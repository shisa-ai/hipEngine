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
models h32 **63.653 tok/s (+0.605%)**, still below Vulkan. The exact composite
was primitive- and runtime-admitted, passed both short clean orders and all
long-context mechanical gates, then failed the complete category gate because
train aggregate TTFT regressed **0.780%** beyond +0.5%. Runtime integration is
removed; the primitive and all evidence remain, while canonical default/topline
stay unchanged. Evidence:
[`gated design`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-gated-design.json),
[`gated primitive`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-gated-correctness.json),
[`gated runtime`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-gated-runtime-correctness.json),
[`gated rejection`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-gated-rejected.json).

Post-gated re-ranking selected and repository-admitted a distinct exact internal
D9 contraction: wave 0 reproduces the retained 256-thread RMS tree after one
LDS publication, then publishes the identical scalar with one final barrier.
All **47/47** actual sparse-layer hidden/norm outputs are BF16-bit exact;
repeated layers 1/47 improve event/wall **2.78-2.87%**. Codegen contracts
**9 -> 2 barriers** and **266 -> 225 instructions** at fixed LDS1024. A
temporary c=1 owner passed exact full state and cache-only
**47-candidate/678-model-kernel** tracing, but the frozen short gate failed in
both orders. Runtime ownership is removed; retained D9 and the canonical topline
remain unchanged.

Post-wave0 re-ranking selected and repository-admitted only the 12 global
current-P4 head+KV calls. The exact local256 wave-0 primitive preserves every
F32 Q/K, BF16 K/V, metadata, and `KVLiveSpans` field across all **12/12** actual
global layers; direct cached-C layers 0/44 improve **20.57-22.55%** event/wall.
The same body regresses SWA **7.21-7.38%**, so SWA remains untouched. A
temporary false/default-off global-only owner passed exact shared-weight state
and **12 candidate + 36 retained SWA / 678-kernel** tracing, but the frozen short
gate failed child/span guards across the two orders. Runtime ownership is
removed; retained current-P4 global and the canonical topline remain unchanged.

Post-global-head re-ranking selected and repository-admitted the 47-call
BF16-hidden/F32-weight router projection wave-0 tree. The separately registered
primitive keeps one local256 block per expert and every existing F32 dot
boundary; wave 0 exactly replays the current stride-128/64/32 partial tree and
uses five shuffles for strides 16..1 after one LDS publication. All **47/47**
actual router outputs are F32-bit exact and finite; repeated layers 1/47 improve
event/wall **8.04-8.13% / 8.02-8.12%**. This is independent of the rejected
expert-tile2/4/8 and persistent/selector owners. A temporary gfx1100 c=1 owner
passed exact shared-weight state and **47-candidate / zero retained decode
projection / 678-model-kernel** tracing, but order A failed the child guard and
order B failed kernel/span guards at the frozen short gate. Runtime ownership is
removed; the diagnostic primitive, retained projection, and canonical topline
remain unchanged.

Post-router-projection re-ranking closes the remaining ranked model-kernel
leaves under their retained or directly rejected exact owners. The first
materially independent residual was therefore the five synchronous runtime
copies outside the **678 model kernels/token**. A false/default-off shared c=1
control owner passed fake-runtime ownership/reset, fresh shared-weight KL0/
top-1 100%, and cache-only **683 -> 681 dispatches/token = five -> three copies
+ 678 identical model kernels**. The frozen short gate nevertheless rejects it:
both orders regress the unchanged model-kernel sum **0.315%/0.336%**; order B
also regresses cycle span **0.700%** and child throughput **0.706%**. Runtime
sharing/borrowing and CLI integration are removed before longer contexts or
categories. Separate publication and canonical **63.270 tok/s** remain the
defaults; the independent reset-position correctness fix is retained.

Post-control-publication re-ranking selects the independent two-read argmax
readback seam. The unchanged stage-2 kernel writes an int64 ID and FP32 value
through separate raw pointers after the explicit fence. A default-off 12-byte
owner now exposes those same pointers at +0/+8 and replaces two D2H calls with
one exact read. RED/GREEN, fresh KL0/top-1-100% state, and exact **683 -> 682 =
five -> four copies + 678 kernels** tracing pass. The frozen short gate rejects
ownership: both orders improve kernel sum, but order B regresses span **1.184%**
and child throughput **1.306%**. The paired owner/runtime/CLI route is removed
before longer contexts/categories; separate owners and canonical **63.270
tok/s** remain.

Post-pair rejection selects a materially different output boundary: point the
unchanged stage-2 raw output pointers at one HIP-registered mapped host page,
retain the explicit fence, and parse the already-visible +0 int64/+8 FP32 bits
without any D2H copy. One actual-vocab process preserves all **15/15** equal-max
tie fixtures and improves every repetition of the complete argmax+fence+host
boundary, **83.311 -> 39.178 us/token (-52.974%)**. Its false/default-off owner
passes RED/GREEN, fresh shared-weight KL0/top-1-100% full state, exact **-2
device allocations / -12 bytes / +4,096 pinned host bytes**, and one cache-only
**681 = three H2D copies + 678 unchanged model kernels / zero D2H** trace. The
frozen short gate nevertheless rejects both process orders. Order A improves
kernel sum **0.201%** but regresses span **1.301%** and child throughput
**0.523%**; order B regresses kernel sum **0.021%** and span **0.790%**. The
mapped owner/runtime/CLI route is removed before longer contexts/categories;
the general host-mapping ABI and canonical **63.270 tok/s** remain.

Post-mapped rejection selects synchronization scheduling rather than another
result owner. Keep both ordinary device outputs and both D2H copies, register one
non-mapped pinned host page, enqueue the two copies on the producing stream, and
replace the current pre-readback fence plus two blocking reads with one final
fence. One frozen actual-vocab process preserves all **15/15** equal-max ID/FP32
fixtures on both default and nonblocking streams; every timing row improves and
the complete boundary moves **82.129 -> 51.353 us/token (-37.472%)**. This is a
design-only/default-off selection with projected topology still **683 = five
copies + 678 model kernels**. No runtime owner, default, or canonical **63.270
tok/s** change exists yet.

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

#### 4.6.1 Matched natural-completion closure

A later repository-owned server harness removes those protocol differences. It
uses the same model SHA-256, all 18 train+heldout prompt token streams, natural
greedy sampling, BF16 K/V, FA on, context 4096, h16/h32, one W7900 queue, and the
same post-TTFT transition count. The frozen engine order is hipEngine A ->
llama.cpp A -> llama.cpp B -> hipEngine B, with four repetitions per process.
Rates pool raw decode seconds rather than averaging per-process rates:

| Matched post-TTFT AR | hipEngine | llama.cpp HIP | hipEngine delta |
| --- | ---: | ---: | ---: |
| h16, 144 runs / 2,160 transitions each | **64.094 tok/s** | 49.290 tok/s | **+30.034% / 1.300x** |
| h32, 144 runs / 4,464 transitions each | **63.431 tok/s** | 49.964 tok/s | **+26.954% / 1.270x** |

Both hipEngine processes pass Poolside KL **0.000156823**, top-1 **100%**, exact
serial/bulk/repeat trajectories, stable cross-process IDs, and complete tracked
teardown. llama.cpp is built from verified `c0bc8591e` source plus one declared
content-only response patch that runs after generation. The accepted server
keeps the clean build's complete **269-file byte-identical HIP bundle**
(`a3c0786d...ce40`; primary `libggml-hip.so` `a3d9e7b8...9faad`). Every native
`prompt_n`, `predicted_n`, and `predicted_ms` row is valid.

The primary normalization is essential: llama.cpp starts `predicted_ms` after
its first sampled token while `predicted_n` includes that token, so its
comparable numerator is `predicted_n - 1`. c0bc can omit entries from SSE token
arrays, making returned-array completeness and cross-engine generated-ID
matches diagnostics rather than timing gates. This is therefore a true 1:1
**protocol/storage/timing** comparison, not bit-identical arithmetic: each
engine retains its own kernels, reductions, and scheduling. The result closes
llama.cpp **HIP** decisively but does not replace or claim victory over the
separate device-pinned Vulkan target. Evidence:
[`matched ABBA artifact`](../benchmarks/results/2026-07-28-gfx1100-laguna-q2-xl-hipengine-vs-llamacpp-hip-matched-abba.json).

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

The separately gated runtime eligibility was narrower than primitive validity:
full-attention c=1 at live<=126 only after split selection was false and only
with paired gate/output pointers. It replaced **12 attention + 12 gate calls
with 12 composites**, contracting eligible short topology **678 -> 666 model
kernels/token** without allocation/workspace/weight changes. At live>=127 the
retained split-gated owner still won first at **678 kernels/token**.

The false/default-off owner passed shared-weight bulk prefill, all **48 hidden /
47 routed** boundaries, 16 transitions crossing live127, active K/V and every
span field, reset/re-prefill, KL **0**, top-1 **100%**, unchanged
**40,459,057,576-byte / 1,500-allocation** peak, and complete teardown. Cached
tracing proved four **12-composite / zero-scalar / zero-standalone-gate /
666-kernel** windows followed by three live127+ **12-score / 12-gated-reducer /
678-kernel** windows, unchanged **36+36 SWA / 45 IQ3** calls, exact resources,
IDs, finite logits, lifecycle, and no compiler.

Both frozen short process orders improved attention+gate **17.772%/18.287%**,
complete kernel sum **0.870%/1.235%**, span **1.054%/0.895%**, and profiled-child
throughput **0.327%/0.235%**. The 512/1K/3968 pairs mechanically preserved zero
candidate and the retained split-gated 678-kernel route. This legally unlocked
the complete 18-prompt two-order gate: aggregate h16/h32 decode improved
**0.844%/0.809%**, every train/heldout category decode row improved at both
horizons, and E2E/prefill/quality/lifecycle checks passed. Promotion nevertheless
fails without rerun because train aggregate TTFT regresses **0.780%**, beyond
+0.5%; overall **+0.071%** and heldout **+0.064%** cannot waive it. Candidate
h32 **63.853 tok/s** remains **0.885%** below Vulkan **64.418**.

Remove the capability/resolver/session/allocator/CLI/route seams and temporary
refactor row; the five production/primitive-test surfaces return byte-for-byte
to primitive commit `b49250b57`. Retain the exact composite, all correctness and
performance evidence, scalar below live127, and split-gated live>=127. Canonical
h32 remains **63.270 tok/s / 678 kernels** with no benchmark rollup change. Do
not retry unchanged ownership. Evidence:
[`design`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-gated-design.json),
[`primitive`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-gated-correctness.json),
[`runtime`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-gated-runtime-correctness.json),
[`rejection`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-gated-rejected.json).

### Post-gated selection: exact MoE-tail wave-0 RMS tree

Re-ranking all **28** stable token windows from the two immutable retained
short controls excludes every rejected Q5/Q6/IQ2/IQ3/IQ4, SWA, router,
staged-add, single-page, and gated-single-page premise. The next independent
surface is instead the already-retained D9 composite itself: its **47**
`laguna_aggregate_moe_tail_next_rmsnorm_out_kernel` calls cost **0.396687
ms/token** and still execute a full local256 LDS reduction with nine dynamic
barriers. No producer/consumer boundary or launch owner is reopened.

The selected sibling preserves both BF16 adds, the hidden store/reload,
per-thread square order, `rsqrtf`, norm multiply association, local256 ownership,
1 KiB LDS, and both outputs. After all 256 partials are published, wave 0 loads
the eight lane-aligned values and exactly replays stride 128, 64, and 32; five
wave32 shuffle-down additions replay strides 16, 8, 4, 2, and 1. Wave 0 then
publishes the same scalar for the unchanged normalization pass. The retained D9
body plus the registered BF16 add + BF16 add + F32-weight RMSNorm chain remain
required fallbacks.

The separately registered gfx1100 primitive now passes **4/4** focused
RED/GREEN coverage, hidden17/3072 edge fixtures against both retained D9 and the
registered add+add+RMSNorm fallback, and ten independent CPU-reference cases at
KL **0** / top-1 **100%**. The one repository actual-input process proves all
**47/47** sparse-layer hidden and normalized outputs byte-exact, then repeats
layers 1/47 with 50 warmups, 15 counterbalanced repetitions, and 1,000
launches/sample. Layer 1 event/wall improves **2.851%/2.806%**; layer 47
improves **2.777%/2.866%**.

Integrated Clang-22 codegen changes logical VGPR **13 -> 14**, keeps
SGPR24/LDS1024/private/spills0, contracts **266 -> 225 instructions** and
**1,404 -> 1,276 B**, and removes seven of nine barriers. A non-profiled exact-
cache preflight precedes `rocprofv3`; the distinct symbol executes at grid/local
**256/256**, allocated VGPR16/SGPR128/LDS1024/scratch0, **13.320 us**, exact
finite outputs, complete teardown, and no compiler under profiling.

Applying only the smallest design endpoint ratio to the immutable D9 family
models **0.010967 ms/token** and h32 **63.270 -> 63.314 tok/s (+0.069%)**. That
remains a planning ceiling, not throughput evidence, and still leaves **1.744%**
to Vulkan. A temporary false/default-off owner substituted only the exact key.
Shared-weight `mixed_ja_en_review` proved bulk prefill, all **48 hidden + 47
routed** boundaries, 16 decode transitions, full logits/IDs, active K/V and
every live-span field, reset, and lifecycle byte-exact at KL0/top-1 100%, with
zero allocation delta. Cache-only tracing recorded **47 candidates/token**,
zero retained-D9 calls, unchanged **45 IQ3 wave10 / 678 model kernels/token**,
local256/VGPR16/LDS1024/scratch0, and no compiler.

The frozen clean gate rejects ownership in the first short context without
rerun. Order A improves the D9 family **3.056%** but regresses complete kernel
sum **0.0506%**, span **0.7993%**, and profiled-child throughput **1.4766%**.
Order B improves kernel/span/child but regresses D9 **0.8431%**. Pooled D9/
kernel/span/child changes are **-1.107%/-0.0355%/+0.214%/-0.443%**, but pooled
rows cannot waive either order. All IDs, exact **47/45/678** topology,
resources, lifecycle, and no-compiler gates pass. Per the frozen stop rule,
512/1K/3968 and categories are skipped. Remove the capability, resolver, plan/
session/CLI/telemetry seams; retain the diagnostic primitive, gfx1151
exclusion, retained D9, and registered add+add+RMSNorm fallback. No benchmark
rollup changes. Evidence:
[`design`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-moe-tail-wave0-tree-design.json),
[`primitive`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-moe-tail-wave0-tree-correctness.json),
[`runtime`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-moe-tail-wave0-tree-runtime-correctness.json),
and
[`rejection`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-moe-tail-wave0-tree-rejected.json).

### Post-wave0 selection: exact global-head wave-0 tree

Re-ranking the same two immutable retained short controls gives **28** stable
windows at **678 kernels/token**. After excluding every closed quant,
attention, SWA-reducer, selector/composite-router, add/norm, D9-wave0,
submission, and prompt-conditioned premise, the next open boundaries are the
47-call F32 router projection at **0.319143 ms/token** and the 48-call current-P4
head+KV family at **0.246175 ms/token**. The latter separates into **0.077259
ms/token / 12 global calls** and **0.168916 ms/token / 36 SWA calls**.

An exact router-projection tile2/4/8 screen first reuses each BF16 hidden load
across multiple F32 expert rows without touching selection. Both actual endpoint
layers remain F32-bit exact, but every tile regresses event and wall by
**8.07-45.48%**; reject it out of tree. Head local32/64/128 screens likewise
prove that geometry cannot be shared across attention types: local128 improves
both global endpoints **17.60-20.40%**, but every right-sized SWA row
regresses **8.83-36.99%**. Do not change SWA ownership.

Select only
`hip_gfx1100/head_rmsnorm+partial_rotary+kv_write/laguna_f32_weight/global_wave0_tree_f32_bf16_spans`.
It keeps local256, dynamic LDS1024, all page-256/`KVLiveSpans` translation and
metadata, every F32 Q/K and RNE-BF16 K/V store, each source square, `rsqrtf`,
norm/RoPE association, and the exact stride-128/64/32/16/8/4/2/1 tree. After
one publication, wave 0 gathers the eight lane-aligned partials, replays strides
128/64/32, applies five shuffles for 16..1, and publishes the identical scalar.
The retained current-P4 global body is the mandatory fallback and the current-P4
SWA body remains the sole SWA route.

The only measured selected-source process uses actual `code_merge_intervals`
inputs, 50 warmups/mode, 15 counterbalanced repetitions, and 1,000 launches per
sample. Query/key F32 bits, key/value BF16 bits, live count, token position, and
eviction metadata are exact. Layer 0 event/wall improves **5.76273 -> 4.55536 us
(-20.951%) / 5.77485 -> 4.55726 us (-21.084%)**; layer 44 improves **5.77917 ->
4.70952 us (-18.509%) / 5.79250 -> 4.74422 us (-18.097%)**. The same local256
wave-0 source regresses layers 1/47 SWA event/wall **7.21-7.38%**, mechanically
fencing the future owner to global only. Tracked ownership returns to zero.

Integrated Clang-22 codegen contracts the global branch **814 -> 662
instructions** and **3,976 -> 3,296 B**, logical VGPR **15 -> 12**, and SGPR
**69 -> 67**, with wave32, dynamic LDS1024, private/spills0. Each workgroup
executes **9 -> 2 barriers**; four static barrier opcodes remain in both objects
because query and key branches each contain their own reduction. Ten
`ds_bpermute` instructions are the two branch copies of the five lower-tree
shuffles.

Primitive admission now freezes this exact global-only key. RED failed **4/4**
solely on the absent source/symbol, strict wrapper, package/backend scope, and
executable primitive; GREEN passes **4/4**, with **15/15** adjacent P4/one-page
nodes and **39/39** registry/build nodes. Synthetic positions 0/255/256/4095 are
bit-exact to both retained P4 and the registered unfused chain and clear the CPU
KL <=0.05/top-1 >=90% gate. One actual `code_merge_intervals` process proves all
**12/12** global layers field-bit exact; direct cached-C layer 0 event/wall moves
**5.77105 -> 4.58412 us (-20.567%) / 5.78378 -> 4.57823 us (-20.844%)** and
layer 44 moves **5.76477 -> 4.46480 us (-22.550%) / 5.77753 -> 4.47467 us
(-22.550%)**. A wrapper-loop diagnostic is intentionally not used as a kernel
claim because unequal Python validation overhead raises both absolute arms; the
later full-model owner gate must include that overhead.

Repository Clang-22 codegen exactly reproduces the frozen ceiling: **814 -> 662
instructions**, **3,976 -> 3,296 B**, logical VGPR **15 -> 12**, SGPR **69 ->
67**, private/spills0, and **9 -> 2** dynamic barriers. A non-profiled cache
preflight precedes `rocprofv3`; one distinct candidate dispatch runs at
**6.440 us**, grid/local **14,336/256**, allocated VGPR16/SGPR128, static LDS0
plus dynamic 1,024 B, scratch0, with no compiler and clean teardown.

Applying only the smallest selected event/wall gain to the immutable global
family still models **0.013982 ms/token** and h32 **63.270 -> 63.326 tok/s
(+0.089%)**, **1.724%** below Vulkan. This remains a planning ceiling, not
full-model throughput. A separately committed temporary gfx1100 capability was
false/default-off; explicit selection substituted only the exact global key and
failed closed to retained current-P4 global on default, disable, backend, or key
miss. SWA stayed on current-P4 and rows/prefill stayed unchanged.

Shared-weight `mixed_ja_en_review` proves bulk prefill, all **48 hidden + 47
routed** boundaries, 16 decode transitions, full logits/IDs, active K/V and
every live-span field, reset/re-prefill, and lifecycle byte-exact at KL0/top-1
100%, with zero allocation delta. A non-profiled cache preflight precedes full-
model `rocprofv3`; two transitions record exactly **12 candidate global + 36
retained SWA calls/token**, zero retained-global decode calls, unchanged **45
IQ3 wave10 / 678 model kernels/token**, local256/VGPR16/SGPR128/dynamic-LDS1024/
scratch0, and no compiler.

The frozen short gate improves global-head work **28.728%/26.407%** and complete
kernel sum **0.273%/0.013%** in orders A/B. Order A nevertheless regresses
profiled-child throughput **0.859%**, and order B regresses median dispatch span
**0.810%**, both beyond the 0.5% guards. Pooled child **-0.466%** and span
**+0.299%** are inside guard but cannot waive either per-order failure. Stop
before 512/1K/3968 and categories without rerun. Remove capability/plan/session/
CLI seams; retain the exact primitive, current-P4 global/SWA, and registered
unfused fallback. Evidence:
[`design`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-head-wave0-tree-design.json),
[`primitive`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-head-wave0-tree-correctness.json),
[`runtime`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-head-wave0-tree-runtime-correctness.json),
and
[`rejection`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-head-wave0-tree-rejected.json).

### Post-global-head selection: exact router-projection wave-0 tree

The same two immutable retained short controls contain **28** stable token
windows at exactly **678 model kernels/token**. After excluding every closed
quant, attention, SWA-reducer, selector/persistent-router, add/norm, D9,
head+KV, submission, and prompt-conditioned premise, the largest independent
open family is the 47-call BF16-hidden/F32-weight router projection at
**0.319143 ms/token**. The prior expert tile2/4/8 screen remains closed: every
tile is F32-bit exact on layers 1/47 but regresses event/wall **8.07-45.48%**.
The new premise retains one expert per workgroup and changes no cross-row
ownership.

The separately registered key is
`hip_gfx1100/router_logits/f32/bf16_hidden_wave0_tree`; its validated shape is
c=1, hidden 3072, and 256 experts, but it has no runtime owner after clean
rejection. Keep local256, one block per expert, the same BF16 hidden conversions,
F32 weight reads, eight-term source order, K traversal,
per-thread partial, 1,024-byte dynamic LDS, and F32 logit store. After one
publication, wave 0 exactly reconstructs each lane's retained stride-128,
stride-64, and stride-32 additions, then uses five wave32 shuffle-down additions
for strides 16/8/4/2/1. The registered `bf16_hidden` projection followed by the
unchanged correction-only selector remains the mandatory production route.
Rows, prefill, gfx1151, unsupported shapes, and every runtime dispatch must not
inherit the candidate.

The repository actual-input process uses `code_merge_intervals`, proves all
**47/47** sparse router outputs F32-bit exact and finite, then repeats the first
and last boundaries with 50 warmups/mode, 15 counterbalanced repetitions, and
1,000 launches/sample. Layer 1 event/wall moves **4.37731 -> 4.02555 us
(-8.036%) / 4.38663 -> 4.03485 us (-8.019%)**; layer 47 moves **4.38319 ->
4.02671 us (-8.133%) / 4.39292 -> 4.03636 us (-8.117%)**. Tracked ownership
peaks at **40,069,955,636 bytes / 1,159 allocations** and returns to zero. Ten
independent hidden-3072/256-expert CPU cases pass at max KL **5.63e-16** and
top-1 **100%**.

Integrated Clang-22 codegen keeps wave32, logical VGPR22, private/spills0, and
1,024 B dynamic LDS. Control executes one publication plus eight loop barriers
(two static barrier opcodes); the selected body executes only the publication
barrier, adds five `ds_bpermute` shuffles, reduces logical SGPR **30 -> 28**, and
contracts static instructions **226 -> 220**. Text grows **1,140 -> 1,208 B**
despite the instruction contraction. Applying only the smallest endpoint gain
to the immutable family models **0.025819 ms/token** and h32 **63.270 -> 63.374
tok/s (+0.164%)**, still **1.648%** below Vulkan. This is a planning ceiling,
not full-model evidence.

Primitive RED/GREEN, hidden17/3072 synthetic edges, the independent CPU gate,
all-47 actual transfer, integrated codegen, and cache-only distinct-symbol trace
all pass. The trace names one candidate at grid/local **65,536/256**, allocated
VGPR24/SGPR128, static-LDS0 plus the exact 1,024-byte dynamic request, scratch0,
with clean teardown and no compiler. The retained projection plus unchanged
selector remains the executable fallback; gfx1151 aliasing is excluded.

A temporary `LAGUNA_ROUTER_PROJECTION_WAVE0_TREE=False` capability owned only an
explicit c=1 route. Default, disable, exact-key miss, rows/prefill, gfx1151, and
unsupported backends retained the base `bf16_hidden` callable. One shared-weight
`mixed_ja_en_review` process matched bulk prefill, **16/16** decode transitions,
full logits/IDs at KL **0** and top-1 **100%**, all **48 hidden + 47 routed**
boundaries, active K/V and every `base_offsets`, `live_counts`,
`token_positions`, and `evict_mask` byte, positions, reset/eight-token
re-prefill, scratch sizes, ownership, and teardown. Peak tracked ownership was
unchanged at **40,459,057,576 bytes / 1,500 allocations** and returned to zero.

A non-profiled exact-cache preflight then preceded one full-model `rocprofv3`
child. Two transitions recorded exactly **94 candidates = 47/token**, zero
retained decode projections, **47 retained tile4 prefill projections**, unchanged
**90 IQ3 wave10 calls = 45/token**, and a first-candidate stride of **683
profiler dispatches = five runtime copies + exactly 678 model kernels/token**.
Candidate resources were local256/VGPR24/SGPR128/static-LDS0 plus the verified
1,024-byte dynamic request/scratch0. IDs, finite logits, lifecycle, and
no-compiler checks passed.

The frozen short gate rejects ownership without rerun. Order A improves the
projection family **7.527%**, kernel sum **0.793%**, and span **3.767%**, but
regresses profiled-child throughput **0.619%**. Order B improves the projection
family **6.307%** and child throughput **1.977%**, but regresses kernel sum
**0.612%** and span **4.211%**. Favorable pooled projection/kernel/span/child
changes (**-6.917%/-0.092%/-0.566%/+0.650%**) cannot waive either per-order
failure. The stop proof contains exactly four short roots, no 512/1K/3968 roots,
no category outputs, and no third order; GPU ownership returns to baseline.
Remove every capability/plan/session/CLI/telemetry seam and the temporary
refactor row. Retain the exact primitive, gfx1151 exclusion, retained
`bf16_hidden` projection, and unchanged selector. No default, benchmark rollup,
or canonical **63.270 tok/s** change is allowed. Evidence:
[`design`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-router-projection-wave0-tree-design.json),
[`primitive`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-router-projection-wave0-tree-correctness.json),
[`runtime`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-router-projection-wave0-tree-runtime-correctness.json),
and
[`rejection`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-router-projection-wave0-tree-rejected.json).

### Post-router-projection selection: shared c=1 control publication

The same two immutable retained short traces contain **28** stable windows and
exactly **678 model kernels/token**. The admitted router-owner trace also
separates a first-candidate stride of **683 profiler dispatches** into those 678
model kernels plus five runtime copies: three synchronous H2D publications
(token, scratch position, and the same KV row position) followed by the two D2H
argmax reads. All larger ranked kernel families were retained at their best
exact form or closed by direct rejection evidence. The five-copy runtime seam
was therefore screened as the first materially independent residual; it did not
reopen any rejected quant, attention, router, composite, graph, or submission
premise.

Current c=1 ownership allocates three independent eight-byte device buffers:
`LagunaEagerScratch.token_id`, `LagunaEagerScratch.position`, and
`LagunaKVCache._row_position`. `forward_token()` publishes token and position to
the first two; `prepare_position()` publishes the same position a third time to
every layer's shared `KVLiveSpans.row_positions`. The retained default fused
head+KV body reads the span position, while only the registered unfused
head-RMSNorm/RoPE fallback reads `scratch.position`. Both values are identical,
but their allocation and copy ownership is currently separate.

The screened admission ABI used one scratch-owned 16-byte control allocation
with non-owning int64 views at offsets +0 for token and +8 for position. The resident KV cache borrows the +8 view for every global
and SWA append/decode span; standalone KV-cache callers continue to own their
existing row-position allocation. One synchronous 16-byte H2D copy publishes
`(token_id, next_position)`, after which borrowed `prepare_position()` performs
only the existing token-serial validation/state transition. The fused path and
unfused fallback therefore read the same pointer and value. Bulk `prepare_rows`
still publishes its consecutive-chunk start, reset still writes -1 and clears
metadata, and the KV borrower must close before scratch. Borrowed views are
never freed independently. The modeled resident delta is **-2 allocations / -8
bytes** with no kernel, weight, workspace, row/prefill, or sampling-read change.

A direct HIP screen exercises the exact current and candidate host contracts,
not a synthetic kernel. It performs 100 alternating warmups and 15
counterbalanced repetitions of 2,000 varied tokens each. Three synchronous
8-byte H2D copies total **66.99773 us/token median**; one synchronous 16-byte
copy totals **22.64014 us/token**, saving **44.35759 us/token (-66.2076%)**.
Final token/position readback, the +8 alias, and tracked teardown are exact.
This includes the current ctypes host-value construction just as production
does, but excludes all model kernels and the two unchanged D2H sampling reads,
so it is planning evidence rather than a full-model throughput claim.

Subtracting only that measured saving from the canonical **15.805224 ms/token**
boundary models **15.760867 ms/token / 63.448 tok/s (+0.281%)**, still **1.529%**
below matched Vulkan. That remains a planning ceiling, not throughput evidence.
A separate runtime admission was completed behind the then-false capability and
explicit session/benchmark opt-in. Its RED contract failed **6/6** only on the
absent seams; final fake-runtime
coverage passes **7/7** for the exact +0/+8 views, reverse partial cleanup,
borrowed versus standalone KV ownership, one pair publication, token-serial
validation, unchanged bulk/reset behavior, fused/unfused position visibility,
and KV-before-scratch teardown. Reset adds one control-only scratch-position
publication outside decode so both control views are `-1`; the aliased candidate
still performs no duplicate reset copy.

Fresh shared-weight `mixed_ja_en_review` state preserves bulk prefill, 16 c=1
transitions, full logits/IDs, all **48 hidden + 47 routed** boundaries, active
K/V and every `KVLiveSpans` field, reset/re-prefill, and teardown at KL **0** /
top-1 **100%**. It measures the designed **-2 allocations / -8 resident bytes**.
One require-cached full-model trace then proves **681 dispatches/token = three
runtime copies + 678 model kernels**, versus immutable control **683 = five +
678**. The complete 678-kernel name/resource multiset, retained 47 router and 45
IQ3-wave10 calls, IDs `[605, 2825, 268]`, finite logits, lifecycle, and
no-compiler checks are exact. This admitted correctness only; separate
publication remained the default and canonical h32 remained **63.270 tok/s**.

The frozen clean contract at runtime commit `31887ae7a` requires each process
order to keep the complete model-kernel sum non-regressive, cycle-span
regression at or below 0.5%, and profiled-child throughput regression at or
above -0.5%, while preserving exact **683 control / 681 candidate** topology.
An initial analyzer asserted all token positions had the admitted first cycle's
resource hash and exited before a verdict; attention grids are position-
sensitive. A frozen amendment changed only that assertion to corresponding-
token control/candidate multiset equality, pinned the four already-produced
roots, and analyzed them without a profiler rerun or threshold change.

The candidate fails both short orders. Model-kernel sum changes
**+0.315%/+0.336%**; order-A cycle span/child remain inside guard at
**+0.411%/-0.300%**, while order B also fails at **+0.700%/-0.706%**. Pooled
kernel/span/child are **+0.326%/+0.495%/-0.572%** and cannot waive per-order
failures. Generated IDs, finite logits, all corresponding 678-kernel
name/resource multisets, **5/3** copy counts, **-2 allocations / -8 bytes**,
retained selectors/resources, teardown, and no compiler are exact. Stop before
512/1K/3968 and categories with no third order or rerun. Remove the capability,
resolver, 16-byte owner/views, borrowed-KV ABI, pair copy, lifecycle branch,
session/CLI/telemetry, and temporary refactor row. Retain standalone KV
ownership, bulk/reset semantics, and the independent control reset publication
for the unfused scratch-position view. Evidence:
[`design`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-control-publication-design.json),
[`runtime`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-control-publication-runtime-correctness.json),
and
[`rejection`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-control-publication-rejected.json).

### Post-control-publication selection: shared argmax readback

Re-rank the same two immutable retained controls at clean rejection commit
`c1c30238e`: **28** stable windows, exactly **678 model kernels/token**, rank
hash `e3eac267...2312`, and trace hashes `0b934f80...f4f` /
`f758b18e...132`. Every model-kernel family is retained at its best exact route
or closed by direct rejection evidence. The removed shared H2D owner is also
closed. The remaining independent runtime seam is the two synchronous D2H
argmax scalar reads after the explicit stream/device synchronization.

Current scratch owns separate eight-byte `argmax_id` and four-byte
`argmax_value` allocations. The unchanged `argmax_stage2_kernel` takes their raw
pointers and writes an int64 ID plus FP32 value. Both scalar sampling sites then
construct separate ctypes hosts and issue one D2H call per value. Select a
future false/default-off owner that allocates one exact 12-byte result block,
exposes the same int64 pointer at +0 and FP32 pointer at +8, and reads the whole
block with one synchronous 12-byte D2H copy. The kernel signature, reduction,
tie-break, output bits, fence, logits, rows/verifier arithmetic, and fallback
stay unchanged. The resident delta is **-1 allocation / 0 bytes** and the
profiled stride models **683 -> 682 dispatches/token = five -> four runtime
copies + 678 unchanged model kernels**.

Freeze the design before one W7900 process (`58e67079...e1fc`; harness
`5c98a598...a1e`). The screen runs the repository's cached
`argmax_stage1_kernel`/`argmax_stage2_kernel` on **15** deterministic 4,096-logit
fixtures with equal maxima and varying minimum-index tie winners. Control and
candidate launches pass the separate pointers versus exact +0/+8 aliases.
After 100 alternating warmups, 15 counterbalanced repetitions, and 2,000
readbacks/sample, every int64 ID and FP32 value bit is exact. Two scalar reads
cost **45.02288 us/token median**; one 12-byte read costs **22.76360 us/token**,
saving **22.25928 us/token (-49.4399%)**. Tracked **16,456 bytes / six
allocations** return exactly to baseline. Raw/stdout/stderr hashes are
`79d59bae...3c1e` / `79d59bae...3c1e` / `e3b0c442...b855`. No codegen comparison
is applicable because kernel source and pointers are unchanged.

Subtracting only the direct saving from canonical **15.805224 ms/token** models
**15.782965 ms/token / 63.359 tok/s (+0.141%)**, still **1.671%** below matched
Vulkan. This remains a planning ceiling, not a throughput claim.

The separate runtime owner is now admitted behind
`LAGUNA_ARGMAX_PAIR_READBACK=False` and explicit session/benchmark opt-in. RED
failed **5/5** only on the absent capability/resolver, exact owner/views,
one-copy parser and both scalar sampling routes, fallback, and CLI telemetry;
GREEN passes **5/5**. Candidate scratch owns one 12-byte result allocation and
non-owning +0 int64-ID/+8 FP32-value views; separate ownership remains the
fallback. Allocation failure and teardown free only owners, the parser performs
one exact 12-byte D2H copy after the unchanged fence, and argmax source,
signature, pointers, arithmetic, and registry are unchanged.

Fresh shared-weight `mixed_ja_en_review` bulk prefill plus **16** c=1
transitions preserves complete logits/IDs, all **48 hidden + 47 routed**
boundaries, active K/V and every `KVLiveSpans` field, reset/re-prefill, and
lifecycle at KL **0** / top-1 **100%**. The candidate removes one scratch
allocation with **0 resident-byte** change. A non-profiled require-cached child
then precedes one `rocprofv3` process. Two transitions prove **682
dispatches/token = four runtime copies + 678 model kernels**, versus immutable
control **683 = five + 678**. The complete kernel name/resource multiset remains
`2f053ea1...c053`; retained 47 router and 45 IQ3-wave10 calls/token, IDs
`[605, 2825, 268]`, finite logits, **23 scratch / 245 KV allocations**,
lifecycle, and no compiler are exact.

The frozen no-rerun short gate then rejects ownership. Both orders improve
complete model-kernel sum **0.065%/1.298%**. Order A also improves span **0.404%**
and child throughput **0.131%**, but order B regresses span **1.184%** and child
throughput **1.306%**, outside the 0.5% guards. Pooled kernel/span/child changes
are **-0.685%/+0.260%/-0.321%** and cannot waive the per-order failure. Exact
IDs, corresponding **683/682 = 5/4 + 678** topology/multisets, allocations,
resources, lifecycle, and no compiler pass. Stop before 512/1K/3968 and
categories with no rerun or third order. Remove the capability/resolver, paired
owner/views, parser/routing, session/CLI/telemetry, and temporary refactor row;
restore separate owners/reads exactly. Evidence:
[`design`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-argmax-readback-design.json),
[`runtime`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-argmax-readback-runtime-correctness.json),
and
[`rejection`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-argmax-readback-rejected.json).

### Post-paired-readback rejection selection: mapped-host scalar argmax output

Re-rank the same immutable **28** retained short windows at clean rejection
commit `f554a419c`. Every model-kernel leaf, shared H2D publication, and paired
*device*-result owner is retained or directly rejected. The next independent
seam does not merge those rejected D2H reads: it eliminates them by making the
existing stage-2 raw output pointers directly host-visible.

The torch-free runtime already supports `HIP_HOST_REGISTER_MAPPED`,
`host_register()`, `host_get_device_pointer()`, and `host_unregister()`; the
prefill flight recorder independently uses this ABI. Select one future
page-backed 4,096-byte host owner, expose its device-visible int64 ID at +0 and
FP32 value at +8, and pass those addresses to the unchanged
`argmax_stage2_kernel`. Keep both argmax stages, launch sizes, reduction,
minimum-index tie-break, FP32 bits, and the existing stream/device synchronization.
After that fence, parse the mapped host bytes directly. Scalar device ownership
falls by **two allocations / 12 bytes** and D2H copies fall **two -> zero**;
one pinned host page is added. The three H2D publications remain, so modeled
full-token topology is **683 -> 681 profiler dispatches = five -> three runtime
copies + 678 unchanged model kernels**. Rows/prefill/verifier sampling and the
ordinary separate-device owner/read chain remain mandatory fallbacks. No kernel
source, signature, registry key, or codegen comparison changes.

Freeze the screen before launch (`f81d7a9e...e190`; builder
`b9669cd0...e6a4`; harness `ad5b0e01...00f`). Exact command:

```bash
HIP_VISIBLE_DEVICES=0 HIPENGINE_HIP_ARCH=gfx1100 GPU_MAX_HW_QUEUES=1 \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version-laguna-iq2.txt \
HIPENGINE_REQUIRE_CACHED_BUILD=1 PYTHONPATH=. \
uv run python -u /tmp/bench_laguna_mapped_argmax.py \
  --json /tmp/laguna-mapped-argmax-screen.json
```

One W7900 process runs the cached repository argmax at the actual **100,352**
vocabulary / 256 threads / 98 stage-1 blocks. All **15/15** deterministic
fixtures preserve the expected int64 ID, FP32 value bits, equal-maximum
minimum-index tie, and replacement of a distinct mapped sentinel after the
fence. Timing alternates two preloaded actual-vocab logit buffers, uses 100
warmups per arm and 15 counterbalanced repetitions of 1,000 complete boundaries,
and requires every repetition to improve. Control ranges **83.136-91.167
us/token**; mapped output ranges **39.086-42.119 us/token**. Every row wins and
medians move **83.310898 -> 39.177799 us/token**, saving **44.133099 us/token
(-52.973981%)**. Six tracked screen allocations / 804,004 bytes return exactly
to zero, the mapped page unregisters before close, no compiler runs, and stderr
is empty. Raw/stdout/stderr hashes are `574f603f...be32` / `574f603f...be32` /
`e3b0c442...b855`.

Subtracting only the direct saving from canonical **15.805224 ms/token** models
**15.761091 ms/token / 63.447 tok/s (+0.280%)**, still **1.530%** below matched
Vulkan. This remains a planning ceiling, not full-model throughput evidence.
The design was committed before the separate runtime admission below. Evidence:
[`design`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-mapped-host-argmax-design.json).

### Default-off mapped-host scalar argmax runtime admission

Runtime RED at clean design commit `37e2057f7` freezes
`LAGUNA_MAPPED_ARGMAX_OUTPUT=False`, gfx1100-only fail-closed resolution, one
page-backed owner, non-owning +0/+8 device views, direct parsing after the
existing fence, separate-device fallback, and session/CLI telemetry. All
**5/5** nodes fail only on the absent seams (`6cef2be5...90a0`). Commit
`31ded283c` adds those host-runtime routes without changing the two argmax
kernels, raw-pointer ABI, arithmetic, registry, model-kernel count, rows,
verifier, or defaults. Final focused mapped/pair/control coverage passes
**11/11** (`08faea39...869`); adjacent coverage passes **111/111**
(`dac9ff0c...97e3`), CPU deterministic coverage passes **29/29**, and
Ruff/compile, fixtures/registry/build, and lineage pass.

`LagunaMappedArgmaxOutput` owns one anonymous **4,096-byte** page registered
with `HIP_HOST_REGISTER_MAPPED`, exposes device-visible int64 ID +0 and FP32
value +8 views, and unregisters before closing the mapping. Candidate scratch
owns **22** ordinary device buffers instead of 24 and never sends either mapped
view to `hipFree`; allocation/register/device-pointer failures and idempotent
teardown are covered. `_read_laguna_argmax_result()` reads mapped host bytes
directly after the unchanged stream/device synchronization and otherwise keeps
the exact two-read fallback.

Freeze no-source-change validation at runtime commit `31ded283c` before model
load (`f2924ee8...d68ff`; manifest `fbfc3d2e...0059`). Fresh shared-weight
`mixed_ja_en_review` bulk prefill, one all-layer capture, 15 further c=1
transitions, reset, and eight-token re-prefill preserve complete logits/IDs,
all **48 hidden + 47 routed** boundaries, active K/V, and every
`KVLiveSpans` field at KL **0** / top-1 **100%**. Ownership moves **24 -> 22
scratch allocations** and **40,069,953,588 -> 40,069,953,576 device bytes**
while adding one 4 KiB pinned page. All **18** mapped sentinels are replaced
after the fence; every counted candidate decode performs exactly three 8-byte
H2D calls and zero D2H calls. Mapping unregister/close and tracked device
ownership return exactly to baseline. Result/stdout/stderr hashes are
`d78245ea...2114` / `d76f0b6e...05c4` / `e3b0c442...b855`.

A non-profiled require-cached child precedes exactly one flat-CSV `rocprofv3`
process. Two transitions prove **681 dispatches/token = three H2D copies + 678
model kernels / zero D2H**, versus immutable control **683 = five copies +
678**. The complete kernel name/resource multiset remains
`2f053ea1...c053`; retained 47 router and 45 IQ3-wave10 calls/token, IDs
`[605, 2825, 268]`, finite logits, **22 scratch / 245 KV allocations**, mapped
lifecycle, and no compiler are exact. Trace/summary/child/profile-stderr hashes
are `c68caa84...61a6` / `b0f6d0e2...5327` / `839bd299...aa29` /
`089407c7...8441`. This admitted only a false/default-off correctness route. Freeze its clean gate
at commit `90b0d5cfb` before launch (`51f9b950...fef4`; runner/child/analyzer
`3fb7a17e...7958` / `e7692d22...347b` / `ce3ffbdb...4c51`). All four declared
short processes run exactly once. The runner exits before a verdict because its
analyzer compares a post-warmup position-sensitive attention-resource hash to
the admission cycle-zero hash. Freeze analyzer-only amendment
`30ddebc0...ea1c`; it deletes only that redundant assertion, pins all four raw
processes, and preserves corresponding-token multiset equality, thresholds,
orders, and measurement bytes. No profile is rerun.

The official analyzer exits **3** and rejects both short orders. Order A improves
complete model-kernel sum **0.201%** but regresses cycle span **1.301%** and
profiled-child throughput **0.523%**. Order B improves child throughput
**1.263%**, but regresses model-kernel sum **0.021%** and span **0.790%**.
Pooled kernel/span/child changes are **-0.090%/+0.643%/+0.636%** and cannot
waive either per-order failure. Exact generated IDs, finite logits,
**683/681 = 5/3 copies + 678 corresponding kernels**, **3 H2D / 0 D2H**
candidate directions, ownership, resources, mapping/device lifecycle, and no
compiler all pass. Stop proof `8f5faf74...9a7a` records zero 512/1K/3968,
category, third-order, or rerun processes.

Remove the gfx1100 capability/resolver, mapped page owner/views, common parser,
both scalar routes, session/CLI/telemetry, and temporary refactor row. Restore
the backend package, runner, benchmark, and adjacent pair contract byte-for-byte
to design commit `37e2057f7`; keep separate device owners/reads and the general
HIP host-mapping ABI. Rejection cleanup passes **3/3**, adjacent **109/109**, and
CPU/registry/build **25/25**, with Ruff and lineage green. Canonical **63.270
tok/s**, defaults, and benchmark rollups remain unchanged. Do not retry unchanged
mapped ownership without a materially new premise. Evidence:
[`runtime`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-mapped-host-argmax-runtime-correctness.json)
and
[`rejection`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-mapped-host-argmax-rejected.json).

### Post-mapped-output rejection selection: single-fence pinned async readback

Re-rank the same immutable **28** retained short windows at clean rejection
commit `607feec94`. Every model-kernel leaf, shared H2D publication, paired-device
readback, and mapped-host output owner is retained or directly rejected. Do not
retry ownership. The remaining independent scalar boundary is synchronization
scheduling: current decode fences the producing stream/device, then performs two
separate blocking D2H calls for the ordinary int64 ID and FP32 value outputs.

Select a registered **non-mapped** host staging page. Keep both separate device
owners, both D2H copies, argmax stages, pointers, arithmetic, tie-break, model
kernels, and projected **683-dispatch = five-copy + 678-kernel** topology.
Enqueue the unchanged 8-byte and 4-byte D2H copies to host offsets +0/+8 on the
same producing stream, issue one final stream/device fence, then parse the host
bits. Registration uses flags **0**, never calls `host_get_device_pointer()`, and
unregisters before page close. The existing pre-fence plus blocking reads remains
the mandatory fallback. This is materially distinct from merging device outputs
or making argmax write mapped host memory.

Freeze one actual-vocab design process before launch (`85ecada2...cef3`; builder
`f0ecf5fb...66f7`; harness `2f3aa9e2...ad4c`). Exact command:

```bash
HIP_VISIBLE_DEVICES=0 HIPENGINE_HIP_ARCH=gfx1100 GPU_MAX_HW_QUEUES=1 \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version-laguna-iq2.txt \
HIPENGINE_REQUIRE_CACHED_BUILD=1 PYTHONPATH=. \
uv run python -u /tmp/bench_laguna_pinned_async_argmax_readback.py \
  --json /tmp/laguna-pinned-async-argmax-screen.json
```

The sole W7900 process runs the cached repository argmax at vocabulary
**100,352**, local256, and 98 stage-1 blocks. All **15/15** deterministic
fixtures preserve int64 ID, FP32 value bits, equal-maximum minimum-index ties,
and distinct sentinel replacement on both default and nonblocking streams. The
timed default-stream arm uses two alternating actual-vocab buffers, 100 warmups
per arm, and 15 counterbalanced repetitions of 1,000 complete boundaries. Every
row improves: control ranges **82.019-83.968 us/token**, candidate
**51.274-53.584 us/token**, and medians move **82.128664 -> 51.353381
us/token**, saving **30.775283 us/token (-37.472037%)**. Six tracked device
allocations / 804,004 bytes return exactly to zero, the pinned page unregisters
and closes, the explicit nonblocking stream is destroyed, no compiler runs, and
stderr is empty. Raw/stdout/stderr hashes are `cbdaeb11...0b0a` /
`cbdaeb11...0b0a` / `e3b0c442...b855`.

Subtracting only the direct saving from canonical **15.805224 ms/token** models
**15.774449 ms/token / 63.394 tok/s (+0.195%)**, still **1.616%** below matched
Vulkan. This remains a planning ceiling, not full-model throughput evidence.

A default-off runtime owner was admitted at commits `02e04930f` and
`938b55126`. It exposed an explicit capability, constructor option, and
benchmark flag while defaults retained the prior pre-fence plus separate
blocking reads. The owner registered one 4 KiB page with flags 0, never obtained
a device pointer, preserved all **24 scratch device buffers / 654,804 bytes**
and both scalar owners, enqueued exactly two async D2H copies, then issued one
final stream/device fence. Focused owner/rejection coverage passed **14/14**,
the adjacent bundle **114/114**, CPU/registry/build **25/25**, all seven
fixtures, Ruff/compile, lineage, and a cached real-HIP production smoke on both
default/nonblocking streams.

Fresh shared-weight `mixed_ja_en_review` state was exact through bulk prefill,
one all-layer capture, **16** decode transitions, reset, and re-prefill: full
logits/IDs, all **48 hidden + 47 routed** boundaries, active K/V, every
`KVLiveSpans` field, and device positions passed at KL **0** / top-1 **100%**.
All **18** pinned sentinels were replaced; 15 counted transitions each proved
**3 blocking H2D -> 2 async D2H -> one final fence**. Device and pinned-page
lifecycle returned to baseline. Exactly one require-cached profiler process
recorded unchanged **683 dispatches/token = 3 H2D + 2 D2H + 678 model kernels**.
The complete model-kernel name/resource multiset matched immutable control hash
`2f053ea1...c053`; retained router/IQ3 resources, IDs `[605, 2825, 268]`, finite
logits, and no-compiler checks passed.

The frozen clean gate nevertheless rejects the candidate at the first context.
All four declared short roots complete once. The first analyzer exits before a
verdict because two control contracts moved from the compact artifact's
`profile` object to `runtime_implementation`; a frozen one-block amendment
changes only those equivalent schema assertions, pins every raw output, and
performs no profiler rerun or threshold change. Order A improves model-kernel
sum **0.126%**, but cycle span regresses **0.618%** and profiled-child throughput
regresses **0.503%**. Order B improves span **0.344%** and child throughput
**0.052%**, but model-kernel sum regresses **0.126%**. Pooled kernel/span/child
changes are **+0.000025%/+0.361%/-0.235%** and cannot waive either per-order
failure. Exact IDs, five-copy/**683-dispatch/678-kernel** topology, copy
directions, two-async-D2H/one-final-fence order, resources, ownership, lifecycle,
and no-compiler checks pass.

Per the frozen stop rule, 512/1K/3968, categories, a third order, and reruns are
skipped. Remove the capability/owner/session/helper/CLI/telemetry integration,
temporary test, and refactor row; production backend/runner/benchmark and the
mapped/pair rejection tests return byte-identical to design commit `b2f1b4f84`.
Retain the general HIP host-registration ABI and separate blocking scalar reads.
No default, canonical **63.270 tok/s / 678 kernels**, benchmark README, or
changelog changes. Do not retry the unchanged synchronization premise. Evidence:
[`design`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-pinned-async-argmax-readback-design.json),
[`runtime`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-pinned-async-argmax-readback-runtime-correctness.json),
and
[`rejection`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-pinned-async-argmax-readback-rejected.json).

## 9. Stopping point and gfx1151 Q4_K_M handoff (2026-07-27)

### 9.1 Frozen gfx1100 result

This is a good stopping point for small host/runtime work. The retained W7900
h32 result remains **63.270 tok/s / 15.805224 ms/token / 678 model
kernels/token**. Matched device-pinned Vulkan remains **64.418 tok/s /
15.523554 ms/token**, leaving **0.281670 ms/token** or **1.814%** more throughput
to match. The most recent retained win is IQ3 wave10 fusion at commit
`7f1d77ab6` (**2026-07-27 02:17:18 +0900**); it moved h32 **62.318 -> 63.270
tok/s (+1.528%)** at **723 -> 678 model kernels/token**. Every subsequent
candidate was either retained only as a diagnostic primitive or removed after a
full-model guard failed.

The final host candidate is closed, not pending. Single-fence pinned async
readback improved its isolated boundary **82.129 -> 51.353 us/token
(-37.472%)**, but both frozen short process orders failed: order A span/child
regressed **0.618%/0.503%**, and order B model-kernel sum regressed **0.126%**.
The runtime integration is removed at `482696290`; canonical throughput and
rollups do not change. See the [design](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-pinned-async-argmax-readback-design.json),
[runtime gate](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-pinned-async-argmax-readback-runtime-correctness.json),
and [rejection](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-pinned-async-argmax-readback-rejected.json).

A fresh matched same-source HIP audit now clarifies the backend boundary without
changing that retained route. The frozen two-process-per-engine ABBA row pools
hipEngine **64.094/63.431 tok/s** versus llama.cpp HIP **49.290/49.964 tok/s**
at h16/h32, a **30.034%/26.954%** lead over exactly **2,160/4,464 transitions
per engine**. This is accepted natural-completion protocol/storage/timing
parity with BF16 K/V and FA on, not cross-engine bit identity. It proves the
remaining Laguna target is Vulkan-specific; it does not promote a new hipEngine
default or replace canonical **63.270 tok/s**. See the
[matched ABBA artifact](../benchmarks/results/2026-07-28-gfx1100-laguna-q2-xl-hipengine-vs-llamacpp-hip-matched-abba.json).

### 9.2 Next gfx1100 work must have a larger premise

The current short residual profile is the planning basis, not a promise that
saving a leaf transfers one-for-one to wall time. Closing the matched Vulkan gap
requires at least **0.282 ms/token** of clean end-to-end saving. The next lane
should start only when it can plausibly clear that bar or compose with another
independent large surface.

| Rank | Surface in the post-wave10 short trace | Required new premise | First gate |
| ---: | --- | --- | --- |
| 1 | Mixed Q5/Q6 projection **1.930 ms/token**, IQ2 selected **1.896 ms/token**, Q5 attention output **1.558 ms/token**, IQ3 wave10 **1.441 ms/token** | A materially different packed-dot/dequant/reduction schedule. Local-size changes, metadata-only rewrites, SWAR pair variants, and exact owner substitutions are already closed. A changed rounding schedule is admissible only as a quality-gated kernel, never as prompt-specific score tuning. | Actual first/last weights and source/ISA screen, then full state and complete 18-prompt train/heldout categories at KL <=0.05/top-1 >=90%. |
| 2 | SWA reducer **1.492 ms/token** plus score producer **0.238 ms/token** at short, growing sharply with context | A new cooperative/tiled online-softmax or value schedule that keeps the 72-query-head parallelism. Do not retry dim2/local-size/barrier-only variants or the rejected GQA3 collapse. | Live 70/128/257/512 endpoints, exact/quality oracle, full 512/1K/3968 state and both process orders before categories. |
| 3 | Q4 LM head **0.267 ms/token** plus the scalar argmax/readback boundary | Produce reduction candidates while the LM-head producer already owns output tiles, or fuse a mathematically valid final reduction without removing required logits. This must eliminate a logits reread or launch; changing only host ownership/synchronization is closed. | Full 100,352-logit equality or quality gate, minimum-index ties, unchanged device ownership, trace-proven launch/read reduction, then complete model/category gates. |
| 4 | Submission spacing after the arithmetic lanes above | Only a genuinely device-resident multi-launch/persistent schedule with a new correctness premise. Unchanged HIP graph replay, direct AQL, C-side packet batching, shared scalar publication, paired/mapped/pinned readback, and one-doorbell experiments are closed. | One immutable full-cycle trace showing the new owner and a saving larger than profiling variance before any broad implementation. |

Source for the residual family sizes:
[`...q5-swar-output-only-design.json`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-q5-swar-output-only-design.json).
The acceptance rule remains metric improvement **and** every mechanical/quality
check; an isolated positive leaf is not retainable by itself.

### 9.3 gfx1151 Q4_K_M is a different optimization target

Do not infer portability from both devices accepting wave32 code. The Radeon
8060S/gfx1151 is a UMA system with different CU/cache/occupancy and driver
behavior from the discrete W7900/gfx1100. The local Qwen3.6 Q4_K_M file declares
41 blocks, of which the runtime correctly excludes trailing NextN block 40 and
executes **40 AR layers: 10 full-attention + 30 GDN/linear-attention**. It uses
hidden 2,048, 256 experts, top-8 routing, and vocabulary 248,320. Relevant weight
roles are:

| Qwen3.6 Q4_K_M role | Stored type | Consequence for Laguna transfer |
| --- | --- | --- |
| Expert gate/up | Q4_K, production decode repacked to Q4T16 | Laguna raw-Q5/Q6 mixed-attention bodies do not apply. Pairing/metadata ideas may be retuned against the existing T16 route. |
| Expert down | Q5_K for 37 AR layers and Q6_K for 3 | Laguna Q5/Q6 scheduling ideas are relevant only conceptually; gfx1151 already has independently gated Q5T16/Q6T16 selected-down pair reuse. |
| Output head | Q6_K, production decode repacked to Q6T16 | Laguna's retained raw-Q4 local32 LM-head kernel is the wrong quant/layout. Only the ownership/partitioning idea transfers; gfx1151 already retains a measured 5+3 C8 partition. |
| Embedding, dense attention/GDN/shared projections | Mostly Q8_0 | The largest historical gfx1151 decode family is dense Q8, which none of Laguna's IQ2/IQ3/Q5 bodies accelerate. |
| Attention | 10 full-attention layers; no Laguna-style SWA | Global paged-attention concepts can be reconsidered, but all SWA score/reducer kernels and thresholds are inapplicable. The other 30 layers need GDN-specific work. |

The pre-later-tuning gfx1151 decode closure profile recorded **708 kernel
dispatches/token** and, at 4K, about **8.541 ms/token dense Q8**, **4.189 ms/token
selected MoE**, **1.858 ms/token Q6 LM head**, **1.005 ms/token full attention**,
**0.768 ms/token router**, and **0.742 ms/token GDN**. Treat those values only as
an attribution prior: later physical-C8 promotions changed the route, and the
other agent's final clean commit/artifact must become the measurement baseline
at handoff. Evidence:
[`...gfx1151-gguf-decode-closure-profile.json`](../benchmarks/results/2026-07-15-gfx1151-gguf-decode-closure-profile.json).

### 9.4 Transfer matrix

| Laguna/gfx1100 result | gfx1151 Q4_K_M disposition | Handoff action |
| --- | --- | --- |
| Four-axis registry, backend fail-closed capabilities, CPU-reference/full-state gates, immutable trace hashes, counterbalanced process orders, category/heldout anti-gaming gate | **Directly reusable** | Reuse the method unchanged. Never alias a gfx1100 key/default into gfx1151 without its own model/hardware gate. |
| Resident weights, repacked-cache source binding, session pool, native scheduler, state reset/continuation, `KVLiveSpans`, lifecycle accounting | **Directly reusable and largely already shared** | Preserve these runtime wins when swapping hardware. Revalidate memory on gfx1151 UMA and keep the 128K lifecycle blocker separate from kernel speed. |
| Fixed quant metadata hoists, one-wave/two-output ownership, shared activation reads, duplicate-expert reuse, mixed launch collapse | **Portable design pattern; retune required** | Apply to the actual Q4T16/Q5T16/Q6T16/Q8T16 roles and physical C1/C2/C4/C8 shapes. Existing gfx1151 pair-reuse wins show the concept is useful, but their thresholds are distribution- and width-specific. |
| Laguna mixed Q5/Q6 attention projection and Q5 attention-output defaults | **Wrong Qwen role/layout** | Do not port the body. Qwen dense attention is Q8_0; investigate selected-MoE T16 or dense-Q8 equivalents instead. |
| Laguna Q4 local32 LM head | **Wrong output quant; geometry lesson only** | Qwen output is Q6T16. Start from the retained gfx1151 5+3 partition, not the Laguna local32 source. A producer-integrated argmax is a new experiment and must preserve full logits/sampling semantics. |
| IQ2 grid64 and IQ3 wave10 fusion | **Not applicable** | Q4_K_M has no IQ2/IQ3 AR weights. Do not spend the gfx1151 window porting these kernels. |
| SWA tile16/split/wave-local reducers and global/SWA threshold table | **SWA portion not applicable; full-attention idea only** | Keep Qwen's existing full-attention split-K/fixed256 routes. Reuse only the threshold-measurement method; do not copy 65/127/257 thresholds or 72-head geometry. |
| Exact head+KV, reducer+gate, MoE-tail+next-RMS and grouped-combine fusions | **Model-level fusion pattern; independent math/shape gate required** | Audit Qwen's 10 full-attention and 30 GDN boundaries for an equivalent consumer/producer pair. Preserve an unfused registered fallback and all state bytes. |
| Laguna prefill tiled F16, SWA wave32, grouped down/combine | **Mostly model-specific** | Keep the current gfx1151 GDN LCP/direct-conv and compact T16 prefill routes. Port only a quant-neutral scheduler/lifecycle improvement, never the Laguna kernel body by name. |
| Laguna graph/AQL findings | **Do not transfer the conclusion** | Qwen gfx1151 graph replay is independently retained; Laguna's unchanged-graph/AQL failures do not invalidate it. Continue from the final gfx1151 graph owner and compare against same-run eager. |
| Shared control publication; paired-device, mapped-host, and pinned-async argmax readback | **Rejected; do not port** | These passed correctness/topology but failed complete clean Laguna guards. gfx1151's prior no-read 128K diagnostic also did not solve lifecycle. Reopen only with a materially different device-side producer/consumer boundary. |
| gfx1100 local32/local64/local320 choices, VGPR/LDS ceilings, context crossovers | **Architecture-specific evidence only** | Recompile and reprofile on gfx1151. The retained C8 Q6 5+3 partition versus gfx1100 6+2 is direct evidence that identical math needs backend-specific geometry. |

Representative already-retained gfx1151 analogues are
[Q4T16 selected-expert pair reuse](../benchmarks/results/2026-07-20-gfx1151-gguf-selected-expert-pairreuse-c8-retained.json),
[Q5T16 selected-down pair reuse](../benchmarks/results/2026-07-20-gfx1151-gguf-q5t16-selected-down-pairreuse-c8-retained.json),
[Q6T16 selected-down pair reuse](../benchmarks/results/2026-07-20-gfx1151-gguf-q6t16-selected-down-pairreuse-c8-retained.json),
[Q6T16 LM-head 5+3 partitioning](../benchmarks/results/2026-07-20-gfx1151-gguf-q6t16-lm-head-chunk5-c8-retained.json),
[paged-attention token offsets](../benchmarks/results/2026-07-20-gfx1151-gguf-paged-attn-token-offsets-c8-retained.json),
[value-vector2](../benchmarks/results/2026-07-20-gfx1151-gguf-paged-attn-value-vector2-c8-retained.json),
and [GDN state cache24](../benchmarks/results/2026-07-20-gfx1151-gguf-gdn-shared-statecache24-c8-retained.json).
They demonstrate the correct transfer rule: carry the ownership/reuse hypothesis,
then choose a separate gfx1151 registry key and promote only on its own clean
shape and serving evidence.

### 9.5 Hardware-swap checklist

1. Let the gfx1151 agent finish and commit its logical unit. Record its final
   commit, model fingerprint, backend defaults, artifact hashes, and unresolved
   blockers; do not use the July attribution profile as a new baseline if the
   final route changed.
2. Before swapping GPUs, require a clean tracked tree and idle device. Warm each
   architecture's JIT cache outside `rocprofv3`; never reuse a gfx1100 code object
   as gfx1151 evidence.
3. Freeze separate matrices for **prefill versus decode** and **C1 versus packed
   C2/C4/C8**. A Laguna c=1 win does not imply a Qwen physical-C8 win, and a C8
   reuse win may regress unique/random or C1 routing.
4. Reprofile the complete final gfx1151 cycle first. Rank Q8 dense, selected MoE,
   Q6 LM head, full attention, router, and GDN from that trace; select the next
   surface from current numbers rather than stale family totals.
5. Keep backend capabilities and fallbacks separate. Every new gfx1151 kernel
   needs CPU/model correctness, expected-symbol/resource tracing, both clean
   process orders, relevant contexts/widths, and complete prompt categories
   before default promotion.
6. On return to gfx1100, resume only one of section 9.2's materially new
   algorithms. Do not reopen the closed scalar owner/readback, local-size-only,
   or unchanged submission experiments.

### 9.6 gfx1151 Q4_K_M measured roofline and next attack (2026-07-28)

#### Same-GGUF Vulkan control and decode roofline

The same Q4_K_M GGUF now has a one-run llama.cpp Vulkan decode control on the
same 8060S. It is a directional implementation comparator, not a bit-exact
backend race: Vulkan uses F16 KV and llama-bench's random stream, while
hipEngine uses BF16 KV and the retained deterministic trajectory.

| Backend | Decode | Wall/token | Qualification |
| --- | ---: | ---: | --- |
| hipEngine production | **14.800 tok/s** | **67.567 ms** | Seven-pair p512/d128; 127 eager C1 calls; retained exact F16 one-barrier owner |
| llama.cpp Vulkan `c0bc8591e` | **23.348 tok/s** | **42.830 ms** | One-run same-GGUF `tg128`, FA on, depth 512 |
| Gap | **+57.755% Vulkan** | **24.736 ms** | Large implementation gap, not launch-count noise |

The dry resident plan gives an active encoded-weight proxy of
**9.173128 GB/token**: every dense/router/output tensor once, 10/256 of each
rank-3 expert tensor, and one embedding row. Raw GGUF bytes under the same
counting rule are **9.031374 GB/token**. This is not a DRAM counter—it excludes
activations, KV and metadata and cannot see cache reuse—but it is the right
first-order streaming ledger.

| Roofline row | Result |
| --- | ---: |
| Practical read anchor | **221 GB/s** |
| hipEngine resident-byte floor at 221 GB/s | **41.507 ms / 24.092 tok/s** |
| hipEngine resident-byte floor at theoretical 256 GB/s | **35.833 ms / 27.908 tok/s** |
| Current hipEngine wall proxy | **135.22 GB/s / 61.18%** of 221 |
| Current hipEngine device-sum proxy | **137.88 GB/s / 62.39%** |
| Vulkan raw-byte wall proxy | **210.87 GB/s / 95.42%** |

The Vulkan performance logger perturbs its wall
**42.830 -> 45.182 ms/token**, so its family rows are attribution-only. The
logged graph reports about **1,114 operation invocations/token**, more than
hipEngine's **869 kernels/token**. Vulkan is not faster because it launches
less work. hipEngine's unprofiled wall exceeds its traced device sum by only
**2.176 ms/token (3.17%)**, and the trace contains no device overlap.

| Family | hipEngine | Vulkan logger | Recoverable gap | Finding |
| --- | ---: | ---: | ---: | --- |
| Source-F16 attention projections | **30.981 ms** | **24.759 ms** | **6.222 ms** | 5.606 GB family already reaches **180.96 GB/s / 81.88%** of the read anchor |
| All 48 attention layers | **14.613 ms** | **0.909 ms** | **13.703 ms** | **52.96%** of the whole wall gap |
| Selected Q4 gate/up | **8.550 ms** | **7.464 ms** | **1.086 ms** | Secondary |
| Selected Q4/Q6 down | **5.131 ms** | **4.525 ms** | **0.606 ms** | Secondary |
| F16 plus attention | **45.593 ms** | **25.668 ms** | **19.925 ms** | **77.01%** of the whole wall gap |

The attention split is more decisive than its total suggests. Score production
costs only **2.034 ms/token**. Global reduction costs **2.623 ms**, and the 36
SWA reductions cost **9.956 ms**. The retained reducer launches one workgroup
per query head and replays the slot-order value scan independently. At live
512, the physically unique K+V payload is only **100.66 MB/token**, while
query-head repetition models **830.47 MB/token** because one KV head serves
six global or nine SWA heads. Vulkan's fused D128 FlashAttention completes the
12 global layers in **0.227 ms** and the 36 SWA layers in **0.683 ms**.

Vulkan source review also prevents a second wrong transfer:

- C1 uses the one-row MMV/MMV-ID family, not the prefill MMQ path.
- On AMD, Q4_K with K at least 2,048 may reuse Q8_1 activation packing and
  integer dot; Q6_K is deliberately excluded from MMVQ and stays on floating
  dequant.
- RADV uses subgroup64 on this device, while the current HIP bodies are
  wave32. Geometry must be measured independently rather than copied.
- FlashAttention fuses score, softmax and PV. It does not materialize
  hipEngine's score plane and then run the current scalar-order value reducer.

The decode dispatch can be derived more precisely from `c0bc8591e`. Before
tuning, `N=1` and the query/KV ratio is nine, so the host folds all nine SWA
query heads into the row dimension and reduces the Y grid from 72 query heads
to eight KV heads. On this subgroup64/KHR-cooperative-matrix device, the
resulting `N=9`, D128 F16-KV path selects cooperative-matrix FlashAttention
with **Br=16, Bc=64, local256**. The split-K heuristic targets twice the 40-CU
count: at live 512, alignment rounds the key slice to 64 and produces
**8 KV heads x 8 K64 splits = 64 main workgroups**, followed by bounded-state
merge. Each tile carries all nine query rows, reuses its K/V tile across them,
and maintains online max/denominator/output state. This explains the key
difference from hipEngine's retained exact fused GQA2 body: the comparator
gets both full-GQA reuse and enough grid breadth by partitioning keys, rather
than choosing between reuse and occupancy.

#### Qwen3.5 gfx1100-to-gfx1151 lessons for Laguna decode

The Qwen adaptation history is useful chiefly because it shows where
architecture changes invalidate a nominally shared gfx11 body.

| Qwen lesson | Laguna consequence |
| --- | --- |
| Native C2/C4/C8, C8-only pair reuse, and the 5+3 rather than 6+2 LM-head partition were separately admitted | Transfer ownership/reuse hypotheses, not widths, partitions, or thresholds |
| Several leaf wins disappeared or reversed in p512/d128 and serving | Require actual-weight leaf, whole cycle, state, context and category gates |
| Whole-row C4/C8 attention was exact while generic row-chunk2 changed row-local numerics | Treat split/merge width as a numerical contract, not a launch detail |
| Collapsing workgroups can lose on the 40-CU target even when it removes bytes | A Laguna GQA body must regain grid breadth with K splits |
| Ephemeral dense execution rows were separated from stable scheduler/KV/GDN ownership | Preserve the full `KVLiveSpans` ABI; Laguna has no GDN state to port |
| C8 pair reuse says nothing about C1 latency | Keep C1, C2/C4/C8 and verifier-shaped admissions separate |

Two prior failures sharpen the grid rule. The gfx1151 prefill qhead3 screen
collapsed 72 SWA workgroups to 24 and lost **18.1–29.7%** despite 3x K/V
reuse. The W7900 decode GQA3 screen lost **61–90%** at the leaf. Therefore the
next grouped-GQA body must split the 512-key axis into enough independent
K64/K128 tiles to expose roughly **40–80+ workgroups/layer**, then merge
bounded partials. One block per KV head is not a valid gfx1151 design.

Simple F16 thread substitution is also closed. In one full p512/d128 screen
each, changing only rows==1 F16 single/triple launches from local256 to
local64/local128 regressed **14.555 -> 13.855 tok/s (-4.811%)** and
**14.263 tok/s (-2.005%)**. Both also changed the generated trajectory, so
quality work was skipped. Retain local256. The next F16 premise is structurally
different: one local256 block owns eight output columns, each wave owns one
column, and eight per-lane partial sequences reconstruct the current
256-thread reduction order without cross-wave LDS/barriers. That exact wave8
screen is now closed too. It is byte-identical at every natural source-F16
shape, but reducing eight physical waves/output to one doubles the modeled
family from **30.818 -> 64.098 ms/token (+107.99%)**. On the large shapes,
effective weight bandwidth falls from **177.8-202.3 GB/s** to
**85.0-99.2 GB/s** even though LDS falls **512 -> 0 B** and scratch remains
zero; tracing shows VGPR **16 -> 32**. The multi-output block is removed.
LD-2 continues only with a bounded local128 exact owner that keeps four
physical waves/output and removes one of the two retained block barriers. A
narrower same-grid seam lands first: the existing generic reducer performs a
second barrier only to return the final sum to every thread, while GEMV stores
from thread 0 alone. The separately registered local256 one-barrier sibling
keeps all eight waves/output and is byte-exact across all natural shapes.
Stabilized leaf medians improve every role by **0.57-1.71%** and the weighted
family **31.316 -> 31.097 ms/token (-0.698%, -0.219 ms/token)** with identical
VGPR16/LDS512/scratch0 resources. It is now the gfx1151 rows==1 default:
seven exact same-session p512/d128 pairs improve
**14.758912 -> 14.800191 tok/s (+0.280%)**, every candidate sample beats every
control, and cached whole-model tracing records all
**18,288 = 144 x 127** expected candidate calls with zero retained GEMVs.
`HIPENGINE_LAGUNA_F16_DECODE=gemv` remains the exact LD-2 rollback.

The bounded exact local128 follow-up is closed and removed. It recreates the
retained 256 logical accumulation chains and eight ordered wave sums with four
physical waves/output, but every natural role regresses: QKV **15.1-15.3%**,
gate **32.9-33.6%**, and output **31.6-40.7%**. Weighted family time moves
**31.039 -> 39.045 ms/token (+25.79%)**. Cached tracing shows both candidate
and retained at VGPR16/LDS512/scratch0; halving the physical waves is the
failure. Exact one-output F16 owners below eight physical waves/output are now
closed on gfx1151.

An exact local256 two-output block is closed as well. Unlike wave8, it keeps
all 256 logical K chains and all eight ordered wave sums for each output while
sharing the activation load and one barrier across adjacent columns. Odd-width
single/triple fixtures are byte-exact, including pairs that cross projection
boundaries. The reduced output grid still loses on every natural role:
QKV **4.42-4.73%**, gate **28.85-33.12%**, and output **6.14-10.43%**.
Weighted family time moves **31.571 -> 33.466 ms/token (+6.00%)**. Remove the
candidate. LD-2 now treats both eight waves/output and one workgroup/output as
required; further exact screens must preserve the full grid.

The successful LD-2 successor does exactly that. Separate K3072/K6144/K9216
specializations keep the retained local256/eight-wave/one-output grid and every
arithmetic operation, but make the natural K width compile-time constant and
fully unroll each thread's 12/24/36-iteration loop. All single/triple F32/BF16
outputs are byte-exact. Every natural role improves: QKV **20.98-21.15%**,
gate **15.91-15.98%**, and output **19.16-26.93%**. Weighted family time moves
**30.952 -> 24.482 ms/token (-20.90%, -6.469 ms/token)**, reaching the prior
**25.368-ms** unchanged-byte target.

Seven same-session p512/d128 pairs then improve retained one-barrier
**14.786076 -> 16.391201 tok/s (+10.856%, -6.623 ms/token)**. Every fixed-K
sample beats every control; all generated IDs/hashes, final token **74107**,
position **638**, and allocation lifecycle match. Cache-only tracing records
all **18,288 = 144 x 127** fixed-K calls with zero fallback at
local256/VGPR24/LDS512/scratch0. gfx1151 now defaults fixed-K; explicit
`HIPENGINE_LAGUNA_F16_DECODE=onebarrier` retains the exact generic-K rollback.
LD-2's declared family-floor objective is complete. Re-profile the new wall,
then return priority to attention rather than perturbing the now
**228-231 GB/s** large F16 stream.

The post-promotion full trace confirms that re-ranking. Across all **127**
complete decode transitions, fixed-K production has a median **58.890
ms/token** kernel sum and **61.569 ms/token** traced span at **864
dispatches/token**. The family wall is now:

| Fixed-K production family | Calls/token | Median ms/token | Kernel-sum share |
| --- | ---: | ---: | ---: |
| Source-F16 projections | 144 | **24.164** | **41.03%** |
| Global + SWA attention | 96 | **13.778** | **23.40%** |
| Selected Q4 gate/up | 47 | **8.558** | **14.53%** |
| Selected Q4/Q6 down | 47 | **5.153** | **8.75%** |
| Dense/shared quant projections | 144 | **3.711** | **6.30%** |
| LM head + argmax | 3 | **1.122** | **1.91%** |
| Router | 94 | **1.068** | **1.81%** |
| Norm/RoPE/gate | 145 | **1.067** | **1.81%** |

The F16 byte proxy now reaches about **232 GB/s**. Attention is the largest
credible algorithmic gap: the SWA reducer alone is **9.959 ms/token**, global
reduction **2.654 ms**, and both score producers together only **1.148 ms**.
This makes the next exact screen a live-512 SWA reducer specialization that
keeps all **72 workgroups / 288 waves** and the complete scalar/FMA order.
Reducing attention from **13.778 to 3.0 ms/token** models roughly **19.9
tok/s** before any secondary Q4 work.

LD-3 was screened with measured production inputs, not a format assumption. A
raw block32 Q8_0 side representation cuts resident source-F16
bytes **46.875%**. At actual layer-0 full-attention and layer-47 SWA inputs,
one Q8_1 activation pack plus existing dp4a consumers cuts QKV+gate
**70.2-84.9%** and output **85.6%**. The 12-full/36-SWA model is
**31.236 -> 6.675 ms/token (-78.63%)**, a modeled **24.562-ms/token** saving.
Projection error looked encouraging—maximum normalized RMSE **0.00974** and
minimum cosine **0.999952**—but the full-model gate rejects the premise.
Combined all-layer ownership reaches **50.154 ms/token** directionally, yet
teacher-forced max KL is **0.497301** at **15/16** top-1. QKV+gate-only and
output-only are worse at max KL **0.589065/0.619246**. Every structural scope
also fails: even the best 24-layer screens peak at **0.165508/0.205582**.
A second Q8 weight plane grows the sidecar beyond source F16 and still reaches
max KL **0.463224**; a second activation plane regresses it to **0.881135**.
The apparent one-plane accuracy depends on cancellation between the two error
surfaces. Residual experiment kernels are removed and nothing is retained.

LD-4's byte audit corrects the earlier premise. Rank-2 Q4_K pack8 stores
**0.75 byte/weight** (0.5-byte packed quants plus 0.25-byte FP32 scale/min)
versus raw GGUF's **0.5625 byte/weight**, so the resident expansion is
**33.33%**, not a multi-fold decode-traffic increase. On actual dense/shared
weights, the existing raw local128 consumer loses to the production pack8
owner by **63.6-85.4%**. The barrier-free raw wave32 fixed-metadata consumer is
still **2.1% slower** on dense gate and **17.9% slower** on shared down; its
only positive row is a launch-floor **1.0%** shared-gate result. Raw residency
is therefore closed.

A separate pack8 geometry screen found a real but inadmissible speed seam.
Relative to production local32, local128 improves the K3072/N12288 dense gate
**28.60%** and K3072/N1024 shared gate **14.23%**; local64 improves the
K1024/N3072 shared down **6.01%**. Seven same-session p512/d128 pairs move
**14.797582 -> 14.978817 tok/s (+1.225%, -0.818 ms/token)**, with every
candidate faster. The changed reduction partitions are not quality-safe:
the candidate deterministically changes the 128-token trajectory
(final token **74107 -> 340**), and the full 18-prompt/576-step teacher-forced
gate reaches **99.31%** top-1 but max KL **1.002942**. Heldout mixed-JA/EN also
reaches **0.660855**, both far above **0.05**. Remove the architecture owner
and A/B plumbing; production remains **14.800191 tok/s**. Reopen Q4 pack8
geometry only with a reduction-order-preserving design.

Evidence:
[`retained GQA3 score owner`](../benchmarks/results/2026-07-28-gfx1151-laguna-swa-gqa3-scores-retained.json) ·
[`retained fused-GQA2 SWA owner`](../benchmarks/results/2026-07-28-gfx1151-laguna-swa-fused-gqa2-retained.json) ·
[`retained fused-GQA3 local384 SWA owner`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-gqa3-local384-retained.json) ·
[`retained fused-GQA3 V-stage64 SWA owner`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-gqa3-vstage64-retained.json) ·
[`clean fused-GQA3 V-stage64 production`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-gqa3-vstage64-production.json) ·
[`post-V-stage64 wall re-profile`](../benchmarks/results/2026-07-29-gfx1151-laguna-post-vstage64-wall-reprofile.json) ·
[`retained global GQA2 V-stage64`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-gqa2-vstage64-retained.json) ·
[`rejected fused-GQA2 shared-V owner`](../benchmarks/results/2026-07-28-gfx1151-laguna-swa-gqa2-sharedv-rejected.json) ·
[`rejected GQA9 K64 split-K family`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-gqa9-splitk64-rejected.json) ·
[`rejected GQA2 LDS-V staging`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-gqa2-vstage-rejected.json) ·
[`rejected rebalanced cooperative GQA9`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-gqa9-cooperative-rebalanced-rejected.json) ·
[`rejected GQA2 LDS weight cache`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-gqa2-weightcache-rejected.json) ·
[`rejected persistent exact GQA9`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-persistent-gqa9-rejected.json) ·
[`post-GQA3 production wall re-profile`](../benchmarks/results/2026-07-29-gfx1151-laguna-post-gqa3-wall-reprofile.json) ·
[`retained fused one-head global owner`](../benchmarks/results/2026-07-28-gfx1151-laguna-global-fused-gqa1-retained.json) ·
[`retained global fixed-shape reducer`](../benchmarks/results/2026-07-28-gfx1151-laguna-global-fixedshape-reduce-retained.json) ·
[`retained selected natural-shape decode`](../benchmarks/results/2026-07-28-gfx1151-laguna-selected-natural-decode-retained.json) ·
[`retained selected tile8 decode`](../benchmarks/results/2026-07-28-gfx1151-laguna-selected-natural-tile8-retained.json) ·
[`retained F16 one-barrier owner`](../benchmarks/results/2026-07-28-gfx1151-laguna-f16-onebarrier-retained.json) ·
[`retained F16 fixed-K owner`](../benchmarks/results/2026-07-28-gfx1151-laguna-f16-fixedk-retained.json) ·
[`rejected F16 local128 exact owner`](../benchmarks/results/2026-07-28-gfx1151-laguna-f16-local128-exact-rejected.json) ·
[`rejected F16 exact block2 owner`](../benchmarks/results/2026-07-28-gfx1151-laguna-f16-block2-exact-rejected.json) ·
[`candidate F16 Q8 real-input screen`](../benchmarks/results/2026-07-28-gfx1151-laguna-f16-q8-real-input-candidate.json) ·
[`rejected F16 Q8 full-model screen`](../benchmarks/results/2026-07-28-gfx1151-laguna-f16-q8-full-model-rejected.json) ·
[`rejected Q4 pack8 geometry screen`](../benchmarks/results/2026-07-28-gfx1151-laguna-q4-pack8-decode-geometry-rejected.json) ·
[`tensorized SWA decode leaf`](../benchmarks/results/2026-07-28-gfx1151-laguna-swa-decode-hipblaslt-leaf.json) ·
[`fixed-K wall re-profile`](../benchmarks/results/2026-07-28-gfx1151-laguna-fixedk-wall-reprofile.json) ·
[`decode roofline/Qwen/Vulkan review`](../benchmarks/results/2026-07-28-gfx1151-laguna-decode-roofline-qwen-vulkan-review.json).

#### Next decode attack

The **18 tok/s** pause target is now complete under the existing state and
quality contract. Fixed-K LD-2 production reached
**16.391 tok/s**. The first post-reprofile LD-1 reducer win raises that to
**16.834 tok/s**, and the natural-shape global reducer reaches
**16.847 tok/s**. The retained natural-shape selected-MoE owner now reaches
**16.976 tok/s**. Its exact gate/up tile8 successor reaches
**17.007 tok/s**, and exact fused-GQA2 SWA reaches **17.065 tok/s**.
Exact fused one-head global attention now reaches **17.097 tok/s**.
Exact fused-GQA3/local384 SWA reaches **17.140 tok/s**. Reusing each staged
64-slot V tile across its three owned query heads reaches **18.032 tok/s** in
seven exact pairs and **18.027 tok/s** in clean production. Exact global GQA2
with the same staged-V reuse then reaches **18.237 tok/s** in seven exact
pairs and **18.230 tok/s** in clean production. The next bounded milestone is
**20 tok/s**. Vectorizing the saturated SWA V-stage copy then reaches
**18.806 tok/s** in seven exact pairs. Vectorizing the padded global V-stage
copy reaches **19.067 tok/s** in seven exact pairs; same-GGUF Vulkan at
**23.348 tok/s** remains the directional comparator target.

The post-global-GQA2 clean census records **816 dispatches/token**,
**52.577 ms** median kernel sum, and **55.154 ms** median dispatch span.
Exact attention is still the dominant implementation gap:

| Current family | hipEngine ms/token | Vulkan logger ms/token | Delta |
| --- | ---: | ---: | ---: |
| Source-F16 projections | **24.072** | **24.759** | **-0.687** |
| Attention total | **8.065** | **0.909** | **+7.155** |
| └ SWA fused GQA3 V-stage64 | **5.828** | **0.683** | **+5.145** |
| └ Global fused GQA2 V-stage64 | **2.237** | **0.227** | **+2.010** |
| Selected gate/up | **8.420** | **7.464** | **+0.956** |
| Selected down | **4.827** | **4.525** | **+0.302** |

Relative to the post-SWA-V-stage census, global GQA2 V-stage64 improves
global attention **2.878 -> 2.237 ms/token (-22.27%)**, total attention
**8.722 -> 8.065 ms (-7.54%)**, kernel sum **1.29%**, and kernel span
**1.26%** with dispatches unchanged. Source-F16 remains at comparator parity;
attention explains about **59.5%** of the remaining wall gap. Reaching the
existing **3.0-ms** attention checkpoint models about **20.1 tok/s**. The
next bounded screen is vectorized V staging in the retained SWA GQA3/local384
body; it owns **5.828 ms/token**, more than twice the global family. That
screen is retained: the leaf improves **20.19%**, and all seven p512/d128
pairs improve **18.244607 -> 18.806305 tok/s (+3.079%, -1.637 ms/token)**.

LD-1 is now underway with one retained exact substep. Grouping only the SWA
score owner by three query heads cuts its live-512 leaf **42.1-46.8%** and
moves matched production **14.563678 -> 14.740486 tok/s (+1.214%)**, with the
complete trajectory unchanged. Grouping the value reducer by nine heads, three
heads/local128, or three heads/local64 was exact but **5-11% slower** and has
been removed. This confirms the architecture seam: reuse K while preserving
the 72-head reducer grid; recovering V reuse still requires split-K/fused
ownership with enough workgroups for 40 CUs.

The next exact-arithmetic shortcut was also closed. Precomputing each head's
softmax weights once preserved the retained F32 context and BF16 gate bytes,
but live-512 score+prepare+value measured roughly **263-282 us**, slower than
the retained **220-229 us** score+wave-local value pair. Serial softmax
preparation and a global weight plane lose to redundant exponentials running
concurrently; the candidate and registry surface were removed.

LD-1b instead changes contraction ownership. A full-ring C1 leaf gathers
chronological BF16 K/V once, packs all nine query heads per KV head into two
F32 hipBLASLt contractions, applies a 72-wave visibility-aware softmax, and
uses the existing BF16 gate. At position 512 with ring wrap and explicit
eviction, context max error is **3.17e-8** and the standard numerical gate
passes. A 20-warmup/201-sample leaf improves complete gated SWA attention
**0.251302 -> 0.073779 ms/layer (3.406x)**. This is now the bounded runtime
candidate. Its clean p512/d128 screen improves
**14.754991 -> 16.526335 tok/s (+12.005%)**, and the complete 18-prompt lane
improves aggregate h16/h32 decode **11.862%/11.946%** with stable lifecycle.
It nevertheless fails quality: teacher-forced top-1 remains
**565/576 (98.09%)**, but max KL reaches **1.218229**, versus the **0.05**
contract, and only **36/54** free-running pairs remain exact. Production
therefore reverts to the exact split route and **14.740486 tok/s** retained
topline. The next bounded repair preserves the exact GQA3 QK reduction tree
and tests tensorized PV independently; no full-F32 QK default is admissible.
That repair is now implemented default-off: its chronological score plane is
F32-bit identical to the retained producer through wrap and explicit eviction,
and it computes exponentials in parallel while replaying the exact logical-slot
denominator order before tensorized PV. Only **1/9,216** deterministic gated
BF16 values differs from exact, versus three for exact-QK/wave-softmax and two
for the original full-F32 leaf. A clean whole-model speed and category screen
decides whether the remaining PV association is admissible. The answer is no:
clean p512/d128 improves **14.751829 -> 16.547822 tok/s (+12.175%)**, but the
complete lane reaches max KL **2.678710** at **566/576 (98.26%)** top-1.
PV accumulation association alone is enough to violate the contract. Close
tensorized all-layer attention and keep production exact. After the exact
local32 comparator also loses, remove the decode capability, session/runtime
owner, HIP widening/softmax/normalize helpers, and decode-specific category
mode. The retained rolling-SWA prefill hipBLASLt route is independent and
unchanged.

LD-1c then tested a new grid without changing an arithmetic operation: one
local32 workgroup owned 32 output dimensions for three adjacent query heads,
reused each BF16 V load across those heads, and replayed each head's retained
logical-slot softmax, FMA, divide, and gate order. Four dimension partitions
per query triple produced **8 KV heads x 3 triples x 4 = 96 workgroups/layer**,
but that still reduced the active waves carrying independent exp/FMA chains.
The candidate is F32/BF16 byte-exact and resource-clean at local32/VGPR24/
LDS0/scratch0, yet loses at every live length: live 1 is
**7.066 -> 8.290 us (+17.3%)**, live 128 is
**59.588 -> 108.323 us (+81.8%)**, and live 512 is
**237.661 -> 391.161 us (+64.6%)**. Remove it. Exact grouped-value reuse is
closed unless a future design preserves the retained 288-wave concurrency.

That preservation is now a retained exact win. The saturated-512 specialization
keeps all **72 local128 workgroups / 288 waves** and changes no scalar or FMA
operation; it only makes the natural 72Q/8KV/D128 ring and 512-slot bounds
compile-time constants. The complete score+reducer leaf falls
**0.108265 -> 0.081059 ms/layer (-25.13%)**. Seven counterbalanced p512/d128
pairs move current GQA3 **16.386231 -> 16.833740 tok/s (+2.731%, -1.622
ms/token)** with every candidate faster and every trajectory identical.
Cached tracing records **4,572 = 36 x 127** fixed reducers, no generic
fallback, and local128/VGPR16/LDS0/scratch0. Live below 512 retains the
generic exact owner.

The exact global counterpart is smaller but positive at every measured
production live count. It keeps the dynamic score/span ABI, all 48 local256
workgroups, and every scalar/FMA operation while specializing only
48Q/8KV/D128/capacity-4096 dimensions, scratch strides, and bounded address
arithmetic. Complete score+reduce improves **0.7-2.0%** at live 513/576/639.
Seven counterbalanced pairs move **16.832097 -> 16.846689 tok/s (+0.087%,
0.051 ms/token saved)** with every candidate faster and every trajectory exact.
Tracing records **1,524 = 12 x 127** fixed reducers, zero generic fallback,
and local256/VGPR24/LDS512/scratch0. Non-natural shapes/capacities and peer
backends retain the generic reducer.

LD-1d finally fuses the saturated SWA path without changing arithmetic. Five
local256 owners per KV head each carry two adjacent query heads, with the fifth
owner carrying the ninth head alone. The **40 workgroups / 320 wave32s per
layer** share a 6,144-byte exact score/physical plane, reuse each K vector
across a query pair, replay the retained slot-order maximum/exponent/
denominator and per-dimension FMA sequence, apply the same gate/stores, and
remove the global score round-trip plus one launch. Ring wrap and explicit
eviction are F32/BF16 byte-exact.

The cache-hot leaf regresses **0.079502 -> 0.081855 ms/layer (+2.96%)**, but
seven resident-model production pairs improve
**17.013184 -> 17.065241 tok/s (+0.306%, -0.179 ms/token)**, with every
candidate faster and every trajectory/state exact. The apparent contradiction
is the point: the 78.8-GB resident workload rewards five K reads per KV head,
where the leaf keeps K artificially hot. An exact one-head local256 fusion
improves that leaf **8.14%** but reads K nine times and regresses full
production **1.038%**; exact local128 one-head and parallel three-head
reducers also regress **8.19%/25.29%** in the leaf. Remove all three.
Cached tracing confirms the retained body at local256/VGPR32/SGPR128/
LDS6144/scratch0.

An exact paired shared-V follow-up is closed before runtime integration. It
kept the same 40-workgroup/eight-wave score phase and each query's scalar/FMA
order, but let four dimension waves carry both query-head states and fetch
each BF16 V element once. Modeled K+V row reads per KV head fell
**14 -> 10**, and ring-wrap/explicit-eviction F32/BF16 outputs remained
byte-exact. Halving the active PV waves and carrying two softmax/output states
instead regressed the complete saturated leaf
**0.083029 -> 0.123772 ms (+49.070%)**. The candidate is removed. This closes
same-workgroup paired-V serialization; it strengthens the llama.cpp-shaped
requirement to regain breadth with independent K64 splits.

The source-faithful GQA9/K64 screen then validates the performance premise but
rejects all three realizations. The online variant launches **8 KV heads x 8
K64 splits = 64 local256 workgroups**, reuses every K/V slice across all nine
query heads, and merges bounded online-softmax state. Its saturated leaf moves
**0.083713 -> 0.035521 ms/layer (-57.57%)**; cached tracing records the
intended local256/VGPR56/LDS5120/scratch0 main body plus a one-block merge.
Seven resident-model pairs move **17.089951 -> 19.292150 tok/s (+12.886%,
-6.679 ms/token)**, with every candidate sample faster. Changed association is
not admissible: the 128-token trajectory changes, and a same-stream
teacher-forced comparison reaches max KL **0.314247** despite
**125/128 (97.66%)** top-1.

The exact repair materializes the original score plane, replays one retained
slot-order softmax per head, and assigns 64 wave32 PV owners while preserving
every output FMA. It is F32/BF16 byte-exact and wins the leaf
**0.083382 -> 0.061297 ms (-26.49%)**, but its two extra dispatch boundaries
reverse the result in production: **17.081531 -> 16.451712 tok/s (-3.687%,
+2.241 ms/token)**, with all seven pairs losing. Fusing those phases into one
64-block cooperative local256 kernel remains byte-exact but regresses the leaf
**0.083669 -> 0.148301 ms (+77.25%)** because the score phase's block footprint
forces the scalar softmax/PV phases to carry seven idle waves. All GQA9
candidate code, registration, runtime selection, tests, and comparison plumbing
are removed. Production remains exact fused-GQA2 at **17.097044 tok/s**.

The next bounded LD-1 screen stayed inside that retained one-dispatch,
40-workgroup topology. It staged BF16 V through LDS while keeping all eight
waves on their original query/dimension ownership, reusing each V element
across the adjacent query pair without changing arithmetic. Both 64-slot and
128-slot batches are F32/BF16 byte-exact. They nevertheless regress the
complete leaf **19.04%/17.18%**: reducing modeled K+V row reads **14 -> 10**
does not repay LDS fill/read traffic and **16/8** additional block barriers.
Both candidates are removed.

The next exact repair returned to the one-kernel cooperative GQA9 design and
redistributed its post-grid-sync work. The rejected prototype assigned only
one PV wave per local256 block; the corrected screen maps all **72 query heads
x 4 dimension partitions = 288** exact scalar-softmax/PV tasks wave-major
across the 64 blocks, activating four or five waves per block. It preserves
every slot-order denominator/output FMA and uses one grid barrier.

That repair removes the old cooperative leaf disaster: retained fused-GQA2
**0.084802 ms** versus rebalanced GQA9 **0.085175 ms (+0.44%)**, both
F32/BF16 byte-exact. It still loses all seven resident-model production pairs,
**17.095757 -> 16.418310 tok/s (-3.963%, +2.414 ms/token)**, with exact
trajectories. The penalty nearly matches the exact three-dispatch repair's
**+2.241 ms/token**, identifying the per-layer global score/grid phase boundary
rather than idle PV waves as the remaining failure. `rocprofv3` crashes in
`hsa_signal_store_screlease` while tracing this cooperative launch and emits no
CSV, so no resource claim is made. Remove the candidate.

The next bounded exact screen remained inside the retained ordinary GQA2
block. One leader per head wrote the exact 512 weights and denominator to LDS;
the original four dimension waves then replayed the unchanged PV FMA sequence.
It is F32/BF16 byte-exact but neutral:
**0.084066 -> 0.083985 ms (-0.097%)**. The four redundant scalar chains already
execute concurrently, and exchanging them for one extra LDS barrier changes no
resident-model traffic. Remove it without a production screen.

The next ordinary-launch geometry, fused GQA3/local384, is retained. Nine query heads
divide exactly into three owners per KV head, and each workgroup's 12 waves map
directly to three heads x four dimension partitions. The layer therefore keeps
**24 x 12 = 288 active PV waves** versus retained GQA2's **40 x 8 = 320**, while
reducing K owners/reads per KV head **5 -> 3**. It preserves the local score
plane, one ordinary dispatch, exact arithmetic, and avoids all global phase
boundaries. Saturated ring wrap and explicit eviction are F32/BF16 byte-exact.
The cache-hot leaf is tied at **0.082480 -> 0.082807 ms (+0.397%)**, but every
resident-model pair wins: p512/d128 moves **17.100489 -> 17.139971 tok/s
(+0.231%, -0.135 ms/token)** with identical 128-token trajectories and state.
Cached tracing records 24 local384 blocks, VGPR104/SGPR128/LDS8192/scratch0.
gfx1151 selects it only at the natural saturated shape; fused-GQA2 remains the
exact rollback.

The direct cooperative-matrix follow-up explains which Vulkan ideas transfer
under Laguna's stricter quality contract. Compensated WMMA GQA9/K64 plus an
FP64 split merge cuts the leaf **0.18094 -> 0.03421 ms (-81.09%)** and moves
one full p512/d128 pair **17.118 -> 19.542 tok/s**, but the complete
18-prompt/576-step gate reaches max KL **1.754897** at **562/576** top-1.
Replacing WMMA QK with the exact retained dot tree still cuts the leaf
**54.56%**, but split-PV/softmax association reaches max KL **0.810355** at
**559/576** top-1. Preserving exact scores and replaying an ordered split
reducer instead regresses the leaf **21.58%**; a midpoint-classified per-head
repair changes 13 BF16 values and regresses **50.0%**. All four prototypes,
their dispatch surfaces, and the temporary quality harness are removed.

The retained successor takes the exact-safe Vulkan seam instead: tile V, but
do not change arithmetic association. The local384 body stages 64 contiguous
logical slots x D128 in LDS and each of its three query heads consumes the
same BF16 V tile. The 32/64/128-slot sweep improves the exact leaf
**26.38%/26.58%/22.85%**; 64 slots wins. Seven counterbalanced resident-model
pairs move **17.135411 -> 18.032171 tok/s (+5.233%, -2.902 ms/token)**, with
all seven candidate wins and identical 128-token hashes, tokens, positions,
and lifecycle. A focused CPU-reference gate remains F32/BF16 byte-exact
through positions 512-519 after ring wrap and explicit eviction. Cached
tracing records 24 local384 blocks, VGPR144/SGPR128/LDS24576/scratch0 and
cuts its four-observation kernel median **184.085 -> 137.197 us (-25.47%)**.
gfx1151 promotes V-stage64 only at the saturated natural shape; unstaged
local384 remains exact rollback.

LD-1e applies the same fusion to global attention while preserving breadth.
One local256 workgroup remains assigned to each of 48 query heads, so the
layer keeps **48 workgroups / 384 wave32s**. It fuses QK, the existing
eight-wave maximum/denominator partition and merge, PV, gate, and stores,
removing the score/physical round-trip and one launch without adding K reads.
The dynamic score/physical plane costs 8 bytes per live scan slot plus a
64-byte warp buffer. F32 context and gated BF16 output are byte-exact at live
257/513/576/639.

Complete leaves improve **17.55%/7.89%/7.93%** at live 513/576/639. Seven
resident-model pairs move **17.064962 -> 17.097044 tok/s (+0.188%, -0.110
ms/token)**, with every candidate faster and every trajectory/state exact.
Tracing records local256/VGPR24/SGPR128/scratch0. The superficially more
aggressive two-head GQA2 sibling halves K reads, but collapses the layer to 24
workgroups; it regresses resident production
**17.057510 -> 17.036046 tok/s (-0.126%)** and is removed. For global
attention on this 40-CU device, block breadth is worth more than GQA K reuse at
the current 513-639-token scan.

V staging reverses that earlier global result. The exact GQA2 body keeps its
24 local256 paired-head workgroups but adds a 64-slot x D128 LDS V tile, so
each load now feeds both per-head PV chains. At live 513/576/639 the
nine-sample leaf improves GQA1 **9.16%/12.39%/12.22%**, with F32 context and
gated BF16 byte-exact after explicit eviction. Seven resident-model pairs move
**18.034298 -> 18.237090 tok/s (+1.124%, -0.617 ms/token)**; every candidate
wins and the full 128-token trajectory/state is exact. Tracing names
`<2,64>` at local256/VGPR32/SGPR128/static-LDS512/scratch0, with 22,540-24,052
bytes of launch-time dynamic LDS over the measured live range. gfx1151
promotes GQA2 V-stage64 only at natural capacity/shape through live 4000;
GQA1 remains rollback above that LDS bound. Three clean selector-unset
p512/d128 runs measure **18.219717/18.230064/18.235007 tok/s**, median
**18.230064**, with identical generated IDs, final state, and allocation
lifecycle.

The saturated SWA copy-width successor preserves that complete compute body
and changes only its LDS staging transport: one aligned 16-byte transaction
now moves eight adjacent BF16 V values. The wrap/eviction oracle remains
byte-exact. The nine-sample leaf improves **0.133491 -> 0.106533 ms
(-20.19%)**, and all seven resident pairs improve **18.244607 -> 18.806305
tok/s (+3.079%)**. Tracing records `<64,true>` at 24 local384 blocks,
VGPR144/SGPR128/LDS30720/scratch0 and **161.50 -> 112.97 us (-30.05%)**.
Scalar V-stage64 remains exact rollback.
Three tracked-clean selector-unset runs measure
**18.801765/18.814192/18.815353 tok/s**, median **18.814192**. That is
**+3.204%** over prior clean 18.230064 and **+64.077%** over the 11.466687
sprint start, with identical IDs, final state, and allocation lifecycle.

The global copy-width successor preserves GQA2 compute and pads only the
dynamic score/physical prefix so each V-stage transaction is 16-byte aligned.
The live4000 LDS bound remains valid. Live513/576/639 leaves improve
**22.29%/25.82%/25.99%** byte-exactly; seven resident pairs improve
**18.794424 -> 19.066920 tok/s (+1.450%, -0.760 ms/token)**. Tracing records
`<2,64,true>` at local256/VGPR32/static-LDS512/scratch32 and
**141.09 -> 103.29 us (-26.79%)**. Scalar GQA2 remains rollback and GQA1
remains fallback above live4000. Three tracked-clean selector runs measure
**19.053726/19.065940/19.068436 tok/s**, median **19.065940**, with exact
IDs/state/lifecycle.

The post-vec16 clean census records **816 dispatches/token**, **50.238 ms**
kernel sum, and **52.814 ms** span. Attention is now **5.652 ms/token**:
**4.180 ms SWA + 1.472 ms global**, down **29.92%** from the pre-vec16
8.065-ms packet. Against Vulkan's logged 0.909 ms, attention still contributes
**4.742 ms** and about **49.3%** of the remaining wall gap. Re-screen vec16
stage geometry next, beginning with SWA stage128 versus stage64.

That wider stage is exact but rejected: **0.107446 -> 0.109075 ms (+1.516%)**.
The useful residue was in generated code instead. ISA shows the vec16
aggregate temporary occupying **384 x 16 = 6,144 B** of hidden LDS and
shuttling each load through that plane. Branching before assignment lets the
valid path store directly into the real V tile. The exact direct-store sibling
improves the leaf **0.107000 -> 0.105197 ms (-1.686%)**, traced kernel
**101.590 -> 99.227 us (-2.326%)**, and fixed LDS
**30,720 -> 24,576 B**; logical VGPR/SGPR also fall **143/36 -> 138/33**.
Seven resident-model pairs improve **19.070545 -> 19.083269 tok/s (+0.0667%,
-0.0350 ms/token)** with exact trajectories/state, so gfx1151 promotes it at
the saturated natural SWA shape. The old vec16 owner remains exact rollback.
Three tracked-clean selector-unset runs then measure
**19.072126/19.085294/19.089552 tok/s**, median **19.085294**. That is
**+0.1015% / -0.0532 ms/token** versus the prior clean 19.065940 packet and
**+66.441%** over the 11.466687 sprint start. Generated IDs, tokens,
positions, deterministic state, and allocation lifecycle remain exact.
Evidence:
[`clean direct-store production`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-gqa3-vstage64-vec16-direct-production.json).

The global vec16 owner has the same aggregate defect expressed as private
scratch: generated ISA contains one `scratch_load_b128` and two
`scratch_store_b128` operations, with a **32-byte** private segment per thread.
The exact direct-store sibling removes that shuttle without changing QK,
softmax, PV, gate, or store association. Evicted live513/576/639 F32/BF16
outputs remain byte-identical; nine-sample leaves improve
**11.71%/11.83%/11.82%**. Cached tracing improves
**103.354 -> 90.530 us (-12.41%)**, scratch falls **32 -> 0 B**, and all seven
p512/d128 pairs improve **19.077502 -> 19.134537 tok/s (+0.2990%,
-0.1562 ms/token)** with exact trajectories/state. gfx1151 now selects the
direct global form through live4000; the aggregate vec16 owner remains exact
rollback. Evidence:
[`retained global direct vec16 store`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-gqa2-vstage64-vec16-direct-retained.json).
Three tracked-clean selector-unset runs measure
**19.136600/19.146417/19.153280 tok/s**, median **19.146417**. That is
**+0.3203% / -0.1673 ms/token** versus the prior clean 19.085294 packet and
**+66.974%** over the sprint start, with exact IDs/state/lifecycle. Evidence:
[`clean global direct-store production`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-gqa2-vstage64-vec16-direct-production.json).

The clean post-direct-store census keeps **816 dispatches/token** and measures
**50.016 ms** kernel sum / **52.567 ms** span. Attention is now
**5.466 ms/token = 4.152 SWA + 1.314 global**. The global direct-store change
cuts global **10.70%**, total attention **3.28%**, kernel sum **0.44%**, and
span **0.47%** from the last census. Against Vulkan's **0.909 ms** attention,
the residual is **4.557 ms/token**, or **48.5%** of the remaining clean wall
gap. Direct-copy codegen cleanup is exhausted. Evidence:
[`post-direct-store wall census`](../benchmarks/results/2026-07-29-gfx1151-laguna-post-direct-store-wall-reprofile.json).

The next exact scalar seam removes generic exponential-domain work without
changing exponential arithmetic. Raw native and manually reconstructed
bounded exponentials improve leaf/wall but fail the complete category KL gate
at **1.452698/1.888082**, so both are removed. The retained successor keeps
compiler `expf` unchanged and asserts only the proven
`score - wave_max <= 0` invariant. Wrapped/evicted and leaf outputs are
F32/BF16 byte-exact. The leaf improves **0.106007 -> 0.097387 ms (-8.13%)**;
static instructions contract **3,196 -> 2,821 (-11.73%)** and cached tracing
improves **126.838 -> 91.812 us (-27.61%)** at unchanged
VGPR144/LDS24576/scratch0. Seven tracked-clean production pairs improve
**19.140826 -> 19.245912 tok/s (+0.549%, -0.285 ms/token)** with complete
sample separation and exact IDs/state. gfx1151 promotes this only at saturated
natural SWA; the generic-domain direct-store route remains rollback. Evidence:
[`retained exact exp-domain specialization`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-assume-exp-retained.json).
Three tracked-clean selector-unset runs then measure
**19.231940/19.248066/19.242300 tok/s**, median **19.242300**. That is
**+0.501% / -0.260 ms/token** versus the prior clean 19.146417 packet and
**+67.810%** over the 11.466687 sprint start, with exact IDs, state, and
lifecycle. Evidence:
[`clean exp-domain production`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-assume-exp-production.json).

The same exact compiler-domain fact is now admitted as a global-attention
primitive. It remains byte-exact to the retained direct-store route and CPU
oracle at live513/576/639 with explicit eviction. The leaf improves
**1.86-2.34%**, and a cached trace improves aggregate median
**90.490 -> 88.526 us (-2.17%)** at unchanged allocated VGPR32/scratch0.
Seven tracked-clean p512/d128 pairs improve
**19.235596 -> 19.243968 tok/s (+0.0435%, -0.0226 ms/token)** with all seven
candidate wins and exact generated IDs/state/lifecycle. gfx1151 promotes it
through live4000; explicit false restores the generic-domain direct-store
body. Evidence:
[`global exp-domain candidate`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-assume-exp-candidate.json) ·
[`global exp-domain retained`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-assume-exp-retained.json).
Three tracked-clean selector-unset runs then measure
**19.236922/19.250313/19.249443 tok/s**, median **19.249443**. That is
**+0.0371% / -0.0193 ms/token** versus the prior clean 19.242300 packet and
**+67.873%** over the sprint start, with exact IDs, state, and lifecycle.
Evidence:
[`clean global exp-domain production`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-assume-exp-production.json).

The next bounded attack is still the Vulkan-informed cooperative Br16/Bc64
QK/PV tile, but it must preserve compiler-`expf` results and each head's
ordered denominator/PV association or use an explicit high-precision repair
that clears the complete category gate. Do not reopen approximate exponential
or static layer-selection screens.

One prerequisite exact screen closes the ordinary-workgroup alternative.
The earlier scalar GQA2 V-stage64 body was never combined with the later
direct vec16 transport and exact exponential-domain specialization. That
40-local256 candidate is F32/BF16 byte-exact and improves the cache-hot leaf
**0.098081 -> 0.097236 ms (-0.861%)**, but allocates **176 VGPR** versus
retained GQA3's 144. All seven resident-model pairs lose
**19.249050 -> 19.182158 tok/s (-0.3475%, +0.1812 ms/token)**. Five K owners
per KV head and the register footprint outweigh broader block coverage once
the 78.8-GB model is resident. Remove the complete candidate; do not retry
ordinary GQA2 staging. The remaining exact source-shaped seam is a
normal-launch persistent K64 producer plus ordered reducer that avoids the
already-rejected cooperative-launch barrier. Evidence:
[`rejected GQA2 direct-vec16 staging`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-gqa2-vec16-rejected.json).

That persistent seam is exact but also closed. Thirty-two local256 workgroups
produce the comparator-shaped **8 KV heads x 8 K64** score tasks, rendezvous
through a monotonic device counter, then replay the retained ordered
maximum/compiler-`expf`/denominator/PV phase. Positions 512-519 plus explicit
eviction are F32/BF16 byte-exact. The leaf nevertheless regresses
**0.098299 -> 0.395157 ms (+301.99%)**. Tracing shows
**VGPR40/SGPR128/LDS0/scratch0**, so this is not a spill or occupancy accident:
the global score-plane writes/reads and grid rendezvous are the cost. The
candidate fails before a 78.8-GB resident-model run and is fully removed.

This sharpens rather than weakens the llama.cpp result. Vulkan is fast because
its subgroup64 cooperative-matrix **Br16 x Bc64** tile performs full-GQA QK,
online softmax state, and PV reuse as one tensorized operation; **64 K64
workgroups** preserve breadth, and only compact bounded state crosses the
merge. Copying split breadth while retaining Laguna's exact scalar score/PV
association creates a repair boundary more expensive than the retained fused
GQA3 kernel. The next admissible attention candidate must therefore do one of
two materially new things: keep broader grouped-Q reuse inside one fused
ordinary-workgroup phase, or tensorize QK/PV with an independently measured
high-precision repair that passes the complete category gate. Do not retry a
full score plane, cooperative/global phase boundary, or approximate online
merge. Evidence:
[`rejected persistent exact GQA9`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-persistent-gqa9-rejected.json).

The first one-phase compromise between the measured 24- and 40-block
endpoints is retained. Each KV head partitions its nine queries as
**2+2+2+3** across four local384 owners, producing **32 workgroups** per SWA
layer. Pair owners keep their four unused waves barrier-active during each
64-slot vec16 V stage but arithmetic-idle; all active heads preserve the
retained exact QK tree, compiler `expf`, ordered denominator/PV FMA, divide,
gate, and stores. There is no split state or global repair plane.

The wrapped/evicted oracle initially caught idle waves exiting before the
block-uniform V-stage barriers. Keeping them in the copy/barrier phase repairs
the implementation; positions 512-519 plus explicit eviction are then
F32/BF16 byte-exact. Nine cache-hot samples improve
**0.096586 -> 0.091360 ms (-5.41%)**. Cached tracing names the 32-block
`<64,true,true,true,true>` owner and improves **112.931 -> 105.717 us
(-6.39%)** at **VGPR104/SGPR128/LDS24576/scratch0**.

Seven resident p512/d128 pairs improve **19.268862 -> 19.371717 tok/s
(+0.534%, -0.276 ms/token)**; every candidate beats every control, with
identical 128-token trajectories, positions, and lifecycle state. gfx1151
promotes mixed32 only for the saturated natural SWA shape. The 24-block GQA3
owner remains exact rollback; shorter/non-natural and peer-backend routes are
unchanged. This is the useful exact form of the Vulkan lesson: increase
grouped reuse and grid breadth together while staying inside one fused phase.
Evidence:
[`retained mixed32 SWA owner`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-mixed32-retained.json).

Three tracked-clean selector-unset runs confirm
**19.353808/19.370310/19.368763 tok/s**, median **19.368763**. That is
**+0.620% / -0.320 ms/token** versus the preceding clean 19.249443 packet and
**+68.913%** over the 11.466687 sprint start. The mixed32 capability is active
without a comparison selector; IDs, state, and lifecycle remain exact.
Evidence:
[`clean mixed32 production`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-mixed32-production.json).

The clean post-promotion trace keeps **816 dispatches/token** and measures
**49.432 ms/token** kernel sum / **51.982 ms/token** span. Attention is now
**4.873 ms/token = 3.583 SWA + 1.280 global**, down **10.84%** from the
preceding 5.466-ms census. Same-GGUF llama.cpp Vulkan remains at **0.909 ms**,
so attention still leaves **3.964 ms/token** and **45.0%** of the complete
clean wall gap on the table. The priority therefore does not change: audit
the comparator's fused cooperative dataflow against mixed32, then admit only
a structurally new exact or fully category-gated candidate.
Evidence:
[`post-mixed32 wall census`](../benchmarks/results/2026-07-29-gfx1151-laguna-post-mixed32-wall-reprofile.json).

The first post-census Vulkan transfer is exact but rejected. Vulkan computes
one probability tile in LDS `Psh` and consumes it with cooperative PV; the
mixed32 scalar body recomputes softmax in four dimension waves. A four-way
P-cache candidate computes the maximum once, divides independent compiler
`expf` calls across those waves, overwrites the dead LDS score plane with
weights, and replays the denominator once in original slot order. Wrapped and
evicted F32/BF16 output is byte-exact, but the leaf regresses
**0.091439 -> 0.097556 ms (+6.690%)**. Resources stay VGPR104/scratch0 while
LDS rounds **24,576 -> 25,088 B**. Three new full-workgroup barriers and LDS
traffic outweigh removing arithmetic that already overlaps across SIMD
waves. Remove the candidate completely. Do not retry scalar `Psh`; the next
transfer must couple probability reuse to tensorized PV or avoid a
block-wide synchronization boundary.
Evidence:
[`rejected mixed32 P-cache4`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-mixed32-pcache4-rejected.json).

The barrier-free Vulkan transfer is retained. Instead of lane 0 issuing four
independent compiler `expf` calls serially for each four-slot batch, mixed32
exp4 lets lanes 0..3 issue one each and shuffles the results back to every
lane. Lane 0 still adds weights into the denominator in the original item
order, and every dimension keeps the original PV FMA order. There is no new
LDS, barrier, launch, or repair plane. Wrapped/evicted F32/BF16 output is
byte-exact. The leaf improves **0.091487 -> 0.089135 ms (-2.57%)** and the
stable cached kernel window improves **85.414 -> 83.584 us (-2.14%)** at
unchanged **32 local384 blocks, VGPR104/SGPR128/LDS24576/scratch0**. All seven
resident p512/d128 pairs improve
**19.368030 -> 19.432503 tok/s (+0.333%, -0.171 ms/token)**; every candidate
beats every control and the complete generated trajectory, positions, and
lifecycle state are exact. gfx1151 now selects exp4 only within the
already-qualified saturated mixed32 route. The serial-exp mixed32 sibling
remains explicit rollback and other shapes/backends are unchanged. Evidence:
[`retained mixed32 exp4`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-mixed32-exp4-retained.json).

Three tracked-clean selector-unset runs confirm
**19.417147/19.424487/19.429963 tok/s**, median **19.424487**. That is
**+0.288% / -0.148 ms/token** versus clean serial-exp mixed32 and
**+69.399%** over the 11.466687 sprint start. The exp4 capability is active
without a comparison selector; IDs, state, and lifecycle remain exact.
Evidence:
[`clean mixed32 exp4 production`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-mixed32-exp4-production.json).

The clean post-exp4 trace keeps **816 compute dispatches/token** and measures
**49.296 ms/token** kernel sum / **51.850 ms/token** span. Attention is now
**4.745 ms/token = 3.448 SWA + 1.297 global**. Relative to serial-exp mixed32,
SWA falls **3.78%**, total attention **2.64%**, and span **0.25%**. Same-GGUF
llama.cpp Vulkan remains at **0.909 ms**, so attention still leaves
**3.835 ms/token** and **44.3%** of the complete clean wall gap on the table.
Evidence:
[`post-exp4 wall census`](../benchmarks/results/2026-07-29-gfx1151-laguna-post-exp4-wall-reprofile.json).

The exact exp8 successor is retained. Lanes 0..7 issue one independent
compiler `expf` per eight-slot batch and shuffle the weights back to every
lane. Lane 0 still accumulates the denominator in original item order and
every dimension retains the original PV FMA order; no LDS, barrier, launch,
or repair plane is added. Wrapped/evicted F32/BF16 output is byte-exact. The
leaf improves **0.089191 -> 0.083755 ms (-6.09%)**, and the stable cached
kernel window improves **83.557 -> 78.667 us (-5.85%)** at unchanged
**32 local384 blocks, VGPR104/SGPR128/LDS24576/scratch0**. All seven resident
p512/d128 pairs improve
**19.427449 -> 19.510986 tok/s (+0.430%, -0.220 ms/token)**; every candidate
beats every control with exact trajectory/state/lifecycle. gfx1151 selects
exp8 only inside the already-qualified exp4/mixed32 route. Evidence:
[`retained mixed32 exp8`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-mixed32-exp8-retained.json).

Three tracked-clean selector-unset runs confirm
**19.496106/19.515697/19.519033 tok/s**, median **19.515697**. That is
**+0.470% / -0.241 ms/token** versus clean exp4 and **+70.195%** over the
11.466687 sprint start. The exp8 capability is active without a comparison
selector; IDs, state, and lifecycle remain exact. Evidence:
[`clean mixed32 exp8 production`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-mixed32-exp8-production.json).

The exact exp16 successor is retained. Lanes 0..15 issue one independent
compiler `expf` per sixteen-slot batch while lane 0 and each dimension retain
their original denominator/PV order. Wrapped/evicted F32/BF16 output remains
byte-exact. The leaf improves **0.083740 -> 0.082224 ms (-1.81%)**, and the
stable cached window improves **78.814 -> 77.265 us (-1.97%)** with unchanged
**32 local384 blocks, VGPR104/SGPR128/LDS24576/scratch0**. All seven resident
p512/d128 pairs improve
**19.506557 -> 19.523370 tok/s (+0.0862%, -0.0441 ms/token)**; every candidate
beats every control with exact trajectory/state/lifecycle. gfx1151 selects
exp16 only inside the qualified exp8 route. Evidence:
[`retained mixed32 exp16`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-mixed32-exp16-retained.json).

Three tracked-clean selector-unset runs confirm
**19.514684/19.538643/19.530105 tok/s**, median **19.530105**. That is
**+0.0738% / -0.0378 ms/token** versus clean exp8 and **+70.320%** over the
11.466687 sprint start. The exp16 capability is active without a comparison
selector; IDs, state, and lifecycle remain exact. Evidence:
[`clean mixed32 exp16 production`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-mixed32-exp16-production.json).

The bounded issue-width screen closes with exact wave32 issue. All lanes
compute one compiler `expf` per thirty-two-slot batch while lane 0 and each
dimension retain the original denominator/PV order. The leaf improves
**0.082313 -> 0.081551 ms (-0.93%)**, and the stable cached window improves
**77.185 -> 76.838 us (-0.45%)** with unchanged
**32 local384 blocks, VGPR104/SGPR128/LDS24576/scratch0**. All seven resident
p512/d128 pairs improve
**19.524103 -> 19.538164 tok/s (+0.0720%, -0.0369 ms/token)**; every candidate
beats every control with exact state. gfx1151 selects exp32 only inside the
qualified exp16 route. Evidence:
[`retained mixed32 exp32`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-mixed32-exp32-retained.json).

Three tracked-clean selector-unset runs measure
**19.521938/19.530839/19.533770 tok/s**, median **19.530839**. This is
aggregate-flat at **+0.0038% / -0.0019 ms/token** versus clean exp16; retain
on the fully separated seven-pair A/B and positive leaf/trace. Production is
**+70.327%** over sprint start and IDs/state/lifecycle remain exact. Evidence:
[`clean mixed32 exp32 production`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-mixed32-exp32-production.json).

The clean post-wave32 census keeps **816 dispatches/token** and measures
**48.966 ms/token** kernel sum / **51.519 ms/token** span. Attention is now
**4.478 ms/token = 3.181 SWA + 1.289 global**. Since exp4, SWA falls
**7.74%**, total attention **5.62%**, and span **0.64%**. Same-GGUF llama.cpp
Vulkan remains at **0.909 ms**, leaving **3.568 ms/token** and **42.6%** of
the complete wall gap in attention. Evidence:
[`post-exp32 wall census`](../benchmarks/results/2026-07-29-gfx1151-laguna-post-exp32-wall-reprofile.json).

The exact fused GQA4+5/local512 screen transfers more of llama.cpp's GQA
reuse without copying its inexact tensorized arithmetic, but it is rejected.
F32 context and gated BF16 are byte-exact; nevertheless, cutting ordinary-grid
SWA workgroups **32 -> 16** regresses the leaf
**0.081615 -> 0.086053 ms (+5.44%)**. The added K/V reuse cannot repay
gfx1151 underfill even without a global score plane, rendezvous, atomics, or
reducer launch. The implementation is removed before tracing/runtime
integration. Evidence:
[`rejected GQA4+5 local512`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-gqa45-local512-rejected.json).

The final sub-32 ordinary-grid screen is also rejected. Applying the retained
exact wave32 exponential issue to the existing 24-workgroup **3+3+3 GQA3**
owner remains byte-exact, but regresses production's 32-workgroup
**2+2+2+3** leaf **0.083732 -> 0.086531 ms (+3.34%)**. The shorter softmax
does not move the occupancy/reuse seam; the implementation is removed and
ordinary-grid SWA owners below 32 workgroups are closed. Evidence:
[`rejected 24-block GQA3 exp32`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-gqa3-exp32-rejected.json).

The exact global wave32 issue sibling is retained. Each of the eight waves
still owns the same strided token chain, while its 32 lanes issue independent
compiler `expf` calls and shuffle weights back to lane 0 for the original
token-order sum. F32 context and gated BF16 remain byte-exact at evicted
live513/576/639. Leaves improve **2.25%/3.22%/3.79%**; cached tracing improves
**88.486 -> 85.601 us (-3.26%)** at VGPR56/SGPR128/LDS512/scratch0. All seven
resident p512/d128 pairs improve
**19.547209 -> 19.556569 tok/s (+0.0479%, -0.0245 ms/token)** with exact
trajectory, positions, determinism, and lifecycle. gfx1151 selects exp32
only inside the qualified natural global direct-store/assume-exp route; the
serial-issue sibling remains rollback. Evidence:
[`retained global GQA2 exp32`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-gqa2-exp32-retained.json).

Three tracked-clean selector-unset runs confirm
**19.544652/19.561715/19.565127 tok/s**, median **19.561715**. That is
**+0.1581% / -0.0808 ms/token** versus clean wave32-SWA production and
**+70.596%** over the 11.466687 sprint start. The exp32 capability is active
without a comparison selector; IDs, state, and lifecycle remain exact.
Evidence:
[`clean global GQA2 exp32 production`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-gqa2-exp32-production.json).

The clean post-global census confirms the leaf transfer. At unchanged
**816 dispatches/token**, global attention falls
**1.288661 -> 1.244374 ms/token (-3.44%)**; complete attention falls
**4.477845 -> 4.427829 ms/token (-1.12%)**, kernel sum is
**48.954823 ms/token**, and dispatch span is **51.493134 ms/token**.
Same-GGUF llama.cpp Vulkan remains at **0.909423 ms/token** attention.
The residual **3.518406-ms** attention gap is still **42.4%** of the clean
**8.290741-ms/token** wall gap, so attention remains the first comparative
priority. Evidence:
[`post-global-exp32 wall census`](../benchmarks/results/2026-07-29-gfx1151-laguna-post-global-exp32-wall-reprofile.json).

Exact scalar issue/ownership work is now closed for both attention families.
The last untested exact reuse permutation is also closed. A 32-workgroup
GQA9/D32 owner keeps full grid breadth and removes the split-softmax merge by
recomputing exact QK in each of four dimension shards, reducing staged V
traffic about fourfold. It is byte-exact through wrap and eviction, but
regresses the retained mixed32 exp32 leaf
**0.081902 -> 0.138907 ms (+69.60%)**. Redundant QK and nine-head
register/serial pressure dominate the V saving, so the candidate is removed.
Evidence:
[`rejected GQA9/D32`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-gqa9-dim32-rejected.json).

The first post-census tensorized state repair is also rejected. The
compensated-WMMA **GQA9 x K64** tile now carries each split's unnormalized PV
numerator and applies the denominator only once in the FP64 merge, removing
the rejected predecessor's local divide/global multiply round trip. It keeps
the material leaf win, **0.081649 -> 0.034335 ms (-57.95%)**, and lowers the
prior all-layer maximum KL **1.754897 -> 1.426066 (-18.73%)**. That is still
**28.52x** the admitted 0.05 ceiling. The full 18-prompt/576-step gate records
**555/576 (96.35%)** top-1, but every category exceeds the KL budget, so the
kernel, wrapper, diagnostic selector, oracle, and harness are removed.
Evidence:
[`rejected raw-numerator WMMA`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-wmma-raw-numerator-rejected.json).

Output-midpoint repair is also closed. A guarded sibling runs the fast
raw-numerator tile, detects approximate gated values near a BF16 midpoint, and
replays the retained exact mixed32 owner only for ambiguous 2/3-head groups.
Guards of **64/128/256/512** F32 low-mantissa units still leave
**9/4/2/1** BF16 mismatches on the wrap/eviction fixture. The first byte-exact
guard, **1024**, repairs so broadly that the leaf regresses
**0.081869 -> 0.118805 ms (+45.12%)**. The kernel, wrapper, oracle seam, and
harness choice are removed before runtime integration. Evidence:
[`rejected guarded WMMA repair`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-wmma-guarded-repair-rejected.json).

Component-fine repair does not rescue that premise. A second bounded sibling
compacts ambiguous dimensions per head, shares exact QK/softmax across each
2/3-head owner, and replays PV only for the compact list. Guards covering
**50%**, **75%**, and **87.5%** of the BF16 low interval still miss exact
output changes on the wrap/eviction fixture. Only the complete **32768**
interval is byte exact, which reduces the candidate to raw WMMA plus complete
exact replay and regresses **0.081644 -> 0.183208 ms (+124.40%)**. All
candidate code is removed. Evidence:
[`rejected component repair`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-wmma-component-repair-rejected.json).

The llama.cpp review exposes one exact scheduling opportunity that the prior
global ownership conclusion missed. Production already used the 24-block GQA2
exp32 path; the relevant comparison was therefore not the old 48-block GQA1
kernel, but **24 versus 32 resident cooperative blocks**. A new global
mixed32 owner partitions each six-query GQA group as **2+2+1+1**. Pair and
singleton blocks preserve the same QK products, eight-wave maximum and
denominator association, ordered PV chain, gate, and stores. Singleton idle
waves still participate in every staged-V barrier.

The live513/576/639 eviction fixture is byte-exact in both F32 context and
gated BF16 output. Nine-sample leaves improve
**0.080514 -> 0.076339 ms (-5.19%)**,
**0.091570 -> 0.083888 ms (-8.39%)**, and
**0.100351 -> 0.091928 ms (-8.39%)**. Cached tracing names the intended
32-block/local256 specialization at VGPR56/SGPR128/static-LDS512/scratch0.
All seven resident p512/d128 pairs win
**19.641357 -> 19.668893 tok/s (+0.1402%, -0.0713 ms/token)** with tokens
2930/74107, trajectory SHA `94f803f7...ebda32`, positions, determinism, and
lifecycle exact. gfx1151 now selects mixed32 only inside the qualified
capacity-4096/live<=4000 GQA2-exp32 route; the 24-block owner remains exact
rollback. Evidence:
[`retained global mixed32 exp32`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-mixed32-exp32-retained.json).

Three tracked-clean selector-unset runs on `ab2ea899c` measure
**19.660256/19.667705/19.670663 tok/s**, median **19.667705**. That is
**+0.1917% / -0.0975 ms/token** versus clean Q4-dual production and
**+71.520%** over the 11.466687 sprint start. Mixed32 is active without a
comparison selector; all three runs preserve tokens 2930/74107, trajectory,
final position, determinism, and zero tracked allocations after teardown.
Evidence:
[`clean global mixed32 production`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-mixed32-exp32-production.json).

The bounded **40-block 2+1+1+1+1** continuation is rejected. It is F32/BF16
byte-exact and improves live513 **0.076805 -> 0.073259 ms (-4.62%)**, but the
extra fifth K/V owner crosses the reuse/occupancy seam at the representative
live576/live639 points: **+0.21%/+0.11%**. The candidate is removed before
trace/runtime integration. Evidence:
[`rejected global mixed40 exp32`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-mixed40-exp32-rejected.json).

Keeping four K/V owners but separating pair and singleton work is also
decisively rejected. The two-launch split32 candidate uses 16 pair-owner
local256 blocks plus 16 singleton-owner local128 blocks; the latter emulate
the exact eight-wave score reduction on four physical waves and remove idle
singleton PV waves. F32 context and gated BF16 output remain byte-exact at
evicted live513/576/639, but leaf latency regresses
**0.076609 -> 0.185596 ms (+142.26%)**,
**0.083784 -> 0.213137 ms (+154.39%)**, and
**0.091793 -> 0.233758 ms (+154.66%)**. The second launch and virtual-wave
replay overwhelm the saved output work. All candidate code is removed and the
production source is restored byte-for-byte. Evidence:
[`rejected global split32 exp32`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-split32-exp32-rejected.json).

The remaining attention sequence is:

1. Re-profile the clean retained-mixed32 attention wall. **Complete:** the
   tracked-clean 127-transition census records **768 compute + 5 runtime-copy
   dispatches/token**, **48.701 ms/token** kernel sum, and **52.205 ms/token**
   dispatch span. Attention is **4.365 ms/token = 3.183 SWA + 1.182 global**.
   Global mixed32 transfers a **5.00%** family reduction versus the preceding
   GQA2-exp32 census; SWA is flat. Against same-GGUF Vulkan's
   **0.909 ms/token**, attention still leaves **3.456 ms/token**, or **43.1%**
   of the complete **8.015-ms/token** publication-wall gap. Evidence:
   [`post-global-mixed32 wall census`](../benchmarks/results/2026-07-29-gfx1151-laguna-post-global-mixed32-wall-reprofile.json).
2. Target the **3.183-ms saturated SWA family** with one launch. The bounded
   packed-BF16 dot2 screen is **complete and rejected**. The two-term F32
   decomposition regresses the leaf **1.05%**. Dropping the residual buys only
   **0.17%** and fails the canonical 18-prompt/576-step gate at max KL
   **1.265727**, **25.31x** the 0.05 ceiling, despite **564/576** top-1.
   The dot2 body, wrapper, selector, oracle seam, and harness are removed.
   Evidence:
   [`rejected QK dot2`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-qkdot2-rejected.json).
3. The first comparator-audit structural screen is **complete and rejected**.
   It remaps the current exact exp32 template to 40 local256 GQA2 blocks under
   the retained local384 launch bound. That fixes the predecessor's
   **176 -> 104 VGPR** footprint, but the fifth K/V owner still regresses the
   leaf **0.081815 -> 0.086925 ms (+6.25%)**. The candidate is removed before
   production; ordinary 40-block GQA2 is closed independently of register
   pressure. Evidence:
   [`rejected current-template GQA2`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-gqa2-exp32-current-template-rejected.json).
4. The exact synchronization screen is **complete and rejected**. Two
   ping-pong 64-slot V buffers preserve every QK/softmax/PV operation and
   reduce staged-V block barriers **16 -> 8**, but improve the leaf only
   **0.081569 -> 0.081210 ms (-0.44%)**. Static LDS rises
   **24,576 -> 40,960 bytes** and clang allocates **104 -> 224 VGPRs** under
   both before-consume and after-consume copy schedules. It fails the
   predeclared >=5% gate; all candidate code is removed before production.
   Evidence:
   [`rejected V-stage64 ping-pong`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-vstage64-pingpong-rejected.json).
5. The structural precision screen is **complete and rejected**. The exact
   layer pattern is `FULL,SWA,SWA,SWA`, so the predeclared candidate applies
   the already-proven exact-QK/tensorized-PV path to each complete 12-layer
   SWA role (`layer_id mod 4 = 1,2,3`) rather than arbitrary layer IDs.
   Across the complete 18-prompt/576-step gate, roles 1/2/3 reach max KL
   **1.590854/1.690376/4.873391** at
   **562/561/560 of 576** top-1. All fail the 0.05 KL ceiling despite only
   **3.94-3.98%** directional decode speedup. The historical diagnostic
   worktree is discarded; no candidate code or default reaches current main.
   This closes tensorized PV even on an architecture-defined one-third SWA
   scope. Do not bisect arbitrary layer IDs. Evidence:
   [`rejected structural-role tensorized PV`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-tensorized-pv-role-rejected.json).
6. Separating the retained mixed owner by its natural query counts is
   **complete and rejected**. The candidate preserves every arithmetic
   operation but replaces one 32-block local384 dispatch with 24 local256
   pair owners followed by eight local384 triple owners. The wrap/eviction
   oracle is F32/BF16 byte-exact, yet the leaf regresses
   **0.081796 -> 0.178297 ms (+117.98%)**. Eliminating four inactive output
   waves from each pair owner cannot repay two sequential underfilled grids.
   All candidate code is removed; do not retry mixed32 as separate launches.
   Evidence:
   [`rejected mixed32 pair/triple split`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-mixed32-pair-triple-rejected.json).
7. Compiling the unchanged exact body for physical wave64 is **complete and
   rejected**. Width-32 shuffle segments preserve the complete F32/BF16
   result byte-for-byte, but two non-profiled screens regress the leaf
   **3.81%** and **3.56%**. Cached tracing keeps local384/LDS24576/scratch0
   but raises VGPR allocation **104 -> 112**. llama.cpp's wave64 speed is
   therefore inseparable from its wave64-native Br16 x Bc64 cooperative tile;
   physical wave64 alone does not transfer. Evidence:
   [`rejected physical wave64 mixed32`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-mixed32-wave64-rejected.json).
8. Sharing softmax across the two width-32 halves of each physical wave64 is
   **complete and rejected**. Native `ds_bpermute` makes the maximum, exp32
   weights, denominator, F32 context, and gated BF16 result byte-exact without
   LDS or barriers, but the leaf regresses
   **0.081713 -> 0.085229 ms (+4.30%)**. Cross-half permutation and wave64
   issue overhead cost more than the duplicated softmax work. All candidate
   code is removed. Evidence:
   [`rejected wave64 shared softmax`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-wave64-shared-softmax-rejected.json).
9. Reusing the score producers for the exact maximum is **complete and
   retained as a registered primitive**. Each of the twelve score waves
   accumulates one partial maximum per owned query while it already owns the
   score, publishes 36 values before the existing phase barrier, and lets
   each output owner reduce twelve partials instead of rescanning all 512
   scores. QK, score storage, exp32 issue, denominator/PV association, gate,
   stores, grid, and barriers are unchanged. The wrap/eviction oracle is
   F32/BF16 byte-exact; the leaf improves
   **0.081790 -> 0.059101 ms (-27.74%)**. Cached tracing keeps
   local384/VGPR104/SGPR128/scratch0 and raises LDS only
   **24,576 -> 25,088 bytes**. The resident gate then improves all seven
   p512/d128 pairs **19.684442 -> 19.996117 tok/s
   (+1.583%, -0.792 ms/token)** with exact trajectories/state/lifecycle and
   complete sample separation. gfx1151 promotes the candidate only for the
   saturated natural SWA shape; mixed32/exp32 remains rollback. Evidence:
   [`retained producer-max primitive`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-producer-max-leaf.json).
   [`retained producer-max runtime`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-producer-max-retained.json).
10. Publish a tracked-clean selector-unset three-run p512/d128 result.
    **Complete:** **19.978220/19.990914/19.983610 tok/s**, median
    **19.983610**, is **+1.606% / -0.804 ms/token** over the prior clean
    19.667705 packet and **+74.275%** over sprint start. The capability is
    active without a comparison selector and repeated state/lifecycle is
    exact. Evidence:
    [`producer-max production`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-producer-max-production.json).
11. Re-profile the clean attention wall against llama.cpp Vulkan.
    **Complete:** the 127-transition census keeps 768 compute dispatches/token
    and cuts SWA **3.183 -> 2.503 ms/token (-21.36%)**, total attention
    **4.365 -> 3.688 ms (-15.52%)**, kernel sum
    **48.701 -> 47.990 ms (-1.46%)**, and span
    **52.205 -> 50.434 ms (-3.39%)**. Global is flat at **1.178 ms/token**.
    Against Vulkan's **0.909 ms/token** attention, the residual
    **2.778-ms/token** attention gap is **38.5%** of the complete wall gap.
    Evidence:
    [`post-producer-max census`](../benchmarks/results/2026-07-29-gfx1151-laguna-post-producer-max-wall-reprofile.json).
12. Transfer score-producer maxima to global attention. **Complete and
    retained:** the eight existing score waves now
    publish their already-ordered partial maxima through the score barrier.
    This removes the materialized-score reread and one barrier without new
    LDS. The live513/576/639 eviction oracle is F32/BF16 byte-exact and leaves
    improve **4.50%/4.89%/4.88%**; tracing keeps
    grid8192/local256/LDS512/scratch0 and lowers VGPR **56 -> 48**. All seven
    resident p512/d128 pairs improve
    **19.978296 -> 19.993586 tok/s (+0.0765%, -0.0383 ms/token)** with
    complete sample separation and exact trajectories/state/lifecycle.
    gfx1151 promotes the qualified route; mixed32/exp32 remains rollback.
    Evidence:
    [`global producer-max leaf`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-producer-max-leaf.json).
    [`global producer-max retained`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-producer-max-retained.json).
    Tracked-clean selector-unset production is
    **19.982796/19.988868/19.986371 tok/s**, median **19.986371**:
    aggregate-flat-to-positive at **+0.0138%** versus the prior packet and
    exact across repeated state/lifecycle. Evidence:
    [`global producer-max production`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-producer-max-production.json).
13. Re-profile after global producer maxima. **Complete:** the clean
    127-transition trace keeps 768 compute dispatches/token and moves global
    attention **1.178 -> 1.149 ms/token (-2.50%)**, total attention
    **3.688 -> 3.658 ms (-0.80%)**, kernel sum
    **47.990 -> 47.956 ms (-0.07%)**, and span
    **50.434 -> 50.405 ms (-0.06%)**. SWA is flat at **2.497 ms/token**.
    Against Vulkan's **0.909 ms/token** attention, the residual
    **2.749-ms/token** attention gap is **38.2%** of the complete wall gap.
    Evidence:
    [`post-global-producer-max census`](../benchmarks/results/2026-07-29-gfx1151-laguna-post-global-producer-max-wall-reprofile.json).
14. After that gate, revisit llama.cpp's cooperative matrix GQA tile as a
    precision-design problem: retain its compact GQA9/K64 ownership and
    tensorized QK/PV throughput, but establish an independently valid
    score/numerator error bound or a higher-precision cooperative
    accumulation before any whole-model quality run. The measured scalar
    split merge, global score plane, output-derived repair, packed-BF16 dot2,
    and synchronization-only variants remain closed.
15. Screen exact scalar whole-GQA ownership before changing arithmetic.
    **Complete and rejected:** one local384 block per KV head stages all nine
    queries and computes each K/V/exp32 value once while preserving the exact
    denominator and PV order. Despite byte identity, the leaf regresses
    **0.058989 -> 0.138660 ms (+135.1%)**. Tracing exposes only eight blocks,
    VGPR224, LDS44,544, and scratch0. Preventing full loop unrolling worsens
    the leaf to **0.173172 ms (+192.8%)**. All candidate code is removed.
    This confirms that llama.cpp's whole-GQA ownership is inseparable from
    cooperative-matrix parallelism; scalar traffic reuse underfills and
    serializes gfx1151. Evidence:
    [`rejected scalar GQA9`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-gqa9-shared-scalar-rejected.json).
16. Hoist score scaling into the producer. **Complete and rejected:** this
    removes four repeated `dot * scale` evaluations per query/token, but
    production fuses `dot * scale - max`; materializing the scaled score
    rounds before subtraction. F32 context differs by up to **2.79e-9** while
    the leaf is neutral at
    **0.059183 -> 0.059172 ms (-0.018%)**. The non-exact result does not
    justify a model-quality run, and all candidate code is removed. Evidence:
    [`rejected producer-scaled scores`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-producer-scaled-scores-rejected.json).
17. Publish the per-head softplus gate from the score phase. **Complete and
    retained as a registered primitive:** one thread computes the identical
    gate once per owned query and reuses the existing score-to-output barrier.
    The wrap/eviction oracle is F32/BF16 byte-exact; the leaf improves
    **0.059058 -> 0.058680 ms (-0.641%)**. Tracing keeps
    grid32/local384/VGPR104/SGPR128/LDS25,088/scratch0. Production stays
    **19.986371 tok/s** until a matched resident gate. That gate is now
    complete: all seven exact pairs improve and median decode moves
    **19.992650 -> 20.012052 tok/s (+0.097%)**, so gfx1151 promotes the
    specialization. Clean selector-unset production is
    **19.991789/20.003064/20.005123 tok/s**, median **20.003064**:
    **+0.0835% / -0.0418 ms/token** over the prior 19.986371 packet and
    **+74.445%** over sprint start, with exact repeated state/lifecycle.
    Evidence: [`producer-gate leaf`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-producer-gate-leaf.json),
    [`producer-gate retained`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-producer-gate-retained.json),
    [`producer-gate production`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-producer-gate-production.json).
18. Re-profile and close the comparison seam. **Complete:** the clean
    127-transition census keeps 768 compute dispatches/token and measures
    SWA **2.497126 -> 2.490833 ms/token (-0.252%)**, attention
    **3.658463 -> 3.657386 ms (-0.029%)**, kernel sum
    **47.955880 -> 47.935239 ms (-0.043%)**, and span
    **50.405381 -> 50.368946 ms (-0.072%)**. Resources remain
    local384/VGPR104/LDS25,088/scratch0. Against Vulkan's
    **0.909423 ms/token** attention, the residual **2.747963-ms/token** gap is
    **38.36%** of the full wall gap. Remove the temporary session/profile
    comparison seam; retain the architecture capability, cache owner, and
    producer-max rollback. Evidence:
    [`post-producer-gate census`](../benchmarks/results/2026-07-29-gfx1151-laguna-post-producer-gate-wall-reprofile.json).
19. Parallelize the exact selected-Q4 tile8 tail. **Complete and promoted:**
    the retained resident-T16 body had eight independent column reductions but
    completed all of them serially on thread 0. Lanes 0..7 now finish one
    column each while preserving every K/FMA owner, wave32 reduction, ordered
    wave0..3 sum, and BF16 store boundary. The actual layer-1 gate/up leaf
    improves **0.130259 -> 0.128862 ms (-1.072%)**, with all 21 pairs positive
    and zero output mismatches. All seven exact resident p512/d128 pairs then
    improve **19.998518 -> 20.007478 tok/s (+0.0448%)**. Clean selector-unset
    production is **19.996444/20.007890/20.020236 tok/s**, median
    **20.007890**: **+0.0241% / -0.0121 ms/token** over the prior packet and
    **+74.487%** over sprint start. Cached tracing records all **5,969**
    selected gate/up calls in the `true` specialization at
    grid16384x10/local128, VGPR96/SGPR128/LDS512/scratch0. Evidence:
    [`parallel-tail leaf`](../benchmarks/results/2026-07-29-gfx1151-laguna-selected-tile8-parallel-tail-leaf.json),
    [`parallel-tail retained`](../benchmarks/results/2026-07-29-gfx1151-laguna-selected-tile8-parallel-tail-retained.json),
    [`parallel-tail production`](../benchmarks/results/2026-07-29-gfx1151-laguna-selected-tile8-parallel-tail-production.json).
20. Fuse exact BF16 SiLU into the selected tile8 parallel-tail body.
    **Complete and promoted:** gate and up still independently round-trip
    through BF16 before the identical `gate * sigmoid(gate) * up`
    expression, so the expert intermediate is byte-identical. The candidate
    removes the temporary 20,480-byte gate/up plane write plus reread and one
    launch per shared layer. The actual layer-1 gate/up+SiLU leaf improves
    **0.131058 -> 0.129529 ms (-1.167%)**, with all 21 pairs positive and zero
    BF16 mismatches. All seven exact resident pairs improve
    **20.008491 -> 20.063975 tok/s (+0.2773%, -0.1382 ms/token)**. Clean
    selector-unset production is
    **20.053892/20.056756/20.064872 tok/s**, median **20.056756**:
    **+0.2442% / -0.1218 ms/token** over the prior packet and **+74.913%**
    over sprint start. Cached tracing confirms **721** compute dispatches per
    token versus **768**, all **5,969** selected calls in
    `<unsigned short,true,true>`, and unchanged
    grid16384x10/local128/VGPR96/SGPR128/LDS512/scratch0. Evidence:
    [`fused-SiLU leaf`](../benchmarks/results/2026-07-29-gfx1151-laguna-selected-tile8-parallel-silu-leaf.json),
    [`fused-SiLU retained`](../benchmarks/results/2026-07-29-gfx1151-laguna-selected-tile8-parallel-silu-retained.json),
    [`fused-SiLU production`](../benchmarks/results/2026-07-29-gfx1151-laguna-selected-tile8-parallel-silu-production.json).
21. Remove the dead F32 attention-context store. **SWA primitive retained;
    global sibling rejected:** llama.cpp's fused-attention graph writes only
    the output consumed downstream, while Laguna's gated decode runner also
    wrote an F32 context scratch that it never reads. A separately registered
    saturated-SWA specialization keeps a `-123.5` F32 sentinel untouched and
    emits byte-identical gated BF16 output. All nine leaf pairs improve
    **0.058948 -> 0.058681 ms (-0.453%)** at unchanged
    grid32/local384/VGPR104/SGPR128/LDS25,088/scratch0. Do not generalize the
    result: the otherwise identical global specialization regresses all
    live513/576/639 medians by **0.043-0.068%** and wins only 3/2/1 of nine
    pairs, so it is removed. The SWA resident gate is also complete and
    rejected: median decode changes
    **20.060575 -> 20.063738 tok/s (+0.0158%)**, but only 6/7 pairs improve
    and one loses **0.0713%**, larger than the projected saving. Remove the
    capability, cache field, session setter, and profile comparison switch;
    keep only the registered exact primitive for diagnostics. Production
    remains **20.056756 tok/s**. Evidence:
    [`SWA gated-only leaf`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-gated-only-leaf.json),
    [`SWA runtime rejection`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-gated-only-runtime-rejected.json),
    [`global rejection`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-gated-only-rejected.json).
22. Test intrinsically higher-precision PV before another tensorized repair.
    **Complete and rejected:** the candidate keeps the exact retained QK,
    producer maximum, compiler `expf`, ordered F32 denominator, ownership,
    divide, gate, and stores, but accumulates each dimension's 512 PV terms
    with one FP64 FMA chain and rounds once at the existing context boundary.
    The wrapped/evicted oracle remains numerically close
    (**1.58e-8** maximum F32 context error), but still changes **5/9,216**
    gated BF16 values. More importantly, nine 50-launch samples regress
    **0.058978 -> 0.165942 ms/layer (+181.36%)**. Remove the wrapper,
    registry key, harness seam, test extension, and kernel specialization
    before trace or model-quality work. This closes serial FP64 substitution:
    mathematical accuracy alone does not reproduce Laguna's recurrent
    scalar-F32 association, and it cannot bridge to llama.cpp's cooperative
    Br16 x Bc64 PV tile. Production remains **20.056756 tok/s**. Evidence:
    [`rejected FP64 PV`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-fp64-pv-rejected.json).
23. Remove the terminal staged-V consumer barrier. **Complete and rejected:**
    the eighth 64-slot V tile has no successor that can overwrite LDS, so the
    post-consume workgroup barrier is semantically unnecessary. The
    wrapped/evicted oracle is F32/BF16 byte-exact through positions 512-519
    and explicit eviction. A directional 9x50 screen initially improves
    **0.058923 -> 0.058824 ms (-0.168%)** with 8/9 paired wins, but the
    decisive 21x100 gate reverses to
    **0.058735 -> 0.058774 ms (+0.066%)**, with only 11/21 pairs positive.
    Remove the specialization, wrapper, registry key, harness seam, and test
    extension before trace or runtime work. Production remains
    **20.056756 tok/s**. This closes scalar SWA synchronization contraction;
    pivot the next decode candidate to the measured selected-Q4 gate/up and
    down excess. Evidence:
    [`rejected terminal V barrier`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-final-vbarrier-rejected.json).
24. Replay the selected-Q4 tile8 body in one physical wave. **Complete and
    rejected:** one local32 wave reconstructs the production local128 body's
    four logical wave32 K/FMA chains and reductions in their original order,
    preserving the BF16 gate/up round trips, SiLU, and output boundary while
    removing 512 B of LDS and the block barrier. The production-shape fixture
    is byte-exact, but the actual layer-1 K3072/N1024 leaf regresses
    **0.126660 -> 0.188025 ms (+48.45%)** and wins none of nine paired
    samples. Remove the body, export, wrapper, key, harness mode, and test
    extension before trace/runtime work. Four-wave physical concurrency is
    mandatory for this tile; do not retry local32 logical-wave replay.
    Production remains **20.056756 tok/s**. Evidence:
    [`rejected wave32 replay`](../benchmarks/results/2026-07-29-gfx1151-laguna-selected-tile8-wave32-replay-rejected.json).
25. Parallelize the exact selected-down tail. **Retained and published:** Q4
    and planar-Q6 keep their local128 four-wave K/FMA
    bodies, wave trees, LDS publication, barrier, and BF16 boundary. Lanes
    0..15 now each own one independent ordered wave0..3 sum and store instead
    of thread 0 serializing all 16 columns. Actual Q4 down improves
    **0.059300 -> 0.057447 ms (-3.125%, 20/21 wins)** and planar-Q6 improves
    **0.073596 -> 0.072904 ms (-0.940%, 21/21 wins)** with zero BF16
    mismatches. Cached tracing preserves local128/LDS512/scratch0 and VGPR
    **104/80**. All seven resident p512/d128 pairs improve
    **20.052490 -> 20.075641 tok/s (+0.1155%, -0.0575 ms/token)** with exact
    generated trajectory, state, and lifecycle. Tracked-clean selector-unset
    production is **20.069608 tok/s**, **+0.0641% / -0.0319 ms/token** over
    the prior clean packet and **+75.025%** over sprint start. Cached tracing
    proves all **5,969** selected-down calls use the Q4/planar-Q6
    parallel-tail specializations and cuts the family
    **4.836836 -> 4.798765 ms/token (-0.787%)** at unchanged resources.
    Evidence:
    [`selected-down parallel-tail leaf`](../benchmarks/results/2026-07-29-gfx1151-laguna-selected-down-parallel-tail-leaf.json) ·
    [`selected-down parallel-tail retained`](../benchmarks/results/2026-07-29-gfx1151-laguna-selected-down-parallel-tail-retained.json) ·
    [`selected-down parallel-tail production`](../benchmarks/results/2026-07-29-gfx1151-laguna-selected-down-parallel-tail-production.json).
26. Transfer producer-owned softplus gates from SWA to global attention.
    **Complete and rejected:** one thread per owned query computes the same
    FP32 softplus once before the score loop and publishes it through the
    existing score barrier. The live513/576/639 oracle with an explicit
    position-200 eviction remains F32/BF16 byte-exact, but all **27/27**
    paired leaf samples regress. Medians move
    **0.073179 -> 0.075032 ms (+2.53%)**,
    **0.079882 -> 0.081864 ms (+2.48%)**, and
    **0.087466 -> 0.089742 ms (+2.60%)**. The global schedule already hides
    the repeated softplus well enough that the LDS publication/dependency
    costs more than it removes. Remove the specialization, wrapper, registry
    key, harness seam, and test extension before trace or runtime work.
    Production remains **20.069608 tok/s**. Do not transfer SWA producer
    ownership wins to global attention without a direct leaf screen.
    Evidence:
    [`rejected global producer gate`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-producer-gate-rejected.json).
27. Rematerialize global physical slots to cross the dynamic-LDS occupancy
    seam. **Complete and rejected:** the candidate removes the per-token
    `int32` physical-slot LDS plane, recomputes exact visibility once during
    softmax, and rematerializes one slot per 16-thread staged-V vector group.
    QK, maximum, exp32, denominator, PV association, divide, gate, and stores
    remain byte-exact at evicted live513/576/639. The modeled score+V footprint
    falls below three times 64 KiB, but the launch still loses all **27/27**
    leaf pairs:
    **0.073445 -> 0.075353 ms (+2.60%)**,
    **0.079795 -> 0.082079 ms (+2.86%)**, and
    **0.087454 -> 0.089966 ms (+2.87%)**. Reduced dynamic LDS does not create
    useful residency for this local256/VGPR schedule; metadata
    rematerialization is net overhead. Remove the candidate before trace or
    runtime work. Production remains **20.069608 tok/s**. Small exact global
    epilogue/metadata changes are now closed; the material attention route is
    a cooperative core with an independently valid correctness mechanism.
    Evidence:
    [`rejected global physical rematerialization`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-physical-remat-rejected.json).
28. Build the material global GQA6 x K64 cooperative core. **Primitive
    retained; production route rejected:** four QK waves and eight PV waves
    cover one `(KV head, K64 split)` local256 block. F32 query and probability
    operands use three non-overlapping BF16 terms; split maxima/denominators
    and raw numerators feed a 48-block local128 FP64 merge. The explicit
    eviction oracle passes at live513/576/639 with maximum F32 error
    **1.49e-8** and gated BF16 mismatches **0/1/2 of 6,144**. Cached 9x50
    leaves improve
    **0.073109 -> 0.041343 ms (-43.45%)**,
    **0.079822 -> 0.043986 ms (-44.89%)**, and
    **0.087357 -> 0.046227 ms (-47.08%)**. Tracing names the intended
    partial at local256/VGPR96/LDS4,608/scratch0 and merge at
    local128/VGPR24/LDS0/scratch0. The complete 18-prompt/576-step
    saturated-p512 gate is finite with exact span/reset/lifecycle state and
    **559/576 (97.05%)** top-1, but max KL is **2.623766**, or **52.48x**
    the `0.05` ceiling. The temporary global-only selector and quality harness
    are removed; production remains unchanged. This closes unqualified
    cooperative reduction association even when the isolated context error is
    about `1e-8`. The next attention core must preserve retained association
    or carry an independently valid exact-repair proof.
    Evidence:
    [`global three-term WMMA rejection`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-three-term-wmma-rejected.json).
29. Isolate cooperative QK behind the retained exact reducer. **Complete and
    rejected:** the GQA6/K64 three-term WMMA body publishes only head-major
    QK scores and physical slots into the retained split ABI; the unchanged
    fixed-shape reducer preserves token-order exponentials, denominator,
    scalar F32 PV association, gate, and BF16 output. The eviction oracle
    passes with maximum F32 context error **5.59e-9** and gated BF16
    mismatches **1/1/0** at live513/576/639, but the extra scratch traffic
    and second dispatch overwhelm the QK saving. Cached 9x50 medians regress
    **0.073180 -> 0.142756 ms (+95.08%)**,
    **0.079873 -> 0.164129 ms (+105.49%)**, and
    **0.087535 -> 0.179403 ms (+104.95%)**. The leaf stop rule removes the
    candidate before trace, recurrent quality, or runtime integration.
    Production remains **20.069608 tok/s**. Do not retry a two-dispatch
    QK-only route; a viable tensorized-QK candidate must retain exact scalar
    PV ownership inside one dispatch and eliminate global score scratch.
    Evidence:
    [`rejected WMMA QK plus exact reducer`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-wmma-qk-exact-reduce-rejected.json).
30. Keep cooperative QK and ordered scalar PV in one launch. **Primitive
    retained; production route rejected:** the 32-block mixed
    **2+2+1+1** owner replaces only scalar QK with the proven three-term BF16
    WMMA producer. Producer maxima, exp32 ordered softmax, 64-slot staged
    scalar F32 PV, gate, and stores stay inside the same workgroup and
    dispatch. This transfers llama.cpp's tile-local ownership without copying
    its F16 PV contract or paying the rejected global score/reducer boundary.
    The evicted live513/576/639 oracle has maximum F32 context error
    **5.59e-9** and gated BF16 mismatches **1/1/0**. Cached 9x50 medians
    improve **0.073108 -> 0.058893 ms (-19.44%)**,
    **0.079791 -> 0.063105 ms (-20.91%)**, and
    **0.087345 -> 0.073427 ms (-15.93%)**. A cache-only trace names
    32 local256 blocks at VGPR104/SGPR128/LDS512/scratch0. The authoritative
    18-prompt/576-step saturated-p512 gate is finite with exact final
    positions, every `KVLiveSpans` metadata plane, reset state, lifecycle, and
    allocation recovery. It passes suite/category top-1 at **564/576
    (97.92%)**, but max KL is **0.741272**, or **14.83x** the `0.05` ceiling;
    category maxima are code/general-en/general-ja/mixed
    **0.741272/0.451912/0.621148/0.582570**. The temporary selector and quality
    harness are removed. Production remains **20.069608 tok/s**.
    The result validates tile-local ownership as the right mechanical shape
    but closes three-term WMMA QK as a quality-safe production route.
    Evidence:
    [`single-launch WMMA-QK rejection`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-single-launch-wmma-qk-rejected.json).

31. Profile the retained exact global owner and test the F32 output-boundary
    repair hypothesis. **Complete; repair rejected:** the actual gfx1151
    kernel is local256 with VGPR48/SGPR128/LDS512/scratch0. Its ISA already
    emits 128-bit Q/V and 64-bit K loads. Across live513/576/639, measured
    memory-unit busy is only about **18.66-26.18%** while the kernel executes
    about **1.67-2.09M VALU** and **0.41-0.52M LDS** instructions. This is not
    a missing-vector-load or saturated-DRAM problem; exact scalar arithmetic,
    cross-lane reduction, and staged PV are the material work. The previous
    “K64 nibble reuse” item was also factually wrong because attention K is
    BF16, not Q4.

    A separate exact-attention candidate preserved the gate and Q5 O
    projection in F32 and rounded only after projection, testing whether the
    retained BF16 boundary amplified the otherwise tiny WMMA-QK context
    error. The authoritative 18-prompt/576-step gate remains finite and
    reaches **563/576 (97.74%)** top-1, but max KL is **1.162237**, or
    **23.24x** the `0.05` ceiling. The temporary helper, selector, test, and
    harness are removed. The BF16 attention-output boundary is part of the
    admitted recurrent contract, not a repair seam for approximate QK.
    Evidence:
    [`exact attention core audit`](../benchmarks/results/2026-07-29-gfx1151-laguna-exact-attention-core-audit.json).

32. Preserve the retained QK association and reduce exact cross-lane cost.
    **Primitive admitted:** the separate DPP-QK sibling keeps the scalar
    four-FMA QK body and the current **+16,+8,+4,+2,+1** F32 association,
    replacing only five `ds_bpermute` shuffles with `permlanex16` plus DPP
    moves. The explicit-eviction live513/576/639 oracle is F32/BF16
    byte-exact. Cached 9x50 leaves improve
    **0.073112 -> 0.062502 ms (-14.51%)**,
    **0.079892 -> 0.074258 ms (-7.05%)**, and
    **0.087375 -> 0.081498 ms (-6.73%)**. Cache-only tracing names the
    intended grid8192/local256 body at
    VGPR48/SGPR128/LDS512/scratch0 with no compiler under profiling.
    The default-off resident gate then improves all seven p512/d128 pairs with
    complete sample separation and exact trajectories/state. Median decode is
    **20.088665 -> 20.114355 tok/s (+0.128%, -0.064 ms/token)**.
    gfx1151 now defaults DPP transport on the qualified producer-max route;
    the registered shuffle sibling remains exact rollback and peer backends
    are unchanged. Tracked-clean selector-unset production measures
    **20.088017/20.105078/20.116745 tok/s**, median **20.105078**:
    **+0.1767% / -0.0879 ms/token** over the preceding production packet.
    Exact repeat state/lifecycle passes, and the comparison-only selector is
    removed. Evidence:
    [`global DPP-QK primitive`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-dpp-qk-primitive.json) ·
    [`global DPP-QK retained`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-dpp-qk-retained.json) ·
    [`global DPP-QK production`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-dpp-qk-production.json).

33. Apply exact DPP QK transport to the higher-volume saturated SWA owner.
    **Primitive retained; runtime rejected:** the separate local384 sibling
    preserves the 128-term product/addition tree exactly and replaces only
    its **+16,+8,+4,+2,+1** lane transport. The wrapped, explicitly evicted
    CPU/rollback oracle is F32/BF16 byte-exact. Cached 9x50 leaves improve
    **0.058897 -> 0.055084 ms (-6.47%)**, and tracing confirms the expected
    grid12288/local384 specialization at unchanged
    VGPR104/SGPR128/LDS25088/scratch0.

    The complete resident result reverses the leaf: every one of seven
    p512/d128 candidate pairs is slower, with median decode
    **20.103985 -> 20.093891 tok/s (-0.0502%, +0.0250 ms/token)**. Remove the
    comparison-only runtime/session/profile route, retain only the registered
    diagnostic primitive, and leave production at **20.105078 tok/s**. Do
    not repeat a DPP-only substitution on this 384-thread body without
    changing its cooperative tile or resource profile. Evidence:
    [`SWA DPP-QK runtime rejection`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-dpp-qk-runtime-rejected.json).

34. Transfer llama.cpp's vector-valued output ownership without changing
    Laguna's exact scalar association. **Complete and rejected:** the Vulkan
    shader at `c0bc8591e` keeps `FLOAT_TYPEV4 Of` accumulators per thread and
    couples them to blockwise online softmax and K/V reuse. The screened exact
    analogue gives each wave32 lane two adjacent value dimensions, reduces
    the saturated mixed owner from local384/twelve waves to local192/six
    waves, and preserves each dimension's complete slot-order FMA, divide,
    gate, and store sequence. The wrap/eviction oracle is F32/BF16 byte-exact.

    Cached 9x50 leaves nevertheless regress
    **0.058696 -> 0.090448 ms (+54.10%)**, with every candidate sample slower.
    Cache-only tracing shows that local size and grid halve as intended, but
    each surviving wave serializes twice the QK/PV work and VGPR allocation
    rises **104 -> 120** at unchanged SGPR128/LDS25,088/scratch0. This is not
    a launch-bound-only miss: Vulkan's vector output ownership pays because
    it is coupled to its cooperative online tile, not as an isolated scalar
    remap. Skip the resident gate, remove all candidate code, and keep
    production at **20.105078 tok/s**. Do not retry dimension packing without
    also changing probability/KV ownership. Evidence:
    [`rejected exact dim2 output owner`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-dim2-output-rejected.json).

35. Remove mixed32's idle waves without adding a second launch. **Complete and
    rejected:** one local256 dispatch pairs all 72 query heads into 36 fully
    active owners. Thirty-two blocks reuse one KV head; the four pairs that
    cross a nine-query GQA boundary process their two KV heads sequentially.
    This launches exactly 288 useful output waves instead of mixed32's 384
    total/288 useful waves. Each query retains its exact QK, maximum, exp32,
    denominator, PV, divide, gate, and store order. The wrap/eviction oracle
    is F32/BF16 byte-exact.

    Cached 9x50 leaves regress **0.058748 -> 0.103936 ms (+76.92%)**.
    A cache-only trace confirms grid9216/local256 and unchanged
    VGPR104/SGPR128/LDS25,088/scratch0, so this is not a resource-allocation
    accident. Suppressing explicit two-phase loop unrolling is immaterial:
    **0.058756 -> 0.103731 ms (+76.55%)**. The four dual-KV blocks and
    **25%** extra K/V ownership outweigh idle-wave removal. Skip resident
    decode, remove all candidate code, and leave production at
    **20.105078 tok/s**. Together with the 40-block same-KV GQA2 and
    two-launch pair/triple failures, this closes scalar pair ownership around
    the retained mixed32 point. Evidence:
    [`rejected exact pair36 owner`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-pair36-rejected.json).

36. Test the proposed cooperative-result consensus guard before reconstructing
    the removed SWA WMMA body. **Complete and rejected:** the retained global
    GQA6/K64 three-term primitive supplies the same component and split-merge
    arithmetic plus known exact-output errors. Right-associating both
    three-term sums changes **3,152/3,246/3,285** F32 contexts at
    live513/576/639 but zero gated BF16 bins. Reversing the FP64 K64 merge
    changes neither F32 nor BF16 output. Both classifiers therefore recall
    **0/3** known exact BF16 errors; one missed error has bit-identical F32
    component-association outputs. Remove the source-local dump/association
    edits, skip SWA reconstruction and resident quality, and keep production
    at **20.105078 tok/s**. Consensus is not an independently valid error
    bound. Evidence:
    [`rejected consensus association`](../benchmarks/results/2026-07-29-gfx1151-laguna-attention-consensus-association-rejected.json).

37. Reuse exact probabilities through the V-stage publication barrier.
    **Retained and promoted:** one output wave per active query computes the
    identical wave32 `expf` weights and exact slot-order denominator while the
    other waves stage the same K64 V tile. The barrier already required to
    publish V now publishes the **3 x 64** probability tile as well, so all
    four output-dimension waves reuse each weight with no additional barrier.
    Excluding the two or three active probability-producer waves from V
    loading and compacting the same aligned copies across the remaining ten
    or nine waves is essential: the initial duplicate-work schedule regresses
    **0.058816 -> 0.059560 ms (+1.266%)**, while the compact schedule improves
    **0.058734 -> 0.055996 ms (-4.662%)** with complete nine-sample
    separation.

    The wrap/explicit-eviction oracle is F32-context and gated-BF16 byte-exact.
    Cache-only tracing confirms the intended grid32/local384 body at unchanged
    VGPR104/SGPR128/scratch0; LDS rises only **25,088 -> 25,600 bytes**.
    All seven resident p512/d128 candidate samples beat every control, moving
    median decode **20.097968 -> 20.282916 tok/s
    (+0.9202%, -0.4537 ms/token)** with an identical 128-token trajectory,
    positions, determinism, and allocation lifecycle. gfx1151 now selects the
    specialization only for gated saturated natural-shape
    72Q/8KV/D128/SWA512; the producer-max/gate owner remains the exact
    rollback and peer backends are unchanged. This is the first direct
    transfer of llama.cpp's probability-tile reuse that preserves Laguna's
    recurrent arithmetic contract. Tracked-clean selector-unset production is
    **20.260703/20.278430/20.270314 tok/s**, median **20.270314**:
    **+0.8219% / -0.4055 ms/token** over the preceding clean packet and
    **+76.776%** over sprint start. The capability is active without a
    comparison selector and all repeated state/lifecycle checks pass.
    Evidence:
    [`retained stage probability cache`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-stage-pcache-retained.json) ·
    [`clean production`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-stage-pcache-production.json).

38. Re-profile the clean attention wall and pin the comparator mechanism.
    **Complete:** the sorted two-queue trace contains the expected warmup128,
    prefill512, and **127** one-token decode segments. Median decode kernel sum
    is **47.554 ms/token** and span is **49.825 ms/token** at **721** compute
    dispatches. Attention is now **3.354 ms/token = 2.238 SWA + 1.107
    global**, down **0.304 ms / 8.31%** from the post-producer-gate census.
    The interval also contains the retained selected-projection and global-DPP
    changes, so the seven-pair resident artifact remains the causal stage-cache
    wall gate.

    The shader/source audit explains the remaining comparator gap. llama.cpp
    collapses grouped queries into one cooperative tile, keeps online
    maximum/denominator/output state tile-local, and publishes one probability
    tile for reused PV. It also uses F16 K/V and lower-precision cooperative
    QK/PV arithmetic, which is not hipEngine's exact BF16 recurrent contract.
    Probability reuse is therefore the exact transferable mechanism; local
    size, wave64, or cooperative matrix instructions alone are not. Same-GGUF
    Vulkan remains **0.909 ms/token** for attention, leaving **2.444
    ms/token**, or **37.58%** of the clean total wall gap, attributable to
    attention. Transfer the exact K64 probability/V publication schedule to
    the **12 global layers** next. Evidence:
    [`post-stage-cache census`](../benchmarks/results/2026-07-29-gfx1151-laguna-post-stage-pcache-wall-reprofile.json).

39. Audit the proposed global probability-cache transfer and combine the two
    positive exact SWA mechanisms. **Global transfer closed by source audit;
    combined SWA primitive retained:** the global mixed32 kernel already
    computes each `expf` probability once, stores the complete probability
    plane in LDS, normalizes it before PV, and reuses it across all four output
    waves. Replacing that with SWA's staged unnormalized numerator followed by
    divide changes the FP32 association; a literal cache port is redundant or
    non-exact.

    The non-redundant exact candidate instead combines the retained stage
    cache with the previously positive DPP QK transport. It preserves every
    QK product and the **+16,+8,+4,+2,+1** tree while changing only lane
    transport. The wrapped/explicit-eviction oracle is F32-context and gated
    BF16 byte-exact. The cached 9x50 leaf improves
    **0.056299 -> 0.052299 ms (-7.105%)**, with complete sample separation.
    Cache-only tracing keeps grid32/local384, VGPR104, SGPR128, LDS25,600, and
    scratch0. The seven-pair resident p512/d128 gate then rejects the route:
    every paired candidate loses and median decode moves
    **20.276057 -> 20.260314 tok/s
    (-0.0776%, +0.0383 ms/token)**, with exact trajectory, positions,
    determinism, and allocation lifecycle. Remove the comparison capability,
    cache route, profile CLI, and routing-test seam; retain only the registered
    diagnostic primitive, leaf choice, and oracle coverage. Production remains
    **20.270314 tok/s** on stage cache plus shuffle transport. Evidence:
    [`combined stage-cache/DPP primitive`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-stage-pcache-dpp-qk-primitive.json) ·
    [`runtime rejection`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-stage-pcache-dpp-qk-runtime-rejected.json).
40. Recombine exact whole-GQA ownership with the retained probability/V-stage
    overlap. **Complete and rejected:** the prior two-owner-per-KV
    GQA4+5/local512 screen predated stage probability reuse, so the new
    candidate halves staged-V duplication, assigns one exact probability and
    denominator producer per query, and publishes through the existing K64
    V-stage barrier. It preserves producer maxima/gates, every QK product and
    reduction, slot-order denominator/PV chains, divide, and BF16 boundary.
    The wrap/explicit-eviction oracle is F32-context and gated-BF16
    byte-exact, but all nine cache-hot candidate samples lose:
    **0.056133 -> 0.061001 ms (+8.673%)**. Probability reuse narrows the old
    local512 miss but cannot repay cutting ordinary-grid workgroups
    **32 -> 16** on the 40-CU target. Remove the kernel, wrapper, registry,
    oracle call, and leaf choice before trace or runtime integration.
    Production remains **20.270314 tok/s**. Scalar ownership recombinations
    below 32 blocks are now closed even with the retained stage-cache
    schedule. Evidence:
    [`rejected GQA4+5 stage cache`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-gqa45-stage-pcache-rejected.json).
41. Test the intermediate all-active GQA3 stage-cache owner. **Complete and
    rejected:** three local384 owners per KV head keep all 12 waves active,
    reduce V ownership **4 -> 3**, and reuse one exact probability sequence
    per query through the retained K64 publication barrier. The
    wrap/explicit-eviction oracle remains F32/BF16 byte-exact, but every
    cache-hot candidate sample loses and median latency moves
    **0.056152 -> 0.059231 ms (+5.484%)**. The 24-block point still
    underfills the 40-CU device more than its saved V traffic repays.
    Remove the wrapper, template instantiation, registry, oracle call, and
    leaf choice before tracing/runtime work. Production remains the
    **32-block mixed owner at 20.270314 tok/s**. Together with the 16-block
    result, this closes scalar occupancy/reuse below 32 workgroups under the
    retained probability-cache schedule. Evidence:
    [`rejected GQA3 stage cache`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-gqa3-stage-pcache-rejected.json).
42. Schedule pair-owner probability work on otherwise-idle waves. **Primitive
    retained; runtime promotion rejected:** the 24 pair-owner blocks in mixed32 have
    eight active output waves and four idle waves. Production assigns exact
    probability/denominator work to active waves 0 and 4, preventing them from
    helping stage V. The candidate moves those two producers to idle waves 8
    and 9; all eight output waves plus the other idle waves can then compact
    the same aligned K64 V loads. Triple-owner blocks have no idle waves and
    retain the production schedule.

    The wrapped/explicit-eviction oracle is F32-context and gated-BF16
    byte-exact. A 9x50 screen improves **0.056018 -> 0.055849 ms (-0.303%)**;
    the stronger 21x100 confirmation improves
    **0.056164 -> 0.055990 ms (-0.309%)**, with **20/21** paired wins.
    Cache-only tracing keeps grid32/local384, VGPR104, SGPR128, LDS25,600, and
    scratch0 for both control and candidate. Seven counterbalanced resident
    p512/d128 pairs move median decode only
    **20.279694 -> 20.283354 tok/s (+0.0180%, -0.0089 ms/token)**. Six pairs
    win, but the lone **-0.0207% / +0.0102 ms/token** loss is larger than the
    median modeled saving. Reject promotion, remove the comparison cache field,
    session setter, profile CLI, and routing-test seam, and retain only the
    registered diagnostic primitive. Current production remains
    **20.270314 tok/s**. Evidence:
    [`idle-wave producer primitive`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-stage-pcache-idle-producer-primitive.json) ·
    [`runtime rejection`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-stage-pcache-idle-producer-runtime-rejected.json).
43. Replay the exact denominator from the already-published probability tile.
    **Complete and rejected:** production pays 64 wave-shuffle transports per
    query/K64 stage to preserve the exact slot-order denominator chain. The
    candidate removes those shuffles and, after the existing probability/V
    publication barrier, has producer lane zero consume the same 64 visible
    probabilities from LDS in identical order while other waves perform PV.
    The existing end-of-stage barrier preserves the dependency, so there is
    no new barrier or arithmetic reassociation. The wrapped/evicted oracle is
    F32-context and gated-BF16 byte-exact, but every cache-hot sample loses:
    **0.056170 -> 0.060017 ms (+6.849%)**. Serial LDS reads delay the complete
    producer wave more than the removed shuffle transport saves. Remove the
    kernel specialization, wrapper, registry, oracle call, and leaf choice;
    production remains **20.270314 tok/s**. Do not retry scalar LDS
    denominator replay without a parallel exact prefix or materially
    different producer ownership. Evidence:
    [`post-barrier denominator rejection`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-stage-pcache-postbarrier-denom-rejected.json).
44. Vectorize the published-probability denominator replay. **Retained and
    promoted:** the scalar LDS screen established the
    right schedule with the wrong load form. The successor reads the aligned
    K64 probability tile as sixteen `float4` vectors on producer lane zero,
    then issues the same 64 ordered denominator adds while the other waves
    perform PV. Published invisible probabilities are positive zero, so
    consuming them preserves the denominator bits; every QK, maximum, `expf`,
    PV FMA, divide, gate, and store remains unchanged.

    The wrapped/evicted oracle is F32-context and gated-BF16 byte-exact. The
    9x50 leaf improves **0.056281 -> 0.045338 ms (-19.443%)**. A 21x100
    confirmation improves **0.056116 -> 0.045204 ms (-19.445%)**, with all
    21 candidate samples faster than every control. Cache-only tracing
    confirms unchanged grid32/local384, VGPR104, SGPR128, LDS25,600, and
    scratch0. All seven resident p512/d128 candidate samples beat every
    control, moving median decode
    **20.277561 -> 20.368173 tok/s
    (+0.4469%, -0.2194 ms/token)** with identical tokens, 128-token
    trajectory, positions, repeat state, and allocation lifecycle. Promote the
    qualified gated 72Q/8KV/D128/SWA512 gfx1151 capability; the shuffle
    denominator stage cache remains registered exact rollback and peer
    backends are unchanged. Remove the comparison-only profile/routing seam.
    Tracked-clean selector-unset production is
    **20.351478/20.360810/20.358649 tok/s**, median **20.358649**:
    **+0.4358% / -0.2141 ms/token** over the preceding clean packet and
    **+77.546%** over sprint start. The normal route reports the promoted
    capability active without a comparison selector and preserves the exact
    repeated trajectory/state/lifecycle. Evidence:
    [`vec4 denominator primitive`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-stage-pcache-vec4-denom-primitive.json) ·
    [`resident retention`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-stage-pcache-vec4-denom-retained.json) ·
    [`clean production`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-stage-pcache-vec4-denom-production.json).

    The clean 127-transition census attributes the production win directly:
    SWA falls **2.237644 -> 2.018186 ms/token (-9.808%)**, total attention
    falls **3.353534 -> 3.135648 ms/token (-6.497%)**, kernel sum falls
    **47.554087 -> 47.296538 ms/token (-0.542%)**, and span falls
    **49.824983 -> 49.510853 ms/token (-0.630%)**. Global attention is
    effectively flat at **1.110485 ms/token**. Attention remains
    **2.226225 ms/token** above same-GGUF Vulkan, or **35.40%** of the
    remaining production wall gap. The next exact screen vectorizes reads
    from the already-published contiguous K64 probability row inside PV while
    preserving every output dimension's 64-FMA order. Evidence:
    [`post-vector-denominator census`](../benchmarks/results/2026-07-29-gfx1151-laguna-post-vec4-denom-wall-reprofile.json).
45. Vectorize the published probability reads inside PV. **Retained and
    promoted:** the output waves already consume one contiguous
    K64 probability row, but the source expresses 64 scalar LDS reads per
    output dimension. The successor loads sixteen aligned `float4` values and
    issues x/y/z/w FMAs in the identical slot 0..63 sequence. This transfers
    llama.cpp's low-overhead probability-tile reuse without changing BF16 KV,
    QK, maximum, `expf`, denominator, PV association, divide, gate, or store.

    RED fails on the absent wrapper. GREEN passes the wrapped/evicted oracle
    with byte-identical F32 context and gated BF16 output. The 9x50 leaf
    improves **0.045426 -> 0.045278 ms (-0.325%)**, and the stronger 21x100
    screen improves **0.045306 -> 0.045174 ms (-0.290%)** with **21/21**
    paired wins. Cache-only tracing keeps grid32/local384, VGPR104, SGPR128,
    LDS25,600, and scratch0 unchanged and names the distinct candidate with no
    compiler under profiling. All seven counterbalanced resident candidate
    runs beat their paired controls, moving median decode
    **20.366610 -> 20.379415 tok/s
    (+0.06287%, -0.03085 ms/token)**. Every row preserves tokens, the exact
    128-token trajectory, positions, repeat state, and allocation lifecycle.
    Promote the qualified gfx1151 capability, retain scalar PV probability
    reads as exact rollback, and remove the comparison-only CLI/session seam.
    Tracked-clean selector-unset production is
    **20.335685/20.349871/20.352342 tok/s**, median **20.349871**. This
    absolute checkpoint is **0.0431%** below the preceding clean packet,
    inside shared-APU variance; retention rests on the stronger same-process
    gate in which all **7/7** candidate runs beat their paired controls.
    The normal route reports the capability active without a comparison
    selector and preserves exact repeated trajectory/state/lifecycle.
    Evidence:
    [`vectorized probability primitive`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-stage-pcache-vec4-probability-primitive.json) ·
    [`resident retention`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-stage-pcache-vec4-probability-retained.json) ·
    [`clean production`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-stage-pcache-vec4-probability-production.json).
46. Transfer vector probability replay to global attention. **Retained and
    promoted:** unlike SWA, the retained global kernel
    already stores its complete normalized probability plane in LDS, but PV
    still consumes one scalar probability per token. The exact successor pads
    each score/probability plane to a four-float stride, preserving alignment
    at live513/live639, and consumes one `float4` per four original PV
    iterations. Normalization, positivity test, BF16 V conversion, and
    accumulation order remain unchanged.

    RED fails on the absent wrapper. GREEN passes the explicit position-200
    eviction oracle at live513/576/639 with byte-identical F32 context and
    gated BF16 output. The stronger 21x100 leaf moves
    **0.063256 -> 0.055029 ms (-13.006%)**,
    **0.074390 -> 0.061559 ms (-17.248%)**, and
    **0.081525 -> 0.073051 ms (-10.395%)**, with **21/21** paired wins and
    complete sample separation at every shape. Cache-only tracing names the
    distinct `<...,true>` specialization at grid8192/local256, VGPR48,
    SGPR128, static-LDS512, scratch0; no compiler runs under profiling.
    The 12-global-layer resident gate passes with complete separation: all
    seven candidates beat every control, moving median decode
    **20.373406 -> 20.409544 tok/s
    (+0.17738%, -0.08691 ms/token)**. Every run preserves tokens, the exact
    128-token trajectory, positions, repeat state, and allocation lifecycle.
    Promote the qualified gfx1151 capability, retain scalar probability
    replay as exact rollback, and remove the comparison-only CLI/session
    seam. Tracked-clean selector-unset production is
    **20.403940/20.414792/20.418871 tok/s**, median **20.414792**:
    **+0.3190% / -0.1563 ms/token** over the preceding clean packet and
    **+78.036%** over sprint start. The normal route reports the capability
    active without a comparison selector and preserves exact repeated
    trajectory/state/lifecycle.
    This is the directly transferable part of llama.cpp Vulkan's tile-local
    probability reuse; its cooperative lower-precision QK/PV arithmetic
    remains outside the exact BF16 recurrent contract. Evidence:
    [`global vector probability primitive`](../benchmarks/results/2026-07-30-gfx1151-laguna-global-probability-vec4-primitive.json) ·
    [`resident retention`](../benchmarks/results/2026-07-30-gfx1151-laguna-global-probability-vec4-retained.json) ·
    [`clean production`](../benchmarks/results/2026-07-30-gfx1151-laguna-global-probability-vec4-production.json).
47. Re-profile the post-transfer decode wall. **Complete:** one tracked-clean,
    cache-only 127-transition trace keeps **721 compute + 5 runtime-copy
    dispatches/token** and measures **47.174209 ms/token** kernel sum.
    Global attention falls **1.110485 -> 1.005649 ms/token (-9.441%)** while
    SWA is flat at **2.017783 ms/token**; total attention falls
    **3.135648 -> 3.023432 ms/token (-3.579%)**. Source-F16 remains flat at
    **24.027570 ms/token**.

    Same-GGUF Vulkan spends **0.909423 ms/token** in attention. The remaining
    exact-attention gap is therefore **2.114009 ms/token**, **34.35%** of the
    current **6.155-ms/token** production wall gap, and hipEngine attention
    is still **3.32x** the comparator. Continue with the 36-layer saturated
    SWA body, which owns **2.017783 ms/token**, using a single-launch exact
    data-movement or scheduling change. Do not retry lower-precision
    cooperative QK/PV arithmetic already rejected by the recurrent-state
    gate. Evidence:
    [`post-global-probability census`](../benchmarks/results/2026-07-30-gfx1151-laguna-post-global-probability-vec4-wall-reprofile.json).
48. Overlap the exact SWA denominator replay with PV on pair-owner blocks.
    **Retained and promoted on gfx1151:** the
    existing mixed32 launch has 24 pair-owner blocks whose waves 8/9 are idle
    during PV. Those waves now perform the unchanged vectorized 64-term
    denominator replay while all eight active output waves execute their
    unchanged PV chains. The eight triple-owner blocks retain the production
    schedule. Probability generation, compact V loading, barriers, denominator
    add order, and every output FMA remain unchanged.

    RED fails on the absent wrapper. GREEN passes the existing wrap and
    explicit-eviction oracle with byte-identical F32 context and gated BF16
    output. The 9x50 leaf moves **0.045257 -> 0.045117 ms (-0.310%)**. The
    stronger 21x100 screen moves **0.045329 -> 0.045182 ms (-0.324%)** with
    **21/21** paired wins, although the leaf distributions do not completely
    separate. Cache-only tracing names the distinct specialization at
    grid12288/local384, VGPR104, SGPR128, LDS25,600, and scratch0; the
    compiler does not run under profiling.

    All seven counterbalanced p512/d128 resident candidates beat every
    control with complete separation, moving median decode
    **20.411948 -> 20.430138 tok/s
    (+0.08912%, -0.04362 ms/token)**. Every row preserves tokens, the exact
    trajectory, final position, repeat determinism, and allocation lifecycle.
    Promote the qualified gfx1151 capability, retain active-wave denominator
    replay as exact rollback, remove the comparison-only CLI/cache seam, and
    leave peer backends unchanged. Tracked-clean selector-unset production is
    **20.412363/20.425412/20.429048 tok/s**, median **20.425412**:
    **+0.05202% / -0.02547 ms/token** over the preceding clean packet and
    **+78.128%** over sprint start. The normal route reports the capability
    active without a comparison selector and preserves exact repeated
    trajectory/state/lifecycle.
    Evidence:
    [`primitive`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-idle-vector-denom-primitive.json) ·
    [`resident retention`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-idle-vector-denom-retained.json) ·
    [`clean production`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-idle-vector-denom-production.json).
49. Fill the device with a one-launch mixed40 SWA owner. **Retained and
    promoted on gfx1151:** partition each KV
    head's nine queries as **2+2+2+2+1** instead of **2+2+2+3**. The grid
    grows **32 -> 40 blocks**, matching gfx1151's 40 CUs and removing the
    eight triple-query critical blocks. This spends 25% more K/V-owner
    traffic but keeps one launch, every arithmetic operation, idle-wave
    denominator replay, and all output stores unchanged.

    RED fails on the absent wrapper. GREEN passes the wrap and explicit
    eviction oracle with byte-identical F32 context and gated BF16 output.
    The 9x50 leaf moves **0.045322 -> 0.037665 ms (-16.894%)**. The stronger
    21x100 screen moves **0.045322 -> 0.037599 ms (-17.039%)**, with
    **21/21** paired wins and complete sample separation. Cache-only tracing
    names grid15360/local384 at unchanged VGPR104, SGPR128, LDS25,600, and
    scratch0; no compiler runs under profiling. This is the same transferable
    scheduling principle behind llama.cpp Vulkan's K-split breadth: keep
    enough independent cooperative tiles resident to fill the machine.

    All seven counterbalanced p512/d128 resident candidates beat every
    control with complete separation, moving median decode
    **20.433014 -> 20.501083 tok/s
    (+0.33313%, -0.16249 ms/token)**. Every row preserves tokens, trajectory,
    final position, repeat determinism, and allocation lifecycle. Promote the
    qualified gfx1151 capability, retain mixed32 as exact rollback, remove the
    comparison-only CLI/cache seam, and leave peer backends unchanged.
    Tracked-clean selector-unset production is
    **20.479107/20.483884/20.498437 tok/s**, median **20.483884**:
    **+0.28627% / -0.13975 ms/token** over the preceding clean packet and
    **+78.638%** over sprint start. The normal route reports mixed40 active
    without a comparison selector and preserves exact repeated
    trajectory/state/lifecycle. Evidence:
    [`primitive`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-mixed40-primitive.json) ·
    [`resident retention`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-mixed40-retained.json) ·
    [`clean production`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-mixed40-production.json).
50. Re-profile the mixed40 wall before changing the body. **Complete:** the
    tracked-clean 127-transition trace still has **721 compute
    dispatches/token**. SWA falls **2.017783 -> 1.757218 ms/token
    (-12.913%)**, total attention falls **3.023432 -> 2.761582 ms/token
    (-8.661%)**, kernel sum falls **47.174209 -> 46.893051 ms/token
    (-0.596%)**, and kernel span falls **50.598383 -> 49.116885 ms/token
    (-2.928%)**. Global attention and all major projection families remain
    flat.

    Same-GGUF Vulkan remains at **0.909423 ms/token** attention. The exact
    attention gap is now **1.852159 ms/token**, **30.92%** of the current
    **5.989-ms/token** production wall gap, and hipEngine attention is
    **3.04x** the comparator. Keep the 40-block geometry and next move
    probability generation onto otherwise-idle tail waves while idle waves
    8/9 retain denominator replay; this preserves the complete arithmetic
    sequence while removing producer work from the PV-owner waves. Evidence:
    [`post-mixed40 census`](../benchmarks/results/2026-07-30-gfx1151-laguna-post-mixed40-wall-reprofile.json).
51. Separate mixed40 exponent, denominator, and PV roles. **Retained and
    promoted on gfx1151:** pair-owner waves 10/11 now
    generate the two probability rows while waves 8/9 replay their ordered
    denominators and waves 0-7 retain ordered PV; singleton blocks use wave
    11, wave 8, and waves 0-3 respectively. QK, exponent inputs, denominator
    order, every PV FMA, gate, stores, ownership, resident bytes, and launch
    count are unchanged.

    RED fails importing the absent wrapper. GREEN passes the wrapped and
    explicitly evicted oracle with byte-identical F32 context and gated BF16.
    The 9x50 leaf moves **0.036995 -> 0.036961 ms (-0.091%)** with **6/9**
    paired wins. The stronger 21x100 screen moves
    **0.037001 -> 0.036896 ms (-0.285%)** with **20/21** paired wins.
    Cache-only tracing records unchanged grid15,360/local384, VGPR104,
    SGPR128, LDS25,600, and scratch0; no compiler runs under profiling.
    The seven exact counterbalanced p512/d128 resident pairs move median
    decode **20.502555 -> 20.508345 tok/s
    (+0.02824%, -0.01377 ms/token)**. Six pairs improve; the median paired
    gain is **+0.01542%**, and the sole **-0.00998%** loss is smaller. Every
    row preserves tokens, trajectory, positions, repeat determinism, and
    allocation lifecycle. Promote the qualified gfx1151 capability, retain
    the preceding mixed40 schedule as exact rollback, remove comparison-only
    plumbing, and leave peer backends unchanged. Tracked-clean selector-unset
    production is **20.489321/20.503390/20.494732 tok/s**, median
    **20.494732 tok/s**: **+0.05296% / -0.02584 ms/token** over the preceding
    clean packet and **+78.733%** over sprint start. The normal route reports
    tail producers active without a comparison selector and preserves exact
    repeated trajectory/state/lifecycle. Evidence:
    [`primitive`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-mixed40-tail-producer-primitive.json) ·
    [`resident retention`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-mixed40-tail-producer-retained.json) ·
    [`clean production`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-mixed40-tail-producer-production.json).
52. Pipeline K64 V/probability staging across ordered PV. **Rejected and
    removed at the leaf stop:** the exact double-buffer candidate lets idle
    waves prepare stage *k+1* while active waves execute stage *k*, reducing
    the staged-loop barriers **16 -> 9** without changing QK, exponent,
    denominator, or PV order. It adds **17,152 bytes** of dynamic LDS for the
    second probability/V stage and narrows next-stage V loading to the four
    non-output waves on pair-owner blocks.

    The wrapped/evicted oracle is F32/BF16 byte-exact, but the 9x50 leaf
    regresses **0.037106 -> 0.040897 ms (+10.219%)**, losing all nine pairs.
    Skip tracing and resident integration; remove the wrapper, template
    branch, leaf selector, and test call. The retained kernel is not
    barrier-bound enough for a second full LDS stage. Do not retry without a
    materially different loader/overlap mechanism. Production remains
    **20.494732 tok/s**. Evidence:
    [`double-buffer rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-mixed40-double-buffer-rejected.json).
53. Compact mixed40's unused third-query storage. **Rejected and removed at
    the leaf stop:** mixed40 owns at most two queries, so the diagnostic
    shrinks its score/max/gate/probability/denominator and register arrays from
    three query rows to two while preserving the conceptual third output-wave
    slot. Tail exponent waves 10/11, idle denominator waves 8/9, active PV
    waves 0-7, every arithmetic operation, and the 40-block grid remain
    unchanged. This removes **2,360 logical static bytes** before compiler
    alignment.

    The wrapped/evicted oracle is F32/BF16 byte-exact, but the 9x50 leaf moves
    **0.037741 -> 0.037747 ms (+0.017%, 4/9 wins)** and the stronger 21x100
    screen moves **0.037749 -> 0.037764 ms (+0.041%, 9/21 wins)**. Skip
    tracing and resident integration; remove the wrapper, template branch,
    leaf selector, and test call. Storage-only compaction does not address the
    retained 104-VGPR schedule or buy useful occupancy. Production remains
    **20.494732 tok/s**. Evidence:
    [`compact2 rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-mixed40-compact2-storage-rejected.json).
54. Overlap the exact idle-wave denominator's LDS reads. **Primitive retained,
    runtime rejected and removed:** a fresh counter census of the current
    mixed40 body measures median **51.264% memory-unit busy**, **49.742% L2
    hit**, **2,091.125 KiB fetched**, **1.764M VALU**, **0.482M LDS**, and
    **0.641M SALU** instructions across exactly **480 waves**. Together with
    the failed double-buffer and compact2 screens, this identifies a mixed
    instruction/latency body rather than a DRAM or LDS-capacity roof.

    ISA inspection finds the idle denominator critical path issuing sixteen
    `ds_load_b128` operations as sixteen separate load/`lgkmcnt(0)`/four-add
    chains. The exact successor issues four adjacent loads before consuming
    their 16 components in the original order. Code size and static
    instruction count remain **5,740 bytes / 1,070 instructions**, but each
    quartet now uses `lgkmcnt(3/2/1/0)` and overlaps LDS latency. All **64**
    ordered adds, wave roles, barriers, grid, and arithmetic remain unchanged.

    The wrapped/evicted oracle is F32/BF16 byte-exact. The 9x50 leaf improves
    **0.037188 -> 0.037036 ms (-0.410%, 9/9 wins)**; the stronger 21x100
    screen improves **0.037597 -> 0.037559 ms (-0.101%)** with **19 wins,
    one tie, and one loss**. Cache-only tracing confirms unchanged
    grid15,360/local384, VGPR104, SGPR128, LDS25,600, and scratch0.

    The seven exact counterbalanced p512/d128 resident pairs do not retain the
    leaf result: median decode moves **20.497384 -> 20.497114 tok/s
    (-0.00132%, +0.00064 ms/token)**, the median paired change is
    **-0.00919%**, and only **3/7** pairs improve. Tokens, complete generated
    trajectory, positions, repeat determinism, and allocation lifecycle all
    remain exact. Remove the comparison/runtime route and keep only the
    registered diagnostic primitive. Production remains **20.494732 tok/s**.
    Evidence:
    [`primitive`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-mixed40-denom-prefetch4-primitive.json) ·
    [`runtime rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-mixed40-denom-prefetch4-runtime-rejected.json).
55. Pack staged BF16 V reads without reducing output-wave breadth.
    **Rejected and removed at the leaf stop:** all eight active PV waves and
    one accumulator per lane remain unchanged. Each even lane instead issues
    one aligned 32-bit LDS load for a BF16 value pair and delivers the packed
    word to its odd neighbor with row-shift DPP. Every probability, BF16
    conversion, FMA, denominator, divide, gate, and store is unchanged.

    The wrapped/evicted oracle is F32/BF16 byte-exact, but the 9x50 leaf
    regresses **0.036950 -> 0.045846 ms (+24.075%)**. Cross-lane delivery
    costs much more than the avoided per-lane 16-bit LDS read. Remove the
    helper, template branch, export, wrapper, registry key, leaf selector, and
    test call; skip tracing and resident integration. Production remains
    **20.494732 tok/s**. Evidence:
    [`packed-V-DPP rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-mixed40-packed-v-dpp-rejected.json).
56. Remove DPP and rely on same-address packed LDS broadcast. **Rejected and
    removed at the leaf stop:** both lanes in each pair read the same aligned
    32-bit word and select their own BF16 half. This retains all eight output
    waves, one accumulator per lane, and every arithmetic operation without a
    cross-lane instruction.

    The wrapped/evicted oracle remains byte-exact, but the 9x50 leaf regresses
    **0.037053 -> 0.037558 ms (+1.363%)**. The compiler's existing 16-bit LDS
    value replay is faster than same-address dword loads plus half selection.
    Remove the candidate before tracing or resident integration and close
    staged-V word packing. Production remains **20.494732 tok/s**. Evidence:
    [`packed-V-broadcast rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-mixed40-packed-v-broadcast-rejected.json).
57. Make the resident BF16 key layout lane-major for one packed load.
    **Rejected and removed at the leaf stop:** this byte-neutral diagnostic
    transposes each D128 key from `[part4][lane32]` to `[lane32][part4]`.
    Each lane replaces four 16-bit cache loads 32 elements apart with one
    aligned 64-bit load and extracts the same four BF16 values in the original
    QK order. This isolates one concrete llama.cpp Vulkan advantage—vectorized
    cooperative K loading—without changing QK, softmax, PV, ownership, or
    resident bytes.

    The wrapped/evicted oracle is F32/BF16 byte-exact. The 9x50 screen improves
    **0.037159 -> 0.037098 ms (-0.164%, 8/9 wins)**, but the stronger 21x100
    screen collapses to **0.037045 -> 0.037019 ms (-0.069%, 17/21 wins)**.
    That is noise-scale and cannot justify migrating every relevant KV writer
    and reader. Remove the wrapper, layout branch, registry/export, test
    side-buffer, and leaf selector before tracing or resident integration.
    Production remains **20.494732 tok/s**. The missing Vulkan multiplier is
    cooperative K/V tile reuse plus tensorized QK/PV, not a key-layout-only
    vector load. Evidence:
    [`lane-major-key rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-mixed40-lane-major-key-rejected.json).
58. Mark the retained one-phase attention buffers non-aliasing.
    **Rejected and removed at the leaf stop:** apply `__restrict__` to the
    independent query, K, V, context, gate, gated-output, and `KVLiveSpans`
    pointers, matching the contract already used by the adjacent fused KV
    writers. This changes no arithmetic, ABI, bytes, ownership, or dispatch.
    RED is not applicable to a compiler-only pointer contract; the focused
    wrapped/explicitly-evicted oracle remains byte-exact.

    Three independent 21x100 processes before and after the annotation move
    the production-tail median-of-process-medians
    **0.037002 -> 0.037097 ms (+0.259%)**. The compiler already schedules the
    retained body effectively under its existing argument contract. Remove
    the qualifiers and skip resident integration. Production remains
    **20.494732 tok/s**. Evidence:
    [`restrict/noalias rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-mixed40-restrict-noalias-rejected.json).
59. Replace staged-V's global-load/wait/LDS-store chain with a direct
    global-to-LDS load.
    **Unsupported on gfx1151 and removed before benchmarking:** the retained
    mixed40 ISA stages each 16-byte value vector as `global_load_b128`,
    `s_waitcnt vmcnt(0)`, then `ds_store_b128`. LLVM exposes
    `__builtin_amdgcn_global_load_lds`, which would remove the register and
    explicit LDS-store leg without changing bytes, layout, barriers,
    ownership, or arithmetic. The gfx1151 compile rejects it, however:
    **`needs target feature vmem-to-lds-load-insts`**. Do not force an
    unsupported target feature or publish a result from a different target.

    Remove the candidate completely; the production HIP source is byte-clean
    and remains **20.494732 tok/s**. The supported successor is an exact
    source-prefetch screen that keeps ordinary global and LDS instructions but
    issues multiple `global_load_b128` operations before the corresponding
    stores, allowing gfx1151 to overlap memory latency. Evidence:
    [`unsupported global-to-LDS load`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-global-load-lds-unsupported.json).
60. Overlap the supported staged-V source loads in registers.
    **Rejected and removed at the leaf stop:** issue two ordinary 16-byte
    V-cache loads before either corresponding LDS write, preserving every
    staged byte, LDS slot, barrier, owner, arithmetic operation, and dispatch.
    RED fails on the absent wrapper; GREEN is F32/BF16 byte-exact under wrap
    and explicit eviction.

    The paired 9x50 leaf regresses **0.037081 -> 0.045278 ms (+22.106%)**.
    The extra live 128-bit value state is much more expensive than the
    load-to-store dependency it attempts to hide. Remove the kernel branch,
    export, wrapper, registry key, test call, and leaf selector completely;
    skip tracing and resident integration. Production remains
    **20.494732 tok/s**. Do not try a deeper staged-V source prefetch—the
    bounded depth-two screen already fails decisively. Evidence:
    [`value-prefetch2 rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-value-prefetch2-rejected.json).
61. Let the finished probability-producer waves copy the staged-V tail.
    **Retained as an exact diagnostic primitive; rejected for production:**
    on pair owners, the two tail producer waves copy vectors 960..1023 while
    every other loader performs exactly three copies instead of 64 lanes
    performing four. The singleton producer wave similarly copies
    vectors 992..1023. This changes only copy ownership after probability
    publication; staged bytes, barriers, QK, softmax, denominator, PV, and
    stores are unchanged. The wrapped/evicted oracle is F32/BF16 byte-exact.

    The 9x50 and 21x100 leaves improve
    **0.036699 -> 0.035635 ms (-2.898%)** and
    **0.037575 -> 0.036346 ms (-3.270%)**. Cache-only tracing keeps
    grid15,360/local384, VGPR104, SGPR128, LDS25,600, and scratch0. Seven
    counterbalanced actual-model p512/d128 pairs nevertheless move median
    decode **20.509962 -> 20.507264 tok/s (-0.01316%)**; median paired
    change is only **+0.00032%** and **4/7** pairs improve. Generated
    trajectories, positions, repeat determinism, and allocation lifecycle
    remain exact.

    Remove the benchmark-only registry swap and do not promote the primitive.
    Production remains **20.494732 tok/s**. Retain the separate symbol only
    for a compounded wave-scheduling experiment where its strong leaf gain
    may become compositional. Evidence:
    [`producer-value-tail runtime rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-producer-value-tail-runtime-rejected.json).
62. Screen an independently valid cooperative-PV rounding bound before
    rebuilding the removed tensor kernel.
    **Rejected at the analytical precondition:** use the saved global
    three-term GQA6/K64 outputs and their reproducible wrapped/evicted input
    fixture to evaluate a deliberately favorable bound. Assume QK and
    softmax probabilities are already exact and identical, then bound only
    the difference between the retained 512-term scalar F32 PV chain and the
    cooperative K64 partials plus eight-way merge:
    `(gamma_512 + gamma_64 + gamma_8) * sum(abs(p_i * v_i))`. Scale that
    interval through the positive softplus gate and include the final F32
    multiply-rounding term.

    Even before adding the omitted QK, exponential, normalization, and
    decomposition errors, the bound marks **2,846/6,144 (46.32%)**,
    **3,012/6,144 (49.02%)**, and **3,144/6,144 (51.17%)** components
    uncertain at live513/576/639. Median bounded gated-context error is about
    **2.35e-6**. The prior component-replay implementation is already
    **124.40% slower** than retained exact attention when replay is complete;
    this roughly half-dense best-case bound cannot make that topology
    competitive.

    Do not reconstruct the three-term WMMA plus scalar component-replay path
    around a standard gamma bound. Reopen only for a materially tighter
    certified mechanism—hardware directed bounds or a separately cheap
    error-free transform—not another output-derived midpoint or consensus
    heuristic. Production remains **20.494732 tok/s**. Evidence:
    [`PV-bound precondition rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-attention-pv-bound-screen-rejected.json).
63. Test whether F32 Q8_1 metadata repairs the selected gate/up integer-dot
    quality failure before building residual machinery.
    **Rejected and removed:** on actual layer-1 resident gate/up weights at
    `x_rows=1, routes=10, K=3072, N=1024`, the compact FP16-scale Q8_1 pack
    plus dp4a fused-SiLU consumer improves the inclusive leaf
    **0.127712 -> 0.095370 ms (-25.32%)**. Replacing its 36-byte block with
    a 40-byte block carrying F32 `d` and `s` remains fast at
    **0.129319 -> 0.100381 ms (-22.38%)**, but it does not repair accuracy.

    Both variants differ from the exact retained result in **8,451 BF16
    values** with **0.125 max absolute error**, and their complete candidate
    BF16 outputs have the same SHA-256. The selected consumer recomputes the
    quantized-byte sum and never uses stored `s`; widening metadata changes
    neither Q8 bytes nor the eventual BF16 result. Therefore this proposed
    fix cannot explain or remove the earlier full-model one-plane Q8_1
    quality failure.

    Remove the temporary F32 block, quantizer symbol, consumer symbol,
    wrappers, registry key, test imports, and both leaf selectors. The
    **speed mechanism remains open**, but only behind a design that changes
    the activation approximation itself—bounded residual repair, selective
    exact blocks, or another mixed scheme—and then passes the complete KL
    and top-1 gate. Production remains **20.494732 tok/s**. Evidence:
    [`Q8_1 F32-scale rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-selected-q8-f32-scale-rejected.json).
64. Gate the fast compact-Q8 selected gate/up path on recurrent c=1 decode,
    not its previously rejected prefill-only state.
    **Rejected and removed at d16:** add a temporary explicit/default-off
    registry-driven owner that quantizes each c=1 hidden row once and uses
    the existing selected Q4_K T16 dp4a fused-SiLU consumer in all 47 sparse
    layers. It reuses already-resident scratch and leaves prefill exact.
    Focused plan, production-shape runtime, CPU-quality, and fused-rounding
    tests pass.

    After identical exact p512 prefill, the 16-transition teacher-forced gate
    reaches mean/max KL **0.08036/0.59520** and top-1 **15/16 (93.75%)**.
    The first top-1 miss is step 4, where token **268** becomes **26** at KL
    **0.59520**—**11.90x** the maximum-KL ceiling. The single
    non-counterbalanced directional timing also moves
    **50.6409 -> 50.8302 ms/token (+0.374%)**, so the positive isolated leaf
    does not transfer to this complete owner.

    Stop before d128, tracing, or residual repair. Remove the compact
    activation registry key, plan route, session flag/setter, runtime owner,
    focused assertions, and diagnostic harness. This closes blanket compact
    Q8_1 selected gate/up for both prefill and decode; reopen only around a
    materially different activation representation with an independently
    positive inclusive owner. Production remains **20.494732 tok/s**.
    Evidence:
    [`selected Q8 decode rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-selected-q8-decode-rejected.json).
65. Remove staged V only from mixed40's singleton-query tail owners.
    **Rejected and removed at the leaf stop:** eight of the 40 SWA blocks own
    one query each, so they have no cross-query V reuse. The diagnostic
    computes the singleton's complete FP32 probability plane once, preserves
    its exact denominator and PV order, and then reads V directly, avoiding
    all eight K64 V-stage copies and sixteen staged-loop barriers in those
    blocks.

    The wrapped/evicted oracle is F32-context and gated-BF16 byte-exact. The
    first 9x50 leaf regresses
    **0.036936 -> 0.202381 ms (+447.929%)**, but that diagnostic also fully
    unrolled 128 probability vectors. A controlled-unroll rerun removes that
    code-generation confound and still regresses
    **0.036765 -> 0.122887 ms (+234.251%)**. Lanes within an output wave
    remain contiguous, so the corrected finding is not uncoalesced addresses:
    four scalar-load PV waves expose far less V-level parallelism than the
    retained twelve-wave aligned 16-byte cooperative copy.

    Remove the template branch, export, wrapper, and leaf selector; skip
    tracing and resident integration. Keep coalesced staged-V transport even
    for singleton owners. Production remains **20.494732 tok/s**. Evidence:
    [`singleton direct-PV rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-singleton-direct-pv-rejected.json).
66. Normalize each global probability once before its 128-lane PV replay.
    **Retained and promoted on gfx1151:** the retained global
    path stores each exponential plus one reciprocal, then all 128 output
    lanes repeat the same FP32 multiplication. The new registered sibling
    performs that identical multiply once per probability in LDS and adds one
    block barrier before PV. The positive test, BF16 V conversion, and every
    per-dimension PV addition retain their original order and bits.

    The live513/576/639 position-200 eviction oracle is F32-context and
    gated-BF16 byte-exact. Strong 21x100 leaves improve
    **1.640%/1.503%/0.128%**, with **21/21, 21/21, and 19/21** paired wins.
    Cache-only tracing names the distinct final-`true` specialization at
    grid8192/local256, VGPR48, SGPR128, static-LDS512, and scratch0; no
    compiler runs under profiling.

    Seven counterbalanced actual-model p512/d128 pairs move the 12-global-layer
    decode median **20.501353 -> 20.503954 tok/s (+0.01269%)**, saving
    **0.00619 ms/token** by independent medians. The candidate wins **5/7**
    pairs with a median paired saving of **0.00558 ms/token**; every pair
    preserves the same generated-state SHA, tokens, position, and lifecycle.
    gfx1151 now selects the qualified capability, other backends remain
    unchanged, and the prior exact probability replay remains the fallback.
    Tracked-clean selector-unset production is
    **20.489386/20.496816/20.498178 tok/s**, median
    **20.496816 tok/s** (**48.788 ms/token**): **+0.01017%** over the
    preceding clean packet and **+78.751%** over sprint start. All runs retain
    the exact trajectory/state/lifecycle.
    Evidence:
    [`primitive`](../benchmarks/results/2026-07-30-gfx1151-laguna-global-probability-prenorm-primitive.json),
    [`retention`](../benchmarks/results/2026-07-30-gfx1151-laguna-global-probability-prenorm-retained.json),
    [`production`](../benchmarks/results/2026-07-30-gfx1151-laguna-global-probability-prenorm-production.json).
67. Re-profile the clean post-pre-normalization wall and re-anchor the llama
    gap. **Complete:** the cached 127-transition p512/d128 trace measures
    **46.910112 ms/token** kernel sum and **49.119568 ms/token** dispatch
    span. Attention is **2.746352 ms/token**:
    **1.754009 ms** across 36 SWA calls plus **0.992343 ms** across 12 global
    calls. Against the post-mixed40 census, global falls
    **1.004364 -> 0.992343 ms/token (-1.197%)** and SWA remains flat.

    Same-GGUF llama.cpp Vulkan logs **0.909423 ms/token** attention. The
    remaining **1.836929-ms** attention deficit is **30.83%** of the complete
    **5.958544-ms/token** production wall gap. Source-F16 projection remains
    the largest absolute family at **24.037891 ms/token**, but it is already
    ahead of the comparator; the largest identified comparator-relative gap
    remains attention. Evidence:
    [`post-pre-normalization wall census`](../benchmarks/results/2026-07-30-gfx1151-laguna-post-global-prenorm-wall-reprofile.json).
68. Amortize final SWA normalization through one reciprocal per query.
    **Rejected and removed:** the existing final K64 denominator producer
    computes one reciprocal per query and publishes it before the already
    required barrier, replacing **9,216 output-lane divisions with 72
    divisions plus multiplies** without a new launch or barrier.

    The 9x50 leaf moves only
    **0.037304 -> 0.037159 ms (-0.391%)**, an estimated
    **0.00525 ms/token** across all 36 SWA layers. F32 context changes by at
    most **1.86e-9** while this fixture's gated BF16 output remains identical,
    but the gain is far below the material gate for a new numerical surface.
    Stop before tracing, resident integration, or recurrent quality testing
    and remove the diagnostic. Scalar final division is not the llama gap;
    cooperative QK/PV remains the target. Evidence:
    [`reciprocal-normalization rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-reciprocal-normalize-rejected.json).
69. Store scaled SWA scores in FP16 while retaining scalar F32 QK and PV.
    **Rejected and removed:** the diagnostic keeps the complete F32 QK
    reduction, rounds each final scaled score once into FP16, and uses that
    value for producer maximum and softmax. This halves the three-query score
    plane **6,144 -> 3,072 bytes** and removes replayed score-scale
    multiplications without changing QK accumulation or scalar PV order.

    The 9x50 leaf is flat/regressive:
    **0.037107 -> 0.037111 ms (+0.011%)**. It also changes **30** gated BF16
    values on the fixture, with F32 context max error **6.29e-8**. Stop before
    trace, resident integration, or recurrent quality testing and remove the
    diagnostic. The performance side of llama.cpp's F16 contract is not score
    storage alone; its cooperative QK/PV execution is the material mechanism.
    Evidence:
    [`FP16 scaled-score rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-fp16-scaled-scores-rejected.json).
70. Screen the untested GQA9/K128 split-softmax point using the recovered
    llama-shaped ownership. **Rejected and removed at the leaf gate:** the
    cached earlier K64 code object reveals its actual scalar topology: keep all
    nine queries resident, partition K tokens across eight waves, publish the
    score/weight tile, and flatten 9 x 128 outputs so one V load updates up to
    five query accumulators. K128 halves the merge from eight to four split
    states, but also yields only **8 KV heads x 4 splits = 32 workgroups** on
    the 40-CU gfx1151.

    Three deliberately simpler drafts establish the scheduling hazards:
    serialized staged-V, wave-per-query direct-V, and wave-per-query
    published-weight bodies regress **381.27%/322.94%/369.44%**. The recovered
    two-axis ownership is much better but still moves the 9x50 leaf
    **0.037223 -> 0.095496 ms (+156.55%)**. Its fixture error is small
    (**2.61e-8** F32, four gated BF16 values), but performance alone rejects
    it before recurrent quality or resident integration.

    Cache-only tracing proves this is not a compiler-spill accident: the main
    body is grid32/local256, **VGPR56/SGPR128/LDS9728/scratch0**, with an
    **84.96-us** warm dispatch plus **1.36-us** merge. K128 underfills the
    device while doubling serial token work per block. Remove the kernel,
    export, wrapper, registry, and harness selector. Production remains
    **20.496816 tok/s**. The mandatory lineage script was independently
    blocked by its missing read-only Atlas checkout; in-tree Laguna and cached
    K64 lineage were audited directly. Evidence:
    [`GQA9/K128 rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-gqa9-splitk128-rejected.json).
71. Raise the retained mixed40 block from local384 to local512 without
    collapsing its 40-owner grid. **Retained and promoted on gfx1151:** four
    extra wave32s expand score/transport concurrency while the same 40
    workgroups continue to fill all 40 CUs. Every QK reduction, exponent,
    denominator addition, PV FMA, gate, and output conversion retains the
    local384 arithmetic association.

    The 9x50 and stronger 21x100 leaves improve
    **0.037079 -> 0.030474 ms (-17.814%)** and
    **0.037041 -> 0.030142 ms (-18.623%)**. Both F32 context and gated BF16
    are byte-exact, and all 21 strong candidate samples beat their controls.
    The launch bound also changes code allocation from **104 -> 32 VGPRs**;
    cache-only production tracing records local512, grid40,
    **SGPR128/LDS25,600/scratch0**.

    Seven counterbalanced actual-model p512/d128 pairs all improve, moving
    median decode **20.472516 -> 20.542123 tok/s (+0.34000%)**, or
    **-0.16552 ms/token** by independent medians. Median paired saving is
    **0.16731 ms/token**. Every run preserves the identical 128-token
    trajectory SHA, first/final tokens, position, and lifecycle. A separate
    127-transition trace names exactly **4,572 = 36 x 127** local512 calls,
    proving production dispatch. The local384 body remains the exact rollback;
    other backends are unchanged. Tracked-clean selector-unset production is
    **20.550788/20.559001/20.557302 tok/s**, median
    **20.557302 tok/s** (**48.645 ms/token**): **+0.29510% /
    -0.14355 ms/token** over the preceding clean packet and **+79.278%** over
    sprint start. All three runs report the local512 capability active and
    retain exact trajectory/state/lifecycle. Evidence:
    [`local512 retention`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-mixed40-local512-retained.json),
    [`clean production`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-mixed40-local512-production.json).
72. Raise the retained 32-owner global block from local256 to local512 while
    freezing its exact denominator tree. **Retained and promoted on gfx1151:**
    the naive first draft correctly fails byte identity at live513 because
    spreading denominator terms over sixteen waves changes FP32 association.
    The retained sibling lets all sixteen waves partition independent QK and
    value transport, but keeps the original eight-wave softmax issue and
    denominator ownership.

    The repaired 9x50 leaves improve live513/576/639 by
    **25.47%/23.44%/30.48%**. Strong 21x100 leaves improve
    **0.054995 -> 0.040636 ms (-26.11%)**,
    **0.060632 -> 0.046423 ms (-23.43%)**, and
    **0.072917 -> 0.050667 ms (-30.51%)**. Every F32 context and gated BF16
    output is byte-identical. Cache-only tracing names the intended
    grid32/local512 specialization at **VGPR48/SGPR128/LDS512/scratch0**.

    All seven actual-model p512/d128 pairs improve:
    **20.581562 -> 20.726022 tok/s (+0.70189%)**, or
    **48.5872 -> 48.2485 ms/token (-0.33865 ms)**. Median paired saving is
    **0.34034 ms/token**. Every run preserves the identical generated-token
    SHA, first/final tokens, position, deterministic state, and allocation
    lifecycle. local256 remains the exact rollback; peer backends remain
    unchanged. Evidence:
    [`global local512 retention`](../benchmarks/results/2026-07-30-gfx1151-laguna-global-mixed32-local512-retained.json).

    Tracked-clean selector-unset production at `96a19bc81` measures
    **20.705514/20.727439/20.717479 tok/s**, median
    **20.717479 tok/s (48.268 ms/token)**. This is **+0.77917% /
    -0.37609 ms/token** over the preceding clean packet and **+80.675%** over
    sprint start. All runs report the capability active and preserve exact
    trajectory/state/lifecycle. Evidence:
    [`clean global local512 production`](../benchmarks/results/2026-07-30-gfx1151-laguna-global-mixed32-local512-production.json).

    A tracked-clean 127-transition cache-only census confirms the cumulative
    local512 gains in the complete decode graph. Global attention falls
    **0.992343 -> 0.659276 ms/token (-33.564%)**, SWA falls
    **1.754009 -> 1.565501 ms/token (-10.747%)**, and total attention falls
    **2.746352 -> 2.228985 ms/token (-18.838%)**. Median kernel sum/span are
    now **46.404/48.563 ms/token**. The trace names exactly 12
    grid32/local512 global calls and 36 grid40/local512 SWA calls per token,
    with no scratch. Evidence:
    [`post-local512 wall census`](../benchmarks/results/2026-07-30-gfx1151-laguna-post-local512-wall-reprofile.json).

73. Revisit producer-wave V-tail transport only after local512 changes its
    loader balance. **Retained and promoted on gfx1151:** the two exact
    tail-probability waves copy the final 64/32 staged-V vectors in pair and
    singleton blocks, while the other 14/15 waves copy the prefix. This keeps
    QK, softmax, denominator, PV, gate, and store association unchanged.

    The 9x50 leaf improves **0.031528 -> 0.030106 ms (-4.512%)**. The stronger
    21x100 leaf improves **0.031737 -> 0.030061 ms (-5.282%)**, wins all 21
    pairs, and is byte-identical in both F32 context and gated BF16. Native
    tracing confirms grid40/local512, **VGPR32/SGPR128/LDS25,600/scratch0**.
    All seven counterbalanced actual-model p512/d128 pairs improve
    **20.718104 -> 20.737481 tok/s (+0.09353%, -0.04510 ms/token)** with
    exact trajectory/state/lifecycle. The previous local512 route remains
    exact rollback, peer backends remain unchanged, and the temporary
    comparison selector is removed.
    Evidence:
    [`local512 value-tail retention`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-local512-value-tail-retained.json).

    Tracked-clean selector-unset production at `e69f28bc6` measures
    **20.715636/20.731612/20.732043 tok/s**, median
    **20.731612 tok/s (48.236 ms/token)**. This is **+0.06821% /
    -0.03290 ms/token** over the preceding clean packet and **+80.799%** over
    sprint start. The new capability is active in all runs; exact
    trajectory/state/lifecycle passes. Evidence:
    [`clean local512 value-tail production`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-local512-value-tail-production.json).

74. Re-screen four-vector denominator prefetch only after local512 and
    producer-value-tail transport materially change the kernel. **Rejected
    and removed at the resident gate:** the candidate issues four adjacent
    probability `ds_load_b128` operations before consuming the same 16
    components in the original order. QK, softmax, denominator, PV, gate, and
    store association remain unchanged.

    The 9x50 and 21x100 byte-exact leaves improve
    **0.031314 -> 0.030273 ms (-3.322%)** and
    **0.031099 -> 0.030302 ms (-2.563%)**, winning all **9/9** and **21/21**
    pairs. The complete model rejects the isolated saving: seven
    counterbalanced p512/d128 pairs move **20.734191 -> 20.731204 tok/s
    (-0.01440%, +0.00695 ms/token)**, median paired change is **-0.00713%**,
    and only **3/7** pairs improve. Exact trajectory/state/lifecycle passes.
    Remove the kernel, capability, runtime selector, and comparison CLI.
    Production remains **20.731612 tok/s**. Evidence:
    [`local512 denominator-prefetch4 rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-local512-denom-prefetch4-runtime-rejected.json).
75. Re-screen V-stage128 only after local512, shared probability replay, tail
    producers, and producer-wave V-tail transport change the synchronization
    balance. **Retained and promoted on gfx1151:** the exact successor keeps
    the 40-owner/local512 grid and every QK, softmax, denominator, PV, gate,
    conversion, and store association, while the fixed 512-slot replay falls
    from eight stages/sixteen barriers to four stages/eight barriers.

    The 9x50 and stronger 21x100 leaves improve
    **0.031164 -> 0.028979 ms (-7.011%)** and
    **0.031216 -> 0.029120 ms (-6.717%)**. Every candidate leaf sample wins,
    and both F32 context and gated BF16 output are byte-identical. Cache-only
    tracing names grid40/local512 at
    **VGPR176/SGPR128/LDS43,008/scratch0**.

    Seven counterbalanced actual-model p512/d128 pairs move
    **20.736052 -> 20.745421 tok/s (+0.04518%, -0.02178 ms/token)**;
    median paired change is **+0.04193%** with **6/7** wins. Every run
    preserves tokens **2930/74107**, trajectory SHA `94f803f7...bda32`,
    position 638, determinism, and allocation recovery. Promote V128 at the
    already qualified gfx1151 natural shape, retain the V64 symbol as exact
    rollback, remove the temporary comparison selector, and leave peer
    backends unchanged. Evidence:
    [`local512 V-stage128 retention`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-local512-vstage128-retained.json).

    Tracked-clean selector-unset production at retained `568e8ae93` is
    **20.728553/20.744351/20.751098 tok/s**, median
    **20.744351 tok/s (48.206 ms/token)**. This is **+0.06145% /
    -0.02962 ms/token** over the preceding clean packet and **+80.910%** over
    sprint start, with exact repeated trajectory/state/lifecycle:
    [`V-stage128 production`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-local512-vstage128-production.json).
76. Bound V-stage128's 32-item vector probability/PV loop to unroll factor 16.
    **Rejected and removed:** the byte-exact 9x50 and 21x100 leaves improve
    **0.030454 -> 0.029083 ms (-4.503%)** and
    **0.030667 -> 0.029348 ms (-4.302%)**, with all 21 strong pairs positive.
    The scheduling change does not achieve its resource goal: control and
    candidate both trace at **VGPR176/SGPR128/LDS43,008/scratch0**.

    Seven counterbalanced actual-model pairs move
    **20.752041 -> 20.751527 tok/s (-0.00248%)**, or
    **48.18803 -> 48.18923 ms/token (+0.00119 ms)**. Median paired change is
    **+0.00554%** with **5/7** wins, but the independent headline regresses.
    Remove the kernel, wrapper, registry entry, runtime selector, comparison
    CLI, and oracle addition. Production remains **20.744351 tok/s**.
    Pragma-only unroll tuning is closed; the next candidate must structurally
    shorten the live range and measurably reduce VGPR or complete-model wall.
    Evidence:
    [`V-stage128 bounded16 rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-vstage128-bounded16-rejected.json).
77. Audit the retained V64/V128 device code before another register-pressure
    experiment. **Priority corrected:** the authoritative gfx1151 code-object
    metadata reports V64/V128 at **32/35 logical VGPR**, **32/32 SGPR**, zero
    spills/private segment, and **25,564/42,716 B LDS**. The trace column's
    V128 value **176** is not 176 live logical registers.

    V128's 40 workgroups exactly match the device's 40 CUs, with 16 wave32s
    per workgroup. Its LDS footprint prevents a second resident workgroup, but
    this dispatch has no second workgroup per CU to schedule. The bounded16
    result therefore failed because it did not materially change code
    resources, not because the compiler ignored a 176-register emergency.

    Applying the complete **6.717%** leaf reduction to the pre-V128
    **1.565501-ms/token** SWA census gives an optimistic
    **0.1052-ms/token / 0.219%** whole-decode ceiling. More pragma/stage-width
    micro-tuning cannot close the **5.376-ms/token** Vulkan wall gap. Require
    a genuinely new arithmetic/traffic premise before returning to SWA and
    move next to the larger selected-MoE comparator gap:
    [`V-stage code-object audit`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-vstage-codeobject-audit.json).
78. Fully unroll the fixed twelve-block K3072 loop in the exact fused selected
    gate/up tile8 owner. **Rejected and removed:** the actual-weight 9x50 and
    21x100 leaves improve **0.134683 -> 0.133444 ms (-0.920%)** and
    **0.134171 -> 0.132723 ms (-1.080%)**, with **18/21** strong paired wins
    and zero BF16 mismatches.

    The authoritative code object improves logical VGPR **96 -> 94** and SGPR
    **34 -> 31**, with zero spills and unchanged 256-byte fixed LDS, but
    expands kernel text **3,400 -> 17,220 bytes (5.06x)**. Cache-only tracing
    names both local128/grid16384x10 specializations at allocated VGPR96,
    LDS512, and scratch0.

    Seven counterbalanced actual-model p512/d128 pairs reverse the leaf result:
    **20.743597 -> 20.689042 tok/s (-0.2630%)**, or
    **48.20765 -> 48.33476 ms/token (+0.12712 ms)**. Production wins
    **7/7** pairs; median paired change is **-0.27974%**. Tokens, trajectory,
    positions, determinism, and allocation teardown remain exact. Remove the
    kernel, wrapper, registry route, oracle addition, leaf selector, runtime
    route, and comparison CLI. Production remains **20.744351 tok/s**.
    Fixed-K full unrolling is closed for this selected owner; the next LD-4
    candidate must reduce weight/address traffic or fuse useful cross-output
    work without multiplying instruction footprint:
    [`selected gate/up unroll12 rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-selected-gate-up-unroll12-rejected.json).
79. Transfer SWA's wider exact value staging back to the retained global
    local512 owner. **Rejected and removed:** V-stage128 halves the staged-V
    iterations and barrier pairs while preserving every QK, maximum, `expf`,
    denominator, probability pre-normalization, PV, gate, conversion, and
    store association. The wrapped/evicted oracle is F32/BF16 byte-exact.

    Cache-hot 9x50 leaves improve **1.522%/1.164%/1.233%** at
    live513/576/639. Native tracing confirms the intended
    grid32/local512 specializations at unchanged allocated
    **VGPR48/SGPR128/scratch0**; candidate dynamic LDS remains bounded at
    **38,960/39,680/40,448 bytes**.

    The complete p512/d128 gate is flat and fails retention:
    **20.736505 -> 20.737910 tok/s (+0.00678%, -0.00327 ms/token)**,
    with only **3/7** paired wins and a **-0.00540%** median paired change.
    Every trajectory, position, and allocation lifecycle remains exact.
    Remove the kernel, wrapper, registry route, oracle addition, leaf selector,
    runtime selector, and comparison CLI. Production remains
    **20.744351 tok/s**. Exact attention stage-width micro-tuning is now
    closed on both SWA and global; move to the traced
    **2.234-ms/token** dense/shared dual-Q4 owner:
    [`global V-stage128 rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-global-local512-vstage128-rejected.json).
80. Contract the exact dual-Q4 owner's repeated scale/min metadata loads with
    one loader per four-lane column quad and shuffle broadcasts. **Rejected
    and removed:** the candidate preserves every dot, FMA, reduction, BF16
    conversion, and store boundary and passes the focused exact oracle, but
    loses all nine leaf pairs at both natural decode shapes.

    Shared M1 K3072 N1024 regresses
    **0.013525 -> 0.034564 ms (+155.555%)**; dense M1 K3072 N12288 regresses
    **0.472077 -> 0.514232 ms (+8.930%)**. Cache-only tracing confirms both
    control and candidate use local32 at grid4096/grid49152 with
    **VGPR96/SGPR128/LDS512/scratch0**. The redundant metadata loads are
    evidently cheap cache/coalescing hits; 32 quad-local shuffles per lane per
    K iteration cost much more than the bytes they replace.

    Skip the resident gate, remove the kernel, wrapper, oracle addition, and
    leaf harness, and keep production at **20.744351 tok/s**. Within-kernel
    metadata broadcast is closed. Next screen the exact consumer boundary:
    fuse the dense/shared dual-Q4 gate/up result with its SiLU-product consumer
    to remove intermediate BF16 traffic and one launch per layer:
    [`dual-Q4 quad-metadata rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-q4-pack8-dual-quadmeta-rejected.json).
81. Fuse the retained dual-Q4 gate/up owner with its immediate BF16
    SiLU-product consumer. **Retained/default:** gate and up preserve their
    independent BF16 rounding points in registers; the kernel widens those
    exact values, performs the existing sigmoid and product expression, and
    writes the same BF16 intermediate. Capability/registry/layout/shape/quant
    misses retain the unfused pair-plus-SiLU chain.

    Actual-weight 21x100 leaves improve shared M1 K3072 N1024
    **0.014770 -> 0.012433 ms (-15.824%, 21/21 wins)** and dense M1 K3072
    N12288 **0.474136 -> 0.469647 ms (-0.947%, 19/21 wins)** with zero BF16
    mismatches. Cache-only tracing names the intended `true` specialization at
    grid4096/grid49152, local32, **VGPR96/SGPR128/LDS512/scratch0**.

    All seven actual-model p512/d128 pairs improve
    **20.756829 -> 20.810024 tok/s (+0.2563%)**, or
    **48.17692 -> 48.05376 ms/token (-0.12315 ms)**. Median paired speedup is
    **+0.26088%**; tokens, trajectory, position, and allocation lifecycle are
    exact. The default removes **48 launches/token** plus 483,328 bytes/token
    of temporary gate/up write-read traffic. Three tracked-clean,
    selector-unset production runs measure
    **20.785471/20.803189/20.804183 tok/s**, median
    **20.803189 tok/s (48.06955 ms/token)**. This is **+0.28363% /
    -0.13634 ms/token** over the preceding clean packet and **+81.423%** over
    sprint start, with exact repeated trajectory/state/lifecycle:
    [`dual-Q4 plus SiLU retention`](../benchmarks/results/2026-07-30-gfx1151-laguna-q4-pack8-dual-silu-retained.json),
    [`clean production`](../benchmarks/results/2026-07-30-gfx1151-laguna-q4-pack8-dual-silu-production.json).
82. Re-profile the tracked-clean post-fusion wall. **Accepted attribution
    checkpoint:** compute dispatches fall exactly **721 -> 673/token**. The
    direct dual-plus-SiLU window falls
    **2.286671 -> 2.216017 ms/token (-3.090%)**; total median kernel sum and
    span fall **46.404232 -> 46.244170 ms (-0.345%)** and
    **48.563428 -> 48.300880 ms (-0.541%)**.

    Source-F16 remains the largest absolute family at **24.050752 ms/token**,
    but it is already **0.707816 ms/token faster** than Vulkan's perturbed
    logger row. The remaining named comparator gaps are attention
    **1.219931 ms/token**, selected gate/up **0.915715**, and selected down
    **0.270471**. Do not re-rank source-F16 merely by absolute time. The next
    bounded boundary is exact T16 selected down plus slot-ordered routing
    weighting, transferring the existing in-tree IQ3 composite without
    changing the ten per-route BF16 projection boundaries:
    [`post-fusion wall census`](../benchmarks/results/2026-07-30-gfx1151-laguna-post-q4-dual-silu-wall-reprofile.json).
83. Fuse natural T16 selected down with slot-ordered routing weights.
    **Complete, rejected, and removed:** one local128 workgroup owns each
    16-column output tile across all ten routes. Every route keeps the retained
    K/FMA tree, ordered four-wave sum, and BF16 projection boundary before the
    unchanged FP32 `fmaf` routing chain. Q4 and planar-Q6 focused oracles are
    bit-exact.

    The strong 21x100 actual-weight leaf is mixed: Q4 moves
    **0.061010 -> 0.061287 ms (+0.455%)**, while Q6 moves
    **0.076066 -> 0.074024 ms (-2.685%)**. A Q6-only temporary runtime owner
    then fails the complete three-pair p512/d128 gate:
    **20.806487 -> 20.632451 tok/s (-0.8364%)**, or
    **48.06193 -> 48.46734 ms/token (+0.40540 ms)**. All next/final tokens,
    the 128-token trajectory hash, positions, and allocation lifecycle remain
    exact.

    Serial route ownership removes one launch and 61,440 scratch bytes per Q6
    layer but destroys too much route-level memory parallelism. Remove the
    kernel, wrappers, key, runtime selector, comparison CLI, harness seam, and
    oracle extension. Production stays **20.803189 tok/s**. Do not retry
    serial top-10 ownership for natural T16; a viable weighted fusion must
    retain route-parallel weight streaming:
    [`Q6 natural-weighted rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-q6-natural-weighted-decode-rejected.json).
84. Transfer llama.cpp's grouped-query V reuse into two exact scalar PV
    accumulators per wave. **V64 leaf rejected and removed:** the source audit
    confirms the exact comparator path at `c0bc8591e`: RADV exposes subgroup64
    plus `VK_KHR_cooperative_matrix`; host GQA folding changes decode `N=1` to
    global/SWA `N=6/9` before final tuning, selecting coopmat1
    **Br16 x Bc64/local256** plus K64 split-K. The transferable premise is one
    V tile serving multiple query accumulators, not the Vulkan API.

    The exact HIP screen keeps mixed40/local512/V64 ownership but maps each
    natural query pair onto the same four output waves. Every query retains its
    original 512 slot-order FMA chain while both accumulators reuse one staged
    BF16 V load. RED fails on the absent wrapper; GREEN is F32-context and
    gated-BF16 byte-exact through wrap and explicit eviction.

    The 9x50 cached leaf regresses
    **0.031300 -> 0.034642 ms (+10.675%)**, losing all nine pairs. Halving LDS V
    reads cannot repay halving active scalar PV wave breadth; cooperative
    grouped-query reuse does not transfer as serial scalar accumulation. Remove
    the kernel, wrapper, test, and harness seam before commit. Production
    remains **20.803189 tok/s**. This closes only the unchanged V64 form; the
    next distinct exact screen keeps production V128/PV breadth and assigns two
    exponent-producer waves per query:
    [`paired-PV V64 rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-paired-pv-v64-rejected.json).
85. Split each retained V128 probability row across two tail waves.
    **Retained and promoted on gfx1151:** pair blocks use waves 12/13 for the
    first query and 14/15 for the second; singleton blocks use waves 14/15.
    Each producer evaluates two 32-slot chunks per V128 stage instead of one
    wave evaluating four chunks. The eight pair-block PV waves, four
    singleton PV waves, ordered denominator replay, QK reductions, output
    FMAs, gates, stores, launch count, and resident bytes remain unchanged.

    RED fails importing the absent wrapper. GREEN passes the wrapped/evicted
    CPU-reference oracle with byte-identical F32 context and gated BF16
    output. The 9x50 leaf moves **0.030255 -> 0.029404 ms (-2.811%)**.
    The stronger 21x100 leaf moves
    **0.030752 -> 0.029131 ms (-5.271%)**. Cache-only native tracing names the
    final-`true` specialization at grid40/local512 with the same
    **VGPR176/SGPR128/LDS43,008/scratch0** resource row as retained V128.

    Seven counterbalanced actual-model p512/d128 pairs move median decode
    **20.806774 -> 20.809401 tok/s
    (+0.01262%, -0.00607 ms/token)** with **5/7** wins. Every row preserves
    tokens **2930/74107**, trajectory SHA `94f803f7...bda32`, final position
    638, repeat determinism, and allocation teardown. This small exact win is
    the scalar part of llama.cpp's probability-lane distribution that
    transfers without its F16 cooperative arithmetic. Promote it for the
    already-qualified gfx1151 saturated shape, retain single-producer V128 as
    exact rollback, remove comparison-only registry replacement, and leave
    peer backends unchanged:
    [`retention`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-dual-tail-producer-vstage128-retained.json).

    Tracked-clean selector-unset production at `72ed34b08` is
    **20.798934/20.811150/20.803739 tok/s**, median
    **20.803739 tok/s (48.06828 ms/token)**. That is a noise-floor
    **+0.00264% / -0.00127 ms/token** over the preceding clean packet and
    **+81.428%** over sprint start; exact repeated state and lifecycle pass:
    [`clean production`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-dual-tail-producer-vstage128-production.json).
86. Move V128 probability production onto the output waves.
    **Retained and promoted on gfx1151:** each of the four pair-block output
    waves per query now evaluates one 32-slot probability shard before
    consuming the unchanged ordered denominator and scalar F32 PV chain.
    Singleton output waves use the same mapping. All 16 waves cooperatively
    load staged V, replacing the asymmetric dual-tail loader schedule without
    changing QK, denominator, PV, gate, store, launch, or resident boundaries.

    RED fails importing the absent wrapper. GREEN is byte-identical for F32
    context and gated BF16 through wrap and explicit eviction. The 9x50 leaf
    moves **0.030667 -> 0.028742 ms (-6.278%)**; the stronger 21x100 leaf moves
    **0.030266 -> 0.028760 ms (-4.976%)**. Cache-only tracing names the intended
    final-`false,true` specialization at grid40/local512 with unchanged
    **VGPR176/SGPR128/LDS43,008/scratch0**.

    Seven counterbalanced actual-model p512/d128 pairs move median decode
    **20.803377 -> 20.816723 tok/s
    (+0.06415%, -0.03082 ms/token)**, with **7/7** paired wins and median
    paired saving **0.03614 ms/token**. Every run preserves tokens
    **2930/74107**, trajectory SHA `94f803f7...bda32`, final position 638,
    repeat determinism, and allocation teardown. Promote the exact output-wave
    schedule for the already-qualified saturated gfx1151 shape, retain the
    dual-tail V128 key as exact rollback, and leave peer backends unchanged:
    [`retention`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-output-sharded-probability-vstage128-retained.json).

    Tracked-clean selector-unset production at `a8a91efab` is
    **20.798681/20.814372/20.800509 tok/s**, median
    **20.800509 tok/s (48.07575 ms/token)**. That is a noise-floor
    **-0.01553% / +0.00746 ms/token** versus the preceding clean packet and
    **+81.399%** over sprint start. The retention claim therefore rests on the
    exact **4.976%** leaf win and **7/7** positive interleaved full-model pairs,
    not this noisy three-run publication. Exact repeated state/lifecycle pass:
    [`clean production`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-output-sharded-probability-vstage128-production.json).
87. Parallelize the output-sharded producer-maximum replay.
    **Complete, rejected, and removed:** lanes 0-15 load the sixteen score-wave
    maxima and replay their exact maximum through a wave32 shuffle tree instead
    of lane 0 reading them serially. The score maximum, scalar F32 QK,
    denominator, PV, gate, store, launch, and resident boundaries are
    unchanged.

    RED fails importing the absent wrapper. GREEN is byte-identical for F32
    context and gated BF16 through wrap and explicit eviction. The 9x50 leaf
    improves **0.029815 -> 0.028933 ms (-2.957%, 9/9 wins)**; the stronger
    21x100 leaf improves
    **0.029644 -> 0.028706 ms (-3.166%, 21/21 wins)**. Cache-only tracing
    names the intended final-`true` specialization at grid40/local512, but
    both variants remain **VGPR176/SGPR128/LDS43,008/scratch0**.

    Seven counterbalanced actual-model p512/d128 pairs reject the candidate:
    median decode moves **20.815600 -> 20.813188 tok/s
    (-0.01159%, +0.00557 ms/token)**, with only **1/7** wins and a negative
    **-0.70697 ms/sample** median paired saving. Tokens **2930/74107**,
    trajectory SHA `94f803f7...bda32`, final position 638, repeat
    determinism, and allocation teardown remain exact. Remove the kernel,
    wrapper, test extension, leaf seam, and comparison route. Production
    remains **20.800509 tok/s**. Maximum-replay-only scheduling is closed
    unless a larger score-production or synchronization topology changes:
    [`rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-output-sharded-parallel-max-rejected.json).
88. Reprofile the retained output-sharded production wall and select the next
    bandwidth gate.
    **Complete, accepted attribution checkpoint:** a cache-only
    `rocprofv3 --kernel-trace` run on tracked-clean `36cfef876` records 127
    exact decode transitions and **673 dispatches/token**. Median device
    kernel sum is **46.214841 ms/token** and kernel span is
    **48.262162 ms/token**. Relative to the post-Q4-SiLU census, attention
    improves **2.129354 -> 2.093607 ms/token (-1.679%)**, driven by SWA
    **1.468158 -> 1.428844 ms/token (-2.678%)**; global is flat within trace
    noise at **0.657957 -> 0.659514 ms/token**.

    The same-GGUF Vulkan family bridge now leaves three named positive gaps:
    attention **+1.184184 ms/token**, selected gate/up
    **+0.922848 ms/token**, and selected down **+0.272274 ms/token**.
    Source-F16 remains **0.711684 ms/token faster** than Vulkan. Attention is
    still the largest comparator gap, but the current scalar-exact topology's
    bounded ownership, replay, stage, and register screens are exhausted; a
    successor needs a materially different exact or quality-gated topology.

    Selected gate/up provides the next concrete production screen. Its
    retained T16 owner streams **1.709507 GB/token** in **8.386938 ms**, or
    **203.83 GB/s**. The scale/min expansion alone makes T16 **2.778% larger**
    than the existing exact byte-neutral qmicro layout. Qmicro would stream
    **1.663304 GB/token**, with a **7.526 ms** floor at the measured
    **221 GB/s** read anchor, nearly Vulkan's **7.464090 ms**. Build a
    production-shaped tile8, parallel-tail, fused-SiLU qmicro consumer before
    changing resident selection. Tokens **2930/74107**, trajectory, final
    position, determinism, and allocation recovery remain exact:
    [`wall census`](../benchmarks/results/2026-07-30-gfx1151-laguna-output-sharded-wall-reprofile.json).
89. Consume byte-neutral qmicro in the production tile8 fused-SiLU boundary.
    **Complete, rejected, and removed:** build an exact qmicro sibling with
    the retained K ownership, FMA order, wave32 tree, wave-0..3 merge, BF16
    gate/up round trips, SiLU expression, and BF16 output. It reduces the
    actual gate/up pair from **931,135,488 -> 905,969,664 resident bytes**
    and remains BF16-byte exact.

    Three actual layer-1 9x50 screens reject the decode cost. Cooperative LDS
    record expansion regresses **0.143908 -> 0.183350 ms (+27.408%)**.
    Removing all expansion barriers and decoding records per lane improves the
    candidate but still regresses **0.139026 -> 0.168395 ms (+21.124%)**.
    One unaligned dword load on lane 0 plus wave broadcast regresses
    **0.135095 -> 0.197774 ms (+46.396%)**. The 2.778% byte reduction cannot
    pay for scale/min unpack at c=1. Remove every new kernel, wrapper,
    conversion helper, test, and comparison seam; keep T16 production and the
    older generic qmicro primitive as diagnostic evidence only. Production
    remains **20.800509 tok/s**:
    [`rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-q4-qmicro-tile8-silu-rejected.json).
90. Reuse the packed T16 Q payload across adjacent output columns.
    **Complete, retained, and promoted pending clean publication:** the
    resident T16 Q plane already places adjacent output-column nibbles in one
    byte. Load that byte once per pair and extract low/high nibbles while
    preserving the T16 layout, coefficient loads, K ownership, FMA order,
    wave32 tree, wave merge, BF16 gate/up boundaries, SiLU expression, and
    BF16 output.

    The actual layer-1 counterbalanced 21x100 leaf improves
    **0.131761 -> 0.129199 ms (-1.945%)** with zero BF16 mismatches. Cached
    tracing names the intended scalar-Q and pair-Q specializations and keeps
    both at local128/VGPR96/SGPR128/LDS512/scratch0. Seven resident p512/d128
    pairs all improve, moving median decode
    **20.811539 -> 20.820664 tok/s (+0.04385%, -0.02106 ms/token)** with
    exact tokens, trajectory, final position, determinism, and allocation
    recovery. Promote pair-Q behind the existing production variant name;
    retain the scalar-Q sibling briefly as an explicit compiler/codegen
    rollback:
    [`retention`](../benchmarks/results/2026-07-30-gfx1151-laguna-q4-t16-pairq-retained.json).

    **Clean publication passes:** tracked-clean selector-unset production is
    **20.823569/20.830515/20.832851 tok/s**, median
    **20.830515 tok/s (48.00649 ms/token)**. This is **+0.14426% /
    -0.06925 ms/token** versus the prior clean checkpoint and **+81.661%**
    over sprint start. All repetitions preserve exact trajectory/state and
    allocation teardown:
    [`production`](../benchmarks/results/2026-07-30-gfx1151-laguna-q4-t16-pairq-production.json).

Current exact decode checkpoint:

| Backend / checkpoint | Decode | Wall/token | Relative to sprint start |
| --- | ---: | ---: | ---: |
| hipEngine sprint start | **11.466687 tok/s** | **87.209 ms** | baseline |
| hipEngine current production | **20.830515 tok/s** | **48.006 ms** | **+81.661%** |
| same-GGUF llama.cpp Vulkan | **23.348381 tok/s** | **42.830 ms** | directional comparator |
| Remaining wall gap | — | **5.177 ms/token** | hipEngine is **10.784%** below Vulkan throughput |

The producer-max and local512 results capture two exact pieces of llama.cpp's
advantage: cooperative work should be computed by the waves that already own
the data, and a block should expose enough independent work to cover the
machine. Local512 is the natural saturated-SWA endpoint because its 512 lanes
cover one logical token each; larger blocks add no score ownership. Global
local512 confirms that the same saturation axis transfers when the original
eight-wave denominator tree is held fixed: the extra waves help only the
independent QK and value-transport phases. The post-fusion census leaves SWA
as the dominant exact-attention family at **1.468158 ms/token**. Total
attention is **2.129354 ms/token**, or **2.341x** llama.cpp Vulkan's logged
**0.909423 ms/token**, a **1.219931-ms/token** gap representing **23.28%** of
the complete remaining wall gap. Device-code inspection now closes
pragma-only register/stage tuning:
V128 is **35 logical VGPR with no spills**, not 176 live registers, and its
grid already maps one workgroup to each of the 40 CUs. The first selected-MoE
address/K-loop contraction is now closed: full K3072 unrolling wins the
isolated leaf but loses all seven resident pairs while expanding code 5.06x.
The traced dense/shared dual-Q4 owner now consumes its SiLU boundary in-kernel,
removing 48 launches/token and exact temporary BF16 traffic; tracked-clean
production confirms the complete-model win. Quad-local metadata broadcast
remains closed by decisive natural-shape regressions. Exact T16 selected down
plus routing weighting is also closed: serial top-10 ownership regresses
complete decode by **0.8364%** despite an isolated planar-Q6 leaf win. Return
to attention only for a materially new exact-association or quality-safe
cooperative premise:
both SWA and global stage-width successors win isolated leaves but fail to
produce a reliable complete-model improvement.
The comparator audit confirms why direct copying fails: the same-GGUF
llama.cpp command requests F16 K/V, and its non-BF16 cooperative shader uses
F16 accumulator/output types. hipEngine's BF16-KV recurrent contract rejects
even tensorized PV alone on every complete SWA role. A new candidate must
therefore preserve scalar PV association or prove a genuinely tighter
higher-precision cooperative result; selective deployment is not a repair.
The approximate output itself is not a usable repair oracle. Reusing scalar
ownership, global score planes, approximate online merges, or output-derived
midpoint repair at either owner or component granularity is closed by the
measured failures. Packed-BF16 QK dot2 is also closed: its compensated path is
slower and its one-term path spends far more than the complete quality budget
for a sub-percent leaf gain.

Scalar ownership is closed around the retained mixed32/mixed40 points: 24, 32,
36, 40, split pair/triple, and whole-GQA owners have all been measured. Stage
probability caching and pre-normalization now supply the exact coupling in both
SWA and global attention. Both saturated retained owners now use local512 at
unchanged grid breadth, with softmax association frozen where required.
On the retained global three-term GQA6/K64 sibling,
left-versus-right component association changes thousands of F32 contexts but
zero gated BF16 bins, while forward-versus-reverse FP64 split merge changes no
output; both miss all **3/3** known exact BF16 errors. Do not reconstruct the
removed SWA WMMA body or use approximation agreement as a replay oracle.

LD-4 meanwhile transfers an exact gfx1100 structural lesson without changing
geometry. The existing local32 Q4 pack8 dual body now owns c=1 gate/up for all
**47 shared layers plus the leading dense layer**. Each projection preserves
the singleton K partition, FMA/reduction tree, and BF16 store. Direct leaves
improve **20.25-24.81%** at the shared shapes and **12.48%** at dense
K3072/N12288, with byte-identical outputs.

All seven wired p512/d128 pairs improve
**19.556271 -> 19.645185 tok/s (+0.4547%, -0.2314 ms/token)** with exact
tokens, trajectory hash, positions, determinism, and lifecycle. The required
cache-only trace records **5,969 shared + 127 dense** dual launches at
local32/VGPR96/SGPR128/LDS512/scratch0, removes exactly **48 compute
launches/token (816 -> 768)**, and cuts the complete Q4 family
**3.018303 -> 2.836943 ms/token (-6.01%)**. Two singleton launches remain the
registered backend/shape/layout fallback. Three tracked-clean selector-unset
runs publish **19.620780/19.630076/19.639015 tok/s**, median
**19.630076**: **+0.3495% / -0.1780 ms/token** over the preceding clean
19.561715 packet and **+71.192%** over the 11.466687 sprint start. Evidence:
[`retained exact Q4 pack8 dual decode`](../benchmarks/results/2026-07-29-gfx1151-laguna-q4-pack8-dual-decode-retained.json) ·
[`clean Q4 pack8 dual production`](../benchmarks/results/2026-07-29-gfx1151-laguna-q4-pack8-dual-decode-production.json).

LD-4's first exact seam is now retained. The gate/up sibling fixes
`x_rows=1, rows=10, K3072, N1024`; the Q4 and planar-Q6 down siblings fix ten
distinct intermediate rows at `K1024, N3072`. They retain the full local128
grid, every thread's K ownership, FMA/reduction order, resident T16 bytes, and
BF16 boundary. The corrected actual-weight leaf improves gate/up
**0.144513 -> 0.142154 ms (-1.63%)**, Q4 down
**0.074293 -> 0.058550 ms (-21.19%)**, and Q6 down
**0.082136 -> 0.073648 ms (-10.33%)**, all byte-exact. An earlier diagnostic
down row that broadcast one activation was discarded before integration; the
retained wrapper requires all ten production rows.

Seven counterbalanced p512/d128 pairs improve current production
**16.850003 -> 16.976046 tok/s (+0.748%, -0.441 ms/token)**. Every candidate
beats every control, trajectories and lifecycle match, and cached tracing
records exactly **5,969 Q4 gate/up + 3,048 Q4 down + 2,921 Q6 down** natural
calls with zero generic selected-T16 decode fallback. All bodies are
local128/SGPR128/LDS512/scratch0; VGPR is **200/104/80** respectively.
gfx1151 now selects the natural siblings automatically, while peer backends
and non-natural shapes retain the generic routes.

LD-4's bounded accumulator-width sweep then retains an exact gate/up tile8
owner. Each resident 16-column T16 tile is divided across two workgroups, but
each output column keeps the natural owner's thread K partition, FMA order,
wave32 tree, ordered wave-0..3 sum, and BF16 store. This lowers allocated VGPR
**200 -> 96** with unchanged local128/LDS512/scratch0 and no new resident
bytes. The actual-weight leaf improves **5.35-7.13%** with zero mismatches.
Tile4 and two separate single-projection launches are exact but regress
**10.51%/9.15%** and are removed.

Seven counterbalanced production pairs move
**16.991621 -> 17.007001 tok/s (+0.091%, -0.053 ms/token)** with **7/7**
wins, identical generated hashes/final state, and exact lifecycle recovery.
Cache-only tracing records all **5,969 = 47 x 127** tile8 calls and zero
natural tile16/generic gate/up fallback. The profiled selected family is now
about **13.224 ms/token**. gfx1151 defaults tile8 only inside the already
qualified natural shape; the registered tile16/generic paths remain rollback
and shape/backend fallbacks.

1. **LD-1 — D128 grouped-GQA split-K attention.** Adapt the proven Qwen
   topology, not its D256/qgroup8 constants. Register separate qgroup6 global
   and qgroup9 SWA bodies, split K by 64/128 to retain grid breadth, fuse
   score/softmax/PV, and merge plus gate through bounded state. First require
   attention at or below **3.0 ms/token** at p512; stretch is **1.5 ms**.
   The retained GQA3 score owner is LD-1a. The exact saturated-512 reducer is
   retained at **+2.731%** production and lowers the SWA reducer from about
   **9.959 to 8.322 ms/token**. The natural global specialization is also
   retained at **+0.087%**, putting the global reducer at about
   **2.628 ms/token**. The exact fused-GQA2 saturated SWA owner is now retained
   at **17.065241 tok/s (+0.306%)**, proving that resident-aware K reuse can
   win even when its cache-hot leaf loses. The exact fused one-head global
   owner adds **+0.188%** and reaches **17.097044 tok/s** while preserving all
   48 workgroups; global GQA2 is closed at **-0.126%** because 24 workgroups
   undersubscribe the 40 CUs when it has K reuse alone. The repaired global
   GQA2 V-stage64 body also reuses V across its query pair, wins the leaf
   **9.16-12.39%**, and improves all seven resident pairs
   **18.034298 -> 18.237090 tok/s (+1.124%)**, so it supersedes GQA1 at the
   natural shape. Exact fused-GQA3/local384 then reduces saturated
   SWA K ownership **5 -> 3** while keeping 288 active PV waves; all seven
   resident pairs improve **17.100489 -> 17.139971 tok/s (+0.231%)**, so it is
   now the natural-shape gfx1151 default with fused-GQA2 rollback. Total
   attention still remains far
   above the **3.0 ms/token** target. LD-1b's full-F32 and exact-QK/
   tensorized-PV owners are both quality-rejected. LD-1c's exact local32 GQA3
   reducer, SWA one-head fusion, and paired shared-V GQA2 owner are
   performance-rejected; shared-V alone is **49.070%** slower. LD-1f's
   source-faithful SWA **GQA9 x K64 split-K** proves the missing topology with a
   **57.57%** leaf and **12.89%** production speedup, but max KL **0.314247**
   rejects its online merge. The exact three-dispatch repair wins the leaf
   **26.49%** but regresses production **3.69%**; exact cooperative fusion
   regresses the leaf **77.25%**. All three are removed. Exact fused-GQA2
   64/128-slot LDS V staging also loses **19.04%/17.18%** despite preserving
   eight waves and reducing pairwise V traffic. Rebalancing cooperative GQA9
   from one to four/five active PV waves per block restores cache-hot parity
   (**+0.44%**) but still regresses production **3.963%**; the global
   score/grid phase boundary is the blocker. Exact per-head LDS weight/
   denominator caching is then neutral at **-0.097%** in the leaf. The
   GQA3/local384 body wins once its three heads reuse a 64-slot LDS V tile:
   the exact leaf improves **26.58%** and all seven resident pairs improve
   **17.135411 -> 18.032171 tok/s (+5.233%)**. It is now the saturated
   natural-shape default with unstaged local384 rollback. The pre-V-stage
   clean re-profile measures attention at **11.764 ms/token**, versus
   Vulkan's logged **0.909 ms/token** and about **70%** of the remaining wall
   gap. A cached-binary numerical ablation exactly reproduces the online
   GQA9 result (**19.263 vs 17.081 tok/s**, max KL **0.314247**), but all three
   static one-third SWA depth policies are worse at max KL
   **0.470646/0.762249/0.737907**. The errors partly cancel across depth, so
   layer-bounded deployment is closed as fragile and prompt-overfit-prone.
   Re-running unstaged exact global GQA2 on the post-GQA3 baseline is neutral-
   negative (**17.082284 -> 17.074471 tok/s, -0.0457%**) and confirms the
   24-workgroup schedule needs both K and V reuse. Source review sharpens the
   comparator gap: Vulkan uses a real subgroup64 cooperative-matrix
   **Br16 x Bc64** GQA9 tile before its K64 merge, whereas the rejected HIP
   kernel copied only GQA ownership and split breadth. A true compensated-WMMA
   GQA9/K64 body then proves an **81.09%** leaf win, but fails the complete
   gate at max KL **1.754897**; exact-QK/PV split association also fails at
   **0.810355**. Exact ordered split repair is **21.58%** slower. Cooperative
   split-K is therefore closed under the current exactness contract. A final
   normal-launch persistent GQA9/K64 score-plane repair is byte-exact but
   regresses the leaf **301.99%** at VGPR40/scratch0, proving that changing the
   launch API does not remove the global phase tax. The one-phase
   **2+2+2+3 mixed32** owner is retained instead: leaf **-5.41%**, all seven
   resident pairs **19.268862 -> 19.371717 tok/s (+0.534%)**, exact state. The
   exact GQA4+5/local512 successor is byte-exact but regresses the leaf
   **5.44%** because 16 ordinary-grid blocks underfill gfx1151, so it is
   removed. The 24-block GQA3 exp32 screen is likewise exact but **3.34%**
   slower, closing scalar ordinary-grid SWA ownership below 32 blocks. The
   final 32-block GQA9/D32 dimension-sharded screen is exact but **69.60%**
   slower: fourfold lower V traffic cannot repay fourfold redundant QK and
   nine-head issue pressure. A single-normalization raw-numerator WMMA repair
   retains a **57.95%** leaf win and improves the prior WMMA max KL
   **1.754897 -> 1.426066**, but remains 28.52x over budget and is removed.
   Grouped BF16-midpoint repair is also removed: the first byte-exact guard
   repairs nearly every owner and regresses **0.081869 -> 0.118805 ms
   (+45.12%)**.
   Component-fine replay is worse: guards through **28672/32768** still miss
   BF16 changes; the byte-exact full-interval form regresses
   **0.081644 -> 0.183208 ms (+124.40%)** and is removed.
   The higher-precision cooperative follow-up is also closed. Three
   non-overlapping BF16 components for every F32 query and probability,
   independent F32 WMMA accumulators, raw K64 split numerators, and a
   query-head-parallel FP64 merge reduce the current production leaf
   **0.058925 -> 0.021090 ms (-64.21%)**. The primitive is pointwise
   close—maximum F32 context error **2.24e-8** and only **3/9,216** BF16
   mismatches—but the complete saturated-p512 18-prompt/576-step lane reaches
   max KL **1.353728** at **560/576 (97.22%)** top-1, still **27.07x** over
   budget. A low-32-bit-bin grouped exact replay still leaves two BF16
   mismatches and already regresses the leaf **0.058737 -> 0.090739 ms
   (+54.48%)**. The kernel, runtime selector, oracle extension, and harness
   seams are removed. This isolates changed F32 reduction association—not
   BF16 operand truncation, split normalization, or serialized merge—as the
   remaining tensor-core blocker.
   Exact wave32 exp issue in global GQA2 is retained at **2.25-3.79%** leaf
   and **+0.0479%** complete decode. The next material SWA step requires an
   exact-association cooperative path or a cheap independently valid interval
   bound that proves the retained BF16 rounding bin before sparse replay. Do
   not resume more BF16 decomposition terms, scalar ownership, full-score
   repairs, output-derived midpoint repair, or approximate split softmax/PV.
2. **LD-2 — exact fixed-K F16 GEMV. Complete.** Compile-time
   K3072/K6144/K9216 preserves the proven local256/eight-wave/one-output
   geometry and every arithmetic operation. The weighted family reaches
   **24.482 ms/token**, beating the declared **25.368-ms** target, and exact
   production reaches **16.391201 tok/s (+10.856%)** versus one-barrier.
   Trace coverage is **18,288/18,288** with zero fallback. Local128, wave-owned
   multi-output, and local256 block2 are closed at **+25.79%**, **+107.99%**,
   and **+6.00%** family regressions. Stop F16 geometry work unless a new
   byte-reduction or cache-counter premise appears.
3. **LD-3 — decode-only compressed source-F16 representation.** Closed for
   plain block32 Q8_0/Q8_1, structural layer subsets, and one-sided residual
   weight/activation repair. The best fast all-layer row reaches
   **50.154 ms/token** but max KL **0.497301**; no tested scope clears
   **0.05**. Reopen only for a materially different calibrated representation
   with a predeclared full-model quality premise, not another Q8 residual pass.
4. **LD-4 — selected and dense/shared Q4 decode.** Raw residency and
   reduction-order-changing thread geometry remain closed. Pack8 is only
   **33.33%** larger than raw; raw local128 loses **63.6-85.4%** at actual
   shapes and raw wave32 has no family-wide win. The shape-tuned pack8 owner
   is a repeatable **+1.225%** full-cycle win but fails quality at max KL
   **1.002942** despite **99.31%** top-1. The exact LD-2-style natural-shape
   specialization is retained at **+0.748%** complete decode and lowers the
   profiled selected family from about **13.711 to 13.307 ms/token**. The
   exact tile8 successor then reaches **17.007001 tok/s**, lowers the family
   to about **13.224 ms/token**, and halves gate/up VGPR **200 -> 96**.
   The exact local32 dense/shared Q4 gate/up pair is now retained as well:
   seven pairs improve **19.556271 -> 19.645185 tok/s (+0.4547%)**, remove
   **48 launches/token**, and lower the complete Q4 pack8 family **6.01%**
   with byte-identical leaves and exact generated state. Continue with a
   paired-output+SiLU boundary or bounded address/loop contraction; plain
   paired metadata decode is complete. Tile4 and projection-split ownership
   are closed. Any new ownership or reduction geometry must re-enter the full
   quality gate.
5. **LD-5 — replay and residual fusion.** Launch/submission work is bounded to
   roughly 2.18 ms/token. It follows attention and F16; launch-count reduction
   alone is not a promotion gate.

LD-1 must pass live **65/127/257/512**, global page
**255/256/511/512/1023/1024**, SWA ring **511/512/513**, explicit
eviction/wrap, and p128/p512/p1K/p4K depth screens before p512/d128
publication. Every candidate preserves logits, hidden states, K/V,
`KVLiveSpans`, reset and allocation lifecycle. Any changed F32 association also
runs the heldout category gate at KL at most 0.05 and top-1 at least 90%.

## 10. Do not chase without new evidence

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

## 11. Evidence map

| Question | Evidence |
| --- | --- |
| Can the real-input source-F16 Q8 direction pass the full-model gate? | [`...f16-q8-full-model-rejected.json`](../benchmarks/results/2026-07-28-gfx1151-laguna-f16-q8-full-model-rejected.json): no. Combined all-layer max KL is **0.497301**; stage-only, structural subsets, and one-sided weight/activation residual repairs all fail. Production remains exact. |
| Is compressed source-F16 decode worth a full-model gate on gfx1151? | [`...f16-q8-real-input-candidate.json`](../benchmarks/results/2026-07-28-gfx1151-laguna-f16-q8-real-input-candidate.json): yes directionally—actual-input modeled family time falls **31.236 -> 6.675 ms/token (-78.63%)**, with projection normalized RMSE at most **0.00974**. It is not yet a full-model quality or production claim. |
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
| Does hipEngine beat same-source llama.cpp HIP under a true matched natural-completion protocol? | [`...hipengine-vs-llamacpp-hip-matched-abba.json`](../benchmarks/results/2026-07-28-gfx1100-laguna-q2-xl-hipengine-vs-llamacpp-hip-matched-abba.json): yes, **64.094/63.431** versus **49.290/49.964 tok/s** at h16/h32 (**+30.034%/+26.954%**) over equal 144-run and 2,160/4,464-transition pools. This is protocol/storage/timing parity, not cross-engine arithmetic identity or a Vulkan claim. |
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
| What happened to the gated one-page composite? | [`design`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-gated-design.json), [`primitive`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-gated-correctness.json), [`runtime`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-gated-runtime-correctness.json), and [`rejection`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-gated-rejected.json): primitive only after category rejection. Full state and exact **12-composite/zero-gate/666-kernel** topology pass; both short orders improve family/kernel/span/child and aggregate h32 reaches **63.853 tok/s (+0.809%)** with every category decode row positive. Train aggregate TTFT still regresses **0.780%** beyond +0.5%, so runtime integration is removed without rerun and canonical **63.270 tok/s / 678 kernels** remains. |
| What is selected after wave-0 MoE-tail rejection? | [`design`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-head-wave0-tree-design.json), [`primitive`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-head-wave0-tree-correctness.json), [`runtime`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-head-wave0-tree-runtime-correctness.json), and [`rejection`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-head-wave0-tree-rejected.json): primitive only after clean rejection. Synthetic/CPU/**12/12** actual and full-state/**12 candidate + 36 retained SWA / 678-kernel** gates pass. Both short orders improve global-head work **28.728%/26.407%** and kernel sum **0.273%/0.013%**, but order A child regresses **0.859%** and order B span regresses **0.810%**. Runtime integration is removed before long contexts/categories; current-P4 and canonical **63.270 tok/s** remain. |
| What happened after global-head wave-0 rejection? | [`design`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-router-projection-wave0-tree-design.json), [`primitive`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-router-projection-wave0-tree-correctness.json), [`runtime`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-router-projection-wave0-tree-runtime-correctness.json), and [`rejection`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-router-projection-wave0-tree-rejected.json): primitive only after clean rejection. Synthetic/CPU/**47/47** actual, full-state, and exact **47-candidate/678-kernel** gates pass. Both short orders improve projection work **7.527%/6.307%**, but order A child regresses **0.619%** and order B kernel/span regress **0.612%/4.211%**. Runtime integration is removed before long contexts/categories; retained `bf16_hidden` and canonical **63.270 tok/s** remain. |
| What happened after router-projection ownership rejection? | [`design`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-control-publication-design.json), [`runtime`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-control-publication-runtime-correctness.json), and [`rejection`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-control-publication-rejected.json): rejected and removed. Ownership/reset and full state pass at KL0/top-1 100%; tracing proves five→three copies and **683→681 dispatches/token** with exact corresponding 678-kernel multisets. Both short orders regress model-kernel sum **0.315%/0.336%**; order B also fails span/child at **+0.700%/-0.706%**. Longer contexts/categories stop; sharing/borrowing integration is removed, while reset-position correctness and canonical **63.270 tok/s** remain. |
| What happened after shared-control rejection? | [`design`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-argmax-readback-design.json), [`runtime`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-argmax-readback-runtime-correctness.json), and [`rejection`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-argmax-readback-rejected.json): paired argmax readback is rejected and removed. RED/GREEN, KL0 full state, and exact **683 -> 682 = five -> four copies + 678 kernels** tracing pass; both short orders improve kernel sum, but order B span/child regress **1.184%/1.306%**. Longer contexts/categories stop; separate owners and canonical **63.270 tok/s** remain. |
| What happened after paired-readback rejection? | [`design`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-mapped-host-argmax-design.json), [`runtime`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-mapped-host-argmax-runtime-correctness.json), and [`rejection`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-mapped-host-argmax-rejected.json): mapped-host scalar argmax output is rejected and removed. Direct timing, RED/GREEN, KL0 full state, exact **683 -> 681 = five -> three H2D copies + 678 corresponding kernels / zero D2H**, and **-2 device allocations / -12 bytes / +4 KiB pinned host** all pass. Both short orders still fail: A regresses span/child **1.301%/0.523%**; B regresses kernel/span **0.021%/0.790%**. Longer contexts/categories stop; separate device owners/reads and canonical **63.270 tok/s** remain. |
| What happened after mapped-output rejection? | [`design`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-pinned-async-argmax-readback-design.json), [`runtime`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-pinned-async-argmax-readback-runtime-correctness.json), and [`rejection`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-pinned-async-argmax-readback-rejected.json): single-fence pinned async readback is rejected and removed. Direct timing improves **82.129 -> 51.353 us/token (-37.472%)**, full state is KL0/top-1 100%, and exact **683 = 3 H2D + 2 D2H + 678 identical kernels** plus two-async-copy/one-final-fence telemetry pass. Both short orders still fail: A regresses span/child **0.618%/0.503%**; B regresses kernel sum **0.126%**. Longer contexts/categories stop; the blocking fallback and canonical **63.270 tok/s** remain. |
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
Post-gated ranking excludes every rejected owner and selects only D9's internal
RMS synchronization: an exact wave-0 tree preserves both BF16 boundaries and
all FP32 association while contracting **9 -> 2 barriers**. The separately
registered primitive passes CPU, **47/47** actual-layer, codegen, and cache-only
symbol/resource gates; repeated endpoint event/wall improves **2.78-2.87%**.
The conservative modeled h32 is **63.314 tok/s**, not a performance claim. A
temporary c=1 owner passed exact full state and cache-only **47-candidate/
678-kernel** topology/resource gates, then failed both frozen short orders and
was removed before long contexts/categories. Retained D9 and the topline remain
unchanged. Post-wave0 re-ranking then selects only the 12 global current-P4
head+KV calls. The exact wave-0 primitive passes synthetic/CPU/all-12 actual,
codegen, and cache-only symbol gates; direct cached-C layers 0/44 improve
**20.57-22.55%**, while SWA regresses and is excluded. Its false/default-off
owner passed exact 16-transition full state and cache-only **12 candidate + 36
retained SWA / 678-kernel** topology/resource gates. Both short orders improve
global-head work and kernel sum, but order A fails the child-throughput guard
(**-0.859%**) and order B fails span (**+0.810%**). Runtime integration is
removed before long contexts/categories; current-P4 global and canonical
**63.270 tok/s** remain the defaults. Post-global-head re-ranking then selects
the 47-call router projection wave-0 primitive. It passes synthetic/CPU/**47/47**
actual, full-state, and exact **47-candidate/678-kernel** gates, but the frozen
short screen rejects runtime ownership: order A child throughput regresses
**0.619%**, while order B kernel sum and span regress **0.612%/4.211%**. The
capability/plan/session/CLI route is removed before longer contexts/categories;
retained `bf16_hidden` remains canonical and the exact primitive is diagnostic.
Post-router-projection re-ranking then moved outside model-kernel leaves to the
five synchronous runtime copies per token. The scratch-owned 16-byte
publication passed ownership/full-state and exact **683 -> 681 = five -> three
copies + 678 model kernels** admission, but failed the frozen short clean gate.
Both orders regress model-kernel sum **0.315%/0.336%**; order B also regresses
cycle span **0.700%** and child throughput **0.706%**. The direct **44.358
us/token** isolated host-contract saving does not transfer under complete clean
decode. Sharing/borrowing runtime integration is removed before longer contexts
or categories; separate publication and canonical **63.270 tok/s** remain.
Post-rejection re-ranking therefore selects only the independent two-read D2H
argmax seam. One exact 12-byte owner preserves the unchanged stage-2 ID/value
pointers at +0/+8 and all **15/15** executable tie fixtures bit-for-bit while
moving direct readback **45.02288 -> 22.76360 us/token (-49.440%)**. Its
false/default-off route passes RED/GREEN, KL0 full state, and exact **682 = four
copies + 678 kernels** tracing, but short order B regresses span/child
**1.184%/1.306%**. Runtime integration is removed before longer contexts or
categories; separate owners and canonical **63.270 tok/s** remain. The next
selected design eliminates, rather than merges, those reads: unchanged stage-2
raw pointers target HIP-registered mapped host memory. All **15/15** actual-vocab
tie fixtures and all timing repetitions pass at **83.311 -> 39.178 us/token
(-52.974%)**. Its false/default-off runtime owner passes RED/GREEN, fresh
shared-weight KL0/top-1-100% state, exact **-2 allocations / -12 device bytes / +4
KiB pinned host**, and one cache-only **681 = three H2D copies + 678 identical
kernels / zero D2H** trace. Complete short decode still rejects both process
orders: A regresses span/child **1.301%/0.523%**, while B regresses kernel/span
**0.021%/0.790%**. Runtime integration is removed before longer contexts or
categories; the general host-mapping ABI remains, and no default, rollup, or
canonical **63.270 tok/s** change is claimed.

Post-mapped rejection therefore selected the independent synchronization
schedule: preserve both separate device outputs and both D2H copies, enqueue
them into a registered non-mapped host page, and use one final fence instead of
a pre-fence plus two blocking reads. All **15/15** actual-vocab tie fixtures,
full state, and exact **683-dispatch / five-copy / 678-kernel** topology passed,
but the frozen short gate rejected both process orders: A regressed span/child
**0.618%/0.503%**, while B regressed model-kernel sum **0.126%**. Runtime
integration is removed and the blocking fallback remains canonical.

The final same-source HIP ABBA closes the comparison ambiguity rather than
reopening a rejected owner. Across equal 144-run and 2,160/4,464-transition
h16/h32 pools, hipEngine measures **64.094/63.431 tok/s** versus llama.cpp HIP
**49.290/49.964 tok/s**, a **30.034%/26.954%** lead. All protocol, native-timing,
source, device-bundle, correctness, and lifecycle gates pass. Cross-engine
arithmetic remains intentionally native, and the independently pinned Vulkan
objective remains open.
