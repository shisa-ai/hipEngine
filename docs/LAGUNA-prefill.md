# Laguna S 2.1 Prefill Attack Plan

Last updated: 2026-07-27

Status: active successor to the completed LPF/AR-O campaign in
[`LAGUNA.md`](LAGUNA.md). The prior bounded tasks are closed; this plan starts a
new arithmetic and data-layout campaign. It does not reactivate the rejected
expert-major F16 runtime routes. LAP-1 is complete with a direct resident-T16
MMQ32 consumer. The first LAP-2 three-plane/guarded/exact-repair primitives are
implemented and traced. The original one-scale-per-32 one-plane integrated
candidate crossed 350 tok/s but failed the complete category quality gate.
The repaired gate/up route uses one FP32 scale per 16 activations in the same
160-byte block and widens the Q4 consumer to 128 columns x 32 rows. Its original
shipping-relative category gate reached maximum KL 0.0407248, but LAP-Q0 found
that direct production-versus-all-exact reached 0.0535024. The admitted
quality schedule uses hipBLASLt heuristic 2 for the K3072xN72 SWA gate through
M128 and heuristic 4 elsewhere; the clean absolute gate passes at maximum KL
0.0495426 and 316/320 top-1. The row-qualified M512 schedule is clean
production. The compounded routes are gfx1151 package defaults. The exact
pair-decode wave-column D8 gate/up remap plus Q4-only D4 down remap first
reached **448.203 tok/s**; Q6 down retains its bit-identical row-vector stage.
A direct per-column Q4 gate/up decode, the corresponding Q4-down decode,
parallel stable compaction, and exact eight-token router-logit reuse are now
retained production. The direct attention-RMSNorm cast is also retained after
complete-state exactness and clean selector-unset publication. The subsequent
byte-neutral Q6 qmicro layout remains production. Exact cached-only qrow4
scheduling cuts traced attention **219.709 -> 176.580 ms (-19.63%)** and
improves clean selector-unset pp512 **505.084 -> 526.451 tok/s (+4.230%)**.
The 500 gate is closed. A subsequent exact cached-metadata policy is now
production after matched pp512 improves **533.507 -> 542.785 tok/s
(+1.739%, 7/7 wins)** with complete output/state exactness and clean
selector-unset publication reaches
**542.088 tok/s** median. Exact MMQ grouped-combine reuse then removes one
routed-output round trip and launch per sparse layer; clean selector-unset
publication reaches **543.807 tok/s** median. Exact selected-down scratch reuse
and an explicit-BF16-boundary dual-SiLU pack remove another launch and
intermediate per sparse layer; clean selector-unset publication reaches
**546.100 tok/s** median. The subsequent M2048 matrix policy does not claim a
pp512 win, but raises clean production 1K/4K to **506.299/410.099 tok/s**
while pp512 remains **545.015 tok/s** within run variance. Exact global qrow6
then raises clean selector-unset 512/1K/4K production to
**547.064/513.180/428.628 tok/s** and cuts traced attention to **152.406 ms**.
The campaign remains active toward the 700 stretch.
The latest exact Q6 padded-activation specialization publishes clean
selector-unset **551.459/517.307/432.099 tok/s** at 512/1K/4K. Its small
gain closes activation-stage padding as a useful local lever. The byte-neutral
Q4 qmicro subsequently passed exact decode but failed its actual-weight M512
prefill gate; all prefill candidate surfaces were removed. The next screen
tested 40/48-row direct-wave gate/up tiles; both were exact but slower and
were removed. A three-query single-wave attention point was also exact, but it
lost **3.22%** to qrow4 on the weighted mix and **7.31%** to the qualified
production policy; every candidate surface was removed. The active bounded
screen then swapped the Q4 gate/up grid axes to run routed-row tiles fastest.
That path was BF16-bit exact but regressed the natural-M512 leaf **0.18%** and
was removed. Axis order alone is therefore closed. The subsequent
source-F16 Q/K/V grouping screen is also closed: a row-major concatenated
contraction is F32-bit exact but models only **2.891 ms** pp512 saving before
the mandatory output restride, while hipBLASLt `GroupedGemm` exposes zero
algorithms for the full QKV problem on gfx1151 at both zero and 64-MiB
workspace. All candidate surfaces were removed. Dense-initial attention then
reached **559.290/523.090/439.044 tok/s**. The latest exact source-F16
boundary fusion now publishes clean selector-unset
**559.554/523.912/440.809 tok/s** and removes **96** pp512 dispatches while
preserving complete state exactly. Selected-down persistence and row64
screens are closed. Exact shared/routed MoE branch concurrency now supersedes
that packet. The exact after-router, least-priority shared schedule now
publishes **568.849/527.113/444.508 tok/s**. The automatic two-queue policy
protects router selection before releasing shared work and cuts pp512 kernel
span **898.024 -> 890.769 ms** versus priority-0 after-router overlap. Moving
shared work after gate/up is rejected. The next bounded screen holds priority
+1 constant and tests whether eager release can eliminate the remaining
**0.853-ms** secondary spill without reintroducing router contention. That
screen is now rejected at **-0.198%, 1/7 wins**. Scheduling is frozen at the
after-router, least-priority boundary. Two byte permutes now replace scalar
Q6 qmicro quartet unpack without changing resident bytes or arithmetic: the
actual leaf improves **2.67%**, tracing cuts the 115-call Q6 body **1.23%**,
and clean selector-unset planar-Q6 production reaches
**573.354/530.351/446.189 tok/s**. Cooperative Q4 row64 and byte-neutral Q4
qmicro direct-wave consumers are now both exact but decisively rejected.
Production remains unchanged after a hybrid Q4 metadata layout also fails:
one packed coefficient plane still raises VGPR 88 -> 120 and loses 3.74%.
Q6 selected-down integer WMMA then preserves every tested BF16 bit and improves
the actual layer-1 leaf **4.20%**. Hoisting its invariant activation fragments
adds another exact **1.136%** leaf win and publishes selector-unset
**577.396/545.366/459.716 tok/s** at 512/1K/4K. Dense-initial F32 hipBLASLt
attention then cuts traced pp512 attention **143.669 -> 82.763 ms (-42.39%)**
and publishes selector-unset **623.050/563.399/462.430 tok/s**. The complete
category gate remains max KL **0.049542582**, **316/320** top-1, and the
route-specific pp512 all-exact KL improves **0.003246 -> 0.002214**. The
successor packs query heads so one wide QK and one PV replace sixteen calls,
then assigns each causal-score row to one local32 wave without LDS/barriers.
Clean selector-unset production then reaches
**632.618/568.845/464.606 tok/s** at 512/1K/4K. Attention falls to
**69.983 ms** in the refreshed trace. The next exact Q6 body overlaps the
next planar-qmicro K32 global fetch with current integer-WMMA compute,
publishing **636.073 tok/s** at pp512 while 1K/4K remain flat within
**0.12%** at **568.765/464.061 tok/s**. Its 23-call pp512 window falls
**112.746 -> 101.963 ms (-9.564%)**. A second exact register pipeline carries
the next compact Q8 half-row during the same current-K32 compute. Clean
512/1K/4K reaches **639.114/569.880/464.280 tok/s** and the Q6 window falls
again to **100.367 ms**. Shape-qualified raw-nibble P8 prefetch then carries
only the next Q4 K32 payload at M512+ and publishes
**643.554/573.066/466.290 tok/s**. The same payload-only pipeline is now
admitted for the Q4 selected-down consumer at M512+: its traced 72-launch
window falls **217.416 -> 212.090 ms (-2.450%)**, and seven complete-state
pp512 pairs improve **639.574 -> 643.166 tok/s (+0.562%, 7/7 wins)**. Clean
selector-unset 512/1K/4K was **643.141/573.717/466.913 tok/s**. Qualifying the
K3072xN72 source-F16 schedule by rows now publishes
**645.803/575.942/468.311 tok/s**, improving **0.414%/0.388%/0.299%**.
Caller-stream physical-byte and overlap reductions remain the active campaign.
The execution order below was re-audited on
2026-07-26 after
correcting both the Vulkan comparator geometry and the absolute quality
baseline.

## Outcome

Close the resident c=1 Laguna S 2.1 Q4_K_M prefill gap on Radeon 8060S/gfx1151
without weakening hipEngine's quality, fallback, memory, or plugin contracts.
The primary external control is the current local llama.cpp Vulkan build at
`c0bc8591e8815c63cb01dd3f051a8b0df02501c9`, which measures
**344.56 +/- 3.16 tok/s** at pp512. The pre-campaign hipEngine
matrix512/attention128 default measured **76.226 tok/s**, a **4.520x** gap.
The quality-admitted production default now measures **645.803 tok/s**
selector-unset, **8.472x** the old row and **87.428%** above the Vulkan
control.

That Vulkan row is now a compatibility floor, not the optimization ceiling.
Strix Halo has a **256 GB/s** theoretical LPDDR5X roof and the existing
local/reference large-read evidence is about **221 GB/s**. The first exact
active-byte lower bound put pre-campaign selected gate/up at only **9.85 GB/s**,
the Vulkan family at about **56.1 GB/s**, and the direct-T16 leaf extrapolation
at about **80.3 GB/s**. Parity therefore still leaves most of the measured
memory roof unused. Before any final throughput target is called complete, this
campaign must rerun a same-host cold-stream read with locked/recorded clock
policy, publish encoded and physical bytes for every family, and report GB/s
plus percent of achievable bandwidth. The interim streaming-family target is
**at least 70% of the measured same-host read ceiling** (about **155 GB/s** if
the 221 GB/s anchor reproduces).

The first design is:

1. source-arithmetic Q4_K/Q6_K packed integer-dot MMQ over natural
   expert-major rows, with geometry calibrated to the shader that actually
   runs on gfx1151;
2. one activation quantization per producer row, before top-10 expansion where
   possible;
3. residual Q8_1 planes plus conservative BF16-boundary detection;
4. sparse exact recomputation with a bounded, fail-closed queue;
5. the existing exact `gguf_q4_k_t16_v1` expert layout as the sole resident
   set, with a direct T16 packed-dot consumer rather than a per-dispatch
   transpose;
6. exact fallbacks selected by quant, projection role, and measured shape—not
   prompt, token, or hand-picked layer ID.

The original `LAP-*` sequence is now substantially complete. Exact
register-resident wave-column consumption is promoted for D8 gate/up, while
row-vector staging remains promoted for Q4/Q6 down. The active post-350 queue
transfers the wave-column premise to down, then returns to the production
bandwidth ledger and further counter-directed expert work. The synchronous-LDS
key-parallel attention premise completed a negative gate, but an exact
cache-ordering schedule subsequently removed **43.129 ms** from the attention
family. Submission and graph work remain deferred because the current trace
leaves only 1.53% of traced wall outside summed kernels.

The first 2026-07-25 layout checkpoint changed item 5. X8 remains the fastest
proven MMQ32 input and an important arithmetic control, but its optimized
exact fallback is **1.11093x** retained T16 at c=1 and **1.02987x** at c=2 on
the actual layer-1 gate/up pair. It catches T16 at c=4/c=8 and is BF16-bit
exact, but the campaign target is c=1 and the <=2% decode gate is mandatory.
The prior “X8 wins” resident decision is therefore reversed: do not add a
complete T16 sidecar to X8, and do not integrate X8 into the runtime. The
direct T16 consumer now passes the frozen leaf gate at
**2.502x/3.959x/5.502x** retained on M128/M256/M512 and within
**4.66%/4.05%/3.02%** of X8. Its guarded repair primitives are also
implemented. Admission, gfx1151 default promotion, and clean production
publication are now complete. The absolute-bandwidth/KL audit remains post-350
roofline work.

This document uses stable `LAP-*` labels (“Laguna arithmetic prefill”). Numeric
task-tracker IDs may be assigned separately; the labels deliberately do not
reuse historical task numbers.

## Scope

The campaign target is frozen to:

| Item | Contract |
| --- | --- |
| Model | `/models/gguf/laguna-s-2.1-Q4_K_M.gguf`, SHA-256 `7da520c5f44bc3c79d4eeebfd1151ba7114c5d7568e72a995638417093c5753f` |
| Backend | `hip_gfx1151`, Radeon 8060S / Ryzen AI MAX+ 395 |
| Runtime | torch-free, one resident weight set, c=1 model math |
| Storage | existing Q4_K/Q6_K/F16/F32 model tensors and BF16 KV |
| Headline | 512-row resident prefill with matrix chunk 512 and the retained attention policy |
| Shape coverage | canonical prompt rows plus 128/256/511/512/513/1K/4K milestone screens |
| Quality | repository primitive gate plus the complete ten-prompt, four-category train/heldout lane |
| Non-regression | h16/h32 decode within 2%, category E2E gates, lifecycle, and bounded memory |

Model load remains outside prefill timing. DFlash, c>1 throughput, loader speed,
sampling, and decode optimization are not campaign credits. They are rerun only
when shared runtime behavior changes.

## Baseline and the bridge to Vulkan

### Retained hipEngine state

The completed campaign moved repeated 512-row prefill from the pre-matrix
**47.395 tok/s** row to **76.226 tok/s**. The retained 512/1K/4K screen is
**76.226/74.538/70.885 tok/s**; the canonical category-weighted short-prompt
result is **69.761 tok/s**. The main retained changes were:

- 512-row matrix scratch and independent 128-row attention chunks;
- exact device-resident expert grouping and adaptive grouped-small-M down;
- compensated F16 WMMA on the 36 SWA layers;
- online global and SWA attention with exact fallbacks;
- exact chunk, cursor, KV, lifecycle, and complete-category gates.

The latest cleanup commit `e4ab85d59` removed only rejected expert-major runtime
experiments. It did not change the shipping path.

At 76.226 tok/s, pp512 is **6.7169 seconds**. The Vulkan control is
**1.4860 seconds**, so parity requires removing about **5.2309 seconds**, or
77.9% of current wall.

### Family bridge budget

LAP-0 replaced the pre-campaign inference with a clean cached trace at unchanged
shipping defaults. The single profiled 512-row pass measures **76.381 tok/s**,
**6.703260 seconds** synchronized wall, **6.689356 seconds** kernel sum, and
**6.699478 seconds** kernel span. This does not replace the repeated retained
**76.226 tok/s** headline; it is the internally consistent attribution row used
for the bridge.

| Cumulative modeled step | Modeled pp512 wall | Modeled tok/s | Evidence used |
| --- | ---: | ---: | --- |
| Current shipping trace | 6.7033 s | 76.381 | fresh matrix512/attention128 pass |
| Apply measured LAP-1 direct-T16 leaf ratio | 3.6933 s | 138.6 | scale gate/up by 9.5966 / 52.7988 ms; not integrated |
| Match Vulkan selected Q4 gate/up | 3.6707 s | 139.5 | save 3.6786 - 0.6461 s |
| Then match Vulkan selected Q4/Q6 down | 2.9372 s | 174.3 | save 1.1001 - 0.3665 s |
| Then match Vulkan dense/shared quant | 2.3586 s | 217.1 | save 0.6415 - 0.0629 s |
| Then match Vulkan source-F16 | 1.7450 s | 293.4 | save 0.8941 - 0.2805 s |
| Then match measured current attention to Vulkan | 1.5064 s | 339.9 | save 0.2779 - 0.0393 s |
| llama.cpp Vulkan control | 1.4860 s | 344.56 | user unprofiled pp512 |

This is an Amdahl model, not a performance claim. It assumes independent family
savings across different numerical/runtime contracts. The five mapped kernel
gaps explain **99.740%** of the fresh hipEngine-minus-Vulkan kernel-sum gap and
leave **20.4 ms** between the modeled hipEngine wall and the user Vulkan wall.
A new runtime, graphs, Python removal, or a different benchmark definition is
not required to explain the 4.5x gap.

The table is useful for attribution but is no longer the completion target. Its
comparator is itself far below the memory roof. At M512 the routing capture
touches **10,237 / 12,032 = 85.08%** of all layer/expert groups. Multiplying
that fraction by the raw **905,969,664-byte** gate/up pair and 47 sparse layers
gives a **36.228 GB encoded-weight lower bound** for the selected gate/up
family:

| Selected gate/up path | Family wall | Encoded-weight-equivalent GB/s | % of 221 GB/s read anchor |
| --- | ---: | ---: | ---: |
| Current shipping trace | 3.6786 s | 9.85 | 4.46% |
| llama.cpp Vulkan | 0.6461 s | 56.1 | 25.4% |
| LAP-1 direct-T16 leaf, `47 x 9.5966 ms` | 0.4510 s | 80.3 | 36.3% |
| Interim bandwidth target | 0.2337 s | 155 | 70.1% |

These are source-encoded lower-bound rates, not memory-controller counters:
T16 physically reads 2.778% more bytes, padding/reloads can add traffic, and
other tensors overlap the family window. LAP-BW0 therefore must publish both
encoded-equivalent and measured/counter-derived traffic. The full 62–68 GB
whole-pass traffic estimate from review is plausible but is not admitted until
the per-family byte ledger is computed directly from the manifest and routing
capture.

The new LAP-1 row is also modeled, not a full-model claim. Applying its clean
actual-layer direct-T16 M512 ratio
(**9.5966 / 52.7988 = 0.18176**) to the measured **3.6786-second** gate/up
family gives **0.6686 seconds**, within 3.5% of Vulkan's **0.6461-second**
family. This says the sole-resident body can close the first mapped gap; repair
and runtime integration must now prove that the ratio transfers across all 47
sparse layers. X8 remains the measured body ceiling, not the selected resident
layout.

There is an unresolved bridge inconsistency: the family trace averages
**78.27 ms/layer**, whereas the retained layer-1 leaf is **52.80 ms**.
Likewise, ratio scaling predicts **0.6686 s**, but summing the direct-T16
layer-1 leaf across 47 layers predicts **0.4510 s**. Layer/routing variation,
kernel-family attribution, and one-layer representativeness must be reconciled
with an all-layer candidate trace before either projection gates LAP-3.

At 512 rows, selected Q4 gate/up is **3.6786 seconds / 54.99%**, selected
Q4/Q6 down **1.1001 seconds / 16.45%**, source-F16 **0.8941 seconds / 13.37%**,
dense/shared quant **0.6415 seconds / 9.59%**, and measured global+SWA
attention **0.2779 seconds / 4.16%**. The respective hipEngine/Vulkan ratios
are **5.694x/3.001x/3.188x/10.198x/7.075x**. Named non-`other` families cover
**99.653%** of kernel time, while span-minus-sum is **0.151%**. Gate/up remains
the largest family, but opportunity/risk order now moves the already-measured
source-F16 and dense/shared routes ahead of selected-family promotion.

The source-F16 library ceiling is materially stronger than the old LAP-6
checkpoint. At M512, `12 x 2.583908 + 36 x 2.981794 = 138.351 ms` for the
measured inclusive hipBLASLt full/SWA families, versus **894.070 ms** shipping
and **280.5 ms** Vulkan. That is a potential **755.719 ms** reduction and about
**2.03x** faster than the comparator family. It is still a ceiling, not a
runtime result: timing buffers were zero-filled, the inclusive path includes a
BF16→F32→FP16 activation cast, and real-input range/quality are unproven.

### LAP-0 cumulative quality and shape evidence

The all-exact versus shipping-control category run passes, but the remaining
approximate budget is narrow. Shipping improves weighted prefill **53.596 ->
70.546 tok/s (1.31627x)** and h16/h32 E2E **1.18198x/1.12459x**, while decode
is neutral. Across 320 teacher-forced steps it reaches maximum KL
**0.0459275**, **319/320 (99.6875%)** top-1, and at least **98.4375%** top-1
in every category. Only **0.0040725** remains below the 0.05 KL ceiling; the
`mixed_ja_en_translate` trajectory is the only non-exact free-running pair.
New approximate paths therefore compare directly with all-exact, and repaired
BF16 equality is strongly preferred.

The **0.0459275** debt is not yet attributed to individual admitted
approximations. Before another approximate runtime path is promoted, run
one-factor ablations for compensated source-F16, global online attention, and
SWA online attention against the all-exact lane. This decides whether
hipBLASLt/FP32 accumulation buys back enough KL headroom for simpler expert
arithmetic. It also prevents spending LAP-2 effort to solve a budget constraint
whose primary consumer can be removed faster.

Natural routing showed that literal 32-row padded arithmetic was not viable.
At M512, padding factors are **1.0219/1.0684/1.1650/1.3801/1.8662x** for
2/4/8/16/32-row tiles; M256 reaches **2.9295x** at tile32. LAP-1 now keeps the
32x32 shared tile but bypasses dot accumulation for padded routes. That makes
all seven natural shapes positive without a second small-row kernel; geometry
and weight loads remain padded and are revisited only if the integrated trace
shows a material ceiling.

Two repeated BF16 activation captures at depths 2/11/20/30/39/48 are
bit-identical for M32/55/64/122/128/256/512 without persisting raw activations.
Late-layer residuals contain sparse extreme outliers: at depth 48/M512,
absolute p99 is
**16.25**, p99.9 **127,488**, and maximum **950,272**, while row-RMS p95 is
only **7.67**. These are post-layer proxies rather than exact projection
inputs, but they already reject a single global or row-wide scale as the LAP-2
premise. Exact projection-input calibration remains required before selecting
residual planes.

LAP-0 used `performance_level=auto` and recorded only a post-run idle sample
(`622 MHz` gfx, `1000 MHz` memory). That is insufficient for close cross-backend
or roofline claims: a 6.7-second HIP run and a 1.49-second Vulkan run can have
different power/thermal trajectories on the shared-memory APU. LAP-BW0 and all
new external/roofline rows must pin the supported performance policy when
possible and record in-kernel or sampled load clocks; otherwise the result is
explicitly qualified as clock-unbounded.

The supported `high` policy is not a production speed lever. A root-applied
`auto -> high -> auto` screen under the complete 512/1K/4K production protocol
measures pp512 **560.898 -> 557.949 -> 560.759 tok/s**. High is **0.514%**
below the median of the surrounding auto runs, lowers 1K, and is inconsistent
at 4K. The original `auto` state was restored. This closes `high` as a
deployment default; manual clock locking remains useful only to tighten future
roofline experiments. Evidence:
[`2026-07-26-gfx1151-laguna-clock-high-rejected.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-clock-high-rejected.json).

The compact LAP-0 evidence packet is
[`2026-07-24-gfx1151-laguna-prefill-lap0-control.json`](../benchmarks/results/2026-07-24-gfx1151-laguna-prefill-lap0-control.json).

The rejected expert-major F16 diagnostic independently supports the same
conclusion. At M512 it reached **176.001 tok/s** versus **76.395 tok/s** for the
retained route. Subtracting the unchanged non-expert wall implies an expert
sub-window near the Vulkan expert budget. That inference needs a direct trace,
but it shows that expert-major reuse—not a theoretical hardware limit—is the
missing performance mechanism.

## What the latest Vulkan implementation is doing

The read-only checkout `/home/lhl/llama.cpp/llama.cpp-vulkan` is clean at
`c0bc8591e8815c63cb01dd3f051a8b0df02501c9`, build 10107. This is the same
revision as the retained pp512 profile; its current HEAD contains no newer
Laguna-specific backend change. The latest Laguna model-support commit in that
history is `1f66c3ce1`. This identity and history were rechecked on 2026-07-25;
the MMQ and attention mechanisms audited below are still the current backend
implementation. The relevant source is:

- `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
- `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp`
- `ggml/src/ggml-vulkan/vulkan-shaders/mul_mmq.comp`
- `ggml/src/ggml-vulkan/vulkan-shaders/mul_mm_id_funcs.glsl`
- `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm1.comp`
- `src/models/laguna.cpp`

The pp512 operation ledger is:

| Vulkan family | Time | Share |
| --- | ---: | ---: |
| Selected Q4 gate/up | 0.6461 s | 43.69% |
| Selected Q4/Q6 down | 0.3665 s | 24.78% |
| Source-F16 projection | 0.2805 s | 18.97% |
| Dense/shared quant projection | 0.0629 s | 4.25% |
| Flash attention | 0.0393 s | 2.66% |
| Router/norm/RoPE/activation/miscellaneous | 0.0836 s | 5.65% |

The mechanisms matter more than the API:

1. Contiguous F32 activation rows are converted once to Q8_1 and cached in a
   reusable preallocated buffer.
2. A device pass counts routes per expert. Subgroup ballots compact matching
   `(token, route-slot)` row IDs into natural expert-major tiles.
3. Q4_K/Q6_K `MUL_MAT_ID` uses packed integer dot, not cooperative matrix.
   On this RADV device the actual Q4_K pp512 comparator is the **medium**
   `matmul_id_subgroup_q4_k_q8_1` pipeline, not the small pipeline. The backend
   disables large routed matmul on AMD; with `m=1024`, `n=512`, neither
   dimension satisfies the small `<=32` branch, so `!mm_l` selects medium.
4. The medium K-quant specialization is local128 over two wave64 subgroups with
   **BM=64 output columns, BN=64 routed rows, BK=32, WMITER=1, TM=2, TN=2**,
   32 FP32 accumulators per lane, and about 3.9 KiB LDS. Routed MMQ forces
   `BK_STEP=1`; the non-ID dense shader defaults to `BK_STEP=4`.
5. Flash attention owns 16 query rows by 64 key rows in a 256-thread
   cooperative-matrix block, performs both QK and PV, and maintains online
   softmax state. The graph gives each layer all 512 query rows instead of four
   128-row launches.
6. Graph-pattern fusions remove several tails, but they are secondary for
   hipEngine: pp512 kernel span exceeds kernel sum by only **0.144%**.

The Vulkan subgroup is 64 and its attention KV is F16, while hipEngine uses
wave32 kernels, BF16 KV, `KVLiveSpans`, and stricter quality gates. The plan
therefore transfers the tiling/dataflow, not literal shader constants or
unchecked numerical policy.

This correction changed the MMQ target. hipEngine's first local128 body is
**32 columns x 32 rows over four wave32s**, with `TM=1`, `WNITER=8`, and eight
accumulators per lane. Per K32 interval it performs about 64 packed dots per
lane between the same two workgroup barriers; the running Vulkan medium shader
performs about 256. The first body remains a valid and fast LAP-1 leaf, but it
is not source-faithful geometry. Direct 128x64, 64x64, 256x32, and coalesced
raw-nibble screens are now rejected below. Simple rectangular and staging
changes to this body are closed; revisit expert scheduling only with hybrid
large-expert or counter evidence that isolates a new limiter.

hipEngine also retains two structural advantages the comparator lacks:
device-resident expert compaction launches only populated tiles, and the dual
gate/up body can reuse one activation tile for both projections. Vulkan
dispatches expert/row tiles broadly, scans the route-ID matrix inside surviving
workgroups, and issues gate and up as separate `MUL_MAT_ID` operations.
Matching its per-tile efficiency should therefore beat, not merely tie, its
family wall.

## What prior hipEngine work proved

### Laguna-specific results

| Experiment | Result | Meaning for this plan |
| --- | --- | --- |
| Direct Q8_1/dp4a selected Q4 gate/up | +4.070% category prefill, but max KL 0.171561 | Quantize-before-expansion is viable; one-plane Q8_1 arithmetic is not promotable. |
| Exact scalar grouped gate/up C4/C8/C16 | Production M55 best candidate still lost to direct | Do not retry scalar row reuse under a new tile name. |
| Diagnostic raw-Q4 DS4 WMMA32 | About 1.41x faster than selected WMMA in a synthetic shape | Integer arithmetic has potential, but this was independent-wave global loading, not Vulkan's tiled MMQ. |
| Diagnostic resident-T16 DS4 WMMA32 | About 1.48x synthetic speedup | T16 can feed a fast prototype, but the body/layout and quality contract were incomplete. |
| Expanded-Q4 LDS staging | 2.22x slower than raw WMMA32 | Staging without enough tile reuse is negative. |
| Packed-Q4 LDS staging | Recovered some loss but remained 38% slower than raw WMMA32 | Do not repeat per-block pack/sync without a complete shared-tile schedule. |
| WMMA64 widening | Only about 0.63% over WMMA32 | More independent waves are not the missing architecture. |
| Pre-unpacked Q4 preview | 1.46x slower than raw DS4 WMMA32 | Metadata decode alone is not the bottleneck. |
| Expert-major compensated F16 WMMA | 176.001 tok/s at M512; full suite max KL 0.527791 | Natural-row matrix reuse is fast enough; arithmetic accumulation is the blocker. |
| Gate/up-only / down-only F16 bisection | KL 0.988050 / 1.183662 | Neither projection can be admitted alone; combined error partly cancels. |
| Global-only / SWA-only F16 bisection | KL 0.628301 / 1.205779 | No architecture-defined layer scope is safe; arbitrary layer tuning is forbidden. |
| Byte-neutral X8 MMQ32 with live-row skip | **1.197/1.567/1.704/2.526/2.587/4.092/5.614x** retained at M32/55/64/122/128/256/512 | The packed-dot body and natural-shape schedule pass. X8 remains the prefill ceiling/control, not the resident winner. |
| Exact X8 decode, direct/staged/transformed | Direct X8 is **4.693x** T16; raw LDS staging is **2.081x**; the optimized transform is exact but clean c1/c2 is **1.11093x/1.02987x** T16 | Per-dispatch layout recovery cannot meet the <=2% c=1 decode gate. Keep T16 resident and add a direct T16 MMQ address specialization. |
| Direct resident-T16 MMQ32 | **1.174/1.528/1.662/2.464/2.502/3.959/5.502x** retained at M32/55/64/122/128/256/512 | LAP-1 passes: T16 matches X8 BF16 bits, stays within 4.66%/4.05%/3.02% at primary shapes, and needs no transpose or sidecar. |
| Guarded T16 D4x3 primitive | Projection relative L2 **0.002922 -> 0.001826** on the finite CPU fixture; all-queued and forced-overflow correction are BF16-bit exact; dirty actual leaf is **1.289x/2.510x** retained at M128/M512 | LAP-2 arithmetic/repair foundation is implemented (`d9bb6ad88`); real-input threshold, repair rate, and runtime quality remain open. |
| hipBLASLt inclusive source-F16 ceiling | M512 weighted 12-full/36-SWA family is **138.351 ms** vs **894.070 ms** shipping and **280.5 ms** Vulkan | Move the real-input library route ahead of selected-family promotion; zero-filled timing and BF16→FP16 range remain explicit blockers. |
| Dense/shared quant family | **0.6415 s** shipping vs **0.0629 s** Vulkan; principal Q4 kernel is one wave32 | Low-risk direct reuse target with no routing or new quality surface; execute before selected down. |

The older scalar and independent-WMMA variants in
`hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_q8_1_selected_prefill.{hip,py}`
remain negative controls. The shared-tile X8 MMQ32 symbol remains the
arithmetic/performance control. The direct T16-native sibling is now the
integrated production primitive; the scalar/WMMA controls have no production
route.

### Transfer from the successful Qwen3.x campaign

Qwen3.6-35B-A3B UD-Q3_K_M on gfx1100 progressed through:

| Retained step | 512 / mixed-4K tok/s | Transferable lesson |
| --- | ---: | --- |
| Exact fully bulk | 218.598 / 211.936 | Fix execution granularity first. |
| Exact dense-Q8 pack8 | 364.414 / 342.902 | Use output reuse before changed arithmetic. |
| Exact Q8 row reuse | 573.288 / 523.321 | Keep one encoded weight row across prompt rows. |
| Exact IQ4 one-wave K512 | 693.325 / 613.576 | Specialize real underfilled shapes. |
| Exact Q8 16x4 | 707.420 / 626.077 | Admit tile shapes only at measured crossovers. |
| Exact GDN LDS32 | 763.221 / 670.417 | Preserve ordered arithmetic while changing ownership. |
| Exact IQ3 rowbatch4 + GQA attention | 774.185 / 741.180 | Batch independent rows and reprofile after every promotion. |
| Guarded residual-D4x3 MMQ | 848.543 / 831.393 | Changed algebra can ship when uncertain BF16 outputs are repaired exactly. |

The guarded Qwen route is the closest internal precedent. It runs a
source-faithful raw-Q8_0 x residual-D4 Q8_1 128x128 K256 MMQ, queues outputs
within `1e-5` of a BF16 rounding boundary, and recomputes those outputs with the
exact reduction. Its 18-workload by 9-position continuation gate is
logit-bit-exact. The policy admits only two measured winning shapes; every
other shape retains exact 16x4/8x4/8x2/pack8 fallbacks.

Laguna must not copy Qwen's threshold or geometry blindly. Q4_K/Q6_K metadata,
K3072/K1024 shapes, top-10 expansion, nonlinear gate/up boundary, and gfx1151
are different. The transferable method is residual reconstruction,
BF16-boundary risk detection, bounded exact repair, and shape-scoped admission.

## Proposed production dataflow

The target gate/up flow is:

```text
BF16 hidden [M, 3072]
  -> same-byte Q8 pack once per token row, one FP32 scale per 16 values
  -> existing device route count/prefix/compact metadata
  -> resident-T16 packed-dot Q4_K MMQ, 128 columns x 32 routed rows
  -> BF16 candidate gate/up
  -> existing exact SiLU/product boundary
```

The down flow starts after the exact SiLU/product boundary:

```text
compact BF16 expert intermediates [M * top_k, 1024]
  -> range-safe D4 Q8 pack once per compact route row
  -> resident-T16 packed-dot Q4_K/Q6_K MMQ to 3072 outputs
  -> BF16 candidate down
  -> existing ordered route-weighted combine/shared/residual chain
```

Important ownership rules:

- Gate/up quantization is over the original `M` producer rows. The compact
  metadata maps routed lanes back to those Q8 rows; it must not quantize or
  store the same input ten times.
- Down input is route-specific after SiLU, so it is packed over compact rows.
- Gate/up, SiLU, down, weighted combine, and residual remain separable
  registered primitives until each fused boundary is independently proven.
- Queue count, indices, thresholds, overflow state, and exact correction stay
  on device. No scalar D2H scheduling boundary is admitted.
- A queue overflow executes the complete exact projection or fails the
  candidate closed. It never truncates repairs.

### Activation metadata range is a correctness gate

The original DS4 block stored both activation scale and raw 32-value sum in
FP16. That was not safe to assume for every Laguna projection role:

- a block sum can overflow FP16 once a 32-element block's magnitude is roughly
  above 2,048;
- gate/up inputs are post-RMSNorm and are probably the safer case, but this must
  be measured rather than assumed;
- down inputs are `SiLU(gate) * up` with no normalization immediately before
  packing and are the primary overflow/range exposure;
- late-layer massive-activation rows also contain quiet blocks whose DS values
  can become FP16 subnormals with little effective mantissa.

The integrated path stores metadata as FP32 in the existing 160-byte
activation block, eliminating FP16 overflow/subnormal exposure. The first
one-plane D4 candidate nevertheless failed the complete 320-step quality gate:
maximum KL was **0.0767056** with **318/320** top-1, and both failing prompts
were in `mixed_ja_en`. This separates quantization granularity from metadata
range; FP32 storage alone was not enough.

The repaired gate/up pack uses eight FP32 scales plus 128 int8 values per
128-element block—still **160 bytes**—so each 16-value half-block has its own
scale. The Q4 consumer reconstructs the two signed quant sums used by the
min-term and applies the corresponding scale without a side buffer. The down
projection remains the faster D4 route. Across the complete 320-step
teacher-forced diagnostic, D8-gate/D4-down reaches maximum KL
**0.040724836**, **317/320 (99.0625%)** top-1, and at least **96.875%**
category top-1. The canonical clean category gate, not this diagnostic, is the
promotion authority. The three-plane repair primitive remains a retained
fallback/research control, but it is no longer on the immediate production
path.

## Resident weight-layout decision

Weight layout is a system decision, not a prefill-only microbenchmark result.
LAP-1 compared raw source blocks, byte-neutral X8, and the current T16
replacement under a strict one-resident-set contract:

- no persistent raw-plus-replacement or X8-plus-T16 expert family;
- temporary one-layer comparison buffers are allowed only for a leaf screen;
- the sole resident representation must preserve exact decode within 2%;
- every sidecar must publish family bytes, total peak, scratch, and context
  capacity before it can be considered;
- layout remains a quant-plugin concern, with no backend/quant branch in model
  or generic runtime code.

The prefill-only screen initially selected X8. It preserves all 144 bytes of
each source Q4_K block in
`[expert,out_pack8,k_block,col_in_pack8]`, occupies **905,969,664 bytes** for
the layer-1 gate/up pair, and improves raw MMQ32 by **9.82–12.14%**. The
live-row schedule then makes all frozen natural shapes positive.

The exact-decode screen reverses that system decision:

| Layout | Gate/up pair bytes | Actual exact selected decode | Decision |
| --- | ---: | --- | --- |
| Current T16 | 931,135,488 | c1/c2/c4/c8 **0.157223/0.351996/0.687016/1.350421 ms** | Sole resident baseline; exact decode already qualified |
| Byte-neutral X8 | 905,969,664 | **0.174663/0.362511/0.686471/1.332379 ms**, zero BF16 mismatches | Reject as sole c=1 layout: **1.11093x** T16 at c1 and **1.02987x** at c2 |
| Raw source rows | 905,969,664 | Existing exact/raw controls; slower MMQ32 than X8 | Diagnostic only |

The final X8 kernel is not a naive scalar fallback. It processes 16 gate and
16 up columns per local128 block, transposes Q4 nibbles and expands metadata
once per K256 interval into T16-shaped LDS, then uses the exact T16 arithmetic
and reduction order. Direct X8, raw LDS staging, and this complete transform
measure roughly **4.69x**, **2.08x**, and **1.11x** T16 at c1. The remaining
tax is layout recovery itself. Adding a full T16 sidecar would erase X8's only
resident-memory advantage and violate the one-set premise.

The selected production premise is therefore **T16 resident, T16-native
MMQ32**. T16 is only **25,165,824 bytes (2.778%)** larger than X8 for the
actual gate/up pair, is already the shipping allocation, and adds zero bytes
relative to the current runtime. The direct consumer reads T16's expanded
`d/dmin/scale/min` and interleaved Q4 payload while building the same 20-byte
per-column MMQ cache used by the proven raw/X8 body; it never transposes T16
back to raw/X8 in LDS.

X8 remains a frozen upper-bound control. Clean direct T16 is positive at every
natural shape, reaches **2.502x/3.959x/5.502x** retained on
M128/M256/M512, and is within **4.66%/4.05%/3.02%** of X8. The leaf decision
therefore passes. No new materializer is needed; LAP-2 repair and LAP-3
integration must preserve the existing one-set T16 residency and exact decode.

That decision unblocks current work but does not prove expanded metadata is the
best permanent streaming layout. The 2.778% is paid on every bandwidth-bound
pass. One bounded replacement screen was therefore run:

- **T16-lite:** keep T16's 16-column Q4 nibble interleave and FP16 `d/dmin`, but
  retain the source-packed 6-bit scale/min field. Per 16 columns/K256 this is
  `2048 + 32 + 32 + 192 = 2304 bytes`, byte-neutral with raw/X8 instead of
  T16's 2,368 bytes.
- **X16:** the cheaper control, grouping 16 source blocks without expanded
  scale/min metadata.

T16-lite is now **closed**. The final byte-plane-major layout is exactly
2,304 bytes and its best consumer expands the 192 packed metadata bytes once
per K256 tile into 512 bytes of LDS. It is BF16-bit exact at c1/c2/c4/c8, but
regresses current T16 by **17.63%/12.66%/11.95%/11.22%**
(T16 **0.161214/0.351595/0.688972/1.351014 ms** versus T16-lite
**0.189635/0.396094/0.771278/1.502643 ms**). Earlier direct packed-decode
controls were roughly 3x T16, so the optimized result is the relevant bound.
The layout fails its exact-decode prerequisite and does not receive an MMQ,
materializer, or runtime route. The host byte-neutral roundtrip oracle remains
for any genuinely different microtile premise. Evidence:
`benchmarks/results/2026-07-26-gfx1151-laguna-q4-k-t16-lite-decode-rejected.json`.

X16 is also now **closed**. Its one-pack exact consumer beats X8 at every
screened shape and reaches parity/wins at c2/c4/c8, but natural c1 remains
**7.654% slower** than resident T16: T16/X16 is
**0.163258/0.175753 ms** at c1, **0.352933/0.359698 ms** at c2,
**0.691072/0.683010 ms** at c4, and **1.368045/1.329822 ms** at c8. It
therefore fails before prefill, materialization, or runtime integration. The
temporary decoder is removed; the byte-neutral host roundtrip oracle remains.
Evidence:
`benchmarks/results/2026-07-26-gfx1151-laguna-q4-k-x16-decode-rejected.json`.

The stronger byte-neutral premise **passes exact decode**. It keeps the proven
T16-local nibble payload but replaces expanded scale/min bytes with exact
**four-column, three-byte** 6-bit metadata records. All 128 work items expand
the gate/up records cooperatively. Balanced c1/c2/c4/c8 timing improves T16
**4.929%/0.781%/3.691%/4.633%**, with zero BF16 mismatches, no sidecar, and
**25,165,824 fewer bytes** for the actual layer-1 gate/up pair. The exact
decoder runs at local128/VGPR192/SGPR128/LDS1536B/scratch0. It is retained as
a primitive; materialization and runtime were held unchanged pending the
actual-weight natural-M512 selected-prefill result below. Decode evidence:
`benchmarks/results/2026-07-26-gfx1151-laguna-q4-k-qmicro-exact-decode-retained.json`.

That selected-prefill gate is now **closed and rejected**. Three BF16-bit-exact
MMQ32 consumers were measured on the actual layer-1 K3072/N1024 gate/up pair
with natural M512 routing:

| Q4 qmicro metadata consumer | Resident T16 | Qmicro | Delta |
| --- | ---: | ---: | ---: |
| Direct per-column packed record | **9.402044 ms** | **9.570781 ms** | **+1.795%** |
| Wave-broadcast packed record | **9.385769 ms** | **10.281055 ms** | **+9.539%** |
| Quartet-owned LDS `dm` writer | **9.411384 ms** | **9.934902 ms** | **+5.563%** |

The direct body is the relevant bound: the 2.778% byte reduction does not pay
for packed scale/min extraction in the MMQ inner loop. Wave shuffles and
concentrating four FP16 scale products on one lane make it worse. The Q4
qmicro MMQ body, wrappers, fixtures, and benchmark mode are removed; only the
host byte-lossless oracle and already-retained exact decode primitive remain.
No materializer, quant key, resident allocation, or runtime route was added,
and production remains **551.459 tok/s**. Evidence:
`benchmarks/results/2026-07-26-gfx1151-laguna-q4-k-qmicro-prefill-rejected.json`.

## Quality strategy

### Three comparison lanes

LAP-0 establishes three explicit resident modes:

| Lane | Purpose |
| --- | --- |
| All-exact oracle | Exact tiled source-F16, exact global/SWA attention, exact grouped experts, and existing exact dense/shared paths |
| Shipping control | Current gfx1151 defaults, including admitted compensated/online paths |
| Candidate | Shipping control plus exactly one new LAP component |

Every new arithmetic path is compared both incrementally and cumulatively:

1. primitive output versus the CPU/source or exact GPU projection;
2. candidate versus the shipping control, for isolated performance and drift;
3. candidate versus the all-exact oracle, so separately admitted approximate
   kernels cannot silently spend the same KL budget several times.

LAP-0 first measures shipping-control versus all-exact cumulative KL/top-1. If
the shipping control is already above the repository gate, no additional
approximate route may be promoted until the debt is reconciled. A
BF16-bit-exact repaired projection remains admissible because it adds no debt.

### Repair policy

The preferred promotion target is BF16-bit equality after repair:

- residual D4x1/D4x2/D4x3 packs are evaluated on real production activations;
- the fast body writes a candidate value and an uncertainty measure;
- outputs whose BF16 rounding cell cannot be certified are queued;
- the exact current Q4/Q6 reduction recomputes only queued coordinates;
- an “all queued” test must reproduce the complete exact projection bit for
  bit.

An analytic error bound is preferred. An empirical distance threshold is
allowed only when it is selected on the declared calibration split, frozen
before heldouts, and passes the complete category/continuation suite. Thresholds
may depend on quant, projection role, and shape bucket. They may not depend on
prompt, token IDs, observed logits, category, or arbitrary layer ID.

Every artifact records:

- fast-versus-exact BF16 mismatch count before and after repair;
- maximum absolute/relative projection error;
- risk count, capacity, occupancy distribution, and overflow behavior;
- quantize/MMQ/repair time separately and inclusively;
- full-model cumulative KL/top-1 and complete free-running ID agreement;
- the exact fallback share by quant, role, and shape.

## Campaign sequence

`LAP-*` numbers remain stable work-package names; execution order is now
opportunity/risk ordered:

```text
LAP-0 current oracle/profile
  -> LAP-1 packed-dot body + sole-resident T16 consumer
  -> LAP-BW0 same-host bandwidth/clock/byte ledger + LAP-Q0 KL ablation
  -> LAP-6 source-F16 hipBLASLt real-input route
  -> LAP-5 dense/shared Q4/Q6
  -> LAP-2 real-input DS/risk calibration (primitive already implemented)
  -> LAP-3 selected Q4 gate/up, then LAP-4 selected Q4/Q6 down
  -> LAP-7 tiled attention
  -> LAP-8 residual/final parity
```

Reprofile after every promoted task. A later task does not start from the
pre-campaign Amdahl table.

Current progress:

| Task | State | Result / next condition |
| --- | --- | --- |
| LAP-0 | Complete | Fresh measured bridge, cumulative quality, routing, activation proxies, and unchanged Vulkan identity published. |
| LAP-1 | Complete | Direct resident-T16 MMQ32 is BF16-bit identical to X8, positive at all seven natural shapes, **2.502x/3.959x/5.502x** retained at M128/M256/M512, and within **4.66%/4.05%/3.02%** of X8 with no transpose or sidecar. |
| LAP-2 primitive | Complete | Three-plane pack, direct/guarded T16 MMQ, bounded queue, and overflow-safe exact correction landed in `d9bb6ad88`; 35 focused tests and cached trace pass. |
| LAP-BW0 / LAP-Q0 | Complete | The absolute quality schedule passes at max KL **0.0495426**, **316/320** top-1. Physical counters classify gate/up at **195.88 GB/s / 88.64%** of the stream anchor; Q4 down at **185.68 GB/s / 84.02%** and **87.78%** memory-unit busy; Q6 down at **123.99 GB/s / 56.10%** and **66.96%** memory-unit busy. Scheduled weight traffic explains **96.16%/99.28%** of Q4/Q6 physical fetch. The remaining route-tile reread ceilings are only **19.04/12.03 ms**. |
| LAP-6 | Admitted gfx1151 default | Torch-free, row-scaled hipBLASLt runs all five source-F16 projections on rows>1 real inputs with no added scratch; exact GEMV/tiled routes remain rollback. |
| LAP-5 | Admitted gfx1151 default | Resident Q4 pack8 and raw Q6 use 64x16 wave32 WMMA consumers. Q4 is BF16-bit identical to the raw-Q4 WMMA oracle; Q6 passes its CPU-reference gate and removes the traced 0.365-second dense/shared family bottleneck. |
| LAP-2 calibration / LAP-3 / LAP-4 | Admitted gfx1151 defaults | The original D4-gate/D4-down route reached **355.273/355.721 tok/s** but was rejected at max KL **0.0767056**. Same-byte D8 gate/up plus D4 down passes the clean complete category gate at max KL **0.040724836**, **317/320** top-1, **2.615x** aggregate natural-prompt prefill, flat decode, and exact lifecycle recovery. Its pre-admission pp512 samples were **353.951/356.082/356.473 tok/s**, token 2930. |
| Production publication | Complete/current | The direct all-exact gate remains max KL **0.049542582**, **316/320** top-1, with deterministic repeats, Poolside exact top-1, and exact lifecycle through 4K. Packed-query/wave-softmax attention, exact Q6 weight+activation prefetch, shape-qualified Q4 raw-nibble P8, and the row-qualified source-F16 schedule publish **645.803/575.942/468.311 tok/s** at 512/1K/4K. pp512 wall is **792.811 ms**, leaving **61.383 ms** to 700. |
| Direct Q4 gate/up wave decode | Admitted gfx1151 default | Direct per-column T16 decode removes pair decode/shuffle without changing resident bytes or arithmetic. The actual layer-1 leaf improves **8.107 -> 6.916 ms (-14.69%)**; clean pp512 improves **449.020 -> 474.363 tok/s (+5.644%)**, and cached tracing cuts the family **389.893 -> 317.722 ms (-18.51%)**. |
| Direct Q4-down wave decode | Admitted gfx1151 default | Direct per-column T16 decode removes pair decode/shuffle only for Q4 down while retaining Q6 row-vector production. Clean pp512 improves **473.963 -> 480.629 tok/s (+1.406%)**, and cached tracing cuts the Q4-down consumer **90.280 -> 71.378 ms (-20.94%)**. |
| Q6 qmicro resident payload | Admitted gfx1151 production default | Byte-neutral `[K32][col4][K4][QL8,QH4]` records preserve the 3,360-byte tile and every BF16 result. On the actual layer-1 660.6 MB tensor, natural-M512 selected prefill improves **5.1564 -> 5.0714 ms (-1.65%)** and top-10 exact decode improves **0.0910 -> 0.0846 ms (-6.99%)**. Clean pp512 improves **526.451 -> 530.447 tok/s (+0.759%)** and traced Q6 falls **126.594 -> 123.473 ms (-2.465%)**. Existing cache files convert once before upload; root lm-head and unmeasured backends remain legacy T16. |
| Q6 qmicro permute decode | Admitted gfx1151 production default | Two `v_perm_b32` byte gathers replace scalar quartet unpack without changing the byte-neutral record or arithmetic. The actual leaf improves **4.872 -> 4.741 ms (-2.67%)**, seven complete-state pairs improve **0.276% (5/7 wins)**, and tracing cuts the 115-call Q6 body **1,138.893 -> 1,124.852 ms (-1.23%)** with local128/VGPR80/LDS5120B/scratch0. Clean publication reaches **571.415/529.870/445.164 tok/s**. |
| Q6 compact activation cache | Admitted gfx1151 production default | Q6 never consumes Q8_1 sum metadata. Dropping that field and storing each bounded K16 quant sum as `int16` reduces activation staging **48 -> 40 bytes/row** and kernel LDS **5,632 -> 5,120 B** without changing dots or accumulation. The actual leaf improves **5.082 -> 4.911 ms (-3.36%)**; 15 complete-state pp512 pairs improve **550.584 -> 552.807 tok/s (+0.404%, 15/15 wins)**. Clean 512/1K/4K reaches **550.625/517.017/431.789 tok/s**. |
| Q6 half-row activation staging | Admitted gfx1151 production default | Each of 128 threads stages one 16-byte activation half and one K16 sum instead of leaving half the workgroup idle while 64 threads stage complete rows. Resources stay local128/VGPR88/SGPR128/LDS5120B/scratch0. The actual layer-1 leaf improves **4.902 -> 4.885 ms (-0.351%, 16/21 wins)**; the all-Q6 screen improves **21/23** layers and **111.798 -> 111.490 ms (-0.276%)** with zero BF16 mismatches. Complete pp512 A/B is exact and positive at **552.562 -> 553.018 tok/s (+0.083%)**; clean headline publication is neutral. |
| Q6 padded-activation elision | Admitted gfx1151 production default | Natural M512 has **117,760 useful** versus **362,944 padded** Q6 row slots. Padded slots are never consumed by the guarded dot/store loops, so production skips their zero LDS stores and K16 sum work. It improves **19/23** actual layers and **112.008 -> 111.806 ms (-0.180%)**, with zero BF16 mismatches and unchanged local128/VGPR88/SGPR128/LDS5120B/scratch0. Complete pp512 A/B is exact and positive at **552.983 -> 553.559 tok/s (+0.104%, 7/11 wins)**; clean publication reaches **551.459/517.307/432.099 tok/s**. |
| Q6 selected-down integer WMMA | Admitted gfx1151 production default | Four wave32 groups consume the existing planar-qmicro/D4 caches as 16x16x16 signed-int8 x unsigned-Q6 fragments while preserving the two K16 scales, `-32*sum(x)` correction, ordered FP32 K32 accumulation, and BF16 store. The latest body carries both the next raw qmicro record/metadata and next compact Q8 half-row in registers during current fragment compute, then reuses the same LDS planes. Activation prefetch improves the retained leaf **4.104 -> 4.045 ms (-1.440%, 20/21 wins)**; clean pp512 improves **636.073 -> 639.114 tok/s**, and tracing cuts 23 Q6 calls **101.963 -> 100.367 ms (-1.565%)** at local128/VGPR112/SGPR128/LDS5120B/scratch0. |
| Q4 selected-down raw-nibble P8 | Admitted gfx1151 production default | Payload-only next-K32 prefetch transfers from gate/up to the 64x32/local64 single-output body without changing resident bytes, LDS, scratch, arithmetic, or BF16 output. Three traced M512 arms cut 72 Q4-down calls **217.416 -> 212.090 ms (-2.450%)** at VGPR **88 -> 96**; seven complete-state pp512 pairs improve **639.574 -> 643.166 tok/s (+0.562%, 7/7 wins)**. Clean selector-unset publication is **643.141/573.717/466.913 tok/s**. |
| LAP-7–LAP-8 | Exact cached-only, cached-metadata, qrow6, and dense-initial policies admitted | Complete M128 tiles append before cached-only attention while partial, wrapped SWA, verifier, explicitly evicted, and unmeasured paths retain exact fallbacks. The final initial-fill policy uses global qrow4/qrow6 and SWA qrow4 without per-token position/eviction reads. Matched pp512 improves **552.144 -> 559.539 tok/s (+1.339%)**; clean publication reaches **559.290 tok/s**, and tracing cuts attention **153.226 -> 141.846 ms (-7.43%)**. Scalar split-state, M16xK64 WMMA, M8xK64 WMMA, qrow8, head2, qhead3, and nine-wave GQA sharing remain closed. |

## Post-500 campaign — 700 production stretch

The 350 and 500 tok/s milestones prove the compounded production package, but
they are not roofline results. Current clean production measures **0.796093
seconds** synchronized pp512 wall. The clean post-selected-down-P8 trace at
`60f089650` measures **0.853428 seconds** kernel span and **1.119634 seconds**
inclusive kernel sum. The profiler perturbs this
library-heavy route, so the
unprofiled wall is the production claim and the cached trace supplies family
attribution only. The sum exceeds span because two streams overlap; inclusive
family durations are not additive Amdahl savings.

The achieved 500 gate required at least three clean selector-unset pp512
repetitions with median and every sample at or above 500 tok/s. The next
production gate is **700 tok/s** under the same model/quant/KV/queue policy and
all existing correctness, quality, decode, determinism, memory, and lifecycle
gates. The 700 row is a target, not a performance claim, until LAP-BW0 supplies
locked-clock physical traffic and achievable-bandwidth evidence.

| Current production family | Inclusive pp512 kernel time | Inclusive-sum share | Remaining decision |
| --- | ---: | ---: | --- |
| Selected D8 Q4 gate/up | **334.201 ms** | **29.85%** | Shape-qualified raw-nibble P8 is the gfx1151 default. Physical counters before P8 reached **195.88 GB/s / 88.64%** of the read anchor. Compact metadata prefetch and non-temporal payload loads are both measured regressions; the next screen must remove physical bytes, cross-tile work, or a caller boundary. |
| Activation/reduce/residual | **269.925 ms** | **24.11%** | This inclusive bucket moved **+3.294%** between single profiled runs, confirming overlap noise rather than an additive ceiling. The prior queue union showed only **0.826 ms** secondary-only; reopen only with caller-stream relief, reduced bandwidth contention, or a fused producer that wins the complete wall. |
| Selected D4 Q4/Q6 down | **171.040 ms** | **15.28%** | Direct Q4 decode and byte-neutral planar-Q6 integer WMMA are retained. Q4 selected-down P8 is confirmed at local64/VGPR96/LDS1536B/scratch0 and moves this inclusive family **-0.398%** versus the prior trace. Further work requires fewer physical weight bytes or a new cross-tile schedule. |
| Static-range direct hipBLASLt source-F16 | **124.668 ms** | **11.13%** | All five contractions and fused producer boundaries are included. Exact fusion removes **96** standalone casts. Concatenated QKV still has only a **2.891-ms** modeled ceiling before restride, and layout-preserving `GroupedGemm` exposes zero gfx1151 algorithms. |
| Q4/Q6 WMMA dense/shared | **95.167 ms** | **8.50%** | This inclusive family overlaps routed work. The secondary shared branch remains hidden. An exact shared gate/up+SiLU leaf improved **14.56%** yet regressed production **0.52%**; reopen only after queue-exclusive caller-stream evidence changes that premise. |
| Global + SWA attention | **67.923 ms** | **6.07%** | Qualified positions 128/256/384 use exact BF16 cache widening, packed F32 query/output tiles, one wide QK and one wide PV hipBLASLt contraction, and one wave32 per causal-score row. Partial, wrapped, explicitly evicted, verifier, decode, and unmeasured routes retain exact fallbacks. |
| Router | **22.883 ms** | **2.04%** | The after-router boundary remains production. Eight-token reuse is retained; eager least-priority release regresses **0.198%** and is closed. |
| Norm/RoPE/gates, metadata, KV/tails and other | **33.827 ms** | **3.02%** | No individual exact subfamily currently has the 5% perfect-removal ceiling needed to displace the selected-projection campaign. |

The current trace gives concrete Amdahl checkpoints; the clean publication
below is a retained performance claim:

- The clean production median is now **645.803 tok/s**. The selector-unset
  1K/4K medians are **575.942/468.311 tok/s**. Row-qualified source-F16
  scheduling improves the preceding packet at every length. The latest clean
  trace predates only that descriptor change and has **2,417** dispatches,
  **1,119.634 ms** inclusive kernel sum, and
  **853.428 ms** kernel span. Selected gate/up remains largest at
  **334.201 ms**; activation/reduce/residual is second at **269.925 ms**.
  Selected Q4/Q6 down is **171.040 ms**, with the admitted Q4 P8 body observed
  at local64/VGPR96/LDS1536B/scratch0. The declared 500 gate is closed.
- Dense-initial metadata elision cuts global+SWA attention
  **153.226 -> 141.846 ms (-7.43%)** with the intended exact launch mix.
- The clean wall must fall from **792.811 ms** to **731.429 ms** for 700 tok/s,
  a further **61.383 ms**. The current profiled kernel span is **121.999 ms**
  above that wall, so sufficient work exists, but inclusive buckets cannot be
  added across the two streams. The next material screen must change selected
  projection physical bytes, cross-tile reuse, a producer/consumer boundary,
  or another measured caller-stream latency limiter and demonstrate a
  multi-millisecond named-family win before a complete-model run.
- Queue-exclusive attribution closes the apparent shared-expert ceiling.
  The refreshed caller stream spans **852.825 ms** with **787.420 ms** of
  kernels. The secondary shared stream contains **340.456 ms** of kernels and
  is hidden except for **0.826 ms**: it starts **76.139 ms** after the request
  and ends **6.038 ms** before it. The two queues execute concurrently for
  **339.630 ms**. Its **257.508-ms** standalone SiLU cost is therefore not an
  Amdahl saving. The observed **64.579-ms** both-idle time is a profiled-trace
  quantity, not a clean-wall ceiling; the profiled span is already
  **51.715 ms** slower than the unprofiled production wall.
  The exact dual-pack8 gate/up+SiLU leaf cut its actual-weight operation
  **0.50183 -> 0.42874 ms (-14.56%)**, then lost the complete pp512 wall
  **580.394 -> 577.374 tok/s (-0.52%, 1/7 wins)**. Shared work is frozen
  unless a new trace shows unhidden spill or reduced caller-stream
  contention.
- The old active-expert-once lower bound made gate/up appear to sustain only
  **115.24 GB/s**. Production rereads a full expert weight for every 32-row
  route tile: **10,237 active groups become 14,034 row tiles**, so the
  schedule-correct resident request is **51.045 GB** and the pre-concurrency
  attribution rate was **162.09 GB/s / 73.34%** of the existing 221 GB/s
  anchor. Gate/up therefore clears the interim 70% requested-byte floor
  outside cross-stream contention. Down requests **27.524 GB**
  across Q4 row32 and Q6 row64 grids; using the refreshed **191.098-ms**
  family window leaves the rounded rate at **144.03 GB/s / 65.17%**.
  These are requested-byte rates, not controller counters; locked-clock
  physical traffic remains the final LAP-BW0 step for selected down.
- The gate/up physical-counter half of LAP-BW0 is complete. The production
  layer-1 consumer fetches **1,325,709,312 bytes** from video memory per
  dispatch and sustains **195.88 GB/s** at the unprofiled **6.768-ms** median,
  or **88.64%** of the existing stream anchor. It is **80.89%** memory-unit
  busy, **95.77%** occupied, **53.25%** L2-hit, and only **1.38%**
  ALU-stalled by LDS. Natural routing creates **297** 32-row expert tiles from
  **228** active experts, a **1.3026x** complete-weight reread factor. The
  remaining gate/up problem is physical bytes, not barriers or occupancy.
  The root-owned performance policy could not be pinned by the benchmark
  user; the evidence records `auto` and a **2.54-GHz** median in-kernel clock.
  Evidence:
  `benchmarks/results/2026-07-26-gfx1151-laguna-gate-up-physical-counters.json`.
- LAP-BW0 selected-down counters are also complete. Across the 24 Q4 layers,
  physical fetch is **13.405 GB** in **72.195 ms**, or **185.68 GB/s /
  84.02%** of the stream anchor, with **87.78%** duration-weighted memory-unit
  busy. The 32-row grid turns **5,144** active expert groups into **7,088**
  weight passes (**1.378x**); scheduled weights explain **96.16%** of physical
  fetch. Across the 23 Q6 layers, physical fetch is **14.740 GB** in
  **118.888 ms**, or **123.99 GB/s / 56.10%**, with **66.96%** memory-unit
  busy. Its 64-row grid is already only **1.113x** active groups and scheduled
  weights explain **99.28%** of physical fetch. Perfect removal of all Q4 and
  Q6 route-tile rereads is only **19.04 + 12.03 = 31.07 ms**, which would put
  the current pp512 wall near **587.0 tok/s**, not 700. Down-specific K1024
  persistence may still buy part of Q4's 19-ms ceiling, but selected down is
  not the sole 700 lever. Evidence:
  `benchmarks/results/2026-07-26-gfx1151-laguna-selected-down-physical-counters.json`.

The quality contract remains binding. LAP-Q0 found that the prior
**0.040724836** result compared current production with an already approximate
shipping control and was not an absolute budget measurement. Direct
production-versus-all-exact reached **0.053502420** and therefore failed. A
row-qualified hipBLASLt schedule—heuristic 2 for the K3072xN72 SWA gate
through M128, heuristic 4 above M128 and everywhere else—passes at
**0.049542582**, leaving only
**0.000457418** below the 0.05 ceiling. The rejected D4 gate candidate already
showed that another approximate shortcut can hold 355+ tok/s while failing
quality at KL **0.0767056**. New approximate paths are closed unless they first
buy back absolute quality budget; prefer exact data-movement/scheduling wins
and preserve K accumulation order. The production-absolute harness is now
repaired to follow the current qrow4, double-buffered gate/up, 64-row Q6-down,
and range-direct F16 selectors instead of the superseded pre-350 lane. Its
320-step revalidation reproduces the published **0.049542582** / **316 of
320** result.

Immediate execution queue:

1. Padded-activation elision is now clean production at
   **551.459/517.307/432.099 tok/s**. It improves the full 23-layer exact Q6
   sub-window only **0.180%** and confirms that activation padding is not the
   missing route-tile architecture. Freeze the current **190.363-ms**
   selected-down body. X16 is now closed after exact c1 regressed **7.654%**
   despite c4/c8 wins. The sole-resident byte-neutral
   **T16-local-Q + four-column/three-byte metadata** microtile passed exact
   c1/c2/c4/c8 by **4.929%/0.781%/3.691%/4.633%**, but its best exact
   actual-weight M512 selected-prefill consumer regresses T16
   **9.402044 -> 9.570781 ms (+1.795%)**. Wave broadcast and quartet-owned
   LDS expansion regress **9.539%/5.563%**. The prefill consumer is removed;
   do not materialize or integrate qmicro for Q4. Return now to an expert
   schedule that reduces Q4 route-tile rereads without larger accumulator
   state or F32 partial spills. The bounded intermediate-tile sweep is also
   closed: rows40 reduces all-layer route tiles **8.32%** but regresses the
   actual leaf **2.40%**, while rows48 reduces tiles **13.15%** but regresses
   **1.71%**. Both exact candidates are removed.
   A row-tile-fast grid axis swap is also closed after the exact actual-weight
   leaf regressed **6.908966 -> 6.921503 ms (+0.181%)**. Do not retry a launch
   axis permutation without counter evidence or a schedule that actually
   shares a resident weight slice across workgroups.
   Keep byte-neutral Q6
   qmicro and direct Q4 decode. The exact MMQ
   grouped-combine reuse is now clean production: it removes 47 launches and
   the routed-output round trip. Do not repeat
   Q4 activation double buffering, Q6 local64/local256 workgroup changes,
   Q6 128-column/local256 widening, Q4-down 128-column widening,
   static-upper sentinel grids, launch-bounds occupancy hints, duplicate-decode
   row halves, 64-row Q4 accumulation, paired-scale metadata, or F32 partial
   spills. The exact fused selected-SiLU pack is now clean production at
   **546.100 tok/s**. A heavy-expert 64x128/local256 Q6 body is also closed:
   the best valid actual-weight leaf saves only **0.017 ms** before its
   required extra metadata schedule/launch, while the >=129-row tail regresses
   **2.14%**. Pursue a different cross-tile/expert schedule rather than another
   larger local256 row tile.
   A transposed **32-column x 128-row/local128** Q6 qmicro body is also
   closed. It held the production **32 F32 accumulators/lane** and reduced
   the natural all-Q6 route grid **5,671 -> 5,253 tiles (-7.37%)**, but
   per-expert padding expanded the layer-1 scheduled rows
   **15,808 -> 29,696**. The exact actual-weight leaf regressed
   **4.8492 -> 6.7545 ms (+39.29%)** with zero BF16 mismatches. Do not retry
   a narrower-column/wider-row rectangle unless its scheduler avoids padding
   amplification rather than merely changing the tile aspect ratio. A
   padding-free hybrid control is closed too: four complete 128-row prefixes
   hold both schedules at 512 rows and the same total workgroup count, yet
   32x128 still regresses **0.44755 -> 0.47493 ms (+6.12%)**. Extra
   activation/LDS cost, not only padding, defeats this geometry.
2. Keep exact cached-metadata attention in production. Clean selector-unset
   512/1K/4K improves **2.195%/1.213%/1.665%**; traced attention falls
   **175.802 -> 160.123 ms (-8.92%)** with the qualified 12-global-start0,
   36-global-metadata, and 144-SWA-metadata policy. The prior scalar-split,
   tiled-WMMA, head-pair, qhead3, and nine-wave GQA bodies remain closed. The
   new exact global-only qrow6 primitive is the active bounded screen:
   qrow4 -> qrow6 improves **1.202x/1.262x/1.278x** at global starts
   128/256/384, is neutral at start 0, and models **6.083 ms** pp512 saving.
   Its SWA sibling lost **10.9–18.4%** and is removed. The qualified global
   policy now passes its repeated complete-state gate:
   **546.056 -> 548.774 tok/s (+0.498%, 7/7 wins)** with every compared
   output/state digest exact. It is the gfx1151 default with explicit qrow4
   rollback. Clean selector-unset 512/1K/4K reaches
   **547.064/513.180/428.628 tok/s**, and tracing cuts attention
   **158.702 -> 152.406 ms (-3.97%)** while observing the exact qualified
   12-qrow4/36-qrow6/144-SWA-qrow4 pp512 launch mix. Qrow3 is now closed:
   although it is F32-bit exact and beats cached-metadata qrow4 on global
   tiles, it loses every SWA position and measures **13.7874 ms** versus
   **13.3577 ms** for weighted qrow4 and **12.8481 ms** for the qualified
   production policy. Its global-start0 result merely ties the actual
   non-metadata production body (**0.18634 vs 0.18580 ms**).
   SWA qrow5 is also closed: it is F32-bit exact but regresses qrow4 by
   **1.66%/3.92%/5.00%/3.21%** at starts 0/128/256/384. The complete
   production-shaped policy moves **11.8174 -> 12.1906 ms (+3.16%)**.
   Together with the larger qrow6 losses, this closes wider SWA adjacent-row
   accumulation without a new state-compression mechanism.
   The next exact dense-initial leaf is positive and retained for immediate
   runtime qualification. Before the first wrap, complete preappended tiles
   have identity token positions and no eviction, allowing the global/SWA
   kernels to remove per-token position/eviction reads while retaining the
   complete `KVLiveSpans` ABI and base-offset mapping. Global qrow4/qrow6 and
   SWA qrow4 are F32-bit exact at every pp512 position; the qualified
   production-shaped leaf improves **12.8348 -> 11.8695 ms (1.0813x)** and
   models **11.584 ms** pp512 saving. Integrate only for runtime-proven
   complete initial no-wrap tiles; partial, wrapped, verifier, gfx1100, and
   unmeasured routes remain on their exact fallbacks.
   Integration now passes: the runtime additionally invalidates the fast path
   after any explicit eviction. Seven matched complete-state pairs improve
   cached-metadata rollback **552.144 -> 559.539 tok/s (+1.339%)**, saving
   **12.255 ms** at the medians with identical logits, hidden states, KV,
   token/logit, and cursor. The gfx1151 capability is the default with
   `prefill_dense_initial=false` rollback. Clean selector-unset publication
   reaches **559.290/523.090/439.044 tok/s**, and tracing cuts attention
   **153.226 -> 141.846 ms (-7.43%)** while observing the exact
   12-global-qrow4/36-global-qrow6/144-SWA-qrow4 dense-initial mix. This
   checkpoint is complete; keep the automatic exact fallbacks.
3. Freeze source-F16 grouping. One combined row-major QKV contraction is
   F32-bit exact but saves only **2.891 ms** across the 12 full and 36 SWA
   layers before splitting `[M,Q+K+V]` back into the three contiguous
   production outputs. The layout-preserving hipBLASLt `GroupedGemm` route
   returns zero algorithms for the full QKV problem with either zero or
   64-MiB workspace on gfx1151. Do not add concatenated resident weights or a
   restride kernel for this ceiling; reopen only if the installed library
   gains a viable grouped algorithm or consumers accept the combined stride.
   A distinct boundary-fusion premise is now retained in gfx1151 production:
   RMSNorm and softplus gating emit the exact FP16
   representation of their existing BF16 output, removing 96 casts at pp512.
   Primitive shapes improve **0.040472 -> 0.021564 ms** and
   **0.192665 -> 0.135213 ms**; seven matched full-model pairs improve
   **554.909 -> 559.320 tok/s (+0.795%, 6/7 wins)** with identical token and
   logit. A second seven-pair gate preserves logits, both hidden snapshots,
   complete KV, token/logit, and cursor exactly. Clean selector-unset
   publication reaches **559.554/523.912/440.809 tok/s**; tracing removes all
   96 standalone casts and records **1,696** pp512 dispatches.
4. **Complete:** LAP-BW0 physical counters classify gate/up and Q4 down as
   controller-bound at **195.88/185.68 GB/s**. Q6 down reaches only
   **123.99 GB/s**, but scheduled weights already explain **99.28%** of its
   physical fetch and its route reread ceiling is only **12.03 ms**. Q4's
   **1.378x** row-tile reread is the next selected-down screen. A
   down-specific persistent K256 body is materially narrower than the rejected
   gate/up body: K1024 instead of K3072, one projection instead of two, and
   four partial passes instead of twelve. That exact screen is now closed:
   natural layer 10 regresses **2.872974 -> 18.455975 ms (6.424x slower)**
   despite zero BF16 mismatches. Three FP32 writes plus three FP32 reads of the
   full 5,120x3,072 accumulator plane, together with Q8 rereads across 48
   output tiles, cost much more than the removed weight passes. The candidate
   is fully removed. The no-partial MMQ64x64 follow-up is also closed. The
   shared-weight body regresses natural layer 10 **2.948389 -> 5.200135 ms**;
   restoring per-lane direct decode narrows that to
   **2.951132 -> 3.790972 ms (+28.46%)**. A fully occupied 64-row control
   still loses **0.075383 -> 0.081069 ms (+7.54%)**, so padding-free hybrid
   prefixes cannot recover it. All candidate surfaces are removed.
   Selected-down scheduling is closed unless a materially new byte model
   appears; return to the 314.920-ms gate/up family. Retire the pre-admission
   **78.27 ms/layer versus 52.80 ms layer-1** bridge instead of scaling it
   into new forecasts.
5. After down, revisit gate/up only from physical counters or a new
   cross-tile/expert schedule. The corrected requested-byte ledger already
   reaches **73.37%** of the read anchor, so a local body tweak must explain
   how it reduces route-tile rereads or raises measured bandwidth. The first
   such byte-removal screen is now closed. A 64x64 body decoded each K256
   weight slab once and kept its F32 partial plane in LDS; it is BF16-byte
   exact but traces at **248 VGPR / 39,936 B LDS** and regresses the actual
   layer-1 pack-inclusive leaf **6.628 -> 30.191 ms (4.56x slower)**. Removing
   the slab partial plane and carrying all **32 F32 accumulators/lane** in
   registers improves the candidate to **11.433 ms**, still **66.5% slower**
   than production because the 64-row route expands padding and doubles
   column workgroups. Both implementations and every diagnostic hook were
   removed. Reopen cross-row sharing only if the scheduler avoids per-expert
   64-row padding as well as the second-launch/local256 costs already closed
   above. Evidence:
   [`2026-07-26-gfx1151-laguna-gate-k256-ldsacc-rejected.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-gate-k256-ldsacc-rejected.json).
6. **Complete:** M2048 is the gfx1151 default while attention remains M128.
   Matched M512 -> M2048 improves 1K/4K **5.420%/5.752%**, keeps pp512 within
   **-0.358%**, and passes full-logit quality at max relative KL
   **0.000012503** with 100% top-1. Clean selector-unset production reaches
   **506.299/410.099 tok/s** at 1K/4K. Exact cursor, multi-wrap KV, deterministic
   repeats, 1.755-GB scratch, and lifecycle recovery are published. This
   receives no pp512 credit.
7. Do not retry T16-lite: its best exact byte-plane/LDS decoder loses
   **11.22–17.63%** at c1/c2/c4/c8. X16 is also closed: its exact one-pack
   consumer loses **7.654%** at c1 even though it wins at c4/c8. T16-local Q
   with four-column/three-byte metadata also closes at prefill: exact decode
   is positive at every shape, but all three exact MMQ consumers lose at
   natural M512. T128 is also closed:
   column-major payload locality bought **12.23%** at the M512 leaf but the
   best exact virtual-thread decoder still lost **6.10–6.86%**. Do not retain
   a second resident view or pay a prefill-to-decode transpose.
8. Screen exact shared/routed MoE branch concurrency before reopening another
   expert kernel body. The always-on shared branch is independent of router
   selection and routed gate/up/down until the final combine, so it can run on
   a nonblocking secondary stream with one input-ready and one output-ready
   event. The default-off Q4 production-style fixture is BF16-byte identical
   to the sequential route. Its absolute ceiling is the traced **53.257-ms**
   dense/shared family: perfect hiding would move the then-current wall only to
   **594.1 tok/s**, so this is additive—not the 700 solution by itself. Run a
   clean counterbalanced A/B with `GPU_MAX_HW_QUEUES=2` in both arms, require
   complete-state equality, and retain only if a cached trace proves real
   overlap without slowing the controller-bound routed families.
   The candidate passes that gate. Seven queue-matched pp512 pairs improve
   **560.837 -> 567.577 tok/s (+1.202%, 7/7 wins)** with every full-state
   digest exact. Clean default-off 512/1K/4K reaches
   **565.457/525.733/443.027 tok/s**. Cached tracing places **188** shared
   kernels on the secondary stream and overlaps **100.390/101.241 ms
   (99.16%)** with caller-stream kernels; despite contention, total kernel span
   falls **909.598 -> 896.871 ms (-12.727 ms)**. The gfx1151 capability and
   two-queue process policy are now promoted with automatic single-queue
   fallback and explicit session rollback. Clean selector-unset publication
   reaches **565.447/526.711/443.444 tok/s**; production tracing observes two
   queues/two streams and overlaps **76.883/77.763 ms (98.87%)** while cutting
   kernel span **909.598 -> 898.334 ms**.
9. **Rejected and removed:** delaying secondary-stream shared work until
   routed gate/up completed preserved complete state but regressed the
   queue-matched pp512 median **566.394 -> 565.011 tok/s (-0.244%)**, won only
   **2/7** pairs, and produced a **535.465 tok/s** low tail. No trace was
   warranted and every launch-phase selector was removed. Do not retry this
   short overlap window.
10. **Retained candidate:** preserve the long gate/up-plus-down overlap window
   but place the dependency event after router selection. Seven exact
   complete-state pairs improve **567.767 -> 568.181 tok/s (+0.073%, 5/7
   wins)**. Tracing proves router recovers **44.075 -> 23.356 ms**, but shared
   contention moves into gate/up **322.200 -> 344.619 ms**, leaving only a
   **0.310-ms** kernel-span win. The gfx1151 capability and clean
   selector-unset publication now pass at **566.839/527.381/444.447 tok/s**.
   Do not model this verified micro-win as material progress toward 700.
11. **Retained production:** create the after-router secondary stream at the
   device's lowest scheduling priority, if gfx1151 exposes a non-degenerate
   priority range. The current trace slows secondary work to **269.084 ms** and
   raises gate/up **22.418 ms**, while more than **500 ms** of routed gate/up
   and down remains available for hiding it. Require exact state, a positive
   seven-pair gate, and a trace showing gate/up recovery without shared work
   spilling materially past the final combine. The candidate passes: priority
   **0 -> +1** improves exact matched pp512 **568.106 -> 570.914 tok/s
   (+0.494%, 6/7 wins)**. Tracing recovers gate/up **344.619 -> 337.502 ms**
   and cuts kernel span **898.024 -> 890.769 ms (-7.255 ms)**. Shared work
   slows **269.084 -> 337.239 ms**, but **99.75%** remains hidden and only
   **0.853 ms** is unoverlapped. The gfx1151 capability and clean
   selector-unset publication pass at **568.849/527.113/444.508 tok/s**.
12. **Rejected:** hold the shared stream at priority +1 in both arms
   and compare eager release against the retained after-router boundary.
   Earlier release restores the longest possible overlap window and may remove
   the remaining **0.853-ms** shared spill; priority protection may be enough
   to keep router and gate/up on the critical path. Require seven
   counterbalanced complete-state-exact pairs and trace only if eager release
   is positive. No new production surface is needed for this screen. Eager
   release preserves complete state but regresses **570.796 -> 569.666 tok/s
   (-0.198%, 1/7 wins)** and adds **1.339 ms** at the median paired wall. No
   trace or production change is retained.
13. **Retained production:** reduce Q6 qmicro decode instructions without
   expanding its byte-neutral 12-byte quartet record. Gather the two
   interleaved low-nibble words into per-column words with gfx11 `v_perm_b32`,
   then combine the existing high-two-bit word with masks and shifts. Keep the
   64-column x 64-row/local128 geometry, activation cache, FP32 accumulation
   order, resident bytes, and BF16 boundary unchanged. Gate first on the
   uneven/empty-expert CPU-reference fixture and actual layer-1 BF16 identity;
   retain only if a counter-rotated actual-weight leaf improves before any
   full-model integration. This is a new instruction-path premise, not a
   retry of non-temporal loads, paired-scale decode, K64 staging, or a larger
   row tile. The actual leaf is exact and improves **4.872 -> 4.741 ms
   (-2.67%)**. Seven complete-state pairs improve
   **567.998 -> 569.563 tok/s (+0.276%, 5/7 wins)**. Cached tracing executes
   all **115** intended calls and cuts their total
   **1,138.893 -> 1,124.852 ms (-1.23%)** with VGPR **88 -> 80**, unchanged
   LDS, and zero scratch. Clean publication reaches
   **571.415/529.870/445.164 tok/s**.
14. **Retained production:** make the same 12-byte
   qmicro record planar:
   store its four `ql01` bytes in the first dword, four `ql23` bytes in the
   second, and retain the four high-bit bytes in the third. This is
   byte-neutral and lets selected prefill load both low-nibble column words
   directly, removing the two now-proven `v_perm_b32` gathers. Update every
   qmicro consumer and the one-time legacy-to-qmicro adapter together; keep
   legacy T16 and the current interleaved-qmicro decoder as explicit controls.
   Gate on byte-neutral roundtrip, exact c1/c2/c4/c8 decode, the
   uneven/empty-expert CPU oracle, and a counter-rotated actual-weight
   natural-M512 leaf before any runtime promotion. This is a layout-order
   screen, not a larger record or a duplicate sidecar. The byte-neutral
   roundtrip, uneven/empty-expert oracle, and exact c1/c2/c4/c8 decode are
   green. On the actual 660.6-MB layer-1 tensor, 21 counter-rotated samples
   improve current permute prefill **4.7718 -> 4.7568 ms (-0.314%)** and c1
   decode **0.08564 -> 0.08415 ms (-1.736%)**, with zero BF16 mismatches.
   Clean cached tracing executes the intended planar prefill and decode
   templates at local128/VGPR80/LDS5120B/scratch0; complete-state full-model
   A/B uses two opposite resident-owner-order blocks because the byte layouts
   cannot share one 77.4-GB owner. Across 14 samples per arm, planar is
   **+0.013% by mean / +0.139% by median**, and the order-adjusted median
   delta is **+0.010 tok/s**: aggregate-neutral, with complete state exact.
   The verified leaf/decode sub-window wins therefore retain and enable
   `LAGUNA_Q6_QMICRO_PLANAR`. Clean selector-unset publication reaches
   **573.354/530.351/446.189 tok/s**, improving all lengths
   **0.339%/0.091%/0.230%** with deterministic tokens, exact positions, and
   full allocation recovery.
15. **Rejected and removed:** build a cooperative Q4 gate/up
   **128-column x 64-row/local256** body as two independent 128-thread row32
   teams. Decode and stage each T16 weight tile once for both teams, but keep
   each lane at the production 32 FP32 accumulators. This directly targets the
   measured **1.3026x** Q4 route-tile reread without repeating the rejected
   local128 row64 body (64 accumulators/lane), 64-column row64 body (duplicate
   activation loads), or 256-column row32 body (10,240-byte weight tile).
   The uneven/empty-expert CPU oracle passes and the candidate is BF16-bit
   identical to production. The fully padded body nevertheless regresses
   actual layer-1 M256/M512 **38.07%/53.69%**. Pairing only adjacent complete
   row32 tiles and leaving odd tails on production removes padding
   amplification but still regresses **22.36%/34.39%**. Cached tracing shows
   local256/VGPR96/LDS8192B/scratch0; broadcasting the 5,120-byte decoded
   weight tile through LDS costs more than rereading compact T16 into
   per-wave registers. Every candidate surface is removed. Evidence:
   [`2026-07-26-gfx1151-laguna-q4-cooperative-row64-rejected.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q4-cooperative-row64-rejected.json).
16. **Rejected and removed:** combine two independently positive premises:
   the byte-neutral Q4 qmicro layout already improves exact c1/c2/c4/c8
   decode and removes **25,165,824 bytes** from the actual gate/up pair, while
   production direct-wave T16 decode removed pair shuffle and cut the family
   **18.51%**. Build a qmicro consumer around the production
   **128-column x 32-row/local128** register schedule, extracting only packed
   scale/min metadata per lane while retaining direct per-column quant decode.
   This is distinct from the removed shared-MMQ32 qmicro body measured at
   9.57 ms. The uneven/empty-expert CPU-reference gate passes and the
   candidate is BF16-bit identical to production. On actual layer-1 weights,
   however, an unaligned-dword metadata decoder regresses M256/M512
   **4.92%/6.17%**. Replacing it with explicit three-byte loads still regresses
   M512 **6.861 -> 7.087 ms (+3.31%)**. Cached tracing shows packed
   coefficient extraction raises VGPR **88 -> 120** with unchanged
   local128/LDS3072B/scratch0. Every prefill candidate surface is removed;
   no materializer or runtime route was added. Evidence:
   [`2026-07-26-gfx1151-laguna-q4-k-qmicro-direct-wave-rejected.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q4-k-qmicro-direct-wave-rejected.json).
17. **Rejected and removed:** test a byte/decode midpoint rather than another
   fully packed qmicro consumer. Keep one T16 coefficient plane expanded and
   pack only the other as four 6-bit values per three-byte record. The
   resulting 2,336-byte tile saves **32 bytes / 1.351%** versus T16 while
   requiring only one packed coefficient extraction per lane. Screen both
   scale-expanded/min-packed and min-expanded/scale-packed orderings on the
   production direct-wave body. Both scale-packed and min-packed orderings
   round-trip raw Q4_K exactly, pass the 13-case CPU-reference gate, and are
   BF16-bit identical to production. Interleaved three-byte records regress
   the actual M512 leaf **3.62%/3.59%**. Reordering the same 96 bytes into
   three planar byte planes still regresses **3.83%/3.74%**. Final tracing
   shows both one-plane candidates at local128/VGPR120/LDS3072B/scratch0
   versus production VGPR88. All candidate surfaces are removed. Evidence:
   [`2026-07-26-gfx1151-laguna-q4-k-hybrid-metadata-rejected.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q4-k-hybrid-metadata-rejected.json).
18. **Admitted production default:** the Q6-only integer-WMMA
   selected-down body consumes the existing D4 activation cache and
   byte-neutral planar qmicro weights without a sidecar. Four wave32 groups
   own independent 16-row bands and each issue two signed-int8 x unsigned-Q6
   16x16x16 fragments per K32. The two integer results retain the existing
   per-half Q6 scales, `-32*sum(x)` correction, FP32 K32 accumulation order,
   BF16 store, 64-row route map, and exact fallback. The uneven/empty-expert
   CPU oracle is BF16-byte exact. On actual layer-1 natural-M512 weights, 21
   counter-rotated burst-seven pairs improve **4.7654 -> 4.5655 ms
   (-4.20%, 21/21 wins)** with zero BF16 mismatches and complete memory return.
   Cached tracing names the intended template at
   local128/VGPR96/SGPR128/LDS5120B/scratch0 versus retained VGPR80. Clean
   selector-unset 512/1K/4K improves
   **573.354/530.351/446.189 -> 576.137/543.213/459.054 tok/s** with
   deterministic tokens, exact positions, and complete allocation return.
   Refresh the Q6 family attribution before deciding whether another fragment
   geometry is justified. Evidence:
   [`2026-07-26-gfx1151-laguna-q6-selected-down-integer-wmma-candidate.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q6-selected-down-integer-wmma-candidate.json).
   [`production`](../benchmarks/results/2026-07-26-gfx1151-laguna-q6-selected-down-integer-wmma-production.json).
19. **Admitted production default:** hoist each wave's invariant
   activation fragments outside the four-column-fragment loop in the
   production Q6 integer-WMMA body. The source now constructs the same two K16
   `a` vectors once per `(wave,row,K32)` rather than once per 16-column
   fragment. Weight fragments, result mapping, two Q6 scales,
   `-32*sum(x)` correction, FP32 K32 order, and BF16 stores remain exact.
   Twenty-one actual layer-1 natural-M512 pairs improve
   **4.5645 -> 4.5126 ms (-1.136%, 20/21 wins)** with zero BF16 mismatches
   and complete memory return. Cached tracing stays at
   local128/VGPR96/SGPR128/LDS5120B/scratch0, identical to the current
   integer-WMMA body. Clean selector-unset 512/1K/4K improves
   **576.137/543.213/459.054 -> 577.396/545.366/459.716 tok/s
   (+0.218%/+0.396%/+0.144%)** with deterministic tokens, exact final
   positions, and complete allocation return. Evidence:
   [`2026-07-26-gfx1151-laguna-q6-integer-wmma-hoist-activation-candidate.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q6-integer-wmma-hoist-activation-candidate.json).
   [`production`](../benchmarks/results/2026-07-26-gfx1151-laguna-q6-integer-wmma-hoist-activation-production.json).
20. **Rejected and removed:** broadcasting each result row's invariant `d`
   and two K16 sums from lanes 0/16 is BF16 exact, but two wave shuffles per
   result row are much more expensive than gfx1151 same-address LDS service.
   Twenty-one actual layer-1 natural-M512 pairs regress
   **4.5149 -> 6.3418 ms (+40.46%, 0/21 wins)**. The distinct HIP
   specialization, wrapper selector, test parameter, and harness mode were
   removed.
21. **Rejected and removed:** compacting the Q6 integer-WMMA shared weight
   record from **40 -> 36 bytes/column** by staging the source FP16 `d` and two
   int8 scales in one dword is BF16 exact, but reconstructing the combined
   FP32 scales grows the kernel body and regresses twenty-one actual layer-1
   natural-M512 pairs **4.5137 -> 4.8221 ms (+6.834%, 0/21 wins)**.
   The logical shared tile falls **5,120 -> 4,864 bytes**, but the hardware
   allocation remains rounded to **5,120 bytes**; local128, VGPR96, SGPR128,
   and scratch0 are unchanged. The candidate was removed. Q6-local metadata
   variants and single-stage pseudo-K64 loop unrolling are closed: adjacent
   planar K32 records are independent, so a loop-only K64 form cannot reuse
   quant bytes or remove either synchronization boundary. Resume this family
   only with a physical-byte or cross-tile reuse mechanism.
22. **Rejected and removed:** fuse the two resident-pack8 shared Q4 gate/up
   projections with SiLU while preserving both existing BF16 projection
   boundaries. The actual layer-1 M512xK3072xN1024 operation improves
   **0.501830 -> 0.428741 ms (-14.565%, 21/21 wins)** with zero BF16
   mismatches, local32/VGPR80/LDS0/scratch0, and complete 77.287-GB owner
   recovery. Production rejects it: seven complete-state-exact pp512 pairs
   move **580.394 -> 577.374 tok/s (-0.520%)**, add **4.088 ms** at the
   paired median wall, and win only **1/7**. The refreshed queue ledger
   explains the result: all **325.222 ms** of secondary-stream work is already
   nested inside the caller-stream span and ends **6.535 ms** early. Every
   candidate kernel, wrapper, registry, runtime-mode, and test surface is
   removed. Evidence:
   [`2026-07-26-gfx1151-laguna-shared-pack8-dual-silu-rejected.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-shared-pack8-dual-silu-rejected.json).
23. **Admitted gfx1151 production default:** use the complete
   dense-initial `KVLiveSpans` contract to widen resident BF16 K/V exactly,
   then run zero-workspace F32 hipBLASLt QK/PV around a causal F32 softmax.
   Start 0 remains on qrow4; partial, wrapped, explicitly evicted, verifier,
   decode, unsupported-head, and context-above-512 paths retain established
   fallbacks. Twenty-one samples at every qualified context improve global
   **0.3785/0.5869/0.8003 -> 0.2823/0.3453/0.4365 ms** and SWA
   **0.6195/1.0079/1.4014 -> 0.3626/0.4634/0.6015 ms**, all 21/21.
   Seven complete pp512 diagnostics improve **576.076 -> 602.518 tok/s**
   median with 6/7 wins and deterministic state per mode. The association
   change passes the long-shape distribution gate: pp512 all-exact KL
   improves **0.003246 -> 0.002214**, while top-1 remains 2930. The route
   owns **23,068,672 bytes** of scratch, uses no hipBLASLt workspace, and
   retains complete memory recovery. Clean selector-unset publication reaches
   **623.050/563.399/462.430 tok/s**, improving the previous production
   **7.907%/3.307%/0.590%**. Corrected cached tracing measures **82.763 ms**
   pp512 attention, down from **143.669 ms**. The next screen replicates the
   eight KV heads into query-head-major scratch so one QK and one PV
   strided-batch contraction replace sixteen smaller calls.
   Evidence:
   [`candidate`](../benchmarks/results/2026-07-26-gfx1151-laguna-attention-hipblaslt-candidate.json) ·
   [`production`](../benchmarks/results/2026-07-26-gfx1151-laguna-attention-hipblaslt-production.json).
24. **Rejected and removed:** replicate each widened KV head into
   query-head-major scratch so one 48/72-way QK and one PV contraction replace
   sixteen smaller calls. The CPU-reference route remains within **4.10e-8**
   absolute error, but scratch grows **23.1 -> 56.6 MB**. After sweeping all
   32 zero-workspace heuristics per contraction, the qualified 48-layer model
   regresses **75.380 -> 105.483 ms (+39.94%)** and loses every context
   256/384/512 sample; SWA context 512 is **73.02%** slower. Every candidate
   kernel, wrapper, route, and test surface is removed. The next formulation
   packs only the 4.7-MB query/output tiles and leaves K/V unreplicated.
   Evidence:
   [`2026-07-26-gfx1151-laguna-attention-replicated-heads-rejected.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-attention-replicated-heads-rejected.json).
25. **Admitted gfx1151 production default:** transpose only the
   4.7-MB F32 query/output tile into head-major order, leaving K/V
   unreplicated, so one eight-way wide QK and one wide PV batch replace
   sixteen calls. All 32 zero-workspace algorithms were screened per
   contraction. The qualified 48-layer leaf model improves **74.976 ->
   71.169 ms (-5.08%)**, with **21/21 wins** at every global/SWA context
   256/384/512 and at most **4.10e-8** absolute output error. Seven
   counter-rotated pp512 pairs improve **621.806 -> 627.217 tok/s (+0.870%,
   6/7 wins)** and save **7.416 ms** at the paired median. The wider F32
   association is quality-gated: all-exact KL improves **0.002214 ->
   0.002097**, production-vs-candidate KL is **0.000119**, and all top-1 IDs
   remain 2930. Scratch grows only **23.1 -> 27.8 MB**. gfx1151 now enables
   the capability. Clean selector-unset publication improves
   **623.050/563.399/462.430 -> 629.101/566.858/463.903 tok/s
   (+0.971%/+0.614%/+0.318%)**, with deterministic tokens, exact positions,
   and complete allocation recovery. Cached pp512 tracing measures attention
   **82.763 -> 73.330 ms (-11.40%)** and dispatches **4,145 -> 2,417**.
   The remaining 700 gap is **82.431 ms**; selected projection physical-byte
   or cross-tile scheduling work is next because attention is now fifth.
   Evidence:
   [`candidate`](../benchmarks/results/2026-07-26-gfx1151-laguna-attention-packed-query-candidate.json) ·
   [`production`](../benchmarks/results/2026-07-26-gfx1151-laguna-attention-packed-query-production.json).
26. **Rejected and removed:** keep production row32 tiles except when an
   expert ends in `32 + remainder`, replacing that pair with one row40 tile
   for remainders 1..8 or one row48 tile for remainders 9..16. Across all 47
   natural-M512 sparse layers this removes **1,246/14,034 tiles (-8.88%)**.
   The reduced grid is still slower: combined row40+row48 regresses M256
   **4.3543 -> 4.6782 ms (+7.44%)** and M512
   **6.6991 -> 7.0457 ms (+5.17%)**. Row40-only regresses
   **2.10%/1.66%** and row48-only **3.92%/1.09%** at M256/M512. The focused
   BF16 fixture is bit exact and actual-weight checksums agree, but the extra
   live accumulators and separate tail launches cost more than the avoided
   weight rereads. All candidate kernel, wrapper, harness, and test surfaces
   are removed. Production remains **629.101 tok/s**. Reopen intermediate
   row counts only with a mechanism that does not increase per-lane
   accumulator lifetime.
   Evidence:
   [`2026-07-26-gfx1151-laguna-q4-mixed-tail-rows-rejected.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q4-mixed-tail-rows-rejected.json).
27. **Admitted gfx1151 production default:** replace the
   dense-initial causal-softmax block reduction with one wave32 per score row.
   The established body launches local256 for each row, merges eight wave
   partials through LDS, and crosses four workgroup barriers. The candidate
   launches local32, uses wave shuffles only, and keeps the complete
   `KVLiveSpans` qualification, causal mask, F32 score ABI, exp/inverse
   operations, and packed-query QK/PV contractions. Screening one/two/four/
   eight independent rows per workgroup selects the simplest one-row policy.
   The qualified 48-layer packed-attention model improves
   **72.738 -> 62.755 ms (-13.73%)**. Seven complete pp512 pairs improve
   **614.668 -> 620.032 tok/s (+0.873%, 6/7 wins)** and save **7.206 ms** at
   the paired median wall. Reassociation is distribution-gated: all-exact KL
   improves **0.002097 -> 0.001796**, production-to-candidate KL is
   **0.0000971**, and all top-1 IDs remain 2930. Cached tracing names the
   retained kernel at local32/VGPR24/SGPR128/LDS0/scratch0. gfx1151 enables
   the capability with an explicit block256 rollback. Clean selector-unset
   512/1K/4K publication improves
   **629.101/566.858/463.903 -> 632.618/568.845/464.606 tok/s
   (+0.559%/+0.351%/+0.152%)** with deterministic tokens, exact positions,
   and complete allocation recovery. The refreshed trace keeps **2,417**
   dispatches and cuts pp512 attention **73.330 -> 69.983 ms (-4.56%)**.
   The 700 wall gap is now **77.907 ms**.
   Evidence:
   [`candidate`](../benchmarks/results/2026-07-26-gfx1151-laguna-attention-wave-softmax-candidate.json) ·
   [`production`](../benchmarks/results/2026-07-26-gfx1151-laguna-attention-wave-softmax-production.json).
28. **Rejected and removed:** use a 256-thread, 128-column x 64-row Q4
   gate/up tile so eight waves preserve production's 32 FP32
   accumulators/lane while serving twice the routed rows. The synthetic
   empty/uneven/65-row fixture is BF16-bit exact versus row32 and passes the
   CPU KL/top-1 gate. Natural routing defeats the premise before integration:
   row64 padding removes only **5.44%/16.84%** of M256/M512 tiles.
   Cooperative shared-weight reconstruction regresses the actual layer-1 leaf
   **4.377 -> 9.498 ms (+116.98%)** and
   **6.804 -> 13.836 ms (+103.34%)**. Retaining direct per-column decode
   avoids LDS reconstruction but still regresses
   **4.420 -> 5.671 ms (+28.31%)** and
   **6.902 -> 8.233 ms (+19.29%)**. Every candidate kernel, wrapper, harness
   mode, and test is removed. Reopen row64 only with a variable-row or
   persistent cross-tile mechanism that avoids both per-expert padding and
   local256 residency loss.
   Evidence:
   [`2026-07-26-gfx1151-laguna-q4-row64-local256-rejected.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q4-row64-local256-rejected.json).
29. **Rejected and removed:** keep the exact production
   128-column x 32-row/local128 body, but map one workgroup to one expert and
   serially walk that expert's existing row32 tiles. This avoids row64
   padding, local256 occupancy, larger accumulator state, and any sidecar
   while testing whether the second K3072 weight sweep can hit cache. The
   empty/uneven/33-row fixture is BF16-bit exact versus production and passes
   the CPU KL/top-1 gate. Twenty-one counter-rotated actual-weight samples
   reject it at both natural shapes: M256 regresses
   **4.395 -> 4.627 ms (+5.28%)** and M512
   **6.835 -> 7.268 ms (+6.33%)**. A complete 128-column K3072 sweep is too
   large to remain live for the next row tile, while serial expert tails
   reduce parallelism. Every candidate kernel, wrapper, harness mode, and
   test is removed. Do not retry row-outer persistence; any further
   cross-row design must keep each K tile live while multiple row tiles
   consume it or change the contraction architecture.
   Evidence:
   [`2026-07-26-gfx1151-laguna-q4-persistent-expert-rows-rejected.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q4-persistent-expert-rows-rejected.json).
30. **Rejected and removed:** use exact integer WMMA in a local128,
   64-column x 64-row gate/up tile. This is distinct from the rejected
   local256 row64 body: it preserves 32 FP32 accumulators/lane and keeps each
   staged K32 weight tile live across two row32 groups. The empty/uneven/
   33-row fixture is BF16-bit exact versus production and the complete
   12-case CPU-reference matrix passes. Full row64 padding regresses the
   actual layer-1 leaf **4.386 -> 8.391 ms (+91.34%)** at M256 and
   **6.807 -> 10.067 ms (+47.90%)** at M512. A padding-free split schedule
   then sends only complete row32 pairs through integer WMMA and every odd
   tail through production. It still regresses
   **4.428 -> 4.881 ms (+10.23%)** with 13 row64 pairs at M256 and
   **6.893 -> 6.937 ms (+0.64%)** with 50 pairs at M512. Integer-WMMA operand
   setup, synchronization, and the second launch consume all saved weight
   traffic. Every candidate kernel, wrapper, harness mode, and test is
   removed; no retained kernel trace is warranted. Do not retry this exact
   64x64 contraction.
   Evidence:
   [`2026-07-26-gfx1151-laguna-q4-integer-wmma-row64-rejected.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q4-integer-wmma-row64-rejected.json).
31. **Rejected and removed:** transfer one-scale-per-32 D4 arithmetic
   into production's proven 128-column x 32-row/local128 direct-wave,
   row-vector, activation-double-buffer body. This changes no resident bytes
   and keeps D8 production as rollback. The complete 12-case CPU-reference
   matrix passes. On actual layer-1 weights, producer-pack-inclusive timing is
   neutral at M128, then improves M256 **4.406 -> 4.225 ms (-4.09%)** and
   M512 **6.855 -> 6.008 ms (-12.36%)**. Five counterbalanced complete pp512
   diagnostics improve **631.251 -> 665.020 tok/s (+5.350%)** and save
   **41.187 ms** at the medians; every D4 sample is
   **663.143–668.584 tok/s**. Cached tracing observes the intended local128,
   16x297-workgroup specialization (rocprof thread grid 2048x297). This is
   not production. The clean direct-all-exact 320-step gate keeps strong
   **315/320 (98.438%)** top-1, and eight of ten prompts pass, but maximum KL
   reaches **0.127536** on `mixed_ja_en_translate`; the mixed category fails
   the 0.05 contract. Unqualified D4 therefore cannot ship. The allowed
   globally data-dependent per-K32 repair is also closed. A scale-ratio
   policy selecting D4 for **50.58%/78.68%/96.40%** of M512 K32 blocks
   regresses the pack-inclusive leaf from **6.8269 ms** D8 to
   **7.6757/7.6628/7.6677 ms (+12.24% to +12.43%)**. The uniform workgroup
   pays the selection and dual-arithmetic cost even when almost every block
   is D4. Every hybrid pack/consumer/test/harness surface was removed. The
   committed D4 export, runtime mode, leaf mode, absolute-quality lane, and
   focused tests were then removed as required by the refactor trigger.
   Evidence:
   [`2026-07-26-gfx1151-laguna-q4-d4-direct-wave-quality-pending.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q4-d4-direct-wave-quality-pending.json),
   [`2026-07-26-gfx1151-laguna-q4-d4-direct-wave-absolute-rejected.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q4-d4-direct-wave-absolute-rejected.json),
   [`2026-07-26-gfx1151-laguna-q4-d4-selective-repair-rejected.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q4-d4-selective-repair-rejected.json).
32. **Rejected and removed:** precompute D8's eight int8 half-block sums in
   the once-per-source-row pack instead of rebuilding them for every routed
   row tile. The exact 192-byte temporary block is BF16-bit identical to the
   160-byte production block. It improves the actual leaf only
   **0.38%/0.94%/0.83%** at M128/M256/M512. In the complete pp512 gate,
   production/candidate medians are **620.085/620.278 tok/s (+0.031%)**;
   after the first cold pair the candidate wins only **3/6** and saves a
   noise-level **0.339 ms** at the paired median. That is **0.44%** of the
   **77.907-ms** gap to 700, so no production selector or wider scratch ABI
   survives.
33. **Quality-pending candidate:** split gate and up by global projection role
   so each branchless kernel consumes one uniform activation format. On
   actual layer-1 natural M512 routing, D4-gate/D8-up improves the
   pack-inclusive leaf **6.8616 -> 6.6175 ms (-3.56%)**; D8-gate/D4-up reaches
   **6.6065 ms (-3.72%)**. Both model at least 11 ms across 47 layers.
   D4-gate/D8-up is the complete-wall leader after an exact separate-input
   fused SiLU/down-pack boundary: seven paired pp512 medians improve
   **617.519 -> 629.151 tok/s (+1.884%)**, saving **15.329 ms** with token
   2930 throughout. The fused and unfused candidate paths produce identical
   complete-state hashes. Cached tracing records the role body at local128,
   VGPR88, SGPR128, zero scratch and the exact fused pack at local128, VGPR16,
   512 B LDS, zero scratch. Production remains D8 at **632.618 tok/s** until
   the clean direct-all-exact 320-step category gate passes. Evidence:
   [`2026-07-27-gfx1151-laguna-q4-role-split-quality-pending.json`](../benchmarks/results/2026-07-27-gfx1151-laguna-q4-role-split-quality-pending.json).
34. **Rejected assignment:** D4 gate + D8 up improves the earlier all-D4
   maximum KL from **0.127536 to 0.061203** and keeps **317/320 (99.063%)**
   suite top-1, but still violates the absolute contract. The mixed-language
   prompts peak at **0.061203** and **0.053487**; all other prompts are within
   budget. Poolside remains exact top-1, category prefill is **4.432x**
   all-exact, decode is flat, and all tracked allocations return to zero.
   Production remains D8. The alternate D8-gate/D4-up role assignment has
   essentially identical wall economics and a different error path through
   SiLU, so it receives the same complete gate before producer-row repair.
   Evidence:
   [`2026-07-27-gfx1151-laguna-q4-gate-d4-up-d8-absolute-rejected.json`](../benchmarks/results/2026-07-27-gfx1151-laguna-q4-gate-d4-up-d8-absolute-rejected.json).
35. **Rejected assignment:** D8 gate + D4 up keeps **317/320 (99.063%)**
   suite top-1 and nine of ten prompts inside budget, but
   `mixed_ja_en_review` reaches max KL **0.203467** at step 1. Poolside remains
   exact top-1, category prefill is **4.442x** all-exact, decode is flat, and
   lifecycle accounting returns to zero. Projection-wide D4 is therefore
   closed in both roles. Evidence:
   [`2026-07-27-gfx1151-laguna-q4-gate-d8-up-d4-absolute-rejected.json`](../benchmarks/results/2026-07-27-gfx1151-laguna-q4-gate-d8-up-d4-absolute-rejected.json).
36. **Quality-pending shape candidate:** the new global matrix-row bucket keeps
   production D8 and the existing dual fused boundary below 512 rows, then
   selects D4-gate/D8-up and the exact separate-input fused boundary at
   M512+. Seven paired complete pp512 medians improve
   **619.782 -> 630.215 tok/s (+1.683%)**, saving **13.676 ms** with token
   2930 throughout. The selector adds no kernel, resident sidecar, or
   prompt/token/layer policy. The short-row Q4 production-shape GPU oracle is
   BF16-bit exact. Admission now requires both the ordinary short category
   no-change gate and a full-logit gate where every canonical prompt stream
   is deterministically extended to exactly 512 rows while attention remains
   tiled at 128. Evidence:
   [`2026-07-27-gfx1151-laguna-q4-m512-role-split-quality-pending.json`](../benchmarks/results/2026-07-27-gfx1151-laguna-q4-m512-role-split-quality-pending.json).
37. **Short no-change gate passed:** the clean 128-row category run reproduces
   production's admitted **316/320 (98.75%)** top-1 and max KL
   **0.049542582** exactly, with every category inside contract, Poolside
   exact top-1, deterministic repeats, and complete lifecycle recovery.
   Diagnostic prefill is **4.505x** all-exact and decode is flat. This proves
   the M512 selector is invisible below its threshold; it does not admit the
   accelerated branch. Evidence:
   [`2026-07-27-gfx1151-laguna-q4-m512-role-split-short-absolute-passed.json`](../benchmarks/results/2026-07-27-gfx1151-laguna-q4-m512-role-split-short-absolute-passed.json).
38. **Rejected M512-wide D4:** the extended-512 gate keeps
   **313/320 (97.813%)** top-1 but reaches max KL **1.379757**. Nine of ten
   streams exceed 0.05, spanning every category; category maxima are
   **1.379757/0.149638/0.878142/0.326543** for
   code/general-English/general-Japanese/mixed. The candidate is fast and
   general across streams at **628.591 tok/s**, **10.762x** all-exact, with
   flat decode and exact lifecycle recovery, but it cannot ship. Production
   remains D8 at **632.618 tok/s**. Evidence:
   [`2026-07-27-gfx1151-laguna-q4-m512-role-split-long-absolute-rejected.json`](../benchmarks/results/2026-07-27-gfx1151-laguna-q4-m512-role-split-long-absolute-rejected.json).
39. **Rejected selector removed:** the M512 runtime mode, its short/long
   comparison definitions, cumulative lane, and selector-specific tests are
   gone. The deterministic 512-token extension helper and shared D4/D8 role
   kernels survived only through the bounded producer-row repair screen below.
   Production dispatch is unchanged.
40. **Producer-row risk screen passed:** one fixed activation-only rule,
   `row_abs_max >= 2.0`, transfers from five category-balanced calibration
   prompts to five disjoint heldouts. It repairs **19.685%/19.724%** of
   layer-token rows while covering **99.764%/99.758%** of route-weighted SiLU
   error and **96.429%/97.010%** of the worst 1% rows. Each split covers
   **120,320** real producer rows; production D8 remains the authoritative
   model path and the comparisons run off-path. This clears the **<=25%**
   economic gate for a GPU sparse-repair candidate. Evidence:
   [`2026-07-27-gfx1151-laguna-q4-role-risk-calibration-heldout.json`](../benchmarks/results/2026-07-27-gfx1151-laguna-q4-role-risk-calibration-heldout.json).
41. **Sparse second-pass economics rejected:** the same real routing maps show
   that a **19.685%** producer-row repair rate expands to **30.266%** of active
   experts and **26.784%** of padded MMQ32 rows. Only **57.45% (135/235)** of
   layer/prompt pairs have no repair rows, while ten require every row. A
   second sparse weight pass therefore gives up too much of the original
   15-ms role-split saving. The retained opportunity is a whole-layer GPU
   gate: use specialized D4-gate/D8-up only when the layer has no risk rows,
   otherwise run production dual D8.
42. **Rejected and removed:** the whole-layer `any_absmax_ge_2` candidate
   saved **3.426 ms** in its seven-pair pp512 screen, but the clean extended
   M512 absolute gate reaches max KL **1.265492** despite
   **314/320 (98.125%)** top-1. Every category violates the 0.05 KL contract:
   **1.265492/0.212004/0.655027/0.293393** for
   code/general-English/general-Japanese/mixed. Candidate prefill is
   **617.423 tok/s**, Poolside remains exact top-1, and lifecycle accounting
   returns to zero, so this is a numerical rejection rather than a runtime
   failure. The layer selector, risk pack, conditional MMQ/SiLU packers, both
   projection-role modes, calibration harness, deterministic extension lane,
   and focused tests are removed. Per-row mixed arithmetic and one-grid
   uniform dynamic arithmetic were already removed after
   **5.594 vs 3.600 ms** and **593.700 -> 481.054 tok/s (-18.97%)**
   regressions. Activation-only D4/D8 projection-role repair is closed;
   production remains **632.618 tok/s**. Evidence:
   [`2026-07-27-gfx1151-laguna-q4-layer-risk-absolute-rejected.json`](../benchmarks/results/2026-07-27-gfx1151-laguna-q4-layer-risk-absolute-rejected.json).
43. **Published production:** the exact Q6 integer-WMMA
   selected-down body now register-prefetches the next planar-qmicro K32
   record and its `d`/scale metadata while the current K32 fragments execute.
   It reuses the existing **5,120-byte LDS** tile, adds no resident bytes or
   scratch, and preserves the current activation fragments, scaled K32
   accumulation, correction, and BF16 boundary. Twenty-one actual layer-1
   samples improve **4.518 -> 4.104 ms (-9.156%, 21/21 wins)** with zero
   BF16 mismatches. Seven complete pp512 pairs improve
   **618.294 -> 623.900 tok/s (+0.907%)** with identical token, logit bits,
   full logits, final/post-layer hidden, KV, and cursor in all 14 runs.
   Cached tracing reports local128/**VGPR104**/SGPR128/LDS5120B/scratch0
   versus VGPR96 for the rollback. gfx1151 enables it behind an explicit
   session rollback. Clean selector-unset pp512 improves
   **632.618 -> 636.073 tok/s (+0.546%)**; 1K/4K remain flat within
   **0.12%** at **568.765/464.061 tok/s**. The 23-call pp512 Q6 body falls
   **112.746 -> 101.963 ms (-9.564%)** in the refreshed cached trace.
   Evidence:
   [`candidate`](../benchmarks/results/2026-07-27-gfx1151-laguna-q6-wmma-weight-prefetch-candidate.json),
   [`production`](../benchmarks/results/2026-07-27-gfx1151-laguna-q6-wmma-weight-prefetch-production.json).
44. **Published production:** the same Q6 body now also carries each next
   compact Q8 activation half-row in registers while the current K32 WMMA
   executes. It publishes the exact bytes into the unchanged activation LDS
   tile on the next iteration. The actual leaf improves
   **4.104 -> 4.045 ms (-1.440%, 20/21 wins)** with zero BF16 mismatches;
   seven complete pp512 pairs improve
   **634.447 -> 637.752 tok/s (+0.521%, 5/7 wins)** with identical full
   state. Clean selector-unset 512/1K/4K improves
   **636.073/568.765/464.061 -> 639.114/569.880/464.280 tok/s**.
   Cached tracing reports local128/VGPR112/SGPR128/LDS5120B/scratch0 and cuts
   the 23-call Q6 window **101.963 -> 100.367 ms (-1.565%)**. Evidence:
   [`candidate`](../benchmarks/results/2026-07-27-gfx1151-laguna-q6-wmma-activation-prefetch-candidate.json),
   [`production`](../benchmarks/results/2026-07-27-gfx1151-laguna-q6-wmma-activation-prefetch-production.json).
45. **Rejected and removed:** timestamp-unioned attribution on the refreshed
   trace shows **339.630 ms** with both queues active, only **0.826 ms** with
   the secondary queue alone, and the secondary branch ending **6.038 ms**
   before pp512 completes. Its **257.508-ms** shared-SiLU inclusive sum is
   starvation/overlap, not an additive ceiling. An exhaustive 256-KiB device
   table removed scalar `expf` while preserving every BF16 gate encoding
   bit-for-bit, but isolated M512 regressed
   **0.021211 -> 0.021876 ms (+3.136%, 2/21 wins)** because the indexed global
   read costs more than the native exponential. All LUT surfaces are removed;
   shared SiLU remains closed. Evidence:
   [`2026-07-27-gfx1151-laguna-shared-silu-bf16-lut-rejected.json`](../benchmarks/results/2026-07-27-gfx1151-laguna-shared-silu-bf16-lut-rejected.json).
46. **Rejected and removed before integration:** put two recursively quantized
   D4 activation planes into production's 128-column x 32-row direct-wave
   body as a possible quality midpoint between the fast one-plane D4 gate and
   exact split16 D8. The production-geometry CPU-reference gate passes, but
   the second plane increases LDS and packed-dot work enough to regress the
   pack-inclusive actual leaf at every natural shape:
   **3.729 -> 4.107 ms (+10.12%)** at M128,
   **4.434 -> 5.864 ms (+32.25%)** at M256, and
   **6.882 -> 10.100 ms (+46.76%)** at M512. No absolute category run is
   warranted; all candidate surfaces are removed. Residual activation planes
   are closed unless an accompanying mechanism removes equivalent weight
   traffic. Evidence:
   [`2026-07-27-gfx1151-laguna-q4-d4x2-wave-direct-rejected.json`](../benchmarks/results/2026-07-27-gfx1151-laguna-q4-d4x2-wave-direct-rejected.json).
47. **Rejected and removed before integration:** expand qmicro's packed
   scale/min records cooperatively into LDS once per K256 inside production's
   direct-wave Q4 body. This is distinct from the old quartet-owned FP32-`dm`
   writer: each wave owns exactly the 32 columns it expands and reads, and the
   quant payload plus D8 arithmetic remain unchanged. The CPU-reference gate
   and actual BF16 identity pass. Fully unrolled expansion raises VGPR
   **88 -> 152** and regresses M256/M512 **19.80%/16.39%**. A deliberately
   rolled correction recovers VGPR to **120** but still grows LDS
   **3,072 -> 5,120 bytes** and regresses the paired leaf
   **4.394 -> 4.940 ms (+12.44%)** at M256 and
   **6.793 -> 7.385 ms (+8.71%)** at M512. The physical layout would save
   **25,165,824 bytes (2.778%)**, but its coefficient decode remains more
   expensive than those bytes. All candidate surfaces are removed; do not
   reopen packed Q4 metadata inside the current 32-accumulator direct-wave
   body without a mechanism that preserves the production VGPR class.
   Evidence:
   [`2026-07-27-gfx1151-laguna-q4-qmicro-metadata-lds-rejected.json`](../benchmarks/results/2026-07-27-gfx1151-laguna-q4-qmicro-metadata-lds-rejected.json).
48. **Published production:** carry only the next
   K32 interval's eight raw T16 nibble words in registers, decode them in
   place on the following interval, and demand-load `d`/scale/min metadata.
   This avoids the rejected complete decoded-record prefetch's VGPR104 cost:
   the new P8 body is local128/**VGPR96**/SGPR128/LDS3072B/scratch0 versus
   production VGPR88. It is BF16-identical and passes the uneven/empty-expert
   CPU-reference gate. Forty-one counter-rotated actual layer-1 samples show
   that P8 is shape-sensitive: M256 regresses
   **4.4213 -> 4.4306 ms (+0.211%)**, while M512 improves
   **6.8727 -> 6.7389 ms (-1.948%)**. Production therefore enables P8 only
   for producer chunks of at least 512 rows and keeps the previous body below
   that threshold. Seven complete pp512 pairs improve
   **636.367 -> 640.003 tok/s (+0.571%, 7/7 wins)** with exact token, logit
   bits, full logits, final/post-layer hidden, KV, and cursor. The gfx1151
   package default now selects this shape policy. Clean selector-unset
   512/1K/4K improves
   **639.114/569.880/464.280 -> 643.554/573.066/466.290 tok/s
   (+0.695%/+0.559%/+0.433%)**. The pp512 wall is **795.583 ms**, leaving
   **64.154 ms** to 700. Tokens are deterministic, final positions are exact,
   and all **78,805,563,028** tracked bytes are recovered. Evidence:
   [`candidate`](../benchmarks/results/2026-07-27-gfx1151-laguna-q4-raw-prefetch-p8-candidate.json),
   [`production`](../benchmarks/results/2026-07-27-gfx1151-laguna-q4-raw-prefetch-p8-production.json).
49. **Rejected and removed before integration:** extend retained P8 with the
   next K32 interval's `d/dmin` FP16 bits and scale/min bytes packed into two
   additional registers. The CPU-reference gate and actual BF16 identity
   pass, but the candidate restores the rejected full-prefetch resource class:
   VGPR rises **96 -> 104** with LDS3072B/scratch0 unchanged. Forty-one
   counter-rotated M512 samples regress
   **6.7265 -> 7.0330 ms (+4.556%)**. No full-model run is warranted and
   every metadata-prefetch surface is removed. Retained P8's successful
   payload-only register set is the ceiling for this one-interval schedule.
   Evidence:
   [`2026-07-27-gfx1151-laguna-q4-p8-metadata-prefetch-rejected.json`](../benchmarks/results/2026-07-27-gfx1151-laguna-q4-p8-metadata-prefetch-rejected.json).
50. **Rejected and removed before integration:** apply non-temporal loads only
   to retained P8's next-K32 raw nibble payload. The candidate is
   BF16-identical and keeps local128/VGPR96/SGPR128/LDS3072B/scratch0, so this
   isolates cache policy rather than register pressure. Forty-one
   counter-rotated M512 samples regress
   **6.5634 -> 6.9727 ms (+6.236%)**. The ordinary cache path is materially
   helping the mixed streamed-weight/reused-activation working set. No
   full-model run is warranted and every non-temporal surface is removed.
   Evidence:
   [`2026-07-27-gfx1151-laguna-q4-p8-nontemporal-rejected.json`](../benchmarks/results/2026-07-27-gfx1151-laguna-q4-p8-nontemporal-rejected.json).
51. **Rejected and removed before integration:** interleave two or four
   adjacent routed-row workgroups inside each output-column tile while keeping
   the exact production 128-column x 32-row/local128 D8 P8 body unchanged.
   This hybrid order targets L2 reuse without repeating the rejected
   64-row/local256 accumulator or LDS schedules. Both variants pass the
   uneven/empty-expert CPU-reference gate and are BF16-bit identical.
   Counter-rotated actual layer-1 timing rejects them at the primary M512
   shape: retained P8 **6.7168 ms**, row-group2 **6.7696 ms (+0.787%)**, and
   row-group4 **6.7332 ms (+0.245%)**. Row-group2 saves only **0.323%** at
   M256. Every candidate surface is removed; launch order alone remains
   closed without explicit shared residency or physical-byte reduction.
   Evidence:
   [`2026-07-27-gfx1151-laguna-q4-p8-rowgroup-order-rejected.json`](../benchmarks/results/2026-07-27-gfx1151-laguna-q4-p8-rowgroup-order-rejected.json).
52. **Admitted gfx1151 production default:** qualify the source-F16
   SWA-gate quality schedule by matrix rows. The complete category lane uses
   at most M128 and therefore keeps the admitted hipBLASLt heuristic 2 for
   K3072xN72 exactly as before. M512 now returns to the retained heuristic 4.
   The independent library screen measures **0.097302 -> 0.036308 ms** per
   M512 SWA gate, modeling **2.196 ms** across 36 layers. Six steady
   counter-rotated pp512 pairs, excluding the first explicitly cold pair,
   measure **797.030 -> 794.718 ms (-2.312 ms)** at the medians with **4/6**
   wins; all seven pairs give **5/7** wins. The M512 all-exact comparison
   remains finite at KL **0.00407713** with top-1 **2930**, while the
   descriptor fixture proves M128 still selects heuristic 2, so the existing
   320-step **0.049542582 / 316-of-320** category result is unchanged.
   Selection depends only on M/K/N, never prompt, token, category, or output.
   Clean selector-unset 512/1K/4K improves
   **643.141/573.717/466.913 -> 645.803/575.942/468.311 tok/s
   (+0.414%/+0.388%/+0.299%)**. pp512 wall falls **796.093 -> 792.811 ms**,
   leaving **61.383 ms** to 700; tokens, positions, repeats, and lifecycle
   recovery pass.
   Evidence:
   [`candidate`](../benchmarks/results/2026-07-27-gfx1151-laguna-f16-quality-row-schedule-candidate.json) ·
   [`production`](../benchmarks/results/2026-07-27-gfx1151-laguna-f16-quality-row-schedule-production.json).
53. **Refreshed production attribution:** clean revision `285b2638c`
   records **2,417** pp512 dispatches, **1,112.508 ms** inclusive kernel sum,
   and **853.021 ms** kernel span. The caller queue contains **781.331 ms**
   of work; the **331.178-ms** secondary queue is almost entirely overlapped.
   The main ceilings are selected Q4 gate/up **333.998 ms**, selected Q4/Q6
   down **170.295 ms**, source-F16 **122.924 ms**, attention **68.058 ms**,
   norm/RoPE/gate **26.331 ms**, and router **22.976 ms**. Row-qualified
   source-F16 falls another **1.744 ms** versus the prior trace. This confirms
   that launch/submission and the shared branch are not the remaining
   61.383-ms route to 700.
54. **Rejected and removed:** retain the first two M128 attention slices but
   merge rows 256..511 into one packed-query M256 x context512 hipBLASLt
   composite. The route passes the widened helper/CPU-reference gate, but its
   larger masked dense contraction regresses steady pp512 wall
   **792.662 -> 811.343 ms (+18.680 ms, -2.302% throughput)**. All 32
   zero-workspace QK and PV algorithms were screened for 48 and 72 query
   heads; the best four indices can recover only **0.783 ms** across the
   12 full and 36 SWA layers. The candidate code, selectors, widened scratch,
   and tests are removed. The naïve generic M256-online control is separately
   worse at **~504–506 tok/s** versus **~646–648** for M128 production.
   Attention-row widening is closed unless a fused causal library kernel
   avoids computing the masked upper triangle.
   Evidence:
   [`2026-07-27-gfx1151-laguna-production-trace-attention-m256-rejected.json`](../benchmarks/results/2026-07-27-gfx1151-laguna-production-trace-attention-m256-rejected.json).
55. **Rejected and removed before integration:** rebuild the 64-row Q4
   gate/up tile on the current direct-wave, activation-double-buffered,
   raw-prefetch-P8 body. This differs materially from the old 64-row screen,
   but it remains slower: all-expert M256/M512 move
   **4.3871 -> 5.0045 ms (+14.07%)** and
   **6.5282 -> 7.2574 ms (+11.17%)**. Restricting row64 to naturally dense
   experts does not rescue it. The best threshold (>=96 rows; only one M256
   and five M512 experts) still regresses **5.46%/6.07%** because the larger
   accumulator lifetime and second dispatch exceed the saved weight reads.
   The CPU-reference gate passes and candidate/production BF16 outputs match.
   All kernel, wrapper, fixture, and harness candidate surfaces are removed.
   Evidence:
   [`2026-07-27-gfx1151-laguna-q4-p8-row64-current-body-rejected.json`](../benchmarks/results/2026-07-27-gfx1151-laguna-q4-p8-row64-current-body-rejected.json).
56. **Rejected and removed before integration:** pair matching 128-column
   gate/up tiles in one local256 workgroup, preserve both BF16 projection
   boundaries in LDS, apply the exact BF16 SiLU boundary, and emit selected
   down D4 blocks directly. This removes the compact gate/up global tensor
   traffic and the standalone fused-SiLU pack launch, but the inclusive
   actual-weight leaf regresses M256 **4.4607 -> 4.8841 ms (+9.49%)** and
   M512 **6.9100 -> 7.4451 ms (+7.74%)**. Candidate and production D4 byte
   streams have identical SHA-256 at both shapes. Local256 residency,
   **19.5 KB LDS**, and cross-wave BF16 exchange cost more than the removed
   materialization. Every candidate surface is removed.
   Evidence:
   [`2026-07-27-gfx1151-laguna-q4-paired-silu-pack-rejected.json`](../benchmarks/results/2026-07-27-gfx1151-laguna-q4-paired-silu-pack-rejected.json).
57. **Rejected and removed before integration:** reduce retained raw-nibble
   prefetch from all eight next-K32 words to four and demand-load the other
   four. P4 is exact but loses to P8 at both actual-weight shapes:
   **4.4541 vs 4.4321 ms (+0.50%)** at M256 and
   **6.8635 vs 6.7625 ms (+1.49%)** at M512 across 41 counter-rotated
   samples. P8's complete payload coverage is earning its VGPR cost; partial
   coverage does not offer a better resource/latency balance. The P4 export,
   wrapper selector, fixture case, and harness mode are removed.
   Evidence:
   [`2026-07-27-gfx1151-laguna-q4-raw-prefetch-p4-rejected.json`](../benchmarks/results/2026-07-27-gfx1151-laguna-q4-raw-prefetch-p4-rejected.json).
58. **Rejected and removed before integration:** pair-share P8's next-K32
   raw-nibble gathers between adjacent output-column lanes. The candidate is
   exact, but eight wave shuffles cost more than the duplicated logical loads
   that already coalesce into the same memory transactions:
   **5.5939 vs 4.4182 ms (+26.61%)** at M256 and
   **8.1325 vs 6.6970 ms (+21.43%)** at M512 across 41 counter-rotated
   samples. The specialization, export, wrapper option, fixture case, and
   harness mode are removed.
   Evidence:
   [`2026-07-27-gfx1151-laguna-q4-p8-pair-shared-prefetch-rejected.json`](../benchmarks/results/2026-07-27-gfx1151-laguna-q4-p8-pair-shared-prefetch-rejected.json).

### Next exact and quality-gated attacks

The activation-only Q4 repair branch is exhausted and removed. Raw-nibble P8
prefetch is now exact Q4 gate/up production and has transferred successfully
to Q4 selected down:

1. **Trace complete:** clean selector-unset production remains
   **645.803/575.942/468.311 tok/s**. The refreshed pp512 trace attributes
   **504.292 ms** to selected gate/up plus down and leaves attention at
   **68.058 ms**.
2. Attack the selected-expert path with a mechanism
   that changes physical bytes, cross-tile reuse, or a measured
   synchronization/latency limiter. Payload-only P8 is retained; decoded,
   metadata-carrying, non-temporal, packed-metadata, pure axis-swap, and
   two/four-row-group launch-order schedules are now rejected. Direct row64
   is also closed on the current P8 body, including dense-expert partitioning;
   paired local256 gate/up-to-D4 fusion is closed as well. The next expert
   candidate must reduce weight bytes without extending accumulator lifetime,
   widening the workgroup, exchanging full results through LDS, or weakening
   P8's complete next-K32 payload coverage. Do not pair-share those payload
   gathers through wave shuffles; coalescing already removes the physical
   traffic duplication.
3. Do not widen the 128-row attention slice through another dense
   QK/softmax/PV formulation: M256 remains slower even after exhaustive
   library-algorithm tuning. Reopen attention only for a fused causal
   primitive that avoids the masked upper triangle.
4. Reopen any other closed family only if a future trace leaves a **>=5%**
   perfect-removal ceiling or a newly supported library algorithm changes a
   prior premise. No further activation-only D4 role policy is admissible
   without a new numerical representation.

The stretch target remains **>=700 tok/s**, i.e. **<=731.429 ms** for pp512.
Current production is **645.803 tok/s / 792.811 ms**, leaving **61.383 ms**.
The rejected D4 role split cannot contribute to that gap; reaching 700 now
requires a retained physical-byte, cross-tile-reuse, or newly enabled library
win.

Post-350 exclusions:

- do not spend a campaign round on source-F16, dense/shared, graphs,
  submission, router, norm/RoPE, or tails without a new trace or a newly
  supported grouped-contraction capability reopening them;
- do not retry the rejected raw-sum D8 or D4-gate quality shortcuts;
- do not add a duplicate resident expert-weight sidecar or weaken c=1 exact
  decode to buy prefill;
- do not retry 40/48/64-row Q4 gate/up accumulation or a row64/row32
  density split without a new mechanism that avoids the additional live
  accumulator and second-dispatch costs;
- do not retry paired local256 selected gate/up+SiLU packing without a
  mechanism that avoids its 19.5-KB LDS result exchange and residency loss;
- do not retry qrow3 attention without a mechanism that changes the SWA
  K/V-reuse or accumulator-cost tradeoff;
- do not retry Q4 gate/up grid-axis permutations without a cross-workgroup
  weight-sharing mechanism or physical cache-counter evidence;
- do not claim 500 or 700 from a leaf, explicit session selector, dirty tree,
  single sample, or incomplete quality lane.
- do not retry shared gate/up or gate/up+SiLU fusion from an isolated leaf
  win while the least-priority secondary stream remains fully hidden; require
  a current trace proving shared spill or caller-stream recovery first.

The current campaign authority is the retained production packet and trace
below. Every new modeled table is rebuilt from the most recently promoted
trace rather than the pre-campaign 76 tok/s bridge.

First post-350 screen: **rejected**. A BF16-bit-identical T16 K64/K128 staged
gate/up body amortized two workgroup barriers across two/four K32 intervals, but
multiplied LDS from 6,656 bytes to 13,312/26,624 bytes without eliminating a
resident-T16 weight read. A counterbalanced dirty-tree full-model diagnostic
measured K32/K64/K128 medians **353.516/318.850/269.071 tok/s**, always token
2930. The variants were removed. The raw-source K64 “both nibble planes from
one byte” lever does not transfer to T16: its resident payload stores K32
subblocks separately. Do not retry multi-K LDS staging unless a different
resident layout or asynchronous copy mechanism changes that premise.

Second post-350 screen: **rejected and removed**. At pp512, 64-row routing
would reduce the measured 47-layer tile count from **14,034 to 11,408
(-18.71%)**, but neither tested geometry converts that reduction into wall
time:

- 128x64 doubled accumulators from 32 to 64 per lane and measured
  **345.141 tok/s** versus **353.787 tok/s** production median
  (**-2.44%**);
- Vulkan-calibrated 64x64 restored 32 accumulators per lane but doubled
  output-column workgroups and increased repeated activation loading; it
  measured **344.606 tok/s** versus **354.693 tok/s** production median
  (**-2.84%**).

Each result is a three-repeat, counterbalanced, same-resident-load pp512
diagnostic on gfx1151 with matrix512/attention128, one queue, and token 2930 in
every run. Both kernels passed the uneven/empty-expert CPU-reference
KL/top-1 fixture before the full-model rejection. Production code and metadata
remain unchanged. Do not retry a 64-row tile without a mechanism that avoids
both extra per-lane accumulators and repeated activation reads.

Third post-350 screen: **rejected and removed**. A 256x32/local256 D8 gate/up
body kept 32 accumulators per lane and halved workgroups plus activation-tile
reloads, but increased the weight LDS tile from 5,120 to 10,240 bytes and
doubled workgroup residency granularity. The same-load three-repeat pp512
diagnostic measured **350.813 tok/s** versus **353.380 tok/s** production
median (**-0.73%**), with token 2930 in every sample. Its CPU-reference
KL/top-1 fixture passed before the full-model screen. The specialization,
selector, and widened-only test fixture were removed. The production
128x32/local128 occupancy remains the stronger schedule.

Fourth post-350 screen: **rejected and removed**. The unchanged
128x32/local128 body coalesced each resident-T16 K32 quant payload into a
3,072-byte raw-nibble-plus-FP32-metadata stage instead of the 5,120-byte
expanded weight cache. Per-lane unpack then reconstructed the identical eight
packed operands before dot work. The CPU-reference gate passed, but the
same-load three-repeat pp512 diagnostic measured **314.082 tok/s** versus
**344.866 tok/s** production median (**-8.93%**), always token 2930. The
scalar nibble reconstruction cost dominates the cleaner global access and
smaller LDS allocation. The candidate was removed. Do not revisit raw LDS
staging without a wave-transpose/unpack primitive that avoids per-lane scalar
reconstruction.

Fifth post-350 screen: **retained production**.
The online global and SWA kernels now share each streamed BF16 K/V row across
four adjacent queries on complete 128-row attention tiles; short and residual
tiles retain qrow2. The wrapped/evicted 508..515 fixture, including a
seven-row partial group, is F32 byte-identical between qrow4 and qrow2. Cached
gfx1151 tracing names the expected qrow4 global/SWA templates at local32,
VGPR **72/80**, SGPR128, and zero LDS/scratch; qrow2 remains the residual path
because its VGPR **48/56** footprint wins the eight-row fixture.

The one-load, three-repeat, counterbalanced explicit screen measured qrow4
global+SWA at **365.249 tok/s** median
(**365.249/366.556/364.684**) versus qrow2 production at **353.836**
(**353.836/353.437/353.926**), always token 2930: **+3.23%**. SWA-only
qrow4 reached **363.214**, while global-only was neutral at **353.722**.
After committing the M128-qualified gfx1151 defaults, the clean
selector-unset confirmation measured **364.839 tok/s** median
(**365.309/364.839/363.944**) versus the paired qrow2 **353.181**, again
always token 2930: **+3.30%**. The cached all-family trace measures
**366.260/339.178/282.939 tok/s** at 512/1K/4K. Qrow4 cuts global/SWA
attention **46.736/227.989 -> 43.577/185.603 ms**, saving **45.544 ms**
and reducing the combined attention share from **19.25% to 16.59%**; kernel
sum falls **45.676 ms**. Evidence:
[`2026-07-25-gfx1151-laguna-attention-qrow4-candidate.json`](../benchmarks/results/2026-07-25-gfx1151-laguna-attention-qrow4-candidate.json).
Production:
[`2026-07-25-gfx1151-laguna-prefill-qrow4-production.json`](../benchmarks/results/2026-07-25-gfx1151-laguna-prefill-qrow4-production.json).

Sixth post-350 screen: **rejected and removed**. Reusing each global K/V row
across eight queries is
F32 byte-identical to qrow2 on the wrapped/evicted full-eight and seven-row
partial fixture. The final cached trace names global qrow8 at local32,
VGPR112, SGPR128, and zero LDS/scratch. A five-repeat matched pp512 screen
measured **366.126 tok/s** median versus qrow4 production **365.471**
(**+0.179%**), always token 2930, but the clean committed gate reversed that
signal: selector-unset qrow8 measured **361.055** versus qrow4 **363.475
tok/s (-0.666%)**. The analogous SWA qrow8 route measured
**349.177** versus **365.392 tok/s** and was removed; SWA stays qrow4.
Global qrow8 is now removed as well; qrow4 remains production and the topline
stays **364.839 tok/s**. Evidence:
[`2026-07-25-gfx1151-laguna-global-qrow8-candidate.json`](../benchmarks/results/2026-07-25-gfx1151-laguna-global-qrow8-candidate.json).

Seventh post-350 screen: **rejected and removed before integration**. Across
the frozen natural pp512 routing, 1,931 experts above 32 rows carry 147,237
lanes; routing only those experts through 64-row tiles would reduce their tile
count **5,728 -> 3,102 (-45.8%)**, while 8,306 small experts remain on
MMQ128x32. The explicit hybrid was BF16 byte-identical on mixed
0/7/18/33/65-row expert fixtures. On actual layer-1 K3072/N1024 gate/up
weights and natural M512 routing, however, pack-inclusive production measured
**12.332 ms** median and the hybrid **13.179 ms (+6.87%)**. The larger
accumulator footprint plus a second filtered launch outweigh the saved tiles.
All candidate surfaces were removed. Evidence:
[`2026-07-25-gfx1151-laguna-hybrid64-expert-rejected.json`](../benchmarks/results/2026-07-25-gfx1151-laguna-hybrid64-expert-rejected.json).

Eighth post-350 screen: **rejected and removed**. A qrow4 SWA workgroup placed
three wave32 query heads from the same qgroup9 KV head together, cutting
workgroups per row tile **72 -> 24** and sharing K/V through LDS. The exact
K8/float-LDS form passed the full-eight and odd-seven wrap/eviction fixture
byte-for-byte, but measured **298.652 tok/s** versus **364.738 tok/s**
production (**-18.1%**) across five counterbalanced pp512 repetitions. A
K32/BF16-LDS follow-up cut barrier frequency 4x and LDS bytes per value in
half, yet fell further to **256.697** versus **364.943 tok/s (-29.7%)**.
Every run selected token 2930. Cross-wave barriers and LDS occupancy outweigh
the 3x K/V load reduction; all candidate C/Python/registry/runtime/test
surfaces were removed. This does not reject a true key-parallel online tile,
but it closes cross-wave GQA row sharing with synchronous LDS staging.
Evidence:
[`2026-07-25-gfx1151-laguna-swa-qhead3-rejected.json`](../benchmarks/results/2026-07-25-gfx1151-laguna-swa-qhead3-rejected.json).

Ninth post-350 screen: **retained production**. The
single-wave qrow4 SWA body now qualifies current/cache K/V loads after
visibility is known. Current-chunk logical slots no longer fetch cached K/V
when every visible row uses current K/V; prior slots do not fetch current K/V.
Dot, online-softmax, PV, and output order are unchanged. Full-eight and
odd-seven wrap/eviction outputs are F32 byte-identical to production qrow4,
and the 33-test attention/backend bundle passes. Cached tracing names
`laguna_swa_attention_prefill_qrows_online_bf16_kernel<4, true>` at local32,
VGPR80, SGPR128, LDS0, and scratch0.

The initial one-load five-pair pp512 screen measures **368.531 tok/s** median
(minimum **367.010**) versus qrow4 **365.584** (maximum **366.503**):
**+0.806%**, always token 2930. The gfx1151 M128 selector now uses the
qualified body while residual tiles retain qrow2. At clean committed revision
`36b318ac9`, selector-unset production measures **366.933 tok/s** median
versus explicit old qrow4 **364.753 (+0.598%)**, always token 2930. Cached
all-family tracing measures **369.532/342.620/285.563 tok/s** at 512/1K/4K
and cuts SWA **185.603 -> 173.749 ms (-6.39%)**, combined attention
**229.181 -> 217.249 ms (-5.21%)**, and kernel sum **11.818 ms**.
Evidence:
[`2026-07-25-gfx1151-laguna-swa-sourcequal-candidate.json`](../benchmarks/results/2026-07-25-gfx1151-laguna-swa-sourcequal-candidate.json).
Production:
[`2026-07-25-gfx1151-laguna-swa-sourcequal-production.json`](../benchmarks/results/2026-07-25-gfx1151-laguna-swa-sourcequal-production.json).

Tenth post-350 screen: **retained production**. The D8 MMQ128x32 gate/up
consumer now assigns one thread to each routed activation row, reads
`compact_to_source` once per K32 interval, and stages the row through two
aligned 16-byte loads instead of reconstructing eight int32 packs byte-by-byte
across the workgroup. Resident T16 weights, D8 bytes/FP32 metadata, weight
decode, packed dots, accumulation order, and BF16 output are unchanged. The
uneven/empty-expert fixture is BF16 byte-identical to old D8 and passes every
CPU-reference D4/D8 configuration. Cached leaf tracing records local128,
VGPR80, SGPR128, 6,656 B LDS, zero scratch, and
**264.416 -> 226.144 us**.

The dirty one-load screen measured **368.450 -> 379.661 tok/s (+3.043%)**.
After commit `bd76e452d`, the clean five-pair selector-unset gate measured
old D8 **368.203** versus row-vector **379.811 tok/s (+3.153%)**, with every
candidate sample above every baseline sample and token 2930 throughout.
Cached all-family tracing measures **381.448/351.663/292.417 tok/s** at
512/1K/4K and cuts selected gate/up **581.061 -> 537.923 ms (-7.42%)**;
kernel sum falls **1,369.727 -> 1,326.263 ms (-3.17%)**. The new template
boolean also exposed and repaired a suffix-sensitive trace-classifier bug;
that repair changes attribution only. Evidence:
[`2026-07-25-gfx1151-laguna-gate-rowvec-candidate.json`](../benchmarks/results/2026-07-25-gfx1151-laguna-gate-rowvec-candidate.json).
Production:
[`2026-07-25-gfx1151-laguna-gate-rowvec-production.json`](../benchmarks/results/2026-07-25-gfx1151-laguna-gate-rowvec-production.json).

Eleventh post-350 screen: **retained production**. The same row-vector
activation stage now covers compact D4 Q4 and Q6 down independently. Both
consumers preserve D4 metadata, resident T16 weight decode, packed-dot and
accumulation order, and BF16 output. Q4 dual/single and Q6
uneven/empty-expert fixtures are BF16 byte-identical to scalar staging; the
production-shape synthetic MoE is also byte-identical.

The one-load five-pair actual-model screen measures old **381.211**, Q4-only
**384.594 (+0.888%)**, Q6-only **382.981 (+0.464%)**, and combined
**386.612 tok/s (+1.417%)**, with every combined sample above every baseline
sample and token 2930 throughout. Cached pp512 tracing names Q4
`<1, true, false, 64, true>` and Q6 `<1, true>`, cuts them
**139.554 -> 126.972 ms (-9.02%)** and
**132.467 -> 122.312 ms (-7.67%)**, and records local128/LDS4096B/scratch0
with VGPR56/72. gfx1151 now selects only the combined mode; the temporary
quant-scoped runtime selectors are removed.

At clean committed revision `69cc0d369`, the five-pair gate measures scalar
down **379.827** versus selector-unset row-vector down **385.997 tok/s
(+1.625%)**, with complete sample separation and token 2930 throughout. This
is **+1.629%** over the prior published production. Cached all-family tracing
measures **388.014/358.319/296.060 tok/s** at 512/1K/4K, cuts selected down
**276.556 -> 254.006 ms (-8.15%)**, and cuts kernel sum
**1,326.263 -> 1,304.061 ms (-1.67%)**. Evidence:
[`2026-07-25-gfx1151-laguna-down-rowvec-candidate.json`](../benchmarks/results/2026-07-25-gfx1151-laguna-down-rowvec-candidate.json).
Production:
[`2026-07-25-gfx1151-laguna-down-rowvec-production.json`](../benchmarks/results/2026-07-25-gfx1151-laguna-down-rowvec-production.json).

Twelfth post-350 screen: **rejected and removed**. A true contiguous-key
split was applied inside one source-qualified qrow4 workgroup while preserving
one K/V read per token across the four query rows. Each wave produced a local
online max, denominator, and 128-dimensional PV state; the workgroup merged
those states in split order through LDS. The wrap/eviction oracle passes at
`rtol=2e-5, atol=2e-6`.

Four key waves regress paired pp512 **385.998 -> 379.597 tok/s (-1.658%)**;
two waves regress **386.075 -> 377.219 (-2.294%)**. All runs select token 2930.
The traced four-way kernel is local128, VGPR88, SGPR128, LDS8704B, scratch0.
The extra waves, two barriers, partial-PV LDS, and merge arithmetic outweigh
the parallel key ranges. All code/registry/runtime/test surfaces are removed.
This closes scalar qrow4 state splitting, not the M16xK64 tiled-QK/PV premise.
Evidence:
[`2026-07-25-gfx1151-laguna-swa-keysplit-rejected.json`](../benchmarks/results/2026-07-25-gfx1151-laguna-swa-keysplit-rejected.json).

Thirteenth post-350 screen: **rejected and removed**. A true tiled-WMMA SWA
body used four wave32 BF16 WMMA waves for a 16-query x 64-key QK tile, shared
each staged K/V tile across adjacent queries, and accumulated cooperative
online-softmax/PV state. The 16/15-row 500..515 wrap/eviction oracle passed at
`rtol=2e-5, atol=2e-6`. It also exposed two correctness landmines that are now
recorded for any future tiled body: mixed current/cache slot indices require an
explicit cross-wave phase barrier, and logically invalid cache payload must be
sanitized before branch-free zero-weight PV because `0 * NaN` is NaN.

The fully correct M16 body traced at local128, VGPR248, SGPR128, LDS50,688B,
scratch0 and regressed paired pp512 **386.631 -> 370.586 tok/s (-4.150%)**.
An M8 pre-wrap specialization retained the proven qrow4 fallback at/after ring
wrap, reused its K LDS allocation for V to cut LDS to 22,016B, and reduced
VGPR to 224. It still regressed **386.539 -> 352.446 (-8.820%)** because it
doubled workgroups and wasted half of each 16-row WMMA query tile. Every correct
full-model run selected token 2930. All C/Python/registry/runtime/test surfaces
were removed. Synchronous-LDS tiled attention is closed until a different
async-copy, supported-library, or fused-softmax premise changes the resource
model. Evidence:
[`2026-07-25-gfx1151-laguna-swa-wmma-tiled-rejected.json`](../benchmarks/results/2026-07-25-gfx1151-laguna-swa-wmma-tiled-rejected.json).

Fourteenth post-350 screen: **rejected and removed before integration**. A
persistent D8 gate/up kernel assigned one local128 workgroup to each active
expert/output128 tile, staged all eight decoded K32 weight tiles for one K256
slab, and processed the expert's 32-row tiles sequentially. This preserved the
production split16 packed-dot and K order; F32 partial outputs carried state
between K256 slabs. The 0/7/18/65-row K512/N128 primitive was BF16
byte-identical to production.

The body traced at VGPR248, SGPR128, LDS42,496B, scratch0 and requires
**40 MiB** of F32 partial workspace at the actual pp512 leaf. Running it for
all active experts costs **37.547 ms** versus **11.463 ms** production. More
decisively, restricting it to experts above 32 rows still costs **13.278 ms**,
already **16.14% slower** than the complete **11.433 ms** production leaf
before adding the required small-expert launch. The candidate was removed
without a full-model screen. Do not retry persistent K256 slabs while
accumulation requires global partial spills. Evidence:
[`2026-07-25-gfx1151-laguna-persistent-expert-rejected.json`](../benchmarks/results/2026-07-25-gfx1151-laguna-persistent-expert-rejected.json).

Fifteenth post-350 screen: **neutral and removed before integration**. The hot
D8 body stores each decoded output column as a 40-byte LDS record: eight packed
quant words followed by two FP32 metadata values. A structure-of-arrays
specialization made each wave's same-plane loads contiguous while preserving
global bytes, arithmetic, and K order. The focused 0/7/18/33-row K512/N128
oracle was BF16 byte-identical to production.

On the actual layer-1 pp512 leaf, 31 counter-rotated pack-inclusive samples move
only **10.709 -> 10.696 ms (-0.124%)**. The candidate traces with exactly the
production resource footprint: local128, VGPR80, SGPR128, LDS6656B, scratch0.
That is noise-scale and cannot materially move the 537 ms all-layer family, so
all candidate surfaces were removed. The next expert body must reduce global
decode/load work or change the wave-level consume schedule, not just rearrange
the current LDS record. Evidence:
[`2026-07-25-gfx1151-laguna-weight-soa-rejected.json`](../benchmarks/results/2026-07-25-gfx1151-laguna-weight-soa-rejected.json).

Sixteenth post-350 screen: **neutral and removed before integration**. The D8
body rereads each column's FP16 T16 `d` and `dmin` base on every K32 subblock.
An exact specialization retained the metadata tile pointer and both bases
across all eight subblocks of a K256 slab, removing an estimated 3,584 bytes
per output128/K256 slab while leaving the quant payload, scaled metadata
arithmetic, packed dots, and K order unchanged. ISA inspection confirms the
base loads moved behind the subblock-zero path.

The focused oracle is BF16 byte-identical, but 31 counter-rotated actual-weight
samples move **11.443 -> 11.446 ms (+0.027%)**; means differ by only -0.082%.
The candidate again traces at local128, VGPR80, SGPR128, LDS6656B, scratch0.
The invariant bases are evidently cache-resident and not limiting. All
candidate surfaces were removed. Evidence:
[`2026-07-25-gfx1151-laguna-weight-meta-hoist-rejected.json`](../benchmarks/results/2026-07-25-gfx1151-laguna-weight-meta-hoist-rejected.json).

Seventeenth post-350 screen: **rejected and removed before integration**.
Experts with one through eight rows are numerous—**3,906/10,237 (38.16%)**
active pp512 expert groups and **27.83%** of MMQ32 tiles across 47 layers—but
contain only 6.42% of routed rows. An exact local32 output128 x rows8
specialization kept the production T16 LDS decode and packed-dot order while
assigning all live rows to its single wave. The hybrid packed activations once,
ran production rows32 for experts at or above nine rows, and ran local32 for
the small experts; the extra launch was included.

After repairing the candidate's initially incomplete column-metadata load, the
0/3/7/8-row K512/N128 CPU quality fixture passed. The actual layer-1 hybrid
still regressed **11.463 -> 13.195 ms (+15.106%)**. Tracing explains why:
local32 serializes output128 weight-cache population and compiles at VGPR224,
SGPR128, LDS5632B, scratch0, versus production VGPR80. Removing three idle
compute waves cannot repay that loader/register cost. All candidate surfaces
were removed. Evidence:
[`2026-07-25-gfx1151-laguna-small8-hybrid-rejected.json`](../benchmarks/results/2026-07-25-gfx1151-laguna-small8-hybrid-rejected.json).

Eighteenth post-350 screen: **rejected and removed before integration**. The
earlier 128x64 body doubled each lane's accumulator footprint; this distinct
follow-up used local256 so eight wave32 row groups covered 64 rows while each
lane retained the production **32 accumulators** and the weight LDS tile stayed
at 128 columns. The 0/7/18/33-row CPU-reference quality fixture passed, as did
all six existing 32-row configurations after templating.

All-expert 64-row padding regressed the actual layer-1 pp512 leaf
**11.440 -> 12.840 ms (+12.23%)**. The decisive hybrid kept production
128x32 for experts at or below 32 rows and used local256 128x64 only above
32; one D8 pack and both launches measured **11.437 -> 11.819 ms (+3.34%)**
across nine counter-rotated burst-three samples. Production and candidate
outputs were finite with identical BF16 checksum. All diagnostic HIP, wrapper,
harness, and test surfaces were removed. The 64-row route is now closed for
both accumulator mappings; reopen it only with a premise that also avoids the
local256/second-launch cost. Evidence:
[`2026-07-26-gfx1151-laguna-mmq128x64-t256-rejected.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-mmq128x64-t256-rejected.json).

Nineteenth post-350 screen: **retained production**. The wave-column remap
assigns each of four wave32 groups 32 output columns and all 32 routed rows.
Each lane retains **32 accumulators**;
an even lane decodes one adjacent T16 column pair, a wave shuffle distributes
the high-nibble column, and decoded weights remain in registers. This removes
the 5,120-byte shared weight cache without changing D8 activation staging,
packed-dot arithmetic, K accumulation order, resident T16 bytes, or output
boundaries.

The uneven/empty-expert fixture is BF16 byte-identical to row-vector
production and passes the independent CPU-reference gate. The actual layer-1
natural pp512 leaf improves **11.467 -> 8.086 ms (1.418x; -29.49%)**,
including the D8 pack. Seven counterbalanced full-model repetitions improve
**385.941 -> 433.380 tok/s (+12.29%)**, with complete sample separation and
token 2930 in every run. Cached tracing names
`<1,false,true,128,true,true>` at local128, VGPR80, SGPR128, **1,536 B LDS**,
and scratch0 versus row-vector production's 6,656 B LDS. Clean
selector-unset publication improves the row-vector rollback
**385.602 -> 432.355 tok/s (+12.125%)** across seven counterbalanced
repetitions; candidate samples are **431.106–433.943**, all token 2930.
Direct all-exact quality is unchanged at maximum KL **0.049542582** and
**316/320** top-1, with neutral decode, deterministic repeats, Poolside,
lifecycle, and exact allocation recovery all passing. Cached all-family
tracing independently measures **434.994/397.128/323.536 tok/s** at
512/1K/4K and cuts selected gate/up to **388.719 ms / 33.49%** of pp512
kernel sum. gfx1151 selects wave-column production; the old row-vector body
remains explicit rollback through the next retained checkpoint. Evidence:
[`2026-07-26-gfx1151-laguna-wavecols-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-wavecols-production.json)
and the implementation-worktree
[`candidate packet`](../benchmarks/results/2026-07-26-gfx1151-laguna-wavecols-candidate.json).

Twentieth post-350 screen: **Q4 retained production; Q6 rejected**. The
64-column transfer uses two wave32s, each owning 32 output columns and all 32
routed rows. Q4 pair decode/shuffle removes its decoded-weight LDS tile while
preserving D4 row-vector activation staging, packed dots, K order, resident
T16 bytes, and BF16 outputs. The Q4 body moves
local128/VGPR56/LDS4096B to local64/VGPR80/LDS1536B with zero scratch.

The quant-isolated actual-model gate is decisive. Across seven
counterbalanced repetitions per mode, row-vector production measures
**433.791 tok/s**, Q4-wave/Q6-row **448.945 (+3.493%)**, Q4-row/Q6-wave
**428.184 (-1.293%)**, and both-wave **442.941 (+2.109%)**. Every run returns
token 2930; the Q4/Q6 primitive candidates are independently BF16
byte-identical and pass their CPU-reference gates. gfx1151 therefore selects
Q4-only `mmq64x32_d4_f32_wavecols_q4`; Q6 remains row-vector, and its
quartet-shuffle runtime routes are removed.

Clean committed publication confirms all-row-vector rollback
**433.081 -> 448.203 tok/s (+3.492%)** across seven counterbalanced
repetitions with complete sample separation and token 2930. Direct all-exact
quality remains maximum KL **0.049542582**, **316/320** top-1, minimum
category agreement **96.875%**, neutral h16/h32 decode, deterministic repeats,
Poolside exact top-1, and exact lifecycle/allocation recovery. Cached tracing
independently measures **449.522/409.990/332.286 tok/s** at 512/1K/4K and
cuts selected down **257.747 -> 216.616 ms (-15.96%)**. Q4 is
local64/VGPR80/LDS1536B/scratch0; retained Q6 is
local128/VGPR72/LDS4096B/scratch0. Evidence:
[`2026-07-26-gfx1151-laguna-down-wavecols-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-down-wavecols-production.json)
and the implementation-worktree
[`candidate packet`](../benchmarks/results/2026-07-26-gfx1151-laguna-down-wavecols-candidate.json).

Twenty-first post-350 screen: **alternate gate/up wave-column geometries
rejected and removed**. The exact local64 variant assigned two waves 64 output
columns total; the exact 256x32 variant kept local128 but assigned two columns
and 64 accumulators to every lane. Both preserved T16 bytes, D8 activation
staging, packed-dot/K order, and BF16 outputs.

On actual layer-1 weights and natural M512 routing, nine counter-rotated
burst-three samples measure production 128x32 **8.048 ms**, local64 64x32
**8.087 ms (+0.486%)**, and two-columns-per-lane 256x32
**9.702 ms (+20.550%)**. Cached tracing shows local64 provides no register
relief at VGPR80, while the wide tile rises to VGPR128; all three use 1,536 B
LDS and zero scratch. The production 128x32/local128 geometry is retained and
all candidate HIP/wrapper/harness/test surfaces are removed. Evidence:
[`2026-07-26-gfx1151-laguna-gate-wavecols-geometry-rejected.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-gate-wavecols-geometry-rejected.json).

Twenty-second post-350 screen: **non-temporal T16 weight loads rejected and
removed**. A separate production-geometry export applied
`__builtin_nontemporal_load` only to the streamed T16 Q4 quant and metadata
loads. Extracted gfx1151 ISA proves this was not a no-op: all 32
`global_load_u8` quant loads and both `global_load_d16_b16` metadata loads
gain `slc dlc`, while activation/routing loads, the 13,704-byte kernel body,
and arithmetic remain unchanged.

The focused CPU-reference bundle passes all 13 cases, and both actual-layer
paths produce the same finite BF16 checksum **1114.1769413301445**. Nine
counter-rotated burst-three natural-M512 samples nevertheless regress
production **7.811 -> 10.355 ms (+32.584%)**. Bypassing the default cache
policy is therefore actively harmful for this resident-T16 access pattern.
All diagnostic HIP, wrapper, harness, and test surfaces were removed. Evidence:
[`2026-07-26-gfx1151-laguna-gate-wavecols-nontemporal-rejected.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-gate-wavecols-nontemporal-rejected.json).

A Q6-specific follow-up reaches the same decision on the current dominant
selected-down leaf. Applying the hint only to the three aligned qmicro quant
record loads remains BF16-byte exact, but actual layer-1 natural-M512 timing
regresses **5.066 -> 5.173 ms (+2.121%)** across eleven counter-rotated
burst-five samples. The candidate is fully removed. This closes non-temporal
loads for both selected Q4 gate/up and selected Q6 down unless a future
counter trace first demonstrates a new cache-pollution limiter. Evidence:
[`2026-07-26-gfx1151-laguna-q6-qmicro-nontemporal-rejected.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q6-qmicro-nontemporal-rejected.json).

Twenty-third post-350 screen: **Q6 local128 row-half wave mappings rejected
and removed**. Four wave32s retained the production 16 accumulators per lane:
waves 0/1 covered columns 0-31/32-63 for rows 0-15, and waves 2/3 repeated
those column halves for rows 16-31. One variant retained quartet decode plus
wave shuffles; the other decoded each lane's column directly. Both removed the
4,096-byte shared weight cache while necessarily decoding each streamed Q6
weight tile twice.

The six-case CPU-reference gate passes and both candidates are BF16-byte
identical to row-vector production. In a one-owner, seven-repetition
matrix512/attention128 pp512 screen with retained Q4 wave columns unchanged,
production measures **447.756 tok/s**. Row-half quartet/shuffle falls to
**411.122 (-8.182%)**; direct per-column decode improves that result but still
lands at **434.797 (-2.894%)**. Every run selects token 2930. Candidate text
also grows from production's 8,372 bytes to 14,008/11,128 bytes. All kernel,
wrapper, runtime, and test surfaces were removed. Evidence:
[`2026-07-26-gfx1151-laguna-q6-row-half-wavecols-rejected.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q6-row-half-wavecols-rejected.json).

Twenty-fourth post-350 screen: **direct per-column Q4 gate/up decode retained
as the next default candidate**. The production wave-column body made each
even lane decode an adjacent T16 column pair and shuffled the second column to
its odd neighbor. The candidate instead has every lane decode its own column.
It preserves the 128x32/local128 geometry, D8 activation bytes, resident T16
layout, packed-dot arithmetic, K accumulation order, and BF16 stores; only the
decode ownership changes.

The nine-case Q4 CPU-reference gate passes and the candidate is BF16-byte
identical to pair-decode production. Nine counter-rotated burst-three samples
on actual layer-1 weights and natural M512 routing improve the pack-inclusive
leaf **8.107 -> 6.916 ms (-14.693%)**, with identical finite checksum
**1114.1769413301445**. A seven-repeat one-owner matrix512/attention128 screen
then improves integrated pp512 **447.582 -> 472.533 tok/s (+5.575%)** with
complete sample separation and token 2930 throughout. Cached tracing names
template `<1,false,true,128,true,true,128,true>` at local128, VGPR88,
LDS1536B, and zero scratch. Its **13,416-byte** text is 752 bytes smaller than
pair decode in the same object. At this checkpoint it remained a
candidate—not a production claim—until committed clean selector-unset timing,
direct all-exact quality, lifecycle, and refreshed all-family tracing
completed. Candidate evidence:
[`2026-07-26-gfx1151-laguna-q4-direct-wavecols-candidate.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q4-direct-wavecols-candidate.json).

Clean publication is now complete. Seven counterbalanced repetitions improve
the explicit pair-decode rollback **449.020 -> 474.363 tok/s (+5.644%)**;
all selector-unset samples are **471.774–476.132**, completely separated from
rollback, and select token 2930. The direct all-exact lane passes at maximum KL
**0.049542582**, **316/320** top-1, minimum category agreement **96.875%**,
neutral h16/h32 decode, deterministic repeats, Poolside exact top-1, and exact
lifecycle/allocation recovery. Cached tracing measures
**475.267/429.785/343.453 tok/s** at 512/1K/4K, cuts gate/up
**389.893 -> 317.722 ms (-18.51%)**, and leaves only **53.3 ms** of traced
pp512 wall to the 500 milestone. Evidence:
[`2026-07-26-gfx1151-laguna-q4-direct-wavecols-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q4-direct-wavecols-production.json).

Twenty-fifth post-350 screen: **direct per-column Q4-down decode retained in
production**. The 64x32/local64 Q4 down body now gives each
lane ownership of its resident-T16 column instead of having even lanes decode
adjacent pairs and shuffle the second column. The D4 activation stage,
resident layout, packed-dot arithmetic, K order, BF16 stores, and Q6
row-vector path are unchanged.

All ten Q4 primitive configurations pass the CPU-reference gate, the direct
single-Q4 body is BF16-byte identical to pair-decode wave columns, and the
production-shape Q4/Q6 runtime oracle remains byte-exact. With the retained
direct Q4 gate/up default fixed, seven counterbalanced one-owner
matrix512/attention128 repetitions improve Q4-down pair decode
**473.774 -> 483.409 tok/s (+2.033%)**. Every direct sample
**478.856–486.240** exceeds every pair-decode sample, and every run selects
token 2930. Cached tracing names
`<1,true,false,64,true,true,64,true>` at local64, VGPR88, LDS1536B, and zero
scratch.

Clean publication is complete at revision `d39cbb5ba`. Seven
counterbalanced repetitions improve explicit Q4 pair-decode rollback
**473.963 -> 480.629 tok/s (+1.406%)**; every selector-unset sample
**477.298–485.019** exceeds every rollback sample and selects token 2930. The
direct all-exact lane passes at maximum KL **0.049542582**, **316/320**
top-1, minimum category agreement **96.875%**, neutral h16/h32 decode,
deterministic repeats, Poolside exact top-1, and exact lifecycle recovery.
Cached tracing measures **481.997/435.961/346.675 tok/s** at 512/1K/4K,
cuts the Q4-down consumer **90.280 -> 71.378 ms (-20.94%)**, and leaves
**34.5 ms** of pp512 kernel span to the 500 milestone. Production evidence:
[`2026-07-26-gfx1151-laguna-q4-down-direct-wavecols-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q4-down-direct-wavecols-production.json).
Candidate evidence:
[`2026-07-26-gfx1151-laguna-q4-down-direct-wavecols-candidate.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q4-down-direct-wavecols-candidate.json).

Twenty-sixth post-350 screen: **direct 64x32/local64 gate/up rejected and
removed**. The exact body improves the actual layer-1 natural-M512 leaf
**6.920 -> 6.839 ms (-1.17%)**, but seven counterbalanced one-owner pp512
repetitions move only **481.323 -> 481.619 tok/s (+0.061%)**. Candidate and
production ranges overlap, and the candidate owns the lowest sample at
**475.974 tok/s**. All outputs are BF16-byte exact and every run selects token
2930, but there is no system-level separation; production remains
128x32/local128. The required lineage command also remains blocked by the
absent read-only Atlas checkout, with no external source copied. Evidence:
[`2026-07-26-gfx1151-laguna-gate-direct-local64-rejected.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-gate-direct-local64-rejected.json).

Twenty-seventh post-350 screen: **direct 256x32/local256 gate/up rejected and
removed**. Eight waves each own one output column and all 32 routed rows, so
the exact body halves workgroups and repeated activation staging without the
64-accumulator pressure of the earlier two-columns-per-lane mapping. Actual
layer-1 natural-M512 pack-inclusive time nevertheless regresses
**6.868 -> 7.181 ms (+4.559%)**, and all nine counter-rotated samples lose.
The checksum remains exactly **1114.1769413301445**. The screen stopped before
runtime integration and every candidate surface was removed; production
remains 128x32/local128. Evidence:
[`2026-07-26-gfx1151-laguna-gate-direct-local256-rejected.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-gate-direct-local256-rejected.json).

Twenty-eighth post-350 screen: **Q6 dense/shared 16x32 retained as a candidate
default**. The refreshed trace exposed the production 64x16 kernel at VGPR256
with **236 B/thread scratch**. The exact 16x32 schedule traces at
local32/VGPR136/LDS0/scratch0 and remains BF16-byte identical across all six
supported tiles on actual weights. The precise pp512 call mix is 23
M512/K1024/N3072 shared-down calls plus one M512/K12288/N3072 layer-0 down
call—not the transposed K3072/N1024 shape in the prior queue text. Their leaf
medians fall **0.942 -> 0.306 ms/call** and **10.629 -> 3.616 ms**,
respectively, a call-weighted **32.301 -> 10.660 ms (-67.00%)**.

Seven dirty-tree one-owner repetitions improve explicit 64x16 rollback
**480.727 -> 488.513 tok/s (+1.620%)** with complete sample separation; all
runs select token 2930. The candidate is default with
`HIPENGINE_GGUF_Q6_K_DENSE_WMMA_TILE=64x16` as rollback, but remains pending a
clean selector-unset publication and refreshed all-family trace. Evidence:
[`2026-07-26-gfx1151-laguna-q6-dense-wmma16x32-candidate.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q6-dense-wmma16x32-candidate.json).

Clean publication is complete at revision `c4e2fbd1d`. Seven counterbalanced
repetitions improve explicit 64x16 rollback **481.950 -> 490.096 tok/s
(+1.690%)**; every selector-unset sample **488.107–494.702** exceeds rollback
**479.521–483.686**, and every run selects token 2930. All 24 actual Q6
dense/shared projection weights have zero BF16 mismatches, so the direct
all-exact maximum KL **0.049542582**, **316/320** top-1, decode, determinism,
Poolside, and lifecycle gates transfer unchanged.

Cached tracing measures **491.171/441.091/351.095 tok/s** at 512/1K/4K,
reduces dense/shared **72.866 -> 54.834 ms (-24.75%)**, and reduces Q6 alone
**29.248 -> 11.131 ms (-61.94%)**. The production 16x32 symbol is
local32/VGPR136/LDS0/scratch0; rollback was VGPR256 with 236 B/thread scratch.
Only **13.6 ms** of traced kernel span and about **20.7 ms** of clean median
wall remain to 500. Production evidence:
[`2026-07-26-gfx1151-laguna-q6-dense-wmma16x32-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q6-dense-wmma16x32-production.json).

Twenty-ninth post-350 screen: **Q4 pack8 per-shape WMMA tiles retained as a
candidate default**. The real mix is 94 M512/K3072/N1024 shared gate/up calls,
24 M512/K1024/N3072 shared-down calls, and two M512/K3072/N12288 layer-0
gate/up calls. Nine counter-rotated burst-three samples across all six exact
tiles keep the first shape at 64x16, select 64x32 for shared down, and select
32x32 for layer 0. The call-weighted leaf window falls **34.782 -> 33.031 ms
(-5.03%)**.

All six tiles are BF16-byte identical on each screened actual weight, and a
direct candidate-versus-64x16 pass across all 120 resident Q4 projections
reports zero mismatches. Dirty-tree one-owner pp512 improves **489.036 ->
491.014 tok/s (+0.404%)**; the candidate wins six of seven paired repetitions
and every run selects token 2930, but samples overlap. The exact micro-win is
therefore retained under the gfx1151 four-axis registry with gfx1100 unchanged
and `HIPENGINE_GGUF_Q4_K_DENSE_WMMA_TILE=64x16` as rollback. Clean
selector-unset publication and a refreshed family trace remain required.
Evidence:
[`2026-07-26-gfx1151-laguna-q4-pack8-shape-policy-candidate.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q4-pack8-shape-policy-candidate.json).

Clean publication is complete at revision `3c1e5b452`. Matched pp512 improves
explicit 64x16 **488.692 -> 489.922 tok/s (+0.252%)** with four of seven
paired wins and token 2930 throughout. The distributions overlap and the
absolute median is **0.036%** below the prior 490.096 publication, so the
system wall is flat within noise. Cached tracing provides the retainable
attribution: **492.717/442.555/351.533 tok/s** at 512/1K/4K, Q4 dense
**43.702 -> 41.936 ms (-4.04%)**, and total dense/shared
**54.834 -> 52.989 ms (-3.36%)**. All 120 actual Q4 outputs remain
byte-identical, so the direct all-exact maximum KL **0.049542582** and
**316/320** top-1 transfer unchanged. Production evidence:
[`2026-07-26-gfx1151-laguna-q4-pack8-shape-policy-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q4-pack8-shape-policy-production.json).

Thirtieth post-350 screen: **Q6 shared-weight local64 rejected and removed**.
The candidate kept the production 64-column/32-row tile and one 4 KiB LDS
weight decode, but assigned two waves 16 rows each instead of four waves eight
rows each. It is therefore distinct from the already-closed row-half variants
that duplicated streamed weight decode. The uneven/empty-expert oracle and
actual layer output are BF16-byte exact.

Actual layer-1 natural-M512 timing nevertheless regresses **5.223 -> 5.308 ms
(+1.635%)** across nine counter-rotated burst-three samples. Doubling
accumulators per lane and making each thread fill two weight-cache entries
costs more than the smaller workgroup saves. Every candidate surface was
removed before runtime integration; Q6 local128 remains production. Evidence:
[`2026-07-26-gfx1151-laguna-q6-down-shared-weight-local64-rejected.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q6-down-shared-weight-local64-rejected.json).

Thirty-first post-350 screen: **Q6 64-row selected down retained in
production**. Unlike the rejected local64 schedule, this body keeps
local128 and one shared 64-column weight decode, but gives each of four
wave32s 16 routed rows. A registry-backed tile64 device map rebuilds metadata
only for Q6 down after Q4 gate/up has consumed its 32-row map; Q4 down remains
on the retained 64x32 direct wave-column body.

On actual layer-1 Q6 weights and natural M512 routing, the runtime upper-bound
grid falls **408 -> 332 workgroups per output tile (-18.63%)**. Nine
counter-rotated burst-three samples improve **5.260 -> 5.161 ms (-1.879%)**
with zero BF16 mismatches. Seven dirty-tree one-owner pp512 repetitions improve
the explicit 32-row rollback **490.105 -> 491.335 tok/s (+0.251%)**, all
token 2930. Cached tracing names the intended `<1,true,false,128,64>` body at
local128/VGPR88/LDS5632B/scratch0; across the 23 full-M512 Q6 calls it cuts
**127.888 -> 126.040 ms** despite the added tile64 map. The exact candidate is
default in the implementation tree, with the prior
`mmq64x32_d4_f32_wavecols_direct_q4` mode retained as rollback.

Clean committed publication at `f9a39715b` improves the explicit 32-row
rollback **489.110 -> 492.640 tok/s (+0.722%)**. The candidate wins all seven
paired repetitions, reduces median wall **7.501 ms**, and selects token 2930
throughout. The cached all-family trace independently reaches
**493.509/443.214/351.871 tok/s** at 512/1K/4K; pp512 wall/span/kernel sum are
**1,037.468/1,033.496/1,021.905 ms**. Absolute quality remains
**0.049542582** maximum KL and **316/320** top-1 by BF16-byte-exact transfer.
Evidence:

- [`candidate`](../benchmarks/results/2026-07-26-gfx1151-laguna-q6-down-rows64-candidate.json)
- [`production`](../benchmarks/results/2026-07-26-gfx1151-laguna-q6-down-rows64-production.json)

Thirty-second post-350 screen: **stable parallel MoE compaction retained in
production**. The prior one-workgroup metadata kernel scanned all 5,120 routed
lanes twice per layer and consumed **16.752 ms** across pp512. The replacement
uses one workgroup per expert for counts, one exact prefix stage, and
wave-ballot ranks to scatter every expert's lanes in ascending source order.
It adds no caller-visible scratch and leaves gfx1100 plus explicit serial
rollback unchanged.

The M512/top10/E256 metadata leaf improves **0.348880 -> 0.058969 ms
(-83.10%)** with starts, active IDs/count, lanes, source rows, and weights all
exact. Complete production-shape MoE output is BF16-byte identical. A clean
seven-repeat one-owner pp512 A/B improves serial rollback **490.824 -> 497.408
tok/s (+1.341%)**, wins all seven pairs, reduces median wall **13.808 ms**, and
selects token 2930 throughout. Cached tracing independently reaches
**500.325/449.468/355.606 tok/s** at 512/1K/4K; pp512 wall/span/kernel sum are
**1,023.336/1,018.444/1,006.892 ms**, and parallel count/prefix/scatter total
**2.564 ms**. The 500 gate remains open because it requires at least three
clean samples with both minimum and median at or above 500.

Absolute quality remains **0.049542582** maximum KL and **316/320** top-1 by
byte-exact transfer. Evidence:
[`2026-07-26-gfx1151-laguna-parallel-compact-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-parallel-compact-production.json).

Thirty-third post-350 screen: **one-block parallel prefix retained**. The
parallel compactor's remaining one-thread loop over 256 expert counts was
replaced by a Blelloch exclusive scan plus ballot active-ID compaction.
Production-shape metadata and complete MoE BF16 output remain byte-exact.
Cached tracing cuts prefix **32.34 -> 2.404 us/layer**, projecting
**1.407 ms** pp512 savings without caller-visible scratch. Evidence:
[`2026-07-26-gfx1151-laguna-parallel-prefix-scan.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-parallel-prefix-scan.json).

Thirty-fourth post-350 screen: **eight-token router-logit reuse retained in
production; 500 gate closed**. The wider workgroup preserves every
token/expert K traversal, per-thread products, 256-thread reduction tree, and
F32 store. Production router logits, selected IDs, scaled routing weights, and
complete MoE BF16 output are byte-exact. The M512 leaf improves
**0.583252 -> 0.434974 ms (1.341x)**.

Clean committed seven-pair pp512 improves explicit tile-4 rollback
**497.625 -> 503.349 tok/s (+1.150%)**, wins every pair, reduces median wall
**11.701 ms**, selects token 2930 throughout, and keeps every production sample
above 500 (**minimum 501.698 tok/s**). Cached all-family tracing independently
measures **504.631/452.733/357.083 tok/s** at 512/1K/4K and cuts router
**30.658 -> 23.315 ms**. Absolute quality remains **0.049542582** maximum KL
and **316/320** top-1 by exact transfer. Evidence:
[`2026-07-26-gfx1151-laguna-router-token-tile8-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-router-token-tile8-production.json).

Thirty-fifth post-350 screen: **nine-wave GQA K/V sharing rejected and
removed before integration**. One 288-thread workgroup assigned a wave32 to
each of the nine query heads sharing one KV head. Every token8 step staged
both current and cached K/V in 16 KiB of LDS, preserving ring-wrap source
qualification without increasing each wave's qrow4 online-softmax state.
The rows-8 and odd-7 wrap oracle was F32-bit exact to retained source-qualified
qrow4, and tracked allocations returned to zero.

Counterbalanced eleven-sample production-shape leaf medians regress at every
128-row pp512 slice: **0.360 -> 0.417 ms** at position 0,
**0.898 -> 1.246 ms** at 128, **1.423 -> 2.091 ms** at 256, and
**1.951 -> 2.803 ms** at 384. The four-slice sum is
**4.633 -> 6.557 ms (0.706x)**. Reduced global K/V reads do not repay the
288-thread barriers, four-way current/cache LDS traffic, and occupancy cost.
All candidate code, dispatch, test, and harness surfaces were removed.
Evidence:
[`2026-07-26-gfx1151-laguna-swa-gqa-tiled-rejected.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-swa-gqa-tiled-rejected.json).

Thirty-sixth post-350 screen: **DPP adjacent-pair T16 decode rejected and
removed before integration**. The candidate revisited the old pair decoder
with a new lane-transfer mechanism: even lanes decoded both nibbles from each
packed T16 byte once, and odd lanes received the adjacent column through a
row-shift-right-one DPP instruction instead of eight generic shuffles. It
retained production's activation double buffer, resident layout, D8 bytes,
packed dots, FP32 K order, and BF16 boundary.

The uneven/empty expert oracle is BF16-byte exact, and actual layer-1 natural
M512 gate/up output has zero BF16 mismatches versus production. Eleven
counter-rotated burst-five medians nevertheless regress
**6.727 -> 8.255 ms (+22.7%; 0.815x throughput)**, with no candidate wins.
The dependent DPP chain costs more than duplicated adjacent packed-byte loads.
All candidate code, wrapper, test, and harness surfaces were removed.
Evidence:
[`2026-07-26-gfx1151-laguna-gate-dpp-pair-decode-rejected.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-gate-dpp-pair-decode-rejected.json).

Thirty-seventh post-350 screen: **Q6 K64 synchronization staging rejected and
removed**. The clean cached-attention trace splits selected down into Q6
**126.594 ms**, Q4 **72.358 ms**, and activation packing **4.970 ms**. Q6 is
therefore the larger remaining down target.

The candidate kept production's 64-column/64-row/local128 geometry and staged
two ordered K32 weight/activation slices before each barrier, preserving the
established K32 dot and FP32 accumulation sequence while halving
synchronization intervals. The uneven/empty-expert CPU-reference quality gate
passes, and every full-model run selects token 2930. The larger live stage is
decisively negative: VGPR rises **88 -> 128**, LDS doubles
**5,632 -> 11,264 B**, and traced Q6 regresses
**126.254 -> 144.607 ms (+14.54%)**. Three counter-rotated pp512 pairs regress
**528.123 -> 518.568 tok/s (-1.81%, 0/3 wins)**. All kernel, wrapper, runtime,
test, and harness candidate surfaces were removed. Evidence:
[`2026-07-26-gfx1151-laguna-q6-down-k64-stage-rejected.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q6-down-k64-stage-rejected.json).

Thirty-eighth post-350 screen: **Q6 paired-scale metadata decode neutral and
removed**. Production uses two threads per output column to load the same FP16
block multiplier and one adjacent int8 scale each. The candidate used one
thread per column to load the multiplier once and compute both scales. It kept
the resident bytes, quant decode, packed dots, FP32 K order, and BF16 boundary
unchanged.

The uneven/empty-expert oracle is BF16-byte exact. Five counter-rotated pp512
pairs are noise at **529.210 -> 529.334 tok/s (+0.023%, 3/5 wins)**, and cached
tracing moves Q6 slightly backward **126.899 -> 126.947 ms (+0.038%)** while
both bodies remain local128/VGPR88/LDS5632B/scratch0. Therefore scale metadata
traffic is not the limiter. All kernel, wrapper, runtime, test, and harness
candidate surfaces were removed. The next Q6 premise attacks its twelve
scattered packed-quant loads per work item with a byte-neutral contiguous
resident micro-layout. Evidence:
[`2026-07-26-gfx1151-laguna-q6-down-paired-scales-rejected.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q6-down-paired-scales-rejected.json).

Q6 qmicro implementation checkpoint: **retained in gfx1151 production**. The
CPU materializer/inverse stores
the unchanged 288-byte `d/scales` metadata followed by records ordered
`[K32][col4][K4][QL8,QH4]`. Each selected-prefill work item therefore owns one
aligned 12-byte record instead of twelve scattered byte addresses. The
transform is bit-lossless and remains exactly **3,360 bytes** per
16-column/K256 tile, equal to legacy T16 and raw Q6_K.

The direct, grouped-small-M, and MMQ consumers are BF16-byte exact. An
11-sample, counter-rotated actual-weight gate on layer 1 measures natural-M512
selected prefill **5.1564 -> 5.0714 ms (-1.65%)** and top-10 exact decode
**0.0910 -> 0.0846 ms (-6.99%)**. Cached tracing observes the intended
`QMICRO=true` prefill body at local128/VGPR88/LDS5,632B/scratch0 and reduces
direct-decode VGPR **96 -> 88**. gfx1151 converts only sparse
`ffn_down_exps` payloads after reading the existing legacy cache, so there is
no cache rebuild, byte growth, duplicate sidecar, or root-lm-head change.
gfx1100 and unmeasured backends remain legacy. Clean committed
selector-unset 512/1K/4K reaches **530.447/473.118/381.375 tok/s**, improving
the prior production packet by **0.759%/1.127%/0.918%**. Full tracing cuts Q6
**126.594 -> 123.473 ms (-2.465%)**, total selected down
**203.923 -> 200.510 ms (-1.673%)**, and kernel sum
**947.513 -> 941.469 ms (-0.638%)**.
Evidence:
[`production`](../benchmarks/results/2026-07-26-gfx1151-laguna-q6-qmicro-production.json) ·
[`leaf`](../benchmarks/results/2026-07-26-gfx1151-laguna-q6-qmicro-candidate.json).

Thirty-ninth post-350 screen: **exact cached-metadata qrow4 policy retained as
the gfx1151 default candidate**. The planned three-wave GQA follow-up
was closed during source audit rather than implemented: it would combine the
already-rejected serial multi-head register growth with already-rejected
cross-wave synchronous sharing. The distinct retained premise starts after
preappend, when current K/V and visibility metadata are already complete.
Global and SWA candidates derive visibility only from `KVLiveSpans`, removing
current-vs-cache source bookkeeping while preserving the ordered qrow4 dot,
wave32 reduction, online-softmax/PV order, and every F32 output bit.

Eleven-sample, burst-25, four-mode counter-rotated leaf timing covers pp512
positions 0/128/256/384. SWA improves **1.128/1.113/1.110/1.108x**. Global
regresses **0.897x** at position 0 but improves
**1.010/1.040/1.052x** thereafter, so the integration policy keeps position 0
on the existing cached body. The qualified 12-full/36-SWA leaf model improves
**14.6024 -> 13.3230 ms (1.096x)**, projecting **15.353 ms** pp512 saving.
Cached tracing names global `<4,true,true>` and SWA `<4,true,true,true>` at
local32/VGPR64/SGPR128/LDS0/scratch0. Qualified runtime integration selects SWA
for every safe pre-wrap M128 tile and global only from position 128. Seven
alternating one-owner full-model pairs improve source-qualified rollback
**533.507 -> 542.785 tok/s (+1.739%)**, all seven pairs win, and median wall
falls **959.688 -> 943.283 ms**, a measured **16.405-ms** saving. All fourteen
runs have identical logits, final/post-layer hidden state, KV, next token/logit,
and cursor. The affected backend/runner/attention bundles report **52 passed**.
Clean selector-unset 512/1K/4K reaches **542.088/478.856/387.725 tok/s**,
with every pp512 sample above 542. Tracing observes 12 global start-0,
36 global cached-metadata, and 144 SWA cached-metadata calls; combined attention
falls **175.802 -> 160.123 ms (-8.92%, 15.679 ms saved)**. Evidence:
[`2026-07-26-gfx1151-laguna-attention-cached-meta-candidate.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-attention-cached-meta-candidate.json) ·
[`2026-07-26-gfx1151-laguna-attention-cached-meta-default.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-attention-cached-meta-default.json) ·
[`2026-07-26-gfx1151-laguna-attention-cached-meta-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-attention-cached-meta-production.json).

Fortieth post-350 screen: **Q6 local256 workgroup widening rejected and
removed before integration**. The candidate kept production's byte-neutral
qmicro bytes, 64-column/64-row tile, one shared weight decode, 5,632-byte LDS
footprint, and ordered K32 arithmetic. It assigned eight wave32 row groups
instead of four, reducing each lane's F32 accumulator count from 32 to 16.
This is distinct from the rejected local64 and K64-stage premises.

The uneven/empty-expert CPU-reference gate passes, and the actual layer-1
natural-M512 output is BF16-byte identical to local128. Eleven counter-rotated
burst-five samples nevertheless regress **5.0602 -> 5.9237 ms (+17.07%,
0/11 wins)**. Tracing shows local256 lowers VGPR **88 -> 72**, keeps
LDS **5,632 B** and scratch zero, yet remains slower; workgroup widening does
not produce useful additional latency hiding for this body. The HIP export,
Python selector, test parameter, and harness mode were removed. Production
remains local128. Evidence:
[`2026-07-26-gfx1151-laguna-q6-down-local256-rejected.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q6-down-local256-rejected.json).

Forty-first post-350 screen: **Q6 128-column barrier amortization rejected and
removed before integration**. This distinct local256 body kept production's
64-row ownership and 32 F32 accumulators per lane, but doubled output ownership
from 64 to 128 columns. It therefore halved output workgroups and reused each
activation stage across twice the columns while preserving byte-neutral qmicro
weights, ordered K32 arithmetic, and the BF16 boundary.

The CPU-reference gate passes and actual layer-1 output is BF16-byte identical
to production. Eleven counter-rotated burst-five samples regress
**5.0672 -> 5.3894 ms (+6.36%, 0/11 wins)**. Tracing keeps VGPR at 88 and
scratch at zero, but local256 plus the doubled shared weight tile raises LDS
**5,632 -> 8,192 B**. The activation/barrier amortization does not repay that
schedule. All candidate surfaces were removed; the 64-column/local128 body
remains production. Evidence:
[`2026-07-26-gfx1151-laguna-q6-down-cols128-rejected.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q6-down-cols128-rejected.json).

Forty-second post-350 screen: **Q4-down 128-column direct-wave widening
rejected and removed before integration**. The candidate reused the already
proven gate/up template for the single Q4-down ABI: four wave32s owned 128
columns and all 32 rows, versus production's two waves and 64 columns. It
halved output workgroups and activation staging while keeping 32 accumulators
per lane, direct register-resident weight decode, D4 activation bytes, K order,
BF16 stores, VGPR88, LDS1,536B, and scratch zero.

The uneven/empty-expert CPU-reference gate and actual layer-6 byte comparison
pass. Eleven counter-rotated burst-five samples nevertheless regress
**2.9716 -> 3.0188 ms (+1.59%, 2/11 wins)**. The candidate is close but
negative, so every candidate surface was removed and production remains
64-column/local64. Evidence:
[`2026-07-26-gfx1151-laguna-q4-down-cols128-rejected.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q4-down-cols128-rejected.json).

Forty-third post-350 screen: **Q6 grid and launch-bounds scheduler controls
rejected and removed**. Two remaining no-math-change premises were measured
before changing architecture. First, the production body launched the runtime
upper grid of 332 row tiles and used `-1` sentinels for the 85 entries above
layer 1's actual 247 tiles. Eleven counter-rotated samples are timing-equivalent
to the exact grid at **5.0896 -> 5.0785 ms (-0.22%)**: empty sentinel
workgroups return cheaply, so host grid construction is not material.

Second, the exact production 64-column/64-row qmicro body changed only from
`__launch_bounds__(128, 1)` to `(128, 2)`. The CPU-reference gate and actual
BF16 byte comparison pass, and the leaf reports a nominal
**5.0759 -> 5.0635 ms (-0.24%, 7/11 wins)**. Cached tracing, however, emits
identical local128/VGPR88/SGPR128/LDS5,632B/scratch0 resources and launch
geometry; its isolated candidate call is slightly slower at
**5.204 -> 5.254 ms**. The compiler hint did not change the machine schedule,
so the sub-quarter-percent delta is noise. Both harness modes and every lb2
candidate surface were removed. Production remains **542.088 tok/s**.
Evidence:
[`2026-07-26-gfx1151-laguna-q6-down-scheduler-controls-rejected.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q6-down-scheduler-controls-rejected.json).

Forty-fourth post-350 screen: **exact MMQ grouped-combine reuse retained as the
default candidate**. The active MMQ route used the exact sorted-lane weighted
sum before launching the shared expert, materialized a 512x3072 BF16
`routed_output`, then launched a separate BF16 add. The already registered
grouped-combine composite preserves that boundary exactly—ten slot-order F32
FMAs, selected BF16 rounding, shared BF16 add, final BF16 rounding—so MMQ can
defer its reduction until the shared output is ready. The primitive unfused
chain remains registered.

RED failed on the missing MMQ fusion policy. GREEN passes both actual
production-shape Q4_K/Q6_K MoE oracle cases byte-for-byte against a
forced-unfused MMQ path. Seven counter-rotated pp512 pairs preserve complete
logits/hidden/KV/token/cursor state; **4/7** candidate pairs win, median paired
wall improves **3.687 ms**, and paired geometric throughput improves
**0.302%**. The noisy absolute medians cross, so that alone is not used as the
claim. An independent traced pair proves the physical win: dispatches fall
**1,887 -> 1,840**, pp512 kernel span **943.200 -> 936.635 ms (-6.565 ms)**,
kernel sum **929.664 -> 924.797 ms (-4.867 ms)**, and all 47 sparse-layer
selected-sum plus add pairs become 47 composite calls. The candidate is
retained as default; clean selector-unset 512/1K/4K publication is the next
gate, so the production headline remains **542.088 tok/s** here. Evidence:
[`2026-07-26-gfx1151-laguna-mmq-combine-candidate.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-mmq-combine-candidate.json).

Clean publication at revision `b6bfc4a0b` promotes the candidate to current
production. Selector-unset 512/1K/4K medians improve
**542.088 -> 543.807 (+0.317%)**, **478.856 -> 480.017 (+0.243%)**, and
**387.725 -> 388.595 tok/s (+0.224%)**. All next tokens, final positions,
repeats, and tracked teardown pass. Cached tracing names 47 composite calls,
removes exactly 47 dispatches (**1,886 -> 1,839**), and cuts the
activation/reduce/residual family **17.914 -> 17.221 ms (-3.87%)**. The
independent trace reaches **544.994 tok/s**. Production is now
**543.807 tok/s**, leaving **210.081 ms** to the 700 wall. Evidence:
[`2026-07-26-gfx1151-laguna-mmq-combine-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-mmq-combine-production.json).

The next exact data-movement candidate removes the standalone selected SiLU
materialization without changing its numerical boundary. Gate/up writes its
packed **62.9-MB** BF16 tensor into the larger **73.4-MB** selected-down
allocation; the fused pack reads it there, evaluates the same SiLU expression,
rounds to BF16, converts that rounded value back to FP32, and runs the unchanged
range-safe D4 pack into the existing gate/up allocation. The registered
standalone SiLU plus ordinary pack remain the unfused fallback. Production
Q4_K and Q6_K MoE fixtures are BF16-byte exact, as are all seven
token/logit/hidden/KV/cursor pp512 pairs. The candidate wins **7/7**; median
paired wall improves **4.636 ms**, mean paired wall **6.098 ms**, and paired
geometric throughput **0.651%**. Cached tracing removes exactly 47 dispatches
(**1,840 -> 1,793**) and replaces **5.346 ms** of standalone SiLU plus
**4.954 ms** of ordinary pack with **6.377 ms** of fused pack, a
**3.924-ms / 38.09%** target-window reduction. It is retained as the gfx1151
default candidate. Clean selector-unset publication at revision `c0730bb94`
then improves 512/1K/4K medians **543.807 -> 546.100 (+0.422%)**,
**480.017 -> 481.640 (+0.338%)**, and
**388.595 -> 389.686 tok/s (+0.281%)**, with every expected next token,
deterministic final position, and exact tracked teardown. The independent
cached trace reaches **549.845 tok/s**, removes another 47 dispatches
(**1,839 -> 1,792**), names exactly 47 local128/VGPR16/LDS512B/scratch0
fused packs, and records zero standalone selected-SiLU calls. Production is
now **546.100 tok/s**, leaving **206.129 ms** to the 700 wall. Evidence:
[`2026-07-26-gfx1151-laguna-fused-silu-pack-candidate.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-fused-silu-pack-candidate.json).
[`2026-07-26-gfx1151-laguna-fused-silu-pack-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-fused-silu-pack-production.json).

The first post-546 selected-down screen is rejected and fully removed. A
64-column x 128-row/local256 Q6 qmicro body keeps **32 FP32 accumulators per
lane** while attempting to reduce repeated weight tiles only for heavy routed
experts. The CPU-quality fixture and actual rows64-versus-rows128 output are
BF16-byte exact; tracing records local256/VGPR88/LDS8704B/scratch0.

On the actual layer-1 weight and natural pp512 routing, the >=65-row subset
collapses **32 -> 17** tiles but improves only
**1.355870 -> 1.338579 ms (-1.27%, 0.017291 ms)** before the extra production
metadata schedule and launch. The supposedly strongest >=129-row tail
collapses **14 -> 8** tiles yet regresses
**0.673548 -> 0.687981 ms (+2.14%)** in a valid serial run. The additional
waves/occupancy cost erases the traffic reduction, so every kernel, wrapper,
test, and harness surface was removed and production remains
**546.100 tok/s**. Two overlapping exploratory GPU processes are explicitly
excluded from the evidence. Evidence:
[`2026-07-26-gfx1151-laguna-q6-down-rows128-heavy-rejected.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q6-down-rows128-heavy-rejected.json).

The next register-only streaming screen is also rejected and fully removed.
The production 128x32/local128 direct-wave Q4 gate/up body prefetched its next
decoded 40-byte T16 K32 record into registers while current dots executed.
Unlike the rejected K64 stage, this changed no activation/weight LDS,
barriers, resident bytes, output ownership, packed-dot/K order, or BF16
boundary. The CPU-reference gate and actual-weight BF16 checksum pass.

Nine counter-rotated actual layer-1 samples nevertheless regress the
pack-inclusive leaf **6.802111 -> 7.270426 ms (+6.885%, 0/9 wins)**.
Cached tracing holds LDS at 3,072 bytes and scratch at zero but raises VGPR
**88 -> 104**. The second live decoded record therefore costs more occupancy
and scheduling capacity than software overlap recovers. Every candidate
kernel, wrapper, test, and harness surface was removed. Do not retry
register-only K32 weight prefetch without a mechanism that keeps the
production VGPR footprint. Evidence:
[`2026-07-26-gfx1151-laguna-q4-wave-weight-prefetch-rejected.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q4-wave-weight-prefetch-rejected.json).

The first post-546 structural screen is retained for long prompts. Projection
and MoE capacity rises from M512 to M2048 while every attention and KV
operation remains independently sliced at M128. A wide pending KV transaction
may span the physical 512-token SWA ring, but no physical operation may do so;
the 640-row oracle is byte-identical to five separately committed M128
transactions across repeated wraps.

Two clean counter-rotated repetitions measure M512 -> M2048 at
512/1K/4K as **547.663/483.675/388.760 ->
545.703/509.891/411.121 tok/s**. That is **-0.358%/+5.420%/+5.752%**;
aggregate wall improves **5.256%**. M2048 uses **1,755,275,296 bytes** of
row/MoE scratch, remains within the existing 2-GiB admission floor, is
repeat-deterministic, and returns every tracked allocation. Full final-logit
comparison against M512 has maximum KL **0.000012503**, 100% top-1, and finite
outputs. Clean selector-unset publication on the promoted revision measures
**545.015/506.299/410.099 tok/s** at 512/1K/4K. The pp512 path receives no
speed credit because the actual transaction still contains 512 rows; the win
is retained for 1K/4K production. Evidence:
[`2026-07-26-gfx1151-laguna-m2048-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-m2048-production.json).

The subsequent exact Q4 gate/up scheduling screen is rejected and fully
removed. It kept the production 128x32/local128 D8 direct-wave body and merely
swapped grid axes so routed-row tiles ran fastest within a weight-column tile.
The actual layer-1 natural-M512 fixture is BF16-bit identical and the focused
GPU file passes **12 tests**, but twelve counter-rotated burst-three samples
regress **6.908966 -> 6.921503 ms (+0.181%)**. Axis order does not create
useful cross-workgroup weight reuse on this schedule. Production remains
**551.459 tok/s**. Evidence:
[`2026-07-26-gfx1151-laguna-q4-gate-rowfast-grid-rejected.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q4-gate-rowfast-grid-rejected.json).

The following source-F16 grouping screen is also rejected and fully removed.
At M512, one F32-bit-exact row-major QKV contraction improves the 12-full /
36-SWA synthetic family by only **2.891 ms** before any layout repair. The
combined output is `[M,Q+K+V]`, while the current attention path requires
three independently contiguous Q/K/V matrices; splitting or restriding that
output and maintaining concatenated resident weights would consume the small
ceiling. The layout-preserving alternative was screened through
`hipblaslt_ext::GroupedGemm`, but the installed gfx1151 library returns zero
algorithms for the full QKV problem with either zero or 64-MiB workspace. The
temporary C++/Python shim, harness, and RED fixture were removed. Production
remains **551.459 tok/s**. Evidence:
[`2026-07-26-gfx1151-laguna-f16-qkv-grouping-rejected.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-f16-qkv-grouping-rejected.json).

The next exact attention specialization is retained as a kernel candidate.
For complete initial no-wrap preappended tiles, logical and absolute token
positions are identical and no cache slot is evicted. The new global/SWA
bodies still consume the full `KVLiveSpans` ABI, preserve physical
`base_offsets`, and validate boundary metadata, but remove per-token
position/eviction loads and branches. Global qrow4, global qrow6, and SWA
qrow4 match the existing F32 output bit-for-bit at starts 0/128/256/384.
Eleven counter-rotated samples improve every natural point; the qualified
global-qrow4/qrow6 plus SWA-qrow4 policy moves **12.8348 -> 11.8695 ms
(1.0813x)** per four-layer pattern, modeling **11.584 ms** pp512 saving.
Cached tracing reports local32, zero LDS/scratch, and VGPR64/88/64 for global
qrow4/global qrow6/SWA qrow4. Runtime/default promotion remains open behind a
strict complete-initial-tile gate. Evidence:
[`2026-07-26-gfx1151-laguna-attention-dense-initial-candidate.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-attention-dense-initial-candidate.json).
The strict runtime gate now requires consecutive complete M128 positions,
capacity/no-wrap safety, untouched eviction metadata, and the existing
non-verifier preappend schedule. Seven matched full-model pairs improve
cached-metadata rollback **552.144 -> 559.539 tok/s (+1.339%, 5/7 wins)** and
save **12.255 ms** at the medians, while every compared output/state digest is
exact. gfx1151 defaults the capability with an explicit session rollback.
That matched gate admitted the default; clean publication evidence follows:
[`2026-07-26-gfx1151-laguna-attention-dense-initial-default.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-attention-dense-initial-default.json).
Clean selector-unset publication is now complete at
**559.290/523.090/439.044 tok/s**, improving the previous production
**1.420%/1.118%/1.607%**. Cached tracing reaches **559.225 tok/s**, cuts
attention **153.226 -> 141.846 ms (-7.43%)**, and observes exactly the
qualified 12/36/144 dense-initial launch mix. Evidence:
[`2026-07-26-gfx1151-laguna-attention-dense-initial-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-attention-dense-initial-production.json).

Production evidence:

- [`2026-07-26-gfx1151-laguna-q6-qmicro-planar-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q6-qmicro-planar-production.json)
  is current production: clean selector-unset medians are
  **573.354/530.351/446.189 tok/s** at 512/1K/4K. Planar dwords preserve the
  12-byte record and every output/state digest; the exact actual leaf improves
  **0.314%** and c1 decode improves **1.736%**.
- [`2026-07-26-gfx1151-laguna-q6-qmicro-permute-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q6-qmicro-permute-production.json)
  is the superseded interleaved-qmicro production packet.
- [`2026-07-26-gfx1151-laguna-moe-shared-low-priority-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-moe-shared-low-priority-production.json)
  is the superseded scheduling packet: clean selector-unset medians are
  **568.849/527.113/444.508 tok/s** at 512/1K/4K. Seven matched pairs
  preserve logits, both hidden snapshots, complete KV, token/logit, and cursor
  exactly. Cached tracing reaches **574.011 tok/s**, recovers **7.116 ms** of
  gate/up, and cuts kernel span **7.255 ms**.
- [`2026-07-26-gfx1151-laguna-f16-boundary-fusion-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-f16-boundary-fusion-production.json)
  is the superseded source-F16 boundary packet: clean selector-unset medians are
  **559.554/523.912/440.809 tok/s** at 512/1K/4K. Seven matched pairs
  preserve logits, both hidden snapshots, complete KV, token/logit, and cursor
  exactly. Cached tracing reaches **561.019 tok/s**, removes 96 standalone
  casts, and records **1,696** pp512 dispatches.
- [`2026-07-26-gfx1151-laguna-attention-dense-initial-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-attention-dense-initial-production.json)
  is the superseded dense-initial packet: clean selector-unset median **559.290 tok/s**,
  minimum **558.935 tok/s**, and 1K/4K **523.090/439.044 tok/s**. Seven
  matched pairs preserve complete state and improve **1.339%**. Cached
  tracing reaches **559.225 tok/s**, observes the intended 12/36/144
  dense-initial launch mix, and cuts attention to **141.846 ms**.
- [`2026-07-26-gfx1151-laguna-q6-skip-padded-activation-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q6-skip-padded-activation-production.json)
  is the superseded Q6 staging packet: clean selector-unset median **551.459 tok/s**,
  minimum **551.206 tok/s**, and 1K/4K **517.307/432.099 tok/s**. The
  repeated exact 23-layer screen improves **19/23** layers and
  **112.008 -> 111.806 ms (-0.180%)**; eleven complete-state pairs are exact
  and positive. Cached tracing observes local128/VGPR88/SGPR128/LDS5120B/
  scratch0. Its one Q6 trace is noisy at **118.802 ms**, so the repeated
  exact sub-window and clean wall medians—not that sample—are the retention
  evidence.
- [`2026-07-26-gfx1151-laguna-q6-half-row-activation-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q6-half-row-activation-production.json)
  is the superseded Q6 staging packet: clean selector-unset median
  **549.150 tok/s**, 1K/4K **514.956/430.300 tok/s**, and a clean traced Q6
  slice of **118.568 ms**.
- [`2026-07-26-gfx1151-laguna-global-qrow6-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-global-qrow6-production.json)
  is the superseded attention packet: clean selector-unset median **547.064 tok/s**,
  minimum **546.934 tok/s**, and 1K/4K **513.180/428.628 tok/s**. Seven
  matched pairs are complete-state exact and win 7/7. Cached tracing observes
  12 global-qrow4 / 36 global-qrow6 / 144 SWA-qrow4 calls and cuts attention
  **158.702 -> 152.406 ms (-3.97%)**.
- [`2026-07-26-gfx1151-laguna-m2048-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-m2048-production.json)
  is the superseded scheduling packet: clean selector-unset median **545.015 tok/s**,
  minimum **544.501 tok/s**, and 1K/4K **506.299/410.099 tok/s**. The matched
  policy screen improves long-prompt throughput **5.420%/5.752%** with maximum
  relative KL **0.000012503**, 100% top-1, deterministic repeats, an exact
  multi-wrap KV oracle, and exact lifecycle recovery.
- [`2026-07-26-gfx1151-laguna-fused-silu-pack-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-fused-silu-pack-production.json)
  is the superseded matrix512 production packet: clean selector-unset median **546.100 tok/s**,
  minimum **543.299 tok/s**, 1K/4K **481.640/389.686 tok/s**, and unchanged
  maximum KL **0.049542582**. Cached tracing removes 47 launches, records no
  standalone selected-SiLU calls, and observes the fused pack at
  local128/VGPR16/LDS512B/scratch0.
- [`2026-07-26-gfx1151-laguna-mmq-combine-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-mmq-combine-production.json)
  is the superseded grouped-combine publication: clean selector-unset median **543.807 tok/s**,
  minimum **541.485 tok/s**, 1K/4K **480.017/388.595 tok/s**, and unchanged
  maximum KL **0.049542582**. Cached tracing removes 47 launches and observes
  the exact composite at local128/VGPR8/LDS0/scratch0.
- [`2026-07-26-gfx1151-laguna-attention-cached-meta-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-attention-cached-meta-production.json)
  is the superseded attention publication: clean selector-unset median **542.088 tok/s**,
  minimum **542.022 tok/s**, 1K/4K **478.856/387.725 tok/s**, and unchanged
  maximum KL **0.049542582**. Cached tracing observes the intended qualified
  policy and cuts global+SWA attention **175.802 -> 160.123 ms (-8.92%)**.
- [`2026-07-26-gfx1151-laguna-attention-cached-meta-default.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-attention-cached-meta-default.json)
  is the retained gfx1151 default-candidate provenance: matched pp512 improves
  **533.507 -> 542.785 tok/s (+1.739%, 7/7 wins)** with complete output/state
  exactness.
- [`2026-07-26-gfx1151-laguna-q6-qmicro-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q6-qmicro-production.json)
  is the superseded Q6-layout publication: clean selector-unset median **530.447 tok/s**,
  minimum **525.864 tok/s**, cached trace **535.006 tok/s**, and unchanged
  maximum KL **0.049542582**. The byte-neutral layout is BF16-byte exact;
  tracing cuts Q6 selected down **126.594 -> 123.473 ms (-2.465%)** and total
  selected down **203.923 -> 200.510 ms (-1.673%)**.
- [`2026-07-26-gfx1151-laguna-attention-preappend-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-attention-preappend-production.json)
  is the superseded attention publication: clean selector-unset median **526.451 tok/s**,
  minimum **526.288 tok/s**, cached trace **532.101 tok/s**, and unchanged
  maximum KL **0.049542582**. Matched seven-pair A/B isolates the exact
  cached-only attention schedule at **+4.214%** with **7/7** wins; tracing
  cuts global+SWA attention **219.709 -> 176.580 ms (-19.63%)**.
- [`2026-07-26-gfx1151-laguna-gate-activation-doublebuf-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-gate-activation-doublebuf-production.json)
  is the superseded gate synchronization publication: clean median **505.084 tok/s**,
  minimum **504.984 tok/s**, cached trace **509.777 tok/s**, and unchanged
  maximum KL **0.049542582**. Matched seven-pair A/B isolates the exact
  one-barrier gate/up body at **+0.284%** and tracing cuts gate/up
  **318.559 -> 314.378 ms (-1.313%)**.
- [`2026-07-26-gfx1151-laguna-f16-output-range-direct-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-f16-output-range-direct-production.json)
  is the superseded output-boundary publication: conservative clean median
  **505.185 tok/s**, clean minimum **503.198 tok/s**, cached trace
  **510.946 tok/s**, and unchanged maximum KL **0.049542582**. Both
  source-F16 boundaries are static-range direct and exact.
- [`2026-07-26-gfx1151-laguna-f16-norm-direct-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-f16-norm-direct-production.json)
  is the superseded norm-only direct publication: clean median **503.869 tok/s**, clean
  minimum **501.790 tok/s**, cached trace **507.067 tok/s**, and unchanged
  maximum KL **0.049542582**. The direct attention-norm boundary is exact and
  cuts cached source-F16 **134.442 -> 128.274 ms**.
- [`2026-07-26-gfx1151-laguna-router-token-tile8-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-router-token-tile8-production.json)
  is the superseded router-token publication: conservative clean median **503.349 tok/s**, clean
  minimum **501.698 tok/s**, cached trace **504.631 tok/s**, and unchanged
  maximum KL **0.049542582**. The 500 production gate is closed.
- [`2026-07-26-gfx1151-laguna-parallel-compact-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-parallel-compact-production.json)
  is the superseded exact parallel-compaction publication at
  **497.408 tok/s** median and **500.325 tok/s** cached trace.
- [`2026-07-26-gfx1151-laguna-q6-down-rows64-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q6-down-rows64-production.json)
  is the superseded Q6 rows64 publication at **492.640 tok/s** median and
  **493.509 tok/s** cached trace.
- [`2026-07-26-gfx1151-laguna-q4-pack8-shape-policy-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q4-pack8-shape-policy-production.json)
  is the superseded exact Q4 shape-policy publication at **489.922 tok/s**
  median and **492.717 tok/s** cached trace.
- [`2026-07-26-gfx1151-laguna-q6-dense-wmma16x32-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q6-dense-wmma16x32-production.json)
  is the superseded pre-Q4-shape-policy publication at **490.096 tok/s**
  median and the Q6 tile provenance.
- [`2026-07-26-gfx1151-laguna-q4-down-direct-wavecols-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q4-down-direct-wavecols-production.json)
  is the superseded direct-Q4-down publication at **480.629 tok/s** median and
  the direct all-exact quality source.
- [`2026-07-26-gfx1151-laguna-q4-direct-wavecols-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-q4-direct-wavecols-production.json)
  is the superseded gate/up-direct publication at **474.363 tok/s** median.
- [`2026-07-26-gfx1151-laguna-down-wavecols-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-down-wavecols-production.json)
  is the superseded pair-decode publication at **448.203 tok/s** median.
- [`2026-07-26-gfx1151-laguna-wavecols-production.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-wavecols-production.json)
  is the superseded gate/up-only wave-column publication at **432.355 tok/s**.
- [`2026-07-26-gfx1151-laguna-production-absolute-quality.json`](../benchmarks/results/2026-07-26-gfx1151-laguna-production-absolute-quality.json)
  is the superseded pre-wave-column absolute-quality publication at
  **386.552 tok/s** median.
- [`2026-07-25-gfx1151-laguna-down-rowvec-production.json`](../benchmarks/results/2026-07-25-gfx1151-laguna-down-rowvec-production.json)
  is the superseded pre-absolute-audit publication at **385.997 tok/s** and the
  latest all-family trace.
- [`2026-07-25-gfx1151-laguna-gate-rowvec-production.json`](../benchmarks/results/2026-07-25-gfx1151-laguna-gate-rowvec-production.json)
  is the superseded row-vector D8 gate/up publication at **379.811 tok/s**.
- [`2026-07-25-gfx1151-laguna-swa-sourcequal-production.json`](../benchmarks/results/2026-07-25-gfx1151-laguna-swa-sourcequal-production.json)
  is the superseded source-qualified SWA publication at **366.933 tok/s**.
- [`2026-07-25-gfx1151-laguna-prefill-qrow4-production.json`](../benchmarks/results/2026-07-25-gfx1151-laguna-prefill-qrow4-production.json)
  is the superseded qrow4 publication at **364.839 tok/s** median.
- [`2026-07-25-gfx1151-laguna-prefill-350-production.json`](../benchmarks/results/2026-07-25-gfx1151-laguna-prefill-350-production.json)
  is the superseded 350-milestone publication artifact.
- [`2026-07-25-gfx1151-laguna-prefill-350-production-default.json`](../benchmarks/results/2026-07-25-gfx1151-laguna-prefill-350-production-default.json)
  is the raw selector-unset 27-row timing/state screen. Its historical
  cross-matrix byte-equality policy correctly rejects the already-admitted
  approximate arithmetic; publication accepts only those two declared legacy
  failures and independently requires same-mode determinism and lifecycle.
- [`2026-07-25-gfx1151-laguna-prefill-350-production-trace.json`](../benchmarks/results/2026-07-25-gfx1151-laguna-prefill-350-production-trace.json)
  attaches the cached-only all-family trace; the 1.5 MiB raw CSV remains
  uncommitted and is bound by SHA-256.

### LAP-0 — freeze the current control and cumulative quality ledger (complete)

Deliverables:

- run a clean cached 128/512/1K/4K profile at `e4ab85d59` or its unchanged
  descendant;
- replace the inferred post-SWA attention time with measured current family
  attribution;
- preserve exact commands, kernel sum/span, calls, resources, model/hash,
  clocks, and lifecycle in one compact artifact;
- add an explicit all-exact session configuration and measure all-exact versus
  shipping-control quality over the complete category lane;
- capture compact activation/routing statistics needed by LAP-1/LAP-2 without
  committing prompt activations or raw logs;
- freeze the local Vulkan comparator revision/build and reuse its existing
  artifact unless source, binary, model, driver, or hardware changed.

Exit gate: one current bridge table whose families sum to at least 99.5% of
kernel time, plus a cumulative quality baseline. No optimization code lands in
this task.

Result: passed at
[`2026-07-24-gfx1151-laguna-prefill-lap0-control.json`](../benchmarks/results/2026-07-24-gfx1151-laguna-prefill-lap0-control.json).
Named non-`other` coverage is **99.653%** at M512; cumulative quality is finite
at max KL **0.0459275** and **319/320** top-1; all profile, routing, activation,
cursor, determinism, Poolside, and tracked-lifecycle checks pass. Public runtime
defaults are unchanged.

### LAP-1 — establish packed-dot reuse and choose the resident layout

Before implementation, read [`KERNELS.md`](KERNELS.md) and run:

```bash
python3 scripts/check_lineage.py --kind kernel --diff stat
```

Deliverables:

- implement one standalone staged Q4_K x Q8_1 packed-dot MMQ body and establish
  the first natural-shape crossover;
- use actual Laguna K3072/N1024 expert weights and natural M32/55/64/122/128/
  256/512 routing replays;
- compare raw source blocks, X8, and a direct current-T16 consumer;
- trace packed-dot instructions, workgroup, VGPR/SGPR, LDS, scratch, and tile
  occupancy;
- prove that the body, not just the activation pack, beats the current selected
  family before runtime integration;
- preserve the existing exact T16 decode leaf and prove one T16 resident set
  serves both decode and candidate prefill.

The current diagnostic scalar DS4, independent WMMA32/64, expanded-LDS,
packed-LDS, preview, and direct-T16 **WMMA** paths are controls only. The new
T16 MMQ kernel must materially differ by implementing the proven complete tile
reuse without a layout transpose.

Exit gate: direct-T16 MMQ is at least 2x inclusive over the retained expert body
on M128/M256/M512, positive on every natural shape, within 10% of the frozen
X8 control on the primary shapes, and uses no full expert sidecar or
per-dispatch layout transpose. Existing exact T16 decode remains bitwise and
performance unchanged. A smaller exact non-regressive sub-window may still be
retained under repository policy, but it does not advance the parity campaign.

Result: the first gfx1151 body uses a 32-column by 32-row Q4_K x Q8_1 tile over
four wave32s in one 128-thread workgroup. It stages
20 bytes of Q4_K data per column and 36 bytes of DS4-Q8_1 data per routed row
for each K32 interval, reuses both tiles across the workgroup, and emits native
packed integer dot instructions. Q8_1 is packed once per producer row; compact
expert rows carry only a source-row index.

The post-LAP-1 source audit corrects the original attribution: Vulkan's actual
gfx1151 comparator is the medium **64x64** routed tile over two wave64s, not
this 32x32 tile. LAP-1 remains complete because its gates were measured against
retained and X8 bodies, not because the geometry matched Vulkan. A widened
64x64-class schedule, K64 nibble reuse, and more work per barrier remain active
performance levers for later expert integration.

On actual layer-1 K3072/N1024 gate/up weights and natural routing counts,
including the producer-row pack, the raw body moves M256 **26.612 -> 10.047 ms
(2.649x)** and M512 **52.522 -> 12.720 ms (4.129x)** versus the retained
direct leaf. The current T16 WMMA diagnostic measures **6.297/9.307 ms**, but
uses a larger resident representation and its arithmetic is not quality-safe.
The raw layout is **864 MiB** for the pair versus **888 MiB** for T16.
Synthetic source-Q4_K x DS4-Q8_1 fixtures pass at maximum softmax KL
**4.745e-5** and **100%** top-1. A cached trace reports local128, allocated
VGPR120, LDS 2,048 bytes, zero scratch, and 64 static
`v_dot4_i32_iu8` instructions per wave.

The clean LAP-1 routing capture now covers all declared shapes. Across all 47
sparse layers, natural tile32 padding is **10.857/8.558/7.873/5.108/4.928/
2.930/1.866x** at M32/55/64/122/128/256/512, versus **2.911/2.402/2.260/
1.721/1.691/1.335/1.165x** for tile8. The actual layer-1 inclusive MMQ32
speedups are **0.680/0.899/0.985/1.515/1.551/2.645/4.117x** over retained
direct at the same shapes. Literal tile32 therefore loses at M32–M64, is
positive but below the 2x gate at M122/M128, and passes only M256/M512.

Three follow-up 8-row designs close the smaller packed-dot branch. A one-wave
32x8 body, four-wave cooperative 64x8 body, and paired-lane wave-local 16x8
body all lose to MMQ32 at every declared natural shape. At M128 they reach only
**1.098/0.926/1.269x** retained-direct versus **1.563x** for MMQ32; at M512
they reach **1.962/1.616/2.096x** versus **4.174x**. The first two reproduce
MMQ32 checksums exactly; the 16x8 primitive passes its focused KL/top-1 gate but
its natural diagnostic omitted one FP16 metadata-rounding step and was removed
without repair after the performance rejection. Cached traces show that lower
padding and VGPR do not offset repeated K3072 weight decode: the one-wave 32x8
tile costs about 405 us and the cooperative 64x8 tile about 522-531 us, versus
about 41 us per MMQ32 tile at the natural M128 leaf.

The whole-expert mixed screen also rejects the existing exact grouped-small-M
leaf as a tail. Threshold 1 is simply all-MMQ32; every threshold that sends
even one active expert to exact is slower at every shape. The lightest true
mixed case raises M128 **8.867 -> 11.625 ms (+31.10%)** and M512
**12.524 -> 13.294 ms (+6.15%)** before any device merge/scatter. All-exact
grouped-small-M itself is **43.622/136.742 ms** at M128/M512.

The first, prefill-only resident-layout screen selected the existing byte-exact
Q4_K X8 format. Raw and X8 share the complete packed-dot arithmetic body; X8
changes only the weight-block address. Two uneven/empty-expert fixtures,
including a nonidentity source-row map, are BF16-bit identical to raw and pass
the independent CPU KL/top-1 gate.

On the clean actual-weight screen, X8 improves raw MMQ32 by
**12.14/11.81/11.79/11.53/11.70/11.47/9.82%** at
M32/55/64/122/128/256/512. Its inclusive speedups over retained direct are
**0.766/1.011/1.105/1.693/1.735/2.957/4.554x**. Raw and X8 checksums match
exactly at every shape, both gate/up pairs occupy **905,969,664 bytes**, and
all tracked temporary buffers return to zero. That layout-only screen made X8
the provisional resident winner but did not change a runtime default because
M32 still lost and M128 had not yet satisfied the LAP-1 2x gate. The later
exact-decode screen above supersedes the resident conclusion while preserving
X8 as the fastest MMQ control.

The retained live-row schedule closes that body/shape gap without a second
geometry: it clamps the natural row count once per tile and skips packed-dot
accumulation for padded routes while preserving live-output arithmetic. Clean
producer-pack-inclusive X8 now measures **3.309/4.064/4.283/5.211/5.331/
6.515/9.330 ms**, or **1.197/1.567/1.704/2.526/2.587/4.092/5.614x**
retained at M32/55/64/122/128/256/512. Relative to the prior X8 screen, time
falls **36.45/35.57/35.22/32.77/32.77/27.41/18.65%**. Raw and X8 checksums
remain exact at every shape; the focused bundle reports 29 passes. Cached
tracing is local128, raw/X8 VGPR **40/48**, SGPR128, LDS 2,048 bytes, and zero
scratch.

This schedule has a recorded boundary: an all-full synthetic tile moves
**0.3881 -> 0.4204 ms (+8.34%)**. Natural Laguna routing is positive at every
frozen shape, so do not add another tail geometry now. If the integrated trace
shows full-tile predicate cost is material, separate full and tail metadata
into two symbols; otherwise avoid the extra launch and code.

The exact X8-native branch is now closed. After correcting the first
local256/eight-wave reduction-order bug, the final local128 kernel is BF16-bit
exact and dynamically constructs a T16-shaped 16-column tile in LDS. The clean
actual layer-1 c1/c2/c4/c8 medians are T16
**0.157223/0.351996/0.687016/1.350421 ms** versus X8
**0.174663/0.362511/0.686471/1.332379 ms**. X8 is **11.093%** slower at c1 and
**2.987%** slower at c2, then neutral/positive at c4/c8. All gate/up BF16
mismatch counts are zero, the temporary comparison peak is
**1,837,482,624 bytes**, and tracked ownership returns to zero. The c=1 target
therefore rejects X8 as the sole resident representation.

The direct-T16 branch closes LAP-1. Its clean producer-pack-inclusive times are
**3.383/4.173/4.419/5.399/5.543/6.769/9.597 ms**, or
**1.174/1.528/1.662/2.464/2.502/3.959/5.502x** retained direct at
M32/55/64/122/128/256/512. T16 is only **4.66%/4.05%/3.02%** behind X8 at
the primary shapes. T16/X8 BF16 checksums match at every shape, focused tests
report 31 passes, and cached tracing reports local128/VGPR48/LDS2048B/scratch0
with packed-dot ISA. The guarded LAP-2 primitive subsequently landed, and
later sections record the completed calibration, selected-family integration,
and production promotion. No small-row threshold, X8 materializer, or
duplicate weight sidecar is retained.
Evidence:
[`2026-07-24-gfx1151-laguna-q4-k-mmq32-leaf.json`](../benchmarks/results/2026-07-24-gfx1151-laguna-q4-k-mmq32-leaf.json).
The all-shape crossover packet is
[`2026-07-24-gfx1151-laguna-q4-k-mmq32-shape-screen.json`](../benchmarks/results/2026-07-24-gfx1151-laguna-q4-k-mmq32-shape-screen.json).
The rejected small-row packet is
[`2026-07-24-gfx1151-laguna-q4-k-mmq8-tail-rejected.json`](../benchmarks/results/2026-07-24-gfx1151-laguna-q4-k-mmq8-tail-rejected.json).
The rejected whole-expert mixed packet is
[`2026-07-24-gfx1151-laguna-q4-k-mixed-exact-rejected.json`](../benchmarks/results/2026-07-24-gfx1151-laguna-q4-k-mixed-exact-rejected.json).
The retained X8 layout packet is
[`2026-07-24-gfx1151-laguna-q4-k-x8-mmq32-layout-retained.json`](../benchmarks/results/2026-07-24-gfx1151-laguna-q4-k-x8-mmq32-layout-retained.json).
The retained live-row schedule packet is
[`2026-07-24-gfx1151-laguna-q4-k-x8-mmq32-live-row-retained.json`](../benchmarks/results/2026-07-24-gfx1151-laguna-q4-k-x8-mmq32-live-row-retained.json).
The exact X8 decode rejection is
[`2026-07-25-gfx1151-laguna-q4-k-x8-exact-decode-rejected.json`](../benchmarks/results/2026-07-25-gfx1151-laguna-q4-k-x8-exact-decode-rejected.json).
The retained direct-T16 consumer is
[`2026-07-25-gfx1151-laguna-q4-k-t16-mmq32-retained.json`](../benchmarks/results/2026-07-25-gfx1151-laguna-q4-k-t16-mmq32-retained.json).

### LAP-2 — calibrate residual Q8_1 and exact repair

Deliverables:

- extend the Qwen residual-D4 and BF16-boundary machinery to Q4_K and Q6_K
  without coupling model code to Qwen;
- compare one, two, and three residual planes, block32 versus DS4/block128
  scaling, FP16 versus FP32 scale storage, and accumulation order;
- evaluate gate, up, Q4 down, and Q6 down independently on production
  activations and weights from all declared prompt categories;
- implement bounded risk queues, all-queued exact correction, deterministic
  overflow fallback, and queue-memory accounting;
- select a global `(quant, role, shape)` policy on calibration data and freeze
  it before heldout/category admission;
- report inclusive quantize + MMQ + repair time, not a prequantized body alone.

Exit gate: all-queued correction is BF16-bit exact; the selected policy passes
the primitive gate; post-repair mismatch is zero or the complete repository
quality lane passes without worsening the cumulative exact-oracle ledger; and
inclusive speed still satisfies LAP-1's primary body premise.

Primitive result: `d9bb6ad88` adds byte-stable DS4x3 packing, direct and guarded
three-pass T16 MMQ32, a bounded 16-column risk queue, and exact correction with
a deterministic full-projection overflow fallback. Both all-queued and forced
overflow tests are BF16-bit exact; the focused bundle reports **35 passed** and
cached tracing names all three new kernels. Dirty actual layer-1 inclusive
D4x3 is **1.289x** retained at M128 and **2.510x** at M512. Subsequent
real-input calibration selected same-byte D8 gate/up plus D4 down, and the
complete category/decode/determinism/lifecycle lane admitted that combination.
The guarded D4x3 primitive remains an exact fallback rather than the production
route.

### LAP-3 — promote selected Q4 gate/up

Deliverables:

- quantize each BF16 token row once before top-10 expansion;
- consume the existing device-resident route count/prefix/compact map;
- process natural rows per expert and return compact gate/up rows in the
  existing lane order;
- test separate versus paired gate/up only after the shared-tile body works;
- preserve the separate exact SiLU chain and current selected/grouped fallback;
- treat direct 128x64/64x64/256x32 routing, resident-T16 K64/K128 staging, and
  scalar coalesced-raw LDS staging as rejected; reopen only for a counter-backed
  hybrid large-expert path or a wave-transpose load/unpack primitive;
- do not retry K64 nibble reuse unless the resident layout changes: T16 stores
  the two K32 subblocks separately, so the raw-Q4 one-fetch premise does not
  apply;
- choose row/occupancy crossovers from M32/55/64/68/96/122/128/256/512
  measurements, not a blanket M32 policy;
- run the full canonical category, Poolside, h16/h32, lifecycle, and cached
  trace gates before changing the backend capability.

Planning checkpoint: the first integrated route must transfer the clean leaf
gain across all 47 sparse layers and resolve the 78.27-versus-52.80 ms/layer
bridge discrepancy. The family is not complete merely at Vulkan parity:
report encoded and physical GB/s, and continue toward at least **70% of the
LAP-BW0 achievable-read result** unless profiling proves a different limiter.
Any exact same-suite non-regressive win is retained even if it misses that
checkpoint.

First integration result: the explicit `mmq32_d4x3` session route quantizes
the 512 producer rows once per layer, builds stable compact/source and 32-row
tile metadata on device, emits compact gate/up, and passes that compact SiLU
output directly to exact grouped Q4/Q6 down. It does not allocate a weight
sidecar and keeps c=1 plus rows below 32 on the exact direct route. A
same-session dirty-tree actual pp512 diagnostic measures **6.7003 -> 4.0123
seconds**, or **76.414 -> 127.607 tok/s (1.670x)**, with next token **2930** in
both modes. This proves the production graph uses the intended MMQ body; it is
not a retained performance or quality claim until the clean canonical gate.

Second integration result: range-safe FP32 metadata and a 64-column x 32-row
T16-native one-plane MMQ consumed Q4 gate/up plus Q4/Q6 down without a raw/X8
transpose or weight sidecar. With LAP-5/LAP-6 compounded it reached
**355.273/355.721 tok/s**, but the complete category lane rejected it at
maximum KL **0.0767056** despite **318/320** top-1. The one-prompt KL
**0.001146** screen was therefore not representative and must not be used for
admission.

Third integration result: gate/up now quantizes in 16-value groups while
preserving the 160-byte block footprint; down remains one-scale-per-32 D4. The
Q4 dual consumer widens to 128 columns x 32 rows, with 32 FP32 accumulators per
lane, while reconstructing each half-block quant sum for the Q4 min term.
The clean shipping-relative category gate passes at maximum KL **0.040724836** and
**317/320** top-1. It improves aggregate natural-prompt prefill
**70.192 -> 183.563 tok/s (2.615x)**, h16/h32 E2E
**1.552x/1.322x**, keeps decode within 0.01%, passes Poolside at KL
**0.0000175** with equal top-1, and returns tracked allocations exactly to
zero. Exact reconstructed-sum pp512 repeats at
**353.951/356.082/356.473 tok/s**, always token **2930**. The tempting raw-sum
variant was faster but failed quality and was removed. gfx1151 now defaults to
this D8 gate/up route and the admitted D4 down route. The clean selector-unset
publication initially closed the 350 check at **354.820 tok/s** median.
Subsequent exact attention, expert, dense/shared, metadata, and router
improvements plus direct attention-norm consumption raised the then-current
production row to **526.451 tok/s** after the separate absolute-quality hipBLASLt repair,
static-range direct output boundary, exact activation double buffer, and exact
cached-only M128 attention scheduling.

Non-temporal weight loads are not a default lever here. Existing gfx1151
cold-DRAM decode evidence found a **+14%** isolated rows=1 bandwidth gain but a
**0.68x** rows>1 regression and flat/slower end-to-end decode. Permit one
rows>1 MMQ screen only after the byte/counter audit shows cache pollution is a
measured limiter; otherwise preserve row reuse through cache.

### LAP-4 — promote selected Q4/Q6 down

Deliverables:

- pack the exact BF16 SiLU/product output once per compact routed row;
- add separate source-arithmetic Q4_K and Q6_K packed-dot leaves;
- repair before the BF16 down boundary;
- retain the current ordered route-weighted combine as an unfused fallback;
- test weighted/fused output only after the unfused projection is admitted;
- carry forward the gate/up default and reprofile all families.

Planning checkpoint: selected gate/up plus down must both report
GB/s/%-of-achievable against LAP-BW0. Continue each streaming family toward
the 70% floor unless a measured arithmetic, occupancy, or repair limiter
supersedes the bandwidth model. The prior 176 tok/s F16 diagnostic remains a
demonstrated scheduling checkpoint, not the campaign target or a quality
claim.

### LAP-5 — reuse the MMQ engine for dense and shared experts

Execute this immediately after LAP-6, before selected down. Shipping is
**0.6415 s** versus **0.0629 s** Vulkan, the worst mapped ratio, and the family
has no routing metadata or new projection-role quality surface.

Integrated candidate:

- `pack8_wmma_prefill_bf16_bf16_out` consumes the already-resident Q4 pack8
  words plus FP32 effective scale/min planes directly. It adds no weight
  sidecar and does not invalidate the 66-GiB repacked cache.
- One wave computes a 64-column x 16-row tile with FP16 WMMA operands and FP32
  accumulation. It is BF16-bit identical to the existing raw-Q4 WMMA kernel
  on the independent synthetic fixture and passes that kernel's CPU-reference
  KL/top-1 tolerance.
- The M512/K3072/N1024 leaf improves **1.2695 -> 0.2407 ms (5.275x)**. A
  same-session compounded pp512 screen improves the retained dense route
  **154.071 -> 162.274 tok/s** with 64x32; the selected 64x16 default then
  reaches **163.881 tok/s**, always with next token 2930.
- Cached tracing names
  `gguf_q4_k_pack8_prefill_wmma_kernel<unsigned short,unsigned short,64,16>`
  at **23.244 us** on the boundary fixture, local32, VGPR88, SGPR128, zero
  LDS, and zero scratch.

Integrated Q6 extension:

- raw-Q6 dense/shared projections now use a 64x16 source-GGUF WMMA consumer;
  two aligned/boundary CPU-reference fixtures pass;
- the exact pre-change 320 tok/s trace attributes only **28.866 ms** to this
  Q6 family, down from the prior **0.365 s** retained path;
- use dense row tiles directly—no route-count or padded expert machinery;
- target a 64x64/128x128-class dense tile with four K32 stages per barrier
  (`BK_STEP=4` control), rather than inheriting routed 32x32 geometry;
- pair gate/up only where the inclusive real-model leaf wins;
- preserve exact rank-2 pack8/raw-Q6 fallbacks and shared-expert addition order;
- reject another duplicate pack8/T16 sidecar unless the total resident/context
  budget is explicitly better than the replacement-layout design;
- reprofile before selected down or attention work.

The dense/shared checkpoint is closed: the compounded stack is above 350 tok/s.
Reprofile only after the complete quality/default admission, not to justify
more dense-kernel work.

### LAP-6 — close the source-F16 projection gap

The existing compensated WMMA path is about **6.285x** faster than exact at the
weighted M128 projection screen, but reaches only about 45-52% of the measured
inclusive hipBLASLt ceiling. It is retained only on SWA layers because the
all-layer quality route failed.

Execute this before LAP-5/LAP-2 integration. The measured M512 inclusive
hipBLASLt family is **138.351 ms**, not 350 ms: it is about **2.03x** faster
than the Vulkan source-F16 family and offers a measured **755.719 ms** reduction
from shipping if the real-input contract passes.

Deliverables:

- compare the custom compensated path with a torch-free, raw-pointer
  hipBLASLt route using the already measured inclusive conversion contract;
- validate nonzero real projection buffers and BF16 dynamic range; screen a
  per-row power-of-two scale for BF16→FP16 conversion so scale/unscale itself
  is exact in binary and overflow is impossible;
- reduce the current high-VGPR custom path only when a profile identifies a
  concrete occupancy or data-movement limit;
- add BF16-boundary exact repair or a higher-accuracy accumulation mode so
  coverage is selected by arithmetic/shape rather than global-versus-SWA layer
  identity;
- preserve exact tiled projection as the registered fallback;
- include Q/K/V/O and per-head attention-gate projections in the full model
  gate and cumulative exact-oracle ledger.

Planning checkpoint: first reproduce **0.14–0.18 seconds** on real inputs; do
not weaken the target to 0.35 seconds unless the measured nonzero-data/range
contract explains the gap. Reprofile overall throughput after promotion.

First integration result: the session-local `hipblaslt_scaled` route casts one
BF16 producer row to finite FP16 with an exact power-of-two scale, caches seven
zero-workspace shape descriptors, and restores FP32/BF16 outputs before their
existing consumers. It reuses the post-embedding token-ID buffer for row
scales, so bounded scratch and resident weights do not grow. With D4x3 MMQ held
constant, a same-session real pp512 diagnostic moves **4.0053 -> 3.3178
seconds**, or **127.831 -> 154.321 tok/s (1.207x)**, while both routes select
token **2930**. The measured **687.5 ms** wall reduction captures most of the
755.7 ms library opportunity. This is an integrated candidate, not a default
promotion; cumulative KL/category/lifecycle and a clean A/B remain mandatory.

### LAP-7 — `KVLiveSpans`-aware attention (cache-order schedule admitted)

Start only after a fresh post-LAP-6 profile puts attention at 10% or more of
kernel time, or the remaining comparator gap is dominated by it.

Deliverables:

- one in-tree M16-query x K64-key online-softmax design for head dimension 128;
- tiled QK and PV with FP32 accumulation and an explicitly gated BF16-KV to
  matrix-input conversion;
- complete `KVLiveSpans` handling: global spans, SWA physical rings, absolute
  positions, eviction masks, causal partial tiles, and 511/512/513 boundaries;
- raise the attention chunk above 128 only after full cursor/KV equivalence;
- retain exact global/SWA and current online row2 kernels as fallbacks;
- cover prior-context 0/64/128/384/896/1920/3968, SWA wraps, partial query
  tiles, and all canonical prompts.

Do not retry paired row2 score materialization, qgroup9, or the invalid
head-dim-128 AOTriton adapter. Those premises are closed.

Tiled-kernel result: the start threshold was satisfied, but correct M16xK64 and
M8xK64 tiled-WMMA bodies regressed pp512 **4.15%** and **8.82%** respectively.
The resource floor was VGPR248/LDS50,688B for M16 and VGPR224/LDS22,016B for
the K/V-reusing M8 specialization. Both were removed. Source-qualified qrow4
remains the arithmetic body and fallback.

Retained scheduling result: complete M128 global tiles and pre-wrap SWA tiles
now append current K/V through the existing BF16 writer before attention, then
run an exact cached-only qrow4 specialization. Partial tiles, wrapped SWA,
staged verifier transactions, gfx1100, and unmeasured backends preserve the
old ordering. Nine counter-rotated leaf samples improve global/SWA by
**1.305x/1.142x** at start 0 and **1.305x/1.186x** at start 384. Clean
selector-unset pp512 improves **505.084 -> 526.451 tok/s (+4.230%)** and the
trace cuts attention **219.709 -> 176.580 ms (-19.63%)**. Primitive output and
full-model state are exact. Further LAP-7 work requires a different async-copy,
supported-library, or materially fused-softmax premise.

### LAP-8 — final residual profile and qualified parity

Deliverables:

- capture a new complete 128/512/1K/4K profile and rebuild the homologous
  Vulkan family table;
- touch router/norm/RoPE/tails only when one named family has a measured
  end-to-end ceiling of at least 5%;
- consider cross-operation fusion only with the required unfused registry
  fallback and bit/quality gate;
- keep graph/submission work closed until kernel span minus kernel sum exceeds
  5% or API tracing shows a repeated synchronization/copy boundary;
- rerun the external Vulkan row only if its source/build/runtime identity
  changed, and retain all protocol qualifications;
- publish the final benchmark artifact, rollup, changelog, kernel catalog,
  refactor cleanup, and WORKLOG handoff.

## Milestones

These are planning checkpoints, not promises or minimum thresholds for keeping
a valid smaller win:

| Milestone | pp512 target | Interpretation |
| --- | ---: | --- |
| Historical gate/up checkpoint | 135-140 tok/s | Primary mapped gap is materially closed; not an exit target. |
| Historical selected-expert checkpoint | 165-175 tok/s | Fast expert-major scheduling is quality-safe; not an exit target. |
| Historical all-quant checkpoint | >=200 tok/s | Dense/shared reuse is working; not an exit target. |
| Historical linear checkpoint | 275-290 tok/s | Linear projection architecture is comparator-class; not an exit target. |
| Gap substantially closed | >=310 tok/s | Within 10% of the 344.56 Vulkan control. |
| Compatibility floor | >=344.56 tok/s | Match/beat the qualified external row; no longer definition-of-done by itself. |
| Production target | >=350 tok/s | Clean selector-unset gfx1151 default under the complete quality/lifecycle protocol. |
| Next production milestone | >=500 tok/s | Every clean pp512 sample and the median clear 500 under the same contract. |
| Stretch production milestone | >=700 tok/s | Same contract; requires measured expert/attention roofline progress. |
| Streaming-family floor | >=70% of measured read roof | About 155 GB/s if the same-host anchor is 221 GB/s; report each mapped family. |
| Roofline system target | Set by LAP-BW0 | Exact active-byte ledger plus non-streaming wall; the review's ~650–750 tok/s range is a hypothesis until measured. |

The 350 and 500 production targets are achieved and current production is
**645.803 tok/s**. The 700 stretch and stronger streaming/roofline rows remain
active targets.

All headline rows also report canonical category-weighted prefill and
128/1K/4K behavior. A repeated-token 512 number cannot promote a path by itself.

## Promotion gates

Every new or ported kernel follows [`TESTING.md`](TESTING.md) and
[`BENCHMARK.md`](BENCHMARK.md). At minimum:

### Primitive and dispatch

- RED test before implementation where practical;
- CPU/source oracle at tiny and production dimensions;
- KL <= 0.05 and top-1 >= 90% for any non-bit-exact primitive;
- exact raw-pointer ABI and four-axis registry resolution;
- no backend/quant branch in engine, model, or generic dispatch code;
- cached `rocprofv3 --kernel-trace` proving the intended symbol, duration,
  workgroup, VGPR/SGPR, LDS, and scratch;
- unfused/exact fallback and automatic fallback below unmeasured shapes.

### Full-model quality

- all-exact, shipping-control, and candidate lanes in one resident load;
- all ten canonical prompts across `code`, `general_en`, `general_ja`, and
  `mixed_ja_en`, including heldouts;
- 320-step teacher-forced full-vocabulary comparison;
- deterministic free-running h16/h32 repeats and complete ID reporting;
- frozen Poolside first-token oracle;
- suite-wide and per-category top-1 >= 90%, max KL <= 0.05;
- cumulative candidate-versus-all-exact result recorded;
- no prompt-conditioned or observed-output-conditioned policy.

### Performance and ownership

- at least three counterbalanced same-session timing repetitions;
- aggregate and every-category prefill positive for a default promotion;
- aggregate h16/h32 E2E positive, every category/horizon E2E >= 0.98x;
- decode within 2%;
- 128/512/1K/4K milestone reporting;
- model load excluded and exact command recorded;
- all allocations freed, repeated sessions deterministic, and no hidden D2H
  control boundary;
- queue/sidecar/scratch bytes and context-capacity effect stated.

## Stop rules and closed work

Stop or change premise when:

- a prequantized MMQ body is not at least 2x on the primary expert shapes;
- inclusive pack + MMQ + repair loses the body advantage;
- exact repair requires a full-family sidecar or unbounded queue;
- the replacement layout cannot provide an exact decode/fallback path;
- a policy passes a short shape but fails the complete category/heldout lane;
- a post-promotion profile moves the bottleneck to another family;
- the remaining family has less than a 5% perfect-removal ceiling.

Do not repeat:

- expert-major compensated F16 component or layer-family bisection;
- arbitrary layer subsets selected from prompt outcomes;
- one-scale-per-32 one-plane direct Q8_1 gate/up promotion;
- scalar grouped gate/up C4/C8/C16;
- independent WMMA wave widening;
- the D8 integer-WMMA 128x32 selected consumer; it is BF16-byte exact but
  regresses the actual-weight leaf **6.902 -> 8.179 ms**;
- per-block LDS unpack/staging without complete tile reuse;
- X8 exact decode via local256, direct raw addressing, raw LDS staging, output
  widening alone, or dynamic X8-to-T16 reconstruction;
- per-dispatch T16-to-raw/X8 shared transposes. Direct T16 MMQ addressing is
  already implemented and is not part of this closed work;
- adjacent-column T16 pair decode through row-shift DPP; exact halved packed
  byte-load instructions regress the natural-M512 gate/up leaf **22.7%**;
- blanket non-temporal weight loads for rows>1 without a new cache/traffic
  profile; both the prior gfx1151 control and the production-geometry T16
  Q4 `slc dlc` screen regress decisively, and the Q6-specific qmicro screen
  also regresses **2.121%**;
- Q6 K64 multi-stage synchronization: doubling live stages raises VGPR
  **88 -> 128**, doubles LDS, and regresses the traced family **14.54%**;
- direct-wave Q4 register weight prefetch: the second decoded K32 record raises
  VGPR **88 -> 104** and regresses the actual gate/up leaf **6.885%**;
- Q6 paired-scale metadata decode: removing the duplicate FP16 multiplier load
  leaves the traced family flat/slower, so metadata traffic is not the limiter;
- Q6 WMMA result-metadata half-wave broadcast: replacing same-address LDS reads
  with two wave shuffles per result row is exact but regresses the actual
  natural-M512 leaf **4.5149 -> 6.3418 ms (+40.46%, 0/21 wins)**;
- Q6 WMMA compact shared weight metadata: shrinking the logical staged record
  **40 -> 36 bytes/column** is exact, but LDS allocation remains **5,120 B**
  after rounding and exact scale reconstruction regresses the actual
  natural-M512 leaf **4.5137 -> 4.8221 ms (+6.834%, 0/21 wins)**;
- Q6 static-upper sentinel grids and launch-bounds occupancy hints: unused
  workgroups are effectively free, while `(128,2)` emits the same
  VGPR/LDS/scratch resources and no repeatable speed change as `(128,1)`;
- shared pack8 gate/up+SiLU fusion: the exact actual-weight leaf improves
  **14.56%**, but the already-hidden low-priority branch regresses pp512
  **580.394 -> 577.374 tok/s (-0.52%, 1/7 wins)** and adds **4.088 ms** at
  the paired median wall;
- qgroup9, paired-row exact attention, or row2 score materialization;
- dense M256 attention-row merging or generic M256 online attention without a
  fused causal primitive;
- single-wave qrow4 two-head GQA fusion; exact K/V reuse regresses all measured
  512/1K/4K diagnostic lengths;
- nine-wave qrow4 GQA token-tile sharing; exact current/cache K/V reuse
  regresses every pp512 slice and totals **0.706x** retained;
- fused-four source-F16 F32 row-scale restore; reducing **192 -> 48** launches
  regresses the exact 48-layer sequence **3.474 -> 6.114 ms (+76.0%)**;
- AOTriton Laguna head-dim-128 adaptation without a newly supported geometry;
- graph replay or launch-count work while span-minus-sum is sub-percent.

## Expected implementation surfaces

Likely reused or extended files:

- `hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_q8_1_selected_prefill.{hip,py}`
- `hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_mmq_prefill.{hip,py}`
- `hipengine/kernels/hip_gfx1100/quant/gguf_k_t16_selected_prefill.{hip,py}`
- `hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_t16_selected_prefill.{hip,py}`
- `hipengine/kernels/hip_gfx1100/quant/gguf_x8_selected_gemv.{hip,py}`
- `hipengine/kernels/hip_gfx1100/linear/laguna_f16_projection.{hip,py}`
- `hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.hip`
- `hipengine/runtime/laguna_moe.py`
- `hipengine/runtime/laguna_gguf_runner.py`
- `hipengine/loading/laguna_gguf_materialize.py`

Likely focused tests/harnesses:

- `tests/test_gguf_q4_k_q8_1_selected_prefill.py`
- `tests/test_laguna_q4_k_x8_exact_decode_bench.py`
- `tests/test_gguf_q8_0_mmq_prefill.py`
- `tests/test_laguna_moe_gpu.py`
- `tests/test_laguna_f16_projection.py`
- `tests/test_laguna_kv_attention.py`
- `tests/test_laguna_gguf_runner.py`
- `scripts/laguna_prefill_profile.py`
- `scripts/laguna_routing_replay.py`
- `scripts/laguna_q4_k_x8_exact_decode_bench.py`
- `scripts/laguna_grouped_down_category_bench.py`

Create a new calibration or category harness only when the existing generic
Laguna harness cannot express the three-lane exact/shipping/candidate contract.
Temporary selectors must receive a `docs/REFACTOR.md` removal trigger when they
land.

## Definition of done

The campaign is complete when one of these conditions is documented:

1. hipEngine reaches the LAP-BW0 roofline-derived pp512 target under its
   retained quality/lifecycle protocol, with no 128/1K/4K or category
   regression, and each mapped streaming family reaches at least 70% of the
   same-host achievable read ceiling (or has a measured non-bandwidth limiter);
   or
2. every mapped family has a retained non-regressive route or a prospectively
   rejected new arithmetic premise, a fresh profile explains at least 99.5% of
   remaining wall, and the residual blocker is explicit enough to require a new
   architecture rather than more local tuning.

Matching **344.56 tok/s** is the first external floor. Because the Vulkan
control uses a different token stream, F16 KV, and backend numerical policy,
“beat llama.cpp” still requires a matched timing/token/KV contract or an
explicit qualification. The engineering goal is now stronger: reduce the
current conservative **0.887-second** pp512 wall from the achieved 500 tok/s
production gate toward the 700 tok/s stretch, then continue until the major
streaming families are close to the same-host bandwidth roof while preserving
hipEngine's stricter correctness contract.

## Evidence index

Primary Laguna evidence:

- `benchmarks/results/2026-07-27-gfx1151-laguna-q4-p8-pair-shared-prefetch-rejected.json`
- `benchmarks/results/2026-07-27-gfx1151-laguna-q4-raw-prefetch-p4-rejected.json`
- `benchmarks/results/2026-07-27-gfx1151-laguna-q4-paired-silu-pack-rejected.json`
- `benchmarks/results/2026-07-27-gfx1151-laguna-q4-p8-row64-current-body-rejected.json`
- `benchmarks/results/2026-07-27-gfx1151-laguna-q4-p8-nontemporal-rejected.json`
- `benchmarks/results/2026-07-27-gfx1151-laguna-q4-p8-metadata-prefetch-rejected.json`
- `benchmarks/results/2026-07-27-gfx1151-laguna-q4-raw-prefetch-p8-production.json`
- `benchmarks/results/2026-07-27-gfx1151-laguna-q4-raw-prefetch-p8-candidate.json`
- `benchmarks/results/2026-07-27-gfx1151-laguna-q4-qmicro-metadata-lds-rejected.json`
- `benchmarks/results/2026-07-27-gfx1151-laguna-q4-layer-risk-absolute-rejected.json`
- `benchmarks/results/2026-07-27-gfx1151-laguna-q4-layer-risk-quality-pending.json`
- `benchmarks/results/2026-07-27-gfx1151-laguna-q4-role-risk-calibration-heldout.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-q4-d4-selective-repair-rejected.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-q6-qmicro-planar-production.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-q6-selected-down-integer-wmma-candidate.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-q6-selected-down-integer-wmma-production.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-q6-integer-wmma-hoist-activation-candidate.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-q6-integer-wmma-hoist-activation-production.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-q6-qmicro-planar-candidate.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-q6-qmicro-planar-leaf.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-q6-qmicro-permute-production.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-f16-boundary-fusion-production.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-attention-preappend-production.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-attention-preappend-candidate.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-attention-qrow3-rejected.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-f16-qkv-grouping-rejected.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-q4-gate-rowfast-grid-rejected.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-mmq-combine-candidate.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-mmq-combine-production.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-fused-silu-pack-candidate.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-fused-silu-pack-production.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-q6-down-rows128-heavy-rejected.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-q4-down-cols128-rejected.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-q6-down-scheduler-controls-rejected.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-q6-down-cols128-rejected.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-q6-down-local256-rejected.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-q6-down-k64-stage-rejected.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-selected-weight-traffic-ledger.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-gate-dpp-pair-decode-rejected.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-swa-gqa-tiled-rejected.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-router-token-tile8-production.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-f16-norm-direct-candidate.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-f16-scale-restore-fused4-rejected.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-router-token-tile8-candidate.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-parallel-prefix-scan.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-parallel-compact-production.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-q6-down-rows64-production.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-q6-down-rows64-candidate.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-q6-down-shared-weight-local64-rejected.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-q4-pack8-shape-policy-production.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-q4-pack8-shape-policy-candidate.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-q6-dense-wmma16x32-production.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-q6-dense-wmma16x32-candidate.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-gate-direct-local256-rejected.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-gate-wavecols-geometry-rejected.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-down-wavecols-production.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-down-wavecols-candidate.json`
- `benchmarks/results/2026-07-26-gfx1151-laguna-wavecols-production.json`
- `benchmarks/results/2026-07-25-gfx1151-laguna-small8-hybrid-rejected.json`
- `benchmarks/results/2026-07-25-gfx1151-laguna-weight-meta-hoist-rejected.json`
- `benchmarks/results/2026-07-25-gfx1151-laguna-weight-soa-rejected.json`
- `benchmarks/results/2026-07-25-gfx1151-laguna-persistent-expert-rejected.json`
- `benchmarks/results/2026-07-25-gfx1151-laguna-swa-wmma-tiled-rejected.json`
- `benchmarks/results/2026-07-25-gfx1151-laguna-swa-keysplit-rejected.json`
- `benchmarks/results/2026-07-25-gfx1151-laguna-down-rowvec-production.json`
- `benchmarks/results/2026-07-25-gfx1151-laguna-down-rowvec-candidate.json`
- `benchmarks/results/2026-07-25-gfx1151-laguna-gate-rowvec-production.json`
- `benchmarks/results/2026-07-25-gfx1151-laguna-gate-rowvec-candidate.json`
- `benchmarks/results/2026-07-25-gfx1151-laguna-swa-sourcequal-production.json`
- `benchmarks/results/2026-07-25-gfx1151-laguna-prefill-350-production.json`
- `benchmarks/results/2026-07-25-gfx1151-laguna-prefill-350-production-default.json`
- `benchmarks/results/2026-07-25-gfx1151-laguna-prefill-350-production-trace.json`
- `benchmarks/results/2026-07-24-gfx1151-laguna-prefill-lap0-control.json`
- `benchmarks/results/2026-07-23-gfx1151-laguna-llamacpp-vulkan-pp512-profile.json`
- `benchmarks/results/2026-07-23-gfx1151-laguna-swa-qrow2-online-retained.json`
- `benchmarks/results/2026-07-24-gfx1151-laguna-expert-major-wmma-screen.json`
- `benchmarks/results/2026-07-24-gfx1151-laguna-expert-major-wmma-category-rejected.json`
- `benchmarks/results/2026-07-24-gfx1151-laguna-expert-major-wmma-component-rejected.json`
- `benchmarks/results/2026-07-24-gfx1151-laguna-expert-major-wmma-layer-family-rejected.json`
- `benchmarks/results/2026-07-24-gfx1151-laguna-q4-k-x8-mmq32-layout-retained.json`
- `benchmarks/results/2026-07-24-gfx1151-laguna-q4-k-x8-mmq32-live-row-retained.json`
- `benchmarks/results/2026-07-25-gfx1151-laguna-q4-k-x8-exact-decode-rejected.json`
- `benchmarks/results/2026-07-25-gfx1151-laguna-q4-k-t16-mmq32-retained.json`
- `benchmarks/results/2026-07-25-gfx1151-laguna-prefill-350-category.json`
- `benchmarks/results/2026-07-25-gfx1151-laguna-prefill-350-d8-category.json`
- `benchmarks/results/2026-07-23-gfx1151-laguna-prefill-ar-o1-q8-dp4a-category-rejected.json`
- `benchmarks/results/2026-07-23-gfx1151-laguna-f16-wmma-comp-swa-retained.json`
- `benchmarks/results/2026-07-23-gfx1151-laguna-f16-library-ceiling.json`

Internal transfer evidence:

- [`GGUF-Q3-OPT.md`](GGUF-Q3-OPT.md)
- `benchmarks/results/2026-07-20-gpu1-q3-guarded-d4x3-mmq-prefill.json`
- `benchmarks/results/2026-07-20-gpu1-q3-exact-q8-row-reuse-prefill.json`
- `benchmarks/results/2026-07-20-gpu1-q3-exact-iq3-rowbatch4-prefill.json`
- `benchmarks/results/2026-07-15-gfx1100-gguf-q8-mmq-source-audit.json`

The complete historical Laguna support and prior-campaign record remains in
[`LAGUNA.md`](LAGUNA.md). This file owns the next optimization order and its
stop conditions.
