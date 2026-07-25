# Laguna S 2.1 Prefill Attack Plan

Last updated: 2026-07-25

Status: active successor to the completed LPF/AR-O campaign in
[`LAGUNA.md`](LAGUNA.md). The prior bounded tasks are closed; this plan starts a
new arithmetic and data-layout campaign. It does not reactivate the rejected
expert-major F16 runtime routes. LAP-1 is complete with a direct resident-T16
MMQ32 consumer. The first LAP-2 three-plane/guarded/exact-repair primitives are
implemented and traced. The original one-scale-per-32 one-plane integrated
candidate crossed 350 tok/s but failed the complete category quality gate.
The repaired gate/up route uses one FP32 scale per 16 activations in the same
160-byte block and widens the Q4 consumer to 128 columns x 32 rows. The clean
complete category gate admits it at maximum KL 0.0407248, 317/320 top-1, and
2.615x aggregate natural-prompt prefill. The four compounded routes are now
gfx1151 package defaults. Clean selector-unset production publication passes at
**354.820 tok/s** median (**353.421/355.584/354.820**), with every pp512
sample above the 350 tok/s target. A cached-only production trace independently
reproduces **354.763 tok/s** and names every intended kernel family. The 350
milestone is complete; the campaign remains active for post-milestone roofline
and long-prompt work. The execution order below was re-audited on 2026-07-25
after correcting the Vulkan comparator geometry and adding an absolute
bandwidth target.

## Outcome

Close the resident c=1 Laguna S 2.1 Q4_K_M prefill gap on Radeon 8060S/gfx1151
without weakening hipEngine's quality, fallback, memory, or plugin contracts.
The primary external control is the current local llama.cpp Vulkan build at
`c0bc8591e8815c63cb01dd3f051a8b0df02501c9`, which measures
**344.56 +/- 3.16 tok/s** at pp512. The pre-campaign hipEngine
matrix512/attention128 default measured **76.226 tok/s**, a **4.520x** gap.
The quality-admitted production default now measures **354.820 tok/s**
selector-unset, **4.655x** the old row and **2.978%** above the Vulkan control.

That Vulkan row is now a compatibility floor, not the optimization ceiling.
Strix Halo has a **256 GB/s** theoretical LPDDR5X roof and the existing
local/reference large-read evidence is about **221 GB/s**. The first exact
active-byte lower bound puts current selected gate/up at only **9.85 GB/s**,
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

The remaining work is no longer strictly numeric by `LAP-*` label. First
attribute the shipping KL debt, then take the already-measured hipBLASLt
source-F16 opportunity, then the low-risk dense/shared quant family. In
parallel, finish real-input DS/repair calibration before promoting selected
gate/up/down. Cooperative tiled attention remains last. Submission and graph
work remain deferred.

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
arithmetic/performance control, while the direct T16-native sibling is the
retained LAP-1 candidate primitive. Neither has a runtime route yet.

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
pass. One bounded replacement screen remains open:

- **T16-lite:** keep T16's 16-column Q4 nibble interleave and FP16 `d/dmin`, but
  retain the source-packed 6-bit scale/min field. Per 16 columns/K256 this is
  `2048 + 32 + 32 + 192 = 2304 bytes`, byte-neutral with raw/X8 instead of
  T16's 2,368 bytes.
- **X16:** the cheaper control, grouping 16 source blocks without expanded
  scale/min metadata.

Both are compatible with the closed-work rules: neither is X8 direct decode,
a sidecar, nor a per-dispatch T16→raw transpose. Admit at most one
materializer/decode/MMQ screen after the higher-value library/dense work. It
must beat current T16 on both exact c1 decode and prefill GB/s before replacing
the resident format. The prior 4.69x direct-X8 result is recorded as an
untuned-kernel failure, not proof that byte-neutral layouts are intrinsically
bad.

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
| LAP-BW0 / LAP-Q0 | Deferred by integrated result | The absolute bandwidth/quality-debt ledger remains useful for further roofline work, but it no longer blocks testing the now-above-target compounded candidate. |
| LAP-6 | Admitted gfx1151 default | Torch-free, row-scaled hipBLASLt runs all five source-F16 projections on rows>1 real inputs with no added scratch; exact GEMV/tiled routes remain rollback. |
| LAP-5 | Admitted gfx1151 default | Resident Q4 pack8 and raw Q6 use 64x16 wave32 WMMA consumers. Q4 is BF16-bit identical to the raw-Q4 WMMA oracle; Q6 passes its CPU-reference gate and removes the traced 0.365-second dense/shared family bottleneck. |
| LAP-2 calibration / LAP-3 / LAP-4 | Admitted gfx1151 defaults | The original D4-gate/D4-down route reached **355.273/355.721 tok/s** but was rejected at max KL **0.0767056**. Same-byte D8 gate/up plus D4 down passes the clean complete category gate at max KL **0.040724836**, **317/320** top-1, **2.615x** aggregate natural-prompt prefill, flat decode, and exact lifecycle recovery. Its pre-admission pp512 samples were **353.951/356.082/356.473 tok/s**, token 2930. |
| Production publication | Complete | Clean selector-unset M512/attention128 pp512 samples are **353.421/355.584/354.820 tok/s** (median **354.820**), all token 2930. The fail-closed publication binds those timings to the retained 320-step quality artifact and exact package capabilities. Cached-only tracing independently measures **354.763 tok/s** and executes D8-pack/128-column gate-up MMQ, D4 Q4/Q6 down MMQ, Q4/Q6 WMMA dense/shared, scaled hipBLASLt, and online global/SWA attention with zero tracked allocations after close. |
| LAP-7–LAP-8 | Deferred | Reprofile after linear work; attention starts only at its measured threshold. |

## Post-350 campaign — 500 production gate, 700 stretch

The 350 tok/s milestone proves the compounded production package, but it is
not a roofline result. The clean cached production trace measures **1.443218
seconds** synchronized pp512 wall, **1.440122 seconds** kernel span, and
**1.427220 seconds** kernel sum. Only **15.997 ms / 1.11%** of wall lies
outside the summed kernels, so graphs, Python removal, and submission tuning
are explicitly closed until a later trace changes that conclusion.

The next primary gate is **at least 500 tok/s selector-unset production
pp512**: at least three clean repetitions, median and every sample at or above
500 tok/s, with the same model/quant/KV/queue policy and all existing
correctness, quality, decode, determinism, memory, and lifecycle gates. The
stretch gate is **700 tok/s** under the same contract. The 700 row is a target,
not a performance claim, until LAP-BW0 supplies locked-clock physical traffic
and achievable-bandwidth evidence.

| Current production family | pp512 kernel time | Kernel-sum share | Remaining decision |
| --- | ---: | ---: | --- |
| Selected D8 Q4 gate/up | **581.799 ms** | **40.76%** | Multi-K, 64-row, local256, and coalesced-raw screens are rejected below. Park simple body tuning; retain only a bounded hybrid-large-expert screen after new trace/counter evidence. |
| Selected D4 Q4/Q6 down | **276.169 ms** | **19.35%** | Carry the winning expert schedule into Q4 and Q6 down, then reprofile the combined 60.11% expert window. |
| Global + SWA attention | **274.724 ms** | **19.25%** | The first LAP-7 step retains four-query K/V reuse on complete M128 tiles at **365.048 tok/s** dirty selector-unset median. Clean confirmation/reprofile it, then continue toward the `KVLiveSpans`-aware M16-query x K64-key tiled path. |
| Scaled hipBLASLt source-F16 | **130.373 ms** | **9.13%** | Freeze unless a new trace exposes conversion overhead; this is already at the measured inclusive library ceiling. |
| Q4/Q6 WMMA dense/shared | **70.098 ms** | **4.91%** | Freeze. It is only about 7.2 ms behind the homologous Vulkan family and cannot move the next milestone materially. |
| All remaining named/other kernels | **94.058 ms** | **6.59%** | Do not tune router, norm/RoPE, reductions, KV write, or tails without a new >=5% family ceiling. |

The current trace gives concrete Amdahl checkpoints, not performance claims:

- **1.25x/1.5x/2x** combined selected-expert throughput models to about
  **403/442/505 tok/s** with every other family unchanged.
- Reducing global+SWA attention from **274.7 ms** toward **80 ms** models to
  about **410 tok/s** with every other family unchanged.
- Combining **2x** selected experts with an **80 ms** attention window models
  to about **625 tok/s**. Reaching 700 requires a stronger measured expert
  bandwidth result, additional attention reduction, or both.
- The frozen routing byte lower bound makes the current gate/up window about
  **62.3 GB/s encoded-weight-equivalent**, only **28.2%** of the existing
  221 GB/s same-host read anchor. This is evidence of likely headroom, not a
  controller-bandwidth claim; LAP-BW0 must replace it with encoded and physical
  traffic plus in-load clocks.

The quality contract remains binding. The production route sits at maximum KL
**0.040724836**, leaving only **0.009275164** below the 0.05 ceiling. The
rejected D4 gate candidate already showed that another approximate shortcut can
hold 355+ tok/s while failing quality at KL **0.0767056**. Prefer exact
data-movement/scheduling wins and preserve K accumulation order; run LAP-Q0
before admitting any new approximation.

Immediate execution queue:

1. Publish the post-admission LAP-BW0 ledger from the final all-layer trace:
   locked/recorded clocks, per-family encoded and physical bytes, and
   counter-derived traffic. Retire the pre-admission **78.27 ms/layer versus
   52.80 ms layer-1** bridge instead of scaling it into new forecasts.
2. Clean-confirm and reprofile the retained LAP-7 qrow4/M128 step, then
   continue cooperative tiled attention. Its primary residual gates are
   pp512, 1K, and 4K family wall plus full `KVLiveSpans`/causal/ring
   correctness and the complete quality lane.
3. Use the next trace/counters to decide one bounded hybrid expert screen:
   retain 128x32 for <=32-row experts and route only >32-row experts through a
   64-row consumer. Do not build it without per-route evidence that the large
   groups individually win.
4. Apply any winning expert schedule to Q4/Q6 down, then run clean selector-unset
   pp512 and the complete category/decode/determinism/lifecycle gate. Retain
   every exact same-suite non-regressive improvement; 500 tok/s closes the next
   production milestone.
5. Raise the currently hard-capped matrix capacity and screen **1024/2048**
   chunks for 1K/4K prompts while retaining independent 128-row attention
   slices. Publish scratch/context admission and exact cursor/KV/lifecycle
   evidence. This receives no pp512 credit.
6. Screen byte-neutral T16-lite/X16 only after those larger opportunities. Its
   2.778% Q4 metadata saving is a permanent but roughly **1.1% pp512**
   gate/up-byte upper bound at the current family share.

Post-350 exclusions:

- do not spend a campaign round on source-F16, dense/shared, graphs,
  submission, router, norm/RoPE, or tails without a new trace reopening them;
- do not retry the rejected raw-sum D8 or D4-gate quality shortcuts;
- do not add a duplicate resident expert-weight sidecar or weaken c=1 exact
  decode to buy prefill;
- do not claim 500 or 700 from a leaf, explicit session selector, dirty tree,
  single sample, or incomplete quality lane.

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

Fifth post-350 screen: **retained candidate pending clean confirmation**.
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
After installing the M128-qualified gfx1151 defaults, the dirty
selector-unset confirmation measured **365.048 tok/s** median with a
**363.735** minimum versus the paired qrow2 **352.920**, again always token
2930. This is not yet the new production claim: commit the candidate and run
the clean selector-unset production gate first. Evidence:
[`2026-07-25-gfx1151-laguna-attention-qrow4-candidate.json`](../benchmarks/results/2026-07-25-gfx1151-laguna-attention-qrow4-candidate.json).

Production evidence:

- [`2026-07-25-gfx1151-laguna-prefill-350-production.json`](../benchmarks/results/2026-07-25-gfx1151-laguna-prefill-350-production.json)
  is the retained publication artifact.
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
with packed-dot ISA. The guarded LAP-2 primitive has since landed; follow the
revised LAP-BW0/LAP-Q0 → LAP-6 → LAP-5 execution queue before selected-family
calibration and integration. No threshold, small-row prototype, X8
materializer, or runtime default is retained.
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
D4x3 is **1.289x** retained at M128 and **2.510x** at M512. The task remains
open only for real gate/up/down captures, FP32 DS/range screening, threshold and
queue occupancy, and complete model admission.

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
The clean complete category gate passes at maximum KL **0.040724836** and
**317/320** top-1. It improves aggregate natural-prompt prefill
**70.192 -> 183.563 tok/s (2.615x)**, h16/h32 E2E
**1.552x/1.322x**, keeps decode within 0.01%, passes Poolside at KL
**0.0000175** with equal top-1, and returns tracked allocations exactly to
zero. Exact reconstructed-sum pp512 repeats at
**353.951/356.082/356.473 tok/s**, always token **2930**. The tempting raw-sum
variant was faster but failed quality and was removed. gfx1151 now defaults to
this D8 gate/up route and the admitted D4 down route. The clean selector-unset
publication closes the production check at **354.820 tok/s** median, with all
three samples above 350 tok/s.

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

### LAP-7 — build `KVLiveSpans`-aware cooperative tiled attention

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

Planning checkpoint: reduce current measured attention toward **0.08 seconds or
less**, exceed **310 tok/s**, and enter the 10%-of-Vulkan control band.

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
| Streaming-family floor | >=70% of measured read roof | About 155 GB/s if the same-host anchor is 221 GB/s; report each mapped family. |
| Roofline system target | Set by LAP-BW0 | Exact active-byte ledger plus non-streaming wall; the review's ~650–750 tok/s range is a hypothesis until measured. |

The production target is achieved. The stronger streaming/roofline rows remain
post-350 optimization targets, not qualifications on the retained 354.820
tok/s production claim.

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
- per-block LDS unpack/staging without complete tile reuse;
- X8 exact decode via local256, direct raw addressing, raw LDS staging, output
  widening alone, or dynamic X8-to-T16 reconstruction;
- per-dispatch T16-to-raw/X8 shared transposes. Direct T16 MMQ addressing is
  already implemented and is not part of this closed work;
- blanket non-temporal weight loads for rows>1 without a new cache/traffic
  profile; the prior gfx1151 control regressed rows>1 to 0.68x;
- qgroup9, paired-row exact attention, or row2 score materialization;
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
explicit qualification. The engineering goal is now stronger: remove the
measured 5.23-second deficit, then continue until the major streaming families
are close to the same-host bandwidth roof while preserving hipEngine's stricter
correctness contract.

## Evidence index

Primary Laguna evidence:

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
