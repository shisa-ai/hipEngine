# Laguna S 2.1 Prefill Attack Plans

Last updated: 2026-07-30

## Active W7900 / gfx1100 UD-Q2_K_XL prefill port

Status: **WPF-H1 through WPF-H5B changed-arithmetic runtimes are rejected;
exact H5L weight-major Q5, H5I ordered-Q6, and H5J IQ row ownership are
production; H5H closes larger Q5 tiles, H5K closes larger resident-IQ3 row
batches, and 16K+ remains deferred**. This section is the
authority for the Radeon Pro W7900 / `hip_gfx1100` Laguna `UD-Q2_K_XL` port.
The longer gfx1151/Q4 campaign record begins below and remains evidence, not a source of automatic defaults or tile
geometry.

H5E compounds WPF-1T's exact Q5/Q6 output-column tiling,
matrix512/attention128, pair16 grouped IQ projection, C256-qualified qrow4 SWA,
and H5D's transient exact-F32 Q5 producer. Production-ordered local128
8x4/4x16/8x8/16x4 consumers now own all eight measured roles while preserving
each output's K/FMA/wave/store sequence. The final-source role gate moves H5D
weighted event/wall **1,085.630/1,040.166 -> 951.876/961.993 ms
(-12.320%/-7.515%)**; 1x64/2x32 are removed as universal regressions. Clean
selector-unset production is **184.997/172.104/131.496 tok/s** at 512/1K/4K,
improving H5D by **+3.166%/+2.941%/+1.944%**. H5F retains 12x4 only for F32
N48. H5G adds exact 8x10/16x5/8x12/12x8 on five roles; its strong gate cuts
H5F **8.639%/7.479%** by event/wall and clean publication reaches
**188.393/175.042/132.743 tok/s**. H5I reuses the same plane for four exact-Q6
roles and publishes **191.713/178.080/134.411 tok/s
(+1.762%/+1.736%/+1.256%)** over H5G. All 48 hidden boundaries, logits,
K/V/live spans, repeats, and lifecycle are byte-exact at KL 0; every publication
row is exact.

The apples-to-apples external target remains same-model, same-512-ID,
context4096, direct-M512, FlashAttention, BF16-K/V, one-queue llama.cpp HIP at
**694.184 tok/s** and token **2930**. The retained H5L package reaches
**237.956 tok/s**, token **2930**, exact position 511, and full allocation
recovery: a current **2.917x** gap. The clean post-H4 **169.516 tok/s** row
remains the frozen attribution control. Same-revision Vulkan
native F16 is **56.274 tok/s**, and llama.cpp HIP server-native is a secondary
**649.321 tok/s** row. Cross-engine arithmetic is not a correctness oracle;
every candidate still passes hipEngine's complete quality gate.

The post-H4 cached exact trace records **3,001.692-ms** kernel sum in a
**3,016.780-ms** span across **1,477** dispatches. Only **15.087 ms (0.500%)**
is outside kernels. Q5 exact coltile owns **1,270.458 ms (42.325%)**, exact
selected IQ3/IQ4 down **557.091 ms**, exact attention **488.304 ms**, selected
gate/up **460.143 ms**, and Q6 exact coltile **157.073 ms**. This closes H5's
reprofile step and selects **WPF-H5A**: transient raw-Q5-to-F32 dequantization,
exact BF16-to-F32 activation widening, and F32 rocBLAS SGEMM. It avoids the
rejected Q8_1/F16 operand rounding, adds no persistent weight sidecar, fails
closed to exact coltile, and still requires the complete 18-prompt/576-step
quality lane because GEMM reassociates reduction.

The standalone H5A leaf now passes that first bound. A role-qualified policy
keeps the regressive F32 K3072/N48 gate on exact coltile and selects the
exact-value F32 stack for the other seven shapes. Across the actual **235**
M512 Q5 calls, HIP-event medians move **1,256.936 -> 221.137 ms (5.684x,
-82.407%)** and synchronized wall moves **1,223.263 -> 231.966 ms (5.273x)**.
Every candidate output is finite at maximum mean KL **1.59e-9**, maximum row KL
**5.79e-8**, and top-1 **100%**; raw-Q5 F32 reconstruction and BF16 widening
are bit-exact. The selected stack is still **3.751x** llama.cpp's matched Q5
trace and models the complete kernel sum at **1,952.371 ms**. Its default-off
owner allocates one admitted **195,035,136-byte** plane set and passes natural
M512 at KL **0.0003742**, top-1 **100%**, deterministic complete state, and
exact teardown. The binding 18-prompt/576-step lane nevertheless rejects SGEMM
reassociation at maximum KL **1.143627 > 0.05** despite **564/576 (97.917%)**
top-1, deterministic repeats, lifecycle recovery, and diagnostic prefill
**152.359 -> 202.707 tok/s (1.330x)** with all categories positive. Remove the
owner/workspace/capabilities/tests and retain exact production plus standalone
leaf evidence.

H5B screens the existing complete-`KVLiveSpans` BF16-cache-to-F32 route on
W7900. The basic eight-QK/eight-PV composition regresses and is rejected;
packed two-call QK/PV plus wave32 softmax with gfx1100 QK algorithms **2/1/3**
at contexts **256/384/512** and PV algorithm **2** moves the selected-context
48-layer leaf **109.897 -> 62.655 ms (1.754x)**. Every selected global/SWA shape
wins; max-row KL is **1.10e-15**, top-1 **100%**, and max abs **4.84e-8**.
An explicit natural-M512 owner passes KL **0.000429**, token **2930**, complete
deterministic state/KV/`KVLiveSpans`, and teardown. Cached tracing observes
**144** complete widen/QK/softmax/PV stacks, keeps all 48 start-0 calls exact,
and moves attention **488.304 -> 60.669 ms (8.049x)** plus full kernel sum
**3,001.692 -> 2,603.520 ms (-13.265%)**.

The binding gate deterministically extends each of the same 18 committed natural
prompts to M512 within its train/heldout split and observes all **10,512**
expected changed-association launches with the six measured algorithm pairs.
Runtime promotion is rejected at maximum KL **0.444675 > 0.05** despite
**564/576 (97.917%)** top-1, deterministic repeats, lifecycle recovery, and
prefill **165.555 -> 190.103 tok/s (1.148x)** with every category positive.
Remove the gfx1100 capability/component policy, algorithm map, owner propagation,
generic map seam, and focused tests. Exact qrow4/M128 remains production; clean
512/1K timing does not run after the failure. H5C returns to Q5 with transient
exact-value expansion followed by a custom F32-weight reduction that preserves
production coltile K ownership, FMA order, and wave/cross-wave tree.

H5C/H5D clears the exact leaf and first runtime gates. One local64 producer
materializes only the current raw-Q5 projection as exact row-major F32;
local128 **8x4/4x8** consumers preserve each output's coltile K ownership, FMA
sequence, wave32 tree, serial wave-0..3 sum, and final BF16/F32 store. H5E
extends the same template to **4x16/8x8/16x4** with 64 accumulators/thread.
Rows17/33 tails and all eight actual roles remain byte-exact. The final-source
235-call policy improves H5D **1,085.630 -> 951.876 ms (1.141x)** by events and
**1,040.166 -> 961.993 ms (1.081x)** by wall; a 15-repeat N6144 adjudication
confirms 16x4 at **1.075x/1.122x**. The runtime still owns one projection-local
**150,994,944-byte** plane with no sidecar. Package-default M512 is KL0 and
byte-exact across all 48 boundaries, logits, K/V, repeat, and teardown. Cached
tracing observes exactly **235** producers and 235 consumers; the retained new
bodies are local128/VGPR136/SGPR128/LDS1024/scratch0. H5F's constant-48 screen
compiles at VGPR104/LDS1024/scratch0 and rejects every broad role, but a stronger
borderline gate retains 12x4 for F32 N48 at **1.187%/0.496%** event/wall. H5G
retains 8x10/16x5/8x12/12x8 on five roles; the final strong subset moves
**892.586/896.357 -> 815.474/829.319 ms (-8.639%/-7.479%)** by event/wall.
Constant-80/96 bodies are VGPR168/200, LDS1536, scratch0. Complete state remains
KL0 and clean package throughput is **188.393/175.042/132.743 tok/s**. H5H
rejects constant-112 despite scratch-free VGPR232 and constant-128 after both
universal role regressions and a VGPR256/28–52 B scratch cliff. All seven bodies
are removed ([H5H rejection](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-f32-ordered-register-boundary-rejected.json)).

The retained H5G request segment contains **1,720 dispatches / 2,667.034 ms**
in a **2,702.091-ms** kernel span. Disjoint physical families are exact Q5
producer+ordered consumer **920.633 ms**, IQ3/IQ4 down **560.642 ms**,
IQ2/special-IQ3 gate/up **470.116 ms**, exact attention **468.533 ms**, exact
Q6 coltile **177.047 ms**, and all remaining request kernels **70.063 ms**.
Against the matched llama.cpp trace, their named gaps are **861.682/405.146/
65.138/446.807/162.128 ms**. Q5 remains largest but H5H closes its exact
geometry; the exact attention schedules and changed-association transfer lanes
are also closed. **WPF-H5I** therefore reuses the existing 150,994,944-byte
serial plane for exact raw-Q6-to-F32 expansion plus the production-ordered
F32-weight consumer. Its initial all-geometry screen and 15-repeat/five-launch
adjudication retain only `8x4`, `16x4`, and `16x5`; seven non-owning Q6
composites are removed. Across all seven configurations/**146 calls**, strong
producer-inclusive event timing moves **194.758 -> 119.751 ms (1.626x,
-38.513%)** and synchronized wall moves **189.722 -> 121.353 ms (1.563x,
-36.037%)**. Four roles select candidates while BF16 K9216/K12288 and F32
N9216 retain exact raw coltile. Q5 and Q6 reuse the same **150,994,944-byte**
plane/library, adding no allocation. Complete M512 state is KL0/byte-exact
through all boundaries/logits/KV/live spans/repeat/teardown. Cached tracing
records **143** Q6 producers, **143** ordered consumers, and **3** raw-coltile
fallbacks, moving Q6 **177.047 -> 110.170 ms (-37.774%)** and request kernel
sum **2,667.034 -> 2,600.260 ms (-2.504%)**. Clean selector-unset 512/1K/4K
promotes **191.713/178.080/134.411 tok/s**. The reconciled H5I trace now assigns
Q5 **922.619 ms**, IQ3/IQ4 down **556.749 ms**, attention **471.150 ms**,
gate/up **469.311 ms**, Q6 **110.170 ms**, and remaining **70.261 ms**. IQ down
retains a **401.254-ms** matched gap. H5J's strict IQ3 leaf decodes each fixed
segment once per active expert/output block and replays the unchanged eight-row
phases. A generated one-BF16-ULP RED rejects the first separately compiled IQ4
constant-K body; the corrected wrapper launches the retained exact physical
body at local32. The final actual-weight/routing screen wins every **45 IQ3 + 2
IQ4** layer on both clocks. IQ3 event timing moves **541.137 -> 491.481 ms
(-9.176%)**, IQ4 **26.137 -> 8.696 ms (-66.730%)**, and combined selected down
**567.274 -> 500.176 ms (-11.828%)**; synchronized wall corroborates
**-11.746%**. Complete M512 state is KL0 and byte-exact through all 48 hidden
boundaries, logits, K/V/live spans, repeat, and teardown. Integrated tracing
selects exactly **45+2** calls, moves selected down **556.749 -> 497.145 ms
(-10.706%)**, and cuts request kernel sum **2,600.260 -> 2,532.020 ms
(-2.624%)** at unchanged **1,862** dispatches. Clean selector-unset 512/1K/4K
promotes **196.103/181.859/137.169 tok/s (+2.290%/+2.122%/+2.052%)** over H5I.
No allocation, workspace, or sidecar is added; every map/shape/key miss and
gfx1151 retain exact fallback. H5K's scratch-free rowbatch12 and rowbatch16
extensions then lose both clocks on all **45/45** actual IQ3 layers: event/wall
sums regress **+6.893%/+5.771%** and **+10.770%/+9.870%**. Every byte and
lifecycle matches; all temporary code is removed and H5J remains production at this checkpoint.
The unchanged request reclassifies to Q5 **919.697 ms**, IQ down **497.145**,
attention **468.007**, gate/up **466.826**, Q6 **110.293**, and remaining
**70.051 ms**. Q5 ordered consumers own **904.399 ms**, with two roles at
**741.721 ms (82.0%)**. H5L changes only linear workgroup ownership so each
weight tile is revisited across row batches before the next output tile. Six
material roles qualify while F32 N48/N72 retain H5G; the final-source 235-call
event/wall sums fall **44.857%/46.544%** with exact bytes. Complete M512 state
is KL0/byte-exact, integrated Q5 falls **919.697 -> 466.986 ms (-49.224%)**, and
request kernel sum falls **2,532.020 -> 2,074.261 ms (-18.079%)** at unchanged
**1,862** dispatches. Clean package-default 512/1K/4K promotes
**237.956/217.888/157.366 tok/s (+21.342%/+19.812%/+14.725% over H5J)** with no
new allocation or sidecar. Prior rowbatch16 spill, output tiling, and source MMQ
remain closed
([H5L production](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-f32-weight-major-production.json) ·
[H5L leaf](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-f32-weight-major-candidate.json) ·
[post-H5K residual](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-post-h5k-residual.json) ·
[H5K rejection](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-iq3-larger-resident-rowbatch-rejected.json) ·
[H5J production](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-iq-row-ownership-production.json) ·
[H5J leaf](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-iq-row-ownership-candidate.json) ·
[H5I production](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q6-k-f32-ordered-production.json) ·
[post-H5I residual](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-post-h5i-residual.json) ·
[H5I leaf](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q6-k-f32-ordered-candidate.json) ·
[post-H5G residual](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-post-h5g-residual.json)).

### Matched M512 kernel/module gap ledger

The preserved matched traces support a finer, exhaustive attribution than the
original four-family conservative view. The rows below are disjoint physical
kernel/module groups and reconcile exactly to both captured kernel sums. Delta
is **hipEngine minus llama.cpp HIP**, so a positive value is remaining
hipEngine work. This is summed GPU-kernel time, not wall time, and no benchmark
was rerun to produce this reclassification. This ledger and the H1-H5 order
below supersede the linked artifact's original conservative `gap_attribution`
and `campaign_revision` fields; its measurements, hashes, and source audit
remain the evidence inputs.

| Kernel/module | hipEngine ms | llama.cpp HIP ms | Delta ms |
| --- | ---: | ---: | ---: |
| Embedding lookup | 0.020 | 0.000 | **+0.020** |
| Q5_K projections | 1,274.342 | 58.951 | **+1,215.391** |
| Q6_K projections, including llama dequant/casts/GEMM | 156.111 | 14.916 | **+141.195** |
| Q8_0 dense-MLP projections | 7.484 | 0.206 | **+7.278** |
| Q4_K LM-head projection | 0.422 | 0.268 | **+0.154** |
| IQ2_XS selected gate/up | 430.054 | 404.903 | **+25.151** |
| IQ3_XXS special gate/up | 31.822 | 0.075 | **+31.747** |
| IQ3_XXS selected down | 531.414 | 152.380 | **+379.034** |
| IQ4_XS selected down | 26.428 | 3.146 | **+23.282** |
| Miscellaneous F32 matrix-vector | 0.000 | 0.012 | **-0.012** |
| Global + SWA attention core | 490.919 | 21.725 | **+469.194** |
| KV writes / `set_rows` | 0.694 | 0.894 | **-0.200** |
| BF16-to-F16 attention/KV conversion | 0.000 | 0.862 | **-0.862** |
| Q/K RMSNorm + RoPE | 7.201 | 7.791 | **-0.590** |
| Other RMSNorm; hipEngine post-attention add is fused | 1.671 | 1.608 | **+0.063** |
| Softplus attention gating | 3.462 | 1.796 | **+1.667** |
| MoE router logits | 12.382 | 7.505 | **+4.876** |
| MoE sigmoid correction / top-k | 0.863 | 0.449 | **+0.414** |
| MoE compact/inverse map vs ID scheduler | 22.908 | 21.716 | **+1.192** |
| MoE hidden gather / `get_rows` | 6.854 | 0.007 | **+6.847** |
| Standalone SiLU-times-gate activation | 0.278 | 4.340 | **-4.062** |
| MoE expert weighting/reduce | 3.105 | 6.904 | **-3.799** |
| Residual/elementwise adds | 1.401 | 4.098 | **-2.697** |
| Q8_1 activation quantization | 0.000 | 5.798 | **-5.798** |
| Runtime copy/fill kernels | 0.000 | 3.950 | **-3.950** |
| Argmax | 0.003 | 0.000 | **+0.003** |
| **Total summed kernel time** | **3,009.837** | **724.299** | **+2,285.538** |

The semantic matching uses demangled symbols, launch shapes, quant type, and
model-graph role. Fusion differs, so support work remains visible rather than
being silently charged to a favored family. In particular:

- hipEngine attention is **114.812 ms global + 376.107 ms SWA**. llama.cpp's
  same-symbol M512 path is **20.884 ms FlashAttention main + 0.841 ms stream-K
  fixup**; the trace has no trustworthy per-launch global/SWA marker, so only
  the combined attention delta is claimed.
- llama.cpp's **14.916-ms Q6_K stack** is **9.583 ms Q6 rocBLAS + 1.983 ms
  dequantization + 1.948 ms F32-to-F16 + 1.385 ms F16-to-F32 + 0.017 ms
  MMVQ**. The separate **7.505-ms** rocBLAS router work is charged to router
  logits, not Q6.
- hipEngine's selected gate/up and Q/K norm+RoPE bodies fuse work that llama.cpp
  launches separately. The standalone activation/support rows keep the two
  totals exhaustive despite that difference.
- llama.cpp exposes no distinct embedding or GPU argmax body in this capture.
  Conversely, its **3.950-ms** process-level copy/fill tail has no counterpart
  inside hipEngine's embedding-to-argmax request segment. These small negative
  rows are reconciliation items, not optimization targets.

Four rows explain almost the complete gap:

| Order | Direct target | hipEngine / llama.cpp | Exact delta | Gap share | Cumulative share | Modeled hipEngine kernel sum after parity |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Q5_K projections | **21.617x** | **1,215.391 ms** | **53.18%** | **53.18%** | **1,794.446 ms** |
| 2 | Global + SWA attention | **22.597x** | **469.194 ms** | **20.53%** | **73.71%** | **1,325.252 ms** |
| 3 | IQ3_XXS selected down | **3.487x** | **379.034 ms** | **16.58%** | **90.29%** | **946.218 ms** |
| 4 | Q6_K projections | **10.466x** | **141.195 ms** | **6.18%** | **96.47%** | **805.023 ms** |

The last column is a kernel-sum replacement model: it substitutes llama.cpp's
measured module cost into hipEngine one row at a time. It is not a tok/s or
wall-time forecast. The important planning fact is exact: these four measured
module deltas total **2,204.814 ms**, or **96.47%** of the complete
**2,285.538-ms** kernel gap.

The exact ledger supersedes the original conservative family rollup for
campaign ordering. Q5_K, attention, IQ3_XXS down, and Q6_K account for
**96.47%** of the complete kernel gap. llama.cpp still launches **2,824**
kernels versus hipEngine's **1,477**, yet sums only **0.724 s** versus
**3.010 s**. Launch count and Python submission are therefore not the cause.

For these four lanes, the audited llama.cpp HIP algorithm is the **primary
implementation candidate**, not a fallback delayed behind another exact-only
tiling campaign. Port it in-tree from `c0bc8591e`, cite the exact source files
and commit, preserve hipEngine's raw-pointer/four-axis registry and
`KVLiveSpans` contracts, and retain the current exact kernel as the unfused or
changed-arithmetic fallback. Copying the proven dataflow, tile ownership, and
WMMA strategy is encouraged; production promotion still requires hipEngine's
correctness and complete quality lanes.

The revised execution order is:

1. **WPF-H1 source-faithful Q5_K Q8_1/WMMA MMQ is rejected for runtime use.**
   The primitive reaches **16.094x** weighted leaf speedup and natural-prompt
   prefill reaches **1.348x**, but the complete lane fails at maximum KL
   **4.162014** with **561/576** top-1. Remove its runtime owner/workspace and
   keep only explicit primitive/ceiling evidence; do not stack it into H2.
2. **WPF-H2 source-faithful full-M512 FlashAttention is rejected for runtime
   use.** The retained standalone whole-tile primitive reproduces
   `flash_attn_ext_f16<128,128,8,8>` ownership behind complete `KVLiveSpans` and
   moves the weighted attention family **490.919 -> 21.719 ms (22.603x)** versus
   llama.cpp's **21.725-ms** trace. Runtime natural-prompt prefill improves
   **152.087 -> 156.219 tok/s (1.027x)**, but complete quality reaches max KL
   **1.804860** at **564/576** top-1. F32 PV, global-only, and SWA-only followups
   fail too. Remove the runtime owner/selector; retain only the corrected leaf
   and exact qrow4/M128 production. Proceed to H3.
3. **WPF-H3 source-faithful IQ3_XXS/IQ4_XS selected down MMQ is rejected for
   runtime use.** The standalone leaf moves all 47 actual M512 layers
   **565.437 -> 115.951 ms (4.877x)** and IQ3 is **27.145% below** llama.cpp's
   matched trace. Runtime natural-prompt prefill improves **152.276 -> 181.556
   tok/s (1.192x)**, but complete quality reaches max KL **0.373028** at
   **567/576** top-1. Keeping IQ4 exact still reaches **0.372917**, isolating
   source IQ3 arithmetic. Remove the runtime owner/selector and retain exact
   grouped production plus separately registered leaf evidence. Proceed to H4.
4. **WPF-H4 source-faithful Q6_K dequantize-plus-rocBLAS is rejected for
   runtime use.** The standalone fused producer/F16-compute-GEMM leaf moves the
   six-shape/144-call M512 family **174.351 -> 14.349 ms (12.151x)**, **3.825%
   below** llama.cpp's matched **14.919865-ms** stack. Natural-prompt prefill
   improves **151.784 -> 158.205 tok/s (1.042x)** with every category positive,
   but complete changed-arithmetic quality reaches max KL **0.338657** at
   **567/576** top-1. Remove the owner/selector/rocBLAS handle/97,517,568-byte
   workspace/capabilities; retain exact coltile and separately registered leaf
   evidence.
5. **WPF-H5 residual tail:** the clean exact-production M512 reprofile is
   complete at **169.516 tok/s**, **3,001.692-ms** kernel sum, and only
   **15.087 ms / 0.500%** span-minus-sum. Q5 remains first at **1,270.458 ms /
   42.325%**. H5A's transient exact-value F32 Q5 producer plus SGEMM leaf is
   admitted at **1,256.936 -> 221.137 ms (5.684x)** with N48 exact fallback,
   max mean KL **1.59e-9**, and top-1 **100%**. It adds no sidecar but remains
   **3.751x** the matched llama.cpp Q5 trace. Its default-off bounded owner
   passes natural M512 at KL **0.0003742**, but the complete lane rejects it at
   max KL **1.143627** with **564/576** top-1 despite **1.330x** diagnostic
   prefill. Remove all runtime ownership and retain the standalone leaf only.
   H5B's existing complete-`KVLiveSpans` packed F32 attention route passes the
   W7900 transfer screen at **109.897 -> 62.655 ms (1.754x)** leaf and cached
   attention **488.304 -> 60.669 ms (8.049x)**, but complete quality reaches
   max KL **0.444675** at **564/576** top-1 despite **1.148x** diagnostic prefill
   and all **10,512** expected launches. Remove runtime ownership/map/policy and
   retain exact attention. H5C's exact-value expansion plus production-ordered
   8x4/4x8 consumers established the byte-exact route. H5E adds 4x16/8x8/16x4
   and moves H5D's final-source 235-call policy **1,085.630 -> 951.876 ms
   (1.141x)** by events and **1,040.166 -> 961.993 ms (1.081x)** by wall. The
   bounded owner remains KL0/byte-exact through complete state; selector-unset
   production reaches **184.997/172.104/131.496 tok/s** through 4K. H5F retains
   only N48 12x4 at **1.187%/0.496%** event/wall. H5G retains four
   constant-80/96 geometries on five roles and publishes
   **188.393/175.042/132.743 (+2.192%/+2.055%/+1.329%)** over H5F. H5H rejects
   and removes constant-112/128 after no role wins and the VGPR256 spill cliff.
   The post-H5G request reclassifies at **2,667.034 ms**, led by Q5 **920.633**,
   IQ down **560.642**, attention **468.533**, and Q6 **177.047 ms**. H5I's
   exact-Q6 expansion plus ordered consumer clears all 146 calls at strong
   **194.758/189.722 -> 119.751/121.353 ms** event/wall, with four selected
   roles, two exact long-K fallbacks, and one exact wide-N fallback. Complete
   state and integrated tracing pass; clean 512/1K/4K promotes
   **191.713/178.080/134.411 tok/s**. H5J then promotes exact K1024 IQ3
   resident-segment reuse across all 45 calls plus a two-call IQ4 wave32
   sibling. Complete state is KL0; integrated selected down falls **10.706%**
   and clean 512/1K/4K reaches **196.103/181.859/137.169 tok/s**. H5K closes
   larger exact resident IQ3 rows: rowbatch12/16 lose all 45 layers by
   **5.8–10.8%** and are removed. Post-H5K attribution selects H5L's exact Q5
   weight-tile-major linear workgroup traversal after ordered consumers retain
   **904.399 ms** and the top two roles **741.721 ms**. H5L promotes six exact
   roles, cuts integrated Q5 **49.224%**, and publishes
   **237.956/217.888/157.366 tok/s** at 512/1K/4K. Reprofile the retained request
   before selecting a distinct next exact lane; do not stack H1-H5B or reopen
   larger row ownership, regressive output tiling, source MMQ, or P6.
6. Keep 16K+ closed. First reach direct-M512 parity at **694.184 tok/s**, then
   collect a matched llama.cpp HIP M4K comparator before reopening long-context
   work. Keep **800/700 tok/s** at M512/M4K as stretch targets rather than the
   only evidence of attainable hardware performance.

Quality-lane calibration is diagnostic only. It cannot waive the repository
KL/top-1 contract or promote D4, D8, D8R8, P6, or another failing approximate
variant. The fixed M512 stream is attribution-only and cannot promote a
candidate; every changed-arithmetic port uses the complete train+heldout
category suite and category heldouts. A source-faithful port that fails those
gates remains a valuable measured ceiling, but cannot become production.

The historical **150 tok/s** short dependency passes at both 512 and 1K, and
its required restored 4K gate is complete. WPF-1T reached it with exact
M512 state and clean **169.253/159.229 tok/s** production, but the matched HIP
result proves that 150 was only an intermediate gate, not a credible endpoint.

### Frozen target and current evidence

| Item | W7900 contract |
| --- | --- |
| Model | `/models/gguf/Laguna-S-2.1-UD-Q2_K_XL.gguf` |
| SHA-256 | `8fe1170f012723f6f7d6c9b08d8f928b0b3d8bffc32926f33a930148a1d62679` |
| Logical / tensor size | 117.562B parameters / 39,680,849,600 tensor bytes |
| Backend | `hip_gfx1100`, AMD Radeon Pro W7900, one explicitly selected device |
| Active iteration shapes | clean prefill-only 512/1K plus restored 4K production; BF16 KV; exact positions and deterministic next token; 16K+ closed below the 800/700 stretch gate |
| Control | clean `67ab7e5a8`, matrix128 / attention128, direct GGUF, cached-only builds |
| Control wall | **41.720 tok/s at 512**, **36.407 tok/s at 4K** |
| Profile repeat | **41.438 / 36.245 tok/s**, raw trace SHA-256 `87f02952...f6f5b3` |
| WPF-1 scalar A/B | **40.636/39.174 -> 79.009/73.654 tok/s** at 512/1K (**+94.431%/+88.018%**) |
| WPF-1W paired RB8 -> RB16 / RB32 | RB16 **+5.445%/+5.220%**; RB32 **79.023/73.610 -> 85.174/78.946 tok/s (+7.783%/+7.249%)** at 512/1K; every RB32 sample wins |
| WPF-C1 M128 -> M256 / M512 | M256 **85.028/78.672 -> 85.855/79.526 tok/s (+0.973%/+1.086%)**; direct M512 **+0.939%/+1.062%**; M256 wins direct aggregate wall by 0.027% and uses half M512 scratch |
| WPF-2 exact grouped IQ down | M256 direct -> grouped **86.175/79.924 -> 96.643/89.049 (+12.147%/+11.417%)**; M512 **86.129/79.887 -> 98.289/90.555 (+14.118%/+13.354%)**; M512 grouped beats M256 grouped **1.703%/1.691%** in the paired admission rows |
| WPF-2b exact pair16 grouped gate/up | All 46 actual IQ2 roles are exact and faster; inclusive leaf **1343.915 -> 482.040 ms (2.788x)**. Complete state is KL0; clean 512/1K improves **99.230/91.559 -> 118.705/107.804 (+19.626%/+17.743%)**. |
| WPF-3 exact qrow4 SWA default | One wave retains one head and four causal rows; the C256 policy uses exact wave32 below crossover. Four M128 slices improve **21.059 -> 9.389 ms (2.243x)** with zero F32-bit mismatches. The package-default M512 gate matches all 48 boundaries/KV spans at KL0. Clean 512/1K improves **118.705/107.804 -> 131.919/125.960 tok/s (+11.131%/+16.842%)**; traced SWA falls **55.411%/59.449%**. |
| Rejected WPF-3 online qrow4 SWA | The complete 18-prompt/576-step M512 lane improves natural-prompt prefill **117.170 -> 118.335 tok/s (+0.995%)** and h16/h32 E2E **+0.764%/+0.609%**, but fails at maximum KL **0.394600** despite **564/576 (97.917%)** top-1. Poolside, repeat determinism, and lifecycle pass; exact qrow4-C256 remains default. |
| WPF-1T exact Q5/Q6 coltile default | All 15 actual M512 configurations are exact/faster; the 381-invocation weighted `(4,8)` leaf sum is **2699.147 -> 1828.710 ms (1.476x)**. Four role keys select `(2,16)`, saving another **36.773 ms (2.011%)** from that family. The frozen same-resident gate improves **+0.545%/+0.459%** at 512/1K; a package repeat is exact/positive at **+0.382%/+0.242%** but misses its repeated 1K magnitude threshold. Complete state is KL0. Clean all-`(4,8)` publication remains **169.253/159.229 tok/s**; dense/shared falls **38.546%/38.875%**. |
| Current canonical clean selector-unset | **237.956/217.888/157.366 tok/s** at 512/1K/4K, matrix512/attention128/H5L exact-weight-major-Q5/H5I exact-ordered-Q6/pair16-grouped-gate/H5J resident-IQ-down/qrow4-C256-SWA, **+21.342%/+19.812%/+14.725%** over H5J. M512 full state is KL0/byte-exact and lifecycle recovers. |
| Matched external M512 targets | Same 512 IDs/context4096/direct M512/first token 2930: current hipEngine BF16 **237.956**, llama.cpp HIP BF16 **694.184 (2.917x)**, llama.cpp Vulkan native F16 **56.274 tok/s**. HIP server-native is **649.321 tok/s**. Direct HIP parity is the active external target; 800 remains stretch. |
| Rejected WPF-H1 Q5 source-MMQ runtime | All eight actual M512 roles and 235 calls: exact coltile **1,562.932 ms** -> aligned DS4/WMMA policy **97.110 ms (16.094x)**, with N48/N72 exact. Complete quality nevertheless reaches max KL **4.162014** at **561/576** top-1 despite **1.348x** diagnostic natural-prompt prefill. Runtime ownership is removed; retain only the explicit primitive. |
| Rejected WPF-H3 IQ3/IQ4 source-MMQ runtime | All **45 IQ3 + 2 IQ4** actual M512 selected-down layers move exact grouped **565.437 -> 115.951 ms (4.877x)** and runtime natural-prompt prefill improves **152.276 -> 181.556 tok/s (1.192x)**, but complete quality reaches max KL **0.373028** at **567/576** top-1. An IQ3-source/IQ4-exact followup still reaches **0.372917**. Runtime ownership is removed; retain exact grouped production plus the explicit leaf. |
| Rejected WPF-1B screens | D4 **129.572/116.116 tok/s**, max KL **0.624304**; D8 **129.083/115.802**, max KL **0.400292**; D8R8 **123.466/111.324**, max KL **0.964321** at 512/1K |
| Rejected P6 / P6-repair screen | Existing IQ2 MMQ gate/up is **3.336x** faster over 46 actual M512 layers and reaches diagnostic **122.135/110.761 tok/s (+23.082%/+20.972%)**, but complete quality reaches max KL **0.683239** at **565/576** top-1. P6 repair stops at **85.946%** uncertain coordinates and **99.496%** touched active output rows; WPF-1R's separately measured raw-Q5/Q6 screen is also rejected. |
| Rejected WPF-1R raw-Q5/Q6 repair | All **381/381** projection tensors are captured at M512; 333 are D8R8-eligible and 48 narrow gates remain exact. Measured BF16 mismatches touch **72.266-100%** of output-weight rows and imply **0.160-1.686x** exact-RB32 family reads; the conservative midpoint envelope reaches **9.142-93.418%** coordinates and **2.925-29.894x** reads. No repair queue/kernel/runtime route is admitted. |
| Current attribution | The retained H5L request segment is **2,074.261 ms / 1,862 dispatches** in a **2,100.389-ms** kernel span. H5L cuts Q5 **919.697 -> 466.986 ms (-49.224%)** while H5I Q6 and H5J IQ ownership remain. Residual span-minus-sum is **26.128 ms / 1.244%**, keeping launch/fusion deferred. Reprofile/reconcile the post-H5L families before selecting the next distinct exact lane. |
| Compact evidence | [`roofline/plan`](../benchmarks/results/2026-07-28-gfx1100-laguna-q2-xl-prefill-roofline-plan.json) · [`WPF-1 RB8 production`](../benchmarks/results/2026-07-28-gfx1100-laguna-q2-xl-q5-q6-rowbatch8-production.json) · [`WPF-1W RB32 production`](../benchmarks/results/2026-07-28-gfx1100-laguna-q2-xl-q5-q6-rowbatch32-production.json) · [`WPF-C1 M256 production`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-matrix256-retained.json) · [`WPF-2 grouped-IQ production`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-grouped-iq-matrix512-retained.json) · [`WPF-2 grouped-IQ correctness`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-grouped-iq-exact-correctness.json) · [`WPF-2b pair16 candidate`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-pair16-grouped-gate-up-candidate.json) · [`WPF-2b production`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-pair16-grouped-gate-up-production.json) · [`WPF-3 exact qrow4 candidate`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-swa-qrow4-exact-candidate.json) · [`WPF-3 default promotion`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-swa-qrow4-default-promotion.json) · [`WPF-3 production`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-swa-qrow4-exact-production.json) · [`WPF-3 online rejection`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-swa-qrow4-online-rejected.json) · [`WPF-1T candidate`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-q5-q6-coltile4-rowbatch8-candidate.json) · [`WPF-1T default`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-q5-q6-coltile4-rowbatch8-default-promotion.json) · [`WPF-1T production`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-q5-q6-coltile4-rowbatch8-production.json) · [`WPF-1T role policy`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-q5-q6-coltile-role-policy.json) · [`matched llama.cpp HIP/Vulkan attribution`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-llamacpp-prefill-matched-attribution.json) · [`WPF-H1 Q5 source-MMQ candidate`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-q5-k-source-mmq-candidate.json) · [`WPF-1B D4 primitive`](../benchmarks/results/2026-07-28-gfx1100-laguna-q2-xl-q5-q6-mmq32-primitive.json) · [`D4 rejection`](../benchmarks/results/2026-07-28-gfx1100-laguna-q2-xl-q5-q6-mmq32-d4-runtime-rejected.json) · [`D8 primitive`](../benchmarks/results/2026-07-28-gfx1100-laguna-q2-xl-q5-q6-mmq32-d8-primitive.json) · [`D8 rejection`](../benchmarks/results/2026-07-28-gfx1100-laguna-q2-xl-q5-q6-mmq32-d8-runtime-rejected.json) · [`D8R8 primitive`](../benchmarks/results/2026-07-28-gfx1100-laguna-q2-xl-q5-q6-mmq32-d8r8-primitive.json) · [`D8R8 rejection`](../benchmarks/results/2026-07-28-gfx1100-laguna-q2-xl-q5-q6-mmq32-d8r8-runtime-rejected.json) · [`P6/P6-repair rejection`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-p6-iq2-mmq-matrix512-rejected.json) · [`WPF-1R raw-Q5/Q6 repair rejection`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-q5-q6-d8r8-repair-density-rejected.json) · [`H5L production`](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-f32-weight-major-production.json) |

WPF-1 established the first retained W7900 prefill default. One shared
resident-weight process preserves logits bit-for-bit, all 48 hidden boundaries,
active K/V, every `KVLiveSpans` field, positions, tokens, and lifecycle across
scalar, rowbatch4, rowbatch8, rowbatch16, rowbatch32, and an RB32 repeat. WPF-1W
supersedes rowbatch8 with rowbatch32 after clean paired gains at both short
shapes. WPF-C1 then promoted matrix256 while keeping both attention capacities at 128.
WPF-2 preserves retained gate/up arithmetic, compacts exact post-SiLU rows once,
and reuses each grouped IQ3/IQ4 down segment across up to eight routed rows.
The exact capacity gate selects M512 and its **439,021,600-byte** planned row/MoE
scratch. WPF-2b then moves IQ2 gate/up to the exact local64/pair16 expert-major
rowbatch8 owner while preserving c=1 and unsupported-key route-major fallbacks.
WPF-3 then reuses each K/V row across four adjacent causal SWA queries while
preserving every row's production arithmetic. WPF-1T subsequently reuses each
activation across four Q5/Q6 output columns by default and across two columns
for four qualified role keys, while retaining RB32 arithmetic.
H5D then adds transient exact-F32 Q5 expansion and ordered reduction; H5E
extends exact ownership to all eight roles without retaining a weight sidecar.
Complete state is KL0; clean package throughput reaches
**184.997/172.104/131.496 tok/s** at 512/1K/4K. The short and restored 4K gates
pass, while 16K+ remains closed
below the 800/700 stretch target.

The bounded post-WPF-2 P6 repricing changes no production code. Across all 46
actual IQ2 gate/up layers, compact D4-Q8_1 pack plus the existing signed-byte
MMQ reduces the summed family leaf **1297.436 -> 388.901 ms (3.336x)**. Its
independent fixed-shape diagnostic reaches **122.135/110.761 tok/s** at 512/1K,
but the complete 18-prompt lane rejects it at maximum KL **0.683239** despite
**565/576 (98.090%)** top-1. Exact sparse repair is also closed before
implementation: maximum BF16 mismatch density is **85.946%** and touched active
expert-output-row density is **99.496%**, versus frozen **5%/20%** limits.
Exact matrix512/grouped-down therefore remains production. The subsequent
separate WPF-1R screen captures all **381/381** raw-Q5/Q6 projection tensors at
M512. Among 333 D8R8-eligible tensors, measured BF16 mismatches touch at least
**72.266%** of output-weight rows and modeled coordinate repair rereads up to
**1.686x** the complete exact-RB32 family. A conservative midpoint envelope is
larger still. This fails the frozen prospective gates before implementation.
WPF-2b runtime admission and post-publication cleanup are complete. The cleanup
removes rejected MMQ ownership and only the unowned Laguna grouped-gate
variants; Qwen3.5's separately retained grouped-dual production chain remains.
WPF-3's exact qrow4/C256 policy is now cleanly published as the gfx1100 package
default at local32, VGPR72, LDS0, and scratch0. Its no-override deep-state,
paired throughput, selector-unset, and cached-trace gates pass. The
changed-association online body is faster but rejected at maximum KL
**0.394600** on the complete lane; its gfx1151-owned registry surface remains,
but gfx1100 production stays exact. WPF-1T's exact dense output tiling is now
cleanly published and traced; the package keeps `(4,8)` generally and selects
`(2,16)` for only four measured role keys. The public benchmark-only variant
setter/constructor is removed, while RB32 remains the generic fallback. The
150 short gate and restored 4K run pass; 16K+ and launch/fusion work remain
closed.

The rejected broad shared-source candidate combined M2048 matrix/global
transactions with packed/block F32 attention, dense initial cache, and rolling
M128 SWA. Its diagnostic wall moved **41.554/39.453/36.244 ->
44.234/43.773/43.021 tok/s** at 512/1K/4K, but the mandatory complete quality
lane failed at max KL **1.11869** and every copied gfx1100 capability was
removed. That bundle also raised tracked residency **40.077 -> 43.581 GB**.
It does not contaminate retained WPF-C1: the isolated screen changes only
matrix capacity, forces ordinary/global/SWA attention rows to 128, and proves
complete state exactness versus M128. That WPF-C1 M256 packet remains the
explicit direct-route rollback; every copied broad capability remains removed.

### What transfers from gfx1151, and what does not

| Seam | Prior evidence | W7900 rule |
| --- | --- | --- |
| Matrix chunk capacity | The broad M2048+attention bundle is quality-rejected; isolated M256/M512 is measured | WPF-2 selects exact M512/attention128 only after grouped IQ down beats grouped M256 **1.703%/1.691%** at 512/1K. Keep explicit M128/M256 fallbacks. |
| Qwen chunk geometry | gfx1151's 256-row linear/MoE policy regressed W7900 **6.4-8.8%** | Never copy chunk/tile width by architecture name. Screen natural 128/256/512 rows. |
| Wave widening | gfx1151 keeps four-wave Q8 through 64K; W7900 keeps two-wave only through 4K | Wave count and crossover are backend capabilities, not shared constants. |
| Compiler resources | A selected-MoE body became VGPR256/private176B/75 spills on W7900 until the outer loop rolled | Every candidate records VGPR/SGPR/LDS/private/spills and rejects a spill-based “win”. |
| Launch seams | Router logits 512->256 threads, selector 512->128, and stream-ordered metadata transferred after independent gates | Transfer small launch hypotheses only with exact 512/1K A/B evidence, and only after the current **0.499%/0.504%** span-minus-sum gaps become material. |
| Queue overlap | gfx1151's first AOTriton on/off order gave a false negative; reverse order and 15 samples found the small win | Counterbalance process order. No favorable rerun or pooled waiver. |
| Quant arithmetic | gfx1151 Q4_K_M uses selected Q4/Q6 T16 D8/D4 MMQ; Q2 XL uses raw IQ2/IQ3/IQ4 experts and mostly raw Q5/Q6 dense/shared weights | Do not alias `LAGUNA_DENSE_Q4_PREFILL_MODE`, selected-MMQ, F16, concurrency, or queue policies. Add separate registered gfx1100 keys. |
| Hotspot migration | Tiled Conv mattered on gfx1151 but fell to 0.87-1.09% on W7900 after transfer | Reprofile immediately after every promotion and discard the source architecture's remaining checklist. |

WPF-C1 is deliberately narrower than the rejected broad bundle: it changes
only matrix capacity and keeps quant/MMQ, attention geometry, source-F16,
MoE-concurrency, and queue policies unchanged. M256 passes all 48 hidden
boundaries, complete K/V/live spans, shared-prefix routing, repeats, memory, and
lifecycle. Explicit M128 overrides preserve rollback; unsupported primitive keys retain
their registered fallbacks.

### W7900 profile and roofline

The corrected request segmenter recognizes the Q2 XL Q5_K embedding and IQ
selected families. The clean cached-only trace is:

| Shape | Kernel sum / span | Dispatches | Dense/shared Q5/Q6 | Selected IQ | Attention |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 | 12.303416 / 12.356689 s | 4,214 | 8.509884 s / 69.167% | 2.734115 s / 22.222% | 1.022529 s / 8.311% |
| 4K | 112.589175 / 113.007464 s | 33,726 | 68.352701 s / 60.710% | 21.885515 s / 19.438% | 22.054476 s / 19.588% |

Span-minus-sum is only **53.273 ms (0.433%)** at 512 and **418.289 ms
(0.372%)** at 4K. Graph/submission work is therefore deferred. The top Q5/Q6
body is `gguf_k_prefill_out_kernel<...,5|6>` with `Grid_Size_Y=128`: one
output-column owner is repeated for every prompt row. It does not tile the row
dimension, so it rereads the same encoded weight for every token.

The now-superseded post-WPF-1 rowbatch8 cached trace was:

| Shape | Kernel sum / span | Dispatches | Dense/shared Q5/Q6 | Selected IQ | Attention |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 | 6.305941 / 6.358115 s | 4,210 | 2.838547 s / 45.014% | 2.482874 s / 39.374% | 0.945927 s / 15.001% |
| 1K | 13.633213 / 13.736498 s | 8,426 | 5.659063 s / 41.509% | 4.985759 s / 36.571% | 2.911707 s / 21.357% |

That rowbatch8 trace selected WPF-1W but is no longer the performance headline.
The large Q5 role's per-element `q5_k_weight(...)` block locate, metadata loads,
6-bit scale unpack, nibble/high-bit extraction, scale/offset, and resulting FMA
made dequant/instruction issue the primary limiter. Exact wider slabs amortize
that chain without changing any row arithmetic. The retained RB32 actual-role
K3072/N12288 leaf is **5.977 ms** versus RB8 **9.021 ms**.

The now-superseded post-WPF-1W matrix128 cached trace is:

| Shape | Kernel sum / span | Dispatches | Dense/shared Q5/Q6 | Selected IQ | Attention |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 | **5.812069 / 5.859423 s** | 4,210 | **2.417919 s / 41.602%** | **2.447247 s / 42.106%** | **0.909824 s / 15.653%** |
| 1K | **12.662342 / 12.756807 s** | 8,426 | **4.844460 s / 38.259%** | **4.925955 s / 38.902%** | **2.817675 s / 22.252%** |

The retained post-WPF-C1 matrix256 trace is:

| Shape | Kernel sum / span | Dispatches | Dense/shared Q5/Q6 | Selected IQ | Attention |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 | **5.777317 / 5.804027 s** | 2,294 | **2.372139 s / 41.060%** | **2.480490 s / 42.935%** | **0.892694 s / 15.452%** |
| 1K | **12.586818 / 12.640568 s** | 4,595 | **4.761770 s / 37.831%** | **5.000716 s / 39.730%** | **2.760052 s / 21.924%** |

The preceding post-WPF-2 matrix512/grouped-exact trace was:

| Shape | Kernel sum / span | Dispatches | Dense/shared Q5/Q6 | Selected IQ gate/up + down | Attention |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 | **5.076126 / 5.091956 s** | 1,479 | **2.349246 s** | **1.242117 + 0.564660 = 1.806777 s** | **0.865893 s** |
| 1K | **11.146106 / 11.179360 s** | 2,962 | **4.728202 s** | **2.512102 + 1.132326 = 3.644428 s** | **2.664366 s** |

Versus the clean M256 trace, selected IQ falls **27.160%/27.122%**, kernel sum
**12.137%/11.446%**, and span **12.269%/11.560%**. IQ3 grouped rowbatch8 is
local128/VGPR48/SGPR128/LDS512/scratch0; grouped IQ4 is
local128/VGPR64/SGPR128/LDS512/scratch0. RB32 keeps zero scratch/private memory
and its prior code-object resources. The M512 trace measures the wider grids,
launch count, and durations rather than inheriting them from M128/M256.

The retained post-WPF-2b matrix512/pair16-grouped trace is:

| Shape | Kernel sum / span | Dispatches | Dense/shared Q5/Q6 | Selected IQ gate/up + down | Attention |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 | **4.245775 / 4.261498 s** | 1,479 | **2.326023 s** | **0.465182 + 0.528174 = 0.993356 s** | **0.869329 s** |
| 1K | **9.454265 / 9.487593 s** | 2,962 | **4.676829 s** | **0.933246 + 1.058688 = 1.991934 s** | **2.670429 s** |

Versus WPF-2, gate/up falls **62.549%/62.850%**, total selected IQ falls
**45.021%/45.343%**, and kernel span falls **16.309%/15.133%** with unchanged
dispatch count. The IQ2 body is local64/VGPR104/SGPR128/LDS512/scratch0. Dense
is now **54.78%/49.47%** of kernel sum, while SWA remains **91.48%/88.48%** of
attention at **0.795/2.363 s**. Span-minus-sum remains only **0.37%/0.35%**.

WPF-1B proves substantial but unretainable changed-arithmetic headroom. D4,
D8/S8, and two-stage D8R8/S8 improve clean 512/1K to
**129.572/116.116**, **129.083/115.802**, and **123.466/111.324 tok/s**.
D8R8 reduces maximum actual-role KL to **8.241e-7**, roughly 68x below D8, but
the complete 576-step lane becomes worse: max KL **0.964321** and **562/576
(97.569%)** top-1. D4 and D8 were already rejected at max KL
**0.624304/0.400292**. This non-monotonic autoregressive sensitivity closes
blind D16 or further residual-precision variants; it does not weaken the
quality contract. Production remains exact rowbatch32.

WPF-1R measures BF16 mismatch, conservative rounding-risk density, distinct
touched output-weight-row density, and modeled repair bytes on every actual
Q5/Q6 role. The frozen prospective stops were **5% uncertain output
coordinates**, **20% touched output weight rows**, and **25% of the exact
family's modeled source reads**. At M512 the screen captures all **381/381**
projection tensors: 333 are D8R8-eligible and 48 N48/N72 gates remain exact.
Measured BF16 mismatch density is **0.500-5.270%**, but those mismatches already
touch **72.266-100%** of output-weight rows in every eligible tensor. Exact
coordinate repair would reread **0.160-1.686x** the complete RB32 family;
**331/333** tensors exceed 25%, and the other two still fail touched-row
density. The per-tensor max-error BF16-midpoint envelope is
**9.142-93.418%** of coordinates, touches **99.512-100%** of rows, and implies
**2.925-29.894x** exact-family reads. All finite/RNE, inventory, token 2930,
position 511, and lifecycle checks pass.

All **333/333** eligible tensors therefore fail every conservative prospective
gate. Stop before a queue, repair kernel, overflow route, full-state/category
lane, or timing gate; the later **1.25x repaired-family / 1.10x complete
512/1K** thresholds are unreachable for this formulation. P6's separate IQ2
rejection was not used to make this decision. Exact matrix512/RB32 remains
production; cleanup is complete and WPF-3 now takes priority. Evidence:
[`WPF-1R raw-Q5/Q6 repair rejection`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-q5-q6-d8r8-repair-density-rejected.json).

The actual GGUF inventory gives this active linear ledger:

| Family | Logical parameters/token | GFLOP/token | Source bytes for one complete weight scan |
| --- | ---: | ---: | ---: |
| Attention projections | 2.803B | 5.606 | 1.976 GB |
| Dense layer MLP | 0.113B | 0.226 | 0.083 GB |
| Routers | 0.037B | 0.074 | 0.148 GB |
| Shared experts | 0.444B | 0.887 | 0.326 GB |
| Selected experts (top-10 active arithmetic) | 4.435B | 8.871 | 1.436 GB active-route equivalent/token |
| **Total** | **7.832B** | **15.665** | — |

All 256 selected-expert tensors across 47 sparse layers occupy **36.761 GB**.
Measured M128/M256/M512 top-10 chunks contain **1,280/2,560/5,120** route rows;
mean rows per all/active expert are **5.0/6.70**, **10.0/12.25**, and
**20.0/23.43**. Exact RB8 full-slab utilization rises
**56.34% -> 73.39% -> 85.54%** and RB16 **34.73% -> 52.89% -> 72.13%**. This
is measured grouping opportunity, not a geometry-only speed promise: selected
IQ regressed **1.36%/1.52%** in the direct M256 trace before WPF-2, while the
retained exact grouped owner now cuts that family **27.160%/27.122%** versus the
clean M256 profile. Planned row/MoE scratch is **109,761,568 / 219,514,912 /
439,021,600 bytes**. M512 is retained and M128/M256 remain explicit exact
rollback capacities. A chunk still activates most experts, so conservative
one-scan bounds use all 36.761 GB until physical traffic is measured. The dense/shared Q5/Q6 source set is
2.385 GB; adding the separately attributed router set gives 2.533 GB for all
non-selected linear weights. The pre-WPF scalar 512 path implies **1,221.316
GB** of dense/shared source-encoded reads and only **143.517 GB/s**, or
**19.69%** of the measured stream ceiling. Its selected active-route
equivalent is **735.220 GB** and
**268.906 GB/s**. These are encoded-equivalent ledgers, not DRAM counters.

A fresh 4-GiB vector-read kernel on GPU0 measures
**728.237/729.067/729.860 GB/s min/median/max** over 31 samples. The existing
W7900 4096-cube BF16 WMMA reference is **84.8 TFLOP/s**. At 15.665 GFLOP per
token, the dense-linear compute ceiling is **5,413 tok/s**. Combining those
measured ceilings with conservative source bytes gives:

| Shape/policy | BW floor | Compute floor | Simplified roof |
| --- | ---: | ---: | ---: |
| 512, M128, one all-expert+dense scan per four chunks | 0.2156 s | 0.0946 s | **2,375 tok/s** |
| 512, ideal future M512 expert-major one scan | 0.0539 s | 0.0946 s | **5,413 tok/s** |
| 4K, M128, 32 scans | 1.7247 s | 0.8166 s including 5.084-TFLOP attention ledger | **2,375 tok/s** |
| 4K, M2048, two scans | 0.1078 s | 0.8166 s | **5,016 tok/s** |

This is a target model, not a performance claim. In particular, the M512 row
assumes a future expert-major owner that actually reuses each weight scan; an
isolated matrix-capacity change alone does not guarantee it. The model excludes
activation and metadata traffic, quantization, nonlinear kernels, the final
one-row lm-head, and achieved efficiency of each IQ integer-dot body. Even so, **800/700 tok/s**
is only **14.78%/13.96%** of the simplified 512/4K roof. The original
41.720-tok/s control delivered **0.652 TFLOP/s** of active linear math; retained
matrix512/pair16-grouped at 118.705 tok/s now delivers **1.859 TFLOP/s**, still
only **2.193%** of the measured BF16 reference. The simplified roof still leaves substantial
software headroom.

### Cross-campaign calibration and the IQ format boundary

The gfx1151 Q4_K_M and W7900 Q2 XL artifacts have the same network dimensions
and therefore the same **15.6647424 GFLOP/token** active-linear ledger. This
makes one directional comparison useful: gfx1151 Q4 began its matrix512 /
attention128 campaign at **76.226 tok/s**, close to W7900's WPF-1 rowbatch8
checkpoint at **79.585 tok/s**; WPF-2 reached **99.230** and WPF-2b now
reaches **118.705**. Q4 then reached **654.249 tok/s (8.583x its own starting
row)** after the full LAP campaign. Current endpoints are **1.859 versus
10.249 TFLOP/s** of active linear math.
The similar early rows despite W7900's larger CU and
measured read ceilings are strong evidence of remaining software headroom, not
proof that codebook cost is a fixed percentage of the gap. Quantization,
devices, memory systems, and retained schedules differ. In particular,
**157.177 GB / 6.433 s = 24.4 GB/s** for current M128 is only the conservative
*one-scan-per-chunk source-byte model* divided by wall, not a DRAM counter or
the current kernels' physical traffic; repeated row-batch reads can exceed
that source ledger. The Q4 source is also not a deployable alternative on the
48-GB W7900: its frozen artifact contains **75.169 GB** of tensors, versus
**39.681 GB** for Q2 XL before runtime allocations.

The real format distinction is affine versus codebook:

- Q5_K/Q6_K codes admit integer-dot and activation-sum factoring because their
  decoded weights are affine in the packed magnitudes. Exact row batching now
  amortizes scalar dequant across 8/16/32 rows, but the retained body still
  performs FP32 per-element accumulation and has not acquired the Q4/Q6
  activation-sum/integer-MMQ campaign used on gfx1151.
- IQ2_XS/IQ3_XXS indices must first expand through their grid plus sign parity;
  a dot over the packed index itself is invalid. The current exact owner stores
  eight expanded values as FP32 and performs eight BF16-to-FP32 FMAs per
  segment. Post-expansion magnitudes do fit signed bytes, so integer dot/WMMA
  remains available *after* lookup and sign application.
- The proposed scale hoist is format-specific. IQ2's byte supplies two
  independent nibbles, each shared by **16**, not 32, weights; its exact
  pair16/shared-scale prefill was already measured and removed after regressions
  up to 5.25%, although pair16 remains useful for decode. IQ3's `aux >> 28`
  scale is shared across 32 weights. Any new hoist must preserve the existing
  `scale * sum` boundary and prove an ISA/load reduction rather than assuming
  source-level common subexpressions survive compilation.
- The IQ2 and IQ3 constant grids are each **1 KiB**. Staging the active grid in
  LDS once per workgroup is a legitimate exact screen, but not an assumed win:
  initialization/barrier cost, bank behavior, constant-cache behavior, and the
  final resource envelope must be measured on both routed shapes.

Most importantly, the post-expansion integer path already exists and was
repriced rather than reimplemented. The explicit P6 IQ2 MMQ32 primitive expands
raw IQ2 into signed-byte fragments, stages a 32-column x K256 tile in **10,240
B LDS**, and consumes caller-owned D4-Q8_1 activations with RDNA3 integer WMMA.
Its earlier quantizer-inclusive synthetic E256/K3072/N1024/top-10 screen
improved exact auto by **22.49-28.76% at 256 tokens** and **45.03-49.86% at
512**, with a conservative 2,560-compact-row crossover.

The actual M512 screen confirms the arithmetic ceiling but rejects production.
Across 46 IQ2 gate/up layers, summed leaf time falls **1297.436 -> 388.901 ms
(3.336x)** and every layer is faster. A temporary full-model session reaches
diagnostic **122.135/110.761 tok/s (+23.082%/+20.972%)** at 512/1K. The
complete 576-step train+heldout lane nevertheless reaches maximum KL
**0.683239 > 0.05** with **565/576** top-1. Sparse repair also fails immediately
at **85.946%** uncertain coordinates and **99.496%** touched active output
rows. No runtime mode, repair queue, or default is added; exact grouped WPF-2
remains production. The separate P2 `dp4a` decode was already rejected and
removed, and neither P2 nor P6 should be retried without a materially different
exact arithmetic/dataflow hypothesis.

### llama.cpp dataflow to transfer

The same-hardware speed ratios justify direct, source-faithful ports rather
than treating llama.cpp as inspiration only. The source below is byte-identical
to llama.cpp `c0bc8591e`; the committed measurement patch touches only
`tools/llama-bench`. Port the algorithm, producer/consumer contract, tile
ownership, and traced starting geometry into hipEngine's registry/ABI model;
do not port llama.cpp's host API or contiguous-cache ABI. The starting source
map at that commit is:

- H1/H3 producer and MMQ: `ggml/src/ggml-cuda/quantize.cu`, `mmq.cu`,
  `mmq.cuh`, and `mmq-config-rdna4.cuh`; routed compaction additionally audits
  `mmid.cu`.
- H2 attention: `ggml/src/ggml-cuda/fattn-mma-f16.cuh` and
  `fattn-common.cuh`.
- H4 Q6 selection/dequantization: `ggml/src/ggml-cuda/ggml-cuda.cu`,
  `mmq.cu`, and `convert.cu`, plus hipEngine's existing torch-free rocBLAS
  binding. Re-audit exact call ownership before implementation rather than
  copying llama.cpp host dispatch.

- `MMVQ_MAX_BATCH_SIZE=8`; M512 takes MMQ. RDNA3's high-expert rule selects MMQ
  for the 256-expert routed matrices. The host chooses the AMD-WMMA config,
  whose Q5_K/IQ2_XS/IQ3_XXS/IQ4_XS cases offer 256-thread, I128, J16..128,
  K256 tiles. The trace chooses J128 and reports VGPR232-248/scratch0 for the
  dominant MMQs.
- `quantize_mmq_q8_1` converts F32 source rows in 128-value records, generally
  one scale per 32 values for these formats, four packed int8 stores per lane,
  and format-specific partial sums for affine min terms. Broadcast gate/up
  uses a device inverse map so each physical token is quantized before its
  top-k compact slots are written. hipEngine already owns compact expert-major
  metadata; transfer the producer/consumer contract, not `mm_ids_helper`.
- Dense Q5_K at M512 uses custom Q8_1 MMQ: **58.951 ms** for the same 235
  projections versus hipEngine's **1,274.342 ms**, a **21.617x** body ratio.
  Port this complete path first. Dense Q6_K does **not** use that MMQ: M512
  exceeds gfx1100's Q6 `<=128` threshold, so Q6 uses a bounded dequant/cast/F16
  rocBLAS stack. Its correctly isolated **14.916 ms** compares with
  hipEngine's **156.111 ms**, a **10.466x** ratio. The old conservative
  **84.322-ms** combined charge deliberately included unrelated router/support
  work and is not the implementation ledger.
- IQ3_XXS selected down uses the same compact Q8_1 + 128x128/K256 MMQ framework
  and measures **152.380 ms** versus hipEngine **531.414 ms**, a **3.487x**
  ratio. The two IQ4_XS down calls are **3.146 vs 26.428 ms**. IQ2 gate/up core
  is already comparatively close at **404.903 vs 430.054 ms**; do not reopen
  P6 ahead of down.
- FlashAttention is `flash_attn_ext_f16<128,128,8,8>` plus one general stream-K
  fixup per layer. It stages Q/K/V and mask tiles dynamically, groups eight GQA
  heads, and uses F16 WMMA across 128 query rows. Its **21.725 ms / 96 calls**
  versus hipEngine **490.919 ms / 192 calls** is not explained by a twofold
  launch reduction. Transfer full-M512 query tiling, head grouping, and
  stream-K ownership behind `KVLiveSpans`; never copy its contiguous-cache ABI.
- The audited Vulkan RADV path remains a compatibility floor. Its direct matched
  native-F16 row is **56.274 tok/s**, versus hipEngine **169.228** and HIP
  **694.184**. Vulkan shader geometry is not the next W7900 campaign.

A source-faithful Q8_1 or F16-WMMA port changes arithmetic. Same first token
**2930** does not make it production-correct. That is a promotion constraint,
not a reason to postpone the port: H1-H4 implement and measure the audited
llama.cpp route first, retain the existing exact registered path as fallback,
and require the complete 18-prompt/576-step train+heldout lane plus category
heldouts before any clean publication.

### WPF execution order

`WPF-*` labels mean “W7900 prefill” and are independent of the historical
`LAP-*` gfx1151 labels.

| Task | State | Gate / next action |
| --- | --- | --- |
| WPF-0 profile + roofline | Complete | Clean 512/4K wall and all-family trace, actual tensor/FLOP ledger, 729.067-GB/s read ceiling, llama.cpp HIP/Vulkan audit, and compact artifact published. |
| WPF-C0 shared-source capacity transfer | Rejected/removed | The bundle was performance-positive but failed the mandatory 576-step quality gate at max KL 1.11869; no copied capability default is retained. |
| WPF-1 retained exact dense/shared row reuse | **Complete; superseded by WPF-1W** | Full state is bit-exact. Scalar **40.636/39.174 -> 79.009/73.654 tok/s** at 512/1K; RB8 selector-unset reached **79.585/74.512** before WPF-1W. rowbatch4/scalar remain explicit rollback/crossover routes; gfx1151 is fail-closed. |
| **WPF-1W exact rowbatch widening** | **Complete; rowbatch32 retained gfx1100 default** | RB16/RB32 are exact through partial 33-row tails, ten actual roles, all 48 hidden boundaries, logits/KV/live spans, and repeat/lifecycle. The **unweighted diagnostic** ten-role sum moves RB8 **45.1883 ms** to RB16/RB32 **41.2040/39.2782 ms**; it is not an end-to-end forecast. Clean paired RB32 improves **+7.783%/+7.249%**, every sample wins, and selector-unset publishes **85.481/79.555 tok/s (+7.408%/+6.768% over RB8)**. RB32 remains scratch/private0 and theoretical 32 waves/CU despite 14/5 Q5/Q6 SGPR spills. |
| **WPF-C1 isolated matrix capacity** | **Complete; superseded by WPF-2** | M256 improves M128 **+0.973%/+1.086%** at 512/1K; direct M512 reaches **+0.939%/+1.062%** but loses aggregate wall by 0.027% and doubles planned scratch. All state gates are exact. M256 remains an explicit rollback. |
| **WPF-2 exact-first routed IQ reuse** | **Complete; explicit preceding rollback** | Exact IQ2/IQ3 grouped rowbatch8 primitives passed tails/resources, but the local256/group8 grouped gate/up body did not preserve pair16/local64 arithmetic. Its unowned Laguna rowbatch8/fused-SiLU surfaces are removed; shared Qwen3.5 base/rowbatch4/adaptive/auto owners remain. The retained rollback keeps route-major gate/up, compacts post-SiLU once, then runs IQ3 rowbatch8/IQ4 auto down plus exact restore. Its clean publication reached **99.230/91.559 tok/s** with exact complete state. |
| **WPF-2b pair16-compatible grouped gate/up** | **Complete; retained gfx1100 default** | Local64 pair16 rowbatch8 preserves production arithmetic while reusing each IQ2 gate/up decode across expert-major rows. All 46 actual M512 roles are exact and faster; complete state is KL0. Clean publication improves **99.230/91.559 -> 118.705/107.804 tok/s (+19.626%/+17.743%)**. Tracing reports local64/VGPR104/SGPR128/LDS512/scratch0 and cuts gate/up **62.549%/62.850%**, selected IQ **45.021%/45.343%**, and span **16.309%/15.133%** at unchanged dispatch count. c=1, unsupported-key, `grouped_exact`, and paired `direct` exact fallbacks remain. |
| **WPF-2P P6 IQ2 signed-byte MMQ** | **Rejected; no runtime owner added** | All 46 actual M512 IQ2 gate/up leaves are faster at **3.336x** summed and the temporary full-model diagnostic reaches **122.135/110.761 tok/s**, but complete quality is max KL **0.683239** at **565/576** top-1. Sparse repair stops at **85.946%** uncertain coordinates and **99.496%** touched active output rows. Keep only explicit primitive evidence; do not retry P2/P6 arithmetic. |
| WPF-1B approximate dense/shared Q8_1 MMQ | **D4/D8/D8R8 rejected** | Fastest clean candidates reach **129.572/116.116**, **129.083/115.802**, and **123.466/111.324 tok/s**, but all fail max-KL quality. D8R8 is the final blind-precision screen: max KL **0.964321**, **562/576** top-1. Keep exact production. |
| **WPF-1R guarded exact repair** | **Rejected before implementation** | All 381 raw-Q5/Q6 tensors are captured at M512; all 333 eligible tensors fail conservative density/touched-row/read stops. Measured mismatches alone touch **72.266-100%** of rows and imply up to **1.686x** exact-family reads. No queue, repair kernel, overflow route, runtime mode, full-state/category lane, or timing gate is added. |
| **WPF-3 short SWA attention** | **Complete; exact qrow4 retained and online qrow4 rejected** | One wave keeps one head and production's per-row two-pass arithmetic while reusing K/V across four causal rows; this is distinct from rejected cross-head/tiled sharing. The C256 policy keeps exact wave32 below crossover. Four M128 slices improve **21.059 -> 9.389 ms (2.243x)** bit-exactly; qrow4 traces at local32/VGPR72/LDS0/scratch0. The no-override M512 gate is KL0 across all 48 boundaries/KV spans. Clean 512/1K improves **+11.131%/+16.842%**, while cached SWA and complete span fall **55.411%/59.449%** and **9.643%/14.228%**. Changed-association online qrow4 improves complete-suite prefill **0.995%** but is rejected at max KL **0.394600** despite **564/576** top-1. |
| **WPF-1T exact dense output tiling** | **Complete; retained gfx1100 production through 4K** | `(2,16)` and `(4,8)` preserve RB32's 32 accumulators/thread, K ownership, FMA order, wave tree, and serial wave sum. Both are byte-exact/faster on all 15 unique actual Q5/Q6 M512 configurations. Production-weighted RB32/`(2,16)`/`(4,8)` sums are **2699.147/2220.526/1828.710 ms** over 381 invocations; `(4,8)` is **1.476x (-32.249%)** and compiles at local128/VGPR72/SGPR50/LDS512/private0 with zero spills. Exactly four role keys select `(2,16)`, reducing the all-`(4,8)` family another **36.773 ms (2.011%)** to **1791.936 ms**. The frozen 512/1K gate passes at **+0.545%/+0.459%**; a package repeat remains exact/positive at **+0.382%/+0.242%** but misses its repeated 1K magnitude threshold. Explicit RB32, smaller slabs, unsupported widths, and gfx1151 remain exact fallbacks; the public variant setter/constructor is removed. No-override M512 is KL0 across all 48 boundaries/KV spans. The pre-H5D canonical 512/1K was **169.253/159.229 tok/s (+28.301%/+26.412%)** and tracing cut dense/shared **38.546%/38.875%**. Restored 4K reached **123.084 tok/s** with deterministic ID/position/lifecycle and allocation recovery. |
| **WPF-H1 source-faithful Q5_K MMQ** | **Rejected; runtime owner removed** | The strict DS4 producer plus isolated fast-math I128/J128/K256 consumer moves the eight-role/235-call leaf **1,562.932 -> 97.110 ms (16.094x)** and natural-prompt prefill **151.252 -> 203.862 tok/s (1.348x)**, but complete quality reaches max KL **4.162014** at **561/576** top-1. Poolside/repeats/lifecycle pass. Remove the constructor switch, activation scopes, DS4 workspace owner, package capability, and dispatch policy; retain only the registered primitive/leaf evidence. Production stays exact. |
| **WPF-H2 source-faithful full-M512 FlashAttention** | **Rejected; runtime owner removed** | The retained standalone BF16-cache/F32-boundary whole-tile body copies F16 `128x128`, eight-query/eight-GQA-head WMMA ownership behind complete `KVLiveSpans`. Weighted 12-global/36-SWA M512 moves **490.919 -> 21.719 ms (22.603x)** and nominally matches llama.cpp's **21.725-ms** trace; natural-prompt prefill improves **152.087 -> 156.219 tok/s (1.027x)**. Complete quality nevertheless reaches max KL **1.804860** at **564/576** top-1. F32 PV/global-only/SWA-only followups fail, and the stream route was already non-finite/slower. Remove runtime ownership/capabilities/tests; retain the corrected standalone primitive and exact qrow4/M128 production. |
| **WPF-H3 source-faithful IQ3/IQ4 selected-down MMQ** | **Rejected; runtime owner removed** | The standalone local `(32,8)` I128/J128/K256 leaf moves all **45 IQ3 + 2 IQ4** actual M512 layers **565.437 -> 115.951 ms (4.877x)**; IQ3 is **27.145% below** llama.cpp. Complete runtime quality nevertheless reaches max KL **0.373028** at **567/576** top-1 despite **1.192x** diagnostic prefill. IQ3-source/IQ4-exact still reaches **0.372917**. Remove runtime ownership/capabilities/tests; retain exact grouped production and the separately registered VGPR152/248 leaf. [`rejection`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-iq3-iq4-source-mmq-rejected.json) · [`leaf`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-iq3-iq4-source-mmq-candidate.json). |
| **WPF-H4 source-faithful Q6_K F16/rocBLAS** | **Rejected; runtime owner removed** | The standalone local64/F16-compute leaf moves the actual six-shape/144-call M512 inventory **174.351 -> 14.349 ms (12.151x)**, **3.825% below** matched llama.cpp. Runtime natural-prompt prefill improves **151.784 -> 158.205 tok/s (1.042x)** with every category positive, but complete changed-arithmetic quality reaches max KL **0.338657** at **567/576** top-1. Remove runtime ownership, rocBLAS handle, 97,517,568-byte workspace, capabilities, and tests; retain exact coltile plus the registered leaf. [`rejection`](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q6-k-f16-rocblas-rejected.json) · [`leaf`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-q6-k-f16-rocblas-candidate.json). |
| **WPF-H5A exact-value Q5 F32/SGEMM** | **Rejected; runtime owner removed** | Clean exact M512 is **169.516 tok/s** versus matched llama.cpp HIP **694.184 (4.095x)**, with Q5 at **1,270.458 ms / 42.325%**. The bounded no-sidecar leaf selects exact N48 fallback and moves the actual 235-call role policy **1,256.936 -> 221.137 ms (5.684x)** by events, corroborated by **5.273x** wall. Raw operand values are exact; max mean/max-row KL is **1.59e-9/5.79e-8** and top-1 **100%**. The default-off owner passes natural M512 at KL **0.0003742**, but complete quality reaches max KL **1.143627** at **564/576** top-1 despite **1.330x** diagnostic prefill. Remove runtime ownership/workspace/capabilities/tests; retain exact coltile plus the registered leaf. [`rejection`](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-f32-sgemm-rejected.json) · [`leaf`](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-f32-sgemm-candidate.json). |
| **WPF-H5B packed F32 dense-initial attention transfer** | **Rejected; runtime owner/map removed** | Reuses the existing complete-`KVLiveSpans` route; no kernel is ported. The tuned packed two-call QK/PV plus wave32 leaf moves **109.897 -> 62.655 ms (1.754x)**, natural M512 passes KL **0.000429**, and tracing moves attention **488.304 -> 60.669 ms (8.049x)**. The binding split-local M512 extension observes all **10,512** expected candidate launches but reaches max KL **0.444675** at **564/576** top-1 despite **165.555 -> 190.103 tok/s (1.148x)** diagnostic prefill. Remove gfx1100 runtime policy/map/propagation/tests; retain exact qrow4/M128 plus the standalone leaf. [`rejection`](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-f32-hipblaslt-attention-rejected.json) · [`leaf`](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-f32-hipblaslt-attention-candidate.json). |
| **WPF-H5C..H5G exact-ordered F32-weight Q5** | **Complete; retained gfx1100 production through 4K** | Reuses H5A's bit-exact transient producer but replaces SGEMM with local128 ordered consumers. Rows17/33 and all roles are byte-exact with one **150,994,944-byte** bounded plane/no sidecar. H5F retains 12x4/VGPR104 only for N48. H5G adds 8x10/16x5/8x12/12x8 on five roles; its strong gate cuts H5F **8.639%/7.479%** by event/wall, with VGPR168/200/LDS1536/scratch0. Package-default M512 remains KL0 across all 48 boundaries/logits/KV/repeat/teardown and clean publication is **188.393/175.042/132.743 tok/s (+2.192%/+2.055%/+1.329%)** over H5F. H5H rejects/removes every constant-112/128 body after universal regressions and the VGPR256 spill cliff. [`production`](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-f32-ordered-production.json) · [`leaf`](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-f32-ordered-candidate.json). |
| **WPF-H5I exact-ordered F32-weight Q6** | **Complete; retained gfx1100 production through 4K** | One exact raw-Q6-to-F32 producer reuses H5G's existing plane/library and ordered consumer. Four roles select `16x5`/`16x4`/`8x4`; three large roles retain raw coltile. Complete state is KL0/byte-exact across all boundaries/logits/KV/live spans/repeat/teardown with no new allocation. Cached tracing records **143+143** candidate launches and three fallbacks, cutting Q6 **177.047 -> 110.170 ms (-37.774%)** and request sum **2,667.034 -> 2,600.260 ms (-2.504%)**. Clean selector-unset 512/1K/4K is **191.713/178.080/134.411 tok/s (+1.762%/+1.736%/+1.256%)** over H5G. [`production`](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q6-k-f32-ordered-production.json) · [`leaf`](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q6-k-f32-ordered-candidate.json). |
| **WPF-H5J exact IQ row ownership** | **Complete; retained gfx1100 production through 4K** | IQ3 retains one decoded segment across unchanged rowbatch8 phases; IQ4 launches the retained exact body at local32 after a generated one-ULP RED rejects duplicate constant-K math. All **45+2** actual M512 layers are byte-exact and both-clock positive, moving event/wall sums **567.274/567.056 -> 500.176/500.448 ms (-11.828%/-11.746%)**. Complete state is KL0; cached integrated selected down falls **10.706%** and request sum **2.624%**. Clean selector-unset 512/1K/4K is **196.103/181.859/137.169 tok/s (+2.290%/+2.122%/+2.052%)** over H5I. No allocation/workspace/sidecar is added; bounded misses and gfx1151 retain exact fallback. [`production`](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-iq-row-ownership-production.json) · [`leaf`](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-iq-row-ownership-candidate.json). |
| **WPF-H5K larger exact resident IQ3 row batches** | **Rejected; all temporary surfaces removed** | Rowbatch12/16 are byte-exact, scratch-free at VGPR48/56, and lifecycle-clean, but lose both clocks on every **45/45** actual IQ3 layer. H5J event/wall **489.058/492.542 ms** regresses to **522.768/520.964 (+6.893%/+5.771%)** and **541.730/541.156 (+10.770%/+9.870%)**. Do not extend resident IQ3 ownership beyond eight. [`rejection`](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-iq3-larger-resident-rowbatch-rejected.json). |
| **WPF-H5L exact Q5 weight-tile-major traversal** | **Complete; retained gfx1100 production through 4K** | Six role-qualified mappings preserve H5G geometry/FMA/wave/store order; F32 N48/N72 retain H5G. The final-source leaf improves event/wall **1.813x/1.871x**. Complete state is KL0; integrated tracing records **235 producers + 188 candidates + 47 fallbacks**, cutting Q5 **49.224%** and request sum **18.079%**. Clean 512/1K/4K reaches **237.956/217.888/157.366 tok/s (+21.342%/+19.812%/+14.725% over H5J)** with the unchanged F32 plane. [`production`](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-f32-weight-major-production.json) · [`leaf`](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-f32-weight-major-candidate.json). |
| WPF-Q lane sensitivity calibration | Diagnostic only | Explain non-monotonic autoregressive amplification; never change thresholds or use calibration to promote a failing approximate path. |
| WPF-4 launch/fusion | Deferred | Fresh H5L M512 span-minus-sum is only **26.128 ms / 1.244%**, and llama.cpp remains faster despite more launches. Start only after span-minus-sum or launch-only boundaries exceed 5% of retained wall. |
| WPF-5 long context | 4K complete; 16K+ hard deferred | Clean H5L 4K is **157.366 tok/s (+14.725%)** over H5J. First reach matched direct-M512 HIP parity **694.184 tok/s**, then collect a matched llama.cpp HIP M4K row before reopening 16K+. Keep 800/700 at M512/M4K as stretch, not the sole hardware-ceiling evidence. |

### Admission and stop rules

For every WPF kernel or owner:

1. RED first against NumPy/CPU or the registered exact GPU fallback. Exact
   candidates preserve BF16/F32 bytes on synthetic edges and actual first/last
   production roles. Reassociated/Q8_1 candidates must additionally pass
   KL <= 0.05, top-1 >= 90%, and the complete 18-prompt train+heldout lane.
2. Record kernel name, grid/local size, VGPR/SGPR/LDS/private/spills, duration,
   and a cached-only `rocprofv3` trace. Any private memory or spill regression
   requires an explicit measured justification.
3. Use one resident weight set. No persistent full-family Q5/Q6/IQ alternate
   weight sidecar fits the 48-GiB budget. Bounded producer/risk/repair queues
   such as the current 4,325,376-byte M128 D8R8 workspace are allowed when
   capacity is explicit, liveness-aliased, fail-closed, and teardown-exact.
4. H1-H4 start with actual-shape M512 leaves, then gate 512/1K in both process
   orders with exact IDs/positions/state, deterministic repeats, finite logits,
   and allocation recovery. Do not favorable-rerun or waive a per-order
   regression by pooling. Small wins use at least 15 samples or the established
   full-model repetition protocol. Do not spend iteration time beyond the
   already published 4K row until direct M512 reaches the **694.184 tok/s**
   external parity target; then measure matched llama.cpp HIP at M4K before
   defining the next long-context gate.
5. Promote only architecture-qualified package capabilities and registered
   four-axis keys. Keep the unfused/M128/exact fallback. Never add backend or
   quant branches to generic runner/model dispatch.
6. Reprofile after promotion. Stop working a family once another family is
   larger, the candidate misses its prospective gate, or its perfect-removal
   ceiling falls below 5%.
7. Gate calibration may characterize autoregressive sensitivity but never
   changes the KL/top-1 thresholds. Approximate MMQ must pass them or repair to
   the exact BF16 output; no pooled/top-1-only waiver is allowed.

Do not copy gfx1151's Q4/D8/D4, F16, queue, concurrency, qmicro, prefetch, or
wave-width defaults into gfx1100. Do not optimize the fixed prompt or route
IDs. Do not rerun 4K or resume long-context sweeps merely because M2048 can
complete them; matched direct-M512 HIP parity and the subsequent matched M4K
measurement control when 16K+ reopens.

## Retained gfx1151 / Q4_K_M campaign record

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
Precomputing Q6's exact K16 activation sums once in the unchanged D4 metadata
word then publishes **647.207/576.799/468.431 tok/s** and cuts the traced
23-call Q6 window **100.367 -> 99.459 ms (-0.905%)**.
Precomputing Q4's exact K16 activation sums once in an activation-only sidecar
then publishes **649.791/576.589/468.830 tok/s** and cuts selected gate/up
**334.229 -> 330.720 ms (-1.050%)**. Qualified library PV tiles now stay
head-major through a division-free exact softplus gate: pp512 output-unpack
launches fall **144 -> 0**, total dispatches fall **2,417 -> 2,273**, and the
transpose-plus-gate boundary falls **11.240 -> 10.318 ms (-8.20%)**. Clean
selector-unset continuity is **647.826/575.732/468.103 tok/s**, within
**-0.302%/-0.149%/-0.155%** aggregate variance of the preceding packet.
The fused RMSNorm/RoPE producer now writes the three qualified query tiles
head-major directly. Query-pack launches fall **144 / 4.907 ms -> 0**, total
dispatches fall **2,273 -> 2,129**, and producer-plus-pack falls
**20.530 -> 16.666 ms (-18.82%)**. Clean selector-unset production reaches
**654.249/579.699/468.608 tok/s**, improving
**0.991%/0.689%/0.108%** over the preceding packet.
An exact split gate/up formulation that
writes D4 directly from the up epilogue is slower at both natural primary
shapes and has been fully removed.
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
The quality-admitted production default now measures **654.249 tok/s**
selector-unset, **8.583x** the old row and **89.875%** above the Vulkan
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
| Production publication | Complete/current | The direct all-exact gate remains max KL **0.049542582**, **316/320** top-1, with deterministic repeats, Poolside exact top-1, and exact lifecycle through 4K. Direct packed-query production, packed-query/wave-softmax attention, exact packed-output gating, Q4/Q6 precomputed activation sums, shape-qualified Q4 raw-nibble P8, and the row-qualified source-F16 schedule publish **654.249/579.699/468.608 tok/s** at 512/1K/4K. pp512 wall is **782.577 ms**, leaving **51.148 ms** to 700. |
| Direct Q4 gate/up wave decode | Admitted gfx1151 default | Direct per-column T16 decode removes pair decode/shuffle without changing resident bytes or arithmetic. The actual layer-1 leaf improves **8.107 -> 6.916 ms (-14.69%)**; clean pp512 improves **449.020 -> 474.363 tok/s (+5.644%)**, and cached tracing cuts the family **389.893 -> 317.722 ms (-18.51%)**. |
| Direct Q4-down wave decode | Admitted gfx1151 default | Direct per-column T16 decode removes pair decode/shuffle only for Q4 down while retaining Q6 row-vector production. Clean pp512 improves **473.963 -> 480.629 tok/s (+1.406%)**, and cached tracing cuts the Q4-down consumer **90.280 -> 71.378 ms (-20.94%)**. |
| Q6 qmicro resident payload | Admitted gfx1151 production default | Byte-neutral `[K32][col4][K4][QL8,QH4]` records preserve the 3,360-byte tile and every BF16 result. On the actual layer-1 660.6 MB tensor, natural-M512 selected prefill improves **5.1564 -> 5.0714 ms (-1.65%)** and top-10 exact decode improves **0.0910 -> 0.0846 ms (-6.99%)**. Clean pp512 improves **526.451 -> 530.447 tok/s (+0.759%)** and traced Q6 falls **126.594 -> 123.473 ms (-2.465%)**. Existing cache files convert once before upload; root lm-head and unmeasured backends remain legacy T16. |
| Q6 qmicro permute decode | Admitted gfx1151 production default | Two `v_perm_b32` byte gathers replace scalar quartet unpack without changing the byte-neutral record or arithmetic. The actual leaf improves **4.872 -> 4.741 ms (-2.67%)**, seven complete-state pairs improve **0.276% (5/7 wins)**, and tracing cuts the 115-call Q6 body **1,138.893 -> 1,124.852 ms (-1.23%)** with local128/VGPR80/LDS5120B/scratch0. Clean publication reaches **571.415/529.870/445.164 tok/s**. |
| Q6 compact activation cache | Admitted gfx1151 production default | Q6 never consumes Q8_1 sum metadata. Dropping that field and storing each bounded K16 quant sum as `int16` reduces activation staging **48 -> 40 bytes/row** and kernel LDS **5,632 -> 5,120 B** without changing dots or accumulation. The actual leaf improves **5.082 -> 4.911 ms (-3.36%)**; 15 complete-state pp512 pairs improve **550.584 -> 552.807 tok/s (+0.404%, 15/15 wins)**. Clean 512/1K/4K reaches **550.625/517.017/431.789 tok/s**. |
| Q6 half-row activation staging | Admitted gfx1151 production default | Each of 128 threads stages one 16-byte activation half and one K16 sum instead of leaving half the workgroup idle while 64 threads stage complete rows. Resources stay local128/VGPR88/SGPR128/LDS5120B/scratch0. The actual layer-1 leaf improves **4.902 -> 4.885 ms (-0.351%, 16/21 wins)**; the all-Q6 screen improves **21/23** layers and **111.798 -> 111.490 ms (-0.276%)** with zero BF16 mismatches. Complete pp512 A/B is exact and positive at **552.562 -> 553.018 tok/s (+0.083%)**; clean headline publication is neutral. |
| Q6 padded-activation elision | Admitted gfx1151 production default | Natural M512 has **117,760 useful** versus **362,944 padded** Q6 row slots. Padded slots are never consumed by the guarded dot/store loops, so production skips their zero LDS stores and K16 sum work. It improves **19/23** actual layers and **112.008 -> 111.806 ms (-0.180%)**, with zero BF16 mismatches and unchanged local128/VGPR88/SGPR128/LDS5120B/scratch0. Complete pp512 A/B is exact and positive at **552.983 -> 553.559 tok/s (+0.104%, 7/11 wins)**; clean publication reaches **551.459/517.307/432.099 tok/s**. |
| Q6 selected-down integer WMMA | Admitted gfx1151 production default | Four wave32 groups consume the existing planar-qmicro/D4 caches as 16x16x16 signed-int8 x unsigned-Q6 fragments while preserving the two K16 scales, `-32*sum(x)` correction, ordered FP32 K32 accumulation, and BF16 store. The latest body carries both the next raw qmicro record/metadata and next compact Q8 half-row in registers, and the producer records the two exact K16 sums once in D4's unchanged metadata word. Inclusive leaf improves another **0.818%**; clean 512/1K/4K reaches **647.207/576.799/468.431 tok/s**, and tracing cuts 23 Q6 calls **100.367 -> 99.459 ms (-0.905%)** at local128/VGPR112/SGPR128/LDS5120B/scratch0. |
| Q4 selected-down raw-nibble P8 | Admitted gfx1151 production default | Payload-only next-K32 prefetch transfers from gate/up to the 64x32/local64 single-output body without changing resident bytes, LDS, scratch, arithmetic, or BF16 output. Three traced M512 arms cut 72 Q4-down calls **217.416 -> 212.090 ms (-2.450%)** at VGPR **88 -> 96**; seven complete-state pp512 pairs improve **639.574 -> 643.166 tok/s (+0.562%, 7/7 wins)**. Clean selector-unset publication is **643.141/573.717/466.913 tok/s**. |
| LAP-7–LAP-8 | Exact cached-only, cached-metadata, qrow6, and dense-initial policies admitted | Complete M128 tiles append before cached-only attention while partial, wrapped SWA, verifier, explicitly evicted, and unmeasured paths retain exact fallbacks. The final initial-fill policy uses global qrow4/qrow6 and SWA qrow4 without per-token position/eviction reads. Matched pp512 improves **552.144 -> 559.539 tok/s (+1.339%)**; clean publication reaches **559.290 tok/s**, and tracing cuts attention **153.226 -> 141.846 ms (-7.43%)**. Scalar split-state, M16xK64 WMMA, M8xK64 WMMA, qrow8, head2, qhead3, and nine-wave GQA sharing remain closed. |

## Post-500 campaign — 700 production stretch

The 350 and 500 tok/s milestones prove the compounded production package, but
they are not roofline results. Current clean production measures **0.782577
seconds** synchronized pp512 wall. The clean Q6-precomputed-sum trace at
`15b26fc09` measures **0.843063 seconds** kernel span and **1.106503 seconds**
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
| Selected D8 Q4 gate/up | **332.844 ms** | **30.45%** | Shape-qualified raw-nibble P8 is the gfx1151 default. Physical counters before P8 reached **195.88 GB/s / 88.64%** of the read anchor. Compact metadata prefetch and non-temporal payload loads are both measured regressions; the next screen must remove physical bytes, cross-tile work, or a caller boundary. |
| Activation/reduce/residual | **252.435 ms** | **23.09%** | This inclusive bucket moves with cross-stream overlap and is not an additive ceiling. The prior queue union showed only **0.826 ms** secondary-only; reopen only with caller-stream relief, reduced bandwidth contention, or a fused producer that wins the complete wall. |
| Selected D4 Q4/Q6 down | **170.751 ms** | **15.62%** | Direct Q4 decode and byte-neutral planar-Q6 integer WMMA are retained. Precomputed exact Q6 K16 sums cut the Q6 subwindow **100.367 -> 99.459 ms** with unchanged resources. Further work requires fewer physical weight bytes or a new cross-tile schedule. |
| Static-range direct hipBLASLt source-F16 | **124.308 ms** | **11.37%** | All five contractions and fused producer boundaries are included. Exact fusion removes **96** standalone casts. Concatenated QKV still has only a **2.891-ms** modeled ceiling before restride, and layout-preserving `GroupedGemm` exposes zero gfx1151 algorithms. |
| Q4/Q6 WMMA dense/shared | **92.035 ms** | **8.42%** | This inclusive family overlaps routed work. The secondary shared branch remains hidden. An exact shared gate/up+SiLU leaf improved **14.56%** yet regressed production **0.52%**; reopen only after queue-exclusive caller-stream evidence changes that premise. |
| Global + SWA attention | **60.564 ms** | **5.54%** | Qualified positions 128/256/384 use exact BF16 cache widening, direct packed F32 query production, one wide QK and one wide PV hipBLASLt contraction, one wave32 per causal-score row, and a packed-output-aware exact gate. Partial, wrapped, explicitly evicted, verifier, decode, and unmeasured routes retain exact fallbacks. |
| Router | **22.763 ms** | **2.08%** | The after-router boundary remains production. Eight-token reuse is retained; eager least-priority release regresses **0.198%** and is closed. |
| Norm/RoPE/gates, metadata, KV/tails and other | **37.474 ms** | **3.43%** | Direct packed-query production is included here at **16.666 ms** and the packed gate remains included. No individual exact subfamily currently has the 5% perfect-removal ceiling needed to displace the selected-projection campaign. |

The current trace gives concrete Amdahl checkpoints; the clean publication
below is a retained performance claim:

- The clean production median is now **654.249 tok/s**. The selector-unset
  1K/4K medians are **579.699/468.608 tok/s**, improving
  **0.991%/0.689%/0.108%** over the preceding packet. The latest clean trace
  has **2,129** dispatches, **1,093.173 ms** inclusive kernel sum, and
  **841.892 ms** kernel span. Selected gate/up remains largest at
  **332.844 ms**; activation/reduce/residual is second at **252.435 ms**.
  Selected Q4/Q6 down is **170.751 ms**, with the Q6 body observed at
  local128/VGPR112/LDS5120B/scratch0. The declared 500 gate is closed.
- Direct packed-query production removes all **144 / 4.907-ms** query
  transposes. The fused producer itself moves **15.623 -> 16.666 ms**, so the
  exact producer-plus-pack boundary improves **20.530 -> 16.666 ms
  (-18.82%)**. Total dispatches fall **2,273 -> 2,129** and traced attention
  falls **63.846 -> 60.564 ms (-5.14%)** with no resident/scratch growth.
- Exact packed output removes all **144 / 3.703-ms** attention-output
  transpose launches. The replacement local128/VGPR8 gate raises gate time
  **7.537 -> 10.318 ms**, so the combined boundary still improves
  **11.240 -> 10.318 ms (-8.20%)** with zero resident/scratch growth.
- Dense-initial metadata elision cuts global+SWA attention
  **153.226 -> 141.846 ms (-7.43%)** with the intended exact launch mix.
- The clean wall must fall from **782.577 ms** to **731.429 ms** for 700 tok/s,
  a further **51.148 ms**. The current profiled kernel span is **110.464 ms**
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
59. **Published production:** for Q6 selected down,
   reuse D4's otherwise-unused 32-bit raw-sum field for two exact `int16` K16
   quant sums. The 160-byte activation block is unchanged, while the producer
   computes each sum once instead of every one of the 48 output-column
   workgroups rebuilding it. On the actual 660.6-MB layer-1 tensor, inclusive
   pack plus production integer-WMMA improves
   **4.1501 -> 4.1162 ms (-0.818%)** across 21 samples with zero BF16
   mismatches. Eleven complete-state pp512 pairs are noisier but positive in
   the paired wall: **6/11 wins**, median **-1.407 ms (+0.179%)**, with exact
   logits, hidden state, KV, cursor, and token 2930. Clean selector-unset
   512/1K/4K improves
   **645.803/575.942/468.311 -> 647.207/576.799/468.431 tok/s
   (+0.217%/+0.149%/+0.026%)**, and the refreshed trace cuts the 23-call Q6
   window **100.367 -> 99.459 ms (-0.905%)** at unchanged
   local128/VGPR112/SGPR128/LDS5120B/scratch0. gfx1151 retains it behind an
   explicit rollback through the active 700 campaign.
   Evidence:
   [`candidate`](../benchmarks/results/2026-07-27-gfx1151-laguna-q6-precomputed-activation-sums-candidate.json) ·
   [`production`](../benchmarks/results/2026-07-27-gfx1151-laguna-q6-precomputed-activation-sums-production.json).
60. **Published production:** reopen Q4 D8
   precomputed activation sums only for the intervening P8 production body,
   and keep the sums in a separate activation-only `int16` sidecar rather
   than widening every 160-byte D8 block. The sidecar is **196,608 bytes** at
   M512 and **786,432 bytes** at the M2048 scratch ceiling; resident expert
   weights and D8 bytes are unchanged. On actual layer-1 natural routing,
   pack-inclusive M256 improves **4.4261 -> 4.4069 ms (-0.434%)** and M512
   improves **6.7309 -> 6.6681 ms (-0.933%)** across 41 counter-rotated
   samples with identical output checksums. Eleven complete-state pp512 pairs
   improve **644.427 -> 645.724 tok/s (+0.201%)** at the independent
   medians, save **2.491 ms** at the paired median, and win **9/11** pairs;
   logits, hidden state, KV, cursor, and token 2930 are exact. This differs
   materially from item 32's rejected widened-block result, which saved only
   0.339 ms and won 3/6 post-cold pairs. Keep the explicit rollback through
   the active 700 campaign. The clean cached trace confirms the intended
   local128/VGPR96/LDS3072B consumer and cuts selected gate/up
   **334.229 -> 330.720 ms (-1.050%)**. Clean selector-unset 512/1K/4K is
   **649.791/576.589/468.830 tok/s**: pp512 improves **0.399%**, 4K improves
   **0.085%**, and 1K is aggregate-flat at **-0.036%**. A dedicated
   same-process 1K gate resolves that wobble in the paired wall:
   **-4.428 ms, 7/11 wins**, with exact complete state.
   Evidence:
   [`candidate`](../benchmarks/results/2026-07-27-gfx1151-laguna-q4-precomputed-activation-sums-candidate.json) ·
   [`production`](../benchmarks/results/2026-07-27-gfx1151-laguna-q4-precomputed-activation-sums-production.json).
61. **Rejected and removed before integration:** split the production P8
   gate/up contraction into two local128 launches, retain gate BF16 scratch,
   and form the exact BF16 SiLU plus D4 cache directly in the up epilogue.
   This removes the up BF16 write, the later gate/up reread, and the standalone
   fused pack without local256 or cross-wave result exchange. The cache is
   byte-identical, and both consumers remain VGPR96/LDS3072B/scratch0, but 41
   counter-rotated samples regress M256
   **4.4945 -> 4.5598 ms (+1.453%)** and M512
   **6.8679 -> 6.9840 ms (+1.690%)**. The additional launch costs more than
   the removed pack and traffic repay. Every export, wrapper, fixture, and
   harness mode is removed.
   Evidence:
   [`2026-07-27-gfx1151-laguna-q4-split-fused-silu-pack-rejected.json`](../benchmarks/results/2026-07-27-gfx1151-laguna-q4-split-fused-silu-pack-rejected.json).
62. **Rejected and removed:** keep packed-query F32 hipBLASLt QK, but replace
   wave32 causal softmax, library PV, normalized-score traffic, and output
   unpack with one local128 fused consumer. The pp512 trace gives this
   formulation a **35.528 ms** perfect-removal ceiling. An eight-row tile is
   CPU-reference green across contexts 128/256/384/512 and wins the isolated
   context-128 composite, but seven production pairs regress
   **646.665 -> 643.218 tok/s (-0.533%, 1/7 wins)**. It traces
   VGPR176/LDS16384B. Doubling reuse to 16 rows raises resources to
   VGPR248/LDS32768B and is slower at every context:
   **0.084/0.237/0.333/0.462 ms** versus row8
   **0.063/0.217/0.309/0.422 ms**. The scalar PV association also changes
   complete state, although token 2930 remains stable. Every kernel/export,
   wrapper, session switch, fixture extension, and harness is removed.
   Attention now reopens only for a cooperative causal primitive that also
   avoids masked QK work.
   Evidence:
   [`2026-07-27-gfx1151-laguna-attention-fused-softmax-pv-rejected.json`](../benchmarks/results/2026-07-27-gfx1151-laguna-attention-fused-softmax-pv-rejected.json).
63. **Published exact production:** keep the three qualified library PV
   outputs in their native head-major M128 tiles and consume that mixed
   generic/head-major buffer in one softplus gate. Direct `(row, head)`
   local128 mapping removes runtime division and preserves the exact BF16
   boundary. Eleven complete-state pp512 pairs are exact and improve
   independent medians **645.735 -> 647.920 tok/s (+0.338%)**, while the
   paired median is flat inside noise. The clean trace supplies the retain
   signal: output-unpack falls **144 launches / 3.703 ms -> 0**, total
   dispatches fall **2,417 -> 2,273**, and transpose plus gate falls
   **11.240 -> 10.318 ms (-8.20%)**. The packed gate is
   local128/VGPR8/LDS0/scratch0. Clean selector-unset 512/1K/4K is
   **647.826/575.732/468.103 tok/s**, within
   **-0.302%/-0.149%/-0.155%** aggregate variance of the preceding packet.
   Evidence:
   [`candidate`](../benchmarks/results/2026-07-27-gfx1151-laguna-attention-packed-output-gate-candidate.json) ·
   [`production`](../benchmarks/results/2026-07-27-gfx1151-laguna-attention-packed-output-gate-production.json).
64. **Published exact production:** write the same three qualified query tiles
   head-major directly from the fused per-head RMSNorm/RoPE producer. The
   planner independently qualifies the dense-initial library route before
   production and hard-errors if the later actual preappend/library decision
   disagrees. Eleven complete-state pp512 pairs improve
   **647.210 -> 650.651 tok/s (+0.532%, 7/11 wins)** with every token/logit/
   hidden/KV/cursor hash equal. The clean trace removes all
   **144 / 4.907-ms** query transposes, cuts dispatches **2,273 -> 2,129**,
   and improves producer-plus-pack **20.530 -> 16.666 ms (-18.82%)**. The
   producer is local256/VGPR16/LDS0/scratch0. Clean selector-unset 512/1K/4K
   reaches **654.249/579.699/468.608 tok/s**, improving
   **0.991%/0.689%/0.108%**.
   Evidence:
   [`candidate`](../benchmarks/results/2026-07-27-gfx1151-laguna-attention-packed-query-producer-candidate.json) ·
   [`production`](../benchmarks/results/2026-07-27-gfx1151-laguna-attention-packed-query-producer-production.json).
65. **Rejected and removed:** reduce only rows>1 routed width while preserving
   exact model-declared top-10 c=1 decode. The mechanism has enough wall
   ceiling: five counter-rotated pp512 samples measure
   **648.578/684.313/720.130 tok/s** for top-10/top-9/top-8. Both approximate
   widths fail the clean ten-prompt, four-category, 320-step absolute gate.
   Top-8 reaches max KL **0.671401** and top-9 **0.452960**, each at
   **314/320** top-1; Poolside-only KL remains deceptively small at
   **0.00018003/0.00004888**. The runtime setter, routing-replay changes,
   harness selector, category lanes, and focused tests are removed.
   Production remains top-10 for prefill and decode.
   Evidence:
   [`top-8`](../benchmarks/results/2026-07-27-gfx1151-laguna-prefill-topk8-absolute-rejected.json) ·
   [`top-9`](../benchmarks/results/2026-07-27-gfx1151-laguna-prefill-topk9-absolute-rejected.json).
66. **Measured diagnostic; one bounded candidate remains:** capture normalized
   F32 route weights beside selected IDs without changing normal generation.
   Across all 47 sparse layers at M512, the final route has median mass
   **0.06837** and the final two have median combined mass **0.13948**.
   Dropping two only when their combined mass is at most **0.10** removes just
   **3.064%** of lanes, which cannot close the 700 gap. Threshold **0.15**
   affects **63.788%** of routed rows and removes **12.758%** of lanes; it is
   the sole dynamic-width candidate with a plausible ceiling. It must be a
   model-wide rule, preserve exact c=1 top-10, and pass an extended-512
   complete category/heldout gate before promotion. No prompt-conditioned,
   observed-output-conditioned, or fixed-prefix exemption is admissible.
   Evidence:
   [`2026-07-27-gfx1151-laguna-routing-tail-mass.json`](../benchmarks/results/2026-07-27-gfx1151-laguna-routing-tail-mass.json).
67. **Rejected and removed:** the final-two combined-mass threshold **0.15**
   passes its five-pair pp512 speed screen at
   **641.668 -> 687.804 tok/s (+7.190%, 5/5 wins)**, but fails the
   deterministic extended-512 ten-prompt absolute gate. Suite maximum KL is
   **3.649289** versus **0.05**; top-1 is **297/320 (92.813%)**. Every
   category violates KL at **2.907917/0.625267/3.649289/2.751395** for
   code/general-English/general-Japanese/mixed. Poolside alone keeps matching
   top-1 but reaches KL **0.414191**. Repeats are deterministic and lifecycle
   returns exactly to zero, so this is a numerical rejection. The prune
   kernel, nullable combine variants, runtime setter, harness mode, category
   lane, and focused tests are removed. Production remains exact top-10.
   Evidence:
   [`2026-07-27-gfx1151-laguna-route-tail15-absolute-rejected.json`](../benchmarks/results/2026-07-27-gfx1151-laguna-route-tail15-absolute-rejected.json).
68. **Rejected and removed:** a native triangular BF16-WMMA QK producer
   consumed resident cache K directly, skipped complete causal-mask tiles,
   and shared each key tile across the six/nine GQA query heads while leaving
   production F32 softmax/PV/output unchanged. The global-48-head and
   SWA-72-head primitives pass their CPU tolerance fixtures, and complete
   attention passes at contexts 128/256/384/512. Performance is decisive in
   the other direction: after removing all LDS barriers and folding SWA's
   ninth head into one local288 workgroup, context-256 QK is
   **0.104837 ms** versus tuned packed F32 hipBLASLt **0.085120 ms
   (+23.16%)**; context 512 is **0.207429 vs 0.156053 ms (+32.92%)**.
   The best body is VGPR144/SGPR128/LDS0/scratch0. Every kernel, wrapper,
   runtime branch, and candidate test is removed; production retains packed
   F32 hipBLASLt QK.
   Evidence:
   [`2026-07-27-gfx1151-laguna-attention-triangular-bf16-wmma-qk-rejected.json`](../benchmarks/results/2026-07-27-gfx1151-laguna-attention-triangular-bf16-wmma-qk-rejected.json).

### Next exact and quality-gated attacks

The activation-only Q4 repair branch is exhausted and removed. Raw-nibble P8
prefetch is now exact Q4 gate/up production and has transferred successfully
to Q4 selected down:

1. **Trace/publication complete:** selector-unset production is now
   **654.249/579.699/468.608 tok/s**. The refreshed pp512 trace has
   **2,129** dispatches and cuts the direct packed-query producer boundary
   **20.530 -> 16.666 ms (-18.82%)**. Selected gate/up plus down remains the
   dominant inclusive expert window at **503.595 ms**.
2. Attack the selected-expert path with a mechanism
   that changes physical bytes, cross-tile reuse, or a measured
   synchronization/latency limiter. Payload-only P8 is retained; decoded,
   metadata-carrying, non-temporal, packed-metadata, pure axis-swap, and
   two/four-row-group launch-order schedules are now rejected. Direct row64
   is also closed on the current P8 body, including dense-expert partitioning;
   paired local256 gate/up-to-D4 fusion and the local128 split/direct-D4
   formulation are closed as well. The next expert
   candidate must reduce weight bytes without extending accumulator lifetime,
   widening the workgroup, exchanging full results through LDS, or weakening
   P8's complete next-K32 payload coverage. Do not pair-share those payload
   gathers through wave shuffles; coalescing already removes the physical
   traffic duplication. Fixed prefill top-8/top-9 are category-rejected despite
   reaching **720.130/684.313 tok/s**. The only remaining routed-width screen
   was item 66's final-two combined-mass threshold **0.15**; item 67 rejects
   it on extended-512 quality despite a **687.804 tok/s** speed screen. No
   routed-width approximation remains open.
3. Do not widen the 128-row attention slice or retry a library-QK plus scalar
   fused-softmax/PV tail: M256 remains slower after exhaustive algorithm
   tuning, and row8/row16 fused tails lose production despite removing the
   score round trip. A direct-cache triangular BF16-WMMA QK producer is also
   closed at **+23.16%/+32.92%** for contexts 256/512. Reopen attention only
   for a cooperative causal primitive that avoids masked QK work and the
   normalized-score materialization without paying a packed-F32-to-BF16 query
   conversion or 144-VGPR score-tile lifetime.
4. Reopen any other closed family only if a future trace leaves a **>=5%**
   perfect-removal ceiling or a newly supported library algorithm changes a
   prior premise. No further activation-only D4 role policy is admissible
   without a new numerical representation.

The stretch target remains **>=700 tok/s**, i.e. **<=731.429 ms** for pp512.
Current production is **654.249 tok/s / 782.577 ms**, leaving **51.148 ms**.
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
**654.249 tok/s**. The 700 stretch and stronger streaming/roofline rows remain
active targets.

All headline rows also report canonical category-weighted prefill and
128/1K/4K behavior. A repeated-token 512 number cannot promote a path by itself.

## Current production summary

This is the compact pause-point view. Short repeated medians remain the
promotion headline; the six-shape row is a clean one-pass 128K-capacity
anti-overtuning sweep; decode is a separate three-repeat eager c=1 snapshot.
They share the same Q4_K_M GGUF, BF16 KV, gfx1151 production defaults, and
two-queue policy, but their timing scopes are not interchangeable.

| Metric | Current result | Protocol / status |
| --- | ---: | --- |
| Repeated short prefill, 512/1K/4K | **654.249 / 579.699 / 468.608 tok/s** | Retained selector-unset production headline |
| One-pass capacity sweep, 512/1K/4K/32K/64K/128K | **614.031 / 666.901 / 609.879 / 365.481 / 247.408 / 149.308 tok/s** | Clean anti-overtuning closure; exact positions and lifecycle |
| Simple p512/d128 eager c=1 decode | **11.471 tok/s** | Median of three; exactly 127 timed `forward_token` calls |
| Prefill paired with the decode snapshot | **654.859 tok/s** | Median of three; **+0.093%** versus the canonical pp512 headline |
| Absolute quality | **0.049542582 max KL; 316/320 top-1** | Complete ten-prompt all-exact gate passes |
| Next short-prefill target | **700 tok/s / 731.429 ms** | Current canonical wall **782.577 ms**; **51.148 ms** remains |

All three p512/d128 runs produce token **2930** first, token **74107** last,
position **638**, and the same complete 128-token hash. Tracked allocations
return to zero. This is an eager-decode snapshot, not a decode optimization
claim. The current eager global-attention ABI admits cache capacity only
through 4,096, so the 1K–128K publication remains prefill-only; no
long-context decode number is inferred.

Evidence:
[`p512/d128 snapshot`](../benchmarks/results/2026-07-28-gfx1151-laguna-p512-d128-eager-snapshot.json) ·
[`short production`](../benchmarks/results/2026-07-27-gfx1151-laguna-attention-packed-query-producer-production.json) ·
[`final six-shape sweep`](../benchmarks/results/2026-07-28-gfx1151-laguna-long-context-final-sweep.json).

## 2026-07-27 pause point: 654 tok/s production and six-shape closure

This is a valid pause point, not the end of the campaign. Clean selector-unset
production is **654.249/579.699/468.608 tok/s** at 512/1K/4K, with
**782.577 ms** pp512 wall. The absolute ten-prompt quality gate remains
**0.049542582** maximum KL and **316/320** top-1, leaving only
**0.000457418** KL headroom. Reaching 700 requires **51.148 ms** from the
current wall, or **6.99%** more throughput.

The requested anti-overtuning sweep ran every shape in one resident
128K-capacity production session with matrix chunks of 2,048, attention chunks
of 128, BF16 KV, two queues, and one timing repetition:

| Prompt | Chunks | Wall | Prefill | Versus repeated production median |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 1 | 0.823 s | **622.009 tok/s** | **-4.928%** |
| 1K | 1 | 1.768 s | **579.152 tok/s** | **-0.094%** |
| 4K | 2 | 8.710 s | **470.270 tok/s** | **+0.355%** |
| 32K | 16 | 152.624 s | **214.698 tok/s** | new closure shape |
| 64K | 32 | 496.497 s | **131.997 tok/s** | new closure shape |
| 128K | 64 | 1,812.326 s | **72.323 tok/s** | new closure shape |

Final positions are exact through 131,071, every recorded next token is
deterministic, and all **85.256 GB** of tracked resident allocation returns to
zero active allocations. The 1K/4K single samples agree with their repeated
production medians and the long rows form a smooth scaling curve; there is no
matrix-chunk shape cliff suggesting that the retained kernels were tuned only
for pp512. This is an attribution baseline over one repeated canonical prompt,
not a new multi-prompt quality or repeated-median performance claim.

The pp512 singleton inside a 128K-capacity session is **4.928%** below the
normal repeated median while 1K/4K remain flat. Treat that as an explicit
resident-capacity/bucketing diagnostic, not a proven regression. The
superlinear long-context wall is instead consistent with the known global
attention path above 512: it falls back from the retained dense-initial
hipBLASLt route to scalar online Q-row attention.

Evidence:
[`six-shape sweep`](../benchmarks/results/2026-07-27-gfx1151-laguna-prefill-six-shape-sweep.json) ·
[`current production`](../benchmarks/results/2026-07-27-gfx1151-laguna-attention-packed-query-producer-production.json).

### Latest 350+ sprint: retained production milestones

Timestamps are JST. This is the compact production ledger; the detailed
candidate/default/trace packets remain in the evidence index and benchmark
changelog.

| Timestamp | Retained iteration | Before -> after | Delta |
| --- | --- | ---: | ---: |
| 2026-07-25 10:31 | Start of the latest 350+ sprint | baseline -> **354.820 tok/s** | production gate closed |
| 2026-07-25 17:53 | Exact qrow4 attention | 354.820 -> **364.839** | **+2.824%** |
| 2026-07-25 22:18–22:46 | Gate/up and down row-vector consumers | 368.203 -> **385.997** | **+4.832%** combined |
| 2026-07-26 01:45 | Expert wave-column remap | 385.602 -> **432.355** | **+12.125%** |
| 2026-07-26 03:09–03:31 | Direct Q4 gate/up and down wave decode | 449.020 -> **480.629** | **+7.040%** combined |
| 2026-07-26 04:04–05:59 | Dense Q6, row64, parallel compaction, router tile8 | 481.950 -> **503.349** | **+4.440%** combined; 500 closed |
| 2026-07-26 09:44–13:31 | Preappend/cache attention, qmicro, cached metadata, qrow6 | 505.084 -> **547.064** | **+8.312%** combined |
| 2026-07-26 18:02–21:16 | Dense-initial attention and exact branch concurrency | 551.459 -> **566.839** | **+2.789%** combined |
| 2026-07-26 23:54 | Q6 selected-down integer WMMA | 573.354 -> **576.137** | **+0.485%** pp512; 1K/4K **+2.425%/+2.883%** |
| 2026-07-27 02:20 | F32 hipBLASLt dense-initial attention | 577.396 -> **623.050** | **+7.907%** |
| 2026-07-27 02:57–03:44 | Packed BLAS queries and wave-per-row softmax | 623.050 -> **632.618** | **+1.536%** combined |
| 2026-07-27 08:59–09:23 | Q6 weight and activation prefetch | 632.618 -> **639.114** | **+1.027%** combined |
| 2026-07-27 10:24–14:02 | Q4 raw prefetch, F16 schedule, Q6/Q4 activation sums | 639.114 -> **649.791** | **+1.670%** combined |
| 2026-07-27 15:17 | Packed attention output gate | 649.791 -> **647.826** | aggregate **-0.302%**; retained exact boundary **-8.20%** |
| 2026-07-27 15:52 | Direct packed-query producer | 647.826 -> **654.249** | **+0.991%** |

From the sprint-opening **354.820** to **654.249 tok/s**, retained production
improved **84.389%**. From the pre-campaign **76.226 tok/s** control it
improved **758.301%**. The external 344.56 tok/s Vulkan row is now exceeded by
**89.879%**, subject to its different token/KV/numerical contract.

### Latest 350+ sprint: bounded failures and reversions

These failures are useful closed evidence, not abandoned loose ends. Candidate
code was removed unless a separately exact decode primitive retained value.

| Timestamp | Reverted/rejected iteration | Before -> candidate | Reason |
| --- | --- | ---: | --- |
| 2026-07-25 17:01 | Multi-K expert staging | K32 353.516 -> K64 318.850 / K128 269.071 | **-9.80%/-23.88%** |
| 2026-07-25 17:14 | 64-row expert tile | 353.787 -> 345.141 | **-2.44%** |
| 2026-07-26 16:38 | Byte-neutral Q4 qmicro prefill | 9.402 -> 9.571 ms | **+1.795%** leaf wall |
| 2026-07-27 04:08 | Persistent expert-row path | 2.873 -> 18.456 ms | **6.424x** slower |
| 2026-07-27 04:51–06:44 | D4 role splits and selective repair | speed-positive leaves, max KL **>=0.076** | quality gate failed |
| 2026-07-27 12:20 | M256 merged attention | 792.662 -> 811.343 ms | **-2.302%** throughput |
| 2026-07-27 12:33 | Current-body Q4 row64 | 6.528 -> 7.257 ms | **+11.17%** leaf wall |
| 2026-07-27 12:51 | Paired Q4 SiLU pack | 6.910 -> 7.445 ms | **+7.74%** leaf wall |
| 2026-07-27 12:57–13:02 | Partial P4 and pair-shared raw prefetch | +1.49% / +21.43% wall | both slower |
| 2026-07-27 14:43 | Fused softmax/PV tail | 646.665 -> 643.218 tok/s | **-0.533%** |
| 2026-07-27 16:12–16:20 | Routed top-8/top-9 | +11.03%/+5.51% speed | max KL **0.671401/0.452960** |
| 2026-07-27 17:10 | Low-mass route pruning | 641.668 -> 687.804 tok/s | **+7.190%**, but max KL **3.649289** |
| 2026-07-27 17:29 | Triangular BF16-WMMA QK | 0.085120 -> 0.104837 ms at context 256 | **+23.16%**; context 512 **+32.92%** |

### Long-context optimization campaign: first pass closed

The pp512-to-700 expert lane remains valid but was paused for the bounded
LC-0 through LC-6 campaign.
Gate/up plus down still occupy about **503.595 ms** of the clean
**782.577-ms** pp512 wall, and a credible future body must reduce physical
weight bytes or create real cross-tile reuse. That lane resumes after the
first long-context architecture target; a fresh profile must now decide
whether global attention is still dominant. Do not repeat row64, K64 staging,
non-temporal loads, grid reordering, pair-shared prefetch, metadata-only
repacks, or paired-SiLU fusion without a genuinely new premise.

The clean final sweep uses one 128K-capacity resident session, selector-unset
M2,048 matrix/global attention, M128 SWA, and one pass per shape. The
same-GGUF, same-device llama.cpp Vulkan baseline uses clean `c0bc8591e`,
batch 2,048, ubatch 512, F16 KV, FlashAttention, one resident load, and one
pass per shape:

| Shape | hipEngine LC-0 | hipEngine final | llama.cpp Vulkan | Final over Vulkan |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 622.009 tok/s | **614.031 tok/s** | 341.999 tok/s | **+79.542%** |
| 4K | 466.482 | **609.879** | 333.502 | **+82.871%** |
| 64K | 132.831 | **247.408** | 126.624 | **+95.388%** |
| 128K | 72.139 | **149.308** | 65.584 | **+127.659%** |

The requested six-shape final is
**614.031/666.901/609.879/365.481/247.408/149.308 tok/s** at
512/1K/4K/32K/64K/128K. Relative to the pre-campaign closure it changes
**-1.283%/+15.151%/+29.687%/+70.230%/+87.435%/+106.446%** and saves
**934.451 seconds** at 128K. Its 4K/64K/128K rows reproduce the canonical
LC-3 gates within **-0.313%/+0.354%/-0.251%**, so the pp512 singleton is
capacity-session variance rather than a long-context regression.

KV numerical policy still differs (hipEngine BF16 versus Vulkan F16), and
llama-bench has its own prompt stream/timing boundary, so Vulkan remains a
source/performance floor rather than a quality-equivalent comparator.
Nevertheless, final hipEngine retains **60.349%** of its 64K rate at 128K
versus Vulkan's **51.794%**: the old tail-retention concern is closed for this
campaign.

Evidence:
[`same-GGUF Vulkan long baseline`](../benchmarks/results/2026-07-27-gfx1151-laguna-llamacpp-vulkan-long-context-baseline.json) ·
[`hipEngine LC-0 attack control`](../benchmarks/results/2026-07-27-gfx1151-laguna-lc0-attack-control.json) ·
[`hipEngine pre-campaign closure`](../benchmarks/results/2026-07-27-gfx1151-laguna-prefill-six-shape-sweep.json) ·
[`hipEngine final closure`](../benchmarks/results/2026-07-28-gfx1151-laguna-long-context-final-sweep.json).

#### Long-context sprint ledger

Timestamps are JST. “Superseded” means the result was a successful bounded
stepping stone whose successor kept the architecture while improving its
scratch or wall; “rejected” means production stayed on the prior row.

| Timestamp | Iteration | Before -> after | Decision |
| --- | --- | ---: | --- |
| 2026-07-27 22:49 | LC-0 coherent attribution | baseline -> **72.139 tok/s** at 128K | accepted control |
| 2026-07-28 01:01 | Capacity-sized F32 global hipBLASLt | 72.139 -> **88.073** (**+22.088%**) | retained, then superseded due to 4.298 GB scratch |
| 2026-07-28 02:00 | Exact 4K-block online global | 88.073 -> **99.100** (**+12.521%**) | retained; scratch **-96.655%** |
| 2026-07-28 02:58 | Tensorized rolling SWA | 99.100 -> **103.520** (**+4.460%**) | retained |
| 2026-07-28 03:50 | Global M2,048 query reuse | 103.520 -> **149.684** (**+44.594%**) | retained; wider SWA rejected |
| 2026-07-28 04:32 | Dense-contiguous cache widen | 0.250249 -> **0.234780 ms** (**-6.181%**) | retained exact sub-window; 128K **-0.291%** neutral |
| 2026-07-28 05:12 | M4,096/M8,192 matrix chunks | 149.684 -> **147.939** at M4,096 (**-1.166%**) | rejected; +1.756/+5.268 GB |
| 2026-07-28 05:20 | Capacity/lazy-KV and secondary routes | short shapes **-0.425%/+0.654%/-0.004%** with +6.007 GiB | closed; no production change |
| 2026-07-28 05:43 | Clean six-shape closure | 72.323 -> **149.308** at 128K (**+106.446%**) | accepted; all gates pass |

#### Laguna-specific long-context roofline

Do not use the hybrid Qwen3.x 35B GDN/linear-attention curve as Laguna's
optimal shape. The production GGUF metadata confirms that **all 48 Laguna
decoder blocks use softmax attention**: **12 global** layers
(`layer % 4 == 0`) and **36 sliding-window** layers with a 512-token window.
There are no GDN or other linear-attention layers. Global attention has 48
query heads, while SWA has 72; both use eight KV heads and head dimension 128.
The model also has one dense MLP block and 47 sparse-MoE blocks; those
projections, experts, norms, and routing remain predominantly linear in prompt
length.

For prompt length `C`, the exact attention pair counts are:

```text
global_pairs(C) = C * (C + 1) / 2
swa_pairs(C)    = C * (C + 1) / 2                         when C <= 512
                  512 * 513 / 2 + (C - 512) * 512        otherwise

global_flops(C) = 4 * 128 * 48 * 12 * global_pairs(C)
swa_flops(C)    = 4 * 128 * 72 * 36 * swa_pairs(C)
```

The factor four counts QK and PV multiply-adds. This makes the theoretical
scaling genuinely mixed: SWA becomes linear after 512, while the 12 global
layers remain quadratic.

| Prompt | Global QK+PV | SWA QK+PV | Total attention | SWA / global |
| ---: | ---: | ---: | ---: | ---: |
| 4K | 2.475 TFLOP | 2.610 TFLOP | 5.084 TFLOP | **105.46%** |
| 16K | 39.585 TFLOP | 10.959 TFLOP | 50.544 TFLOP | **27.68%** |
| 64K | 0.633 PFLOP | 0.044 PFLOP | 0.678 PFLOP | **7.00%** |
| 128K | 2.533 PFLOP | 0.089 PFLOP | 2.622 PFLOP | **3.51%** |

This is why the optimization ordering stood but the short screens mattered.
Global attention is the asymptotic target and owns **96.61%** of attention
arithmetic at 128K, yet SWA owns slightly more attention arithmetic than
global at 4K. The retained global specialization delivered the expected
increasing 16K/64K/128K slope while the separate rolling-SWA owner protected
the 4K route.

The gfx1151 theoretical roofs imply deliberately optimistic
all-attention-only lower bounds at 128K of about **88.3 seconds** at
29.7-TFLOP FP32/VOPD or **44.2 seconds** at 59.39-TFLOP BF16 WMMA. Neither is
an end-to-end promise: they omit softmax, cache traffic, projections, MoE,
norms, routing, output projection, occupancy, and achievable clock/issue
efficiency. They do show that the LC-0 **1,816.9-second** hipEngine, final
**877.863-second** hipEngine, and **1,998.5-second** Vulkan walls are not
hardware-optimal. Required total attention arithmetic divided by complete
wall is **1.443/2.987/1.312 TFLOP/s** for LC-0/final/Vulkan; these are
system-equivalent ratios, not measured attention-kernel throughput.

Use measured staged milestones rather than the peak as a promise:

| 128K milestone | Throughput | Wall | Interpretation |
| --- | ---: | ---: | --- |
| LC-0 control | **72.139 tok/s** | 1,816.9 s | Exact scalar online global route |
| Final clean closure | **149.308 tok/s** | **877.863 s** | **2.070x LC-0**; within 0.462% of the strict 150 milestone |
| First architectural milestone | **>=150 tok/s** | <=873.8 s | At least 2.07x LC-0; trace-backed global wall must fall about 2.75x if other work is unchanged |
| Main long-context target | **>=300 tok/s** | <=436.9 s | At least 4.15x current; requires comparator-independent tiled compute near the FP32 global roof or companion non-global wins |
| Roofline-informed stretch | **>=450 tok/s** | <=291.3 s | At least 6.22x current; requires global plus SWA/linear/capacity progress, not a global-only kernel |

LC-0 now makes these trace-informed campaign targets rather than inferred
promises. At measured 64K rates, the 128K split projects about
**1,482 s global + 87 s SWA + 159 s remaining linear work**, with another
**89 s** between that simple projection and the actual control. Thus 300 is
already close to an FP32-global-roof system target if non-global work stays
fixed, while 450 necessarily depends on later LC stages too. These remain
targets, not promotion gates or claims.

#### Fast architectural-development protocol

The campaign intentionally uses cheap evidence while far from the target:

1. Add one fixed **4K/16K/64K/128K** attack set to the strict profiler and
   capture the missing current 16K hipEngine control. Arbitrary user-selected
   lists remain rejected.
2. During architecture development, run one cached-build control/candidate
   pass at **4K/16K/64K** in one 128K-capacity resident session. Rotate order
   or use a same-session selector when the candidate can switch safely.
3. A directional screen passes only when 4K is at least **0.98x** control,
   16K and 64K are both strictly positive, and at least one of 16K/64K is
   **>=1.10x**. Exact smaller wins may be retained under repository policy,
   but they do not close the major stage.
4. Before starting the next major stage, run one full **128K** candidate gate.
   It must be strictly faster than the retained control, preserve exact final
   position and deterministic next token, keep logits finite, and return all
   tracked allocations to baseline. A stage that misses 128K remains open or
   changes premise.
5. Keep one pp512 non-regression sample at **>=0.98x** current when dispatch or
   shared code changes. A route qualified only above 512 should leave pp512
   structurally unchanged.
6. These one-pass rows are development evidence only. Production promotion
   still requires counterbalanced repetitions, the ten-prompt 320-step
   KL/top-1 gate, h16/h32 E2E, decode within 2%, complete state/lifecycle, and
   the ordinary benchmark rollup.

#### LC-0 — baseline, harness, and attribution

- **Done:** capture the same-GGUF Vulkan 512/4K/16K/64K/128K baseline above.
- **Done:** admit the fixed 4K/16K/64K/128K hipEngine attack set. Arbitrary
  lists remain rejected.
- **Done:** capture one coherent current-production 4K/16K/64K/128K control
  in a single 128K-capacity resident session:
  **466.482/307.953/132.831/72.139 tok/s**. Exact positions, deterministic
  tokens, finite outputs, and full tracked-allocation recovery pass.
- **Done:** admit the fixed **16K/64K** cached trace set so LC-0 can attribute
  both useful scaling points without profiling the 30-minute 128K row.
- **Done:** trace cached 16K and 64K controls and attribute all 12 global
  layers, 36 SWA layers, projections, MoE, and residual families without
  profiling the 30-minute 128K row.
- **Done:** record global-attention achieved FLOP/s, requested K/V bytes, GQA
  reread factor, dispatches, resource tuple, and wall-growth fit. The
  instruction-requested traffic is explicitly not a physical DRAM counter.
- **Done:** freeze a bounded-state CPU-reference block-streaming oracle against
  dense GQA at partial query/key tiles, positions 511/512/513, global/SWA
  masks, and the final 128K position.

The cached trace is stable against the unprofiled LC-0 control:
**309.180 tok/s** at 16K (**+0.399%**) and **132.790 tok/s** at 64K
(**-0.031%**). Its kernel span misses complete wall by only
**11.5/19.7 ms**, so long-context prefill is kernel-bound rather than
submission-bound.

| Component | 16K wall | Share | 64K wall | Share | 16K -> 64K |
| --- | ---: | ---: | ---: | ---: | ---: |
| Global attention | **22.670 s** | **42.78%** | **370.549 s** | **75.08%** | **16.345x** |
| SWA attention | **10.462 s** | **19.74%** | **43.499 s** | **8.81%** | **4.158x** |
| Complete wall minus attention | **19.860 s** | **37.48%** | **79.483 s** | **16.10%** | **4.002x** |
| Complete wall | **52.992 s** | 100% | **493.532 s** | 100% | **9.313x** |

The trace therefore resolves the scaling model: global attention is
quadratic, while both SWA and the remaining model are linear after the
512-token window fills. Logical global QK+PV reaches only
**1.746/1.709 TFLOP/s** at 16K/64K; SWA reaches
**1.047/1.020 TFLOP/s**. Holding the measured 64K rates projects
**1,728.3 seconds** at 128K versus the observed **1,816.9 seconds**
(**-4.88%**), so the mixed-attention model explains about 95% of the full
128K wall before capacity/cache/thermal residuals.

The root cause is repeated load work inside the scalar topology:

| Family | Current tile | GQA heads/KV | Load-request amplification | 64K requested K+V |
| --- | ---: | ---: | ---: | ---: |
| Global | qrow6, 22 row groups | 6 | **131.76x** | **108.861 TB** |
| SWA | qrow4, 32 row groups | 9 | **288.00x** | **11.046 TB** |

These are executed BF16 K/V vector-load requests after position 512 relative
to loading each K/V vector once per KV head and 128-query tile. They exclude
the first-512 library route, metadata, cache-line effects, and cache hits; the
implied request rates exceed DRAM bandwidth precisely because many rereads hit
cache. They are not physical DRAM measurements. The dominant global symbol is
local32/VGPR88/SGPR128/LDS0/scratch0 with grid **48 x 22 workgroups** per
launch; SWA is local32/VGPR80/SGPR128/LDS0/scratch0 with **72 x 32**.

Evidence:
[`LC-0 attribution`](../benchmarks/results/2026-07-27-gfx1151-laguna-lc0-long-context-attribution.json).

#### LC-1 — real block-streamed global attention

- Replace token-serial independent-query-head scanning for qualified prompt
  prefill with a K/V tile shared by the complete GQA head group. LC-0 proved
  that row blocking and head sharing must be designed together rather than
  treated as independent sequential wins.
- **Rejected and removed:** the exact single-head **Q16 x K64/local128**
  screen reduced theoretical row-group requests about 2.75x but raised the
  resource tuple to **VGPR248/SGPR128/LDS33,792 B/scratch0**. Direct cached
  leaf timing regressed qrow6 by **5.05x/5.33x/5.04x/4.96x** at
  context 512/4K/16K/64K. It never reached a full-model screen. The inline
  leaf command/raw samples were not preserved, so this is explicitly
  non-promotable rejection evidence; the cached resource trace and exact
  output gate are preserved.
- **Also rejected and removed:** a local192 six-wave body assigned one qrow6
  wave to each of a K/V head's six query heads and staged K/V once for all 36
  queries. K64 remained **VGPR248/LDS33,792 B** and delivered only
  **0.244x-0.264x** qrow6 throughput. K32 with
  `launch_bounds(192,2)` reduced resources to
  **VGPR152/LDS16,896 B**, but still delivered only **0.585x-0.601x** at
  context 512/4K/16K/64K. Both were exact and both lost at every shape.
- This closes scalar LDS staging as the LC-1 premise. The LC-0 load-request
  amplification is predominantly cache-served; suppressing requests does not
  accelerate the serial dot, exponential, and PV loop.
- **Ceiling passed:** the packed M128-by-context F32 hipBLASLt route, including
  BF16-cache widening, query/output transpose, QK, wave-row softmax, and PV,
  beats qrow6 by **2.163x/1.406x/1.263x/1.305x/1.250x** at
  512/4K/16K/64K/128K. Maximum absolute output error is
  **4.622e-8**. Screening all 32 zero-workspace heuristics matters:
  algorithm 0 falsely lost at every long shape, while tuned QK/PV pairs are
  **20/25, 28/1, 28/8, and 28/3** at 4K/16K/64K/128K.
- **Transitional production milestone passed:** the separate capacity-sized
  48-head owner qualified only dense-initial global tiles beginning above 4K.
  The first broad `start >= 512` policy was rejected at paired 4K
  **464.555 -> 449.640 tok/s (0.968x)**. Raising qualification to 4K preserved
  the complete 4K path and produced
  **468.065 -> 468.911 tok/s (+0.181%)** at 4K,
  **308.181 -> 332.617 (+7.929%)** at 16K,
  **131.825 -> 154.151 (+16.936%)** at 64K, and
  **72.139 -> 88.073 (+22.088%)** at mandatory 128K. It proved the tensorized
  arithmetic but still widened the full prefix and materialized an F32
  `[48,128,C]` score tile, costing **4.298 GB** scratch at 128K.
- **Bounded block-streamed production successor passed and replaces it:**
  split the key prefix into 4,096-token blocks, widen only the current block,
  and carry per query-head/row F32 maximum, denominator, and output numerator
  across tuned QK/softmax/PV tiles. The final merge is the exact online
  softmax identity; it never materializes the complete score matrix or
  computes future key blocks.
- Inclusive M128 leaf timing improves qrow6
  **7.588 -> 5.456 ms (1.391x)** at 4K,
  **32.387 -> 21.881 (1.480x)** at 16K,
  **131.157 -> 88.035 (1.490x)** at 64K, and
  **272.887 -> 175.591 (1.554x)** at 128K. It also beats the transitional
  full-score route at every long shape. Maximum absolute output error is
  **3.516e-8** and all outputs are finite.
- Same-session complete-model screens preserve or improve the transitional
  route at 4K/16K/64K:
  **467.930 -> 468.495 (+0.121%)**,
  **332.645 -> 334.686 (+0.614%)**, and
  **153.467 -> 165.002 (+7.516%)**, with identical next tokens.
  The mandatory 128K gate improves
  **88.073 -> 99.100 tok/s (+12.521%)**, or
  **1,488.225 -> 1,322.622 seconds**, saving another **165.603 seconds**.
  Relative to the original LC-0 control this is **+37.374%**; relative to
  same-GGUF llama.cpp Vulkan it is **+51.104%**. Token 22746, final position
  131071, and full allocation recovery pass.
- Scratch falls **4,298,113,024 -> 143,753,216 bytes (-96.655%)** and resident
  accounting falls by **4,154,359,808 bytes**. Cached tracing names the block
  widener at local256/VGPR24, online tile softmax at local32/VGPR16, and
  numerator merge at local256/VGPR24; all use zero LDS and scratch. LC-1's
  exact bounded-state global architecture is therefore closed and retained.
- The retained numerical boundary uses FP32 query/dot/output accumulation and
  exact BF16 cache widening, isolating scheduling/reuse from a new
  approximation.
- The separate dense-prefill primitive retains the generic `KVLiveSpans`
  attention chain as its fallback. Decode, verifier, partial, wrapped, and
  evicted paths remain unchanged.
- A BF16-WMMA/MFMA QK/PV body is a separate quality-gated substage. The
  rejected small triangular WMMA screen does not close a genuinely tiled
  block-streamed design, but its resource/performance failure must inform the
  new geometry.

Evidence:
[`single-head Q16xK64 rejection`](../benchmarks/results/2026-07-27-gfx1151-laguna-lc1-single-head-qtile16-k64-rejected.json) ·
[`GQA6 scalar-staging rejection`](../benchmarks/results/2026-07-27-gfx1151-laguna-lc1-gqa6-scalar-staging-rejected.json) ·
[`long F32 hipBLASLt ceiling`](../benchmarks/results/2026-07-27-gfx1151-laguna-lc1-long-f32-hipblaslt-ceiling.json) ·
[`transitional full-score production`](../benchmarks/results/2026-07-27-gfx1151-laguna-lc1-long-f32-hipblaslt-production.json) ·
[`bounded 4K-block production`](../benchmarks/results/2026-07-27-gfx1151-laguna-lc1-block4096-hipblaslt-production.json).

#### LC-2 — share K/V across Laguna GQA heads

- **Co-designed with tensorized LC-1:** one global K/V head serves six query
  heads. Reuse K/V inside the GEMM/MFMA QK/PV tile or library batch instead of
  staging it for six otherwise-serial waves. A winning combined screen can
  close LC-1 and LC-2 together after the mandatory 128K gate; scalar
  cooperative staging is now closed evidence.
- **Global side closed:** the retained packed-query, eight-way library batch
  plus 4K-block online state consumes each widened global K/V block once per
  KV-head batch and has passed the mandatory 128K gate. It closes LC-1 and
  the global half of LC-2 without an extra replicated-head sidecar.
- **SWA side closed and retained:** one SWA K/V head serves nine query heads.
  For every consecutive non-evicted M128 tile beginning at position 512, gather
  exactly 511 historical BF16 ring rows plus 128 current BF16-rounded rows into
  a fixed 639-key union. One packed eight-way F32 QK/PV batch applies a
  row-shifted 512-key diagonal mask. Decode, verifier, partial rows, explicit
  eviction, nonconsecutive positions, and unmeasured backends retain the
  generic span-aware route.
- All 32 zero-workspace QK/PV algorithms were screened; QK25/PV18 improves the
  complete wrap leaf **3.313 -> 0.684 ms (4.846x)** with maximum absolute
  output error **3.818e-8**. Scratch is fixed at **33,554,432 bytes**.
- Complete-model one-run gates improve
  **579.394 -> 670.326 tok/s (+15.694%)** at 1K,
  **469.356 -> 579.269 (+23.418%)** at 4K,
  **334.430 -> 392.424 (+17.341%)** at 16K, and
  **163.985 -> 177.222 (+8.072%)** at 64K. The 512 path is structurally
  unchanged. All paired tokens, positions, determinism, and lifecycle checks
  pass.
- Mandatory 128K improves the retained LC-1 route
  **99.100 -> 103.520 tok/s (+4.460%)**, or
  **1,322.622 -> 1,266.148 seconds**, saving **56.474 seconds**. This is
  **+43.501%** versus the original LC-0 control and **+57.844%** versus
  same-GGUF llama.cpp Vulkan. Token 22746 and final position 131071 match.
  LC-2 is therefore closed; LC-3 query-chunk widening is next.
- Co-design LDS layout, query fragments, and output accumulators with LC-1;
  report physical K/V bytes and reuse rather than assuming a theoretical 6x.
- Preserve score/token order in the exact FP32 lane. Any reassociated or
  reduced-precision lane must pass the complete quality gate before default
  promotion.

Evidence:
[`rolling-SWA tensorized production`](../benchmarks/results/2026-07-28-gfx1151-laguna-lc2-swa-hipblaslt-production.json).

#### LC-3 — widen the attention query chunk

- **Closed and retained for global attention at M2,048.** Exhaustive
  zero-workspace algorithm screens at a 4K K/V block measure inclusive global
  cost per row at **43.309/35.008/32.123/28.329/26.382 us** for
  M128/M256/M512/M1,024/M2,048. M2,048 is **39.09%** below M128 per row;
  QK15/PV1 serves the first 2K context and QK15/PV2 serves full 4K blocks.
  Maximum absolute error across the screened global shapes is **4.284e-8**.
- The production selector lets each complete M2,048 matrix chunk feed one
  global-attention transaction on the 12 full-attention layers. It preserves
  the exact 4K-block online state and bounded BF16-to-F32 cache widen from
  LC-1. Partial matrix tails, verifier, decode, eviction, and unmeasured
  backends retain the established M128 routes.
- **SWA widening is rejected.** Its union QK multiplies M queries by
  `511 + M` keys even though every row sees only 512 keys, so masked
  query/current work grows with the tile. Inclusive cost per row rises
  **5.370 -> 7.875 -> 11.123 -> 12.439 -> 21.124 us** from M128 through
  M2,048. M128/M256/M512 remain within maximum absolute **5.215e-8** of the
  retained qrow2 oracle. Production SWA therefore remains M128.
- Complete-model gates improve retained LC-2
  **579.269 -> 611.795 tok/s (+5.615%)** at 4K,
  **392.424 -> 472.766 (+20.473%)** at 16K, and
  **177.222 -> 246.537 (+39.112%)** at 64K. Mandatory 128K improves
  **103.520 -> 149.684 tok/s (+44.594%)** and
  **1,266.148 -> 875.657 seconds**, saving **390.491 seconds**. This is
  **+107.494%** versus original LC-0 and **+128.233%** versus same-GGUF
  llama.cpp Vulkan. Tokens 7772/81/69407/22746, final positions, finite
  output, and lifecycle all pass.
- Selector-unset validation reports global attention rows **2,048**, SWA rows
  **128**, and **611.795 tok/s** at 4K. gfx1151 promotes
  `LAGUNA_PREFILL_GLOBAL_ATTENTION_ROWS=2048`. LC-4 dense contiguous-cache
  specialization is next.

Evidence:
[`global-M2048 production`](../benchmarks/results/2026-07-28-gfx1151-laguna-lc3-global-m2048-production.json).

#### LC-4 — dense-initial contiguous cache specialization

- **Closed and retained as an exact kernel sub-window win.** The qualified
  dense-initial global cache is allocated in identity physical order. Its new
  block widen computes `logical_start * width + index` directly, removing
  per-element token-position, eviction, block-table, block-offset, and
  physical-capacity work.
- A paired 100-sample M2,048/4K screen improves the exact BF16 K/V widen
  **0.250249 -> 0.234780 ms (-6.181%)**, saving **15.469 us per 4K block**.
  The complete attention transaction is about **53.95 ms**, so the projected
  system effect is only **0.02-0.03%**. Explicit direct attention remains
  within maximum absolute **3.446e-8** of qrow6.
- The 4K/16K/64K one-run production A/B is
  **+2.759%/+0.304%/-0.415%** with exact tokens, positions, and lifecycle.
  Mandatory 128K is **149.249 tok/s** versus LC-3's **149.684 (-0.291%)**,
  also exact and fully recovered. These aggregate swings are much larger than
  the projected effect and are classified as neutral variance; no new
  complete-model topline is claimed.
- Cached tracing confirms the retained contiguous symbol at
  local256/VGPR16/SGPR128/LDS0/scratch0 and **211.797-217.167 us**, versus
  generic VGPR24 and **324.930 us** under the same profiler. gfx1151 promotes
  `LAGUNA_PREFILL_DENSE_CONTIGUOUS_CACHE=true`. Selector-unset
  512/1K/4K is exact at **628.203/669.454/611.359 tok/s**.
- The full `KVLiveSpans` ABI and generic registered block widen remain the
  fallback. Continuation, verifier, decode, explicit eviction, SWA, and
  unmeasured backends never infer dense identity order. Preappend/query
  producer folding is not attempted because the complete attention body did
  not show a measurable independent win. LC-5 matrix chunks above M2,048 are
  next.

Evidence:
[`dense contiguous-cache production`](../benchmarks/results/2026-07-28-gfx1151-laguna-lc4-dense-contiguous-cache.json).

#### LC-5 — matrix chunks above 2,048

- **Closed and rejected for production.** RED/Green lifts only the explicit
  diagnostic ceiling to M8,192, accounts exact scratch, and proves one
  8,192-row transaction can retain global attention M2,048 and SWA M128.
  gfx1151's package default remains M2,048.
- Bounded rows+MoE scratch is **1.756/3.512/7.024 GB** at
  M2,048/M4,096/M8,192. Thus M4,096 adds **1.756 GB** and M8,192 adds
  **5.268 GB** to every session before timing work.
- M8,192 is rejected before 128K. Its long-capacity 4K/16K/64K result is
  **609.368/479.032/245.564 tok/s**, or
  **-0.397%/+1.325%/-0.394%** versus retained. It is weaker than M4,096 at
  every directional shape while spending another **3.512 GB** scratch.
- M4,096 is exact and directionally positive at
  **611.937/479.656/247.668 tok/s** for 4K/16K/64K
  (**+0.023%/+1.457%/+0.459%**). Mandatory 128K nevertheless falls
  **149.684 -> 147.939 tok/s (-1.166%)** and
  **875.657 -> 885.984 seconds**, losing **10.327 seconds** while consuming
  the extra 1.756 GB. Token 22746, final position 131071, finite output, and
  lifecycle pass, so this is a performance/capacity rejection rather than a
  correctness failure.
- The result falsifies repeated whole-model weight streaming as the remaining
  dominant LC bottleneck at M2,048. Wider routed batches touch more experts
  and trade fewer transactions for more saturated per-transaction work;
  global attention's quadratic component is unchanged.

Evidence:
[`wide matrix rejection`](../benchmarks/results/2026-07-28-gfx1151-laguna-lc5-wide-matrix-rejected.json).

#### LC-6 — capacity buckets, lazy KV, and secondary bandwidth work

- **Closed without a production change.** A clean M2,048 session sized for
  128K carries **6.007 GiB** more resident allocation than the 4K-capacity
  control, yet 512/1K/4K changes only
  **-0.425%/+0.654%/-0.004%**:
  **625.532/673.832/611.334 tok/s** versus
  **628.203/669.454/611.359**. Tokens, positions, and lifecycle match. The
  earlier **-4.928%** pp512 singleton does not reproduce, so it is closed as
  one-run variance rather than evidence for lazy KV or allocation buckets.
- Exact BF16 cache widening is not the remaining wall. LC-4 reduced its 4K
  sub-window to **0.234780 ms** inside about **53.95 ms** of complete
  attention, only **0.02-0.03%**. Q8 KV would add a new quality surface for
  no measured system ceiling and remains closed.
- Existing library screens remain conclusive for this runtime: AOTriton lacks
  Laguna's head-dim-128 GQA causal `KVLiveSpans` geometry; concatenated
  source-F16 QKV saves only **2.891 ms** before output restriding; and
  `hipblaslt_ext::GroupedGemm` exposes zero applicable gfx1151 algorithms.
  Reopen only after a ROCm/Tensile/AOTriton capability change or a new trace
  moves one of these leaves onto the critical path.
- LC-5's diagnostic-only M4,096/M8,192 constructor/profile surface is removed
  now that its stated LC-6 lifetime expired. Production and the explicit
  validation ceiling are both M2,048.
- **Final closure passes.** Clean selector-unset
  512/1K/4K/32K/64K/128K is
  **614.031/666.901/609.879/365.481/247.408/149.308 tok/s**. Every expected
  token and final position passes, all tracked allocation returns to zero,
  and 4K/64K/128K reproduce the retained LC-3 band within **+/-0.354%**.

Evidence:
[`capacity and secondary closure`](../benchmarks/results/2026-07-28-gfx1151-laguna-lc6-capacity-secondary-closure.json) ·
[`final six-shape closure`](../benchmarks/results/2026-07-28-gfx1151-laguna-long-context-final-sweep.json).

#### Next measured avenues after the LC-0 through LC-6 closure

1. **Re-profile the retained 64K route before choosing another kernel.** The
   LC-0 trace predates the 4K-block, SWA, and M2,048-query owners. A cached
   64K trace should re-split global attention, SWA, and linear/MoE wall and
   report tensor-core utilization plus QK/PV/merge launch boundaries.
2. **Build an in-tree fused head-dim-128 GQA FlashAttention owner if global
   still dominates.** Consume `KVLiveSpans` directly, keep bounded online
   softmax state and F32 accumulation, and remove the separate query pack,
   BF16-cache-to-F32 tiles, QK/PV library calls, and merge launches. Installed
   AOTriton cannot supply this geometry; copying its unsupported adapter is
   not the premise.
3. **Decouple attention-query reuse from whole-model matrix width.** LC-5
   rejected M4,096 because it widened every projection/MoE scratch plane. A
   layer-local or two-M2,048 attention window could reuse each global K/V
   block across 4,096 queries without paying the rejected **1.756 GB** or
   changing expert routing geometry. This needs a new scheduling proof and
   the same 4K/16K/64K/128K gates.
4. **Resume pp512 physical-byte work only after that trace.** At short context
   the mapped gate/up plus down window is still about **503.595 ms** of the
   **782.577-ms** production wall. The next credible expert body must remove
   physical weight bytes or cross-tile work; prior row64, K64 staging,
   non-temporal, grid-order, pair-shared, and metadata-only variants remain
   closed.

Do not reopen lazy KV, Q8 KV, M4,096 whole-model chunks, source-F16 grouping,
or unchanged AOTriton/GroupedGemm routes without new capability or trace
evidence.

Long-context stop rules:

- do not use the hybrid Qwen3.x/GDN curve as Laguna's roofline;
- do not materialize O(N^2) scores or add unbounded context-sized scratch;
- do not generalize a dense-prompt route to verifier/decode/eviction semantics;
- do not promote BF16 query arithmetic from a primitive tolerance check alone;
- do not move to the next major LC stage after a failed 128K gate;
- do not pay for repeated 128K medians while an architecture is still outside
  the directional band.

Approximate expert routing is closed under the current **0.05** KL contract:
there is too little quality headroom, and both top-width and low-mass screens
failed by large margins. The next work should remain exact unless an explicit
quality-budget change is approved.

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

- `benchmarks/results/2026-07-28-gfx1151-laguna-p512-d128-eager-snapshot.json`
- `benchmarks/results/2026-07-27-gfx1151-laguna-llamacpp-vulkan-long-context-baseline.json`
- `benchmarks/results/2026-07-27-gfx1151-laguna-prefill-six-shape-sweep.json`
- `benchmarks/results/2026-07-27-gfx1151-laguna-attention-packed-output-gate-production.json`
- `benchmarks/results/2026-07-27-gfx1151-laguna-attention-packed-output-gate-candidate.json`
- `benchmarks/results/2026-07-27-gfx1151-laguna-attention-packed-query-producer-production.json`
- `benchmarks/results/2026-07-27-gfx1151-laguna-attention-packed-query-producer-candidate.json`
- `benchmarks/results/2026-07-27-gfx1151-laguna-q4-precomputed-activation-sums-production.json`
- `benchmarks/results/2026-07-27-gfx1151-laguna-q4-precomputed-activation-sums-candidate.json`
- `benchmarks/results/2026-07-27-gfx1151-laguna-q6-precomputed-activation-sums-production.json`
- `benchmarks/results/2026-07-27-gfx1151-laguna-q6-precomputed-activation-sums-candidate.json`
- `benchmarks/results/2026-07-27-gfx1151-laguna-q4-p8-pair-shared-prefetch-rejected.json`
- `benchmarks/results/2026-07-27-gfx1151-laguna-q4-raw-prefetch-p4-rejected.json`
- `benchmarks/results/2026-07-27-gfx1151-laguna-q4-paired-silu-pack-rejected.json`
- `benchmarks/results/2026-07-27-gfx1151-laguna-q4-split-fused-silu-pack-rejected.json`
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
