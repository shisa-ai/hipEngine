# hipEngine Topline Benchmarks

Last updated: **2026-07-29**

The current Laguna arithmetic-prefill production packet is
[`2026-07-27-gfx1151-laguna-attention-packed-query-producer-production.json`](results/2026-07-27-gfx1151-laguna-attention-packed-query-producer-production.json).
Complete dense-initial M128 tiles beginning at positions 128/256/384 now widen
the resident BF16 K/V cache exactly, produce only the F32 query tile directly
in head-major order, run one zero-workspace eight-way QK and one PV hipBLASLt
contraction into head-major output, then assign
each causal F32 score row to one local32 wave with no LDS or workgroup
barriers. Start 0, partial, wrapped, explicitly evicted, verifier, decode,
unsupported-head, and context-above-512 paths retain the established
`KVLiveSpans` kernels. The exact planar-Q6 selected-down body pipelines both
the next K32 weight record and the next compact Q8 activation half-row while
current integer-WMMA fragments execute. The preceding gate/up P8 publication
measured **643.554/573.066/466.290 tok/s**, improving its prior production
**0.695%/0.559%/0.433%**. Shape-qualified Q4 raw-nibble P8 prefetch carries
only the next K32 payload at producer chunks >=512 and preserves the previous
exact body below 512. The same payload-only pipeline now runs Q4 selected
down at M512+. The source-F16 K3072xN72 SWA-gate quality schedule keeps
hipBLASLt heuristic 2 through M128 and returns larger matrices to heuristic 4.
The Q6 D4 producer also records the two exact K16 activation sums in its
otherwise-unused metadata word, avoiding their reconstruction in 48
output-column workgroups. Clean selector-unset 512/1K/4K now measures
**654.249/579.699/468.608 tok/s**. The fused per-head RMSNorm/RoPE producer
now writes qualified query tiles directly in head-major order: query-pack
launches fall **144 / 4.907 ms -> 0**, total pp512 dispatches fall
**2,273 -> 2,129**, and producer-plus-pack falls
**20.530 -> 16.666 ms (-18.82%)**. Qualified library PV tiles now remain
head-major through the exact softplus gate: output-unpack launches fall
**144 -> 0**, total pp512 dispatches fall **2,417 -> 2,273**, and the
transpose-plus-gate boundary falls **11.240 -> 10.318 ms (-8.20%)** with no
resident or scratch growth. The aggregate medians are within
**-0.302%/-0.149%/-0.155%** of the preceding packet and are classified as
shared-APU variance; complete-state A/B is exact and the named sub-window is
positive. The Q4 D8 producer caches exact K16
integer sums in a bounded activation-only sidecar, cutting traced selected
gate/up **334.229 -> 330.720 ms (-1.050%)**. Relative to the preceding packet,
pp512 improves **0.399%**, 4K improves **0.085%**, and 1K is aggregate-flat
at **-0.036%**; same-process 1K A/B saves **4.428 ms (7/11 wins)**. Tokens
2930/95/7772, positions, full state, and lifecycle remain exact. Matched
full-state A/B improves
selected-down P8 **639.574 -> 643.166 tok/s (+0.562%, 7/7 wins)**, and
tracing cuts its 72 M512 Q4 calls **217.416 -> 212.090 ms (-2.450%)** at
local64/VGPR96/SGPR128/LDS1536B/scratch0. The absolute category gate remains
**0.049542582** maximum KL and **316/320** top-1; direct M512 quality is KL
**0.00407713** with top-1 2930. The clean 700 target now requires another
**51.148 ms** from the **782.577-ms** pp512 median wall. The preceding clean
trace cut the 23-call Q6 window **100.367 -> 99.459 ms (-0.905%)** at
local128/VGPR112/SGPR128/LDS5120B/scratch0.
[`row-schedule candidate`](results/2026-07-27-gfx1151-laguna-f16-quality-row-schedule-candidate.json).

Three exact gfx1100 structural transfers plus eleven gfx1151-native owners now
improve paired p512/d128 gfx1151 eager c=1 decode
**11.466687 -> 19.066920 tok/s (+66.281%)** in the retained same-source gate.
Native
head-RMSNorm + partial-RoPE + BF16 KV-write composites first reach
**11.485885 tok/s**, then the complete global/SWA/tile16 split-attention
**127/65/257** threshold bundle reaches **14.528110 tok/s (+26.487%
incremental)**. The D9 MoE-tail composite then replaces 47 exact
add/add/RMSNorm triples, removes **94 launches/token**, and reaches
**14.555265 tok/s (+0.177%)** between rollback medians
**14.529573/14.525706**. Finally the exact gfx1151 GQA3 SWA score owner reaches
**14.740486 tok/s**, **+1.214%** over its same-commit rollback
**14.563678**, while cutting the live-512 score producer **42.1-46.8%**.
The exact source-F16 one-barrier owner then moves
**14.758912 -> 14.800191 tok/s (+0.280%)** and saves
**0.189 ms/token** by removing one unnecessary reducer barrier without
changing the local256 grid or any arithmetic.
The exact fixed-K successor retains that complete grid and arithmetic while
compile-time-specializing K3072/K6144/K9216. It moves the same-session
one-barrier control **14.786076 -> 16.391201 tok/s (+10.856%)**, cuts the
weighted F16 family **30.952 -> 24.482 ms/token (-20.90%)**, and saves
**6.623 ms/token** of production wall.
Exact fused-GQA3 saturated SWA then stages 64 BF16 V slots in LDS and reuses
them across three query heads, reaching **18.026501 tok/s** in clean
production. The exact global GQA2 successor reuses the same staged V tile
across two query heads and reaches **18.230064 tok/s** in clean production,
**+1.129%** over the SWA-only packet. All 128 generated IDs, positions, and
allocation lifecycle state remain exact.
The staged SWA copy then widens from one BF16 value to one aligned 16-byte
transaction without changing attention arithmetic. Seven exact pairs move
**18.244607 -> 18.806305 tok/s (+3.079%)**, saving **1.637 ms/token**.
The padded global sibling does the same through live4000 and moves seven exact
pairs **18.794424 -> 19.066920 tok/s (+1.450%)**, saving another
**0.760 ms/token**.
The exact saturated-512 SWA successor retains the full
**72-workgroup/288-wave** reducer grid and scalar/FMA order while specializing
the natural 72Q/8KV/D128 ring. It moves the same-session GQA3 control
**16.386231 -> 16.833740 tok/s (+2.731%)**, saves **1.622 ms/token**, and cuts
the complete score+reduce leaf **25.13%**.
The exact global successor retains all 48 local256 reducer workgroups and
dynamic live/span arithmetic while specializing 48Q/8KV/D128/capacity-4096.
It moves same-session production **16.832097 -> 16.846689 tok/s (+0.087%)**,
saves **0.051 ms/token**, and improves complete attention **0.7-2.0%** at
production live 513/576/639.
The exact selected-MoE successor retains the local128 grid, resident T16
bytes, and every arithmetic operation while specializing Laguna's natural
c=1/top-10 gate/up and Q4/planar-Q6 down shapes. Actual-weight leaves improve
**1.63%/21.19%/10.33%**, and seven complete pairs move
**16.850003 -> 16.976046 tok/s (+0.748%)**, saving **0.441 ms/token**. Cached
tracing records **5,969/3,048/2,921** intended role calls with zero generic
selected-T16 fallback.
The exact tile8 successor keeps that natural T16 owner and arithmetic but
splits each 16-column gate/up tile across two 8-column workgroups. It halves
gate/up VGPR use **200 -> 96** with no weight/scratch growth. The actual-weight
leaf improves **5.35-7.13%**; seven exact production pairs move
**16.991621 -> 17.007001 tok/s (+0.091%)**, saving **0.053 ms/token** with
**7/7** wins. Cached tracing records all **5,969** tile8 calls and zero
tile16/generic fallback.
The exact fused-GQA2 SWA successor owns two adjacent query heads per
workgroup, fuses QK, ordered softmax, PV, gate, and stores, and eliminates the
global score plane plus one launch at saturated 512. Seven resident-model
pairs move **17.013184 -> 17.065241 tok/s (+0.306%)**, saving
**0.179 ms/token** with every candidate faster and every trajectory/state byte
unchanged. The cache-hot leaf is **2.96% slower**, but the full resident model
wins because each K vector is reread five rather than nine times per KV head.
Tracing records 40 local256 workgroups / 320 wave32s per SWA layer,
VGPR32/LDS6144/scratch0. Exact one-head fusion wins the hot leaf but regresses
resident production **1.038%** and is removed.
The exact fused one-head global successor keeps all **48 local256
workgroups / 384 wave32s** while fusing QK, the existing eight-wave softmax
association, PV, gate, and stores. It removes the score/physical round-trip
and one launch without increasing K reads. Complete leaves improve
**7.89-17.55%** at live 513/576/639; seven resident-model pairs move
**17.064962 -> 17.097044 tok/s (+0.188%)**, saving **0.110 ms/token**, with
every candidate faster and every trajectory/state exact. A two-head GQA2
version halves K reads but collapses the layer to 24 workgroups and regresses
production **0.126%**, so it is removed.
CPU-oracle, F32/BF16/`KVLiveSpans` byte,
reducer-bit-exact,
complete 128-token trajectory, ID/position, native trace, and lifecycle gates
all pass. Split scratch adds only **1.573 MB**; explicit disables retain both
unfused and serial chains. D9's paired pp512 is **651.504 tok/s**, within
normal variance of
retained **654.249 tok/s** short headline. The eager global-attention ABI still
caps decode cache capacity at 4,096, so the long-context publication below
remains prefill-only.
[`retained GQA3 score artifact`](results/2026-07-28-gfx1151-laguna-swa-gqa3-scores-retained.json) ·
[`retained fused-GQA2 SWA artifact`](results/2026-07-28-gfx1151-laguna-swa-fused-gqa2-retained.json) ·
[`retained fused one-head global artifact`](results/2026-07-28-gfx1151-laguna-global-fused-gqa1-retained.json) ·
[`retained selected tile8 artifact`](results/2026-07-28-gfx1151-laguna-selected-natural-tile8-retained.json) ·
[`retained selected natural-shape artifact`](results/2026-07-28-gfx1151-laguna-selected-natural-decode-retained.json) ·
[`retained global fixed-shape reducer artifact`](results/2026-07-28-gfx1151-laguna-global-fixedshape-reduce-retained.json) ·
[`retained fixed512 reducer artifact`](results/2026-07-28-gfx1151-laguna-swa-fixed512-reduce-retained.json) ·
[`retained F16 fixed-K artifact`](results/2026-07-28-gfx1151-laguna-f16-fixedk-retained.json) ·
[`retained F16 one-barrier artifact`](results/2026-07-28-gfx1151-laguna-f16-onebarrier-retained.json) ·
[`retained D9 artifact`](results/2026-07-28-gfx1151-laguna-d9-moe-tail-retained.json) ·
[`retained split-attention artifact`](results/2026-07-28-gfx1151-laguna-split-attention-retained.json) ·
[`retained head/KV artifact`](results/2026-07-28-gfx1151-laguna-head-kv-fusion-retained.json).

The clean post-D9 production census confirms exactly **869 kernels/token**
versus **963** before D9, with 47 D9 calls and **66.528 ms/token** median
device work. Source-F16 projections now dominate at
**30.981 ms/token (46.568%)**, followed by retained split attention at
**14.617 ms/token (21.972%)** and selected Q4 gate/up at
**8.549 ms/token (12.851%)**. The remaining merged gfx1100 wins are either
quant-incompatible with this Q4_K_M file, bypassed by the faster p512 split
path, or already rejected on gfx1100 with a negligible gfx1151 ceiling.
Further material decode work therefore starts with gfx1151-native source-F16
c=1 projection kernels.
[`transfer audit`](results/2026-07-28-gfx1151-laguna-gfx1100-decode-transfer-audit.json).

The user-requested 512/1K/4K/32K/64K/128K closure sweep now passes in one
128K-capacity resident session. Single-sample production throughput is
**622.009/579.152/470.270/214.698/131.997/72.323 tok/s** with exact final
positions through 131,071, deterministic next tokens, and zero tracked
allocations after teardown. The 1K/4K samples are within **-0.094%/+0.355%**
of their repeated production medians; pp512 is **-4.928%** when measured
inside the 128K-capacity session and remains a capacity-bucketing diagnostic.
The smooth long-context slowdown exposes the scalar online global-attention
fallback above 512 as the next long-context ceiling rather than a
matrix-chunk-specific cliff. This one-repeat attribution baseline does not
replace the repeated short-shape headline or the absolute quality packet.
[`six-shape artifact`](results/2026-07-27-gfx1151-laguna-prefill-six-shape-sweep.json).

LC-6 now closes that capacity diagnostic. With the retained M2,048 geometry,
a 128K-capacity session allocates **6.007 GiB** more than the 4K-capacity
control but changes 512/1K/4K only
**-0.425%/+0.654%/-0.004%**, to
**625.532/673.832/611.334 tok/s**. Exact tokens, positions, and lifecycle
pass, so the prior **-4.928%** singleton is not reproducible and neither lazy
KV nor a capacity bucket is promoted. Exact BF16 cache widening is only
**0.02-0.03%** of the measured complete attention transaction; Q8 KV and
unchanged library capabilities remain closed. The rejected M4,096/M8,192
diagnostic surface is removed and production stays M2,048.
[`LC-6 closure`](results/2026-07-28-gfx1151-laguna-lc6-capacity-secondary-closure.json).

The clean selector-unset long-context publication now passes
512/1K/4K/32K/64K/128K at
**614.031/666.901/609.879/365.481/247.408/149.308 tok/s** in one
128K-capacity session. Relative to the pre-campaign six-shape closure this is
**-1.283%/+15.151%/+29.687%/+70.230%/+87.435%/+106.446%** and saves
**934.451 seconds** at 128K. The 4K/64K/128K rows reproduce the retained LC-3
gates within **-0.313%/+0.354%/-0.251%**; all expected tokens, final
positions, finite state, and tracked-allocation teardown pass. The campaign
therefore more than doubles 128K without overfitting the short gate.
[`final long-context sweep`](results/2026-07-28-gfx1151-laguna-long-context-final-sweep.json).

A same-GGUF llama.cpp Vulkan one-pass baseline on the same Radeon 8060S
measures **341.999/333.502/280.349/126.624/65.584 tok/s** at
512/4K/16K/64K/128K. The coherent hipEngine LC-0 attack control measures
**466.482/307.953/132.831/72.139 tok/s** at 4K/16K/64K/128K, or
**+39.874%/+9.846%/+4.902%/+9.995%** over Vulkan. The earlier matched-capacity
pp512 closure remains **+81.874%** over Vulkan. Although Vulkan retains
**19.177%** of its
pp512 rate through 128K versus **11.054%** for the faster hipEngine headline,
the tail itself favors hipEngine: 64K-to-128K retention is
**54.309% versus 51.794%**, and wall growth is **3.683x versus 3.861x**.
Exact GGUF metadata confirms that all 48 Laguna blocks use softmax attention,
not the hybrid Qwen3.x 35B GDN mix: 12 are global and 36 use SWA-512. Exact
global+SWA QK+PV is **5.084/50.544/677.685/2,622.181 TFLOP** at
4K/16K/64K/128K; SWA/global falls
**105.46% -> 27.68% -> 7.00% -> 3.51%**. Global attention is therefore the
quadratic tail target, while the 4K gate protects a still-material SWA path.
Both complete-model walls remain far below that Laguna-specific roof, so
Vulkan parity is a floor.
[`hipEngine LC-0`](results/2026-07-27-gfx1151-laguna-lc0-attack-control.json) ·
[`Vulkan baseline`](results/2026-07-27-gfx1151-laguna-llamacpp-vulkan-long-context-baseline.json).

Cached-only LC-0 tracing is stable at **309.180/132.790 tok/s** for 16K/64K
and resolves the slowdown. Global attention grows
**22.670 -> 370.549 seconds (16.345x)**, SWA grows
**10.462 -> 43.499 seconds (4.158x)**, and complete wall minus attention grows
**19.860 -> 79.483 seconds (4.002x)** as context grows 4x. At 64K, global
alone is **75.08%** of complete wall and all attention is **83.90%**; kernel
span is within **19.7 ms** of wall. The post-512 scalar global qrow6 topology
issues about **131.76x** the ideal once-per-KV-head/tile K/V loads across 22
row groups and six GQA heads; SWA qrow4 issues **288x** across 32 groups and
nine heads. These include cache hits and are not physical DRAM counters.
[`LC-0 attribution`](results/2026-07-27-gfx1151-laguna-lc0-long-context-attribution.json).

The first LC-1 leaf establishes an important negative boundary. An exact
single-query-head Q16xK64 block-streamed body reached
local128/VGPR248/LDS33,792B and ran **4.96x-5.33x slower** than qrow6 from
context 512 through 64K. The candidate is removed. Its output/resource gate
is preserved, while the ephemeral leaf samples are explicitly non-promotable
because their raw command was not retained. LC-1 and LC-2 are consequently
co-designed around staging once for all six global GQA query heads rather
than spending LDS and barriers on single-head reuse.
[`LC-1 rejection`](results/2026-07-27-gfx1151-laguna-lc1-single-head-qtile16-k64-rejected.json).

Sharing the tile across all six global GQA heads does not rescue the scalar
design. Exact local192 K64 reaches only **0.244x-0.264x** qrow6 throughput;
K32 with two-workgroup launch bounds lowers VGPR **248 -> 152** and LDS
**33,792 -> 16,896 B**, yet still reaches only **0.585x-0.601x** from context
512 through 64K. Both candidates are removed. The result changes LC-1's
premise: request amplification was mostly cache-served, while serial
dot/exp/PV arithmetic remains the ceiling. The next screen is a bounded F32
hipBLASLt ceiling followed by tensorized block-streamed QK/PV.
[`GQA6 rejection`](results/2026-07-27-gfx1151-laguna-lc1-gqa6-scalar-staging-rejected.json).

The tuned packed-F32 hipBLASLt ceiling is the first positive LC-1
architecture. Inclusive BF16-cache widening, query/output transposes, QK,
wave-row causal softmax, and PV improve qrow6
**2.163x/1.406x/1.263x/1.305x/1.250x** at
512/4K/16K/64K/128K, with maximum absolute output error
**4.622e-8**. Untuned algorithm 0 loses at long context; screening all 32
zero-workspace heuristics selects QK28 from 16K onward and is essential. This
is a ceiling rather than production: F32 K/V plus score scratch costs
**2.151 GB** at 64K and **4.298 GB** at 128K. A global-only complete-model
candidate is the next gate.
[`long F32 ceiling`](results/2026-07-27-gfx1151-laguna-lc1-long-f32-hipblaslt-ceiling.json).

That ceiling first promoted a production-qualified global-only owner above
4K. A broad start-512 policy failed paired 4K at **0.968x**, so the qualified
shape policy leaves all work through 4K unchanged. The transitional owner
improved 16K **7.929%**, 64K **16.936%**, and mandatory 128K
**72.139 -> 88.073 tok/s (+22.088%)**, but materialized **4.298 GB** of F32
cache/score scratch at 128K.
[`long F32 production`](results/2026-07-27-gfx1151-laguna-lc1-long-f32-hipblaslt-production.json).

The retained successor is exact bounded-state tensorized attention. It widens
only one 4,096-key block, runs packed F32 QK/online tile softmax/PV, and carries
the query-row maximum, denominator, and output numerator across blocks.
Complete-model 4K/16K/64K screens preserve or improve the full-score owner by
**0.121%/0.614%/7.516%**. Mandatory 128K improves
**88.073 -> 99.100 tok/s (+12.521%)** and
**1,488.225 -> 1,322.622 seconds**, while scratch falls
**4,298,113,024 -> 143,753,216 bytes (-96.655%)**. This is **37.374%** above
the original hipEngine LC-0 control and **51.104%** above the same-GGUF Vulkan
128K row. Token 22746, position 131071, finite output, and complete allocation
recovery pass. The next long-context stage applies tensorized K/V reuse to the
rolling 512-token SWA ring, then widens query chunks.
[`bounded 4K-block production`](results/2026-07-27-gfx1151-laguna-lc1-block4096-hipblaslt-production.json).

LC-2 now tensorizes the rolling SWA half as well. After the 512-token ring
fills, each qualified M128 tile gathers 511 historical BF16 rows plus 128
current BF16-rounded rows into a fixed 639-key union, then runs packed F32 QK,
a row-shifted 512-key wave softmax, and packed F32 PV. The leaf improves
**3.313 -> 0.684 ms (4.846x)** with maximum absolute error **3.818e-8** and
**33.6 MB** fixed scratch. Complete-model gates improve 1K/4K/16K/64K by
**15.694%/23.418%/17.341%/8.072%**. Mandatory 128K improves
**99.100 -> 103.520 tok/s (+4.460%)**, saves **56.474 seconds**, preserves
token 22746/position 131071, and releases every tracked allocation. The
retained 128K route is now **43.501%** above original LC-0 and **57.844%**
above same-GGUF Vulkan.
[`rolling-SWA production`](results/2026-07-28-gfx1151-laguna-lc2-swa-hipblaslt-production.json).

LC-3 now reuses every exact 4K global K/V block across a complete M2,048
matrix chunk while preserving M128 SWA. Exhaustive global screens reduce
inclusive cost per row **43.309 -> 26.382 us (-39.09%)** from M128 to M2,048;
the corresponding SWA screen is rejected because per-row cost rises
**5.370 -> 21.124 us**. Complete 4K/16K/64K improves
**5.615%/20.473%/39.112%**. Mandatory 128K improves
**103.520 -> 149.684 tok/s (+44.594%)**, cuts wall
**1,266.148 -> 875.657 seconds**, preserves token 22746/position 131071, and
releases every tracked allocation. This is **107.494%** above original LC-0
and **128.233%** above same-GGUF Vulkan at 128K.
[`global-M2048 production`](results/2026-07-28-gfx1151-laguna-lc3-global-m2048-production.json).

LC-4 removes redundant span decoding from the metadata-qualified
dense-initial global-cache widen. Paired M2,048/4K timing improves the exact
sub-window **0.250249 -> 0.234780 ms (-6.181%)** and trace resources fall
VGPR24 -> VGPR16. Whole-model 4K/16K/64K samples are mixed within one-run
noise; mandatory 128K is neutral at **149.249 versus 149.684 tok/s
(-0.291%)**, with exact token/position/lifecycle. The canonical LC-3 128K
topline therefore remains unchanged while gfx1151 retains the exact
microsecond win and the generic `KVLiveSpans` fallback.
[`dense contiguous-cache production`](results/2026-07-28-gfx1151-laguna-lc4-dense-contiguous-cache.json).

LC-5 screens matrix chunks above M2,048 while preserving global-attention
M2,048 and SWA M128. M8,192 is directionally dominated and costs **5.268 GB**
extra scratch. M4,096 improves 16K/64K **1.457%/0.459%**, but mandatory 128K
regresses **149.684 -> 147.939 tok/s (-1.166%)**, loses **10.327 seconds**,
and adds **1.756 GB**. Both wider widths are rejected; the package default and
canonical topline remain M2,048/149.684 tok/s. The explicit diagnostics were
kept through LC-6 and are now removed after no bucketed-capacity premise
survived.
[`wide matrix rejection`](results/2026-07-28-gfx1151-laguna-lc5-wide-matrix-rejected.json).

LC-6 and the clean final sweep close the bounded long-context campaign.
Large-capacity short shapes vary only **-0.425%/+0.654%/-0.004%**, so lazy KV
is not promoted. Final 512/1K/4K/32K/64K/128K is
**614.031/666.901/609.879/365.481/247.408/149.308 tok/s**; the 128K result is
**106.446%** above the pre-campaign closure and **127.659%** above same-GGUF
Vulkan, with exact position/token/lifecycle.
[`capacity closure`](results/2026-07-28-gfx1151-laguna-lc6-capacity-secondary-closure.json) ·
[`final sweep`](results/2026-07-28-gfx1151-laguna-long-context-final-sweep.json).

Payload-only P8 has also passed admission for the Q4 selected-down
64x32/local64 body at producer rows >=512. Three traced pp512 arms cut its 72
M512 launches **217.416 -> 212.090 ms (-2.450%)** at
local64/VGPR96/LDS1536B/scratch0, while seven full-state pairs improve
**639.574 -> 643.166 tok/s (+0.562%, 7/7 wins)** with identical logits,
hidden states, KV, cursor, and token. The clean selector-unset publication is
complete at **643.141/573.717/466.913 tok/s**.
[`candidate`](results/2026-07-27-gfx1151-laguna-q4-down-raw-prefetch-p8-candidate.json).

The 256-thread Q4 row64 gate/up screen is exact but rejected and fully
removed. Keeping 32 accumulators per lane avoids the earlier row64 register
failure, but natural routing cuts M256/M512 tiles only **5.44%/16.84%**.
Cooperative shared-weight reconstruction regresses the actual layer-1 leaf
**116.98%/103.34%**; retaining direct per-column decode still regresses
**28.31%/19.29%**. Production remains **632.618 tok/s**.
[`artifact`](results/2026-07-26-gfx1151-laguna-q4-row64-local256-rejected.json).

The mixed short-tail Q4 gate/up schedule is rejected and fully removed.
Replacing only eligible final `32 + remainder` pairs with 40/48-row tiles
would reduce the natural-M512 47-layer grid **14,034 -> 12,788 (-8.88%)**,
but the combined candidate regresses actual-weight M256/M512
**4.3543 -> 4.6782 ms (+7.44%)** and **6.6991 -> 7.0457 ms (+5.17%)**.
The isolated row40 and row48 variants also lose at both shapes. The fixture is
BF16-bit exact and actual-weight checksums agree; all candidate surfaces are
removed and production remains **629.101 tok/s**.
[`artifact`](results/2026-07-26-gfx1151-laguna-q4-mixed-tail-rows-rejected.json).

The byte-neutral Q4 qmicro passes exact c1/c2/c4/c8 decode, but its
actual-weight natural-M512 selected-prefill gate is rejected and fully
removed. Direct per-column packed metadata regresses resident T16
**9.402044 -> 9.570781 ms (+1.795%)**; wave broadcast and a quartet-owned LDS
writer regress **9.539%/5.563%**. All three candidates preserve every BF16
bit. Only the host oracle and separately retained exact-decode primitive
remain; production stays **551.459 tok/s**.
[`artifact`](results/2026-07-26-gfx1151-laguna-q4-k-qmicro-prefill-rejected.json).

Intermediate 40/48-row Q4 gate/up tiles are also rejected and fully removed.
They reduce the frozen 47-layer route grid **14,034 -> 12,866/12,189
(-8.32%/-13.15%)**, but the exact actual-weight M512 leaves regress production
**6.851842 -> 7.016511 ms (+2.40%)** and
**6.824466 -> 6.941421 ms (+1.71%)**. The extra live accumulators cost more
than the avoided weight rereads; production stays **551.459 tok/s**.
[`artifact`](results/2026-07-26-gfx1151-laguna-q4-gate-rows40-48-rejected.json).

The exact qrow3 attention interpolation is rejected and fully removed. It
beats cached-metadata qrow4 on global tiles but loses at every SWA position;
the weighted pp512 leaf regresses **13.3577 -> 13.7874 ms (+3.22%)** and loses
**7.31%** to the qualified production qrow6 policy. At global start 0 it only
ties the actual non-metadata production body (**0.18634 vs 0.18580 ms**).
Production stays **551.459 tok/s**.
[`artifact`](results/2026-07-26-gfx1151-laguna-attention-qrow3-rejected.json).

The exact Q4 gate/up row-tile-fast grid is also rejected and fully removed.
Swapping only the launch axes preserves the production 128x32/local128 body
and every BF16 output bit, but the actual-weight natural-M512 leaf regresses
**6.908966 -> 6.921503 ms (+0.181%)**. Axis order alone does not create useful
cross-workgroup weight reuse; production stays **551.459 tok/s**.
[`artifact`](results/2026-07-26-gfx1151-laguna-q4-gate-rowfast-grid-rejected.json).

Source-F16 Q/K/V grouping is likewise closed. A single row-major combined
contraction is F32-bit exact but models only **2.891 ms** total pp512 saving
before splitting `[M,Q+K+V]` into the three contiguous production outputs.
The layout-preserving hipBLASLt `GroupedGemm` alternative exposes zero
algorithms for the full QKV problem on gfx1151 at both zero and 64-MiB
workspace. All temporary candidate surfaces were removed; production stays
**551.459 tok/s**.
[`artifact`](results/2026-07-26-gfx1151-laguna-f16-qkv-grouping-rejected.json).

The exact dense-initial attention path is retained in production.
On complete pre-wrap initial tiles, logical and absolute token positions are
identical and no slot is evicted, so the separately registered kernels keep
the full `KVLiveSpans` ABI and physical base-offset mapping while removing
per-token position/eviction reads. All outputs are F32-bit exact. The
qualified global-qrow4/qrow6 plus SWA-qrow4 policy improves
**12.8348 -> 11.8695 ms (1.0813x)** per four-layer pattern, modeling
**11.584 ms** pp512 saving. Strict runtime qualification is now the gfx1151
default with explicit rollback: seven matched complete-state pairs improve
cached metadata **552.144 -> 559.539 tok/s (+1.339%)**, save **12.255 ms** at
the medians, and preserve every output/state digest. Clean selector-unset
publication reaches **559.290/523.090/439.044 tok/s** at 512/1K/4K; tracing
cuts attention to **141.846 ms** with the intended launch mix.
[`production artifact`](results/2026-07-26-gfx1151-laguna-attention-dense-initial-production.json) ·
[`default artifact`](results/2026-07-26-gfx1151-laguna-attention-dense-initial-default.json) ·
[`leaf artifact`](results/2026-07-26-gfx1151-laguna-attention-dense-initial-candidate.json).

The exact production path temporarily writes packed gate/up BF16 into the
larger selected-down allocation, then folds the standalone sparse SiLU into
the range-safe down pack while explicitly preserving the BF16 boundary. Seven
complete-state pp512 pairs are exact and win **7/7**; paired geometric
throughput improves **0.651%**. Clean tracing removes another **47 launches**,
names exactly 47 fused packs at local128/VGPR16/LDS512B/scratch0, and records
no standalone selected-SiLU launches.
[`production artifact`](results/2026-07-26-gfx1151-laguna-fused-silu-pack-production.json) ·
[`candidate artifact`](results/2026-07-26-gfx1151-laguna-fused-silu-pack-candidate.json).

The M2048 default uses **1,755,275,296 bytes** of row/MoE scratch within the
existing 2-GiB admission floor. A 640-row pending transaction matches five
separately committed M128 transactions byte-for-byte across SWA wraps; wide
transactions require resident row-position views and every physical
attention/write slice remains bounded.
[`M2048 scheduling artifact`](results/2026-07-26-gfx1151-laguna-m2048-production.json).

The first post-546 selected-down screen is closed and fully removed. An exact
64-column x 128-row/local256 Q6 qmicro body collapses actual >=65-row expert
tiles **32 -> 17**, but saves only **0.017 ms** before its required second
metadata schedule and launch. The >=129-row tail instead regresses
**0.673548 -> 0.687981 ms (+2.14%)** at
local256/VGPR88/LDS8704B/scratch0. The selected-down body remains unchanged.
[`artifact`](results/2026-07-26-gfx1151-laguna-q6-down-rows128-heavy-rejected.json).

Complete M128 tiles now append current F32 K/V through the existing BF16 cache
writer before attention, then read prior and current K/V through one cached
source. Global storage is overwrite-safe; SWA uses the ordering only before
its first 512-slot ring wrap. Partial tiles, wrapped SWA, staged verifier
transactions, gfx1100, and unmeasured backends keep attend-then-append.
Primitive and full-model state are exact. Cached pp512 attribution cuts
global+SWA attention **219.709 -> 176.580 ms (-19.63%, 43.129 ms saved)**.
[`candidate artifact`](results/2026-07-26-gfx1151-laguna-attention-preappend-candidate.json).

The preceding selected-down split was Q6 **126.594 ms**, Q4 **72.358 ms**, and
activation packing **4.970 ms**. A Q6 K64 stage that halves synchronization
intervals is rejected and fully removed: it raises VGPR **88 -> 128**, doubles
LDS **5,632 -> 11,264 B**, regresses the traced Q6 family
**126.254 -> 144.607 ms (+14.54%)**, and regresses three-pair pp512
**528.123 -> 518.568 tok/s (-1.81%, 0/3 wins)**. Then-production remained
**526.451 tok/s**.
[`artifact`](results/2026-07-26-gfx1151-laguna-q6-down-k64-stage-rejected.json).

A Q6 paired-scale metadata screen is also closed and fully removed. Loading
each FP16 block multiplier once per output column instead of once per
column/scale-half leaves local128/VGPR88/LDS5632B unchanged. Five-pair pp512 is
noise at **529.210 -> 529.334 tok/s (+0.023%, 3/5 wins)**, while the traced Q6
family is slightly worse at **126.899 -> 126.947 ms (+0.038%)**. This rules
out scale metadata as the Q6 limiter; then-production remained **526.451 tok/s**.
[`artifact`](results/2026-07-26-gfx1151-laguna-q6-down-paired-scales-rejected.json).

The follow-up byte-neutral Q6 qmicro payload is retained in gfx1151 production.
It groups each four-column K4 quant quartet into one aligned 12-byte record while preserving
the **3,360-byte** T16 tile. Direct, grouped, and MMQ consumers are BF16-byte
exact. On the actual layer-1 660.6 MB Q6 tensor, natural-M512 selected prefill
improves **5.1564 -> 5.0714 ms (-1.65%)** and top-10 exact decode improves
**0.0910 -> 0.0846 ms (-6.99%)**. Clean pp512 improves
**526.451 -> 530.447 tok/s (+0.759%)**; the traced Q6 family falls
**126.594 -> 123.473 ms (-2.465%)**, and total selected down falls
**203.923 -> 200.510 ms (-1.673%)**.
[`production artifact`](results/2026-07-26-gfx1151-laguna-q6-qmicro-production.json) ·
[`leaf artifact`](results/2026-07-26-gfx1151-laguna-q6-qmicro-candidate.json).

The next exact attention policy is retained in gfx1151 production.
After preappend, cached-metadata qrow4 removes current-vs-cache bookkeeping
while preserving every F32 output bit. It selects every safe SWA M128 tile and
global tiles from position 128, retaining the established global position-0
body. Seven alternating one-owner full-model pairs improve
**533.507 -> 542.785 tok/s (+1.739%, 7/7 wins)** and save **16.405 ms**, with
identical logits, hidden states, KV, token/logit, and cursor across all fourteen
runs. Clean selector-unset production reaches **542.088 tok/s** median and
**542.022 tok/s** minimum; tracing cuts global+SWA attention
**175.802 -> 160.123 ms (-8.92%)**.
[`production artifact`](results/2026-07-26-gfx1151-laguna-attention-cached-meta-production.json) ·
[`default artifact`](results/2026-07-26-gfx1151-laguna-attention-cached-meta-default.json) ·
[`leaf artifact`](results/2026-07-26-gfx1151-laguna-attention-cached-meta-candidate.json).

The first post-500 replacement-layout screen is closed. Q4_K T128 improves the
exact actual layer-1 natural-M512 gate/up leaf **7.015 -> 6.157 ms (-12.23%,
1.139x)** with no byte growth versus current T16, but its best exact
sole-resident decode consumer regresses natural c1/c2/c4/c8
**6.79/6.10/6.36/6.86%**, missing the <=2% contract at every shape. All T128
implementation surfaces were removed; production remains T16 at
**503.349 tok/s**.
[`artifact`](results/2026-07-26-gfx1151-laguna-gate-t128-resident-rejected.json).

The byte-neutral Q4_K X16 control is also closed. Its exact one-pack decoder
beats X8 at every screened shape and reaches parity/wins at c2/c4/c8, but c1
regresses resident T16 **0.163258 -> 0.175753 ms (+7.654%)**, failing the
sole-resident prerequisite before any prefill or runtime work. The temporary
decoder was removed; the byte-lossless host roundtrip oracle remains for the
next T16-local-Q/four-column-metadata microtile.
[`artifact`](results/2026-07-26-gfx1151-laguna-q4-k-x16-decode-rejected.json).

That stronger byte-neutral Q4_K qmicro now passes exact decode. It preserves
the T16-local Q payload while packing 64 independently decodable metadata
quartets. Balanced c1/c2/c4/c8 improves resident T16
**4.929%/0.781%/3.691%/4.633%** with zero BF16 mismatches and reduces the
actual gate/up pair **931,135,488 -> 905,969,664 bytes (-2.778%)**. The
local128/VGPR192/LDS1536B/scratch0 decoder is retained as a primitive; no
materializer/runtime default changes before selected-prefill qualification.
[`artifact`](results/2026-07-26-gfx1151-laguna-q4-k-qmicro-exact-decode-retained.json).

The next exact source-F16 screen is also closed. Cached attribution shows that
hipBLASLt itself is **124.927 ms** of the **134.442 ms** family, leaving only
**9.516 ms** of cast/scale/restore glue. Fusing the four independent output
restores reduced launches **192 -> 48** but regressed their exact pp512
sequence **3.474 -> 6.114 ms (+76.0%)**; the candidate was removed.
[`artifact`](results/2026-07-26-gfx1151-laguna-f16-scale-restore-fused4-rejected.json).

A producer-qualified source-F16 candidate is retained. Every attention
RMSNorm output is bounded by **16.34623** from the actual 48 norm weights, so
the input row scale is provably one. A direct BF16-to-FP16 primitive plus
removal of four identity output restores improves the exact pp512 glue mix
**4.434 -> 0.767 ms (-82.70%, 5.780x)** with byte parity. Seven-pair full
pp512 improves **502.348 -> 505.887 tok/s (+0.704%)** with exact
logits/hidden/KV/cursor and all paired wins. The clean selector-unset refresh
retains it in production at **503.869 tok/s** median and **501.790 tok/s**
minimum while cached source-F16 attribution falls **134.442 -> 128.274 ms**.
[`artifact`](results/2026-07-26-gfx1151-laguna-f16-norm-direct-production.json).

A second source-F16 boundary candidate is retained. Per-layer Cauchy-Schwarz
over the actual norm/value/gate weights bounds the worst gated attention BF16
producer at **7,957.539**, leaving **4.116x** FP16 margin after the runtime's
separate 2x reserve. Removing the final row reduction/scale improves the exact
48-layer cast sequence **3.404 -> 2.680 ms (-21.27%)**. Seven-pair pp512 is
complete-state exact and moves **505.805 -> 506.284 tok/s (+0.095%)**; the
aggregate change is inside run variance, so the measured sub-window is the
retention evidence. Clean selector-unset publication retains the route at
**505.185 tok/s** median and **503.198 tok/s** minimum; 1K/4K also improve
**0.326%/0.259%**.
[`artifact`](results/2026-07-26-gfx1151-laguna-f16-output-range-direct-production.json).

The next exact gate/up load screen is closed. Staging each resident-T16 K32
payload with aligned 64-bit loads in 2 KB LDS adds no barrier and is byte-exact,
but regresses the actual layer-1 natural-M512 leaf **6.918 -> 6.990 ms
(+1.05%)**. All candidate surfaces were removed; then-current production stayed
**505.185 tok/s**.
[`artifact`](results/2026-07-26-gfx1151-laguna-gate-wave-lds-stage-rejected.json).

The synchronization follow-up is retained. Ping-ponging the 1.5 KB activation
tile removes one of two barriers per K32, keeps resident bytes and arithmetic
unchanged, and is BF16 byte-exact. The actual layer-1 natural-M512 inclusive
leaf improves **6.995 -> 6.907 ms (-1.258%, +1.274% throughput)**. The route is
now the gfx1151 package default after clean seven-pair pp512 improved direct
rollback **505.970 -> 507.405 tok/s (+0.284%)**, won **5/7** pairs, and kept
complete state exact. Its selector-unset publication is complete below.
[`artifact`](results/2026-07-26-gfx1151-laguna-gate-activation-doublebuf-candidate.json).
[`default artifact`](results/2026-07-26-gfx1151-laguna-gate-activation-doublebuf-default.json).
Clean selector-unset publication is now complete at **505.084 tok/s** median
and **504.984 tok/s** minimum. The unmatched checkpoint is flat within
**-0.020%** run variance versus the preceding **505.185 tok/s** packet, while
cached tracing cuts gate/up **318.559 -> 314.378 ms (-1.313%)** and observes
the intended local128/VGPR88/LDS3072B/scratch0 body 564 times across the
profile.
[`production artifact`](results/2026-07-26-gfx1151-laguna-gate-activation-doublebuf-production.json).

The analogous Q4 selected-down synchronization screen is closed and removed.
It is BF16 byte-exact and leaves Q6 unchanged, but matched seven-pair pp512
regresses **508.788 -> 508.023 tok/s (-0.150%, +1.515 ms)** and wins only
**2/7** pairs. Production remains **505.084 tok/s**.
[`artifact`](results/2026-07-26-gfx1151-laguna-q4-down-activation-doublebuf-rejected.json).

Latest retained hipEngine revisions in this scoreboard:
`b58461a5e` for exact next-K32 Q6 WMMA weight prefetch in gfx1151 Laguna
production,
`fe105632c` for exact padded-slot Q6 activation-stage elision in gfx1151
Laguna production,
`dcf29b1d5` for M2048 projection/MoE transactions with independent M128
attention/KV slices in gfx1151 Laguna production,
`53e3c2468` for exact qualified cached-metadata qrow4 attention in gfx1151
Laguna production,
`7b4f2ef82` for exact cached-only M128 qrow4 attention scheduling in gfx1151
Laguna production,
`647ac846c` for exact activation-double-buffer gate/up production,
`7ecd940b9` for exact static-range direct F16 boundaries in Laguna production,
`1bac6ead5` for the exact direct attention-norm cast in Laguna production,
`238eb28cd1c748c1755ac8871db4c0e140c3fee4` for exact eight-token
Laguna router-logit reuse,
`f91eccb5d16d8269199d4b1060df183c91dafad8` for exact stable parallel
Laguna MoE compaction,
`f9a39715be6a72cb550f058a4c89109e1e265d4b` for the exact 64-row Q6
selected-down body and tile64 metadata in Laguna production,
`3c1e5b452bb101b620c26e149e700277980f657b` for the exact Q4 pack8
64x16/64x32/32x32 shape policy in Laguna production,
`c4e2fbd1ddea8100a663754ceedbceaef174a38e` for scratch-free Q6 16x32
dense/shared WMMA in Laguna production,
`d39cbb5baacc7b44dac4e852c0580ee8b59a1c47` for direct per-column
wave decode in Laguna Q4 selected-down production,
`d7134100df6d211c36a13478f8c5fa84aeb77e57` for direct per-column
wave decode in Laguna Q4 selected gate/up production,
`7086f8cdd113fd2912acac0d129040b3bf84ef5f` for Q4-only selected-down
wave-column production,
`b44c8a5604be29664781201cf6d14ea9c17b7f64` for wave-column D8 Q4 gate/up
production,
`9ae1e4ea609520f476132d50794a74be58b5de5a` for the absolute-quality
hipBLASLt schedule and production-versus-all-exact gate,
`69cc0d369511b993c882df44083a81c680483dda` for bit-identical row-vector
Laguna Q4/Q6 selected-down staging,
`bd76e452dfe7d9713c413f5a2dc523af828d6dfc` for bit-identical row-vector
Laguna D8 gate/up staging,
`36b318ac930a400987cb5f8c4b2c8f8a0144e2ae` for source-qualified M128
Laguna SWA qrow4 production,
`3e1cb59934e6433e54ac0ada07052e2d13b3cdba` for the M128-qualified gfx1151
Laguna online-qrow4 attention production default,
`ab0a8ea3bf783ff033881d7c54a38205b35423e6` for the quality-admitted gfx1151
Laguna D8/D4 MMQ, row-scaled hipBLASLt, and Q4/Q6 WMMA production defaults,
`7b710c09ed7c402317990011d624b6010509eac2` for the clean fail-closed
production publication gate,
`84c50b205b865b54070f96ff9b6d2feea653c295` for the exact byte-neutral
gfx1151 Laguna Q4_K X8 MMQ32 live-row primitive,
`1d6566de3f6ec394d6a3e34e2f37e2a70250368c` for the quality-gated gfx1151
Laguna SWA qrow2 online-softmax prefill default,
`7c211f2412872dab76de5edd70ec155c7ca88f75` for the post-global-online Laguna
512/1K/4K all-family attribution,
`60a1e8f104993b5d50374170959659080151f4e1` for the quality-gated gfx1151
Laguna global qrow2 online-softmax prefill default,
`afdede4286bab27a691e8e137b16229d1baba194` for the post-qrow2 Laguna
512/1K/4K all-family attribution and global AOTriton screen,
`c0dfb324eba69f61b981af57c5af6815af92a6f5` for the exact context-qualified
Laguna SWA qrow2 prefill default,
`cdc43b36635cb23fbfcf674d7cd9698b65630a7a` for the post-matrix512 Laguna
512/1K/4K all-family attribution,
`af59b711e23ea3b9a6ae0f5ed172ef6e5a85a69e` for the exact gfx1151 Laguna
matrix512/attention128 prefill-chunk default,
`2ec20c8a06197a9dff9a5acd30e75d6fe52844b5` for the quality-gated gfx1151
Laguna compensated-WMMA SWA source-F16 prefill default,
`dae2afaad425ca3df32cd5b390bae6178b1e82dd` for its clean compensated-WMMA
production-shape screen,
`4f074642789b83eccda57ad70e3fd5a68dee1aaf` for the preceding direct-WMMA
production-shape screen,
`b7a0d7751a96cac2c0c93530f4f1119a39bd0ad2` for its preceding clean
rocBLAS/hipBLASLt matrix ceiling,
`773ab9033ea55445b254ddf47fa87cd5ff926adc` for the clean matched W7900
Laguna hipEngine-versus-llama.cpp-HIP ABBA harness and measurement packet,
`5a0463c71a198d1fbb1caf8ee632d29b337c5a94` for the exact W7900 Laguna
IQ3 ten-wave fused measurement/runtime packet promoted by this update,
`fe59e77d065c7bf88cfa1635d71cd64b0e415005` for exact Laguna native
scheduler ownership, cancellation acknowledgement, and the retained c1/c2
server policy packet,
`ac83c6e1f01358db65b8cb1c0fd94d6dd31a528f` for the exact gfx1151 Laguna
grouped routing/shared combine default,
`804e9484f3da0031628805f5bbef62a43badffaa` for exact bounded Laguna
stateful-session KV continuation,
`71f2af038cf5eea88f1997d178d815cfaad15681` for prefix-aware exact Laguna
stop-safe streaming,
`a95adcac82d8ae0b018fe1167b5108422afa47a9` for the full exact Laguna
Q4/Q6 grouped-small-M down category gate and gfx1151 default promotion,
`0081d150c08a95423f29fec8fd26779f53c8f730` for request-local exact Laguna
prompt preparation and preprocessing telemetry,
`8ae07d693b6f98d6c44aae90090df6c6d77e8d78` for exact gfx1151 Laguna S 2.1
resident-session pooling and setup telemetry,
`8f8e64ea88cc886bffe600430d091d71b1774e6f` for exact all-local32 W7900 Laguna S 2.1 UD-Q2_K_XL Q5/Q6 mixed projections,
`756a1dcd3bcf240bed9dd787edabc2851b458032` for the exact fixed-metadata W7900 Laguna S 2.1 UD-Q2_K_XL shared-Q5 BF16 pair,
`65f13a87720d4b0e999f5eec5d6fd57a82357841` for exact fixed-Q6 metadata inside the W7900 Laguna S 2.1 UD-Q2_K_XL mixed projection quads,
`dd3b9c646` for the preceding exact W7900 Laguna S 2.1 UD-Q2_K_XL mixed
attention-projection quads,
`b271f1fdc` for the exact W7900 Laguna S 2.1 UD-Q2_K_XL raw-Q5
fixed-metadata siblings,
`853516ecdd3a464b38013064a1d8ccacc20556c5` for the exact W7900 Laguna S 2.1
UD-Q2_K_XL IQ2 expanded-magnitude grid,
`367c1f622167653c733896e3a2a1f5972f9961c4` for exact W7900 Laguna S 2.1
UD-Q2_K_XL current-P4 head RMSNorm+RoPE+BF16-KV fusion,
`46539dedb8b84e4f7511f3320fa740e2f41092a6` for exact W7900 Laguna S 2.1
UD-Q2_K_XL P4.1 split-reducer+gate fusion,
`071331863` for the exact W7900 Laguna S 2.1 UD-Q2_K_XL P2 SWA tile16
score producer,
`ad99721ed6921d86c8cb89975433603881168c80` for its exact P2 split attention
producer/reducer parent,
`54a5751de19e00865754becee3588d041f8d4136` for the exact W7900 Laguna S 2.1
UD-Q2_K_XL P0 IQ3 wave4 route/output producer,
`c7fcf46f9` for the retained explicit-only exact W7900 Laguna S 2.1
UD-Q2_K_XL DFlash IQ3 selected-down tile4 path,
`338d3afca01aa884ff3a68e0175566bc51e5ceae` for the measured exact W7900
Laguna S 2.1 UD-Q2_K_XL raw-Q5 wave32x2 default,
`30cf6f0755ee53afc1c72e9106fbab887ea067bc` for its preceding measured exact W7900
Laguna S 2.1 UD-Q2_K_XL aggregate MoE-tail plus next-RMS default,
`51a437bc729eb5476a8bf71a31b2a9718e880263` for its preceding Q6 attention
pair default (`973382e68` implementation),
`22e6144ce032eda3b42a757faf792cea90997a67` for its preceding Q5 attention
query/gate pair default,
`35b1602e50c3234f7676dea5c62a802f99a67a8e` for its preceding Q5 shared
gate/up pair default,
`73a2583beecc0a92964e9885fc15a2b28802eddf` for its preceding token4 SWA
decode default,
`fe89c210c9129d51a893beaab8c419aa87250fd5` for its preceding IQ3
routing-weighted down default,
`ae20392bb8472a26ad67eba2a82679a83add8576` for its preceding IQ3 K1024
local128 default,
`89939a90b6efee417c8fb8e63946d35d0f09607f` for its preceding wave-uniform
IQ3 selected-down sub-window,
`fc08ca0ed07576c2fcfd632b9a9f51e0d5397d4e` for the exact W7900 Laguna S 2.1
UD-Q2_K_XL dense-decode default,
`6ba1ddec95e224c1cc337c69ac2c4ea611ff0472` for the first W7900 Laguna S 2.1
UD-Q2_K_XL B4 DFlash decode win,
`09cca232f49e73f68fd09d4ace8509fa3201681e` for the first W7900 Laguna S 2.1
UD-Q2_K_XL target-only AR baseline,
`b2618b725a39dc199b0009c23a0ec3d5a6342fa1` for the matched Poolside
llama.cpp Laguna S 2.1 128/512/1K/4K prefill-control harness,
`7ded0d5f42b107d3bf10f1d096f8a93ae194be9b` for the current Laguna S 2.1
128/512/1K/4K all-family prefill attribution,
`8f8baf9a100bc9598b633cb040193dcfcdb80ebe` for the current merged-main
Laguna S 2.1 B4 DFlash economics confirmation,
`c4ac3c60a47ce474ce5aa160d7c3ff8eda1009b5` for the exact explicit-only
Laguna S 2.1 B4 DFlash library/OpenAI route,
`871a22dda42cc612a3a77b2110ac0d1397b5426c` for the original post-prefill
Laguna S 2.1 B4 DFlash economics decision,
`bd8877fdf029a6c23880cc8fb4fb04401adf93a7` for the exact Laguna S 2.1
LPF-5 wave32 SWA full-model gate,
`9fb6bda7f810e3b6ad18603858b72a88bb75ad72` for the Laguna S 2.1 LPF-5
512/1K/4K long-context attribution profile,
`8ceb7d3a0068822b23ba6729cd9c91cacb701309` for the exact Laguna S 2.1
LPF-4 128-row chunk-policy and canonical target-AR gate,
`6b14c2da60bd51d56373e31fe9e16e15f4e969d9` for the exact Laguna S 2.1
LPF-1 tiled source-F16 prefill and canonical target-AR gate,
`dbfeecf83363023d3ac9d72736c68da632af0726` for the Laguna S 2.1 LPF-0
prefill-only shape trace and real-routing replay,
`b83d9aaae0b4afd59508d081f65b2da7c473e59a` for the exact pre-LPF-1
Laguna S 2.1 full-suite DFlash diagnostic and fixed-horizon state gate,
`ee1649e3fa372bd115ae7afed9aa2a0e81932afc` for the exact Laguna S 2.1
canonical target-AR category benchmark and bulk-prefill promotion,
`e99a30cb9183ce342f5a30fa4f774b14dc4c0677` for the Laguna S 2.1
source-bound repacked cache and cold-start reduction,
`8a8ef4816d442b3b8766507c1eac1ae796e882eb` for bounded cold-cohort
admission priority and the clean gfx1151 C8 server-ramp/SLO closure,
`44c76674c2693f7dfc994b40b4cfc3880abbbeac` for the repeated W7900
coding-agent A1 baseline,
`414d6d9e0fc8a1333bbece4db851271f031936bf` for the clean W7900
coding-agent A5 pressure/soak closure and corrected bounded stream lifecycle,
`878d07a9ba8d3cd24cf44bd88d359be7b4921c2e` for the clean W7900
coding-agent A6 broad external-oracle quality packet,
`960d4a98d623c64073e40ffd061cbc25ee38a0fc` for the matched gfx1151
PARO/GGUF/llama-server concurrency refresh and corrected PARO request-lifetime
route validation,
`bb9fc742eb2929af38562de4375fcc64e85d4b17` for the exact parallel W7900 PARO
MTP router top8 gate,
`43abe82ee7094601a0fb3af0b0b71afe0e787e09` for the W7900 PARO N4+
selected-state commit/cursor ownership gate,
`64f80f83e241013f05ac70b22806e3b523826b4f` (clean merged gate
`4e9703be8790a05cae6c99685801a1abc5621350`) for the W7900 PARO N4+
bound-control replay and duplicate-sync removal plus the follow-up provider-residual attribution,
`f3600c248ab8368a2cf808c65e8ace283fc6328c` for exact physical-C8 indexed-GDN
shared state cache 24 and clean direct C8 retention,
`4bec6b20c96a233fa56e0fa732dd6f6ab503a05c` for exact physical-C8
paged-attention value vector 2 and clean direct C8 retention,
`a960e28a3bf129f2f49f30a40b3ca37ef0493cad` for exact physical-C8
paged-attention shared token offsets and clean direct C8 retention,
`11910c273684636855ee3160d70b5bcb6415bdb0` for exact physical-C8
Q6T16 lm-head 5+3 rowtile partition and clean direct C8 retention,
`b03a828fe5e6ddccdf702ff185184e2d09014e21` for exact physical-C8
Q8T16 qkv+gate rowtile4/col8 and clean direct C8 retention,
`c843f836b45d5a5c12e05bae8c62b2cf17af2d60` for exact physical-C8
Q6T16 selected-down expert pair reuse and clean direct C8 retention,
`4cb305434aa11f827bbbd1f021f03d75fbe7398e` for exact physical-C8
Q5T16 selected-down expert pair reuse and clean direct C8 retention,
`1a753e2e99820ec421b6e16f192c3fa70b341e4d` for exact physical-C8
Q4T16 selected-expert pair reuse and clean direct C8 retention,
`bbe328cb828f6708614617f64dc1ed87963d773a` for the clean optimized
physical-C8 gfx1151 GGUF server packet (pressure-gated prefill `d0f45bc9`,
resident telemetry reuse `96598b39`, and terminal-state discard `bbe328cb`),
`2946787417d299ec4b30d1940d008128d4a4442d` for the prior physical-C8
server tree (clean BF16 measurement `8e61b9f9`, resident graph `33e22d85`,
physical-C8 routing `c2492bce`, and corrected harness `b2448f81`) and the
clean corrected-window mirrored-INT8 packet
(implementation `fb926d8e`, quality harness `24d4ad42`),
`184bc4e81ff0aa34961378590ff6afd705c1f050` for the uncontended W7900 PARO
N4 strict economics/on-off bracket and `7afc0d5e` for its cached final-child
HIP API/kernel profiler harness,
`5ef02aff4f97563b867f0ca3945b3a6cf050edf7` for the corrected W7900 PARO
N4 strict-verifier/category correctness packet (runtime fix `b3599958`),
`0d276fdfa5681f8255e11cad1bd9de9a514a8b71` for the gfx1151 packed-owner
workspace lifecycle and high-concurrency GTT repair,
`1163e1bbd28c7cb31325d695388fc05fa4e7d7ab` for the gfx1151 reusable
NativeSpecCycle target graph and N3 public complete-cycle transfer,
`2395ad3319ba6b96fe1a066171c7f77b712ba452` for the W7900 N3P reusable
NextN proposal-submission ownership diagnostic,
`69b7080850dc1f5c38f577e17468331349cacbd2` for the W7900 N3 complete
GGUF MTP cycle/public-adapter ownership diagnostic,
`8893e06a3d5ec5a91597163a12ce4325f4bf5e2e` for the W7900 N2
native accept/selected-state ownership diagnostic,
`0d7b86e76edc46940417e06f5b9e634b907d9fe2` for the faster retained W7900
reusable B1/B2 native-target `llama-compat` MTP route,
`7ab8eb3b60de772f61b1b2d55785e7872586abcd` for real GGUF graph replay
accounting and `b49bc0ef8dd74678e7477541f0e455cf73d11b67` for the Prometheus
surface in the correctness-retained, live-observable gfx1100 GGUF OpenAI
continuous-membership closure (D4 lifecycle source `f03957cc`),
`666a72dbac0af1d27661860e7f09facb77dd1299` for the focused post-sweep gfx1100
GGUF router convergence gates, `d59d7cf0c3532f4fd7a5601a26805c85698f1db8`
for the retained gfx1100 GGUF direct native-c4 graph-scaling closure (graph
runtime `6f7851f3`, clean profiler `a05c560b`, and category provenance
`799d29b9`), `52b0db25a20607f51e08abc89c43d200d2fe0ea5` for the retained
native-c8 profiler/scaling packet (correctness runtime `bbe6deb0`),
`77279adfb42c1f106c0541b34239d083517136c5` for the retained real OpenAI
gfx1100 arbitrary-C server packet (E3 state gate `1dc7076f`, compaction
`be04fa31`), `fcb65c470fd830918255b49f554fe70b08399272` for the retained
explicit gfx1100 PARO selected-batch c2 step,
`2edbb2ee3ca74d7757500b5eafe737d43748489c` for the current gfx1151 IOMMU-off
model/MTP refresh, and `d01952211ebafba2bd9391174369f48450b254f9` for the
profiler-, scaling-, and live-loop-retained gfx1151 direct native-c2/c4/c8
transfer (correctness packet `ab6c6d60`), and
`c6e5443d86e873772d7432dc136170ef5a99916a` for retained gfx1151 E3
arbitrary-C/explicit-compaction correctness, and
`71e2ea9a355fee8502b12d2b0d2210ec50ce6859` for retained real OpenAI
gfx1151 arbitrary-C server scaling, `ef46ee8cb100c291495c233024aa9ca492aea5b6`
for the current-main GGUF direct/server/profiler recertification and physical-c2
exactness repair, `190f208c758aafc72e4c2d7a66c2b91386445db7` for retained
occupancy-adaptive gfx1151 GGUF serving,
`7bb3669b9f585a695f8c6ea8d897d3a726268733` for retained exact gfx1151
physical-C8 Q8T16 pair row amortization (`e99b228b` singleton-indexed GDN
baseline),
`ecaf14d53799d487efdb27e07eb4342e69b3e574` for the accepted gfx1151 GGUF
production load/SLO gate and scoped `fair:256` package default,
`7871c0886f6c674c77ca279bcd3dbee6e7717e71` for the accepted sampled GGUF
OpenAI API-path closure, `f4c826e2` for byte-exact active-current GGUF prefix
reuse, `05dda75b` for its clean paired economics packet, and `13d8beaf` /
`74251bdb` for cache-owned completed-source snapshots and their real correctness
gate, plus `f0a63059` for completed-source economics, and
`8c8cc15ef657cd966ce16793c29ed2eba8533a14` for retained explicit gfx1151
PARO direct native-c2/c4/c8 (c2 transfer parent `778c7a70`), plus clean G5
blocking/SSE measurements `c0e3318c` / `ee8b417e` for the package-default
resident c2/c4/c8 promotion, and clean measured `7c243ed1` plus focused harness
repair `c142730a` for the retained gfx1151 GGUF 64K concurrency / device-KV
pressure closure, and `8405c46723efb136df279e03500bbe2425e40384` for the
tracked-clean matched GGUF C1/C2/C4/C8/C13 OpenAI packet. The gfx1151 GGUF
refresh is retained through 64K; repeated 128K is explicitly blocked by the
residual gfx11 scheduler lifecycle failure rather than carrying a stale number.

This file is the source of truth for repository-level performance tables. It
records which snapshots are eligible for use, the exact protocol behind each
table, the measured source revision and build environment, and the command used
to refresh it. [`README.md`](../README.md) contains copies of the marked export
blocks below; update them with:

```bash
python3 scripts/sync_benchmark_readme.py --write
python3 scripts/sync_benchmark_readme.py --check
```

Machine-readable evidence is under [`benchmarks/results/`](results/). Promotion
requirements are defined in [`docs/BENCHMARK.md`](../docs/BENCHMARK.md).
Reverse-chronological changes are in [`benchmarks/CHANGELOG.md`](CHANGELOG.md).
The previous experiment notebook is preserved in
[`benchmarks/HISTORY.md`](HISTORY.md).

## Status Rules

| Status | Meaning | May appear as a repository topline? |
| --- | --- | --- |
| **Retained** | The artifact passes the protocol's correctness, provenance, and performance gates. | Yes, for the named protocol only. |
| **Diagnostic** | The run is useful but has a known comparability, correctness, repetition, or provenance limitation. | No. Link it from the separate diagnostic section; do not place its numbers in a current table. |
| **Stale** | A measured path, dependency, or required evidence contract changed after the run. | No. It may remain as the last dated snapshot while a refresh is pending. |
| **Blocked** | No row satisfies the protocol. | No numeric topline. Record the blocker and the next command. |

`Latest` means the newest artifact for one exact protocol tuple. A newer
diagnostic does not replace a retained row. A row is identified by:

```text
platform + GPU + model fingerprint + quant + KV type + backend +
workload + concurrency + sampling/speculative policy + timing scope
```

Documentation-only commits do not make a row stale. Changes to a measured
runtime path, model, quant, KV policy, compiler/runtime, benchmark timing scope,
correctness gate, or comparison engine do.

New server, retained PARO, GGUF, and micro artifacts must embed a valid
`hipengine_artifact_provenance` v1 block. The canonical schema is
[`schemas/artifact-provenance.schema.json`](schemas/artifact-provenance.schema.json).
For retained model-performance rows, the resolved backend must be concrete,
the selected target/device must be recorded, the model fingerprint must refer
to existing content, and staged/unstaged/untracked dirtiness must all be false.
Legacy provenance fields remain useful diagnostics but do not satisfy this
contract for a new row.

New non-streaming hipEngine server rows also require a complete
`hipengine.generation_shape` v1 rollup. Route caps retain their
`queue_requests` scope; queue request/prompt counts, actual backend calls and
widths, and verifier rows remain separate and are deduplicated by queue-group
ID. Client concurrency is never substituted for backend or verifier width.

Direct/server comparisons additionally require the
`hipengine_exact_token_oracle` v1 gate from
[`scripts/exact_token_generation.py`](../scripts/exact_token_generation.py).
The committed 512-ID fixture feeds both PARO/GGUF direct generation and
`/v1/completions` without detokenization. HTTP input hashes/counts, exact usage,
and every generated ID must match the direct oracle. The formal contract is
[`schemas/exact-token-oracle.schema.json`](schemas/exact-token-oracle.schema.json).
The 2026-07-11 gfx1151 PARO 512/128 correctness gate passed; it is not a
throughput row and changes no topline.

Unified direct/server reports use `hipengine_benchmark_matrix` v1 from
[`scripts/benchmark_matrix.py`](../scripts/benchmark_matrix.py). The matrix
recomputes exact-ID denominators, enforces timing ownership, preserves backend
and verifier shapes, and attaches memory/profiler summaries. Its schemas are
[`benchmark-matrix.schema.json`](schemas/benchmark-matrix.schema.json) and
[`benchmark-matrix-manifest.schema.json`](schemas/benchmark-matrix-manifest.schema.json).
The committed SOL-E5 PARO manifest is diagnostic: direct-call wall includes
model/session setup while HTTP is client-E2E, so the report intentionally emits
no direct/server speed ratio. A retained matrix requires the normal clean,
repeated, scoped-timing, memory, profiler, correctness, and shape gates.

The accepted gfx1151 GGUF eager correctness gate is
[`2026-07-11-sol-g1-gfx1151-gguf-eager-p512-d4.json`](results/2026-07-11-sol-g1-gfx1151-gguf-eager-p512-d4.json).
For the exact Q4_K_M file and `[9707] * 512` prompt, llama.cpp and hipEngine's
bulk-prefill/eager route both emit five `9707` IDs. Four teacher-forced eager
transitions are byte-exact against fresh serial-prefix recomputation for all 40
layer outputs, 30 Conv/GDN state pairs, and 10 live K/V layer pairs. This
classifies the repeated stream as valid model behavior on gfx1151; it is a
correctness artifact with `performance_claim=false`, not a throughput row.

The accepted SOL-G2 fused/chain prefill gate is
[`2026-07-11-sol-g2-gfx1151-gdn-prefill-exact-matrix.json`](results/2026-07-11-sol-g2-gfx1151-gdn-prefill-exact-matrix.json).
At committed revision `332f01f8`, the GGUF-only raw-Q/K-plus-scale split chain
matches fused production prefill in all 6/6 clean gfx1151 cases: the exact
17-token greeting, repeated-token 512, the 1024/1025 segment threshold, and the
4095/4096 four-chunk boundary. Sampled tokens, FP32 hidden seeds, and all 30
resident Conv/GDN state pairs are byte-exact; greeting and 512 also match every
captured layer output. The earlier
[`104fad87` prefix artifact](results/2026-07-11-sol-g2-gfx1151-gdn-prefill-greeting-prefix.json)
preserves the normalized-Q/K layer-0 recurrent RED. Both artifacts set
`performance_claim=false`; the repeated, interleaved G3 result below selects
the default.

That G3 protocol is now complete in
[`2026-07-11-sol-g3-gfx1151-gdn-prefill-interleaved-ab.json`](results/2026-07-11-sol-g3-gfx1151-gdn-prefill-interleaved-ab.json).
From a clean detached `ad773eba` worktree with one warmup and four balanced
same-session repetitions per mode/context, the exact chain is slower than fused:
`1248.436` versus `1186.842 ms` at 512 (**+5.19% wall**) and `10870.022` versus
`10187.300 ms` at 4096 (**+6.70% wall**). Every timed pair returns exact token
`9707`, and the artifact links the accepted state matrix by SHA-256. This is a
valid retained negative result (`performance_claim=true`): fused remains the
default, and the exact split remains a diagnostic/unfused fallback.

The follow-on GPF-2B candidate performance gate is retained in
[`2026-07-13-gfx1151-gguf-prefill-gpf2-balanced-ab.json`](results/2026-07-13-gfx1151-gguf-prefill-gpf2-balanced-ab.json).
At clean detached `31d4204d` on TheRock HIP 7.15 and TuneD
`accelerator-performance`, one warmup plus four balanced same-session
repetitions move 512 prefill **1212.462 -> 535.136 ms** (**422.281 -> 956.765
tok/s, 2.266x**) and 4096 prefill **9977.239 -> 4848.216 ms** (**410.534 ->
844.847 tok/s, 2.058x**). All 16 timed final IDs are `9707`; the linked
six-case project gate has KL at most `5.39e-5` and 100% top-1. Because the
candidate changes recurrent-state bits, this is a retained candidate
performance result rather than a default/topline replacement. The public GGUF
column remains the fused route until multi-prompt generated-trajectory/decode
and explicit numerical-contract gates pass.

That natural-prompt gate subsequently rejects default promotion in
[`2026-07-13-gfx1151-gguf-prefill-gpf2-trajectory-rejection.json`](results/2026-07-13-gfx1151-gguf-prefill-gpf2-trajectory-rejection.json).
At clean `2670ed04`, all ten prompts and four categories run a fused/candidate
prefill sample plus 24 logit-checked transitions and two balanced 128-step
graph windows per mode. Only **7/10** prompts keep the first 25 samples and
only **3/10** keep the complete 129-token trajectory; first divergence ranges
from transition 4 to 126. The diagnostic execution wall is flat (**53.316 vs
53.324 tok/s**), but seven timing legs execute different outputs, so that
number is not a retained decode comparison. The numerical-contract decision
keeps the predeclared exact natural trajectory requirement; `auto` remains
fused and the tree is an explicit rejected diagnostic.

The exact follow-on is also rejected in
[`2026-07-13-gfx1151-gguf-prefill-gpf2c-ordered-resident-rejected.json`](results/2026-07-13-gfx1151-gguf-prefill-gpf2c-ordered-resident-rejected.json).
GPF-2C keeps four state rows per wave lane in registers while preserving every
ordered shuffle and FMA site. Plain/segment output and FP32 state stay byte-
exact and 46 focused tests pass, but 512/1K/4K prefill is only
**368.702/383.292/354.672 tok/s**, **12.98%/14.58%/13.50% below** the clean
fused control. Decode is within -0.31%..-0.24%. A cache-clean trace attributes
**928.006 ms / 30** to recurrence, 16.86% slower than fused. Register residency
therefore fixes global state traffic but not the ordered cross-lane cost;
`auto` remains fused.

The next exact schedule passes its focused candidate gate in
[`2026-07-13-gfx1151-gguf-prefill-gpf2d-lds32-focus-candidate.json`](results/2026-07-13-gfx1151-gguf-prefill-gpf2d-lds32-focus-candidate.json).
GPF-2D assigns one scalar-exact value column to each thread and retains its
128-row FP32 state in a 16 KiB LDS tile across the token loop. Plain/segment
tile32/tile64 fixtures are byte-exact. After rejecting a forced-unroll build
that spilled 1,880 bytes/thread, the rolled LDS32 kernel uses 64 VGPR and zero
scratch; its cache-clean 512 recurrence is **221.873 ms / 30**, 72.06% below
fused. Focused 512/1K/4K prefill improves **423.708/448.694/410.023 ->
753.489/799.844/686.840 tok/s** (**+77.83%/+78.26%/+67.51%**) with decode
−0.10%/+0.03%/+0.03%. This dirty-tree focus artifact is not a retained topline
or default change. The subsequent clean six-case
[`exact matrix`](results/2026-07-13-gfx1151-gguf-prefill-gpf2d-exact-matrix.json)
is byte-identical for sampled tokens, hidden seed, all resident Conv/GDN state,
and the required layer outputs. A clean balanced
[`A/B`](results/2026-07-13-gfx1151-gguf-prefill-gpf2d-balanced-ab.json) moves
512 **420.959 -> 753.891 tok/s (1.791x)** and 4K **408.359 -> 687.831 tok/s
(1.684x)** with exact timed IDs. The clean ten-prompt
[`trajectory/decode gate`](results/2026-07-13-gfx1151-gguf-prefill-gpf2d-trajectory-decode-gate.json)
passes all **250/250** checked logits with `KL=0`, preserves every timed token,
and moves weighted decode **53.4295 -> 53.4416 tok/s (+0.023%)**. GPF-2D is
now the gfx1151-scoped automatic route; gfx1100 remains fused. Its clean
[`six-shape max-context stress gate`](results/2026-07-13-gfx1151-gguf-prefill-gpf2d-default-six-shape.json)
records **751.993/804.420/688.545/589.866/504.730/372.892 tok/s** prefill with
stable five-run IDs and completes in **66.66 minutes**. That one 128K-sized
session is default/long-context validation, not the canonical right-sized
short-shape memory rollup.

The next selected-MoE schedule is promoted from
[`2026-07-13-gfx1151-gguf-prefill-gpf3a-q4t16-shared-x-replay.json`](results/2026-07-13-gfx1151-gguf-prefill-gpf3a-q4t16-shared-x-replay.json).
GPF-3A shares one activation fragment across the existing two independent
Q4T16 WMMA output halves while preserving each accumulator's K/WMMA order.
BF16/FP16 fixture bytes are exact; the tiny trace is **44.725 -> 33.343 us
(-25.45%)**, and identical real 40-layer routing moves Q4 gate/up
**114.633 -> 97.082 ms (-15.31%)**. Its clean balanced
[`full-model gate`](results/2026-07-13-gfx1151-gguf-prefill-gpf3a-full-model-ab.json)
moves 512/1K/4K prefill **747.764/804.150/687.676 ->
771.027/823.624/701.042 tok/s** (**+3.11%/+2.42%/+1.94%**). All three full
logit vectors are byte-exact, every 128-step measured decode trajectory
matches, and aggregate decode wall is **7527.985 -> 7527.750 ms (-0.0031%)**.
The gfx1151 backend capability now selects shared-X automatically; gfx1100
remains on baseline pending its own transfer gate. A clean selector-unset
[`focus confirmation`](results/2026-07-13-gfx1151-gguf-prefill-gpf3a-default-focus.json)
at promoted `431fe1e4` reproduces **774.653/823.149/701.389 tok/s** prefill and
stable IDs. It uses four measurements in one max-4K session, so it confirms
routing/performance; the later right-sized 1+3 publication rollup supersedes
it for the public throughput and memory rows.

The next exact GDN refinement is promoted on gfx1151. GPF-2E removes
prompt-sized raw Q/K/V materialization and computes one Q/K norm per shared K
head, then reads canonical `conv_out` from the scalar-exact LDS32 recurrence.
The clean [`six-case matrix`](results/2026-07-13-gfx1151-gguf-prefill-gpf2e-exact-matrix.json)
matches fused sampled tokens, FP32 hidden seeds, all resident Conv/GDN state,
and required layer outputs. Its
[`balanced A/B`](results/2026-07-13-gfx1151-gguf-prefill-gpf2e-balanced-ab.json)
moves current-default 512/1K/4K prefill
**776.428/825.319/700.824 -> 823.093/889.209/744.577 tok/s**
(**+6.01%/+7.74%/+6.24%**). The
[`natural/decode gate`](results/2026-07-13-gfx1151-gguf-prefill-gpf2e-trajectory-decode-gate.json)
passes 250/250 exact logits and every timed trajectory; weighted decode is
**53.3282 -> 53.3684 tok/s (+0.075%)**. gfx1151 `auto` now uses direct-conv;
gfx1100 remains fused pending transfer evidence. A clean selector-unset
[`focus confirmation`](results/2026-07-13-gfx1151-gguf-prefill-gpf2e-default-focus.json)
reproduces **821.755/897.160/750.896 tok/s** with stable IDs. The right-sized
publication sweep remains.

The original explicit screen remains in
[`2026-07-13-gfx1151-gguf-prefill-gpf2e-direct-conv-screen.json`](results/2026-07-13-gfx1151-gguf-prefill-gpf2e-direct-conv-screen.json).

The dense Q8T16 follow-on is now retained on gfx1151 through 64K. GPF-5A
pairs two production-order 32-column waves and shares one activation tile in
1 KiB LDS. Tail fixtures and 512/4K full-model state are byte-exact; the clean
focus gate improves 512 by **8.35%** and stable 4K by **2.54%**. The automatic
right-sized 1+3 sweep refreshes 512/1K/4K/32K/64K to
**889.904/919.598/762.940/648.948/546.296 tok/s**, all with three token `9707`
IDs and unchanged memory. A stable same-commit 128K gate rejects two-wave
there (**382.041 vs 392.219 tok/s, -2.59%**), so final package policy restores
the production wrapper above 65,536 prompt tokens. The unchanged accepted
**387.334 tok/s** 128K row carries forward: a final scoped retry completed one
**385.474 tok/s** measurement before reproducing the separately documented
later-pass lifecycle stall, which is not enough to replace the accepted 1+3
row. Evidence:
[`2026-07-14-gfx1151-gguf-prefill-gpf5a-right-sized-3run.json`](results/2026-07-14-gfx1151-gguf-prefill-gpf5a-right-sized-3run.json).

LCP-2A further promotes the exact GDN route on gfx1151. It instantiates the
same rolled scalar recurrence with compiler-cacheable LDS state accesses while
keeping the volatile GPF-2E symbol as rollback. At clean detached `53928aaf`,
the six-case state matrix and all **250/250** natural transitions are byte-
exact. One warmup plus four balanced repetitions moves 512/1K/4K prefill
**900.814/940.736/941.462 -> 1213.912/1285.266/1285.888 tok/s**
(**+34.76%/+36.63%/+36.58%**); every pair and timed ID matches. Weighted
decode is **53.348 -> 53.359 tok/s (+0.021%)**. The named kernel uses 32 VGPR,
16 KiB LDS, and zero scratch versus 64 VGPR for GPF-2E. gfx1151 `auto` uses
LCP-2A; gfx1100 remains fused. It is included in the current clean 512-64K
production refresh. Evidence:
[`2026-07-14-gfx1151-gguf-gdn-lcp2a-clean-promotion.json`](results/2026-07-14-gfx1151-gguf-gdn-lcp2a-clean-promotion.json).

The follow-up LCP-M2 device-metadata path is promoted only through 4K. Clean
512/1K/4K automatic-vs-explicit state is **83/83** exact and balanced prefill
improves **+1.56%/+0.90%/+0.53%**; longer prompts retain synchronous metadata.
That scoped fallback is not the remaining 128K trigger: the final current
production run and an explicit metadata-off/router-rollback control both
complete one warmup then enter the same low-power measured-pass-1 stall. The
current 512-64K refresh is retained, while 128K is withheld from the topline:
[`2026-07-15-gfx1151-gguf-production-refresh-512-64k-128k-blocked.json`](results/2026-07-15-gfx1151-gguf-production-refresh-512-64k-128k-blocked.json).
A matched user-space-stack follow-up does not clear the blocker. HIP 7.13
completes two full warmup+3 gates at **509.659/499.895 tok/s** with all six IDs
`9707`, but a post-HIP-7.15 third gate stalls after one measured pass. HIP 7.15
stalls in both controls. All persistent stalls show 100%/2.9 GHz at only
**42-48 W** with no kernel-journal fault. Therefore HIP 7.13 is not a safe
workaround and no cross-stack 128K number is published:
[`2026-07-15-gfx1151-128k-hip713-vs-715-lifecycle.json`](results/2026-07-15-gfx1151-128k-hip713-vs-715-lifecycle.json).

A persistent same-stream flight recorder now narrows one clean HIP 7.15,
one-queue measured-pass-1 failure. The warmup completes at **503.876 prefill /
27.970 decode tok/s**; the next prefill then remains at **100% / 2.9 GHz** and
median **49 W** for 1,436 seconds through the process bound. A retired chunk
marker proves all work through token 28,672 completed, while the host reaches
the layer-11 full-attention checkpoint in chunk `[28672,32768)` and advances no
further. Because that source records layer entry before synchronous chunk
metadata and the layer call, the safe unresolved window is layer-10
linear-attention retirement, layer-11 metadata, or layer-11 full-attention/MoE
work—not proof that one named kernel launched or failed. Kernel logs remain
clean and `amdgpu_fence_info` exposes no mismatch but still cannot see KFD user
queues. The capture predates merged request/chunk metadata reuse, so the merged
scheduler is a distinct lifecycle experiment rather than an inferred fix:
[`2026-07-16-gfx1151-128k-prefill-flight-recorder-stall.json`](results/2026-07-16-gfx1151-128k-prefill-flight-recorder-stall.json).

That merged scheduler experiment also reproduces and is rejected as a lifecycle
fix. A cached exact 512/1 preflight passes, but 128K stalls during its first
warmup. The last retired marker advances farther, through chunk
`[57344,61440)`, while host execution enters layer 18 linear attention in
`[61440,65536)` and never reaches layer 19. The state persists for **1,636
seconds** at 100% activity, 2.9 GHz, and median **45 W**, again with no relevant
kernel log. The main process thread remains runnable in user space while ROCr
event threads wait in KFD. Failure incidence and location moved relative to the
pre-merge capture, so neither one deterministic chunk/layer nor metadata-copy
frequency explains both observations:
[`2026-07-16-gfx1151-128k-merged-metadata-reuse-stall.json`](results/2026-07-16-gfx1151-128k-merged-metadata-reuse-stall.json).

Removing the remaining compact-WMMA scalar D2H synchronization also fails as a
lifecycle workaround. Its recorder-enabled warmup completes at **507.552 /
28.100 prefill/decode tok/s**, then measured prefill 1 stalls. A marker proves
retirement through token 36,864; with no per-layer readback, the host queues all
40 layers of `[36864,40960)`, enters the next chunk, and stops at the synchronous
token-ID/embedding boundary. The low-power state persists 1,437 seconds at
median **54 W**. Thus synchronous copies/readbacks determine where host waiting
becomes visible, but neither metadata copies nor the compact-MoE D2H is
necessary to trigger device no-progress:
[`2026-07-16-gfx1151-128k-compact-no-read-stall.json`](results/2026-07-16-gfx1151-128k-compact-no-read-stall.json).

Layer-granularity marker instrumentation then completes one full HIP 7.15
warmup+3 process: the final **5,392/5,392** checkpoint cursor retires, all four runs return token
`9707`, and the measured medians are **503.732 / 28.246 prefill/decode tok/s**.
This is a useful perturbation signal—not a stability or performance claim.
Layer mode adds a same-stream system-fence kernel writing host-mapped memory
after every layer; repeated independent processes are required before testing a
decoupled queue heartbeat as a workaround:
[`2026-07-16-gfx1151-128k-layer-marker-completion.json`](results/2026-07-16-gfx1151-128k-layer-marker-completion.json).

Both approved independent repeats then stall, rejecting layer markers as a
reliable mitigation. Repeat 1 completes three prefills and stops with layer 11
retired, layer 12's marker pending, and host execution inside layer 13 at
`[28672,32768)`. Repeat 2 completes one prefill and stops with layer 33 retired,
layer 34's marker pending, and the host inside layer 35 at `[16384,20480)`.
Both hold 100%/2.9 GHz at median **43 W** with clean logs. The exact common
window is now the prior linear layer's post-scalar-read MoE tail/marker through
the current layer's pre-read prefix; it still does not name a failed kernel:
[`2026-07-16-gfx1151-128k-layer-marker-repeat-stalls.json`](results/2026-07-16-gfx1151-128k-layer-marker-repeat-stalls.json).

An inline-interposition rocprofv3 run also stalls in its first prefill: layer 15
retires, layer 16's marker remains pending, and the host enters layer 17 at
`[102400,106496)`. At the same onset, rocprofiler's injected HSA completion
signal stops advancing and is polled more than 153 million times. No trace files
finalize, so no last user kernel is recovered. Because this rocprofiler inline
interposer has a documented independent hang class, the result is deliberately
ambiguous; retry with `ROCPROFILER_QUEUE_INTERPOSITION=0` or a streaming trace:
[`2026-07-16-gfx1151-128k-rocprof-inline-interposition-stall.json`](results/2026-07-16-gfx1151-128k-rocprof-inline-interposition-stall.json).

Two subsequent current-boot KFD-control processes both complete exact
warmup+3 gates, with measured prefill medians **488.431/509.332 tok/s**, all six
IDs `9707`, and final recorder cursors **5,392/5,392**. Healthy snapshots expose
two compute queue objects plus one SDMA queue with zero fault/page counters.
However, `kfd/rls` says `No active runlist` even while telemetry records
97-99% activity and 128-129 W, so that view alone cannot discriminate a stall
on this MES configuration. No HQD dump was taken because no stall reproduced;
these controls are not evidence of a lifecycle fix:
[`2026-07-16-gfx1151-128k-kfd-healthy-controls.json`](results/2026-07-16-gfx1151-128k-kfd-healthy-controls.json).

The first rebooted attempt with `mes_log_enable=1`, `gpu_recovery=1`, and
`send_sigterm=1` then reproduces in its first 128K prefill. The recorder freezes
at submitted/completed **389/339** after retiring `[32768,36864)`. For 175
seconds, all 36 samples remain **100% / 2.9 GHz / 41-49 W** with fixed residency.
One established-stall HQD snapshot shows the primary 1 MiB AQL queue active and
non-empty at `rptr=0x32250`, `wptr=0x32450`—**32 unread AQL packets** after the
gfx11 pointer shift—with zero HQD error/dequeue state. The MES event-log bytes
are identical in healthy-active, stalled, and +30-second snapshots, then change
during monitor-requested SIGTERM teardown. Only one HQD sample exists, so this
proves backlog at one instant, not temporal pointer immobility or one failed
packet/kernel. No autonomous recovery, SIGTERM, or reset fires:
[`2026-07-16-gfx1151-128k-mes-kfd-stall-capture.json`](results/2026-07-16-gfx1151-128k-mes-kfd-stall-capture.json).

The follow-up `sched_policy=2` boot confirms `amdgpu: SW scheduler is used`, but
is rejected before it can isolate the original HWS/MES stall. Exact fresh-process
512/1 controls pass before and after at **1163.527/1199.181 tok/s**, token `9707`,
and finite logits. The intervening 128K process faults before prefill at
`0x00007ff3409ae000` on ring 24 / VMID 8 / PASID 31 with faulty client CPF;
ROCr aborts and the coredump places the main thread in `hipMemcpy` through HSA
executable freeze, code-cache invalidation, and `AqlQueue::ExecutePM4`. No reset
occurs and the GPU returns idle. This rejects the software-scheduler boot as a
workaround, but does not show whether non-HWS would affect the original stall if
128K initialization succeeded:
[`2026-07-16-gfx1151-sched-policy2-128k-vm-fault.json`](results/2026-07-16-gfx1151-sched-policy2-128k-vm-fault.json).

SOL-G4 is accepted on gfx1151 in
[`2026-07-11-sol-g4-gfx1151-gguf-eager-decode-audit.json`](results/2026-07-11-sol-g4-gfx1151-gguf-eager-decode-audit.json).
At clean detached `5f4c6561`, the exact repacked/GEMV eager route measures
**49.285 tok/s** (`20.290 ms/token`) for `[9707] * 512` plus 128 timed decode
steps, using one discarded and four measured full runs; every recorded token is
9707 and the artifact links the G1 state oracle by SHA-256. The same synchronized
p8/d32 protocol localizes the first eager performance change to direct-parent
commit `4499fb13`: **17.799 -> 54.963 tok/s** (**+208.79%, 3.088x**) from loaded
HIP-library memoization. Current p8 remains **55.208 tok/s** (+0.45%). A
24-step marker-only profile records **18.402 ms GPU kernels/token** versus
**20.766 ms profiled host wall/token** (88.62%); raw trace CSVs remain under
`/tmp`, while their hashes and the full family Amdahl table are retained.

The current TheRock HIP 7.15 / TuneD refresh promotes explicit wave/block
indexing for the BF16 Q8T16 dual-split leaf at clean detached `e20cdc13`.
Against clean scalar parent `8184355c`, a control/candidate/control p512/d128
eager A/B moves **20.5342 -> 20.4709 ms/token** (**-0.308%**, **48.699 ->
48.850 tok/s**) with non-overlapping ranges and every token exact. Matching
24-step profiles move the named leaf **4245.4 -> 4188.2 us/token** (**-1.349%**)
and total marked GPU time **19256.1 -> 19199.2 us/token** (**-0.296%**). The
state-bound graph path also improves **20.5736 -> 20.5324 ms/token** across
commits, but current G5 runs on both commits find graph slightly slower than
same-run eager; this refresh makes no new graph-over-eager speed claim.

The historical HIP 7.13 SOL-G5 result is accepted on gfx1151 in
[`2026-07-11-sol-g5-gfx1151-gguf-decode-graph-production-audit.json`](results/2026-07-11-sol-g5-gfx1151-gguf-decode-graph-production-audit.json).
At clean detached `7f611fe3`, the production state-bound graph matches eager
byte-for-byte for all 128 launches across generated tokens, the FP32 hidden
seed, 30 Conv/GDN state pairs, and 10 live BF16 K/V pairs. One warmup and four
rotating same-session repetitions measure capture-inclusive graph wall at
**20.311 ms/token** (**49.233 tok/s**) versus same-run eager at
**20.334 ms/token** (**49.178 tok/s**), a **+0.112%** throughput improvement.
The one capture/instantiate and final destroy are charged to every 128-token
window. Per-token recapture is rejected at **35.429 ms/token**. That result
introduced the graph default only for non-streaming c1 greedy gfx1151 windows
with at least 128 remaining transitions. The current HIP 7.15 refresh above
supersedes its speed-policy conclusion: graph replay remains exact but is
slightly slower than same-run eager on both clean commits. The rollback remains
available and a scoped default-policy follow-up is required.

SOL-G6 is accepted on gfx1151 in
[`2026-07-11-sol-g6-gfx1151-gguf-residency-audit.json`](results/2026-07-11-sol-g6-gfx1151-gguf-residency-audit.json).
At clean detached `d70c9464`, the Q4_K_M p512/d128 BF16-KV production graph
session owns **21.478 GiB**, leaving **2.522 GiB** to the explicit 24 GiB gate.
Its 733 planned source tensors have zero raw+replacement duplicates and zero
enabled optional replacement sidecars: **20.461 GiB** is replacement layout,
**0.503 GiB** is the required raw token embedding, and **0.097 GiB** is dense
metadata. Decode scratch is **0.080 GiB** (including **15 MiB** KV and
**63.75 MiB** linear state), while session/prefill buffers are **0.337 GiB**.
Production `record_steps=0` graph capture adds no tracked buffer and a measured
**308 KiB** HIP graph/exec delta. G5 remains the cryptographically linked exact
and performance non-regression gate; this G6 artifact makes no new speed claim.

SOL-P2 is accepted on gfx1151 in
[`2026-07-11-sol-p2-gfx1151-paro-ragged-lifecycle.json`](results/2026-07-11-sol-p2-gfx1151-paro-ragged-lifecycle.json).
At clean detached `6f1910c9`, exact prompt lengths 449 through 512 shrink from
c8 to c1 without compaction. One row retires through EOS; explicit
cancellations create middle, tail, then front holes while every post-event
width remains exact. All eight generated sequences, all 30 linear recurrent
state families, and all 10 live full-attention K/V families match independent
c1. Ragged prefill uses the explicitly labelled `per_segment_ragged_exact`
fallback; this is a correctness artifact with `performance_claim=false`.

The retained Laguna Q4 pack8 shape-policy production is recorded in
[`2026-07-26-gfx1151-laguna-q4-pack8-shape-policy-candidate.json`](results/2026-07-26-gfx1151-laguna-q4-pack8-shape-policy-candidate.json).
Exact 94+24+2 call-weighted leaf timing improves **34.782 -> 33.031 ms
(-5.03%)** with 64x16/64x32/32x32 shape-specific tiles. All 120 actual Q4
projections have zero BF16 mismatches versus 64x16, while dirty one-owner
matrix512/attention128 pp512 improves **489.036 -> 491.014 tok/s (+0.404%)**
with six of seven paired wins and token 2930. The later clean publication is
linked from the superseded platform row below.

The rejected Laguna Q6 selected-down shared-weight local64 screen is recorded
in
[`2026-07-26-gfx1151-laguna-q6-down-shared-weight-local64-rejected.json`](results/2026-07-26-gfx1151-laguna-q6-down-shared-weight-local64-rejected.json).
Actual layer-1 natural-M512 timing regresses **5.223 -> 5.308 ms (+1.635%)**
despite BF16-byte identity, so every candidate surface is removed and Q6
local128 remains production.

The retained Laguna Q6 selected-down 64-row production is recorded in
[`2026-07-26-gfx1151-laguna-q6-down-rows64-production.json`](results/2026-07-26-gfx1151-laguna-q6-down-rows64-production.json).
Clean matched pp512 improves the explicit 32-row rollback **489.110 ->
492.640 tok/s (+0.722%)**, wins all seven paired repetitions, and selects
token 2930 throughout. Cached tracing independently reaches **493.509 tok/s**
and cuts the 23 full-M512 Q6 calls **127.888 -> 126.040 ms**.

The retained stable parallel Laguna MoE compaction production is recorded in
[`2026-07-26-gfx1151-laguna-parallel-compact-production.json`](results/2026-07-26-gfx1151-laguna-parallel-compact-production.json).
Clean matched pp512 improves serial rollback **490.824 -> 497.408 tok/s
(+1.341%)**, wins all seven paired repetitions, and selects token 2930
throughout. Cached tracing independently reaches **500.325/449.468/355.606
tok/s** at 512/1K/4K and cuts the former 16.752-ms serial metadata window to
**2.564 ms**. The 500 gate remains open because its three-sample minimum and
median contract has not yet passed.

The exact parallel-prefix follow-up is recorded in
[`2026-07-26-gfx1151-laguna-parallel-prefix-scan.json`](results/2026-07-26-gfx1151-laguna-parallel-prefix-scan.json).
It replaces the remaining one-thread 256-expert loop with a one-block
exclusive scan and stable ballot compaction. Cached tracing moves the prefix
from **32.34 us/layer** in the production trace to **2.404 us**, a projected
**1.407 ms** pp512 saving, with exact metadata and complete MoE BF16 output.
The later router-token publication compounds this retained sub-window into the
current production result.

The retained exact router-token production is recorded in
[`2026-07-26-gfx1151-laguna-router-token-tile8-production.json`](results/2026-07-26-gfx1151-laguna-router-token-tile8-production.json).
Clean matched pp512 improves tile-4 rollback **497.625 -> 503.349 tok/s
(+1.150%)**, wins all seven paired repetitions, and keeps every production
sample above 500 (**minimum 501.698 tok/s**). Cached tracing independently
reaches **504.631/452.733/357.083 tok/s** at 512/1K/4K and cuts router
**30.658 -> 23.315 ms**. This closes the declared 500 tok/s production gate.

## Platform Index

| Platform | Benchmark family | Run date | Measured revision / build | Evidence status | Root README | Refresh condition |
| --- | --- | --- | --- | --- | --- | --- |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M exact global GQA2 V-stage64 decode | 2026-07-29 | retained `ce6d178c4` and clean production/census; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; two HIP queues; p512/d128 eager c=1; live513/576/639 CPU/rollback byte identity with eviction, nine-sample leaf, seven resident-model A/B pairs, three clean default runs, and cached native traces | **Retained gfx1151 decode default**: 24 local256 workgroups pair adjacent GQA6 heads and reuse each 64-slot LDS V tile across both exact PV chains. Leaves improve **9.16%/12.39%/12.22%** at live513/576/639; all seven complete pairs improve **18.034298 -> 18.237090 tok/s (+1.124%, -0.617 ms/token)**. Clean default production is **18.230064 tok/s**, **+1.129%** over prior clean 18.026501 and **+58.983%** over the 11.466687 sprint start. The clean census cuts global attention **2.878 -> 2.237 ms/token (-22.27%)**, total attention **8.722 -> 8.065 ms (-7.54%)**, and kernel span **55.855 -> 55.154 ms (-1.26%)**. Trajectories/state remain exact. [`candidate/trace`](results/2026-07-29-gfx1151-laguna-global-gqa2-vstage64-retained.json), [`clean production`](results/2026-07-29-gfx1151-laguna-global-gqa2-vstage64-production.json), [`wall census`](results/2026-07-29-gfx1151-laguna-post-global-gqa2-wall-reprofile.json). | Yes, for natural-shape capacity-4096/live<=4000 gfx1151 global attention and exact p512/d128 scope | Rerun after fused attention arithmetic/V staging, span/page ABI, model/quant, compiler/runtime, or device/queue changes; GQA1 remains exact rollback above live4000 and non-natural capacities/shapes retain prior routes. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M exact GQA3 V-stage64 vec16 saturated SWA decode | 2026-07-29 | retained `8aef53b5a` and clean production; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; two HIP queues; p512/d128 eager c=1; wrap/eviction CPU/rollback byte identity, nine-sample leaf, seven resident-model A/B pairs, three clean default runs, and cached native trace | **Retained gfx1151 decode default**: aligned 16-byte global-to-LDS copies replace scalar BF16 staging while the GQA3/local384 QK, softmax, PV FMA, gate, and store sequence remains unchanged. The leaf improves **0.133491 -> 0.106533 ms (-20.19%)**; all seven complete pairs improve **18.244607 -> 18.806305 tok/s (+3.079%, -1.637 ms/token)**. Clean production is **18.814192 tok/s**, **+3.204%** over prior clean 18.230064 and **+64.077%** over the sprint start. Trace records `<64,true>` at 24 local384 blocks, VGPR144/SGPR128/LDS30720/scratch0 and **161.50 -> 112.97 us (-30.05%)**. Trajectories/state remain exact. [`candidate/trace`](results/2026-07-29-gfx1151-laguna-swa-gqa3-vstage64-vec16-retained.json), [`clean production`](results/2026-07-29-gfx1151-laguna-swa-gqa3-vstage64-vec16-production.json). | Yes, for saturated natural-shape gfx1151 SWA and exact p512/d128 scope | Rerun after staged-copy width/alignment, attention arithmetic, ring/span ABI, model/quant, compiler/runtime, or device/queue changes; scalar V-stage64 remains exact rollback and shorter/non-natural/peer routes are unchanged. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M exact global GQA2 V-stage64 vec16 decode | 2026-07-29 | retained worktree over `6a1f79501`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; two HIP queues; p512/d128 eager c=1; live513/576/639 CPU/rollback byte identity with eviction, nine-sample leaf, seven resident-model A/B pairs, and cached native trace | **Retained gfx1151 decode default**: dynamic score/physical LDS is padded to 16-byte alignment and each copy moves eight BF16 V values; all global attention arithmetic is unchanged. Leaves improve **22.29%/25.82%/25.99%**; all seven complete pairs improve **18.794424 -> 19.066920 tok/s (+1.450%, -0.760 ms/token)** with identical trajectories/state. Trace records `<2,64,true>` at 24 local256 blocks, VGPR32/SGPR128/static-LDS512/scratch32 and **141.09 -> 103.29 us (-26.79%)**. [`artifact`](results/2026-07-29-gfx1151-laguna-global-gqa2-vstage64-vec16-retained.json). | Yes, for natural-shape capacity-4096/live<=4000 gfx1151 global attention and exact p512/d128 scope | Rerun after staged-copy width/alignment, attention arithmetic, span/page ABI, model/quant, compiler/runtime, or device/queue changes; scalar GQA2 remains exact rollback and GQA1 remains fallback above live4000. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M exact GQA3 V-stage64 saturated SWA decode | 2026-07-29 | retained/clean production `e420f8f32`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; two HIP queues; p512/d128 eager c=1; ring/wrap/eviction byte identity, 32/64/128-slot leaf sweep, seven resident-model A/B pairs, three clean default runs, and cached native traces | **Retained gfx1151 decode default; 18 tok/s passed**: 64 contiguous V slots x D128 are staged in LDS once and reused across the three query heads owned by each local384 block. All scalar/FMA associations and output bytes remain unchanged. Seven pairs improve **17.135411 -> 18.032171 tok/s (+5.233%)**; clean default production is **18.026501 tok/s**, **+5.172%** over the prior clean 17.139971. The kernel falls **184.085 -> 137.197 us (-25.47%)**, while the clean census cuts SWA **8.891 -> 5.844 ms/token (-34.27%)**, total attention **11.764 -> 8.722 ms (-25.86%)**, and kernel span **58.846 -> 55.855 ms (-5.08%)**. Trajectory/state remain exact. [`candidate/trace`](results/2026-07-29-gfx1151-laguna-swa-gqa3-vstage64-retained.json), [`clean production`](results/2026-07-29-gfx1151-laguna-swa-gqa3-vstage64-production.json), [`wall re-profile`](results/2026-07-29-gfx1151-laguna-post-vstage64-wall-reprofile.json). | Yes, for saturated natural-shape gfx1151 SWA and exact p512/d128 scope | Rerun after fused attention arithmetic/V staging, ring/span ABI, model/quant, compiler/runtime, or device/queue changes; unstaged local384 remains rollback and shorter/non-natural/peer routes are unchanged. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M exact fused-GQA3 local384 saturated SWA decode | 2026-07-29 | retained `e93415110`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; two HIP queues; p512/d128 eager c=1; ring/wrap/eviction byte identity, nine-sample leaf, seven resident-model A/B pairs, cached native trace, and clean 127-transition production census | **Retained gfx1151 decode default**: three local384 owners per KV head keep 288 active query/dimension waves while reducing saturated K-cache ownership **5 -> 3** versus fused-GQA2. The cache-hot leaf is tied at **+0.397%**, but all seven resident pairs improve **17.100489 -> 17.139971 tok/s (+0.231%, -0.135 ms/token)** with identical trajectories/state. Trace records 24 local384 blocks, VGPR104/SGPR128/LDS8192/scratch0. The clean post-retention census lowers attention **13.778 -> 11.764 ms/token (-14.62%)** and dispatches **864 -> 816**, but Vulkan remains at **0.909 ms/token**. [`retained artifact`](results/2026-07-29-gfx1151-laguna-swa-gqa3-local384-retained.json), [`wall re-profile`](results/2026-07-29-gfx1151-laguna-post-gqa3-wall-reprofile.json). | Yes, for saturated natural-shape gfx1151 SWA and exact p512/d128 scope | Rerun after fused attention arithmetic/ownership, ring/span ABI, model/quant, compiler/runtime, or device/queue changes; fused-GQA2 remains rollback and shorter/non-natural/peer routes retain the existing exact chain. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M exact fused one-head global decode | 2026-07-28 | base `ecdb5b802`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; two HIP queues; p512/d128 eager c=1; live 257/513/576/639 byte identity, nine-sample leaf, seven resident-model A/B pairs, and cached native trace | **Retained gfx1151 decode default**: all 48 local256 workgroups fuse exact QK, the retained eight-wave max/denominator association, PV, gate, and stores, removing the score/physical round-trip and one launch. Complete leaves improve **7.89-17.55%** at live 513/576/639; seven complete pairs improve **17.064962 -> 17.097044 tok/s (+0.188%, -0.110 ms/token)** with every candidate faster and identical trajectories. Trace records 48 blocks / 384 wave32s per layer, VGPR24/SGPR128/scratch0 and dynamic LDS of 8 bytes/live slot. Two-head GQA2 regresses production **0.126%** after collapsing to 24 workgroups and is removed. [`artifact`](results/2026-07-28-gfx1151-laguna-global-fused-gqa1-retained.json). | Yes, for natural-shape capacity-4096 gfx1151 global attention and exact p512/d128 scope | Rerun after fused attention arithmetic/ownership, span/page ABI, model/quant, compiler/runtime, or device/queue changes; non-natural shapes/capacities and peer backends retain exact split score plus fixed-shape reduction. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M exact fused-GQA2 saturated SWA decode | 2026-07-28 | base `ae5f0c463`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; two HIP queues; p512/d128 eager c=1; ring/wrap/eviction byte identity, nine-sample leaf, seven resident-model A/B pairs, and cached native trace | **Retained gfx1151 decode default**: five owners per KV head fuse exact QK, ordered softmax, PV, and gate for query-head pairs, removing the global score plane/launch while preserving all arithmetic and output bytes. Seven complete pairs improve **17.013184 -> 17.065241 tok/s (+0.306%, -0.179 ms/token)** with every candidate faster and identical trajectories. The cache-hot leaf is **2.96% slower**, while one-head local256 fusion improves that leaf **8.14%** but regresses resident production **1.038%** and is removed. Trace records 40 local256 blocks / 320 wave32s per SWA layer, VGPR32/LDS6144/scratch0. [`artifact`](results/2026-07-28-gfx1151-laguna-swa-fused-gqa2-retained.json). | Yes, for saturated natural-shape gfx1151 SWA and exact p512/d128 scope | Rerun after fused attention arithmetic/ownership, ring/span ABI, model/quant, compiler/runtime, or device/queue changes; shorter live counts, non-natural shapes, and peer backends retain exact GQA3 score plus fixed512 reduction. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M exact global fixed-shape reduction | 2026-07-28 | base `2f5d621f5`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; two HIP queues; p512/d128 eager c=1; live 257/513/576/639 byte identity, nine-sample leaf, seven same-session A/B pairs, and cached whole-model trace | **Retained gfx1151 decode default**: natural 48Q/8KV/D128/capacity-4096 global attention keeps dynamic live spans plus 48 local256 reducer workgroups and specializes only fixed dimensions, scratch strides, and bounded addressing. Complete score+reduce improves **0.7-2.0%** at live 513/576/639; production improves **16.832097 -> 16.846689 tok/s (+0.087%, -0.051 ms/token)** with every candidate faster and identical trajectories. Trace records **1,524 = 12 x 127** fixed reducers, zero generic fallback, local256/VGPR24/LDS512/scratch0. [`artifact`](results/2026-07-28-gfx1151-laguna-global-fixedshape-reduce-retained.json). | Yes, for natural-shape capacity-4096 gfx1151 global attention and exact p512/d128 scope | Rerun after global score/reducer arithmetic, span ABI, model/quant, compiler/runtime, or device/queue changes; non-natural shapes/capacities and peer backends keep the generic exact route. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M exact saturated-512 SWA reduction | 2026-07-28 | base `6ce10017d`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; two HIP queues; p512/d128 eager c=1; live/wrap/eviction byte identity, nine-sample leaf, seven same-session A/B pairs, and cached whole-model trace | **Retained gfx1151 decode default**: saturated 512-slot SWA keeps the exact GQA3 score owner plus 72 local128 value workgroups / 288 wave32s and specializes only 72Q/8KV/D128 bounds/addressing. Complete score+reduce improves **0.108265 -> 0.081059 ms/layer (-25.13%)**; production improves **16.386231 -> 16.833740 tok/s (+2.731%, -1.622 ms/token)** with every candidate faster and identical trajectories. Trace records **4,572 = 36 x 127** fixed reducers, zero generic fallback, local128/VGPR16/LDS0/scratch0. [`artifact`](results/2026-07-28-gfx1151-laguna-swa-fixed512-reduce-retained.json). | Yes, for saturated natural-shape gfx1151 SWA and exact p512/d128 scope | Rerun after SWA score/reducer arithmetic, ring/span ABI, model/quant, compiler/runtime, or device/queue changes; live below 512 and peer backends keep the generic exact route. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M exact source-F16 fixed-K decode | 2026-07-28 | base `3a9be397a`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; two HIP queues; p512/d128 eager c=1; natural-K byte identity, seven same-session A/B pairs, and cached whole-model trace | **Retained gfx1151 decode default**: rows==1 QKV/gate/O preserve the local256/eight-wave/one-output reduction exactly while compile-time-specializing K3072/K6144/K9216. All six roles improve **15.91-26.93%** and weighted F16 family falls **30.952 -> 24.482 ms/token (-20.90%)**. Production improves retained one-barrier **14.786076 -> 16.391201 tok/s (+10.856%)**; every candidate beats every control and trajectories are identical. Trace records exactly **18,288 = 144 x 127** fixed-K calls, zero fallback, and local256/VGPR24/LDS512/scratch0. [`fixed-K artifact`](results/2026-07-28-gfx1151-laguna-f16-fixedk-retained.json), [`one-barrier predecessor`](results/2026-07-28-gfx1151-laguna-f16-onebarrier-retained.json). | Yes, for declared gfx1151 rows==1 natural-K source-F16 roles and p512/d128 exact scope | Rerun after F16 projection arithmetic/dispatch, model/quant, compiler/runtime, or device/queue changes; explicit `HIPENGINE_LAGUNA_F16_DECODE=onebarrier` is the exact generic-K rollback. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M exact SWA GQA3 decode score owner | 2026-07-28 | clean candidate and in-process rollback `0249d1534`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; two HIP queues; p512/d128 eager c=1; cached native trace and boundary exactness gate | **Retained gfx1151 decode default**: one score workgroup loads each BF16 SWA key once for three adjacent query heads, independently recreates the retained ordered dot reductions, and preserves the existing score/physical scratch plus wave-local value reducer ABI. Same-commit rollback/candidate medians move **14.563678 -> 14.740486 tok/s (+1.214%)**, saving **0.824 ms/token**; all three candidate samples exceed all three controls. The live-511/512 score producer falls **35.988/35.707/35.787 -> 20.839/20.559/19.035 us (-42.1% to -46.8%)** at local256/VGPR40/LDS0/scratch0. F32 context, gated BF16, complete 128-token trajectory, positions, and lifecycle are exact. Grouped GQA9/GQA3 value reducers were removed after **5-11%** regressions. [`artifact`](results/2026-07-28-gfx1151-laguna-swa-gqa3-scores-retained.json). | Yes, for the declared gfx1151 gated split-SWA and p512/d128 exact scope | Rerun after score/reducer arithmetic, split thresholds, KV/span ABI, model/quant, compiler/runtime, or device/queue policy changes; continue LD-1 with split-K/fused attention rather than grouped value-only ownership. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M direct packed-query production | 2026-07-27 | runtime/default `fc84d87b9`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; automatic two HIP queues, HIP priority range `[+1,-1]`; matrix2048/attention128; F32-bit producer equality, 11 complete-state A/B pairs, cached all-family trace, and three clean selector-unset repetitions | **Current retained quality-gated production default; 500 passed, 700 pp512 active**: the fused per-head RMSNorm/RoPE producer writes qualified M128 query tiles directly head-major for the existing packed zero-workspace QK route. pp512 query transpose falls **144 launches / 4.907 ms -> 0**, total dispatches fall **2,273 -> 2,129**, and producer-plus-pack falls **20.530 -> 16.666 ms (-18.82%)**; the producer is local256/VGPR16/LDS0/scratch0. Clean selector-unset 512/1K/4K is **654.249/579.699/468.608 tok/s**, improving the preceding packet **+0.991%/+0.689%/+0.108%**. Eleven A/B pairs are complete-state exact and improve **647.210 -> 650.651 tok/s (+0.532%, 7/11 wins)**. Canonical quality transfers unchanged at max KL **0.049542582**, 316/320 top-1. [`production artifact`](results/2026-07-27-gfx1151-laguna-attention-packed-query-producer-production.json), [`candidate artifact`](results/2026-07-27-gfx1151-laguna-attention-packed-query-producer-candidate.json). | Yes, for the exact query-producer boundary, launch reduction, selector-unset pp512, and 1K/4K continuity under the declared route/category/lifecycle scope | Rerun after attention/query layout, fused RMSNorm/RoPE, model/quant, compiler/runtime, or device/queue changes. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M packed-output attention-gate production | 2026-07-27 | candidate/default `4659d69e3`, division-free repair/trace `c86f22ee7`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; automatic two HIP queues, HIP priority range `[+1,-1]`; matrix2048/attention128; mixed-layout BF16 equality, 11 complete-state A/B pairs, cached all-family trace, and three clean selector-unset repetitions | **Superseded retained quality-gated production**: qualified dense-initial PV tiles remain head-major until one exact `(row, head)` softplus gate emits the ordinary generic tensor. pp512 output unpack falls **144 launches / 3.703 ms -> 0**, total dispatches fall **2,417 -> 2,273**, and transpose-plus-gate falls **11.240 -> 10.318 ms (-8.20%)**; the packed gate is local128/VGPR8/LDS0/scratch0. Clean selector-unset 512/1K/4K is **647.826/575.732/468.103 tok/s**, within **-0.302%/-0.149%/-0.155%** shared-APU variance of the preceding packet. Eleven A/B pairs are complete-state exact and independent medians improve **645.735 -> 647.920 tok/s (+0.338%)**. Canonical quality transfers unchanged at max KL **0.049542582**, 316/320 top-1. [`production artifact`](results/2026-07-27-gfx1151-laguna-attention-packed-output-gate-production.json), [`candidate artifact`](results/2026-07-27-gfx1151-laguna-attention-packed-output-gate-candidate.json). | Superseded; exact packed-output boundary provenance | Use the direct packed-query production row above for current performance and correctness. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M row-qualified-F16/raw-nibble-P8/precomputed-sum production | 2026-07-27 | Q4 precomputed-sum runtime/default `119ff7700`, category revalidation `e89957333`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; automatic two HIP queues, HIP priority range `[+1,-1]`; matrix2048/attention128; CPU-reference RED/GREEN, exact 512/1K complete-state A/B, transferred category gate, cached family trace, and three clean selector-unset repetitions | **Superseded retained quality-gated production**: the Q4 D8 producer stores exact K16 sums once in a bounded activation-only sidecar, avoiding reconstruction in 16 gate/up output-column workgroups. Clean selector-unset 512/1K/4K is **649.791/576.589/468.830 tok/s**, moving the prior packet **+0.399%/-0.036%/+0.085%** and cutting pp512 wall **791.092 -> 787.946 ms**. The aggregate-flat 1K result is positive same-process at **-4.428 ms paired median (7/11 wins)**. Tokens, positions, complete state, deterministic repeats, and allocation recovery pass. Canonical quality remains max KL **0.049542582**, 316/320 top-1, every category >=96.875%; direct M512 quality is KL **0.00407713** with top-1 2930. The clean trace records **2,417** dispatches and cuts selected gate/up **334.229 -> 330.720 ms (-1.050%)** at local128/VGPR96/LDS3072B/scratch0. [`production artifact`](results/2026-07-27-gfx1151-laguna-q4-precomputed-activation-sums-production.json), [`candidate artifact`](results/2026-07-27-gfx1151-laguna-q4-precomputed-activation-sums-candidate.json). | Superseded; exact Q4 activation-sum provenance | Use the packed-output attention-gate row above for current performance and correctness. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M planar-Q6 WMMA-prefetch production prefill | 2026-07-27 | runtime/default/clean measurement and trace `b58461a5e`, category revalidation `e89957333`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; automatic two HIP queues, HIP priority range `[+1,-1]`; matrix2048/attention128; CPU-reference/BF16 equality, complete-state A/B, transferred category gate, cached all-family trace, and three clean selector-unset repetitions | **Superseded retained quality-gated production**: the selected-down Q6 body prefetches the next planar-qmicro K32 record and metadata while current integer-WMMA fragments execute, with zero resident/LDS/scratch growth. Clean selector-unset pp512 improves **632.618 -> 636.073 tok/s (+0.546%)**; 1K/4K are flat at **568.765/464.061 tok/s**. Cached tracing cuts the 23-call pp512 Q6 body **112.746 -> 101.963 ms (-9.564%)** at local128/VGPR104/LDS5120B/scratch0. [`artifact`](results/2026-07-27-gfx1151-laguna-q6-wmma-weight-prefetch-production.json). | Superseded; exact next-weight prefetch provenance | Use the weight+activation-prefetch row above for current performance and correctness. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M exact planar-Q6 integer-WMMA production prefill | 2026-07-26 | hoist/default `acbb72215`, publication/trace `76c963f2a`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; automatic two HIP queues, HIP priority range `[+1,-1]`; matrix2048/attention128; actual-weight leaf, CPU-reference gate, cached all-family trace, and three clean selector-unset repetitions | **Superseded retained quality-gated production**: each wave reuses its exact K16 activation vectors across four output fragments without changing resident bytes, arithmetic, BF16 output, or kernel resources. Clean selector-unset improves **576.137/543.213/459.054 -> 577.396/545.366/459.716 tok/s (+0.218%/+0.396%/+0.144%)**. The exact actual leaf improves **1.136% (20/21 wins)** at unchanged local128/VGPR96/LDS5120B/scratch0; tracing cuts the 115-call Q6 window another **1.63%**. Absolute quality remains max KL **0.049542582**, 316/320 top-1, every category >=96.875%, deterministic, and lifecycle-exact through 4K. [`artifact`](results/2026-07-26-gfx1151-laguna-q6-integer-wmma-hoist-activation-production.json). | Superseded; exact Q6 arithmetic provenance | Use the dense-initial F32 hipBLASLt-attention row above for current performance and correctness. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M exact low-priority shared-MoE production prefill | 2026-07-26 | capability/measurement `a63a503b3`, matched/trace `d2426ede2`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; automatic two HIP queues, HIP priority range `[+1,-1]`; matrix2048/attention128 | **Superseded retained quality-gated production**: after-router least-priority overlap publishes **568.849/527.113/444.508 tok/s** and exact complete state. Cached tracing reaches **574.011 tok/s** and cuts kernel span **7.255 ms**. [`artifact`](results/2026-07-26-gfx1151-laguna-moe-shared-low-priority-production.json). | Superseded; exact scheduling provenance | Use the Q6 qmicro-permute row above for current performance and correctness. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M exact after-router shared-MoE production prefill | 2026-07-26 | capability/measurement `764de3fc4`, matched/trace `a9883e6fa`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; automatic two HIP queues; matrix2048/attention128; three selector-unset 512/1K/4K repetitions, seven queue-matched complete-state pairs, and cached two-stream trace | **Superseded retained quality-gated production**: router selection completes before the priority-0 shared expert is released. Clean selector-unset reaches **566.839/527.381/444.447 tok/s**. Seven pairs preserve complete state and cached tracing cuts kernel span **898.334 -> 898.024 ms**. Absolute quality remains max KL **0.049542582**. [`artifact`](results/2026-07-26-gfx1151-laguna-moe-shared-after-router-production.json). | Superseded; exact after-router provenance | Use the low-priority shared-MoE production row above for current performance and correctness. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M exact eager shared/routed MoE concurrency production prefill | 2026-07-26 | capability `6e58950e3`, measured/trace `0cfe25bb7`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; automatic two HIP queues; matrix2048/attention128; three selector-unset 512/1K/4K repetitions, seven queue-matched complete-state pairs, and cached two-stream trace | **Superseded retained quality-gated production**: clean selector-unset reaches **565.447/526.711/443.444 tok/s**. Seven pairs preserve complete state and cached tracing cuts the former single-stream kernel span **909.598 -> 898.334 ms**. Absolute quality remains max KL **0.049542582**. [`artifact`](results/2026-07-26-gfx1151-laguna-moe-branch-concurrency-production.json). | Superseded; exact eager concurrency provenance | Use the low-priority shared-MoE production row above for current performance and correctness. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M exact source-F16 boundary production prefill | 2026-07-26 | default `893b39197`, complete-state/trace `6f7cea1c0`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; one HIP queue; matrix2048/attention128; three selector-unset 512/1K/4K repetitions, seven matched complete-state pairs, and cached all-family trace | **Superseded retained quality-gated production**: RMSNorm and softplus-gate producers write exact `FP16(BF16(value))`, eliminating 96 standalone casts. Clean selector-unset reaches **559.554/523.912/440.809 tok/s**. Seven pairs preserve logits, both hidden snapshots, complete KV, token/logit, and cursor exactly. Cached tracing reaches **561.019 tok/s** and records **1,696** pp512 dispatches. Absolute quality remains max KL **0.049542582**. [`artifact`](results/2026-07-26-gfx1151-laguna-f16-boundary-fusion-production.json). | Superseded; exact source-F16 boundary provenance | Use the low-priority shared-MoE production row above for current performance and correctness. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M exact dense-initial attention production prefill | 2026-07-26 | clean measured/default `227398af9`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; one HIP queue; matrix2048/attention128; three selector-unset 512/1K/4K repetitions, seven matched complete-state pairs, and cached all-family trace | **Superseded retained quality-gated production**: complete initial no-wrap tiles preserve the full `KVLiveSpans` ABI while removing per-token position/eviction reads. Clean selector-unset reaches **559.290/523.090/439.044 tok/s**. Matched A/B is complete-state exact at **552.144 -> 559.539 tok/s (+1.339%)**. Cached tracing reaches **559.225 tok/s**, observes 12 global-qrow4 / 36 global-qrow6 / 144 SWA-qrow4 calls, and cuts attention **153.226 -> 141.846 ms (-7.43%)**. [`artifact`](results/2026-07-26-gfx1151-laguna-attention-dense-initial-production.json). | Superseded; exact dense-initial attention provenance | Use the source-F16 boundary row above for current performance and correctness. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M exact padded-Q6 production prefill | 2026-07-26 | clean measured/default `fe105632c`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; one HIP queue; matrix2048/attention128; three selector-unset 512/1K/4K repetitions, eleven complete-state pairs, repeated actual-weight 23-layer timing, and cached all-family trace | **Superseded retained quality-gated production**: never-read padded Q6 slots skip zero activation-cache writes and K16 sums with unchanged useful arithmetic and resources. Clean selector-unset 512/1K/4K reaches **551.459/517.307/432.099 tok/s**, improving the preceding packet **0.420%/0.456%/0.418%**. The repeated exact Q6 window improves **112.008 -> 111.806 ms (-0.180%, 19/23 layers)** and complete-state A/B is exact and positive. Cached tracing reaches **552.796 tok/s** at local128/VGPR88/SGPR128/LDS5120B/scratch0; its one Q6 slice is noisy and not used as promotion evidence. Absolute quality remains max KL **0.049542582**, 316/320 top-1, every category >=96.875%, with deterministic repeats, exact KV/cursors, neutral decode, and exact lifecycle. [`artifact`](results/2026-07-26-gfx1151-laguna-q6-skip-padded-activation-production.json). | Superseded; exact Q6 staging provenance | Use the dense-initial attention production row above for current performance and correctness. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M exact global-qrow6 production prefill | 2026-07-26 | clean measured/default `0f71800dc`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; one HIP queue; matrix2048/attention128; seven matched pp512 pairs, three selector-unset 512/1K/4K repetitions, and cached all-family trace | **Superseded retained quality-gated production**: global qrow6 owns only complete preappended M128 tiles from position 128, while start 0 and all SWA retain qrow4. Clean selector-unset 512/1K/4K reaches **547.064/513.180/428.628 tok/s**, improving the M2048 packet **0.376%/1.359%/4.518%**. The matched gate is complete-state exact and wins 7/7; cached tracing cuts global+SWA attention **158.702 -> 152.406 ms (-3.97%)**. [`artifact`](results/2026-07-26-gfx1151-laguna-global-qrow6-production.json). | Superseded; exact attention provenance | Use the padded-Q6 production row above for current performance and correctness. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M M2048 production prefill | 2026-07-26 | clean screen `9f560a764`; promoted/default `dcf29b1d5`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; one HIP queue; matrix2048/attention128; two matched-policy and three selector-unset 512/1K/4K repetitions | **Superseded quality-gated production packet**: projection/MoE capacity rises M512 -> M2048 while attention stays M128 and physical KV operations remain bounded. Clean selector-unset 512/1K/4K reaches **545.015/506.299/410.099 tok/s**; pp512 is flat within **-0.199%** variance while 1K/4K improve **5.120%/5.238%** over the preceding packet. Matched M2048 improves 1K/4K **5.420%/5.752%**, has maximum relative KL **0.000012503**, 100% top-1, deterministic repeats, exact multi-wrap KV semantics, and exact lifecycle. Scratch is **1,755,275,296 bytes**, within the existing 2-GiB admission floor. [`artifact`](results/2026-07-26-gfx1151-laguna-m2048-production.json). | Superseded; exact M2048 scheduling provenance | Use the padded-Q6 production row above for current performance and correctness. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M exact fused selected-SiLU pack production prefill | 2026-07-26 | clean measured/default `c0730bb94`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; one HIP queue; matrix512/attention128; three selector-unset 512/1K/4K repetitions, seven paired admission runs, and cached all-family trace | **Superseded matrix512 production packet**: reuse selected-down scratch for packed gate/up and fold the standalone sparse SiLU into the range-safe Q8 pack while explicitly preserving its BF16 boundary. Clean selector-unset pp512 improves **543.807 -> 546.100 tok/s (+0.422%)**, with minimum **543.299 tok/s**; 1K/4K improve **480.017 -> 481.640 (+0.338%)** and **388.595 -> 389.686 tok/s (+0.281%)**. Direct all-exact quality transfers unchanged at max KL **0.049542582**, **316/320 (98.75%)** top-1, every category >=96.875%, deterministic repeats, Poolside exact top-1, neutral decode, and exact lifecycle. Cached tracing removes another **47 dispatches**, names 47 fused packs at local128/VGPR16/LDS512B/scratch0, records zero standalone selected-SiLU launches, and reaches **549.845 tok/s**. [`artifact`](results/2026-07-26-gfx1151-laguna-fused-silu-pack-production.json). | Superseded; exact fused-pack and matrix512 trace provenance | Use the M2048 production row above for current performance and correctness. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M Q6-down heavy 128-row tile | 2026-07-26 | rejected leaf on `8538f4bc4`; TheRock HIP 7.15; exact actual layer-1 Q6 weight and natural pp512 routing; serial 11-sample actual-weight screens plus cached resource trace | **Rejected and fully removed**: the 64-column x 128-row/local256 qmicro body is BF16-byte exact and preserves 32 accumulators/lane. The >=65-row subset collapses **32 -> 17** weight tiles but saves only **0.017 ms** before its required second metadata schedule/launch; the >=129-row tail collapses **14 -> 8** tiles yet regresses **0.673548 -> 0.687981 ms (+2.14%)**. Trace resources are local256/VGPR88/LDS8704B/scratch0. [`artifact`](results/2026-07-26-gfx1151-laguna-q6-down-rows128-heavy-rejected.json). | No; rejected leaf evidence only | Do not retry a larger local256 Q6 row tile without a new occupancy mechanism. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M exact fused selected-SiLU pack candidate | 2026-07-26 | candidate on `4cbb50632`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; one HIP queue; matrix512/attention128; seven paired pp512 repetitions and one cached traced pair | **Superseded exact admission evidence**: reuse selected-down scratch for packed gate/up and fold the standalone sparse SiLU into the range-safe Q8 pack while explicitly preserving its BF16 boundary. All seven complete-state pairs are exact and the candidate wins **7/7**; paired geometric throughput improves **0.651%** and median paired wall falls **4.636 ms**. Tracing removes **47 dispatches**, cuts the target window **10.301 -> 6.377 ms (-38.09%)**, and records local128/VGPR16/LDS512B/scratch0. [`artifact`](results/2026-07-26-gfx1151-laguna-fused-silu-pack-candidate.json). | Superseded; exact paired/sub-window provenance | Use the fused selected-SiLU pack production row above for current performance and correctness. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M exact MMQ grouped-combine production prefill | 2026-07-26 | clean measured/default `b6bfc4a0b`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; one HIP queue; matrix512/attention128; three selector-unset 512/1K/4K repetitions, seven paired admission runs, and cached all-family trace | **Superseded exact grouped-combine production**: the registered exact sorted-lane weighted-sum plus shared-add composite follows MMQ selected down, preserving every BF16 boundary while removing one routed-output round trip and one launch per sparse layer. Clean selector-unset pp512 improves **542.088 -> 543.807 tok/s (+0.317%)**, with minimum **541.485 tok/s**; 1K/4K improve **478.856 -> 480.017 (+0.243%)** and **387.725 -> 388.595 tok/s (+0.224%)**. Direct all-exact quality transfers unchanged at max KL **0.049542582**. [`artifact`](results/2026-07-26-gfx1151-laguna-mmq-combine-production.json). | Superseded; exact grouped-combine provenance | Use the fused selected-SiLU pack production row above for current performance and correctness. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M cached-metadata attention production prefill | 2026-07-26 | clean measured/default `53e3c2468`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; one HIP queue; matrix512/attention128; three selector-unset 512/1K/4K repetitions, matched exact A/B, and cached kernel trace | **Superseded exact attention production**: qualified metadata-only qrow4 selects every safe SWA M128 tile and global tiles from position 128 while keeping global position 0 on the established cached body. Clean selector-unset pp512 improves **530.447 -> 542.088 tok/s (+2.195%)**, with minimum **542.022 tok/s**; 1K/4K improve **473.118 -> 478.856 (+1.213%)** and **381.375 -> 387.725 tok/s (+1.665%)**. Direct all-exact quality transfers unchanged at max KL **0.049542582**. Cached tracing cuts global+SWA attention **175.802 -> 160.123 ms (-8.92%, 15.679 ms saved)**. [`artifact`](results/2026-07-26-gfx1151-laguna-attention-cached-meta-production.json). | Superseded; exact cached-metadata attention provenance | Use the MMQ grouped-combine production row above for current performance and correctness. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M Q6-qmicro production prefill | 2026-07-26 | clean measured/default `7aa0e0985`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; one HIP queue; matrix512/attention128; three selector-unset 512/1K/4K repetitions and cached kernel trace | **Superseded exact Q6-layout production**: gfx1151 stores sparse expert Q6 payloads in byte-neutral 12-byte qmicro records while preserving the 3,360-byte T16 tile and every BF16 result. Clean selector-unset pp512 improves **526.451 -> 530.447 tok/s (+0.759%)**, with minimum **525.864 tok/s**; 1K/4K improve **467.846 -> 473.118 (+1.127%)** and **377.905 -> 381.375 tok/s (+0.918%)**. Direct all-exact quality transfers unchanged at max KL **0.049542582**, **316/320 (98.75%)** top-1, every category >=96.875%, deterministic repeats, Poolside exact top-1, neutral decode, and exact lifecycle. Cached tracing reaches **535.006 tok/s**, cuts Q6 selected down **126.594 -> 123.473 ms (-2.465%)**, and cuts total selected down **203.923 -> 200.510 ms (-1.673%)** at local128/VGPR88/LDS5632B/scratch0. [`artifact`](results/2026-07-26-gfx1151-laguna-q6-qmicro-production.json). | Superseded; exact Q6-layout provenance | Use the cached-metadata attention row above for current production performance and correctness. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M eight-token router-logit production prefill | 2026-07-26 | clean measured/default `238eb28cd`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; one HIP queue; matrix512/attention128; seven paired pp512 repetitions, byte-exact router/MoE transfer, and cached all-family trace | **Superseded exact router production**: gfx1151 reuses each router hidden row across eight token reductions per workgroup while retaining the exact K traversal and reduction tree; token tile 4 remains rollback and the unmeasured-backend default. Clean matched pp512 improves tile-4 rollback **497.625 -> 503.349 tok/s (+1.150%)**, wins all seven pairs, places every production sample above 500 (**minimum 501.698 tok/s**), and selects token 2930 throughout. Direct all-exact quality transfers unchanged at max KL **0.049542582**, **316/320 (98.75%)** top-1, every category >=96.875%, deterministic repeats, Poolside exact top-1, neutral decode, and exact lifecycle. Cached tracing independently measures **504.631/452.733/357.083 tok/s** at 512/1K/4K and cuts the router family **30.658 -> 23.315 ms**. [`artifact`](results/2026-07-26-gfx1151-laguna-router-token-tile8-production.json). | Superseded; exact router provenance | Use the activation-double-buffer row above for current production performance and correctness. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M stable parallel MoE-compaction production prefill | 2026-07-26 | clean measured/default `f91eccb5d`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; one HIP queue; matrix512/attention128; seven paired pp512 repetitions, byte-exact metadata/MoE transfer, and cached all-family trace | **Superseded exact MoE-scheduler production**: gfx1151 replaces one-workgroup serial stable compaction with per-expert count, prefix, and ballot-ordered scatter kernels; gfx1100 and the explicit serial rollback are unchanged. Clean matched pp512 improves serial rollback **490.824 -> 497.408 tok/s (+1.341%)**, wins all seven paired repetitions, and selects token 2930 throughout. Direct all-exact quality transfers unchanged at max KL **0.049542582** and **316/320 (98.75%)** top-1. Cached tracing independently measures **500.325/449.468/355.606 tok/s** at 512/1K/4K; the old **16.752 ms** serial compact window falls to **2.564 ms**. [`artifact`](results/2026-07-26-gfx1151-laguna-parallel-compact-production.json). | Superseded; exact parallel-compaction provenance | Use the router-token-tile row above for current production performance and correctness. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M 64-row Q6 selected-down production prefill | 2026-07-26 | clean measured/default `f9a39715b`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; one HIP queue; matrix512/attention128; seven paired pp512 repetitions, exact production-shape transfer, and cached all-family trace | **Superseded exact selected-down production**: gfx1151 uses one local128 64-column x 64-row body for Q6 selected down, reducing its runtime grid from 408 to 332 workgroups per output tile; Q4 selected down and gfx1100 are unchanged. Clean matched pp512 improves the explicit 32-row rollback **489.110 -> 492.640 tok/s (+0.722%)**, wins all seven paired repetitions, and selects token 2930 throughout. Direct all-exact quality transfers unchanged at max KL **0.049542582** and **316/320 (98.75%)** top-1. Cached tracing independently measures **493.509/443.214/351.871 tok/s** at 512/1K/4K; the 23 full-M512 Q6 calls fall **127.888 -> 126.040 ms**. [`artifact`](results/2026-07-26-gfx1151-laguna-q6-down-rows64-production.json). | Superseded; exact Q6 selected-down provenance | Use the parallel-compaction row above for current production performance and correctness. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M exact Q4/Q6 dense-tile production prefill | 2026-07-26 | clean measured/default `3c1e5b452`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; one HIP queue; matrix512/attention128; seven paired pp512 repetitions, all-120-Q4-weight byte-identity transfer, and cached all-family trace | **Superseded exact Q4 shape-policy production**: gfx1151 keeps Q4 shared gate/up at 64x16, selects Q4 shared down 64x32 and layer-0 gate/up 32x32, and retains Q6 16x32; gfx1100 is unchanged. All 120 actual Q4 projections have zero BF16 mismatches versus 64x16. Clean matched pp512 improves **488.692 -> 489.922 tok/s (+0.252%)**, with four of seven paired wins and token 2930. Direct all-exact quality transfers unchanged at max KL **0.049542582** and **316/320 (98.75%)** top-1. Cached tracing independently measures **492.717/442.555/351.533 tok/s** at 512/1K/4K, cuts Q4 dense **43.702 -> 41.936 ms (-4.04%)**, and cuts total dense/shared **54.834 -> 52.989 ms (-3.36%)**. [`artifact`](results/2026-07-26-gfx1151-laguna-q4-pack8-shape-policy-production.json). | Superseded; exact Q4 shape-policy provenance | Use the 64-row Q6 selected-down row above for current production performance and correctness. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M absolute-quality production prefill | 2026-07-26 | clean measured/default `9ae1e4ea6`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; one HIP queue; matrix512/attention128; five pp512 repetitions plus three-repeat ten-prompt all-exact category gate | **Superseded pre-wave-column production**: hipBLASLt heuristic 4 remains selected for every source-F16 shape except the tiny K3072xN72 SWA attention gate, which uses heuristic 2. Clean selector-unset pp512 is **386.552 tok/s** with samples **387.110/386.988/385.748/385.039/386.552**, all token 2930. Direct all-exact quality passes at max KL **0.049542582**, **316/320 (98.75%)** top-1, every category >=96.875%, deterministic repeats, Poolside exact top-1, neutral decode, and exact lifecycle recovery. Natural-prompt production prefill is **53.436 -> 198.461 tok/s (3.714x)** versus all-exact. [`artifact`](results/2026-07-26-gfx1151-laguna-production-absolute-quality.json). | Superseded; absolute-quality schedule provenance | Use the wave-column production row above for current performance and correctness. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M row-vector selected-down production prefill | 2026-07-25 | clean measured/default `69cc0d369`; retained quality `319dfdf3a`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; one HIP queue; matrix512/attention128; five paired repetitions | **Superseded before the absolute-quality audit**: assigning one thread per routed D4 activation row in both Q4 and Q6 down preserves BF16 output byte-for-byte and moves clean paired pp512 **379.827 -> 385.997 tok/s (+1.625%)**, with complete sample separation and token 2930 throughout. Its **0.040724836** KL row was relative to an already approximate shipping control and is not an absolute production-quality result; the 2026-07-26 row above repairs and supersedes that claim. Cached-only tracing remains valid attribution at **388.014/358.319/296.060 tok/s** for the pre-repair schedule. [`artifact`](results/2026-07-25-gfx1151-laguna-down-rowvec-production.json). | Superseded; expert-body and trace attribution only | Use the absolute-quality production row above for current performance and correctness. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M row-vector D8 production prefill | 2026-07-25 | clean measured/default `bd76e452d`; trace attribution repair `dad8c5a8c`; retained quality `319dfdf3a`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; one HIP queue; matrix512/attention128; five paired repetitions | **Superseded retained production row**: assigning one thread per routed Q8 activation row preserves D8/BF16 arithmetic byte-for-byte and moves clean paired pp512 **368.203 -> 379.811 tok/s (+3.153%)**, with complete sample separation and token 2930 throughout; prior published production rises **366.933 -> 379.811 (+3.510%)**. The admitted max KL **0.040724836**, **317/320** top-1, neutral decode, and exact lifecycle carry forward without a new approximation. Cached-only tracing measures **381.448/351.663/292.417 tok/s** at 512/1K/4K, cuts selected gate/up **581.061 -> 537.923 ms (-7.42%)**, and cuts kernel sum **1,369.727 -> 1,326.263 ms (-3.17%)**. [`artifact`](results/2026-07-25-gfx1151-laguna-gate-rowvec-production.json). | Yes, for selector-unset c=1 pp512 and the declared bit-identical expert/quality/lifecycle scope | Superseded by the Q4/Q6 row-vector selected-down production row; retain as gate/up provenance. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M source-qualified SWA production prefill | 2026-07-25 | clean measured/default `36b318ac9`; retained quality `319dfdf3a`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; one HIP queue; matrix512/attention128; five paired repetitions | **Superseded retained production row**: qualifying current/cache K/V loads after visibility preserves qrow4 arithmetic byte-for-byte and moves clean paired pp512 **364.753 -> 366.933 tok/s (+0.598%)**, all token 2930; the prior published production rises **364.839 -> 366.933 (+0.574%)**. The admitted max KL **0.040724836**, **317/320** top-1, neutral decode, and exact lifecycle carry forward without a new approximation. Cached-only tracing measures **369.532/342.620/285.563 tok/s** at 512/1K/4K, cuts SWA **185.603 -> 173.749 ms (-6.39%)**, combined attention **229.181 -> 217.249 ms (-5.21%)**, and kernel sum **11.818 ms**. [`artifact`](results/2026-07-25-gfx1151-laguna-swa-sourcequal-production.json). | Yes, for selector-unset c=1 pp512 and the declared byte-identical attention/quality/lifecycle scope | Superseded by the row-vector D8 production row; retain as attention provenance. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M production prefill | 2026-07-25 | clean selector-unset timing `ab0a8ea3b`; retained quality `319dfdf3a`; promoted defaults `ab0a8ea3b`; fail-closed publication `7b710c09e`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; one HIP queue; matrix512/attention128; three repetitions | **Retained quality-gated production default; 350 target passed**: pp512 moves the prior production **76.226 -> 354.820 tok/s (+365.484%; 4.655x)** with samples **353.421/355.584/354.820**, all token 2930. Selector-unset 1K/4K medians are **322.922/264.245 tok/s**. The complete category lane passes 320 teacher-forced steps at max KL **0.040724836**, **317/320 (99.0625%)** top-1, every category >=96.875%, **2.615x** aggregate natural-prompt prefill, neutral decode, deterministic repeats, Poolside KL **0.0000175125**, and exact lifecycle recovery. The package defaults are D8 128x32 gate/up MMQ, D4 64x32 Q4/Q6 down MMQ, scaled hipBLASLt source-F16, Q4/Q6 64x16 WMMA dense/shared, and online global/SWA attention. Cached-only tracing independently measures **354.763 tok/s** and names every intended family; gate/up is local128/VGPR80/LDS6656B/scratch0. [`publication`](results/2026-07-25-gfx1151-laguna-prefill-350-production.json) · [`quality`](results/2026-07-25-gfx1151-laguna-prefill-350-d8-category.json) · [`trace`](results/2026-07-25-gfx1151-laguna-prefill-350-production-trace.json). | Yes, for selector-unset c=1 pp512 and the declared quality/category/lifecycle scope | Rerun after selected-expert arithmetic/layout, F16 or dense/shared route, matrix/attention policy, model/quant/KV, compiler/runtime, or device/queue policy changes; gate other backends independently. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M LAP-1 direct-T16 MMQ32 natural-shape gate | 2026-07-25 | clean measured `272ee08d5`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; one HIP queue; actual layer-1 K3072/N1024 gate/up weights; natural M32/55/64/122/128/256/512 counts | **Retained explicit primitive; LAP-1 leaf gate passed; runtime default unchanged pending repair**: direct T16 reads the existing resident bytes with no layout transpose or sidecar. Producer-pack-inclusive T16 speedups over retained direct are **1.174/1.528/1.662/2.464/2.502/3.959/5.502x**; M128/M256/M512 are only **4.66%/4.05%/3.02%** behind X8. T16/X8 checksums are exact at every shape; focused tests report 31 passes; cached resources are local128/VGPR48/LDS2048B/scratch0 and device ISA contains `v_dot4_i32_iu8`. [`artifact`](results/2026-07-25-gfx1151-laguna-q4-k-t16-mmq32-retained.json). | Yes for the one-layer natural-routing leaf scope only; no full-model/default claim | Calibrate residual/repair arithmetic, then integrate the sole-resident T16 route and run all-layer performance, category quality, decode, milestone-shape, and lifecycle gates. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M LAP-1 exact X8 decode gate | 2026-07-25 | clean measured `420bf8392`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; one HIP queue; actual layer-1 K3072/N1024 gate/up weights; c1/c2/c4/c8 producer rows with top-10 | **Rejected sole-resident X8 premise; current T16 residency unchanged**: the optimized local128 X8 fallback is BF16-bit exact at every shape and catches T16 at c4/c8, but c1/c2 T16 -> X8 moves **0.157223 -> 0.174663 ms (+11.093%)** and **0.351996 -> 0.362511 ms (+2.987%)**, failing the <=2% decode gate. X8/T16 pair bytes are **905,969,664/931,135,488**; the temporary comparison peaks at **1,837,482,624 bytes** and returns to zero. [`artifact`](results/2026-07-25-gfx1151-laguna-q4-k-x8-exact-decode-rejected.json). | Negative layout decision; no runtime/default claim | Superseded by the retained direct-T16 LAP-1 leaf; do not retry dynamic X8-to-T16 reconstruction or add a complete T16 sidecar. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M LAP-1 X8 MMQ32 live-row screen | 2026-07-24 | clean measured `84c50b205`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; one HIP queue; actual layer-1 K3072/N1024 gate/up weights; natural M32/55/64/122/128/256/512 counts | **Retained explicit prefill control; natural-shape body gate passed; runtime default unchanged**: clamping live rows and bypassing padded-route dot work makes byte-neutral X8 positive at every frozen shape. Producer-pack-inclusive speedups over retained direct are **1.197/1.567/1.704/2.526/2.587/4.092/5.614x**; X8 time falls **18.65–36.45%** versus the prior layout screen. Raw/X8 checksums remain exact; focused tests report 29 passes; cached X8 resources are local128/VGPR48/LDS2048B/scratch0. An all-full synthetic control regresses **8.34%**. The 2026-07-25 exact-decode row supersedes X8 as a resident candidate but preserves it as the MMQ ceiling. [`artifact`](results/2026-07-24-gfx1151-laguna-q4-k-x8-mmq32-live-row-retained.json). | Yes for the exact one-layer natural-routing leaf scope only; no full-model or default claim | Direct T16 now passes the same leaf gate; retain X8 only as the arithmetic ceiling while repair and runtime integration proceed. |
| Radeon Pro W7900, gfx1100 | Poolside Laguna S 2.1 UD-Q2_K_XL matched post-TTFT c1 decode versus llama.cpp HIP | 2026-07-28 | clean hipEngine A/B `a5f0719d4` / `773ab9033`; verified llama.cpp `c0bc8591e` plus declared host-response patch; byte-identical 269-file HIP bundle `a3c0786d...ce40`; BF16 K/V; FA on; one queue | **Accepted protocol-matched ABBA comparison**: over the same 18 natural-greedy train+heldout prompt token streams, h16/h32, and exactly **2,160/4,464 timed transitions per engine**, pooled hipEngine is **64.094/63.431 tok/s** versus llama.cpp HIP **49.290/49.964**, or **+30.03%/+26.95%**. Both hipEngine processes pass Poolside KL **0.000156823**, top-1 100%, exact serial/bulk/repeat trajectories, and teardown. All llama.cpp native timing rows pass; its c0bc response-only patch is source-verified and leaves device code unchanged. This is protocol/storage/timing 1:1, not bit-identical cross-engine reduction order, and it does **not** claim Vulkan is beaten. [`artifact`](results/2026-07-28-gfx1100-laguna-q2-xl-hipengine-vs-llamacpp-hip-matched-abba.json). | Yes, for the declared natural-greedy post-TTFT HIP comparison | Rerun after either engine's model kernels/runtime, llama.cpp source/device bundle, model/prompt, KV/FA policy, ROCm/compiler, or W7900 queue/clock policy changes; keep Vulkan as a separate backend target. |
| Radeon RX 7900 XTX GPU1, gfx1100 | Qwen3.6-35B-A3B UD-Q3_K_M native prefill/decode, c2/c4/c8, and NextN diagnostic | 2026-07-22 | merged branch tip `5c76b408`; exact model/artifact identities and cached GPU1 traces retained | **Accepted direct-Q3 path / rejected MTP diagnostic**: exact fully-bulk prefill reaches **848.543/831.393 tok/s** at 512/4K; graph decode retains **101.216/108.383 tok/s** at c1 and native c8 reaches **207.780/211.177 aggregate tok/s** with exact IDs/full logits and no c>N serial fallback. The blk.40 NextN B1/B2/B3 row is exact but only **0.544x/0.346x/0.271x** AR and remains disabled. | Yes for the declared direct/native-Q3 scopes; no MTP promotion | Rerun after Q3 quant kernels, native-row ownership, MTP transaction/state, model, compiler/runtime, or GPU policy changes. |
| Radeon Pro W7900 + Radeon RX 7900 XTX, gfx1100 | PARO BF16/INT8 KV context capacity and fidelity | 2026-07-13 | clean profile-aware BF16 frontier `5a49b16d`; clean INT8 capacity `d6504544`; clean functional check `2743798f`; clean external-format screen `d0b56364`; current Qwen3.6 packed model fingerprint retained | **Current capacity / correctness outcome**: on the physical 24 GB XTX, the automatic all-768 low-memory prefill profile makes **208 Ki BF16 the recommended safe cap** at **23.623 GiB whole-device peak / 0.361 GiB free**. **220 Ki physically completes** at 23.908 GiB but leaves only **0.076 GiB (~78 MiB)** and is edge-only; a 232 Ki low-profile screen exceeds capacity. Compact 256K INT8 fits at 22.971 GiB tracked but remains unsupported. External-format S1 lowers mean KL to **0.13342**, but the winning Hadamard group32 row rejects 4K/16 at **0.15512 KL** despite **94.12% top-1**. | Current diagnostic table | Rerun after chunk policy, model/runtime, or allocator changes; do not promote 220 Ki without more margin, and require matched-context plus broader task quality before INT8 support. |
| Radeon Pro W7900, gfx1100 | llama.cpp Q8_0 KV protocol/arithmetic isolation | 2026-07-13 | clean harness `a344d32a`; llama.cpp HIP build 9648 / `1ebf790cd`; exact library/model hashes retained; external instrumentation tree disclosed dirty | **Repeated-token pass superseded as representative quality evidence**: native Q8_0/F16 at 4K/16 is **0.000006 KL / 100% top-1** on repeated token 9707 but **0.075654/1.26009 mean/max KL / 94.12% top-1** on exact mixed `mixed_v1`, failing the KL gate; an exact rerun reproduces every row. Mixed K-only and V-only Q8 reach **0.09668** and **0.24322** mean KL, while full Q8 improves through non-additive K/V interaction. The old 128K repeated row remains a saturation control, not broad fidelity evidence. No performance claim. | Current diagnostic table | Require multiple mixed/natural prompt families after cache arithmetic, format, model/build, or protocol changes; do not promote from repeated-token rows. |
| Radeon Pro W7900 + Radeon RX 7900 XTX, gfx1100 | Native GGUF/PARO tail-four Hadamard-group32 mixed KV | 2026-07-15 | clean GGUF closure `c971262f`; therock HIP 7.15; exact Q4_K_M and prompt-suite identities; prior PARO/XTX outcome retained separately | **Quality-safe GGUF explicit diagnostic; no default promotion**: clean GGUF passes all 11 prompts at 512/8 and 4K/16 (**0.0001369/0.009926 mean/max KL, 99.47% aggregate and 94.12% minimum-prompt top-1** at 4K) plus bounded `mixed_v1` 128K/16. Persistent 128K K/V drops **2,689,597,440 -> 2,185,297,920 bytes (-18.75%)** with no persistent BF16 shadow, but production 4K prefill/decode regress **0.67%/0.75%**, 128K decode regresses **3.82%**, and a **1.002 GiB** prefill transient raises allocator high water **24.168 -> 24.700 GiB** despite lowering live owned memory by **0.470 GiB**. Prior PARO quality and 256 Ki capacity blockers remain. Explicit-only; unsupported/default status unchanged. [`clean GGUF gate`](results/2026-07-15-gfx1100-gguf-tail4-hadamard-clean-gate.json) · [`prior split outcome`](results/2026-07-14-gfx1100-native-tail4-hadamard-kv-outcome.json). | Diagnostic link only | Remove the inferred four-layer BF16 prefill transient and optimize long-context group32 attention, then repeat the clean GGUF gate; PARO requires its own quality-safe layout. |
| Radeon Pro W7900, gfx1100 | Qwen3.6 35B model sweep | 2026-07-16 | clean GGUF `28b37356` on therock HIP 7.15; retained PARO `8116c453`; llama.cpp HIP `1ebf790cd` build 9648; Vulkan `263cc04a5` build 9600 | **Accepted current four-column topline**: the GGUF column is the final right-sized 1+3 defaults-only refresh; PARO and llama.cpp columns retain their clean July 12 protocols. All six GGUF shapes have clean provenance, finite/stable IDs, exact Q4_K_M identity, and <=0.658%/0.223% prefill/decode stdev over median. | Yes | Rerun after PARO/GGUF measured paths, graph policy, model, compiler/runtime, llama.cpp builds, or W7900 clock policy changes. |
| Radeon Pro W7900, gfx1100 | GGUF Q4_K_M direct native-c1/c2/c4/c8 graph decode + real OpenAI arbitrary-C concurrency | 2026-07-17 | clean native-c4 graph/equality/profiler/scaling `6f7851f3`/`a05c560b`/`d59d7cf0`, category `799d29b9`, server lifecycle/metrics/accounting `f03957cc`/`b49bc0ef`/`7ab8eb3b`, native-c8 correctness/scaling `bbe6deb0`/`52b0db25`, arbitrary-C state/compaction `1dc7076f`/`be04fa31`, server F1 `77279adf`; TheRock HIP 7.15; exact Q4_K_M/prompt fingerprints; BF16 KV; cached builds | **Retained direct native-c8 and real OpenAI server scaling**: direct one-physical-c8 remains **246.872 aggregate tok/s**, **2.888x c1** and **+34.89%** over c4+c4, with a **748 packed-native / 0 row-local / 0 copy** trace. E3 adds exact C13 eager/graph p512/d128 (**66,560/66,560** hidden rows), sparse cancellation/admission, and nine-move optional compaction with **2/2** graph invalidations. The clean p512/128-output SSE packet retains logical c1/c8/c9/c13/serial-c13 at **25.583/136.122/88.592/111.380/31.708 aggregate tok/s**; grouped C13 is **4.354x** logical-c1 and **3.513x** serial, while all **189** prompt/output rows are exact and the c8→c13 live trace drains ownership to zero. C>8 is explicitly multiple physical buckets, never native c9/c13. [`E3`](results/2026-07-17-gfx1100-gguf-concurrency-e3-arbitrary-c-correctness.json) · [`F1`](results/2026-07-17-gfx1100-gguf-concurrency-f1-server-scaling-closure.json). | Yes, under separate direct graph-step and real SSE cycle-wall scopes | Rerun after physical-group planning, resident stream lifecycle, packed graph/model math, server timing/accounting, prompt/model, compiler/runtime, or device policy changes. |
| Radeon Pro W7900, gfx1100 | GGUF deterministic coding-agent A1, cache off, C1/C4/C8 | 2026-07-21 | clean measured source `44c76674`; system HIP 7.2.53211; exact UD-Q4_K_M/BF16-KV and workload fingerprints; real Uvicorn SSE; one complete warmup + three measurements/configuration | **Retained active-SSE baseline**: small 4K C1/C4/C8 is **16.239/15.995/16.020 exact tok/s**, growing 4K is **15.100/15.231/15.036**, and medium 10,240 is **4.127/4.629/4.339**. Medium C4 is **1.122x C1** and C8 is **0.937x C4**; short/growing scaling is flat. All **702 turns / 17,316 response-owned IDs** pass independent blocking/SSE, strict-tool, variance (<0.91%), and zero-ownership gates. The retained denominator sums measured SSE wave walls; the older first-to-last wall includes inter-turn validation oracles and is diagnostic only. Public timing is buffered tool-ready, not lower-loop TTFT/ITL. GPU0/W7900 is target-exclusive; pinned GPU1/XTX work is a separate device and allowed. Full-vocabulary host logits D2H reaches **1.473 GiB/run**, selecting prefix A/B then native sampling. [`artifact`](results/2026-07-21-w7900-agentic-a1-repeated-baseline.json). | Yes, for exact active-SSE wave goodput and buffered tool-ready latency only | Rerun after prefix/cache, sampled logits/sampler placement, batch/routing policy, tool-envelope streaming, model/quant/KV, compiler/runtime, or GPU0 policy changes. |
| Radeon Pro W7900, gfx1100 | GGUF deterministic coding-agent A2 prefix decision | 2026-07-21 | clean C1 measurement `5d483f36`; prerequisite skip `496dbd60`; lifecycle closure `b8604358`; exact UD-Q4_K_M/BF16-KV and workload fingerprints; active-SSE wave scope | **Rejected; cache-off remains default**: radix versus paired off regresses C1 active-SSE goodput **64.19%/65.63%/26.64%** and worsens buffered tool-ready p50 **181.90%/196.09%/38.81%** for small/growing/medium. Hits are only **0/12, 3/24, 3/18**; all A1 guards fail and growing/medium variance exceeds 5%. C4/C8 is an intentional no-timing skip after the C1 prerequisite fails. Exact IDs/state/KV, lifecycle, cache bounds, and final ownership pass, but cannot override performance. Radix remains explicit diagnostic-only. [`decision`](results/2026-07-21-w7900-agentic-a2-prefix-decision.json). | Negative/default decision; no radix performance promotion | Reconsider only after a model-general LCP/snapshot redesign passes the full C1/C4/C8 suite without prompt-conditioned tuning. |
| Radeon Pro W7900, gfx1100 | GGUF deterministic coding-agent A4 routing/SLO decision | 2026-07-22 | clean balanced screen `fb744f03`; frozen protocol `c445d0ca`; measurement publication `7cc2fee0`; system HIP 7.2.53211; exact UD-Q4_K_M/BF16-KV; real localhost Uvicorn SSE | **Blocked; package routing defaults unchanged**: all **8 candidates x 3 balanced delayed mixed-arrival repetitions** complete (**288 requests / 8,640 response-owned IDs**), but no candidate passes every gate. The exact package control misses TTFT p95 once (**10.983 s > 10 s**); faster alternatives produce **9 late `fixed-0011` p512/d48 mismatches** after 20-24 correct IDs. Native route/final ownership pass, but diagnostic goodput gains up to **+63.81%** cannot override correctness. C1/C2/C4/C8 promotion, strict-tool, and safety timing is intentionally skipped with no inference. [`decision`](results/2026-07-22-w7900-agentic-a4-routing-decision.json). | Negative/default decision; no A4 performance row | Localize and exactness-gate the late p512/d48 state/KV or width transition, then restart all eight frozen candidates. |
| Radeon Pro W7900, gfx1100 | GGUF deterministic coding-agent A5 pressure/soak | 2026-07-22 | clean measured source `414d6d9e`; system HIP 7.2.53211; exact UD-Q4_K_M/BF16-KV; real localhost Uvicorn SSE; cache/native sampler off; `protect_decode:256/burst-1`, zero-ms window | **Accepted bounded correctness/SLO closure; no comparative performance claim**: all nine workloads pass over **122 requests / 2,482 exact observed IDs**: **108 completions / 2,480 completed IDs**, **12 exact retryable overload rejects**, one two-ID disconnect reclaimed in **44.5 ms**, and one distinct deadline. The 80-second soak is **40/40 exact at 11.151 SLO-goodput tok/s**; overload is **20 accepts / 12 rejects at 21.717**. Queue/stream depth stay within **16/1**, KV grows **3 -> 12 pages** then drains, graph/workspace/tracked memory recover, all final owners are zero, and 41 KFD samples show target GPU0 exclusivity. Cache eviction links to the exact A2 p2048/p8192 lifecycle packet because cache off remains the retained default. [`artifact`](results/2026-07-22-w7900-agentic-a5-pressure-soak-closure.json). | Correctness and absolute bounded-SLO evidence only; no tuning/default-speed claim | Rerun after cancellation/deadline, active/queue/stream admission, resident/KV/graph/workspace lifecycle, scheduler defaults, model/quant/KV, compiler/runtime, or device policy changes; a longer soak is required for multi-day reliability claims. |
| Radeon Pro W7900, gfx1100 | GGUF coding-agent A6 broad automatic-tool quality | 2026-07-22 | clean measured source `878d07a9`; system HIP 7.2.53211; exact UD-Q4_K_M/BF16-KV; cache/native sampler off; real localhost blocking OpenAI; 6 committed workloads / 24 turns x 2 repeats; external result/patch/test oracle | **Completed synthetic quality diagnostic; no performance claim**: **10/48 complete turns** pass. Valid-call/correct-tool is **18/48**, exact arguments and independent-oracle pass are **16/48**, safe patch success is **0/6**, and independent test success is **8/8**. Family success is repository **2/16**, general English **4/16**, Japanese **0/8**, and mixed Japanese/English **4/8**. Outcomes are **10 passed / 20 invalid-tool-call / 10 no-tool-call / 6 content-alongside-tool-call / 2 wrong-arguments**. All **24/24** repeat pairs match response IDs/outcomes, all **4,538 IDs** are response-owned, no raw markup leaks, clean provenance/GPU0 exclusivity/final zero ownership pass, and no latency/tok/s/goodput fields exist. [`artifact`](results/2026-07-22-w7900-agentic-a6-broad-quality.json). | Quality diagnostic only; not a public benchmark, cross-model leaderboard, generated-patch execution, or performance row | Expand to independent public task suites and model-generated patch sandboxes before any broad quality claim; prioritize automatic tool envelopes, Japanese argument selection, and safe patch calls from this failure distribution. |
| Radeon Pro W7900, gfx1100 | PARO W4/BF16-KV explicit direct native-c2 selected-batch decode | 2026-07-18 | clean measured `fcb65c47`; TheRock HIP 7.15; exact packed-PARO/prompt fingerprints; cached builds | **Retained for the direct c2 model-step scope**: p512/d128 selected-batch is **121.923 aggregate / 60.962 per-request tok/s**, **+5.09% vs c1 graph** and **+20.81% vs serial c2**. Three fresh processes are <=0.276% stdev/median; primitive, all-layer hidden/Conv/GDN/context/KV, uniform/ragged EOS+cancel immutability, auto-default, and a **10/10 prompt / 330/330 ID** category+heldout gate pass. The fresh L4 trace is `eq_ok`, has **1,306 dispatches**, and records the exact c2 context plus selected fused projection families. Public/OpenAI PARO remains width-1; c4/c8 and gfx1151 are not implied. [`artifact`](results/2026-07-18-gfx1100-paro-g2-selected-batch-c2-retained.json). | Yes, for explicit direct native c2 only | Rerun after PARO c2 math/routing, model/prompt, compiler/runtime, KV policy, or device policy changes; require separate shared-loop and c4/c8 gates before broader production claims. |
| Radeon Pro W7900, gfx1100 | GGUF final architecture-local prefill/decode/memory optimization | 2026-07-16 | clean right-sized rollup `28b37356`; therock HIP 7.15; exact Q4_K_M fingerprint; selector-unset BF16-KV package defaults | **Accepted final gfx1100 GGUF route**: six-shape prefill is **2716.648/3052.541/2953.101/2078.038/1559.878/1037.378 tok/s**, beating llama.cpp HIP by **12.62-30.95%** everywhere and Vulkan from 512-64K; graph decode is **92.833/98.148/100.522/88.240/76.691/62.669 tok/s**, ahead of llama.cpp HIP everywhere and closest to Vulkan at 4K (**-2.47%**). Tracked memory is within **-0.378 to +0.079 GiB** of llama.cpp HIP whole-device readings. All 18 IDs are exact. [`artifact`](results/2026-07-16-gfx1100-gguf-final-optimization-sweep.json). | Yes | Rerun after model/runtime/default-policy, compiler/runtime, or reference-engine changes; decode-to-Vulkan and 128K Vulkan prefill are the concrete residuals. |
| Radeon Pro W7900, gfx1100 | GGUF pp512 request-scoped metadata reuse | 2026-07-15 | clean retained scheduler `e03e5a34`; matched HIP API/kernel traces around the identical source change; system HIP 7.2.53211; exact Q4_K_M fingerprint retained | **Retained scheduler / diagnostic GPF-9C row**: exactly **240 synchronous copies** are removed, matched queue idle falls **27.956 -> 15.163 ms (-45.76%)**, and clean `chain_peer_wave32` pp512 improves **2210.729 -> 2292.186 tok/s (+3.68%)** with stable IDs, unchanged **22.995 GiB** peak, and decode +0.51%. 4K is within -0.44%, but 512 remains **4.98% below** the frozen llama.cpp HIP floor, so exact direct-LDS32 remains production. [`artifact`](results/2026-07-15-gfx1100-gguf-prefill-chunk-metadata-reuse.json). | Diagnostic link only; scheduler code retained | Superseded for the next queue boundary by the compact-WMMA no-read row below; retain as the isolated 240-copy attribution. |
| Radeon Pro W7900, gfx1100 | GGUF pp512 compact-WMMA tight no-read | 2026-07-15 | clean retained gfx1100 default `31c9cdc5`; matched HIP API/kernel traces against `e03e5a34`; system HIP 7.2.53211; exact Q4_K_M fingerprint retained | **Retained gfx1100 scheduler default / diagnostic GPF-9C row**: the tight routing-independent tile bound removes the remaining **40 synchronous D2H copies**, cuts matched queue idle **15.163 -> 11.634 ms (-23.27%)**, and improves clean `chain_peer_wave32` pp512 **2292.186 -> 2334.451 tok/s (+1.84%)** with stable IDs, unchanged **22.995 GiB** peak, and decode within -0.053%. 4K improves +0.70%; pp512 remains **3.23% below** the frozen llama.cpp HIP floor, so exact direct-LDS32 remains production. gfx1151 stays scalar pending its independent gate. [`artifact`](results/2026-07-15-gfx1100-gguf-compact-wmma-tight-no-read.json). | Diagnostic link only; scheduler code retained | Superseded as the final pp512 residual by the no-scratch Conv row below; retain as the isolated 40-copy attribution. |
| Radeon Pro W7900, gfx1100 | GGUF normal-prefill Conv no-scratch exact math | 2026-07-15 | clean retained default `683ddab6`; clean post-profile/floor `c85c2880`; system HIP 7.2.53211; exact Q4_K_M fingerprint retained | **Retained exact default; residual closed**: explicit sequential `v_mul_f32_e32`/`v_add_f32_e32` removes **20 private bytes/thread**, cuts cached pp512 Conv body **8.496 -> 1.894 ms / 30 (-77.71%)**, and improves clean production 512/4K prefill **+1.44%/+1.86%** with exact state/IDs. The clean follow-up puts production exact / peer / llama.cpp kernels at **369.285/203.808/203.301 ms** and GDN at **199.030/20.840/16.522 ms**: the shipped gap is exact GDN. Peer improves clean 512/4K **2334.451/2519.871 -> 2385.677/2585.343 tok/s (+2.19%/+2.60%)**, but 512 remains **1.104% below** its floor, so production is unchanged. [`win`](results/2026-07-15-gfx1100-gguf-conv-no-scratch.json) · [`residual`](results/2026-07-15-gfx1100-gguf-post-conv-residual-attribution.json). | Retained exact-kernel win; peer conclusion superseded by LCP-5A below | Preserve as the post-Conv attribution baseline; the next row closes the peer promotion boundary. |
| Radeon Pro W7900, gfx1100 | GGUF LCP-5A spill-free T16 selected prefill and peer-GDN promotion | 2026-07-15 | clean retained default `487e658c`; system HIP 7.2.53211; exact Q4_K_M/prompt-suite identities retained | **Prefill parity gap closed; gfx1100 default promoted**: rolling only the outer T16 selected-Q4/Q5/Q6 loops removes Q5's 176 private bytes/75 spills, cuts pp512 Q5 **51.009 -> 29.544 ms (-42.08%)**, and moves complete peer kernels/span to **184.513/194.886 ms**, faster than llama.cpp HIP's **203.301/212.236 ms**. The clean 18-prompt gate passes at **0.041737 KL / 445/450 top-1 / -0.103% decode wall**. Selector-unset 512/4K prefill is **2588.231/2757.752 tok/s**, **7.29%/22.29% above** the frozen floors, with stable IDs and **21.670 GiB** tracked peak. [`artifact`](results/2026-07-15-gfx1100-gguf-prefill-lcp5a-spill-free-peer-promotion.json). | Retained default; included in the final `28b37356` six-shape publication above | Rerun after selected-prefill compiler/schedule, peer GDN math, liveness allocation, model, or ROCm changes; keep explicit exact direct-LDS32 rollback for one release. |
| Radeon Pro W7900, gfx1100 | GGUF exact F32-weight cooperative c1 router | 2026-07-14 | clean hipEngine `4c743994` plus persistent-counter default `0ec2a813`; TheRock HIP 7.15; exact Q4_K_M fingerprint retained | **Retained scoped default**: the cooperative fold first improves its complete leaf **17.845 -> 14.666 us (-17.81%)** and clean 4K graph decode **97.234 -> 98.273 tok/s (+1.07%)**. The self-resetting counter then removes 40 reset nodes/token, improves the fused leaf **14.667 -> 10.444 us (-28.79%)**, and cleanly improves 4K graph decode **98.812 -> 100.446 tok/s (+1.65%)**. Every router output bit and all measured IDs/final values are exact; the counters add only eight tracked bytes. | Retained defaults; included in the final `28b37356` six-shape table | Remove temporary cooperative/persistent rollback flags after one release window; rerun after router math, model, compiler/runtime, or graph-policy changes. |
| Radeon Pro W7900, gfx1100 | PARO gfx1151 optimization transfer gate | 2026-07-12 | clean detached hipEngine `255e5aca`; TheRock HIP `7.15.0-0000000`; exact PARO model fingerprint retained | **Retained scoped-default validation / negative chunk decision**: the balanced global-isolation screen is exact at 512/1K/4K. Its 4K/4096-query leg directly validates the merged scoped default with total wall **-0.562%**; 512/1K used 256-query isolation that the final policy intentionally excludes. The gfx1151 linear/MoE-256 profile is rejected at **-7.72%/-8.78%/-6.40% prefill**. | Linked, not a new topline | Rerun after AOTriton/ROCr stream scheduling, PARO chunks, compiler/runtime, or gfx1100 clock policy changes. |
| Radeon Pro W7900, gfx1100 | GGUF NativeSpecCycle N1 fixed-B2 target submission | 2026-07-19 | dirty-tree diagnostic based on `b96fd44c`; system HIP 7.2.53211; exact UD-Q4_K_M/BF16-KV identity; cached launcher and profiler builds | **Superseded diagnostic**: the fixed three-row graph is byte-exact and one C++ submit+sync cuts target forward **29.589 -> 15.493 ms/cycle**, but position-bound recapture regressed its three-cycle screen. The retained reusable B1/B2 row below removes that ownership error. [`artifact`](results/2026-07-19-gfx1100-native-spec-cycle-n1-b2.json). | Historical diagnostic only | Retain as the one-shot rejection control; use the reusable route below for current decisions. |
| Radeon Pro W7900, gfx1100 | GGUF graph AR, exact/default MTP, reusable-native `llama-compat`, and llama.cpp HIP base/MTP | 2026-07-19 | clean retained hipEngine `0d7b86e7`; historical exact suite `202bd2f0`; ROCm 7.2.53211; exact Q4_K_M/prompt fingerprints; llama.cpp HIP `1ebf790cd` build 9648, binary `da974ab…edd2` | **Retained explicit gfx1100 `llama-compat` route; external floor closed**: reusable fixed-address B1/B2 target graphs move the corrected **54.88 -> 122.67 tok/s (+123.52%)** and **18.259 -> 8.186 ms/output (-55.17%)** conservatively. Two clean full-suite processes are **123.33/122.67 tok/s** (0.54% spread), acceptance remains **80.45% / 60.00% accepted-output**, and full/train/heldout plus every category beat their true graph-AR controls. The slower run is **1.2679x** its **96.75 tok/s** AR and **6.26% faster** than llama.cpp's transition-matched **115.44 tok/s** MTP floor. A six-step cached trace has zero measured recaptures and cuts profiler residual **38.41 -> 5.00 ms/step**. `llama-compat` remains explicit/accuracy-traded; exact/default and gfx1151 are unchanged. [`retained route`](results/2026-07-19-w7900-llama-compat-reusable-native-cycle.json) · [`prior baseline`](results/2026-07-19-w7900-hipengine-llama-compat-current-baseline.json) · [`external floor`](results/2026-07-19-w7900-llamacpp-mtp-natural25-refresh.json). | Yes, for the explicit gfx1100 natural24 `llama-compat-native-cycle` protocol | Rerun after target graph metadata/state/KV, MTP proposal/accept/commit, model/prompt, compiler/runtime, or W7900 policy changes; validate gfx1151 and public-generation ownership independently. |
| Radeon Pro W7900, gfx1100 | GGUF NativeSpecCycle N2 device accept and selected-state commit | 2026-07-19 | clean diagnostic hipEngine `8893e06a`; system HIP 7.2.53211; exact UD-Q4_K_M/BF16-KV and prompt identities; cached builds; final-child kernel trace | **Correctness/ownership diagnostic retained; N1 stays the topline**: N2 captures strict acceptance, selected FP32 hidden plus 60 Conv/GDN state buffers, and target cursors behind the reusable B1/B2 submission. The clean full suite is **117.557 tok/s / 1.2723x true AR**, with all **240 IDs / 96 cycles** and **80.45% / 60.00%** acceptance economy identical to N1. A same-tree screen is aggregate-neutral/slightly lower (**117.773 -> 117.235 tok/s, -0.46%**), so N2 does not replace the retained **122.667 tok/s** N1 row. It does cut MTP KV commit **0.135 -> 0.102 ms/output**, target replay/commit **0.059 -> 0.007**, seed upload **0.020 -> 0.002**, and context append **0.0175 -> 0.0001**. The trace records eight accept, chunked-state-commit, and hidden-commit leaves at **5.920/201.187/12.800 us average**. [`artifact`](results/2026-07-19-w7900-llama-compat-native-cycle-n2.json). | Diagnostic link only; no topline promotion | Carry graph-owned verifier rows and N2 state ownership into N3 complete-cycle/public-adapter work; rerun clean N1/N2 if aggregate promotion is reconsidered. |
| Radeon Pro W7900, gfx1100 | GGUF NativeSpecCycle N3 complete cycle/public adapter | 2026-07-19 | clean diagnostic hipEngine `69b70808`; system HIP 7.2.53211; exact UD-Q4_K_M/BF16-KV and prompt identities; cached builds; final-child kernel trace | **Correctness/ownership diagnostic retained; N1 stays the topline**: one public GGUF call owns strict device-chained proposal, N2 verify/accept/selected-state commit, verifier reseed, speculative MTP-KV rollback/repair, and both cursors. The clean full suite is **118.592 tok/s / 1.2858x true AR / 8.497 ms-output**, with all **240 IDs / 96 cycles** and **80.45% / 60.00%** acceptance economy identical to N2. It is +0.88% versus clean N2 but -3.32% versus retained N1, so no topline promotion. The eight-cycle trace contains 16 NextN Q6-X8 stage-1 launches plus eight accept/state/hidden-commit leaves at **606.262/7.165/202.082/12.750 us average**. Proposal submission is superseded by the N3P ownership row below. [`artifact`](results/2026-07-19-w7900-llama-compat-native-cycle-n3.json). | Diagnostic link only; no topline promotion | Use N3P for proposal submission ownership, then migrate the provider-neutral adapter to PARO/DFlash; retain exact unsupported-shape fallback and independent gfx1151 gates. |
| Radeon Pro W7900, gfx1100 | GGUF NativeSpecCycle N3P reusable NextN proposal submission | 2026-07-19 | clean detached diagnostic hipEngine `2395ad33`; system HIP 7.2.53211; exact UD-Q4_K_M/BF16-KV and prompt identities; cached builds; matched HIP API/kernel traces | **Correctness/submission diagnostic retained; N1 stays the topline**: dynamic hidden/embedding/RoPE/context/KV-row inputs feed runner-owned B1/B2 proposal graphs before the unchanged N2 target graph. The clean full suite is **118.183 tok/s / 1.2820x true AR / 8.610 ms-output** and matches clean N3 across all **97 non-timing fields x 96 cycles / 240 IDs**, with unchanged **80.45% / 60.00%** economy. A same-source N3/N3P pair is aggregate-neutral (**116.793 -> 117.589 tok/s**, cycle wall **8.634 -> 8.653 ms-output**), while capture-excluded proposal wall improves **0.964 -> 0.953 ms/output**. Matched eight-cycle tracing changes `hipLaunchKernel` **3273 -> 2731 (-542)**, synchronous `hipMemcpy` **1204 -> 1124 (-80)**, and `hipGraphLaunch` **8 -> 16 (+8)** with the same 22 IDs. This is one proposal plus one target graph launch, not one combined submission. [`artifact`](results/2026-07-19-w7900-llama-compat-native-cycle-n3p.json). | Diagnostic link only; no topline promotion | Validate gfx1151 independently, migrate PARO/DFlash adapters, and combine proposal+target behind one native boundary only if a same-suite wall gate is positive. |
| Radeon Pro W7900, gfx1100 | PARO MTP / DFlash NativeSpecCycle N4 shared target adapter | 2026-07-20 | clean parallel-router `bb9fc742`, selected-commit `43abe82e`; prior N4+ `64f80f83` / merged gate `4e9703be`; system HIP 7.2.53211; current full8192 W4-PARO target + MTP-BF16 sidecar fingerprint; canonical prompt SHA; cached final-child HIP API/kernel profiles | **Selected PARO commit/cursor ownership and exact parallel proposer router retained inside explicit N4; global N4 remains default-off**: three matched arms preserve **10/10 prompts / 240/240 IDs / 214 cycles / 16 accepts**, every train/heldout/category split, and **150/150** expanded native `VERIFY|ACCEPT|COMMIT|UPDATE_CURSORS` records. B1 cycle 7 commits accepted row 1 exactly and cycle 8 remains exact; B2 passes every **60 Conv/GDN + 20 live-KV + 60 scratch-commit + 60 selected-state + 20 selected-KV** comparison. Capture-inclusive on/off/on wall straddles at **16.189/16.277/16.549 ms-cycle**, so no aggregate win is claimed; both candidate arms improve capture-adjusted wall **14.051 -> 13.983/13.992 ms-cycle** across every category. Cached profile wall brackets **16.518/16.322 ms** around **16.413** (mean **+0.007 ms**, neutral) while APIs fall **80.6875 -> 75.6875**, syncs **2 -> 1**, host launches **36.1875 -> 34.1875**, and kernels **1248.5 -> 1247.5** with one graph launch unchanged. [`selected commit`](results/2026-07-20-w7900-paro-mtp-n4plus-selected-commit.json) · [`N4+`](results/2026-07-20-w7900-paro-mtp-n4plus-bound-control.json) · [`provider residuals`](results/2026-07-20-w7900-paro-mtp-n4plus-provider-residuals.json) · [`prior blocker`](results/2026-07-20-w7900-paro-mtp-n4-uncontended-baseline.json) · [`correctness`](results/2026-07-19-w7900-paro-mtp-native-target-graph-n4-correctness.json). | Qualified relative capture-adjusted/API ownership claim only; no AR speedup or global N4 promotion | Selected commit remains default-on only after explicit N4 admission. Exact 256-thread router top8 improves micro-rocprof **94.516 -> 5.395 us/call**, clean complete wall **16.202 -> 15.919/15.951 ms-cycle (-1.75%/-1.55%)**, proposer update **1.222 -> 1.107/1.106 (-9.44%/-9.49%)**, and MTP **65.188 -> 66.303/66.259 tok/s (+1.71%/+1.64%)** with all IDs/accepts/categories unchanged; MTP remains only ~0.592x AR. [`parallel router`](results/2026-07-20-w7900-paro-mtp-n4plus-parallel-router-topk.json) · [`profile`](results/2026-07-20-w7900-paro-mtp-n4plus-proposer-update-residuals.json). Re-profile remaining proposer leaves; keep DFlash/gfx1151 independently gated. |
| Radeon Pro W7900, gfx1100 | PARO/llama.cpp/vLLM concurrency | 2026-07-07 | hipEngine `b4edca09`; same TheRock stack; vLLM `0.22.1rc1.dev499+g470229c37.d20260613` | **Stale diagnostic**: cross-quant and mixed timing scopes; source artifacts set `performance_claim=false`; measured PARO code predates the July concurrency changes | Diagnostic link only | Rerun one timing scope with exact generated-token accounting across all engines |
| Radeon Pro W7900, gfx1100 | Dense 27B DFlash | 2026-06-11 | hipEngine `9faa731c`; ROCm 7.2; artifact records a dirty tree | **Retained under the recorded DFlash gate**, with legacy dirty-source provenance | Yes, qualified | Refresh on a clean tree before changing the public claim |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M native resident scheduler / OpenAI c1-c2 policy suite | 2026-07-23 | clean scheduler core `27d28e1c0`, protect-ttft packet `8648415b`, protect-decode bound `1e4af85ec`, cancellation/fair/protect-decode packets `fe59e77d0`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; one HIP queue; real localhost Uvicorn blocking/SSE | **Accepted exact scheduler ownership and bounded c2 serving; no comparative speedup or policy-default claim**: every physical model transition remains exact c1, while two resident slots admit later work between decode ticks. Across `protect_ttft`/`protect_decode`/`fair`, all c1/c2 blocking rows, SSE oracle reconstructions, routes, and delayed admissions pass with zero blocking mismatches. c2 blocking/SSE is **8.306/8.276**, **8.316/8.263**, and **8.326/8.288 tok/s** respectively. At the declared 0.5-s ITL threshold, c2 SSE SLO runs pass **3/3, 0/3, 2/3**; `protect_ttft` has the lowest TTFT p95 median (**1.241 s**), while `fair` has the highest median exact/SLO goodput (**8.288 tok/s**). `protect_decode` is exact after its finite tick-bound repair but reaches **0.560-s ITL p99**. Dispatched cancellation publishes only after reclaim, bounded overload/recovery and a 20-request host soak pass, explicit S4 reuse remains byte-exact over **277,434,816 KV/span bytes**, and fair/protect-decode c2 GTT returns exactly to baseline after shutdown. [`artifact`](results/2026-07-23-gfx1151-laguna-native-scheduler.json). | Correctness/ownership and absolute c1/c2 server evidence only; no compatibility c2 baseline existed | Rerun after Laguna session/KV ownership, scheduler policy/tick bounds, stop/cancellation/backpressure, model/quant/KV, compiler/runtime, or device policy changes; require broader prompt/decode/load shapes before changing the package scheduler policy. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M compensated-WMMA SWA source-F16 prefill | 2026-07-23 | clean measured route `99ce69780`; promoted default `2ec20c8a0`; compensated shape screen `dae2afaad`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; one HIP queue; 128-row chunks | **Retained quality-gated gfx1151 default**: exact tiled -> compensated WMMA on only the 36 SWA layers from M16 moves weighted ten-prompt prefill **53.388 -> 69.037 tok/s (+29.313%)**, median TTFT **1.529 -> 1.187 s (-22.377%)**, and h16/h32 E2E **+17.004%/+11.663%**; every category prefill/E2E row improves and decode stays neutral. Maximum teacher-forced KL is **0.043888**, suite top-1 is **318/320 (99.375%)**, every category is >=96.875%, Poolside first-token top-1 is exact, repeats/lifecycle pass, and no persistent sidecar is added. Full-attention, M2-15, rows=1, gfx1100, and other backends retain exact paths. [`artifact`](results/2026-07-23-gfx1151-laguna-f16-wmma-comp-swa-retained.json). | Yes, for the declared gfx1151 Q4_K_M 128-row category/quality scope | Rerun after source-F16 residency, compensated reduction/tile schedule, chunk policy, model/quant/KV, compiler/runtime, or device policy changes; gate gfx1100 and other quants independently. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M matrix512/attention128 long-prefill policy | 2026-07-23 | clean measured screen `a6a78bd68`; promoted capability `af59b711e`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; one HIP queue; fixed attention128; two rotating repetitions | **Retained exact gfx1151 default**: M128 -> M512 moves 512/1K/4K prefill **64.997/60.385/49.540 -> 69.069/63.925/51.989 tok/s (+6.266%/+5.862%/+4.943%)** and aggregate median wall **107.516 -> 102.218 s (1.05183x)**. Complete logits, final/post-layer hidden, cursor, and all visible global/SWA K/V plus span bytes match at every row; repeats and lifecycle pass. M512 row/MoE scratch is **411,953,168 bytes**, below the unchanged 2-GiB admission floor. Canonical <=122-token category throughput remains 69.037 tok/s because it does not cross M128; unmeasured backends retain M128. [`artifact`](results/2026-07-23-gfx1151-laguna-matrix-chunk-retained.json). | Yes, for the declared deterministic 512/1K/4K matrix-policy scope | Rerun after matrix/attention chunk policy, row/MoE scratch ownership, source-F16 or selected-expert kernels, KV/span semantics, model/quant/KV, compiler/runtime, or device policy changes; gate other backends independently. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M context-qualified SWA qrow2 prefill | 2026-07-23 | clean measured selector `bb622c9a0`; promoted capability `c0dfb324e`; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; matrix512/attention128; three rotating repetitions | **Retained exact gfx1151 default**: wave32 -> qrow2 moves 512/1K/4K prefill **69.031/63.969/52.017 -> 69.647/64.745/52.557 tok/s (+0.893%/+1.212%/+1.040%)** with complete logits/hidden/KV/span/cursor/repeat/lifecycle equality. The selector applies only to M128 slices with start>=128; the exact ten-prompt category gate is neutral/non-regressive at **0.999652x prefill** and **0.999917/0.999999x h16/h32 E2E** because canonical/partial/verifier rows stay on wave32. [`artifact`](results/2026-07-23-gfx1151-laguna-swa-qrow2-retained.json). | Yes, for the declared deterministic 512/1K/4K selector scope | Rerun after SWA attention math/policy, attention chunk size, KV/span semantics, model/quant/KV, compiler/runtime, or device policy changes; keep independent gates for broader row/context thresholds and other backends. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Poolside Laguna S 2.1 Q4_K_M post-qrow2 attribution / global AOTriton screen | 2026-07-23 | clean profile `afdede428`; cached-only rocprofv3; TheRock HIP 7.15; exact model SHA-256 `7da520c5...5753f`; BF16 KV; matrix512/attention128 | **Accepted diagnostic, no new throughput claim**: post-qrow2 SWA duration falls **9.38%/9.00%/8.99%** at 512/1K/4K and complete kernel sum falls **0.95%/1.24%/1.20%**; global is flat and reaches **16.823 s / 21.62%** at 4K. AOTriton GPU/head-dim-256 controls pass, but all head-dim-128 Laguna V3/V2 attempts return `hipErrorInvalidValue`, so direct adapter work is closed and no selector is added. [`artifact`](results/2026-07-23-gfx1151-laguna-post-qrow2-global-screen.json). | Attribution and mechanical adapter-closure evidence only | Implement and independently gate an in-tree head-dim-128 tiled causal global kernel only; rerun after global/SWA math, attention chunk policy, model/quant/KV, compiler/runtime, or device changes. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Qwen3.6 35B current four-engine model refresh | 2026-07-17 | clean hipEngine `2edbb2ee`; TheRock HIP 7.15; TuneD accelerator-performance; `amd_iommu=off`; current PARO/GGUF routes; llama.cpp HIP/Vulkan five-repetition sweeps | **Accepted current topline through 64K**: PARO and both llama.cpp lanes complete all six shapes; GGUF 512-64K is **1395.379/1481.943/1444.733/1132.215/892.663 prefill** and **52.761/54.658/55.297/45.983/39.388 decode tok/s**, with all 15 IDs exact and <=0.122%/0.028% variance. Across the 11 eligible hipEngine cells, the directional cross-publication average is **+4.60% prefill/+6.20% decode**. GGUF 128K completes warmup plus measured pass 1, then times out and remains blocked. [`artifact`](results/2026-07-17-gfx1151-amd-iommu-off-topline-refresh.json). | Yes through 64K; GGUF 128K blocked | Rerun after model/runtime/default-policy, compiler/runtime, comparison-engine build, or boot IOMMU state changes. A same-commit IOMMU-on reboot is required for causal attribution; IOMMU-off disables XDNA/NPU support. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF Q4_K_M direct native-c1/c2/c4/c8 graph decode + occupancy-adaptive OpenAI concurrency | 2026-07-21 | clean server-ramp `8a8ef481`; clean F3Q direct `f3600c24` (F3P `4bec6b20`; F3O `a960e28a`; F3K `11910c27`; F3I `b03a828f`; F3F `c843f836`; F3E `4cb30543`; F3C `1a753e2e`; F3B `7bb3669b`; F3 baseline `e99b228b`); clean optimized BF16 server `bbe328cb`; pressure-gated prefill `d0f45bc9`; resident telemetry reuse `96598b39`; terminal-state discard `bbe328cb`; terminal packed-owner repair `01589977`; physical-C8 route `c2492bce`; resident graph `33e22d85`; TheRock HIP 7.15; TuneD accelerator-performance; normal HWS/one queue; `amd_iommu=off`; exact Q4_K_M/BF16-KV | **Retained direct and ramp-optimized real-server physical C8**: direct p512/d128 c1/c2/c4/c8 is now **50.335/78.552/108.050/159.487 tok/s** (**1.561x/2.147x/3.169x C1**). F3Q caches 24/128 old FP32 state rows across the indexed-GDN barrier only at gfx1151 physical C8, improving clean C8 **158.804 -> 159.487 (+0.430%)** and **+4.439%** versus F3K at 0.083% variance with exact trajectories. The exhaustive cache sweep selects the four-block/CU 15,360 B LDS point: leaf **204.996 -> 197.844 us (-3.49%)**; trace reports 56 VGPR/zero scratch and GDN **3.770 -> 3.468 ms (-8.02%)** over 30 launches. Cache-120 is removed because one-block occupancy regresses serving. F3P/F3O remain underneath and move paged context **315.073 -> 134.890 us (-57.19%)**; F3I/F3F/F3E/F3C remain underneath; gfx1100 and C1/C2/C4 are unchanged. The combined hidden/state gate is **320/320 exact**. The pre-ramp all-width real-Uvicorn packet is **44.321/59.783/75.580/86.185 blocking** and **42.147/59.102/73.971/84.196 exact SSE**, with delayed C8 **67.788 tok/s**. Against the prior exact F3P packet, F3Q then moved blocking **91.830 -> 91.562 (-0.29%, noisy)**, exact SSE **87.772 -> 88.597 (+0.94%)**, and delayed **69.777 -> 70.414 (+0.91%)**. Bounded cold-cohort admission priority now moves the tracked-clean p512/d128 C8 packet to **101.627 blocking (+10.99%) / 99.018 exact SSE (+11.76%) / 74.450 delayed (+5.73%)** versus F3Q while C1 remains **44.353/43.178 blocking/SSE**. The detached-clean `continuous_fixed` safety gate is **12/12 exact**, **47.121 SLO-goodput tok/s**, and ITL p99 **0.2991 s < 0.5 s**. Direct and server claims are tracked-clean; final graph/KV/workspace ownership is zero. [`bounded admission priority`](results/2026-07-21-gfx1151-gguf-bounded-cold-cohort-admission-retained.json) · [`F3Q GDN state cache 24`](results/2026-07-20-gfx1151-gguf-gdn-shared-statecache24-c8-retained.json) · [`F3P paged-attention value vector 2`](results/2026-07-20-gfx1151-gguf-paged-attn-value-vector2-c8-retained.json) · [`F3O paged-attention token offsets`](results/2026-07-20-gfx1151-gguf-paged-attn-token-offsets-c8-retained.json) · [`F3K lm-head 5+3`](results/2026-07-20-gfx1151-gguf-q6t16-lm-head-chunk5-c8-retained.json) · [`F3I Q8 pair col8`](results/2026-07-20-gfx1151-gguf-q8t16-pair-col8-c8-retained.json) · [`F3F Q6 selected-down reuse`](results/2026-07-20-gfx1151-gguf-q6t16-selected-down-pairreuse-c8-retained.json) · [`F3E Q5 selected-down reuse`](results/2026-07-20-gfx1151-gguf-q5t16-selected-down-pairreuse-c8-retained.json) · [`F3C selected-expert reuse`](results/2026-07-20-gfx1151-gguf-selected-expert-pairreuse-c8-retained.json) · [`F3B direct/profile/server`](results/2026-07-20-gfx1151-gguf-q8t16-pair-rowtile-c8-retained.json) · [`optimized server`](results/2026-07-20-gfx1151-gguf-server-concurrency-optimized.json) · [`prior refresh`](results/2026-07-20-gfx1151-gguf-server-concurrency-refresh.json) · [`F3 direct`](results/2026-07-19-gfx1151-gguf-f3-singleton-gdn-retained.json) · [`E3`](results/2026-07-18-gfx1151-gguf-concurrency-e3-arbitrary-c-correctness.json). | Yes, under separate direct graph-step and complete localhost server-wall scopes; external C>1 remains blocked | Rerun after Q8T16 pair scheduling, occupancy/prefill policy, resident graph/state ownership, server timing/accounting, model/quant/KV, compiler/runtime, or device/boot policy changes; the exact dense/LM-head/short-attention/state lanes are closed; the simultaneous p512 C8 ramp is closed; further serving work requires a new measured bottleneck at staggered or longer-prompt shapes. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF Q4_K_M / mirrored INT8-KV continuous concurrency | 2026-07-20 | clean final `29467874`; implementation `fb926d8e`; quality harness `24d4ad42`; corrected 5 ms generation window / fair 256-token prefill; TheRock HIP 7.15; TuneD accelerator-performance; normal HWS/one queue; `amd_iommu=off`; explicit `int8_per_token_head` + FP16 per-token/head scales; bounded BF16 mirrors; real Uvicorn | **Accepted corrected-protocol explicit short-context C1/C2/C4/C8 serving; no default, memory-saving, or external claim**: blocking is **44.225/60.598/74.631/83.408 tok/s**, exact SSE is **42.759/55.128/71.284/81.140**, and delayed C8 is **65.034**. All **117/117** server rows are exact with zero serial/resident fallback; the full 11-prompt/99-position gate remains **KL=0 / 100% top-1** and ownership drains 65/65. Against the clean corrected-window single-run baseline, C1/C2/C4 stay within **0.56%** while C8 improves **75.513 -> 83.408 (+10.46%)** and delayed C8 **62.188 -> 65.034 (+4.58%)**. The route remains eager because the packed graph ABI is BF16-only, and bounded BF16 attention mirrors mean it does not save KV memory. [`refresh`](results/2026-07-20-gfx1151-gguf-mirrored-int8-concurrency-refresh.json) · [`prior quality/lifecycle`](results/2026-07-19-gfx1151-gguf-mirrored-int8-continuous-concurrency.json). | Yes for exact own-engine explicit short-context scaling/lifecycle only | Rerun after packed/dynamic KV layout, mirror policy, context admission, server scheduling/SLO, model/quant, compiler/runtime, or device/boot changes; remove mirrors and pass independent quality before any memory-saving or direct-INT8 claim, and gate tail4 separately. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | PARO W4 and GGUF Q4_K_M matched real-server concurrency versus llama.cpp HIP/Vulkan | 2026-07-20 | clean GGUF `fcc586ed`; clean PARO `473ec348` with route re-evaluation `960d4a98`; llama.cpp HIP build 9648 / `1ebf790cd` with dirty source disclosed and binary/library hashes pinned; clean Vulkan build 9641 / `6e9007ae`; TheRock HIP 7.15; TuneD accelerator-performance; normal HWS/one queue; `amd_iommu=off`; p512/d128 greedy/no-MTP; real localhost servers | **Qualified unified C1/C2/C4/C8 refresh; external C>N remains diagnostic.** Blocking aggregate tok/s is PARO **46.742/51.485/59.606/60.576**, GGUF **44.511/60.073/75.642/86.601**, llama.cpp HIP raw **46.279/66.725/85.404/103.724**, and Vulkan raw **53.696/75.704/94.520/109.370**. C8/C1 scaling is **1.296x/1.946x/2.241x/2.037x** respectively. Both hipEngine lanes pass all **117/117** oracle/warmup/blocking/SSE/delayed rows; exact delayed C8 is **20.158 PARO / 67.898 GGUF tok/s**. Same-GGUF C1 is the only eligible external comparison: GGUF trails HIP **3.82% blocking / 7.49% SSE** and Vulkan **17.11% / 20.50%**. llama.cpp C2/C4/C8 measured blocking exact fractions are only **50%/33%/37.5% HIP** and **50%/25%/25% Vulkan**, so the high-C rates and GGUF gaps (**-16.51%/-20.82% at raw C8**) are diagnostics, not retained wins/losses. GGUF C8 wall decomposes to **3.404 s recorded prefill + 7.685 s retained direct model wall + 0.735 s residual**; **83.6%** of its wall gap to HIP appears in TTFT. PARO is cross-quant and its direct C4/C8 already plateaus at **100.209/99.943 tok/s**. [`four-lane refresh`](results/2026-07-20-gfx1151-four-lane-server-concurrency-refresh.json) · [`optimized GGUF`](results/2026-07-20-gfx1151-gguf-server-concurrency-optimized.json) · [`PARO direct`](results/2026-07-18-gfx1151-paro-g3-native-c248-direct-retained.json). | Yes for exact hipEngine own-engine scaling and same-GGUF C1; PARO cross-quant and llama.cpp C>N are explicitly non-comparative | First batch/coalesce concurrent prefill and repair PARO delayed admission; then replace row-parallel dense/selected-MoE GEMV with weight-reusing MMQ/WMMA/grouped-expert kernels. Collect gfx1151 counters before calling C8 arithmetic-compute-bound. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF Q4_K_M production OpenAI load/SLO gate | 2026-07-19 | clean package-default `ecaf14d5` (scheduler row acknowledgement `1e104111`); TheRock HIP 7.15; TuneD accelerator-performance; normal HWS/one HIP queue; `amd_iommu=off`; exact Q4_K_M fingerprint; BF16 KV; real localhost Uvicorn SSE | **Accepted F4 production workload closure**: the four-candidate frozen mixed sweep selects scoped `fair:256` at **46.527 exact SLO-goodput tok/s**, **+6.73%** over passing `fair:128`. Static c1/c8, ragged, fixed, Poisson, cancellation, overload, recovery, and 60-second soak all pass declared **10/10/0.5/30 s** queue-p99/TTFT-p95/ITL-p99/end-to-end-p95 SLOs. Cancellation preserves **6/6 exact normal neighbors + one post-token disconnect + one distinct 408 timeout**; overload is **16 exact completions + 16 exact 429 rejects**; soak is **120/120 at 43.314 goodput tok/s**. Bounded queues, route/counter accounting, memory recovery, and scheduler/runner/KV ownership all pass. [`artifact`](results/2026-07-19-gfx1151-gguf-f4-production-load-slo.json). | Yes, for greedy exact Q4_K_M/BF16-KV real-Uvicorn serving under the declared SLO/rate protocol | Rerun after scheduler policy, row cancellation/deadline ownership, resident routing/backpressure, server accounting, model/quant/KV, compiler/runtime, or device/boot policy changes; sampled/API paths, prefix reuse, long-context pressure, and external comparisons remain separate gates. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF Q4_K_M long-context concurrency and device-KV pressure | 2026-07-19 | clean measured `7c243ed1`; focused live-pool artifact repair `c142730a`; TheRock HIP 7.15; TuneD accelerator-performance; normal HWS/one HIP queue; `amd_iommu=off`; exact Q4_K_M fingerprint; BF16 KV; real localhost Uvicorn SSE | **Accepted one-pass long-context server closure through 64K**: c2 1K/4K/32K/64K is **17.638/8.095/1.037/0.416 exact generated tok/s**; mixed 1K/4K/32K is **3/3 exact at 2.585 tok/s**. Across 15 requests, all **642 returned tokens** are exact; the sole non-completion is the required retryable 4K `429 engine_busy` while a 32K row survives at **17 requested / 134 current / 134 capacity pages**. The 64K phase reaches a bounded **519 pages / 2.534 GiB device KV** and **36.017 GiB sampled whole-device peak**. Shrink/regrow returns to **5/5 free, zero refs/pins**, uses disjoint logical IDs **5..133 -> 134..262**, and records **2 captures / 256 replays / 2 invalidations** with zero final ownership. The raw packet's only failed verdict was an empty post-teardown pool query; the retained artifact preserves that original result and the focused pure-host re-evaluation. Non-BF16 continuous concurrency is not claimed. [`artifact`](results/2026-07-19-gfx1151-gguf-long-context-memory-pressure.json). | Yes, for exact greedy Q4_K_M/BF16-KV real-Uvicorn serving through 64K; one-pass rates are not variance-qualified external comparisons | Rerun after long-context attention, dynamic KV ownership/admission, graph retirement, scheduler policy, model/quant/KV, compiler/runtime, or device/boot changes; gfx1100 transfer and non-BF16 c>N remain separate gates. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF Q4_K_M sampled OpenAI API paths | 2026-07-19 | clean measured `7871c088`; sampled resident-model path `ae4a3159`; reusable gate `d38f2f25`; TheRock HIP 7.15; TuneD accelerator-performance; normal HWS/one HIP queue; `amd_iommu=off`; exact Q4_K_M fingerprint; BF16 KV; real localhost Uvicorn | **Accepted F5 correctness/serving-path closure; no throughput claim**: two repeated sampled blocking c4 waves and two repeated SSE c4 waves cover **16 rows** with deterministic text, exact blocking generated-token IDs, finite selected/top logprobs, and exact blocking-to-SSE reconstruction. Repeated `n=3` adds **6 exact choices**. Request EOS, stop, strict tool forcing, and fail-closed structured output pass. The whole packet records **26 host-sampler requests, 88 packed model steps, physical c2/c4 execution, zero serial-decode fallback, and zero resident fallback**; memory and all scheduler/model-runner/KV ownership drain. `host_sampling_required` discloses sampler placement without mislabelling packed model work as serial fallback. [`artifact`](results/2026-07-19-gfx1151-gguf-f5-sampled-openai-api.json). | Correctness and serving-path evidence only | Rerun after GGUF sampled logits/sampler math, API token/logprob/finish semantics, owner routing, model/quant/KV, compiler/runtime, or device/boot policy changes; gfx1100 transfer, prefix reuse, long-context/non-BF16 pressure, and external comparisons remain separate gates. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF Q4_K_M active/completed prefix reuse | 2026-07-19 | clean active runtime/correctness `f4c826e2`; clean active economics `05dda75b`; completed snapshot runtime `13d8beaf`; clean completed correctness `74251bdb`; clean completed economics `f0a63059`; TheRock HIP 7.15; TuneD accelerator-performance; normal HWS/one HIP queue; `amd_iommu=off`; exact Q4_K_M fingerprint; BF16 KV; p256+s1 greedy; fixed-capacity six-page pool | **Retained narrow opt-in F5 prefix slice**: one live exact-current 256-token source shares its page with a non-empty suffix continuation. Output, all **60 Conv/GDN + 20 live K/V** components, source-first lifecycle, final survivor state, and four teacher-forced transitions are byte-exact (`KL=0`, top-1 **100%**). One discarded warmup plus three alternating matched pairs moves synchronized continuation admission-to-first-token median **249.269 -> 21.188 ms (11.765x, -91.50%)**, with **3/3 exact 256-token hits**, no fallback, and live pages **4 -> 3** (**5,242,880 bytes**). Paired HIP-current deltas are **0/203,423,744/0 bytes**, median zero, so no physical-current reduction is claimed. A separate completed-source gate resets/unbinds the source before admission, reports `prefix_snapshot_hit=true`, restores all **66,846,720** hybrid-state bytes, keeps boundary/output/all state/KV/four teacher-forced transitions byte-exact (`KL=0`, top-1 **100%**), and drains cache refs through **1->1->2->1->0**. Its clean matched packet moves continuation TTFT **249.446 -> 22.013 ms (11.332x, -91.18%)** with 3/3 exact snapshot hits. Unique continuation pages stay **2 -> 2**; the retained snapshot + page cost is exactly **72,089,600 bytes**, tracked allocation is **+66,846,720 bytes**, and paired HIP current is **+62,914,560 bytes**. Sampled reuse, packed-suffix byte identity, gfx1100 transfer, broader boundary/LRU pressure, and default-on promotion remain open. [`active correctness`](results/2026-07-19-gfx1151-gguf-active-prefix-reuse-correctness.json) · [`active economics`](results/2026-07-19-gfx1151-gguf-active-prefix-reuse-economics.json) · [`completed correctness`](results/2026-07-19-gfx1151-gguf-completed-prefix-reuse-correctness.json) · [`completed economics`](results/2026-07-19-gfx1151-gguf-completed-prefix-reuse-economics.json). | Performance/correctness retained for explicit active-current and completed-source greedy p256+s1 `radix` reuse on gfx1151, under separate memory contracts | Rerun after cache/state snapshot lifecycle, KV pool/block layout, suffix arithmetic, model/quant/KV, compiler/runtime, or device/boot policy changes; require separate general-cache, sampled, gfx1100, and default-promotion gates. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | PARO W4/BF16-KV explicit direct native-c2/c4/c8 selected-batch decode | 2026-07-18 | clean equivalent tree measured `e175e28f`, pushed `8c8cc15e`; TheRock HIP 7.15; TuneD accelerator-performance; normal HWS with one HIP hardware queue; `amd_iommu=off`; exact packed-PARO/canonical prompt-suite fingerprints; cached builds | **Retained for explicit direct c2/c4/c8 model steps**: p512/d128 is **79.237/100.209/99.943 aggregate tok/s**, with c4/c8 **+41.52%/+41.14% vs c1**. Three fresh processes per width have <=**0.054% stdev/median** and all **5,754/5,754 IDs** exact. True physical c4/c8 use 40/40 selected-batch layers and zero fallback/row chunks; primitive, all-layer hidden/Conv/GDN/context/KV, sparse c8->c1 cancel/EOS/ragged immutability, both ten-prompt category+heldout gates, and the cached **4,644-dispatch** c8 trace pass. c8 is -0.265% aggregate vs c4 but -0.183% step time vs c4+c4. G5 attaches these widths to the shared owner and promotes the gfx1151 package default. [`artifact`](results/2026-07-18-gfx1151-paro-g3-native-c248-direct-retained.json). | Yes, for explicit direct native c2/c4/c8; linked to the separate G5 server claim | Rerun direct gates after PARO math/routing, model/prompt, compiler/runtime, KV policy, or device/boot changes. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | PARO W4/BF16-KV resident OpenAI server scaling | 2026-07-18 | clean blocking measurement `c0e3318c`, clean SSE measurement `ee8b417e`, measured runtime-candidate archive `858488bf`; TheRock HIP 7.15; TuneD accelerator-performance; normal HWS/one HIP queue; `amd_iommu=off`; exact packed-PARO revision | **Retained real OpenAI c2/c4/c8 scaling and gfx1151 package default**: p512/d128 blocking F1 c1/c2/c4/c8 is **47.124/51.962/60.323/61.253 aggregate tok/s** (**1.103x/1.280x/1.300x c1**) with <=0.994% variance, **68/68** exact rows, exact delayed admission, and **18.373/18.840/19.461/20.594 GiB** GTT peaks. Real FastAPI SSE c1/c2/c4/c8/serial-c8 is **36.327/38.666/42.471/41.487/35.633**, all **100/100** rows exact, c8 **1.164x** serial, and live c4->c8 **38.191**. A separate c8 stress adds **72/72** exact rows; a no-flag c4 gate observes native widths 2/4 with 4/4 exact and no fallback. [`F1`](results/2026-07-18-gfx1151-paro-g5-f1-server-scaling.json) · [`SSE`](results/2026-07-18-gfx1151-paro-g5-sse-server-scaling.json) · [`repeat`](results/2026-07-18-gfx1151-paro-g5-c8-sse-repeatability.json) · [`default`](results/2026-07-18-gfx1151-paro-g5-default-openai-c4.json). | Yes, under separate blocking localhost-HTTP and in-process FastAPI-SSE walls | Rerun after owner/session/reset/accounting, width profile, model/prompt, compiler/runtime, KV policy, or device/boot changes; close gfx1100 owner c4/c8 independently. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | PARO exact c1 prefill recovery | 2026-07-12 | clean control `240c5daf` and candidate `9944e481`; TheRock HIP `7.15.0-0000000`; TuneD accelerator-performance; exact PARO model fingerprint retained | **Retained historical promotion gate**: exact linear/MoE 256-row architecture profile improves all six prefill shapes by **14.35%-51.11%**, leaves decode within **-0.25%..+0.26%**, and matches final hidden plus all Conv/GDN/KV state at 512/4K/128K. The July 17 current-default sweep supersedes its public numeric column. | Superseded as topline; retained promotion evidence | Rerun after PARO prefill chunk/staging/math, compiler, model, prompt, or tuned/clock policy changes; validate separately on gfx1100 before transfer. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | PARO 4K-128K AOTriton queue isolation | 2026-07-12 | clean same-commit control/candidate `01e2cec5`; TheRock HIP `7.15.0-0000000`; TuneD accelerator-performance; exact PARO model fingerprint retained | **Retained historical scoped gate**: event-linked isolated AOTriton queue improves matched prefill by **13.32%-23.03%**, leaves decode within **-0.16%..+0.12%**, holds tracked peak unchanged, and matches final hidden plus all 30 Conv/GDN and 10 K/V families at every retained shape. The July 17 current-default sweep supersedes its public numeric column. | Superseded as topline; retained scoped evidence | Validate separately on gfx1100 before transfer; 512/1K remain on the proven-safe caller-stream route. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF eager token/state oracle | 2026-07-12 | clean detached hipEngine `3ce60e56`; TheRock HIP `7.15.0-0000000`; exact Q4_K_M fingerprint and llama binary hashes retained | **Accepted correctness-only gate**: the repeated external and production token stream matches; four hidden/layer/30-Conv-GDN/10-KV transitions are finite and byte-exact. `performance_claim=false`. | Diagnostic link only | Rerun after eager math/state/KV, model, compiler/runtime, or device changes. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF fused/chain GDN prefill correctness and default selection | 2026-07-11 | correctness at clean tracked `332f01f8`; clean performance worktree `ad773eba`; TheRock HIP `7.13.60980-c76140fa27`; exact Q4_K_M fingerprint retained | **Accepted correctness / retained negative performance decision**: exact chain passes 6/6 state cases but is +5.19%/+6.70% slower in balanced 512/4K walls. Fused remains default. | Diagnostic link only | Rerun after GDN math/scheduler/chunk changes; do not retry unchanged split scheduling. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF GPF-2 register-resident GDN diagnostics | 2026-07-13 | clean tree performance `31d4204d`, clean tree trajectory gate `2670ed04`, exact ordered candidate based at `cf3e8250`; TheRock HIP `7.15.0-0000000`; TuneD accelerator-performance; exact Q4_K_M fingerprint retained | **Both default candidates rejected**: relaxed tree balanced 512/4K prefill improves **422.281 -> 956.765 tok/s (2.266x)** and **410.534 -> 844.847 tok/s (2.058x)**, but only **3/10** natural prompts preserve the complete fused 128-step trajectory. Exact ordered residency preserves byte identity but regresses 512/1K/4K by **12.98%/14.58%/13.50%**. `auto` remains fused. | Diagnostic link only | Test scalar-exact value columns with recurrent state resident in a 32/64-column LDS tile; require exact natural trajectories before any default/topline change. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF GPF-2D scalar-exact LDS-resident GDN | 2026-07-13 | clean candidate `a6f389d2`, promoted default `5f082783`; TheRock HIP `7.15.0-0000000`; TuneD accelerator-performance; exact Q4_K_M fingerprint retained | **Promoted gfx1151-scoped default**: six-case matrix and 250/250 natural logits are exact; balanced 512/4K prefill improves **420.959 -> 753.891 tok/s (1.791x)** and **408.359 -> 687.831 tok/s (1.684x)**; decode is +0.023%. The clean automatic max-context stress gate records **751.993/804.420/688.545/589.866/504.730/372.892 tok/s** across six shapes with stable IDs. | Superseded within current GGUF rollup | Keep gfx1100 fused until an independent transfer gate. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF GPF-3A Q4T16 shared-activation prefill | 2026-07-13 | clean A/B `95d484df`, clean automatic confirmation `431fe1e4`; TheRock HIP `7.15.0-0000000`; TuneD accelerator-performance; exact Q4_K_M fingerprint retained | **Promoted gfx1151-scoped default**: BF16/FP16 fixtures and 248,320 full-model logits/shape are byte-exact; balanced 512/1K/4K prefill improves **747.764/804.150/687.676 -> 771.027/823.624/701.042 tok/s (+3.11%/+2.42%/+1.94%)**; every measured 128-step trajectory matches and aggregate decode is -0.0031%. Selector-unset focus medians reproduce **774.653/823.149/701.389 tok/s** with stable IDs. | Included in current GGUF rollup | Keep gfx1100 baseline until an independent transfer gate. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF GPF-2E compact-scale/direct-conv GDN and right-sized rollup | 2026-07-13 | clean exact matrix `c3a065ee`, balanced A/B `ffbcc4d9`, trajectory/decode `5501aeb9`, automatic focus `b8949477`, clean measured sweep `28b45d38`; TheRock HIP `7.15.0-0000000`; TuneD accelerator-performance; exact Q4_K_M fingerprint retained | **Promoted and published on gfx1151**: clean 512/1K/4K A/B is +6.01%/+7.74%/+6.24%; the final 1+3 right-sized prefill row is **819.641/893.266/752.308/640.096/540.850/387.334 tok/s** with <=0.132% stdev/median. Six-case state and 250/250 natural logits are exact. The log-recovered 128K row discloses that the interrupted process did not serialize IDs and links those stronger independent gates. | Superseded by LCP-1/LCP-D1 | Keep gfx1100 fused; investigate later-pass 128K lifecycle no-progress separately from the calibrated performance protocol. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF GPF-5A two-wave dense Q8T16 prefill | 2026-07-14 | clean candidate `4a1fff53`, clean promoted sweep `e9baf563`, final scoped policy `6418b278`; TheRock HIP `7.15.0-0000000`; TuneD accelerator-performance; exact Q4_K_M fingerprint retained | **Promoted through 64K and published**: the clean kernel is byte-exact, uses 80 VGPR/1 KiB LDS/zero scratch, and improves the prior right-sized 512/1K/4K/32K/64K prefill row to **889.904/919.598/762.940/648.948/546.296 tok/s (+1.01% to +8.57%)** with unchanged memory. Stable same-commit 128K rejects two-wave at **-2.59%**, so package policy restores production above 65,536 tokens and carries forward the accepted **387.334 tok/s** row. | Superseded by LCP-1/LCP-D1 | Keep the env rollback and 64K ceiling for one release; validate independently on gfx1100; treat later-pass 128K no-progress as lifecycle diagnosis, not extra timing repetitions. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF LCP-1 tiled convolution and LCP-D1 long-split decode reduction | 2026-07-14 | clean focus/promotion `3ff8e2d7`/`631498dd`, clean reducer `71e61524`, final six-shape sweep `71e61524`; TheRock HIP `7.15.0-0000000`; TuneD accelerator-performance; exact Q4_K_M fingerprint retained | **Promoted and published on gfx1151**: LCP-1's clean 4K body falls **954.134 -> 49.790 ms** and its 512/4K focus is **+1.73%/+22.91%** with 82/82 exact state parts. LCP-D1 cuts the clean 128K reducer **234.714 -> 196.466 us/call (-16.30%)**. Final right-sized prefill is **906.979/929.724/946.366/778.371/636.330/433.811 tok/s** and graph decode is **49.061/51.569/52.432/43.543/37.562/28.047 tok/s**; all 18 IDs are exact, memory unchanged, and variance <=0.140%. | Superseded by LCP-2A prefill promotion | Keep LCP-1's production fallback for one release and validate gfx1100 independently; continue decode only from the measured grouped-GQA context or dense-Q8 residual. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF LCP-2A compiler-cacheable exact GDN | 2026-07-14 | clean detached candidate `53928aaf`; TheRock HIP `7.15.0-0000000`; TuneD accelerator-performance; exact Q4_K_M fingerprint retained | **Promoted on gfx1151 and included in the current 512-64K production refresh**: six-case state and 250/250 natural transitions are exact. Balanced 512/1K/4K prefill improves **900.814/940.736/941.462 -> 1213.912/1285.266/1285.888 tok/s (+34.76%/+36.63%/+36.58%)**; decode is +0.021%. The kernel uses 32 VGPR/16 KiB LDS/zero scratch. | Superseded within targeted prefill gate by LCP-3 | Keep volatile GPF-2E rollback for one release; validate gfx1100 independently. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF LCP-3 four-wave dense Q8T16 prefill | 2026-07-15 | clean detached candidate `d34476da`; TheRock HIP `7.15.0-0000000`; TuneD accelerator-performance; exact Q4_K_M fingerprint retained | **Promoted on gfx1151 through 64K and included in the current production refresh**: clean 512/4K full-model capture is **83/83 exact**. Five balanced pairs improve automatic GPF-5A **1214.510 -> 1220.993 tok/s (+0.53%)** and **1269.030 -> 1288.986 tok/s (+1.57%)**; all 20 timed IDs are `9707`. The named kernel uses 128 threads, 80 VGPR, 1 KiB LDS, and zero scratch. | Superseded within targeted prefill gate by LCP-4A | Keep two-wave then production as rollback paths; validate gfx1100 independently. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF LCP-4A exact F32 router launch geometry | 2026-07-15 | clean detached candidate `3ef55ad4`; TheRock HIP `7.15.0-0000000`; TuneD accelerator-performance; exact Q4_K_M fingerprint retained | **Promoted on gfx1151 and included in the current 512-64K production refresh**: clean 512/4K full-model state is **83/83 exact**. Five balanced pairs improve **1218.536 -> 1252.147 tok/s (+2.76%)** and **1290.923 -> 1333.229 tok/s (+3.28%)**. Clean 512/128 graph decode is exact and **48.987 -> 49.021 tok/s (+0.071%)**. Trace confirms 256 threads, 32 VGPR, and zero scratch. | Superseded within targeted prefill gate by LCP-4B | The 4K refresh is complete; it rejects risky logits+top-k fusion in favor of exact select geometry. Validate gfx1100 independently. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | HIP one-hardware-queue lifecycle stabilization | 2026-07-15 | clean current production `4d0aa281`; TheRock HIP `7.15.0-0000000`; MES `0x88`, MES KIQ `0x6f`; TuneD accelerator-performance; exact Q4_K_M fingerprint retained | **Retained as a risk-reducing gfx1151 process default; not lifecycle-safe for repeated 128K**: default ROCm queue policy enters a reproducible first-warmup 128K stall at 100%/2.9 GHz but only 41-43 W, with four identical host stacks in synchronous metadata H2D and no kernel fault. Changing only `GPU_MAX_HW_QUEUES=1` completes 128K warmup+3 at **499.755 warmup** and **500.210/500.873/500.687 tok/s measured**, all IDs `9707`. Clean 512/4K A/B is non-regressive at **+0.35%/+0.46% prefill** and **+0.066%/+0.072% decode**. | Yes, stability/process-policy gate | Preserve explicit `GPU_MAX_HW_QUEUES` overrides; `=4` restores ROCm's documented default. Current production later reproduces the stall under one queue, so 128K remains blocked. Remove after a fixed gfx11 firmware/runtime passes the same 128K gate. Upstream evidence: [initial ROCm#5107 comment](https://github.com/ROCm/ROCm/issues/5107#issuecomment-4976739824) and [follow-up](https://github.com/ROCm/ROCm/issues/5107#issuecomment-4979442043). |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF LCP-M2 stream-ordered contiguous prefill metadata | 2026-07-15 | clean explicit A/B `6131e891`, clean scoped policy `37b39269`; TheRock HIP `7.15.0-0000000`; one HIP hardware queue; TuneD accelerator-performance; exact Q4_K_M fingerprint retained | **Promoted through 4K and included in the current 512-64K production refresh**: full-model 512/1K/4K automatic-vs-explicit state is **83/83 exact**. Five balanced pairs improve prefill **1261.643 -> 1281.323 tok/s (+1.56%)**, **1333.877 -> 1345.928 (+0.90%)**, and **1356.934 -> 1364.103 (+0.53%)**. The explicit 128K one-queue gate completes warmup at only 483.439 tok/s, then re-enters the low-power GPU-active no-progress state on measured pass 1, so automatic policy retains synchronous metadata above 4K. | Yes, scoped prefill promotion gate | Keep `HIPENGINE_GGUF_PREFILL_DEVICE_METADATA=0|1` for rollback/diagnosis; never extend the 4K ceiling without a completed long-context lifecycle gate. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF LCP-4B exact prefill router-select geometry | 2026-07-15 | clean profiles `37b39269`/`89443a1f`, balanced candidate `c10c794c`, clean promoted policy `89443a1f`; TheRock HIP `7.15.0-0000000`; one HIP hardware queue; TuneD accelerator-performance; exact Q4_K_M fingerprint retained | **Promoted on gfx1151 and included in the current 512-64K production refresh**: the fresh 4K profile leaves router select at 12.539 ms / 0.41%, so a launch-only screen replaces risky logits+top-k fusion. 128 threads is **83/83 exact** and improves five-pair 512/4K prefill **1274.062 -> 1278.414 tok/s (+0.34%)** and **1361.337 -> 1366.173 (+0.36%)**. Clean trace cuts the named select family **12.539 -> 3.741 ms (-70.17%)** with 24 VGPR, 512 B LDS, zero scratch. Faster 64 threads is rejected because 4K full-model state is not exact. | Yes, final targeted prefill gate | Keep `HIPENGINE_GGUF_PREFILL_ROUTER_SELECT_THREADS=512` for rollback; gfx1100 stays 512 and decode stays 256. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF current decode closure profile and graph admission | 2026-07-15 | clean current code `89443a1f`; TheRock HIP `7.15.0-0000000`; one HIP hardware queue; TuneD accelerator-performance; exact Q4_K_M fingerprint retained | **Retained diagnostic / no new kernel promotion**: exact 16-step marker profiles confirm 512/4K dense Q8 at **8.560/8.541 ms/token** and 128K attention/dense-Q8 at **17.504/8.555 ms/token**. Grouped-GQA chunk 128 is +2.89% in isolation but changes one BF16 output; chunk 512 is inexact and slower. Dense-Q8 64 threads is 15.8% slower than 128. Current graph replay remains admitted over eager at **+1.00%/+0.86%** on 512/4K 1+3 and **+0.36%** in the bounded 128K confirmation, all IDs exact. | Diagnostic link; current graph rows retained through 64K, 128K topline blocked by prefill lifecycle | Retain chunk-256 LCP-D1 attention, 128-thread Q8, and graph replay. A future decode attempt needs a new exact algorithm/layout, not another launch-only sweep. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF final IOMMU-on production refresh and residual 128K lifecycle gate | 2026-07-15 | clean detached `61a27d72`; TheRock HIP `7.15.0-0000000`; automatic one hardware queue; TuneD accelerator-performance; exact Q4_K_M fingerprint retained | **Superseded IOMMU-on comparison**: clean right-sized 1+3 prefill is **1294.885/1358.342/1365.720/1034.845/796.083 tok/s** and graph decode is **49.041/51.623/52.422/43.572/37.622 tok/s**. All 15 IDs are `9707`; 128K remains blocked. The July 17 IOMMU-off row is the current topline, while this row remains the directional comparison source. | Historical comparison; no 128K number | Keep one queue as risk reduction, not a lifecycle guarantee. Require a same-commit reboot A/B for IOMMU causality and fixed gfx11 firmware/kernel or a production-quality workaround before restoring 128K. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF 128K HIP 7.13 versus 7.15 lifecycle diagnostic | 2026-07-15 | clean detached `61a27d72`; unchanged kernel `7.1.3-2-cachyos`, MES `0x88`, MES KIQ `0x6f`; one hardware queue; HIP 7.13/AOTriton 0.11.2 versus HIP 7.15/AOTriton 0.11.1 | **Retained diagnostic; no performance claim**: HIP 7.13 completes two independent warmup+3 gates at **509.659/499.895 tok/s** with all six IDs `9707`, then a third gate reproduces the stall after one measured pass. HIP 7.15 fails both matched controls, one in warmup and one after measured pass 1. Persistent states remain 100%/2.9 GHz at only **42-48 W** with no amdgpu/KFD journal fault. | No; full stacks differ and incomplete legs are not topline eligible | Do not recommend a HIP 7.13 downgrade as a lifecycle fix. The common firmware/kernel scheduler path remains the leading suspect; quantify stack-specific incidence only with a larger fixed-stack campaign. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF 128K persistent prefill flight-recorder capture | 2026-07-16 | clean detached `d697b971`; TheRock HIP 7.15; kernel `7.1.3-2-cachyos`; MES `0x88` / KIQ `0x6f`; one hardware queue; chunk markers | **Retained diagnostic; no performance claim**: a **503.876/27.970 tok/s** recorder-enabled warmup completes, then measured prefill 1 times out. The last retired marker certifies chunk `[24576,28672)`; the host reaches layer 11 in `[28672,32768)` and stops. Ordering narrows but does not identify the failing dispatch: layer 10 was enqueued, while layer-11 metadata or full-attention/MoE may not have returned. The persistent state is 100%/2.9 GHz at median 49 W for 1,436 seconds with no amdgpu/KFD/MES journal fault. | No; instrumentation-enabled incomplete run | Gate merged request/chunk metadata reuse separately, then use post-metadata/layer markers or KFD tracing to identify the last submitted and retired dispatch without over-reading `amdgpu_fence_info`. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF 128K merged request/chunk metadata-reuse lifecycle gate | 2026-07-16 | tracked-clean `7d4500c3`; TheRock HIP 7.15; kernel `7.1.3-2-cachyos`; MES `0x88` / KIQ `0x6f`; one hardware queue; chunk markers and task-state sampling | **Rejected lifecycle fix; no performance claim**: cached exact 512/1 passes, but the 128K first warmup times out. The last retired marker certifies `[57344,61440)` and host execution enters layer 18 linear attention in `[61440,65536)` without reaching layer 19. The state remains 100%/2.9 GHz at median 45 W for 1,636 seconds; the main thread stays runnable in user space, KFD event threads wait normally, and kernel logs remain clean. | No; incomplete instrumentation-enabled lifecycle run | Metadata-copy reduction is not sufficient. A next bounded diagnostic may remove the remaining compact-WMMA scalar D2H synchronization, but the existing empty-tile no-read route is a known 4K performance rejection and cannot become production unchanged. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF 128K compact-WMMA no-read lifecycle gate | 2026-07-16 | tracked-clean `d1b9e581`; TheRock HIP 7.15; one hardware queue; `NO_READ_MAX_SELECTED_ROWS=32768`; chunk markers and task-state sampling | **Rejected lifecycle fix; no performance claim**: cached exact 4K/1 passes and the recorder-enabled 128K warmup completes at **507.552/28.100 tok/s**, but measured prefill 1 times out. Retirement stops after `[32768,36864)`; without per-layer D2H, the host queues all 40 layers of `[36864,40960)` and reaches the next chunk embedding boundary. The state remains 100%/2.9 GHz at median 54 W for 1,437 seconds with clean logs. | No; known performance-rejected route and incomplete lifecycle run | Keep the no-read route rejected. Synchronous boundaries expose rather than prevent no-progress. Use layer retirement markers or KFD tracing for further localization; a production no-read path still requires indirect/device-sized launch. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF 128K layer-marker perturbation gate | 2026-07-16 | tracked-clean `9e4e7c37`; TheRock HIP 7.15; one hardware queue; layer-granularity persistent recorder | **Retained diagnostic completion; no performance claim**: one process completes warmup+3 exactly, with the final **5,392/5,392** checkpoint cursor retired, all four IDs `9707`, and measured medians **503.732/28.246 tok/s**. No persistent low-power state appears. | No; instrumentation changes every layer boundary and only one process completed | Superseded by two approved repeats below: the completion was a queue-perturbation signal, not reliable suppression. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF 128K layer-marker repeat lifecycle gate | 2026-07-16 | tracked-clean `5a665731`; TheRock HIP 7.15; one hardware queue; two approved sequential layer-recorder processes | **Rejected mitigation; no performance/correctness claim**: both repeats time out. Repeat 1 completes warmup+2 measured passes, then retires through layer 11 with layer 12 pending and host inside layer 13 at `[28672,32768)`. Repeat 2 completes warmup, then retires through layer 33 with layer 34 pending and host inside layer 35 at `[16384,20480)`. Both persist at 100%/2.9 GHz and median 43 W with clean logs. | No; incomplete instrumentation-sensitive processes | Reject layer markers and a standalone heartbeat as reliable fixes. The shared window is prior linear-layer post-read MoE tail/marker through current-layer pre-read work. Next high-information probe is a separate cached rocprofv3 kernel/HIP/HSA/KFD trace. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF 128K rocprofv3 inline-interposition trace gate | 2026-07-16 | tracked-clean `6e7ab8b0`; rocprofv3 1.3.2; kernel/HIP/HSA/copy/KFD trace requested; one queue; layer markers | **Incomplete/ambiguous diagnostic; no performance/correctness claim**: first prefill stalls after layer 15 with layer 16 pending and host inside layer 17 at `[102400,106496)`. The persistent state lasts 1,556 seconds at 100%/2.9 GHz/median 55 W. rocprofiler's injected completion signal concurrently stops advancing and logs 153M polling iterations; no trace files finalize. | No; profiler inline interception has its own documented hang class | Do not name a user kernel from this run. Retry with `ROCPROFILER_QUEUE_INTERPOSITION=0` or a streaming callback, and preserve the possibility that instrumentation induced this incidence. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF 128K current-boot healthy KFD controls | 2026-07-16 | tracked-clean merge `babbc8c6`; TheRock HIP 7.15; kernel `7.1.3-2-cachyos`; MES `0x88`; one-queue policy; chunk recorder; KFD MQD/sysfs snapshots | **Two exact completions; no performance or lifecycle-fix claim**: both independent warmup+3 processes finish with all six measured IDs `9707`, final cursors **5,392/5,392**, and measured prefill medians **488.431/509.332 tok/s**. Healthy snapshots at 97-99%/128-129 W show two compute queues and one SDMA queue, zero fault/page counters, and only 3-4 ms eviction time. `kfd/rls` nevertheless reports no active runlist. | Correctness only; debugfs/recorder timing is perturbing and no stall/HQD is captured | Retain as healthy MQD/sysfs baseline, not stability evidence. Superseded as the next-action record by the MES-log stalled-HQD capture below. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF 128K MES-log and stalled-HQD capture | 2026-07-16 | public tracked-clean `a7b4fe4b`; TheRock HIP 7.15; kernel `7.1.3-2-cachyos`; MES `0x88`; `mes_log_enable=1`, `gpu_recovery=1`, `send_sigterm=1`; one-queue policy; layer recorder | **Stall captured with mapped unread work; no performance/correctness claim**: first prefill freezes at cursor **389/339** after `[32768,36864)`. All 36 final samples are 100%/2.9 GHz at **41-49 W**. One HQD snapshot shows the active non-empty 1 MiB AQL queue at `rptr=0x32250` / `wptr=0x32450` (**32 unread packets**), with zero error/dequeue state. MES-log bytes remain identical through +30 seconds and change on monitor-requested SIGTERM; no autonomous recovery/reset fires. | No; incomplete diagnostic and only one HQD sample | File the dedicated ROCm/ROCm issue with a redacted bundle. Do not claim frozen hardware pointers or a named packet/kernel; the separate `sched_policy=2` follow-up below faults before prefill, so retain this capture and request AMD's supported scheduler-isolation path plus MES decoder. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF 128K non-HWS (`sched_policy=2`) isolation | 2026-07-16 | tracked-clean `ee932fb9` with runtime source equivalent to tagged `a7b4fe4b`; TheRock HIP 7.15; kernel `7.1.3-2-cachyos`; software scheduler confirmed; one-queue policy; no application rollback selectors | **Rejected before prefill; no performance/correctness claim**: exact 512/1 controls pass before and after at **1163.527/1199.181 tok/s**, but the 128K process aborts in seven seconds with a CPF gfxhub fault at `0x00007ff3409ae000`, ring 24 / VMID 8 / PASID 31. The coredump reaches `AqlQueue::ExecutePM4` through HSA executable freeze and code-cache invalidation; no reset occurs. | No; one 128K initialization fault and no prefill result | Restored and verified `sched_policy=0`; do not infer whether non-HWS changes the original HWS stall. Ask AMD whether this policy is supported on gfx1151 and for a supported scheduler-isolation method. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF correct eager baseline, revision bisect, and decode-only Amdahl | 2026-07-11 | clean detached hipEngine `5f4c6561`; TheRock HIP `7.13.60980-c76140fa27`; exact Q4_K_M fingerprint retained | **Retained**: p512/d128 exact eager is 49.285 tok/s; `4499fb13` is the direct-parent 3.088x speed boundary; 24 exact marker windows isolate the current family profile. | Yes, named repeated-token protocol | Rerun after eager decode math, route, dispatch/build caching, or a material family-kernel change; run separately on W7900. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF Q8T16 dual-split wave/block indexing | 2026-07-12 | clean scalar `8184355c` and promoted `e20cdc13`; TheRock HIP `7.15.0-0000000`; TuneD accelerator-performance; exact Q4_K_M fingerprint retained | **Retained**: clean p512/d128 eager **20.5342 -> 20.4709 ms/token** (-0.308%); marked dual-split leaf **4245.4 -> 4188.2 us/token** (-1.349%); graph route **20.5736 -> 20.5324 ms/token** (-0.200%); every token/state gate exact. | Yes, named repeated-token protocol | Rerun after Q8T16 indexing/layout, compiler, graph policy, or gfx1151 launch geometry changes; validate separately on gfx1100 before transfer. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF state-bound production decode graph | 2026-07-11 historical; 2026-07-12 refresh | clean detached hipEngine `7f611fe3` on HIP 7.13; clean `8184355c`/`e20cdc13` on HIP 7.15; exact Q4_K_M fingerprint retained | **Historical retained / current speed-policy stale**: all 128 graph launches remain byte-exact. HIP 7.13 measured +0.112% over eager; both current HIP 7.15 reruns reject at -0.246%/-0.293%. | Current table reports exact diagnostic wall, not a graph-over-eager win | Run a scoped balanced current-stack A/B; restore eager default if graph does not reproduce a win. Validate separately on gfx1100 before any admission. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF replacement-layout residency and 24 GiB-class gate | 2026-07-11 | clean detached hipEngine `d70c9464`; TheRock HIP `7.13.60980-c76140fa27`; exact Q4_K_M fingerprint retained | **Retained memory/correctness gate**: 733 unique sources, no raw+replacement duplicates or optional sidecars, 21.478 GiB owned/tracked p512/d128 graph session, 2.522 GiB budget margin. `performance_claim=false`; G5 supplies linked speed non-regression. | Diagnostic link only | Rerun after weight materialization/layout, KV/state, prefill scratch, graph allocation, or max-sequence policy changes; context-specific capacity remains separate. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Historical PARO exact c1-c8 shape/routing catalog | 2026-07-11 | clean detached hipEngine `a18ff7bc`; TheRock HIP `7.13.60980-c76140fa27`; exact model and prompt fingerprints retained | **Historical c1 and fail-closed routing anchor**: exact-fixture c1 graph is 66.910 tok/s median; that snapshot's c2-c8 candidates fail independent-c1 equality at index 2 and are explicitly serial. G3/G5 supersede this catalog for direct and resident c2/c4/c8; c3/c5/c6/c7 remain partitions rather than native widths. | Historical/diagnostic; routing superseded by G3/G5 | Rerun after c1 graph/prefill changes or any general native c3/c5/c6/c7 algorithm change. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | PARO ragged c8-to-c1 lifecycle correctness | 2026-07-11 | clean detached hipEngine `6f1910c9`; TheRock HIP `7.13.60980-c76140fa27`; same exact model/fixture fingerprints as P1 | **Accepted correctness-only gate**: eight token sequences, 30 linear-state families, and 10 full-KV families match c1 through EOS and front/middle/tail sparse cancellation. `performance_claim=false`; ragged prefill uses an exact per-segment fallback. | Diagnostic link only | Rerun after ragged prefill, scheduler retirement, slot/state/KV addressing, or true-c1 decode changes; run independently on W7900. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | PARO/llama.cpp concurrency | 2026-06-15 | measured hipEngine revision not recorded in summary; gfx1151 forced through `HIPENGINE_HIP_ARCH` | **Stale diagnostic**: `performance_claim=false`, mixed quant, and incomplete backend provenance | Diagnostic link only | Rerun c=1..8 plus shrinking batches at one clean revision with detected arch and all-choice token counts |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF MTP exact/default, `llama-compat`, and llama.cpp HIP refresh | 2026-07-17 | clean detached hipEngine `2edbb2ee`; TheRock HIP 7.15; TuneD accelerator-performance; `amd_iommu=off`; exact Q4_K_M/prompt fingerprints; llama.cpp HIP build 9648 byte-identical to the prior publication | **Historical pre-NativeSpecCycle compatibility topline; exact/default remains the semantic control**: exact B5 is a narrow aggregate negative **56.386 vs 56.983 AR tok/s (0.9895x)** with heldout **0.9339x**; explicit accuracy-traded `llama-compat` is **81.900 vs 56.783 AR tok/s (1.4423x)** with heldout **1.4306x** and every category above AR. At matched complete decode boundaries hipEngine is **81.745 tok/s** versus transition-normalized llama.cpp **68.153 tok/s**. The repeated-stream byte-exact state oracle passes. [`artifact`](results/2026-07-17-gfx1151-amd-iommu-off-mtp-refresh.json). | Historical qualified absolute; compatibility explicit-only and superseded for current-main serving by the N1/N3 row below | Rerun exact/default after semantic-path changes; use the N1/N3 row for current-main compatibility. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF NativeSpecCycle N1 target graph + N3 public complete cycle | 2026-07-19 | clean detached hipEngine `1163e1bb`; TheRock HIP 7.15; TuneD accelerator-performance; `amd_iommu=off`; one automatic HIP hardware queue; exact UD-Q4_K_M/BF16-KV and category prompt identities; cached builds and final-child trace | **Retained current-main transfer**: N1 is **80.132 tok/s** and public N3 retains **80.099 tok/s**, versus the clean direct-commit control's **70.020 tok/s (+14.39%)**; N3 wall falls **14.314 -> 12.551 ms/output (-12.32%)**. All **240 IDs / 97 cycle semantics** and **77.72% / 59.58%** acceptance economy match, every train/heldout/category rate improves, and N3 is only **0.042%** below N1. The six-step cached trace is **24.891 ms host / 21.674 ms kernels / 3.218 ms residual**, with 940 calls/step and expected zero-scratch metadata leaves. [`artifact`](results/2026-07-19-gfx1151-llama-compat-native-cycle-transfer.json). | Yes, for explicit accuracy-traded `llama-compat`; exact/default is unchanged | Rerun after target graph bindings, N2/N3 state/KV/cursor ownership, verifier math, compiler/runtime, model/prompt identity, or output horizon. N3P proposal graph remains independently unadmitted. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Historical GGUF MTP pre-correctness-pass rows | 2026-07-02–03 | hipEngine exact `44c4d3d4`, `llama-compat` `ca571bf6`; environment provenance incomplete | **Superseded history**: exact 61.98 and compatibility 71.52 tok/s remain useful deltas, but no longer define the current table. | Historical links only | Do not promote without the current state lifecycle, clean provenance, and transition-matched timing contract. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF OpenAI server automatic-route gate | 2026-07-11 | tracked-clean hipEngine `d2b1e742`; TheRock HIP `7.13.60980-c76140fa27`; exact GGUF and prompt-suite fingerprints retained; unrelated untracked files disclosed | **Diagnostic correctness rejection**: compatibility MTP is faster at c1/c2 but changes true-AR IDs on heldouts, so it cannot select automatic routing. One c8 AR repetition also exposes the separate exact-concurrency blocker. | Diagnostic link only | Implement an exact/default server MTP hook, then rerun full plus category-heldout realized-group economics before admitting it to `auto`. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | HIP versus Vulkan timing-contract v2 micro matrix | 2026-07-12 | clean detached hipEngine `50bea8f3`, TheRock ROCm `7.15.0a20260711`, kernel `7.1.3-2-cachyos`, RADV/Mesa `26.1.4`, corrected gfx1151 device wheels | **Matched and strict matrices retained**: each passes 22/22 comparisons and 232 burst GPU rows after portable q8_1 RNE/ties-away rounding eliminates the systematic scale mismatch | Linked, not copied here | Run the portable shader's strict gate on gfx1100 when W7900 access returns; otherwise rerun after a timed kernel/harness, ROCm, Mesa, or clock-policy change |
| Radeon Pro W7900, gfx1100 | HIP versus Vulkan timing-contract v2 micro matrix | 2026-07-11 | clean hipEngine `c57f21b5`; TheRock ROCm `7.15.0a20260711`; RADV/Mesa `26.1.4` | **Retained**, 22/22 comparisons and 232 burst GPU rows pass provenance, correctness, exact-matrix, device, and clock gates | Linked, not copied here | Rerun after a timed kernel/harness, ROCm, Mesa, or device clock-policy change |

## Current Eligible Toplines

Only rows with an eligible evidence status appear in this section. The sync
script copies this marked block into the root README byte-for-byte.

### gfx1100 model throughput

The GGUF column is the clean 2026-07-16 defaults-only right-sized sweep at
`28b37356` on the complete therock HIP 7.15 stack. Each shape uses one discarded
eager warmup plus three measured runs in an independent resident process; every
measurement captures and closes a fresh state-bound decode graph. Package
automatic policy selects peer-wave GDN, spill-free selected prefill, the
persistent cooperative router, and the long-context parallel reducer while KV
remains default BF16. Focused post-sweep transfers now also select the exact
256-thread F32-weight router-logits wrapper and 128-thread bulk router selector
on gfx1100.

Prefill is now
**2716.648/3052.541/2953.101/2078.038/1559.878/1037.378 tok/s**, graph decode is
**92.833/98.148/100.522/88.240/76.691/62.669 tok/s**, and tracked right-sized
memory is **21.228/21.295/21.670/22.234/22.879/24.168 GiB** from 512 through
128K. All 18 final IDs are `9707`; the largest prefill/decode stdev over median
is **0.658%/0.223%**. The six-shape values remain the last clean publication
sweep. A same-session balanced W7900 gate for the newly retained router default
moves focused 512/4K prefill **2689.171 -> 2795.242 (+3.94%)** and **2955.867 ->
3070.905 tok/s (+3.89%)**; graph decode is **-0.022%/+0.159%**, tracked memory
is unchanged, the 4K primitive is bit-exact, and all timed final IDs match. An
incremental 128-thread selector gate on top improves aggregate 512/4K medians
another **+0.32%/+0.81%** (paired medians **+0.30%/+0.12%**), with graph decode
**-0.068%/+0.216%**, unchanged memory, bit-exact selected IDs/routing weights,
and matching final IDs. A direct legacy-512/512 versus final-package stack gate
confirms paired prefill gains of **+3.87%/+4.16%** at 512/4K, graph decode
**+0.11%/+0.07%**, and unchanged memory/IDs. The subsequently retained
stream-ordered metadata path adds aggregate **+0.41%/+2.43%** at 512/4K
(paired **+0.26%/+2.26%**), with non-regressive decode, unchanged memory/IDs,
and an exact metadata primitive. Production peer-wave GDN remains unchanged;
the strict-exact rollback now resolves to nonvolatile direct-LDS32, which moves
volatile-direct 512/4K prefill **+73.01%/+82.46%**, halves VGPR **64 -> 32**,
and preserves byte-exact state, decode, and compact-scratch memory. A final clean
selector-unset confirmation moves the pre-screen 512/4K package baseline
**2699.283/2972.935 -> 2808.249/3173.723 tok/s (+4.04%/+6.75%)**; graph decode
is **-0.26%/+0.24%**, tracked memory unchanged, and all IDs exact.

Relative to the July 14 GGUF table, prefill improves **+35.27% to +118.78%**
and decode improves **+2.24% to +3.46%**. Prefill now beats llama.cpp HIP at
all six shapes by **12.62-30.95%** and beats llama.cpp Vulkan from 512 through
64K by **3.37-17.10%**; only 128K Vulkan prefill remains ahead, by **3.88%**.
Decode beats llama.cpp HIP everywhere by **2.85-26.02%**, while Vulkan remains
ahead by **2.47-13.87%**. The tracked-memory count is within **-0.378 to
+0.079 GiB** of llama.cpp HIP's broader whole-device readings, so memory is at
practical parity but small cross-scope differences are not allocator-efficiency
claims.

Evidence: [`focused convergence confirmation`](results/2026-07-16-gfx1100-gguf-convergence-final-confirmation.json),
[`final optimization sweep`](results/2026-07-16-gfx1100-gguf-final-optimization-sweep.json),
[`256-thread router transfer`](results/2026-07-16-gfx1100-gguf-router-threads256-promotion.json),
[`128-thread router-select transfer`](results/2026-07-16-gfx1100-gguf-router-select-threads128-promotion.json),
[`retained router stack`](results/2026-07-16-gfx1100-gguf-router-stack-promotion.json),
[`device-metadata transfer`](results/2026-07-16-gfx1100-gguf-prefill-device-metadata-promotion.json),
[`nonvolatile exact rollback`](results/2026-07-16-gfx1100-gguf-gdn-nonvolatile-exact-rollback.json),
[`peer-GDN promotion`](results/2026-07-15-gfx1100-gguf-prefill-lcp5a-spill-free-peer-promotion.json),
[`decode attribution`](results/2026-07-15-gfx1100-gguf-decode-lcpd3-attribution.json),
[`LCP-D2 gate`](results/2026-07-14-gfx1100-gguf-decode-lcp-d2-parallel-reduce.json),
[`LCP-M1 memory gate`](results/2026-07-14-gfx1100-gguf-lcp-m1-prefill-scratch-liveness.json),
[`persistent router counter`](results/2026-07-14-gfx1100-gguf-persistent-router-counter.json),
and [`mixed-KV closure`](results/2026-07-15-gfx1100-gguf-tail4-hadamard-clean-gate.json).

PARO remains the clean 2026-07-12 `8116c453` two-warmup/five-measurement row.
llama.cpp HIP/Vulkan remain the matched July 12 Q4_K_M/F16-KV references with
one internal warmup plus five samples per split phase. Every engine uses the
stated graph/eager route and excludes graph capture from steady decode timing.

Bold marks the best raw value in each row. It is descriptive only: PARO is W4
PARO/BF16 KV, while the other columns use the same Q4_K_M GGUF with hipEngine
BF16 KV and llama.cpp F16 KV. Memory scopes also differ.

<!-- BEGIN TOPLINE:W7900_SWEEP -->
#### Prefill tok/s

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **2917.732** | 2716.648 | 2412.320 | 2627.990 |
| 1K/128 | 2995.876 | **3052.541** | 2389.670 | 2631.750 |
| 4K/128 | 2943.038 | **2953.101** | 2255.080 | 2521.770 |
| 32K/128 | **2108.868** | 2078.038 | 1667.640 | 1943.920 |
| 64K/128 | **1584.131** | 1559.878 | 1291.820 | 1414.470 |
| 128K/128 | 1056.252 | 1037.378 | 891.949 | **1079.280** |

#### Decode tok/s

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **115.599** | 92.833 | 80.756 | 107.786 |
| 1K/128 | 103.238 | 98.148 | 80.805 | **107.555** |
| 4K/128 | **105.943** | 100.522 | 79.768 | 103.066 |
| 32K/128 | **92.438** | 88.240 | 74.304 | 91.835 |
| 64K/128 | 78.260 | 76.691 | 69.010 | **83.746** |
| 128K/128 | 60.663 | 62.669 | 60.933 | **70.833** |

#### Peak memory GiB

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **18.144** | 21.228 | 21.606 | 21.260 |
| 1K/128 | **18.367** | 21.295 | 21.618 | 21.220 |
| 4K/128 | **19.161** | 21.670 | 21.674 | 21.278 |
| 32K/128 | **19.864** | 22.234 | 22.216 | 21.855 |
| 64K/128 | **20.403** | 22.879 | 22.895 | 22.512 |
| 128K/128 | **22.124** | 24.168 | 24.089 | 23.824 |
<!-- END TOPLINE:W7900_SWEEP -->

hipEngine memory is its tracked allocator high-water; llama.cpp is absolute
whole-device W7900 VRAM used, sampled from DRM sysfs `card1` every 10 ms. The
host's `rocm-smi` card labels use a different numbering scheme; the retained
artifact validates the 48 GiB W7900 device rather than the idle 24 GiB XTX.
Use memory values for within-column context growth, not small cross-column
allocator-efficiency claims.

Artifacts: [current hipEngine GGUF throughput and memory](results/2026-07-16-gfx1100-gguf-final-optimization-sweep.json),
[superseded July 14 hipEngine GGUF sweep](results/2026-07-14-gfx1100-gguf-optimization-right-sized-3run.json),
[LCP-M1 memory gate](results/2026-07-14-gfx1100-gguf-lcp-m1-prefill-scratch-liveness.json),
[July 12 accepted summary](results/2026-07-12-w7900-v030-8116c453-summary.json),
[hipEngine PARO](results/2026-07-12-w7900-v030-8116c453-hipengine-paro-packed-5run.json),
[superseded hipEngine GGUF](results/2026-07-12-w7900-v030-8116c453-hipengine-gguf-q4km-5run.json),
[llama.cpp HIP](results/2026-07-12-w7900-v030-8116c453-llamacpp-hip-q4km-f16kv.json),
[llama.cpp Vulkan](results/2026-07-12-w7900-v030-8116c453-llamacpp-vulkan-q4km-f16kv.json),
and [W7900 GGUF oracle](results/2026-07-12-w7900-v030-gguf-eager-p512-d4.json).

### gfx1151 model throughput

The current table is the clean 2026-07-17 refresh at hipEngine `2edbb2ee`,
TheRock HIP 7.15, TuneD `accelerator-performance`, automatic one-hardware-queue
gfx1151 policy, and kernel boot option `amd_iommu=off`. PARO uses two warmups
plus five measurements per right-sized process, GGUF uses one plus three, and
each llama.cpp backend uses five repetitions. PARO passes all six shapes. GGUF
512 through 64K passes clean provenance, finite logits, exact final IDs, and the
5% variance gate; maximum prefill/decode stdev over median is only
**0.122%/0.028%**, and all 15 measured IDs are `9707`.

**IOMMU boot note (directional, not causal):** relative to the prior published
IOMMU-on rows, the arithmetic mean change across the 11 eligible current
hipEngine cells is **+4.60% prefill / +6.20% decode**. GGUF 512-64K averages
**+8.84% / +5.84%**, while PARO averages **+1.08% / +6.51%** and has mixed
prefill results. The measured hipEngine revision and some routing changed, so a
same-commit reboot A/B is still required to attribute the delta solely to
IOMMU. This boot has zero IOMMU groups and disables the XDNA/NPU driver
(`amdxdna: Running without IOMMU not supported`).

Repeated GGUF 128K remains **blocked**. Under IOMMU-off it completes warmup and
measured pass 1 at **584.059/583.464 prefill tok/s**, then measured pass 2 fails
to complete before the 1,800-second workload bound. No kernel fault/reset is
logged and the GPU returns idle after termination. Thus `amd_iommu=off` does
not close the lifecycle gate, and no current GGUF 128K throughput or memory
number appears in the topline. Since PARO uses W4 PARO rather than Q4_K_M,
llama.cpp uses F16 rather than BF16 KV, and memory scopes differ, bold values
are descriptive raw leaders rather than same-math allocator claims.

<!-- BEGIN TOPLINE:GFX1151_SWEEP -->
#### Prefill tok/s

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | 1298.259 | **1395.379** | 1184.628 | 1161.498 |
| 1K/128 | 1332.199 | **1481.943** | 1192.768 | 1154.327 |
| 4K/128 | 977.252 | **1444.733** | 1148.155 | 1114.081 |
| 32K/128 | 827.350 | **1132.215** | 843.252 | 873.573 |
| 64K/128 | 690.642 | **892.663** | 632.774 | 702.742 |
| 128K/128 | 498.101 | — (blocked) | 432.033 | **499.728** |

#### Decode tok/s

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **70.750** | 52.761 | 53.222 | 63.795 |
| 1K/128 | **65.905** | 54.658 | 53.044 | 63.391 |
| 4K/128 | **66.728** | 55.297 | 52.338 | 61.863 |
| 32K/128 | **53.458** | 45.983 | 45.946 | 52.286 |
| 64K/128 | 44.793 | 39.388 | 40.353 | **45.160** |
| 128K/128 | 32.615 | — (blocked) | 32.728 | **35.569** |

#### Peak memory GiB

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **18.039** | 21.478 | 21.375 | 21.551 |
| 1K/128 | **18.051** | 21.710 | 21.387 | 21.501 |
| 4K/128 | **19.026** | 22.995 | 21.444 | 21.507 |
| 32K/128 | **19.716** | 23.559 | 21.987 | 22.191 |
| 64K/128 | **20.344** | 24.203 | 22.666 | 22.627 |
| 128K/128 | **21.881** | — (blocked) | 23.862 | 24.254 |
<!-- END TOPLINE:GFX1151_SWEEP -->

The PARO column is W4 PARO/BF16 KV. The other three columns use the same
Q4_K_M GGUF; hipEngine uses BF16 KV and llama.cpp uses f16 KV. Peak-memory
scopes differ: hipEngine reports its tracked allocator high-water, while
llama.cpp reports absolute whole-device amdgpu GTT used, sampled every 10 ms.
Use memory values for within-column context growth; small cross-column deltas
are not allocator-efficiency claims. hipEngine load and graph capture are
excluded from phase throughput.

Artifacts: [current IOMMU-off four-column refresh and 128K blocker](results/2026-07-17-gfx1151-amd-iommu-off-topline-refresh.json),
[preliminary 512/4K IOMMU-off diagnostic](results/2026-07-17-gfx1151-amd-iommu-off-short-context-diagnostic.json),
[previous IOMMU-on GGUF 512-64K refresh](results/2026-07-15-gfx1151-gguf-production-refresh-512-64k-128k-blocked.json),
[HIP 7.13 versus 7.15 128K lifecycle diagnostic](results/2026-07-15-gfx1151-128k-hip713-vs-715-lifecycle.json),
and [dedicated ROCm issue](https://github.com/ROCm/ROCm/issues/6437).

### Speculative decode

The public table includes only contracts with a true same-protocol AR control.
The exact/default and `llama-compat` columns are intentionally separate:
exact/default is the semantic control, while `llama-compat` is the closer
structural comparison with llama.cpp's B2 natural-output-horizon route.

#### Cross-engine GGUF decode timing contract

Use this contract whenever a hipEngine GGUF decode rate is placed beside a
llama.cpp rate:

- hipEngine true AR excludes model load, prefill, its prompt-produced first
  token, and warmup. Timing begins before the first of `N` measured
  `session.step()` calls and ends after the `N`th call returns.
- hipEngine MTP excludes prefill and draft warmup. Cross-engine throughput uses
  complete `cycle_wall_ms`, measured from proposal-cycle entry through draft,
  target verification, recurrent/KV state commit, acceptance, and output
  accounting. The canonical same-harness MTP/AR objective may retain the
  slightly narrower summed stage wall, but that value is not ranked against
  llama.cpp.
- llama.cpp sets `server_slot::t_start_generation` **after** sampling the first
  output token, while `predicted_n` includes that token. Native
  `predicted_n / predicted_ms` therefore counts one untimed token per request.
  To compare `N` timed transitions, request `N+1` outputs and report
  `sum(predicted_n - 1) * 1000 / sum(predicted_ms)`.
- Client/request wall includes prompt processing, HTTP, and response handling;
  it is a separate end-to-end diagnostic and is never compared with direct
  decode-only wall. Record KV dtype differences beside every cross-engine row.

The committed runner emits native, client, and transition-normalized fields.
The exact local llama.cpp source/binary lineage and instrumentation are retained
under [`benchmarks/llama.cpp/`](llama.cpp/).

<!-- BEGIN TOPLINE:SPECULATIVE -->
#### GGUF MTP comparison, Radeon Pro W7900/gfx1100

| Metric | hipEngine GGUF true AR | hipEngine GGUF exact/default | hipEngine GGUF `llama-compat` | llama.cpp HIP base AR | llama.cpp HIP bundled MTP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Route | State-bound graph, no MTP | B3, fixed 10 cycles | B2 natural24, reusable B1/B2 target graphs | Natural25 request / 24 timed transitions | B2, natural25 request / 24 timed transitions |
| Decode | **98.75 tok/s fixed / 96.75 tok/s natural24** | 68.50 tok/s | **122.67 tok/s** | 78.05 tok/s transition-normalized | 115.44 tok/s transition-normalized |
| Own true AR | same route | 98.75 tok/s | 96.75 tok/s | same route | 78.05 tok/s |
| MTP / own AR | 1.0000x | **0.6936x** | **1.2679x** | n/a | **1.4791x** |
| Draft acceptance | n/a | 73.53% | 80.45% | n/a | 81.56% |
| Accepted draft/output | n/a | 50.00% | 60.00% | n/a | 58.40% |
| Complete wall per output/transition | 10.336 ms natural24 | 14.696 ms | **8.186 ms** | 12.812 ms | 8.662 ms |
| State/commit contract | serial autoregressive | serial-prefix preserving | direct partial commit/dp4a; accuracy-traded | native llama.cpp autoregressive | native llama.cpp compatibility target |

The W7900 route now reuses one fixed-address target graph per B1/B2 shape
bucket. Live token, position, context, and cursor metadata are staged on device;
the five two-row output-cap tails use B1 and the four true one-row/no-draft
cycles stay on AR. Unsupported configurations fall back before launch, while a
post-launch failure never re-executes a possibly mutating verifier.

Two clean full-suite processes at `0d7b86e7` measure **123.33 and 122.67 tok/s**
(0.54% spread). The conservative run is **1.2679x** its true graph AR and
**6.26% faster** than llama.cpp's **115.44 tok/s / 8.662 ms-transition** floor,
while complete wall is **5.50% lower** at **8.186 ms/output**. Draft acceptance
and accepted/output remain exactly **80.45% / 60.00%**. All 240 output IDs and
all 96 cycle semantics in both runs match the prior eager-target
`llama-compat` baseline. hipEngine uses BF16 KV versus llama.cpp F16 KV, and the
external row remains `performance_claim=false` because its preserved
instrumentation checkout is dirty.

The target graph also passes the real 35B oracle at two B2 positions plus B1:
target top-1, 16,384 FP32 hidden values, each set of 60 captured and 60 resident
Conv/GDN buffers, all 20 K/V buffers, and cursors are byte-exact. A cached
six-step trace records zero measured recaptures, **18.67 ms host / 13.67 ms
kernels / 5.00 ms residual**, and the expected dynamic-metadata, cursor-advance,
and top-1 widening leaves. The prior eager profiler residual was 38.41 ms.

Exact/default remains the semantic control. `llama-compat` remains explicit-only
because direct partial commit is not serial-prefix-equivalent; this retained
speed result does not make it the automatic exact route. The fixed-cycle exact
and natural24 compatibility rows remain different protocols.

##### W7900 reusable-native `llama-compat` full-suite gate

| Scope | Prompts | True AR tok/s | `llama-compat` tok/s | MTP / AR | Draft acceptance | Accepted/output | Cycle wall/output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 10 | 96.75 | **122.67** | **1.2679x** | 80.45% | 60.00% | 8.186 ms |
| Train | 6 | 96.13 | **124.70** | **1.2973x** | **87.25%** | 61.81% | 8.052 ms |
| Heldout | 4 | 97.68 | **119.73** | **1.2257x** | **71.43%** | 57.29% | 8.388 ms |
| `code` | 4 | 97.06 | **127.81** | **1.3168x** | 93.94% | 64.58% | 7.854 ms |
| `general_en` | 2 | 94.25 | **123.37** | **1.3091x** | 75.68% | 58.33% | 8.138 ms |
| `general_ja` | 2 | 98.04 | **118.42** | **1.2079x** | 69.23% | 56.25% | 8.480 ms |
| `mixed_ja_en` | 2 | 97.40 | **116.78** | **1.1990x** | 72.97% | 56.25% | 8.604 ms |

Every category and the heldout split beat their true same-protocol AR control;
even the slowest category remains above the aggregate external floor in the
conservative run. The corrected 54.88 tok/s eager-target row remains the
optimization baseline, not the current route. Artifacts:
[`retained reusable route`](results/2026-07-19-w7900-llama-compat-reusable-native-cycle.json),
[`N2 ownership diagnostic`](results/2026-07-19-w7900-llama-compat-native-cycle-n2.json),
[`N3 complete-cycle diagnostic`](results/2026-07-19-w7900-llama-compat-native-cycle-n3.json),
[`N3P proposal-submission diagnostic`](results/2026-07-19-w7900-llama-compat-native-cycle-n3p.json),
[`prior baseline`](results/2026-07-19-w7900-hipengine-llama-compat-current-baseline.json),
and [`llama.cpp floor`](results/2026-07-19-w7900-llamacpp-mtp-natural25-refresh.json).

#### GGUF MTP comparison, Radeon 8060S/gfx1151

| Metric | hipEngine GGUF exact/default | hipEngine GGUF `llama-compat` | llama.cpp HIP |
| --- | ---: | ---: | ---: |
| Route | B5, fixed 10 cycles | B2 natural24, NativeSpecCycle N3 | B2, natural25 request / 24 timed transitions |
| Canonical/native MTP decode | 56.39 tok/s (0.9895x own AR) | **80.10 tok/s (1.4282x own AR)** | 70.99 tok/s native (1.3530x own AR; not cross-engine comparable) |
| Cross-engine MTP decode-transition rate | n/a: fixed-cycle horizon | **80.10 tok/s** | 68.15 tok/s |
| Cross-engine own AR transition rate | n/a: fixed-cycle horizon | **56.09 tok/s** | 50.37 tok/s |
| Cross-engine MTP / own AR | n/a | 1.4282x | 1.3530x |
| Draft acceptance | 72.33% | 77.72% | 79.56% |
| Accepted draft/output | 53.49% | 59.58% | 57.60% |
| Full-cycle/predicted wall per counted output or timed transition | 17.808 ms/output | 12.551 ms/output | 14.673 ms/transition |
| State/commit contract | exact/default, serial-prefix preserving | N3 complete public cycle; accuracy-traded | native llama.cpp compatibility target |

The IOMMU-off exact/default B5 route remains the current semantic control at
**56.39 vs 56.98 true-AR tok/s (0.9895x)**. `llama-compat` is separate,
explicit-only, and not serial-prefix-equivalent. On current main, registering
the reusable gfx1151 target graph moves the clean direct-commit control
**70.020 -> 80.132 tok/s (+14.44%)**; N3 public complete-cycle ownership retains
**80.099 tok/s (+14.39%)** and cuts complete wall **14.314 -> 12.551 ms/output
(-12.32%)**. N3 is only **0.042%** below target-only N1.

All **240 output IDs / 97 cycle semantics** match across clean control, N1, and
N3, with unchanged **77.72% draft acceptance / 59.58% accepted-output**. The
prior clean `2edbb2ee` direct-commit row remains slightly higher at **81.90
tok/s** (-2.20% versus current N3), but it is a different revision/run and no
source regression is attributed. Against the preserved transition-normalized
llama.cpp context, current N3 is **80.10 vs 68.15 tok/s (+17.53%)**; BF16 versus
F16 KV and the dirty preserved llama.cpp source remain disclosed.

##### gfx1151 NativeSpecCycle N3 `llama-compat` full-suite gate

| Scope | Prompts | True AR tok/s | N3 tok/s | N3 / AR | Draft acceptance | Accepted/output | Cycle wall/output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 10 | 56.09 | **80.10** | **1.4282x** | 77.72% | 59.58% | 12.551 ms |
| Train | 6 | 55.97 | **80.91** | **1.4457x** | **82.08%** | 60.42% | 12.429 ms |
| Heldout | 4 | 56.26 | **78.91** | **1.4025x** | **71.79%** | 58.33% | 12.733 ms |
| `code` | 4 | 56.12 | **86.08** | **1.5338x** | 91.04% | 63.54% | 11.684 ms |
| `general_en` | 2 | 57.26 | **78.98** | **1.3795x** | 71.79% | 58.33% | 12.716 ms |
| `general_ja` | 2 | 55.61 | **75.12** | **1.3509x** | 69.23% | 56.25% | 13.388 ms |
| `mixed_ja_en` | 2 | 55.35 | **75.66** | **1.3669x** | 69.23% | 56.25% | 13.282 ms |

Every category and the heldout split beats its true same-protocol AR control and
improves versus the clean current-main direct-commit route by **9.91% to
19.45%**. The real 35B N1/N2 oracle passes target IDs, FP32 hidden rows, all 60
Conv/GDN and 20 full-KV buffers, selected commits, and cursors. The six-step
cached trace records zero recaptures, **24.891 ms host / 21.674 ms kernels /
3.218 ms residual**, 940 calls/step, and the expected zero-scratch metadata
leaf. N3P remains unregistered on gfx1151 because it is not needed for this win
and was not the gfx1100 topline. Artifact:
[`gfx1151 NativeSpecCycle transfer`](results/2026-07-19-gfx1151-llama-compat-native-cycle-transfer.json).

#### Dense PARO DFlash

| Path | Platform and protocol | Result | Evidence status |
| --- | --- | ---: | --- |
| DFlash B=4 online-gated | W7900/gfx1100; Qwen3.6-27B PARO target plus Qwen3.6-27B DFlash drafter; 9 prompts; 64 decode tokens | 40.10 vs 32.57 AR tok/s, **1.231x** | Retained under the recorded DFlash gate; source tree was dirty and must be refreshed before changing the claim |
<!-- END TOPLINE:SPECULATIVE -->

Artifacts: [W7900 GGUF MTP transfer](results/2026-07-12-w7900-gfx1100-gguf-mtp-transfer.json),
[W7900 llama.cpp MTP floor refresh](results/2026-07-19-w7900-llamacpp-mtp-natural25-refresh.json),
[current W7900 hipEngine `llama-compat` baseline](results/2026-07-19-w7900-hipengine-llama-compat-current-baseline.json),
[DFlash](results/2026-06-11-hipengine-dflash-27b-dense-hardening-rerun.json),
[current gfx1151 NativeSpecCycle transfer](results/2026-07-19-gfx1151-llama-compat-native-cycle-transfer.json),
[gfx1151 IOMMU-off GGUF MTP refresh](results/2026-07-17-gfx1151-amd-iommu-off-mtp-refresh.json),
and [llama.cpp instrumentation manifest](llama.cpp/manifest.json). Historical
gfx1151 sources remain [exact B5](results/2026-07-02-ar-mtp-default-parallelattn-full.json),
[`llama-compat` B2](results/2026-07-03-ar-mtp-llama-compat-directcommit-nocopy-natural24-cyclecap24-f32head-full.json),
and [llama.cpp B2](results/2026-07-02-llamacpp-mtp-stage-timing-b2-natural24-rerun.json).

The current clean gfx1151 PARO DFlash profile remains outside this eligible
table: it is exact but measures only `9.676` versus `65.266 tok/s` AR
(`0.14825x`), so DFlash stays default-off. Branch-copy is faster but diverges
at generated token 1, and fused target LM-head is 5.16% slower. The diagnostic
artifact is [SOL-S4](results/2026-07-11-sol-s4-gfx1151-paro-dflash-profile.json).

### GGUF decode

These are exact repeated-token SOL-G4/G5 rows, not natural-prompt quality or
speculative-economics results. The graph delta uses its same-run eager control;
the Q8T16 row is the current eager timing while SOL-G4 remains the historical
revision-bisect/Amdahl baseline.

<!-- BEGIN TOPLINE:GFX1151_GGUF_EAGER -->
| Path | Platform and protocol | Result | Evidence status |
| --- | --- | ---: | --- |
| GGUF eager c1 | Radeon 8060S/gfx1151; Qwen3.6-35B-A3B UD-Q4_K_M; BF16 KV; `[9707] * 512`; TheRock HIP 7.15; TuneD accelerator-performance; clean scalar/candidate/scalar, 1 discarded + 4 measured runs per leg; 128 eager steps; graph off | **48.850 tok/s** (`20.471 ms/token`), **+0.309%** vs clean scalar control | Retained for this exact repeated-token protocol; control/candidate ranges do not overlap, every output ID is 9707, and the G1 hidden/state/KV oracle is linked |
| GGUF state-bound graph c1 | Radeon 8060S/gfx1151; same current model/KV/prompt/stack; 1 warmup + 4 measured rotating same-session runs; 128 steps; capture and destroy charged | **48.704 tok/s** (`20.532 ms/token`), **-0.293%** vs same-run eager; **+0.201%** vs scalar graph | Exact 128/128 state/KV/token replay, but current G5 rejects a graph-over-eager speed claim; graph default policy is tracked separately |
<!-- END TOPLINE:GFX1151_GGUF_EAGER -->

Artifacts: [`Q8T16 wave/block production A/B`](results/2026-07-12-gfx1151-q8-t16-waveblock-production.json),
[`SOL-G4 eager audit`](results/2026-07-11-sol-g4-gfx1151-gguf-eager-decode-audit.json),
and [`SOL-G5 production graph audit`](results/2026-07-11-sol-g5-gfx1151-gguf-decode-graph-production-audit.json).

### PARO concurrency and production routing

The current gfx1100 retained row is deliberately narrower than the GGUF server
claim: it is one explicit direct physical-c2 model step, not the public/OpenAI
loop. Canonical `auto` resolves the exact selected-batch MoE route for this
step; grouped-compact remains a slower exact diagnostic.

| gfx1100 direct route | Median aggregate decode | Per-request decode | Classification |
| --- | ---: | ---: | --- |
| c1 graph | **116.022 tok/s** | 116.022 tok/s | independent reference |
| serial c2 bridge | **100.925 tok/s** | 50.462 tok/s | exact fallback control |
| native selected-batch c2 | **121.923 tok/s** | 60.962 tok/s | **retained direct c2; 1.0509x c1 / 1.2081x serial** |

The three selected samples are 121.357/121.923/121.953 aggregate tok/s. All
274 recorded IDs per run match independent c1, the ten-prompt category and
heldout suite matches 330/330 IDs, and all-layer/lifecycle/profiler gates pass.
Artifact: [`gfx1100 selected-batch c2`](results/2026-07-18-gfx1100-paro-g2-selected-batch-c2-retained.json).

Radeon 8060S/gfx1151 retains true physical c2/c4/c8. Physical c4/c8
generalize the selected-batch algorithm; they do not stack c2 groups. G5 now
attaches those identity-matched widths to the fixed-capacity resident owner and
selects them by gfx1151 backend-package capability when the legacy flags are
unset. Public `LLM`, blocking OpenAI, and concurrent SSE share the same stable-
slot session; explicit `HIPENGINE_QWEN35_{RETAINED_BATCH_DEFAULTS,
EXPERIMENTAL_NATIVE_BATCH_DECODE}=0` values remain rollback opt-outs.

| gfx1151 direct route | Median aggregate decode | Per-request decode | Classification |
| --- | ---: | ---: | --- |
| c1 graph | **70.810 tok/s** | 70.810 tok/s | independent reference |
| serial c2 bridge | **65.574 tok/s** | 32.787 tok/s | exact fallback control |
| native selected-batch c2 | **79.237 tok/s** | 39.619 tok/s | **retained direct c2; 1.1190x c1 / 1.2084x serial** |
| true physical c4 | **100.209 tok/s** | 25.052 tok/s | **retained direct c4; 1.4152x c1** |
| true physical c8 | **99.943 tok/s** | 12.493 tok/s | **retained direct c8; 1.4114x c1** |

Three fresh-process direct c2/c4/c8 samples have at most **0.054%**
stdev/median and match **5,754/5,754** recorded independent-c1 IDs. c8 aggregate
throughput is **0.265% below c4**, exposing a genuine c4 bandwidth plateau,
while its median model-step time is still **0.183% faster than two sequential
c4 steps**. The all-layer hidden/Conv/GDN/NumPy-context/KV gate, sparse c8->c1
cancel/EOS/ragged immutability, c4/c8 primitives, both ten-prompt category+
heldout gates, and the cached 4,644-dispatch c8 profiler pass with 40 selected-
batch and zero fallback layers.

The clean socket-level blocking F1 packet uses 512 raw prompt IDs, 128 generated
IDs/request, one discarded warmup, three measured bursts, a fresh server per
width, and barrier-to-last-response localhost HTTP wall. Every one of **68/68**
warmup/measured/live rows matches its independent c1 ID oracle, actual resident
prompt IDs, exact usage, finish metadata, and native route. Static rate variance
is at most **0.994%** and c1/c2/c4/c8 GTT peaks are
**18.373/18.840/19.461/20.594 GiB**.

| Blocking OpenAI client c | Aggregate generated tok/s | Per-request tok/s | Scale vs c1 |
| ---: | ---: | ---: | ---: |
| 1 | **47.124** | 47.124 | 1.000x |
| 2 | **51.962** | 25.981 | **1.103x** |
| 4 | **60.323** | 15.081 | **1.280x** |
| 8 | **61.253** | 7.657 | **1.300x** |

The complementary in-process real FastAPI SSE packet uses exact-roundtrip text
transport plus authoritative reclaim IDs. All **100/100** c1/native-c2/c4/c8/
serial-c8 warmup, measured, and live rows are exact and drain ownership. Median
aggregate rates are **36.327/38.666/42.471/41.487/35.633 tok/s** respectively:
native c2/c4/c8 are **1.064x/1.169x/1.142x c1**, while c8 is **1.164x** the
same-loop serial-c8 control. Delayed c4->c8 admission remains exact at
**38.191 tok/s**. A separate c8 1+7 stress packet adds **72/72** exact static/
live rows; combined with the earlier 40 exact c8 rows and 192 direct staggered
rows, it did not reproduce the one prior diagnostic row divergence. A no-native-
flag OpenAI c4 confirmation runs from `/tmp`, requires the packaged profile,
observes physical widths 2 and 4, keeps **4/4** rows exact, records no fallback
reason, and drains ownership; its d16 no-warmup wall is routing evidence, not a
performance row.

c3/c5/c6/c7 remain exact partitions of certified physical groups, not new native
width claims, and have no separate retained server-speed rows.

<!-- BEGIN TOPLINE:GFX1151_PARO_CURRENT -->
| Client c | Production backend groups | Exact classification | Retained OpenAI aggregate |
| ---: | --- | --- | ---: |
| 1 | `1` | c1 oracle / accepted | **47.124 tok/s** |
| 2 | native `2` | retained physical width | **51.962 tok/s** |
| 3 | `2+1` | exact partition; not native c3 | no separate claim |
| 4 | native `4` | retained physical width | **60.323 tok/s** |
| 5 | `4+1` | exact partition; not native c5 | no separate claim |
| 6 | `4+2` | exact partition; not native c6 | no separate claim |
| 7 | `4+2+1` | exact partition; not native c7 | no separate claim |
| 8 | native `8` | retained physical width | **61.253 tok/s** |
<!-- END TOPLINE:GFX1151_PARO_CURRENT -->

Artifacts: [`direct c2/c4/c8`](results/2026-07-18-gfx1151-paro-g3-native-c248-direct-retained.json),
[`G5 blocking F1`](results/2026-07-18-gfx1151-paro-g5-f1-server-scaling.json),
[`G5 SSE`](results/2026-07-18-gfx1151-paro-g5-sse-server-scaling.json),
[`c8 repeatability`](results/2026-07-18-gfx1151-paro-g5-c8-sse-repeatability.json),
[`package-default OpenAI c4`](results/2026-07-18-gfx1151-paro-g5-default-openai-c4.json),
plus historical [P1 exact catalog](results/2026-07-11-sol-p1-gfx1151-paro-c1-c8-exact-catalog.json),
[P2 ragged lifecycle](results/2026-07-11-sol-p2-gfx1151-paro-ragged-lifecycle.json),
and [G4 resident OpenAI correctness](results/2026-07-18-gfx1151-paro-g4-resident-openai-correctness.json).

The retained gfx1100 and gfx1151 HIP/Vulkan timing-contract v2 micro matrices
are linked from the platform index and
[`docs/HIP-vs-VULKAN.md`](../docs/HIP-vs-VULKAN.md); they are not
model-throughput toplines.


## Merged UD-Q3_K_M GPU1 and W7900 Records

These rows retain exact direct/native evidence under the original RX 7900 XTX
GPU1 and Radeon Pro W7900 scopes. They do not replace the W7900 Q4_K_M or
project-wide serving toplines above. The three W7900 rows are the
correctness-first branch baseline measured at `44a1f963`; current production Q3
code is the later optimized route represented by the GPU1 rows.

| Model | Quant | Backend | Workload | Prefill tok/s | Decode tok/s | Peak GiB | Correctness | Artifact | Last updated | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| Qwen3.6-35B-A3B GGUF | gguf_ud_q3_k_m BF16 KV | `hip_gfx1100` Radeon Pro W7900 historical branch | 512/128 repeated-token bulk prefill + graph decode | 614.089 | 92.285 | 15.692 | finite and bit-identical logits across three measured runs; stable token `9707`; selected-kernel CPU-oracle and zero-scratch gates pass | [`2026-07-19-w7900-qwen36-q3-k-m-benchmark.json`](results/2026-07-19-w7900-qwen36-q3-k-m-benchmark.json) | 2026-07-19 | Historical correctness-first branch baseline; superseded as implementation evidence by the optimized merged Q3 route. |
| Qwen3.6-35B-A3B GGUF | gguf_ud_q3_k_m BF16 KV | `hip_gfx1100` Radeon Pro W7900 historical branch | 1K/128 repeated-token bulk prefill + graph decode | 623.583 | 97.373 | 15.759 | finite and bit-identical logits across three measured runs; stable token `9707`; exact final positions and graph replay pass | [`2026-07-19-w7900-qwen36-q3-k-m-benchmark.json`](results/2026-07-19-w7900-qwen36-q3-k-m-benchmark.json) | 2026-07-19 | Same pinned model, compiler, and one-warmup/three-median protocol as the adjacent W7900 rows. |
| Qwen3.6-35B-A3B GGUF | gguf_ud_q3_k_m BF16 KV | `hip_gfx1100` Radeon Pro W7900 historical branch | 4K/128 repeated-token bulk prefill + graph decode | 616.135 | 98.111 | 16.134 | finite and bit-identical logits across three measured runs; stable token `9707`; 4K final position and graph replay pass | [`2026-07-19-w7900-qwen36-q3-k-m-benchmark.json`](results/2026-07-19-w7900-qwen36-q3-k-m-benchmark.json) | 2026-07-19 | Descriptive same-model Vulkan comparison only; raw source sweeps remain in merged commit `d47e63cd`. |
| Qwen3.6-35B-A3B GGUF | gguf_ud_q3_k_m BF16 KV | `hip_gfx1100` RX 7900 XTX GPU1 native rows | c=2 512/128 | 864.569 | 118.125 | 15.903 | all 129 sampled IDs (128 timed native decode steps) and stateful full logits are exact vs independent c=1 | [`2026-07-21-gpu1-q3-native-cn-retained.json`](results/2026-07-21-gpu1-q3-native-cn-retained.json) | 2026-07-21 | 59.062 tok/s/request; 16.865/17.543 ms p50/p95; aggregate is 1.167× retained c=1. |
| Qwen3.6-35B-A3B GGUF | gguf_ud_q3_k_m BF16 KV | `hip_gfx1100` RX 7900 XTX GPU1 native rows | c=4 512/128 | 864.549 | 151.772 | 16.059 | exact generated IDs/full logits vs independent c=1; varied-prompt confirmation 151.638 tok/s | [`2026-07-21-gpu1-q3-native-cn-retained.json`](results/2026-07-21-gpu1-q3-native-cn-retained.json) | 2026-07-21 | 37.943 tok/s/request; 26.124/27.964 ms p50/p95; aggregate is 1.499× c=1. |
| Qwen3.6-35B-A3B GGUF | gguf_ud_q3_k_m BF16 KV | `hip_gfx1100` RX 7900 XTX GPU1 native rows | c=8 512/128 | 863.901 | 207.780 | 16.372 | exact generated IDs/full logits vs independent c=1; rocprof shows indexed Conv/GDN, row-batched paged attention, selected-row MoE, row lm-head, and row argmax | [`2026-07-21-gpu1-q3-native-cn-retained.json`](results/2026-07-21-gpu1-q3-native-cn-retained.json) | 2026-07-21 | 25.973 tok/s/request; 38.547/39.917 ms p50/p95; 2.053× c=1; varied prompts confirm 210.640 tok/s. |
| Qwen3.6-35B-A3B GGUF | gguf_ud_q3_k_m BF16 KV | `hip_gfx1100` RX 7900 XTX GPU1 native rows | c=2 4K/128 | 900.310 | 130.276 | 17.226 | exact generated IDs vs independent c=1; native split-GQA attention path and full-logit boundary gates pass | [`2026-07-21-gpu1-q3-native-cn-retained.json`](results/2026-07-21-gpu1-q3-native-cn-retained.json) | 2026-07-21 | 65.138 tok/s/request; 15.188/16.056 ms p50/p95; aggregate is 1.202× c=1. |
| Qwen3.6-35B-A3B GGUF | gguf_ud_q3_k_m BF16 KV | `hip_gfx1100` RX 7900 XTX GPU1 native rows | c=4 4K/128 | 897.687 | 157.926 | 17.520 | exact generated IDs vs independent c=1; C=4 full-logit native-row gate passes | [`2026-07-21-gpu1-q3-native-cn-retained.json`](results/2026-07-21-gpu1-q3-native-cn-retained.json) | 2026-07-21 | 39.481 tok/s/request; 25.238/26.022 ms p50/p95; aggregate is 1.457× c=1. |
| Qwen3.6-35B-A3B GGUF | gguf_ud_q3_k_m BF16 KV | `hip_gfx1100` RX 7900 XTX GPU1 native rows | c=8 4K/128 | 894.036 | 211.177 | 18.107 | exact generated IDs vs independent c=1; C=8 full-logit gate and native graph provenance pass | [`2026-07-21-gpu1-q3-native-cn-retained.json`](results/2026-07-21-gpu1-q3-native-cn-retained.json) | 2026-07-21 | 26.397 tok/s/request; 37.888/38.365 ms p50/p95; aggregate is 1.948× c=1. |
| Qwen3.6-35B-A3B GGUF | gguf_ud_q3_k_m | `hip_gfx1100` RX 7900 XTX GPU1 fully-bulk + guarded residual-D4 MMQ wide-Q8 + exact Q8 fallbacks + attention GQA-batch + IQ3 rowbatch4 + GDN LDS32 + IQ4-down wave32 defaults | 512/0 repeated-token prefill | 848.543 | — | 15.821 | Final 18-workload x 9-position continuation suite is logit-bit-exact to the exact-tile control (`KL=0`, top-1 `1.0`); all-queued sparse repair is BF16-bit exact | [`2026-07-20-gpu1-q3-guarded-d4x3-mmq-prefill.json`](results/2026-07-20-gpu1-q3-guarded-d4x3-mmq-prefill.json) | 2026-07-20 | Post-hardening official five-run median moves the prior retained `774.185 -> 848.543 tok/s` (+9.60%, 0.22% stdev); matched mixed-pattern A/B is `760.411 -> 837.417` (+10.13%). The bounded queue adds 16 MiB; non-admitted shapes retain exact fallbacks. |
| Qwen3.6-35B-A3B GGUF | gguf_ud_q3_k_m | `hip_gfx1100` RX 7900 XTX GPU1 fully-bulk + guarded residual-D4 MMQ wide-Q8 + exact Q8 fallbacks + attention GQA-batch + IQ3 rowbatch4 + GDN LDS32 + IQ4-down wave32 defaults | 4K/0 mixed-pattern prefill | 831.393 | — | 17.080 | Matched mixed-pattern exact/candidate full logits preserve token `14626`, KL `0`, and top-1; the 18-workload continuation and focused primitive/full-model gates pass | [`2026-07-20-gpu1-q3-guarded-d4x3-mmq-prefill.json`](results/2026-07-20-gpu1-q3-guarded-d4x3-mmq-prefill.json) | 2026-07-20 | Post-hardening matched mixed-pattern A/B moves `743.906 -> 831.393 tok/s` (+11.76%; +12.17% vs prior retained 741.180); official repeated-token median is 828.003. Cached dense Q8 falls `2,052.066 -> 1,569.232 ms` (-23.53%), total kernel sum `5,350.508 -> 4,815.413 ms` (-10.00%), and trace span falls to 4,973.718 ms; the bounded queue adds 128 MiB. |
| Qwen3.6-35B-A3B GGUF | gguf_ud_q3_k_m | `hip_gfx1100` RX 7900 XTX GPU1 historical native-attention control | 512/128 native-attention + grouped scalar MoE + one-step graph decode | 16.685 | 101.216 | 15.805 | Mixed-64 native-row-bulk vs serial: KL `0`, top-1 `1.0`, max abs `0`; current tail A/B IDs `[220]*6`; 4K IDs/logit exact across both pairs; selected IQ/tail symbols have zero scratch/copies | [`2026-07-20-gpu1-q3-moe-tail-next-rms-retained.json`](results/2026-07-20-gpu1-q3-moe-tail-next-rms-retained.json) | 2026-07-20 | Superseded as the prefill default by the exact fully-bulk rows above; its `101.216 tok/s` result remains the retained decode score. The grouped-prefill portion remains a verified sub-window/default promotion rather than a prefill headline: paired direct is 16.648 tok/s (+0.22%, within 2.08%/1.60% spread), while raw-IQ time is `994.668 -> 613.995 ms` (-38.27%) and total kernel sum `4396.145 -> 4078.667 ms` (-7.22%). `HIPENGINE_GGUF_IQ_GROUPED_PREFILL=0` retains rollback. Task #20 hierarchical top-k was exact but rejected on graph wall (`99.201 -> 98.057 tok/s`, -1.15%); [`rejection artifact`](results/2026-07-20-gpu1-q3-hierarchical-topk-rejected.json). The final-tree task-#32 D0 records `8.82493 ms/token` and `671` launches/token after the retained tail/RMS fusion; dense Q8 remains first at `2.83934 ms/token` (32.17%), followed by attention at `1.42134` (16.11%), lm-head Q6 at `1.05068` (11.91%), weighted IQ4 down at `1.00066` (11.34%), and IQ3 gate/up at `0.70532` (7.99%); [`final D0 artifact`](results/2026-07-22-gpu1-q3-final-decode-d0-profile.json). Its single follow-up premise is also closed: source-shaped raw-Q8 block serialization cut three representative leaves 34–55% but changed every full logit at 512/1K/4K, while exact association regressed those leaves 21–80%; candidate code was removed and the retained wall row is unchanged; [`rejection artifact`](results/2026-07-22-gpu1-q3-q8-blockserial-decode-rejected.json). D1B output tile4 was removed after a 12.50% real-family regression. D1C's retained wave-uniform IQ3 block base reduces IQ3 `11.4966 -> 11.2614 ms` (-2.05%), VGPR `48 -> 40`, and counterbalanced graph decode `100.334 -> 100.536 tok/s` (+0.20%) with exact 512/1K/4K IDs/logits; that within-noise sample did not replace the prior `100.573 tok/s` headline. [`D1C artifact`](results/2026-07-20-gpu1-q3-iq3-wave-base-retained.json). Task #21 now fuses 37 already-weighted MoE tails with the next input RMSNorm, removes 37 graph nodes/token (`708 -> 671`), and moves same-suite graph decode `100.195 -> 101.216 tok/s` (+1.02%) at 512 and `107.366 -> 108.383 tok/s` (+0.95%) at 4K with all five pairs positive. The 16.685 prefill metric is intentionally retained from the grouped-prefill gate because this decode-only route does not touch bulk prefill; slot-weighted Q3/Q4/PARO boundaries retain the exact two-kernel fallback. [`Task-21 artifact`](results/2026-07-20-gpu1-q3-moe-tail-next-rms-retained.json). |
| Qwen3.6-35B-A3B GGUF | gguf_ud_q3_k_m | `hip_gfx1100` RX 7900 XTX GPU1 raw-IQ direct session | 512/128 native-attention + direct `x_rows` MoE + one-step graph decode | 19.452 | 99.015 | 15.805 | Public first token matches pinned llama.cpp; resident eager/graph `[11,11,264]`, KL `0`, top-1 `1.0`; three benchmark IDs stable `[220]*3`; `performance_claim=false` | [`2026-07-19-gpu1-hipengine-qwen36-35b-a3b-ud-q3km-direct-baseline.json`](results/2026-07-19-gpu1-hipengine-qwen36-35b-a3b-ud-q3km-direct-baseline.json) | 2026-07-19 | Correctness/optimization control, not a speed promotion. Selected profiles show 185,078 prefill launches, 708 dispatches/decode token, exact IQ traffic `424,280,064 bytes/token`, and no IQ scratch; same-model qwen-kernel `829.30/189.96` is contextual and was not rerun. |

| Lane | Status | Workload | Same-session AR tok/s | Spec tok/s | Ratio | Correctness | Artifact / source | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| Qwen3.6-35B-A3B GGUF UD-Q3_K_M + blk.40 NextN | **diagnostic no-hold / default disabled** | GPU1 RX 7900 XTX/gfx1100, matched raw `code_python` prompt (21 tokens), D16, fixed B=1/2/3, exact eager shared verifier with stable shape buckets | 8.832 / 9.247 / 8.902 | 4.801 / 3.201 / 2.413 | **0.544x / 0.346x / 0.271x** | exact greedy IDs at every budget; B=1/2/3 full target logits equal scalar; GPU accept equals CPU oracle; reject/partial/full state and live-KV prefix are exact | [`2026-07-21-gpu1-q3-gguf-mtp-e2e-nohold.json`](results/2026-07-21-gpu1-q3-gguf-mtp-e2e-nohold.json) | One accepted draft token over 14 cycles (`1.071` visible/cycle) at every budget. The explicit diagnostic route uses candidate-only `DraftBatch`, root-prefixed `TargetVerifyBatch`, `KVLiveSpans(span_role=verify_chain)`, scheduler `KVTransaction`, GPU accept summary, and stable graph-shaped buffers. Wider B pays more exact target rows without density gain; no public/default promotion. |

## Platform Records And Diagnostics

The dated records below preserve scoped retained rows plus diagnostic protocols,
blockers, commands, and artifact links. Removed or superseded tables remain
recoverable from the linked compact artifacts, changelog, and
[`benchmarks/HISTORY.md`](HISTORY.md).

### gfx1100 Laguna S 2.1 UD-Q2_K_XL target AR, 2026-07-28

Last updated: **2026-07-28**.

**Status: exact dense decode, IQ3 ten-wave fused weighted-down ownership with
wave4 fallback, P2 exact split attention plus SWA tile16 scores, P4.1
split-reducer+gate, current-P4 head RMSNorm+RoPE+BF16-KV, wave-local exact SWA
split reduction, expanded-magnitude IQ2 gate/up, raw-Q5 wave32x2 fixed-metadata
loads, all-local32 Q5/Q6 plus retained local128 Q6/Q8 attention-projection
quads, exact local32 Q4 LM head, token4 SWA, raw-Q6 attention pairing, and
aggregate MoE-tail plus next-RMS are the retained W7900 target-only AR
default.**
The exact D10 token8 SWA candidate improved every clean mechanical profile and
h32 decode but failed aggregate/every-category h16 non-regression. The exact
D11 persistent router removed 47 launches/token and improved isolated router/
span/child rows, but failed the clean short kernel-sum gate. The exact D13 Q5
shared pair+SiLU leaf shortened its local launch window but regressed total
kernel sum at every context. D14's exact head/RoPE+KV composite passed every
mechanical row and improved every category's decode, but failed aggregate and
per-category E2E/TTFT non-regression. D15 paired D14 with exact attention+gate
leaves and improved every mechanical row plus aggregate decode/E2E, but missed
code h16 E2E and the aggregate TTFT guard. D17 replaced D15's SWA leaf with
D10's exact token8 schedule, passed every mechanical and category decode/E2E
row, and crossed 50 tok/s, but still missed the aggregate TTFT guard. All six
historical candidates are removed. The historical D14 route was not restored;
its exact body was separately recomposed and regated over retained P2/P4.1 as
the current-P4 default described below. Clean measured D12 implementation revision
`338d3afca01aa884ff3a68e0175566bc51e5ceae` runs the pinned
`Laguna-S-2.1-UD-Q2_K_XL.gguf`
(SHA-256 `8fe1170f012723f6f7d6c9b08d8f928b0b3d8bffc32926f33a930148a1d62679`)
directly from raw GGUF residency with BF16 KV and a 4-GiB safety reserve. The
canonical protocol covers all 18 `mtpbench-code-general-ja` train+heldout
prompts, all four categories, prompt lengths 68-122, two balanced repetitions,
greedy h16/h32, 128-row prompt chunks, and c=1 eager decode. Model load is
excluded. D0 is the original `09cca232` baseline; D1 moves raw Q4/Q5/Q6/Q8
rows=1 projections to exact decode-specialized leaves; D2 removes four idle
wave32 units from exact IQ3 selected-down K1024 launches; D3 preserves each
route projection's BF16 boundary while contracting scaled routing into that
selected-down leaf. D4 computes four exact SWA slot dots concurrently, then
consumes softmax and values in the baseline logical-slot order. D5 combines
each same-input K3072/N1024 raw-Q5 shared gate/up pair into one registered
launch while preserving both singleton reduction trees and BF16 stores. D6
similarly flattens each unequal-width raw-Q5 attention query/per-head-gate pair
into one exact F32 pair launch at N6144+48 or N9216+72. D7 combines 47
same-input raw-Q6 K/V pairs plus layer 47's Q6 query/per-head-gate pair while
preserving every singleton reduction tree and F32 output. D9 contracts each of
47 sparse BF16 add/add/next-RMS boundaries into one dual-output local256 leaf,
preserving both add roundings and the exact RMS reduction order. D12 replaces
the 47 raw-Q5 attention-output and 47 unequal query/gate pack8 calls with
local32 two-output siblings that reconstruct the same four logical wave
partials without LDS or barriers. P0 then replaces 45 serial weighted IQ3 down
calls with one exact local32 wave per `(route, output)` plus the registered
slot-order reducer.

| hipEngine route | Prefill tok/s | Median TTFT | Decode tok/s, h32 | E2E tok/s, h16 | E2E tok/s, h32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| D0 exact bulk prefill + generic dense decode | 40.091 | 2.018 s | 19.565 | 5.488 | 8.557 |
| D1 exact bulk prefill + exact dense decode | 40.401 | 2.000 s | 35.419 | 6.258 | 10.618 |
| D2 exact D1 + IQ3 K1024 local128 | 43.266 | 1.865 s | 38.301 | 6.713 | 11.403 |
| D2 change vs D1 | +7.093% | -6.743% | +8.135% | +7.279% | +7.393% |
| D3 exact D2 + IQ3 weighted down | 43.264 | 1.865 s | 38.840 | 6.728 | 11.448 |
| D3 change vs D2 | -0.004% | +0.016% | +1.407% | +0.218% | +0.399% |
| D4 exact D3 + token4 SWA | 43.168 | 1.870 s | 43.081 | 6.815 | 11.760 |
| D4 change vs D3 | -0.223% | +0.272% | +10.919% | +1.298% | +2.724% |
| D5 exact D4 + Q5 shared pair | 43.167 | 1.871 s | 44.501 | 6.849 | 11.860 |
| D5 change vs D4 | -0.003% | +0.066% | +3.298% | +0.499% | +0.849% |
| D6 exact D5 + Q5 query/gate pair | 43.159 | 1.870 s | 45.433 | 6.869 | 11.921 |
| D6 change vs D5 | -0.018% | -0.081% | +2.093% | +0.290% | +0.518% |
| D7 exact D6 + Q6 attention pairs | 43.093 | 1.873 s | 46.409 | 6.881 | 11.972 |
| D7 change vs D6 | -0.152% | +0.172% | +2.147% | +0.168% | +0.423% |
| D9 exact D7 + aggregate MoE-tail/next-RMS | 43.190 | 1.871 s | 47.132 | 6.909 | 12.038 |
| D9 change vs D7 | +0.224% | -0.117% | +1.560% | +0.411% | +0.555% |
| D9 counterbalanced control for D12 | 43.021 | 1.873 s | 47.046 | 6.884 | 11.997 |
| D12 exact D9 + raw-Q5 wave32x2 | 43.028 | 1.874 s | 48.987 | 6.923 | 12.117 |
| D12 change vs paired D9 | +0.016% | +0.032% | +4.124% | +0.564% | +1.001% |
| P0 matched D12 control | 43.017 | 1.871 s | 48.780 | 6.917 | 12.103 |
| P0 exact D12 + IQ3 wave4 | **42.992** | **1.877 s** | **50.254** | **6.941** | **12.183** |
| P0 change vs matched D12 | **-0.057%** | **+0.280%** | **+3.022%** | **+0.339%** | **+0.666%** |
| Current-P4 matched P4.1 control | 42.961 | 1.935 s | 51.872 | 6.925 | 12.207 |
| Current-P4 exact head RMSNorm+RoPE+KV | **42.949** | **1.935 s** | **52.391** | **6.932** | **12.232** |
| Current-P4 change vs matched P4.1 | **-0.029%** | **-0.008%** | **+1.001%** | **+0.106%** | **+0.204%** |
| Wave-local matched shared-reducer control | 42.962 | 1.937 s | 52.211 | 6.931 | 12.225 |
| Exact wave-local SWA split reducer | **42.907** | **1.936 s** | **52.514** | **6.928** | **12.229** |
| Wave-local change vs matched shared reducer | **-0.126%** | **-0.027%** | **+0.580%** | **-0.047%** | **+0.033%** |
| IQ2 compact-grid matched control | 42.909 | 1.937 s | 52.650 | 6.930 | 12.237 |
| Exact IQ2 expanded-magnitude grid | **42.878** | **1.938 s** | **54.540** | **6.956** | **12.326** |
| Expanded-grid change vs matched compact grid | **-0.072%** | **+0.042%** | **+3.590%** | **+0.373%** | **+0.730%** |
| Q5 coefficient-publication matched control | 42.967 | 1.937 s | 54.476 | 6.967 | 12.343 |
| Exact Q5 fixed metadata | **42.923** | **1.937 s** | **57.711** | **7.008** | **12.487** |
| Fixed-metadata change vs matched control | **-0.103%** | **+0.022%** | **+5.938%** | **+0.582%** | **+1.163%** |
| Projection pair/singleton matched control | 42.955 | 1.937 s | 57.833 | 7.014 | 12.499 |
| Exact mixed attention projections | **42.887** | **1.938 s** | **58.425** | **7.013** | **12.510** |
| Mixed-projection change vs matched control | **-0.158%** | **+0.035%** | **+1.024%** | **-0.021%** | **+0.087%** |
| Generic-Q6 mixed-projection matched control | 42.963 | 1.937 s | 58.466 | 7.024 | 12.530 |
| Exact fixed-Q6 metadata mixed projections | **42.887** | **1.936 s** | **59.211** | **7.023** | **12.545** |
| Fixed-Q6 metadata change vs matched control | **-0.177%** | **-0.042%** | **+1.275%** | **-0.018%** | **+0.121%** |
| Generic shared-Q5 pair matched control | 43.008 | 1.934 s | 59.500 | 7.045 | 12.586 |
| Exact fixed-metadata shared-Q5 pair | **42.938** | **1.936 s** | **60.942** | **7.053** | **12.631** |
| Shared-Q5 fixed-metadata change vs matched control | **-0.163%** | **+0.091%** | **+2.425%** | **+0.118%** | **+0.357%** |
| Local128 fixed-Q6 mixed-projection matched control | 42.966 | 1.938 s | 60.900 | 7.056 | 12.635 |
| Exact all-local32 Q5/Q6 mixed projections | **42.883** | **1.938 s** | **61.732** | **7.055** | **12.650** |
| All-local32 mixed-projection change vs matched control | **-0.192%** | **+0.000%** | **+1.367%** | **-0.025%** | **+0.118%** |
| Local128 Q4 LM-head matched control | 42.893 | 1.941 s | 61.675 | 7.055 | 12.650 |
| Exact local32 Q4 LM head | **42.804** | **1.941 s** | **61.992** | **7.046** | **12.642** |
| Local32 Q4 LM-head change vs matched control | **-0.208%** | **+0.028%** | **+0.512%** | **-0.131%** | **-0.066%** |
| IQ3 wave4 + weighted-reducer matched control | 42.961 | 1.935 s | 62.318 | 7.073 | 12.692 |
| Exact IQ3 wave10-fused weighted down | **42.880** | **1.937 s** | **63.270** | **7.073** | **12.711** |
| Wave10-fused change vs matched wave4 | **-0.189%** | **+0.105%** | **+1.528%** | **-0.008%** | **+0.145%** |
| D3 token-serial control | 44.396 | 1.800 s | 39.000 | 6.883 | 11.675 |

A fresh same-source HIP closure uses a frozen ABBA engine order rather than the
older forced-token/F16-KV `llama-bench` diagnostic. Each process runs four
repetitions of all 18 natural-greedy train+heldout prompts at h16/h32 with the
same model bytes, prompt token streams, BF16 K/V, FA-on policy, context 4096,
one W7900 queue, and post-TTFT timing boundary. Rates are pooled from raw
seconds rather than averaged:

| Matched natural completion | hipEngine | llama.cpp HIP | hipEngine delta |
| --- | ---: | ---: | ---: |
| h16, 144 runs / 2,160 transitions each | **64.094 tok/s** | 49.290 tok/s | **+30.034% / 1.300x** |
| h32, 144 runs / 4,464 transitions each | **63.431 tok/s** | 49.964 tok/s | **+26.954% / 1.270x** |

Both hipEngine processes pass Poolside KL **0.000156823**, top-1 **100%**, exact
serial/bulk/repeat IDs, stable IDs across processes, and zero final tracked
ownership. llama.cpp uses verified source `c0bc8591e` plus one declared
post-generation content-only response patch; its complete 269-file HIP bundle is
byte-identical to the clean build. Every native `prompt_n`, `predicted_n`, and
`predicted_ms` row is valid. c0bc sometimes omits SSE token-array entries, so
those arrays and cross-engine ID matches are diagnostics, not timing ownership.
This is a true 1:1 **protocol/storage/timing** comparison, not a claim of
bit-identical arithmetic: hipEngine retains `KVLiveSpans` and its exact kernels,
while llama.cpp retains ggml scheduling and reduction order. It also does not
supersede or claim victory over the separately pinned Vulkan target. Evidence:
[`matched ABBA artifact`](results/2026-07-28-gfx1100-laguna-q2-xl-hipengine-vs-llamacpp-hip-matched-abba.json).

P0 also pools two complete process-order pairs. Every category improves h16/h32
decode by **2.80-3.16%** and E2E by **0.30-0.76%**; unaffected prefill is
**-0.057% aggregate** and remains within **-0.152% to +0.077%** by category.
Full logits, all 48 hidden boundaries, all 47 routed outputs, active KV/
`KVLiveSpans`, reset, IDs, Poolside quality, and lifecycle are exact. The
[gfx1100 P0 retained artifact](results/2026-07-24-gfx1100-laguna-q2-xl-p0-iq3-wave4-retained.json)
pins all raw hashes and the unchanged 40.068-GB resident footprint.

D12 pools two complete process-order pairs (control/candidate then candidate/
control), for four effective repetitions and 40 runs per mode. Every category
improves h16/h32 decode by **3.88-4.49%** and E2E by **0.49-1.24%**; unaffected
prefill is **+0.016% aggregate** and stays within **-0.020% to +0.067%** by
category. Every ID/repeat, Poolside quality, full hidden/KV/`KVLiveSpans`, and
lifecycle gate passes. The counterbalance is required because each one-way
process pair retained small order-correlated drift in the unchanged prefill
lane. [D12 retained artifact](results/2026-07-24-gfx1100-laguna-q2-xl-d12-q5-wave32x2-retained.json).

The two D9 default samples are stable: bulk prefill is `43.457/42.925 tok/s`,
h32 decode is `47.352/46.915 tok/s`, and h32 E2E is `12.108/11.969 output
tok/s`. Versus D7, every category improves h32 decode **0.956-1.838%** and
h32 E2E **0.333-0.759%**; unchanged category prefill stays within **+0.114%
to +0.384%**. All 20 serial/bulk pairs are exact at both horizons, every route
repeats deterministically, the independent Poolside first-token gate passes at
KL `0.000156823` and top-1 `1.0`, and tracked peak ownership is
**40,455,911,848 bytes (37.678 GiB)** before teardown returns exactly to zero.

The source harness's serial-vs-bulk predicate remains false because both modes
use the same c=1 decode while token-serial prompt prefill is faster. That is
disclosed rather than hidden: D9 retention compares clean D7 bulk against clean
D9 bulk on the identical full suite. Decode and E2E improve in every category,
while unchanged prompt prefill remains within the declared 0.5% guard. [D9
retained artifact](results/2026-07-24-gfx1100-laguna-q2-xl-d9-moe-tail-next-rms-retained.json).
The [D7 artifact](results/2026-07-23-gfx1100-laguna-q2-xl-q6-attention-pair-retained.json),
[D6 artifact](results/2026-07-23-gfx1100-laguna-q2-xl-q5-query-gate-pair-retained.json),
[D5 artifact](results/2026-07-23-gfx1100-laguna-q2-xl-q5-shared-pair-retained.json),
[D4 artifact](results/2026-07-23-gfx1100-laguna-q2-xl-swa-token4-retained.json),
[D3 artifact](results/2026-07-23-gfx1100-laguna-q2-xl-iq3-weighted-down-retained.json),
[D2 artifact](results/2026-07-23-gfx1100-laguna-q2-xl-iq3-local128-retained.json),
[D1 artifact](results/2026-07-23-gfx1100-laguna-q2-xl-dense-decode-retained.json),
and [D0 artifact](results/2026-07-23-gfx1100-laguna-q2-xl-target-ar.json)
remain frozen baselines.

The exact D8 one-step HIP graph screen is rejected and removed. A 956-step
state trajectory is byte-exact, but counterbalanced steady-state graph
throughput is **2.247%/1.995%/1.502%/1.146% slower** than eager at
short/512/1K/near-4K. Capture-inclusive canonical h16/h32 decode falls
**46.827/46.409 -> 43.480/44.193 tok/s (-7.150%/-4.774%)**, while h16/h32 E2E
falls **6.881/11.972 -> 6.819/11.839 (-0.902%/-1.110%)**; every category
regresses both horizons. [D8 rejection
artifact](results/2026-07-23-gfx1100-laguna-q2-xl-decode-graph-rejected.json).

The exact D10 token8 SWA screen is also rejected and removed under the stricter
non-regression rule. It improves clean short/512/1K/near-4K SWA by
**12.47%/14.81%/14.74%/14.86%**, complete span by
**0.70%/5.81%/5.32%/3.70%**, and diagnostic h32 decode
**47.132 -> 47.872 tok/s (+1.569%)**. However aggregate h16 E2E changes
**6.909 -> 6.905 (-0.055%)**, general-English h16 decode/E2E changes
**-0.535%/-0.254%**, and code/mixed h16 E2E changes **-0.128%/-0.017%**. The
required aggregate and every-category h16/h32 decode/E2E predicate therefore
fails despite exact output/state/KV/lifecycle. Token8 source/dispatch/tests are
removed; eager token4 remained the D9 default at that decision. D12 later
superseded the headline without reviving token8. [D10 rejection
artifact](results/2026-07-24-gfx1100-laguna-q2-xl-d10-swa-token8-rejected.json).

Exact current-default D12 benchmark command:

```bash
HIP_VISIBLE_DEVICES=0 HIPENGINE_HIP_ARCH=gfx1100 GPU_MAX_HW_QUEUES=1 HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version-laguna-iq2.txt HIPENGINE_REQUIRE_CACHED_BUILD=1 PYTHONPATH=. uv run python -u scripts/laguna_target_ar_bench.py /models/gguf/Laguna-S-2.1-UD-Q2_K_XL.gguf --prompts benchmarks/prompts/laguna-target-ar-code-general-ja-heldout.jsonl --template tests/fixtures/laguna_poolside_v1_template.json --oracle tests/fixtures/laguna_poolside_q2_xl_v1_oracle.json --oracle-logprobs tests/fixtures/laguna_poolside_q2_xl_v1_first_token_logprobs.npy --bulk-correctness-artifact benchmarks/results/2026-07-23-gfx1100-laguna-q2-xl-bulk-correctness.json --backend hip_gfx1100 --context-length 4096 --chunk-size 128 --output-horizons 16,32 --repetitions 2 --warmup-output-tokens 2 --compiler-version-file /tmp/hipengine-hipcc-version-laguna-iq2.txt --require-cached-build --direct-gguf --safety-reserve-gib 4 --model-sha256 8fe1170f012723f6f7d6c9b08d8f928b0b3d8bffc32926f33a930148a1d62679 --quant-label UD-Q2_K_XL --output /tmp/laguna-q2-xl-d12-target-ar.json
```

##### Laguna Q2 XL c=1 decode D0

A tracked-clean cached W7900 trace at `e6120872` attributes 16 true c=1 rows
from the retained AR route. The stable 14 rows average **44.572 ms/token** in
kernels across **1,055 dispatches/token**; median embedding-to-argmax span is
**49.929 ms**. Generic dense-Q5 `prefill_out` aliases alone consume **27.303 ms
/ 61.26% / 235 calls** at a **70.7 GB/s** active encoded-weight proxy. SWA
decode is 4.237 ms, selected IQ3 down 4.021 ms, fused IQ2 gate/up 2.318 ms,
dense Q6 2.006 ms, and the Q4 lm-head 1.618 ms. All 26 decode symbols are
classified and scratch-free; final logits and lifecycle pass. This is the
frozen pre-optimization bottleneck diagnostic, not the current throughput
claim. [D0 artifact](results/2026-07-23-gfx1100-laguna-q2-xl-decode-d0-profile.json).

##### Laguna Q2 XL c=1 decode D1

The clean D1 cached trace keeps the same **1,055 dispatches/token** but reduces
stable kernel sum **44.572 -> 23.142 ms/token (-48.08%)** and profiled child
wall **52.703 -> 28.820 ms/token (-45.32%)**. Exact Q5 falls **27.303 -> 7.133
ms (-73.87%)** and the Q4 lm-head falls **1.618 -> 0.376 ms (-76.75%)**.
The new Q5 symbol runs at local128, VGPR48/72, LDS1024, and scratch0. SWA
(**4.212 ms**) and selected IQ3 down (**4.040 ms**) are now the largest
individual short-context families. The raw trace SHA-256 is
`18d02d7896c43d9a6986243e562e741ff520d279d2dcc2995f207c227c61515a`;
this profile is attribution while the unprofiled full-suite D1 row above is the
performance claim.

Reduction-order-exact one-wave and two-wave SWA transfers are rejected and
removed. The SWA family changes **4.212 -> 4.274 ms (+1.49%)** short and **27.823
-> 29.016 ms (+4.29%)** at a 512-token window for wave32; two-wave is neutral
short (**4.210 ms**) but **2.93%** slower at 512. Both pass exact 508-515
wrap/eviction output gates, so this is a performance rejection rather than a
correctness failure. [Rejection
artifact](results/2026-07-23-gfx1100-laguna-q2-xl-swa-decode-rejected.json).

The retained IQ3 address-only follow-up makes each selected-down wave's
super-block base uniform. Two actual `E256/K1024/N3072/top-10` packets are
bit-exact and improve paired medians **0.69%/0.75%**; clean rocprof moves the
45-call family **4.040 -> 4.002 ms/token (-0.94%)** and total kernel sum **23.142
-> 23.097 ms (-0.20%)**. The full clean suite is non-regressive with every
category positive and h32 decode/E2E **+0.510%/+0.457%**, but the conservative
**35.419 tok/s** headline stays unchanged because that separate-process wall
delta exceeds the physically attributable leaf win. [Retained sub-window
artifact](results/2026-07-23-gfx1100-laguna-q2-xl-iq3-wave-base-retained.json).

##### Laguna Q2 XL c=1 decode D2

The K1024 IQ3 selected-down row contains exactly four wave32 work units, so D2
defaults only that shape from local256 to local128 and retains explicit
local256 rollback. Actual `blk.1.ffn_down_exps.weight` output is BF16-bit exact;
its clean paired median improves **43.61%**. Local64 was rejected after one of
30,720 actual outputs changed by one BF16 bit. Clean rocprof keeps **1,055
dispatches/token** and local128/VGPR32/LDS512/scratch0 while moving the 45-call
IQ3 family **4.002 -> 2.258 ms/token (-43.57%)**, total kernel sum **23.097 ->
21.302 ms/token (-7.77%)**, and median dispatch span to **25.524 ms**. The full
suite promotes the new **38.301 tok/s** h32 headline because every category and
all correctness/lifecycle gates pass. [D2 retained
artifact](results/2026-07-23-gfx1100-laguna-q2-xl-iq3-local128-retained.json).

##### Laguna Q2 XL c=1 decode D3

D3 replaces IQ3 selected-single plus scaled weighted-sum with one registered
routing-weighted leaf while retaining both primitives as the unfused fallback.
Every route keeps the D2 local128 reduction and BF16 projection boundary before
slot-ordered FMA. Actual `E256/K1024/N3072/top-10` output is bit-exact and the
clean paired micro median improves **18.13%**. Cached rocprof removes **45
launches/token (1,055 -> 1,010)**, moves IQ3 down plus selected reduction
**2.392 -> 2.115 ms/token (-11.61%)**, total kernel sum **21.302 -> 20.997 ms
(-1.43%)**, and median dispatch span **25.524 -> 25.037 ms (-1.91%)** at
local128/VGPR32/LDS512/scratch0. The complete suite promotes the **38.840
tok/s** h32 headline with every category's decode/E2E positive and all
correctness/lifecycle gates passing. [D3 retained
artifact](results/2026-07-23-gfx1100-laguna-q2-xl-iq3-weighted-down-retained.json).

A clean D3 context extension confirms the changed Amdahl order. At
512/1K/near-4K, stable kernel sum is **47.088/50.102/66.900 ms/token**, median
dispatch span is **51.450/54.520/71.395 ms**, and profiled child throughput is
**18.476/17.418/13.485 tok/s**. Versus the frozen D0 context trace, child
throughput improves **46.76%/44.29%/33.84%**, primarily because dense Q5 falls
about 74% to **7.12 ms/token**. SWA is unchanged at **27.776/27.846/27.901 ms**
and now consumes **58.99%/55.58%/41.71%** of kernel sum; global attention is
**2.976/5.885/22.638 ms**. These are profiler diagnostics over a deterministic
extended token stream, not retained public throughput rows. [Current D3 context
artifact](results/2026-07-23-gfx1100-laguna-q2-xl-decode-context-d3-profile.json);
[frozen D0 context artifact](results/2026-07-23-gfx1100-laguna-q2-xl-decode-context-profile.json).

##### Laguna Q2 XL c=1 decode D4

D4 assigns four logical SWA slots to four wave32 units, stores exact unscaled
dots plus physical slots in **4,120 B dynamic LDS**, and then preserves D3's
logical-slot softmax/value order. The baseline remains registered and explicitly
selectable; gfx1151 and unmeasured backends continue to use it. Focused tracing
moves six calls **792.747 -> 237.722 us median (-70.01%; 3.335x)** at
local128/VGPR24/static-LDS0/scratch0.

Clean full-model traces move SWA **4.202 -> 2.118 ms/token (-49.60%)** short and
**27.776/27.846/27.901 -> 13.111/13.096/13.104 ms/token
(-52.80%/-52.97%/-53.03%)** at 512/1K/near-4K. Corresponding kernel sums fall
**9.59%/31.09%/29.47%/22.17%**, median dispatch spans fall
**8.62%/28.87%/27.45%/21.00%**, and profiled child throughput rises
**8.56%/38.56%/37.52%/25.51%**. Dispatches stay at 1,010/token; exact wrap,
eviction, generated-ID, state, and lifecycle gates all pass. The full suite
therefore promotes **43.081 tok/s** h32 decode. [D4 retained
artifact](results/2026-07-23-gfx1100-laguna-q2-xl-swa-token4-retained.json).

A direct inventory/trace join split D4's remaining dense-Q5 family. Its **235
calls/token consumed 7.123 ms (37.52% of kernel sum)** across a **1.931 GB
encoded-weight proxy**, or 271.1 GB/s; this proxy excludes activations, cache
waste, and dequant compute. The same-input K3072/N1024 shared gate/up subset was
**92 calls and 1.561 ms/token** at only 127.4 GB/s, selecting the bounded D5
candidate. [D4 Q5 profile](results/2026-07-23-gfx1100-laguna-q2-xl-d4-q5-profile.json).

##### Laguna Q2 XL c=1 decode D5

D5 combines each of those 46 pairs into one four-axis `linear_pair` launch.
Independent grid-y workgroups call the exact singleton block body, preserving
each projection's K order, coefficient hoist, reduction tree, and BF16 store;
registry/shape misses still execute the two singleton primitives and rows>1 is
unchanged. Synthetic K3072/N1024 output and actual Q2 XL layer-1/layer-47
oracles are exact. Actual `blk.1` gate/up wall improves **28.148 -> 16.373 us
per pair (-41.83%)**.

Clean short tracing removes **46 launches/token (1,010 -> 964)**, moves the
paired subset **1.561 -> 0.890 ms/token (-42.99%)**, complete dense Q5 **7.123
-> 6.366 ms (-10.62%)**, kernel sum **18.983 -> 18.260 ms (-3.81%)**, and
median span **22.878 -> 21.981 ms (-3.92%)**. At 512/1K/near-4K, kernel sum
improves **2.13%/1.68%/1.18%**, span improves **2.25%/2.02%/1.55%**, and
profiled child throughput improves **1.78%/1.14%/1.50%**. Every trace preserves
generated IDs, finite output, exact lifecycle, and local128/VGPR72/LDS1024/
scratch0 resources. The complete category suite therefore promotes **44.501
tok/s** h32 decode and **11.860 output tok/s** h32 E2E. [D5 retained
artifact](results/2026-07-23-gfx1100-laguna-q2-xl-q5-shared-pair-retained.json).

The same clean trace ranks D5's remaining short families: dense-Q5 BF16/F32,
selected IQ2, SWA, and weighted IQ3 consume **2.744/2.732/2.300/2.120/2.093
ms/token**. All 2.732 ms of Q5 F32 work is 47 same-input attention query/gate
pairs: 35 K3072 N9216+72 SWA rows and 12 K3072 N6144+48 global rows. The gate
side alone costs **0.598 ms / 47 launches** while reading only 6.54 MB at a
10.93 GB/s encoded-weight proxy. The selected D6 candidate maps unequal query
and gate pack ranges into one exact flattened `linear_pair` dispatch, preserving
the two singleton launches as fallback. Perfectly hiding the gate side would
save at most **3.28% of kernel sum / 2.72% of span** and approximate **45.747
tok/s**, so this bounded candidate cannot reach 50 tok/s by itself. [D5
residual profile](results/2026-07-23-gfx1100-laguna-q2-xl-d5-residual-profile.json).

##### Laguna Q2 XL c=1 decode D6

D6 registers `linear_pair/gguf_q5_k/pack8_gemv_decode_bf16_f32_out` and
flattens the independent query and gate output-pack ranges into one grid. The
existing local128 singleton block body owns every pack, so both F32 output
buffers preserve exact K order and bytes. Rows>1, registry/shape misses, mixed
quants, F16 residency, Q6 layer 47, and unmeasured backends keep the unfused
QKV plus gate path. Synthetic production shapes and actual `blk.0/1` weights
are byte-exact; actual global/SWA pair medians improve **23.04%/25.83%**.

Clean profiling removes **47 launches/token (964 -> 917)**. Short/512/1K/
near-4K kernel sum improves **3.13%/1.91%/1.70%/1.21%**, median span improves
**3.21%/2.15%/1.95%/1.38%**, and profiled child throughput improves
**3.89%/2.87%/1.81%/1.48%**. The F32 pair is stable near **2.15 ms/token** and
runs at local128/VGPR48/SGPR128/LDS1024/scratch0. Every trace preserves D5 IDs,
finite output, and exact teardown; the full suite promotes **45.433 tok/s** h32
decode and **11.921 output tok/s** h32 E2E. [D6 retained
artifact](results/2026-07-23-gfx1100-laguna-q2-xl-q5-query-gate-pair-retained.json).

A code-identical reanalysis of those retained traces ranks short attention-output
Q5, selected IQ2, retained query/gate Q5, SWA, weighted IQ3, and attention K/V
Q6 at **2.628/2.285/2.151/2.118/2.110/1.407 ms/token**. The Q6 leaf is exactly
47 same-input K/V pairs at K3072/N1024: **94 launches/token**, **14.920 us**
median per singleton, local128/VGPR48/SGPR128/LDS1024/scratch0. The selected D7
candidate transfers the retained equal-width pair tactic through a separately
registered `linear_pair/gguf_q6_k` key while keeping both singleton fallbacks.
Even perfect one-side overlap saves at most **0.703 ms (3.98% kernel sum / 3.31%
span)** and approximates **46.933 tok/s**, so it cannot reach 50 tok/s alone.
This is attribution and candidate selection, not a new throughput claim. [D6
residual profile](results/2026-07-23-gfx1100-laguna-q2-xl-d6-residual-profile.json).

##### Laguna Q2 XL c=1 decode D7

D7 registers `linear_pair/gguf_q6_k/pack8_gemv_decode_bf16_f32_out` and uses
one flattened pack grid for 47 equal-width K/V pairs plus layer 47's unequal
query/gate pair. Every workgroup preserves the singleton K traversal,
eight-output reduction tree, and F32 store; rows>1, registry/shape misses,
mixed pairs, and unmeasured backends keep both singleton fallbacks. Synthetic
production outputs and all three actual-weight regimes are byte-exact. Global
K/V, SWA K/V, and layer-47 query/gate medians improve **37.36%/36.65%/8.80%**.

Clean tracing removes **48 launches/token (917 -> 869)**. Short/512/1K/
near-4K complete Q6 F32 work falls **35.91%/36.02%/37.30%/36.25%**, kernel sum
falls **2.38%/1.60%/1.46%/0.86%**, median span falls
**2.63%/2.00%/1.82%/1.17%**, and profiled child throughput rises
**3.33%/1.80%/2.31%/1.53%**. The pair runs at
local128/VGPR56/SGPR128/LDS512/scratch0. Every context preserves D6 IDs, finite
output, and exact teardown; the full suite promotes **46.409 tok/s** h32 decode
and **11.972 output tok/s** h32 E2E. [D7 retained
artifact](results/2026-07-23-gfx1100-laguna-q2-xl-q6-attention-pair-retained.json).

The canonical D7 wall is **21.548 ms/token**, still **1.548 ms** above the 20-ms
50 tok/s target. The clean short kernel sum is **17.268 ms** and median span is
**20.715 ms**, so the target remains physically supported rather than achieved.

A code-identical D7 residual analysis ranks short Q5 attention output, selected
IQ2 dual+SiLU, retained Q5 query/gate, weighted IQ3 down, and token4 SWA at
**2.646/2.317/2.171/2.134/2.119 ms/token**. The first family is 47 raw-Q5
BF16-output projections at N3072 and K6144/K9216, local128/VGPR72/LDS1024/
scratch0. Its current pack8 blocks reread a **304.35 MB/token** BF16 activation
proxy around **836.96 MB/token** of encoded weights. One exact 16-output block
would halve those activation reads and reduce the combined proxy **13.33%**;
a traffic-proportional model saves **0.353 ms/token** and reaches only **47.181
tok/s**, so this is a bounded D8 screen rather than a claim or a route to 50 by
itself. Tile16 must be BF16-bit exact and improve both actual global/SWA shapes;
tile32 is secondary and must also beat tile16 without spill. Near-4K remains
global-attention dominated at **22.658 ms/token**. [D7 residual
artifact](results/2026-07-23-gfx1100-laguna-q2-xl-d7-residual-profile.json).

##### Laguna Q2 XL c=1 decode D9

D9 registers
`moe_tail+next_rmsnorm/bf16/laguna_aggregate_gguf_f32_weight_out` for gfx1100
c=1. One local256 workgroup rounds routed+shared to BF16, rereads it for the
post-attention add and second BF16 boundary, stores hidden, then reproduces the
standalone RMSNorm's per-thread square order and stride-128..1 F32 reduction.
It emits both hidden and the next layer's normalized BF16 row; the final sparse
layer targets `output_norm`. Rows>1, gfx1151, registry misses, and explicit
rollback retain the exact registered add/add/RMS chain.

Clean short/512/1K/near-4K tracing removes **94 launches/token (869 -> 775)**.
Candidate versus explicit fallback improves kernel sum
**0.320%/0.515%/0.462%/0.117%**, median span
**2.667%/1.551%/1.431%/0.751%**, and profiled child throughput
**3.104%/2.387%/2.297%/1.015%**. Each token has 47 fused calls; the body costs
**0.389/0.385/0.386/0.385 ms** with **8.12-8.24 us** medians at
local256/VGPR16/SGPR128/LDS1024/scratch0. Every context preserves D7 IDs,
finite output, and exact teardown.

The complete category suite promotes h16/h32 decode
**46.827/46.409 -> 47.576/47.132 tok/s (+1.599%/+1.560%)** and h16/h32 E2E
**6.881/11.972 -> 6.909/12.038 output tok/s (+0.411%/+0.555%)**. Every
category improves both decode and E2E horizons; prefill and TTFT remain within
the 0.5% guard, and all correctness/lifecycle gates pass. Canonical D9 is
**21.217 ms/token**, still **1.217 ms** above 20 ms and requiring another
**6.084% throughput** to reach 50 tok/s. [D9 retained
artifact](results/2026-07-24-gfx1100-laguna-q2-xl-d9-moe-tail-next-rms-retained.json).

##### Laguna Q2 XL c=1 decode D10 (rejected and removed)

D10 tested a local256 token8 sibling of the retained exact token4 SWA reader.
It preserves complete `KVLiveSpans`, BF16 K/V, every score/softmax/value
arithmetic boundary, full logits and argmax bits, all 48 layer states, complete
K/V payload and live spans, reset, and lifecycle. The focused pre-measurement
bundle reported 69 passed.

Clean candidate versus token4 profiles improve SWA
**2.143/13.114/13.128/13.140 -> 1.876/11.171/11.193/11.187 ms/token** at
short/512/1K/near-4K. Kernel sum improves **1.05-6.38%**, span improves
**0.70-5.81%**, and profiled child throughput improves **0.91-5.80%** with 775
dispatches/token and exact IDs/teardown. The complete suite is not
non-regressive: aggregate h16 E2E is **-0.055%**, general-English h16
decode/E2E is **-0.535%/-0.254%**, and code/mixed h16 E2E is
**-0.128%/-0.017%**. Thus diagnostic h32 **47.872 tok/s / 20.889 ms/token** is
not retained. The token8 kernel, wrapper, registry entry, tests, and selector
are removed; post-removal validation reports 69 passed. At the D10 decision,
D9 stayed **47.132 tok/s / 21.217 ms/token** with a **1.217 ms / 6.084%**
throughput gap; D12 later superseded that headline. [D10 rejection
artifact](results/2026-07-24-gfx1100-laguna-q2-xl-d10-swa-token8-rejected.json).

##### Laguna Q2 XL c=1 decode D11 (rejected and removed)

D11 tested a separately registered gfx1100 c=1 persistent router/top-k
composite. It preserves full FP32 logits, unbiased/corrected scores, selected
IDs, normalized/scaled weights, all 47 actual layer routers, full model state,
K/V plus `KVLiveSpans`, reset, and lifecycle bit-for-bit. The self-resetting
counter returns to zero after every launch. Actual-weight events improve the
isolated 47-layer split window **0.820 -> 0.661 ms (-19.37%)**, and tracing
confirms **775 -> 728 dispatches/token** at local256/VGPR32/SGPR128/LDS512/
scratch0.

The predeclared clean mechanical screen nevertheless rejects the route.
Short/512/1K/near-4K isolated router work improves
**9.69%/11.05%/9.66%/9.87%**, span improves
**0.142%/1.012%/0.606%/0.488%**, and profiled-child throughput improves
**2.184%/0.092%/0.896%/0.749%**. Complete kernel sum changes
**+0.169%/-0.269%/-0.135%/-0.160%**. Two extra counterbalanced short pairs
confirm the failure: pooled 42-step kernel sum is **17.269472 -> 17.277499
ms/token (+0.046%)**. The category suite is skipped by policy, and the
composite source/export/wrapper/registry/selector/counter/tests are removed.
Split D9 remains **47.132 tok/s / 21.217 ms/token** for the D11 decision. [D11
rejection artifact](results/2026-07-24-gfx1100-laguna-q2-xl-d11-persistent-router-rejected.json).

##### Laguna Q2 XL c=1 decode D12

D12 defaults two separately registered gfx1100 c=1 role siblings:
`linear/gguf_q5_k/wave32x2_gemv_decode_bf16_bf16_out` for attention output and
`linear_pair/gguf_q5_k/wave32x2_gemv_decode_bf16_f32_out` for unequal query/
gate. One local32 wave owns two outputs while preserving every retained
logical thread's `[t,t+128]` K sequence, four independent 16..1 trees, and
logical-group 0..3 add order. It removes 1,024 B LDS and all block barriers;
pack8 remains the required explicit/unsupported fallback.

The formal 50-warmup/15x200 actual-weight screen is bit-exact and improves all
four required leaves by **13.63-24.80%** in HIP-event time and
**10.39-23.73%** in synchronized wall. Clean short/512/1K/near-4K profiles
improve the two Q5 families **15.06-17.91%**, complete kernel sum
**1.73-4.49%**, span **1.63-4.01%**, and profiled-child throughput
**1.44-5.25%** with unchanged **775 dispatches/token**. Candidate kernels are
local32/VGPR96/SGPR128/LDS0/scratch0.

The counterbalanced canonical gate moves h32 decode **47.046 -> 48.987 tok/s
(+4.124%)** and h32 E2E **11.997 -> 12.117 (+1.001%)**, with every category
positive and unaffected prefill/TTFT inside 0.5%. D12 is **20.414 ms/token**,
still **0.414 ms / 2.068% throughput** from 50 tok/s. [D12 retained
artifact](results/2026-07-24-gfx1100-laguna-q2-xl-d12-q5-wave32x2-retained.json).

##### Laguna Q2 XL P0 exact IQ3 ownership (retained gfx1100 default)

P0 screens both exact schedules from `docs/LAGUNA-decode.md` on actual
E256/K1024/N3072 weights with ten distinct routes. Producer+reducer events pick
wave4 over row4: layer 1 improves **50.101 -> 31.896 us (-36.34%)** versus
row4 **40.626 us (-18.91%)**; layer 45 improves **49.172 -> 33.000 us
(-32.89%)** versus row4 **41.819 us (-14.96%)**. The retained producer launches
one local32 wave per `(route, output)`, preserves four independent K256
shuffle trees plus their original add order, and writes each BF16 route before
the unchanged slot-order FMA reducer. It is VGPR88/SGPR128/LDS0/scratch0 and
adds no allocation or persistent weight copy.

Clean short/512/1K/near-4K profiles reduce the inclusive IQ3 family
**24.98-26.19%**, complete kernel sum **1.00-3.21%**, and dispatch span
**0.63-1.82%**, while profiled-child throughput improves **1.09-1.52%**. The
45 producer plus 47 reducer calls increase dispatches **775 -> 820/token**, but
the device-body saving dominates. The counterbalanced canonical gate moves h32
**48.780 -> 50.254 tok/s (+3.022%)** and E2E **12.103 -> 12.183 (+0.666%)**
with every category/horizon positive. gfx1100 therefore defaults
`wave4_reduce`; `serial_weighted` remains exact rollback and other backends stay
serial. The P0 row4 c=1 runtime mode is removed; its separately measured tile4
leaf remains for explicit DFlash verifier rows. [P0 retained
artifact](results/2026-07-24-gfx1100-laguna-q2-xl-p0-iq3-wave4-retained.json).

##### Laguna Q2 XL P2 exact split attention (retained gfx1100 default)

P2.1 replaces one-block context scans above independently measured crossovers
with local32 score producers plus local256 global/local128 SWA reducers. Every
producer consumes all five `KVLiveSpans` fields; global preserves block-256
paging and SWA preserves ring/wrap/eviction semantics. Synthetic boundaries
select global live count **127** and SWA **65**. Actual layer 0/44 and 1/47
context-128 event windows improve **8.53-13.31%**, all F32 outputs and complete
model state are bit-exact, and the four kernels use VGPR **8/19/7/11** with
zero private scratch. Two reusable session buffers add **1,572,864 bytes**.

Clean short/512/1K/near-4K profiles improve total attention
**15.66%/23.28%/22.98%/22.33%**, complete kernel sum
**2.67%/12.61%/13.44%/16.11%**, span
**4.65%/11.05%/11.60%/14.63%**, and profiled-child throughput
**1.19%/12.19%/12.01%/17.58%**. The two-order 18-prompt category/heldout gate
keeps all IDs/state exact and moves h32 decode **50.093 -> 51.436 tok/s
(+2.681%)** plus E2E **12.098 -> 12.158 (+0.496%)**, with every category
positive and prefill/TTFT inside 0.5%. gfx1100 defaults both thresholds;
`--disable-split-attention`, below-threshold calls, and other backends retain
the registered readers. A retained exact SWA tile16 refinement now takes over
at live `>=257`. Two process orders improve pooled 512/1K/near-4K SWA attention
**0.571%/0.344%/0.208%** and total attention
**0.461%/0.272%/0.056%**, with unchanged dispatches/memory and exact complete
state. The short canonical row remains **51.436 tok/s / 19.441 ms/token**
because it stays below that crossover and still needs **83.75%** more
diagnostic throughput to match Vulkan. Explicit tile16 disable retains P2.1;
other backends do not inherit it. [P2 retained artifact](results/2026-07-24-gfx1100-laguna-q2-xl-p2-split-exact-retained.json) and [tile16 retained artifact](results/2026-07-24-gfx1100-laguna-q2-xl-p2-swa-tile16-retained.json).

The same-model/same-context natural-greedy completion protocol supplies the
cross-engine timing boundary. It uses all **18** category and heldout prompts,
h16/h32, two repetitions, context 4096, and normalizes both engines to
synchronized post-TTFT transitions: hipEngine
`decode_forward_calls/decode_seconds` versus llama.cpp Vulkan
`sum(predicted_n - 1) / sum(predicted_ms)`. The pre-current-P4 audit measured
hipEngine **51.839/51.432 tok/s** and Vulkan **64.213/64.336 tok/s**.

The post-current-P4 reaudit explicitly pins `GGML_VK_VISIBLE_DEVICES=0` and
reproduces Vulkan at **64.245/64.418 tok/s**. Current hipEngine reaches
**52.855/52.391 tok/s**, so it remains **17.73%/18.67% slower** and needs
**21.55%/22.96%** more throughput; Vulkan-beating completion is not achieved.
Two unpinned 57.1-57.6-tok/s Vulkan diagnostics are excluded from the canonical
protocol. The unavoidable KV difference is disclosed (hipEngine BF16
`KVLiveSpans`, Vulkan F16 because this device reports no BF16 support). All 72
pinned Vulkan native prompt/predicted timing rows are valid; SSE `return_tokens`
omits one or more token entries for 18 rows, so returned-array completeness is
not used as the timing gate. The subsequently retained exact wave-local SWA
reducer moves the same hipEngine category boundary to **52.514 tok/s h32**;
against the unchanged pinned target it still needs **22.67%** more throughput.
[Initial audit](results/2026-07-24-gfx1100-laguna-q2-xl-vulkan-matched-completion-audit.json) and [device-pinned current-P4 reaudit](results/2026-07-24-gfx1100-laguna-q2-xl-vulkan-matched-completion-reaudit.json).

##### Laguna Q2 XL P4.1 exact split-reducer+gate (retained gfx1100 default)

P4.1 folds the existing FP32 softplus per-head gate and RNE BF16 store into the
retained exact global/SWA split reducers while retaining the F32 context. The
score producers, logical-slot reduction order, `KVLiveSpans` ABI, and scratch
allocation are unchanged; the registered unfused chain remains the
below-threshold, explicit-disable, registry-miss, and non-gfx1100 fallback.
First/last actual global/SWA layers at live 128/257 are bit-exact and improve
the inclusive event **3.00-10.05%** and synchronized wall **2.89-9.60%**. Full
logits, all 48 hidden and 47 routed boundaries, K/V, every span field, reset,
and lifecycle remain exact; tracing names the expected local256/local128 gated
reducers at VGPR24/LDS512/scratch0 with no standalone gate.

The two-order 18-prompt category/heldout gate moves h16/h32 decode
**51.882/51.497 -> 52.229/51.825 tok/s (+0.669%/+0.637%)**. Every train and
heldout category improves both decode horizons; E2E, prefill, and TTFT remain
inside the 0.5% non-regression guards. Relative to the prior retained 51.436
headline this is **+0.757%**, or **19.296 ms/token**. The formal matched Vulkan
h32 target remains **64.336 tok/s**, so hipEngine still needs **24.14%** more
throughput and completion is not claimed. [Correctness artifact](results/2026-07-24-gfx1100-laguna-q2-xl-p4-split-gate-correctness.json) and [retained artifact](results/2026-07-24-gfx1100-laguna-q2-xl-p4-split-gate-retained.json).

##### Laguna Q2 XL current-P4 exact head RMSNorm+RoPE+KV (retained gfx1100 default)

The current-P4 recomposition ports only historical D14's independently positive
exact c=1 body onto retained P2/P4.1. One local256 block owns each query/KV
head, preserves FP32 RMSNorm and partial-RoPE arithmetic, writes F32 Q/K plus
RNE BF16 K/V, and consumes all five `KVLiveSpans` fields. The registered
head-plus-writer chain remains the rows/prefill, explicit-disable, gfx1151, and
unsupported fallback. First/last actual global/SWA layers are bit exact and
improve inclusive event **33.05-39.36%** and wall **33.41-39.13%**. Full logits,
48 hidden/47 routed boundaries, active K/V and all span fields, reset, and
lifecycle are exact with no allocation or persistent-copy delta.

Clean profiling removes 48 launches/token (**820 -> 772**). The first short
pair had a +0.035% total-kernel noise row despite family **-32.11%**, span
**-2.22%**, and child **+3.12%**; a predeclared reverse confirmation pools 28
stable samples per arm and resolves kernel sum **15.503 -> 15.432 ms/token
(-0.462%)**, family body **-32.81%**, and span **-2.31%**. At 512/1K/near-4K,
kernel sum improves **0.873%/0.507%/0.141%**, span
**1.766%/1.089%/0.775%**, and child throughput
**1.912%/1.408%/1.187%**.

The complete two-order 18-prompt gate moves h16/h32 decode
**52.296/51.872 -> 52.855/52.391 tok/s (+1.068%/+1.001%)** and h32 E2E
**12.207 -> 12.232 (+0.204%)**. Every train/heldout category decode improves;
all E2E/prefill/TTFT guards, IDs, Poolside oracle, state, and lifecycle pass.
Relative to the prior retained 51.825 row, h32 improves **1.092%** to **19.087
ms/token**. The device-pinned matched Vulkan reaudit measures **64.418 tok/s**,
so hipEngine still requires **22.96%** more and completion remains open.
[Correctness artifact](results/2026-07-24-gfx1100-laguna-q2-xl-p4-head-kv-correctness.json), [retained artifact](results/2026-07-24-gfx1100-laguna-q2-xl-p4-head-kv-retained.json), and [matched reaudit](results/2026-07-24-gfx1100-laguna-q2-xl-vulkan-matched-completion-reaudit.json).

##### Laguna Q2 XL exact wave-local SWA split reducer (retained gfx1100 default)

The wave-local reducer is a distinct follow-up to the rejected max-scan-only
edit. Four logical wave leaders independently replay the retained scalar
maximum and denominator order, broadcast four weights at a time with width-32
shuffles, and preserve each dimension's slot-order FMA chain. This duplicates
scalar score/`expf` work but removes every block barrier and all reducer LDS;
score producers, `KVLiveSpans`, workspace, and 772 dispatches/token are
unchanged. Full logits, all 48 hidden and 47 routed boundaries, active K/V plus
every span byte, reset, and lifecycle are exact. Cached tracing records
local128/VGPR24/SGPR128/LDS0/scratch0.

Two clean process orders improve the reducer **4.63-5.22%**, complete SWA
**4.24-4.55%**, total kernel sum **0.94-1.98%**, and span **0.61-1.69%** at
short/512/1K/near-4K; child throughput stays within guard or improves. The
complete two-order 18-prompt gate moves h16/h32 decode
**52.675/52.211 -> 52.949/52.514 tok/s (+0.519%/+0.580%)**. Every
train/heldout category decode improves **0.239-0.706%**; aggregate h32 E2E is
**+0.033%**, prefill **-0.126%**, and TTFT **-0.027%**, with every scoped guard
passing. Relative to the prior retained h32 row this is **+0.235%**, or
**19.042 ms/token**. The pinned Vulkan target remains **64.418 tok/s**, so
another **22.67%** is required and completion stays open. Explicit
`use_swa_split_wave_local=False` / `--disable-swa-split-wave-local` retains the
shared-statistics reducer. [Correctness artifact](results/2026-07-25-gfx1100-laguna-q2-xl-p4-swa-wave-local-correctness.json) and [retained artifact](results/2026-07-25-gfx1100-laguna-q2-xl-p4-swa-wave-local-retained.json).

##### Laguna Q2 XL exact IQ2 expanded-magnitude grid (retained gfx1100 default)

The c=1 sibling replaces the retained 1-KiB packed selector-code table and its
per-use magnitude reconstruction with one canonical 64-bit magnitude entry per
selector. It adds 3 KiB of code-object constants, not model weights or a
persistent sidecar, and keeps parity `popc`, every FMA/reduction, and both BF16/
SiLU boundaries. The hot leaf contracts **1,246 -> 986** disassembly lines,
logical VGPR **132 -> 110**, uint-to-float conversions **66 -> 10**, and
multiplies **78 -> 14**, with no spill. First/last actual layers are bit exact
and improve events **30.78-33.73%** plus synchronized wall **30.00-33.43%**.
Full logits, all 48 hidden and 47 routed boundaries, active K/V plus every span
byte, reset, and lifecycle are exact. Cached tracing records local64/VGPR112/
SGPR128/LDS512/scratch0; rows>1 remain on compact-grid VGPR136.

Two clean process orders improve the complete 46-call IQ2 family
**20.31-21.54%**, kernel sum **1.30-3.70%**, dispatch span **1.20-3.09%**, and
profiled-child throughput **1.19-2.17%** at short/512/1K/near-4K, with exactly
772 dispatches/token. Both complete 18-prompt orders move h16/h32 decode
**53.068/52.650 -> 55.022/54.540 tok/s (+3.683%/+3.590%)**. Every train/
heldout category decode improves **3.426-3.794%** and every E2E row improves;
prefill is **-0.072%** and TTFT **+0.042%**. Relative to the prior retained
52.514 row, h32 improves **3.858%** to **18.335 ms/token**. Pinned Vulkan remains
**64.418 tok/s**, so another **18.11%** is required and completion stays open.
`use_iq2_grid64=False` / `--disable-iq2-grid64` retains the registered
compact-grid fallback; rows>1 and unsupported backends always use it. [Correctness artifact](results/2026-07-25-gfx1100-laguna-q2-xl-iq2-grid64-correctness.json) and [retained artifact](results/2026-07-25-gfx1100-laguna-q2-xl-iq2-grid64-retained.json).

##### Laguna Q2 XL exact Q5 fixed metadata (retained gfx1100 default)

The c=1 D12 siblings replace 32 lane-published Q5 scale/min coefficient
exchanges with two wave-uniform 128-bit metadata loads per superblock. Raw
176-byte Q5_K weights, eight accumulator chains, four reduction trees, group
addition order, and BF16/F32 stores are unchanged. Coefficient plus reduction
`ds_bpermute` contracts **72 -> 40** and logical VGPR **89 -> 72**, with no LDS,
scratch, sidecar, workspace, or dispatch-count change. First/last actual global
and Q5-SWA output/query-gate rows are bit exact and improve HIP events
**19.80-25.19%** plus synchronized wall **17.59-24.07%**. Full logits, all 48
hidden and 47 routed boundaries, active K/V plus every span byte, reset, and
lifecycle are exact.

Both clean process orders improve pooled Q5 **22.68-23.12%**, kernel sum
**2.35-6.34%**, dispatch span **2.17-5.58%**, and profiled-child throughput
**2.26-4.41%** at short/512/1K/near-4K, with exactly 772 dispatches/token and
local32/VGPR72/LDS0/scratch0 resources. Both complete 18-prompt orders move
h16/h32 decode **54.964/54.476 -> 58.243/57.711 tok/s (+5.964%/+5.938%)**.
Every train/heldout category decode improves **5.52-6.48%**, h16/h32 E2E
improves **0.582%/1.163%**, prefill is **-0.103%**, and TTFT **+0.022%**.
Relative to the prior retained 54.540 row, h32 improves **5.813%** to **17.328
ms/token**. Pinned Vulkan remains **64.418 tok/s**, so another **11.62%** is
required and completion stays open. Role-scoped `use_q5_fixed_meta_*=False` /
`--disable-q5-fixed-meta-*` retains the registered coefficient-publication
wave32x2 fallback; rows>1 and unsupported backends retain the existing exact
routes. [Correctness artifact](results/2026-07-25-gfx1100-laguna-q2-xl-q5-fixed-metadata-correctness.json), [retained artifact](results/2026-07-25-gfx1100-laguna-q2-xl-q5-fixed-metadata-retained.json), and [post-Q5 matched Vulkan audit](results/2026-07-25-gfx1100-laguna-q2-xl-vulkan-matched-completion-post-q5.json).

##### Laguna Q2 XL exact mixed attention projections (retained gfx1100 default)

The c=1 four-axis `attention_projection_quad` route flattens each layer's four
independent raw-GGUF projections into one dispatch without changing a weight
byte or accumulator order. Layers 0-46 run four fixed-metadata Q5 wave32 owners
plus generic Q6 local128 packs; corrected layer 47 runs generic Q6 query/gate
plus Q8 K/V packs. Actual layers 0/1/46/47 are F32-bit exact and improve the
inclusive HIP-event window **4.52-16.57%** plus synchronized wall
**3.65-14.23%**. Full logits, all 48 hidden and 47 routed boundaries, active
K/V and every `KVLiveSpans` field, reset, and lifecycle remain exact.

Cached tracing records 47 Q5/Q6 calls plus one Q6/Q8 call and contracts **772 ->
723 dispatches/token**. Q5/Q6 is local128/VGPR88/LDS512/scratch0; Q6/Q8 is
local128/VGPR56/LDS512/scratch0. Both clean process orders improve projection
work **2.02-3.35%**, kernel sum **0.09-0.35%**, span **0.69-1.56%**, and
profiled-child throughput **1.06-2.92%** at short/512/1K/near-4K. Both complete
18-prompt orders move h16/h32 decode **58.367/57.833 -> 58.992/58.425 tok/s
(+1.072%/+1.024%)**. Every train/heldout category decode improves
**0.744-1.471%**; h16/h32 E2E changes **-0.021%/+0.087%**, prefill is
**-0.158%**, and TTFT **+0.035%**, all inside the frozen guards. Relative to the
prior retained 57.711 row, h32 improves **1.237%** to **17.116 ms/token**.
Pinned Vulkan remains **64.418 tok/s**, so another **10.26%** is required and
completion stays open. `use_mixed_q5_q6_attention=False` /
`--disable-mixed-q5-q6-attention` retains the exact registered pair/singleton
fallback; rows>1, shape/registry misses, and unsupported backend defaults also
retain it. [Correctness artifact](results/2026-07-26-gfx1100-laguna-q2-xl-mixed-attention-correctness.json), [retained artifact](results/2026-07-26-gfx1100-laguna-q2-xl-mixed-attention-retained.json), and [post-mixed matched Vulkan audit](results/2026-07-26-gfx1100-laguna-q2-xl-vulkan-matched-completion-post-mixed.json).

##### Laguna Q2 XL fixed Q6 metadata inside mixed projections (retained gfx1100 default)

The retained sibling changes only the Q6-owned workgroups inside the mixed
attention-projection quad. Each local128 pack cooperatively publishes the exact
8x16 `d*scale` metadata while preserving every `[t,t+128]` accumulator,
reduction, Q5 owner, Q8 operation, output byte, and launch. Actual layers
0/1/46/47 improve complete projection event **9.61-41.52%** and synchronized
wall **8.50-38.85%**. Full logits, all 48 hidden/47 routed boundaries, active
K/V and every `KVLiveSpans` field, reset, and lifecycle remain exact.

Cached tracing remains **723 dispatches/token** and records 47 Q5/Q6 plus one
Q6/Q8 candidate call per transition. Q5/Q6 is local128/VGPR88/LDS1024/scratch0;
Q6/Q8 is local128/VGPR48/LDS1024/scratch0. Both clean process orders improve
projection work **8.08-10.10%**, kernel sum **0.73-1.26%**, span **0.57-1.49%**,
and profiled-child throughput **0.01-0.84%** at short/512/1K/near-4K. Both
complete 18-prompt orders move h16/h32 decode **59.038/58.466 -> 59.787/59.211
tok/s (+1.269%/+1.275%)**. Every train/heldout category decode improves
**1.047-1.647%**; category E2E stays within **-0.249% to +0.350%**, aggregate
prefill is **-0.177%**, and aggregate TTFT **-0.042%**. Relative to the prior
retained 58.425 row, h32 improves **1.346%** to **16.889 ms/token**. Pinned
Vulkan remains **64.418 tok/s**, so another **8.79%** is required and completion
stays open. `use_mixed_q6_fixed_meta_attention=False` /
`--disable-mixed-q6-fixed-meta-attention` restores the registered generic-Q6
mixed quad. [Correctness artifact](results/2026-07-26-gfx1100-laguna-q2-xl-mixed-q6-fixed-metadata-correctness.json), [retained artifact](results/2026-07-26-gfx1100-laguna-q2-xl-mixed-q6-fixed-metadata-retained.json), and [post-fixed-Q6 matched Vulkan audit](results/2026-07-26-gfx1100-laguna-q2-xl-vulkan-matched-completion-post-fixed-q6.json).

##### Laguna Q2 XL shared-Q5 fixed metadata (retained gfx1100 default)

The same exact local32 fixed-address-metadata owner now replaces the local128
pack8 pair for the 46 shared gate/up projections in sparse layers 1-46. Raw
weights, BF16 stores, each accumulator/reduction sequence, pair launch count,
standalone SiLU/shared-down chain, and layer-47 Q6 route are unchanged. First/
last actual pairs are byte-exact and improve event/wall **26.88-27.61%**; full
logits, all 48 hidden/47 routed boundaries, active K/V and every live-span
field, reset, and lifecycle remain exact.

Both clean process orders preserve **723 dispatches/token** and improve shared
pair work **45.99-47.13%**, kernel sum **1.32-3.02%**, span **1.43-2.62%**, and
profiled-child throughput **0.89-3.33%** across short/512/1K/near-4K. Both
complete 18-prompt orders move h16/h32 decode **60.083/59.500 -> 61.554/60.942
tok/s (+2.448%/+2.425%)**. Every train/heldout category improves
**2.091-3.007%**; category E2E stays within **-0.031% to +0.551%**, aggregate
prefill is **-0.163%**, and TTFT **+0.091%**. Relative to the prior retained
59.211 row, h32 improves **2.924%** to **16.409 ms/token**. Pinned Vulkan remains
**64.418 tok/s**, so another **5.70%** is required and completion stays open.
`use_q5_shared_fixed_meta=False` / `--disable-q5-shared-fixed-meta` restores the
registered local128 pack8 pair; rows>1, key/shape misses, layer 47, and
unsupported backends retain their existing routes. [Correctness artifact](results/2026-07-26-gfx1100-laguna-q2-xl-shared-q5-fixed-metadata-correctness.json), [retained artifact](results/2026-07-26-gfx1100-laguna-q2-xl-shared-q5-fixed-metadata-retained.json), and [post-shared-Q5 matched Vulkan audit](results/2026-07-26-gfx1100-laguna-q2-xl-vulkan-matched-completion-post-shared-q5.json).

##### Laguna Q2 XL all-local32 mixed Q5/Q6 projections (retained gfx1100 default)

Layers 0-46 now give every Q5 or Q6 output pair one independent local32 wave.
Q5 invokes the retained fixed-address-metadata helper unchanged; Q6 carries the
four original local128 partitions independently, preserves every `k/k+128` FMA,
wave tree, and 0..3 partition addition, and replaces coefficient LDS/barriers
with wave broadcasts. Total global/SWA grid threads and waves are unchanged,
while rounded LDS falls **1,024 -> 0 B** and allocated VGPR **88 -> 80**.
Production outputs, complete model state, and default-vs-local128 rollback are
bit exact. First/last actual projection event/wall improves **11.39-14.77% /
11.24-15.72%**.

Both clean process orders preserve **723 model kernels/token**, 47 candidate
calls plus one retained layer-47 Q6/Q8 call, and improve projection work
**7.00-8.12%**, kernel sum **0.49-2.12%**, span **0.45-2.77%**, and profiled-child
throughput **0.20-1.29%** across short/512/1K/near-4K. Both complete 18-prompt
orders move h16/h32 decode **61.503/60.900 -> 62.354/61.732 tok/s
(+1.383%/+1.367%)**. Every train/heldout category improves **0.981-1.650%**;
category E2E stays within **-0.155% to +0.274%**, aggregate prefill is
**-0.192%**, and TTFT is unchanged. Relative to the prior retained 60.942 row,
h32 improves **1.296%** to **16.199 ms/token**. Pinned Vulkan remains **64.418
tok/s**, so another **4.35%** is required and completion stays open.
`use_mixed_local32_fixed_meta_attention=False` /
`--disable-mixed-local32-fixed-meta-attention` restores the registered local128
fixed-Q6 mixed quad; layer 47, rows>1, registry misses, and unsupported backends
retain their existing routes. [Correctness artifact](results/2026-07-26-gfx1100-laguna-q2-xl-mixed-local32-projection-correctness.json), [retained artifact](results/2026-07-26-gfx1100-laguna-q2-xl-mixed-local32-projection-retained.json), and [post-local32 matched Vulkan audit](results/2026-07-26-gfx1100-laguna-q2-xl-vulkan-matched-completion-post-local32.json).

##### Laguna Q2 XL local32 Q4 LM head (retained gfx1100 default)

The c=1 BF16/F32 LM head now gives one local32 wave two adjacent vocabulary
rows while replaying the retained local128 body's four K partitions, FMA order,
wave trees, and 0..3 partition addition exactly. Total threads and waves remain
**1,605,632 / 50,176**, while local128/LDS1024/VGPR48 becomes
local32/LDS0/VGPR72. All **100,352** F32 logits and the complete default-versus-
local128 rollback trajectory are bit exact; bulk-prefill/verifier projections,
rows>1, gfx1151, and registry misses retain local128.

Both clean process orders improve the LM head **29.07-30.79%**, complete kernel
sum **0.34-1.10%**, and dispatch span **0.25-1.35%** at unchanged **723 model
kernels/token**; profiled-child throughput remains inside the frozen guard. Both
complete 18-prompt orders move paired h16/h32 decode **62.310/61.675 ->
62.638/61.992 tok/s (+0.526%/+0.512%)**. Every train/heldout category improves
**0.247-0.804%**; category E2E stays within **-0.389% to +0.087%**, aggregate
prefill is **-0.208%**, and TTFT **+0.028%**. Relative to the prior retained
all-local32 row, h32 improves **61.732 -> 61.992 tok/s (+0.420%)** to **16.131
ms/token**. Pinned Vulkan remains **64.418 tok/s**, so another **3.91%** is
required. `use_q4_lm_head_local32_fixed_meta=False` /
`--disable-q4-lm-head-local32-fixed-meta` is the explicit local128 rollback.
[Primitive artifact](results/2026-07-26-gfx1100-laguna-q2-xl-q4-lmhead-local32-fixed-metadata-correctness.json),
[runtime artifact](results/2026-07-26-gfx1100-laguna-q2-xl-q4-lmhead-local32-fixed-metadata-runtime-correctness.json),
and [retained artifact](results/2026-07-26-gfx1100-laguna-q2-xl-q4-lmhead-local32-fixed-metadata-retained.json).

##### Laguna Q2 XL IQ3 ten-wave fused weighted down (retained gfx1100 default)

One local320 workgroup now owns one output column and keeps all ten exact
route-parallel wave32 IQ3 projections live. Each route preserves the retained
K1024 dot, shuffle tree, partition addition, and BF16 boundary; thread 0 then
replays the registered F32-weight reducer from `+0.0` in slot order. Explicit
`wave4_reduce`, exact-key miss, rows/prefill, and unsupported backends retain the
registered producer-plus-reducer fallback.

All **45/45** actual layer outputs and the no-argument-default versus explicit
wave4 trajectory are byte exact through bulk prefill, all 48 hidden/47 routed
boundaries, 16 decode transitions, active KV and every `KVLiveSpans` field,
reset, shared ownership, and teardown. Cached tracing contracts **723 -> 678
model kernels/token** with 45 fused calls plus two unchanged reducers; the
candidate runs local320/VGPR88/LDS512/scratch0 and adds no allocation.

Both clean process orders pass at short/512/1K/near-4K: inclusive IQ3 improves
**9.71-11.90%**, kernel sum **0.398-1.082%**, dispatch span **0.813-1.998%**,
and profiled-child throughput stays within the guard (**-0.005% to +2.751%**).
The frozen counterbalanced 18-prompt gate moves h16/h32 decode **62.972/62.318
-> 63.951/63.270 tok/s (+1.554%/+1.528%)**. Every train/heldout category
improves at both horizons; aggregate h16/h32 E2E changes **-0.008%/+0.145%**,
prefill **-0.189%**, and TTFT **+0.105%**, all within guard. Relative to the
prior retained row, h32 improves **61.992 -> 63.270 tok/s (+2.063%)** to
**15.805 ms/token**. Pinned Vulkan remains **64.418 tok/s**, so another
**1.81%** is required. [Primitive artifact](results/2026-07-27-gfx1100-laguna-q2-xl-iq3-wave10-fused-correctness.json),
[runtime artifact](results/2026-07-27-gfx1100-laguna-q2-xl-iq3-wave10-fused-runtime-correctness.json),
and [retained artifact](results/2026-07-27-gfx1100-laguna-q2-xl-iq3-wave10-fused-retained.json).

A clean post-P4.1 short trace then measures **820 dispatches/token**, **15.676
ms** of kernels, **18.760 ms** median dispatch span, and a **3.213 ms**
span-minus-kernel window. A new native submission screen prefilled 820 dependent
HSA AQL packets, rang one doorbell, and waited only on the final signal. With
correct barrier and agent/system fence semantics—and packet construction
excluded in AQL's favor—it is still **0.560-0.758% slower** than the same HIP
kernel chain across five independent 51-repetition processes. Direct AQL is
therefore rejected before runtime integration; D8 graph capture and D16 host
packets remain rejected. The formal Vulkan wall is also **0.132 ms shorter than
the retained kernel sum alone**, so exact device-work/dispatch fusion remains
necessary regardless of submission. [AQL rejection artifact](results/2026-07-24-gfx1100-laguna-q2-xl-p4-aql-submission-rejected.json).

##### Laguna Q2 XL c=1 decode D13 (rejected and removed)

D13 tested an exact local256 Q5 shared gate/up pair+SiLU composite. Synthetic,
actual endpoint, and complete shared-weight model gates preserve every BF16
output, hidden/logit/argmax checkpoint, K/V and `KVLiveSpans` field, reset, and
lifecycle bit-for-bit. The endpoint micro improves the inclusive pair+SiLU
window **2.12-2.95%**, and cached tracing confirms local256/VGPR88/SGPR128/
LDS1536/scratch0.

Clean short/512/1K/near-4K profiles reject the route. The composite body is
**9.28-11.07% slower** than pair+separate SiLU, while removing 46 launch gaps
shortens the inclusive boundary **5.46-6.93%**. Complete kernel sum regresses
**0.594%/0.390%/0.707%/0.444%**; 1K span regresses **0.024%**, and short
profiled-child throughput regresses **1.280%**. The frozen every-context
mechanical gate therefore fails, the category suite is skipped, and the kernel,
wrapper/registry, runtime selector, and tests are removed. D12 remains
**48.987 tok/s / 20.414 ms/token**. [D13 rejection
artifact](results/2026-07-24-gfx1100-laguna-q2-xl-d13-q5-shared-silu-rejected.json).

##### Laguna Q2 XL c=1 decode D14 (rejected and removed)

D14 tested separately registered global/SWA c=1 head-RMSNorm+partial-RoPE+BF16
KV-write composites. Synthetic page/ring transitions and the complete
shared-weight gate preserve every query/key, K/V, `KVLiveSpans`, hidden/logit,
reset, and lifecycle bit. Actual global/SWA endpoints improve the inclusive
two-launch boundary **33.76-41.54%**, and cached tracing confirms
local256/VGPR16/dynamic-LDS1024/scratch0 at 56/80 blocks.

Clean short/512/1K/near-4K profiles improve the fused body
**32.13-40.21%**, inclusive boundary **57.30-62.45%**, complete kernel sum
**0.072-1.478%**, span **0.485-2.717%**, and profiled-child throughput
**0.517-3.639%**, with exact IDs/lifecycle and **775 -> 727 dispatches/token**.
The full suite still fails the frozen non-regression gate. Aggregate h32 decode
improves **48.964 -> 49.274 tok/s (+0.634%)**, and every category's decode is
positive, but h16/h32 E2E changes **-0.140%/-0.050%**, TTFT changes **+0.798%**,
code regresses both E2E horizons, and general-English/general-Japanese regress
h16 E2E. The kernel/wrappers/registry/runtime selector/tests are removed; the
head+writer chain is again the only route. Canonical D12 remains
**48.987 tok/s / 20.414 ms/token**. [D14 rejection
artifact](results/2026-07-24-gfx1100-laguna-q2-xl-d14-head-kv-rejected.json).

##### Laguna Q2 XL c=1 decode D15 (rejected and removed)

D15 tested an all-or-none bundle of D14's exact global/SWA head+KV leaves and
new exact global/SWA attention+softplus-gate leaves. Production page/ring
fixtures, actual layers 0/44/1/47, all 48 context/gated-context/hidden taps,
complete logits/KV/`KVLiveSpans`, reset, and lifecycle are bit-exact. Cached
resources stay within local256/VGPR16/dynamic-LDS1024 for head+KV,
local256/VGPR40 for global attention+gate, local128/VGPR24 for SWA
attention+gate, and scratch0 throughout.

Clean short/512/1K/near-4K profiles remove **96 launches/token (775 -> 679)**.
Head-boundary body improves **29.94-33.48%**, attention-boundary body
**0.34-2.17%**, complete kernel sum **0.32-0.78%**, span **1.18-2.98%**, and
profiled-child throughput **1.87-3.10%**. The counterbalanced full suite remains
exact and moves h32 decode **48.888 -> 49.613 tok/s (+1.484%)** and h32 E2E
**12.104 -> 12.138 (+0.280%)**; aggregate h16 E2E is also **+0.116%** and every
category's decode improves. The frozen gate still fails because code h16 E2E
changes **-0.088%** and aggregate TTFT changes **+0.554%**, outside the 0.5%
guard. All four leaves, wrappers/registrations, selector/branches, aliases, and
candidate tests are removed together; standalone D14 is not restored.
Canonical D12 remains **48.987 tok/s / 20.414 ms/token**. [D15 rejection
artifact](results/2026-07-24-gfx1100-laguna-q2-xl-d15-attention-boundaries-rejected.json).

##### Laguna Q2 XL c=1 decode D16 (selection-rejected)

D16 screened host-only function-pointer packets around three exact adjacent
pairs whose D12 trace gaps sum to **0.5326 ms/token**: Q5 output+add/RMSNorm,
router projection+selection, and Q5 shared pair+SiLU. Both actual attention
widths and two actual router/shared layers preserve every intermediate/output
bit over 50 warmups and 15x500 counterbalanced iterations. The packets do not
reduce device submission cost: HIP-event changes range **-0.198% to +0.161%**
and wall changes **-0.563% to +0.016%**. No source or runtime route is retained;
the apparent trace gaps are queue-submission spacing rather than removable
ctypes overhead. [D16 rejection
artifact](results/2026-07-24-gfx1100-laguna-q2-xl-d16-c-dispatch-rejected.json).

##### Laguna Q2 XL c=1 decode D17 (rejected and removed)

D17 tested one all-or-none bundle of D15's exact global/SWA head+KV and global
attention+gate leaves plus a D10-derived token8 SWA attention+gate leaf. The
production empty/wrap/eviction/adversarial fixture, actual layers 0/44/1/47,
all 48 context/gated-context/hidden taps, complete logits/KV/`KVLiveSpans`,
reset, and lifecycle are bit-exact. Cached resources stay within
local256/VGPR24/dynamic-LDS4136/scratch0 for the token8 leaf.

Clean short/512/1K/near-4K profiles remove **96 launches/token (775 -> 679)**,
improve head-boundary body **25.73-30.75%**, attention-boundary body
**5.64-12.49%**, complete kernel sum **1.16-6.97%**, span **3.91-7.50%**, and
profiled-child throughput **3.89-8.30%**. The counterbalanced full suite moves
h32 decode **48.971 -> 50.668 tok/s (+3.465%)** and h32 E2E
**12.122 -> 12.197 (+0.623%)**; every category improves both decode/E2E
horizons and prefill stays within guard. The frozen gate still fails because
aggregate TTFT changes **1.8705 -> 1.8854 s (+0.795%)**, beyond 0.5%. All D17
source/dispatch/tests are removed without restoring standalone D10/D14/D15.
The **50.668 tok/s** row is diagnostic only; canonical D12 remains
**48.987 tok/s**. [D17 rejection
artifact](results/2026-07-24-gfx1100-laguna-q2-xl-d17-attention-boundaries-rejected.json).

No Q2-to-Q4 speed ratio is claimed: the retained Q4_K_M controls use a
different tensor recipe on gfx1151.

#### Laguna Q2 XL B4 DFlash decode

**Status: current tile4 verifier path retained explicit-only; automatic routing remains off.**
The clean current-revision W7900 gate uses the canonical ten prompts and four
categories plus four heldouts, 128-row chunks, pinned B4 drafter, 4K capacity,
and two complete process orders per horizon. h32 uses two internally
counterbalanced repetitions/process; h128 uses one. Tile4 is exact against
tile1 for every generated ID, full state/oracle, active KV/`KVLiveSpans`,
reset, and lifecycle check.

| Horizon / route | Target AR decode | DFlash decode | DFlash E2E | DFlash / AR decode |
| --- | ---: | ---: | ---: | ---: |
| h32 tile1 control | 48.990 | 32.307 | 12.118 | 0.6595x |
| h32 tile4 | 48.930 | **33.834 (+4.725%)** | **12.289 (+1.413%)** | **0.6915x** |
| h128 tile1 control | 45.775 | 27.790 | 20.452 | 0.6071x |
| h128 tile4 | 45.837 | **29.050 (+4.536%)** | **21.130 (+3.316%)** | **0.6338x** |

Every category and heldout DFlash decode/E2E row improves at both horizons:
h32 decode improves **4.51-4.84%** and E2E **1.18-2.03%**; h128 decode
improves **4.46-4.60%** and E2E **3.01-3.74%**. The 45-call IQ3 family moves
**11.646 -> 7.726 ms/cycle (-33.66%)**, complete target-verifier kernel sum
**64.874 -> 60.968 ms (-6.02%)**, and target-verifier wall
**73.955 -> 70.220 ms (-5.05%)**. Tile2 was exact but slower and is removed;
tile1 stays the unsupported/ordinary fallback.

The tile4 win is retained only for explicit `iq3_selected_down_tile=4` /
`--iq3-selected-down-tile 4` use. It does **not** satisfy the automatic-route
policy (>1.10x true AR with no category/heldout regression versus AR), the Q2
target has no admitted public DFlash route, and gfx1151 is unmeasured. [Tile4
retention artifact](results/2026-07-24-gfx1100-laguna-q2-xl-dflash-iq3-tile4-retained.json).

Historical context: the clean `6ba1ddec95e224c1cc337c69ac2c4ea611ff0472` run predates D1-D12 and
uses the same ten prompts, categories/heldouts, 128-row chunks, two balanced
repetitions, and 32
visible outputs as the Q4 DFlash protocol. It pairs the pinned Q2 XL target with
`poolside/Laguna-S-2.1-DFlash@b0486d1` (BF16 safetensors SHA-256
`f24f08781c697c19952c02fb2e7e9bdf2071b79a711c2a44b836a74b9b62a1f4`)
and alternates true AR/DFlash in one resident process.

| Route | Prefill tok/s | Median TTFT | Decode tok/s | E2E output tok/s |
| --- | ---: | ---: | ---: | ---: |
| True AR | 40.140 | 2.019 s | 19.596 | 8.569 |
| B4 DFlash | 20.477 | 3.929 s | **29.452** | 6.070 |
| DFlash / AR | **0.5101x** | **1.9462x wall** | **1.5030x (+50.30%)** | **0.7084x (-29.16%)** |

The decode promotion gate passes: heldout is **1.3125x**, while
`code/general_en/general_ja/mixed_ja_en` are
**2.1039/1.1699/1.5722/1.1414x**. One low-density prompt,
`mixed_ja_en_review`, is individually `0.9755x`, but its complete category and
the heldout aggregate remain positive. Draft acceptance is **422/872 (48.39%)**
over 218 cycles, with **1.7581 target rows/output**. Proposal, target verify,
and post-verify residual total **2.676/18.003/0.370 s**.

All 20 AR/DFlash pairs are exact, finite, target/drafter-state aligned, and
repeat-deterministic; the Q2 Poolside KL/top-1 gate and lifecycle pass. Combined
target+drafter residency is **42,369,140,733 bytes** and peak tracked ownership
is **42,369,361,917 bytes (39.460 GiB)**; close returns to zero. A cached
single-cycle dispatch trace confirms Q5 embedding, Q5/Q6/Q8 and IQ target
families, B+1 rowtiles, BF16/F32 DFlash WMMA projections, norm/rope/Silu,
`dflash_accept_chain_i32_kernel`, and top-k. Trace SHA-256 is
`39279fe9bcee683af3751a81e6715b6881536503140dbb293a30129aafbd8d5f`.
[Compact artifact](results/2026-07-23-gfx1100-laguna-q2-xl-dflash-b4.json).

The result is retained as a historical **D0-relative decode** win, not a
current/default-route or fixed-32 E2E win. Current D9 AR reaches **47.132
tok/s**, above this old DFlash row's 29.452 tok/s, while the verifier should
also benefit from D2's K1024 workgroup change, D3's weighted-down fusion, D4's
token4 SWA, D5/D6's Q5 pairs, D7's Q6 attention pair, and D9's MoE-tail fusion.
A fresh matched
category run is
therefore required before any current DFlash ratio is claimed. Independently,
DFlash still seeds target captures serially and this capture reported h32 E2E
29.16% slower; a full long-horizon category/public-route gate remains required
before any routing threshold or default promotion.

Exact benchmark command:

```bash
HIP_VISIBLE_DEVICES=0 HIPENGINE_HIP_ARCH=gfx1100 GPU_MAX_HW_QUEUES=1 PYTHONPATH=. uv run python -u scripts/laguna_dflash_category_bench.py /models/gguf/Laguna-S-2.1-UD-Q2_K_XL.gguf /models/hipengine/Laguna-S-2.1-DFlash --prompts benchmarks/prompts/mtpbench-code-general-ja.jsonl --template tests/fixtures/laguna_poolside_v1_template.json --oracle tests/fixtures/laguna_poolside_q2_xl_v1_oracle.json --oracle-logprobs tests/fixtures/laguna_poolside_q2_xl_v1_first_token_logprobs.npy --backend hip_gfx1100 --context-length 4096 --chunk-size 128 --candidate-budget 4 --output-tokens 32 --repetitions 2 --warmup-output-tokens 6 --compiler-version-file /tmp/hipengine-hipcc-version-laguna-iq2.txt --require-cached-build --direct-gguf --safety-reserve-gib 4 --model-sha256 8fe1170f012723f6f7d6c9b08d8f928b0b3d8bffc32926f33a930148a1d62679 --quant-label UD-Q2_K_XL --drafter-sha256 f24f08781c697c19952c02fb2e7e9bdf2071b79a711c2a44b836a74b9b62a1f4 --drafter-revision b0486d1586daa0d56435c508108171fc1c8daff9 --iq3-selected-down-tile 4 --output /tmp/laguna-q2-xl-dflash-category.json
```

### gfx1151 Laguna S 2.1 target AR, DFlash, and cold startup, 2026-07-23

**Status: retained for exact target-only c=1 AR and loader startup; LPF-1's
exact tile, LPF-4's 128-row chunks, LPF-5's wave32-exact SWA fallback, AR-O5's
context-qualified qrow2 SWA reader, and the exact grouped routing/shared combine
are default. The current merged-main
matched B4 DFlash row is exact and reaches
**0.9477x** true-AR decode, but fails aggregate, heldout, and non-code economics,
so DFlash remains off by default and performance-ineligible. Its exact B4 path
is supported only as an explicit library/OpenAI opt-in.** The AR protocol uses the
full ten-prompt `mtpbench-code-general-ja` suite (`code`, `general_en`,
`general_ja`, and
`mixed_ja_en`), prompt lengths 68-122, greedy 16/32-token horizons, two
repetitions, balanced serial/bulk order, and one warmup per route. Model load is
excluded. Every serial/bulk pair and same-route repeat has exact generated IDs;
the frozen Poolside first-token distribution passes at KL `6.6214e-6` and exact
top-1, and lifecycle recovery is exact.

| hipEngine route | Prefill tok/s | Median TTFT | Decode tok/s, h32 | E2E tok/s, h16 | E2E tok/s, h32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Token-serial prefill + eager c=1 decode | 17.425 | 4.625 s | 16.384 | 2.727 | 4.671 |
| Previous 64-row bulk GEMV prefill + same eager c=1 decode | 23.333 | 3.481 s | 16.381 | 3.470 | 5.719 |
| LPF-1 exact tiled prefill, 64-row chunks + same eager c=1 decode | 48.560 | 1.692 s | 16.386 | 5.955 | 8.717 |
| LPF-1 change vs previous default | **+108.12% (2.081x)** | **-51.39%** | +0.030% | **+71.61%** | **+52.42%** |
| Pre-LPF-5 LPF-1 tile + LPF-4 128-row chunks + same eager c=1 decode | 49.641 | 1.639 s | 16.390 | 6.042 | 8.811 |
| LPF-4 paired change vs same-session 64-row control | **+2.27%** | **-3.15%** | -0.010% | **+1.49%** | **+1.08%** |

The first serving-latency lane is now retained independently of model-kernel
prefill. Clean revision `8ae07d693b6f98d6c44aae90090df6c6d77e8d78`
alternates five synchronized fresh/reset first-token samples after one warmup on
the exact Q4_K_M model, BF16 KV, 4K capacity, chunk 128, and frozen 46-token
Poolside prompt. Generator-owned pooling moves median session preparation
**32.399 -> 0.598 ms (-31.800 ms; 54.14x)** and direct preparation-plus-prefill
TTFT **963.262 -> 928.384 ms (-34.877 ms; -3.621%)**. It also removes a
**5.373 ms median** fresh-session close from each request tail. Every paired
setup improves, first token ID `5887` is exact, median TTFT is non-regressive,
and tracked lifecycle returns to zero. This is a direct resident first-token
scope, not a replacement for the canonical ten-prompt AR table or real HTTP
TTFT. `prepare()` constructs the pool during readiness; subsequent requests
reset it under the c=1 lock and expose `session_prepare_ms`/mode telemetry.
[Retained S1 session-pool artifact](results/2026-07-23-gfx1151-laguna-session-pool.json).

Serving lane S2 is also retained independently of GPU/model wall. Clean
revision `0081d150c08a95423f29fec8fd26779f53c8f730` changes each blocking and
streaming Poolside request from six full rendered-prompt encoder calls to one
request-local exact-ID owner. In the FastAPI TestClient/immediate-model scope,
4,096-token useful-content TTFT changes **15.334 -> 7.167 ms (-8.166 ms;
-53.26%)** blocking and **15.483 -> 6.786 ms (-8.697 ms; -56.17%)** streaming.
Across 500 requests/mode on the canonical ten-prompt suite, pooled median prompt
encoder wall changes **0.357 -> 0.065 ms (-81.93%)** blocking and **0.340 ->
0.063 ms (-81.39%)** streaming; useful-content TTFT changes **1.265 -> 0.998 ms
(-21.12%)** and **1.244 -> 0.964 ms (-22.49%)**. Exact prompt usage matches and
every candidate sample encodes the rendered prompt once. This CPU-only isolated
scope uses the Q4_K_M GGUF for tokenizer metadata and an immediate deterministic
fake response; it is not Laguna model TTFT. [Retained S2 prepared-prompt
artifact](results/2026-07-23-laguna-prepared-prompt-fastapi.json).

Serving lane S3 removes blanket longest-stop holdback while preserving exact
suppression. Clean revision `71f2af038cf5eea88f1997d178d815cfaad15681`
uses the production Laguna stream with deterministic 61-ms fake resident-token
arrivals. A nonmatching first token moves useful-content TTFT **184.738 -> 0.203
ms (-184.536 ms; -99.89%)**; a one-token prefix disproved by the second token
moves **184.695 -> 61.452 ms (-123.243 ms; -66.73%)**. Exact four-token stops
remain fully suppressed, all text/IDs/finish details match, and complete E2E is
neutral. This host-only delayed-arrival scope is anchored to the retained 16.384
tok/s rate and proves emission timing, not model throughput. [Retained S3
prefix-aware stop artifact](results/2026-07-23-laguna-prefix-aware-stop-streaming.json).

Serving lane S4 retains one auth/session-scoped, 900-second, exact-prefix Laguna
KV slot. Clean measured revision `804e9484f3da0031628805f5bbef62a43badffaa`
moves nine-token continuation wall from **2,607.195/10,347.413/21,619.917 ms**
with reset/full prefill to **260.699/303.268/314.255 ms** with exact 128/512/1K
prefix reuse: **10.00x/34.12x/68.80x**, saving **2.347/10.044/21.306 s**. A
canonical rendered Poolside chat follow-up moves **1,670.692 -> 411.907 ms
(4.06x)**. Every next ID matches; each first measured shape and chat copy hashes
all **277,434,816 bytes** of global/SWA K/V plus span metadata identically with
exact positions and no pending rows; lifecycle returns to zero. Model load is
excluded. Reuse is explicit-session only and global radix stays off. [Retained
S4 stateful-KV artifact](results/2026-07-23-gfx1151-laguna-stateful-kv.json).

LPF-1 changes prompt execution only; rows=1 decode stays on the original exact
GEMV and is neutral. Every serial/tiled pair and same-route repeat is exact at
both horizons, the Poolside distribution remains KL `6.6214e-6` with exact
top-1, every category independently passes the predeclared
prefill/decode/e2e guard, and tracked ownership returns exactly to baseline.
The measured artifact explicitly records forced `tiled` selection; the
post-gate gfx1151 capability makes that same registered path automatic from two
rows while unsupported backends retain GEMV.

The same-session LPF-1 A/B holds one resident model/runtime and alternates GEMV
and tiled execution at rows 2/3/4/5/7/8/15/16/17/32/55/64/65/122/128. Every
mode/repetition returns the same next token. Tiled is faster at every measured
shape, from **20.568 -> 21.327 tok/s (1.0369x)** at two rows through
**23.460 -> 48.760 (2.0784x)** at 55 and **23.374 -> 50.240 (2.1494x)** at
128; the weighted measured profile improves **2.0538x**. The retained threshold
is therefore two rows, with rows=1 permanently on GEMV.

LPF-0 explains the gain. One physical chunk at rows 16/32/55/64/122/128 had
reached only 23.141/23.421/23.450/23.453/23.368/23.377 tok/s. Its cached
55-row trace assigns **68.99%** of kernel sum to source-F16 QKV/O, **26.45%**
to selected Q4/Q6 direct GEMV, and **0.96%** to attention. LPF-1 replaces that
first family with a reduction-order-preserving 8x4/16x4 tile; the cached
55x9216x3072 O launch is **3.798 ms**, 256 threads, 96 VGPR, 128 SGPR, 512 B
LDS, and zero scratch. A faster reassociated WMMA control was rejected after
changing three free-running trajectories.

Real top-10 routing produces 6,892 nonempty `(layer, expert)` groups at 55 rows;
76.25% have at most four lanes. LPF-2's exact no-padding compact-pair candidate
therefore tested the strongest incremental reuse bound, but regressed weighted
prefill **-11.57%** and was removed. LPF-3 exact dense/shared gate+up pairing
regressed **-0.71%** and was also removed; its real-WMMA follow-up requires a
non-incremental resident-layout/cache change for a family with only a roughly
6% post-LPF-1 Amdahl ceiling.

LPF-4 promotes 128-row chunks. A clean same-session run alternates 64/128 over
two repetitions of every canonical prompt, all of which cross 64 and fit 128.
The candidate moves paired prefill **48.541 -> 49.641 tok/s (+2.27%)**, median
TTFT **1.692 -> 1.639 s (-3.15%)**, and h16/h32 E2E **5.954/8.717 ->
6.042/8.811 (+1.49%/+1.08%)**, with decode neutral within 0.014%. Every category
improves: prefill is **+1.09% to +2.84%**, and fixed-horizon E2E is **+0.48% to
+1.79%**. All chunk pairs/repeats are exact, the Poolside gate remains KL
`6.6214e-6` with exact top-1, lifecycle returns to zero, and bounded resident
ownership rises only **49.1 MiB**. Artifacts: [LPF-0 profile and routing](results/2026-07-23-gfx1151-laguna-prefill-lpf0-profile.json),
[LPF-1 same-session A/B](results/2026-07-23-gfx1151-laguna-prefill-lpf1-ab.json),
[LPF-1 canonical category gate](results/2026-07-23-gfx1151-laguna-prefill-lpf1-tiled.json),
and [LPF-4 chunk-policy gate](results/2026-07-23-gfx1151-laguna-prefill-lpf4-chunk128.json).

LPF-5's clean one-pass attribution baseline uses the retained 128-row chunks at
512/1K/4K. Prefill is **43.732/39.697/33.745 tok/s** while attention grows from
**1.896 s / 16.25%** through **6.115 s / 23.78%** to **42.609 s / 35.19%** of
kernel sum. At 4K, global/SWA own **16.908/25.701 s (13.96%/21.23%)**. Exact
final cursors/IDs, the existing 511/512/513 CPU/GPU fixtures, trace segmentation,
and lifecycle pass. This is diagnostic attribution—not a speedup, repeated
throughput row, or long-context support claim—and ranks the bounded but serial
SWA reader before global attention. [LPF-5 profile](results/2026-07-23-gfx1151-laguna-prefill-lpf5-long-context-profile.json).

The promoted LPF-5 wave32-exact reader reconstructs the baseline 128-thread
reduction tree without per-token block barriers. The clean shared-weight gate
moves 512/1K/4K prefill **43.760/39.748/33.800 -> 47.395/44.855/38.552 tok/s
(+8.31%/+12.85%/+14.06%)**, saving **0.898/2.933/14.939 s**. Complete FP32
logits, final/pre-final BF16 hidden, next-logit bits, IDs, and cursors match at
every length; lifecycle is exact. A prior complete timing pass independently
reproduced **1.082/1.128/1.140x** before a post-timing harness failure. gfx1151
therefore initially made wave32 exact the gfx1151 default; AR-O5 now retains it
as the automatic short/partial fallback and explicit rollback beneath qrow2,
while unmeasured backends remain unchanged. [Retained LPF-5 SWA gate](results/2026-07-23-gfx1151-laguna-prefill-lpf5-swa-wave32.json).

AR-O5's exact qrow2 follow-up shares each BF16 K/V load across two adjacent
query rows while preserving both rows' logical scan, reduction, softmax, and
value-accumulation order. A measured two-axis policy uses it only for complete
M128 attention slices beginning at absolute position 128 or later; empty,
short, partial, and verifier rows automatically stay on wave32. The clean final
three-repeat gate moves 512/1K/4K prefill **69.031/63.969/52.017 ->
69.647/64.745/52.557 tok/s (+0.893%/+1.212%/+1.040%)**, saving
**0.066/0.192/0.811 s**, with complete logits/hidden/KV/span/cursor/repeat and
lifecycle equality. The full category gate is exact across all 30
free-running pairs and 320 teacher-forced steps and non-regressive at
**0.999652x prefill** and **0.999917/0.999999x h16/h32 E2E**. gfx1151 first
defaulted to this context-qualified selector; after the online promotion below
it remains the primary exact rollback, with wave32 and unmeasured-backend
fallbacks unchanged. [Retained AR-O5 qrow2 gate](results/2026-07-23-gfx1151-laguna-swa-qrow2-retained.json).

The post-qrow2 cached trace measures **69.467/64.676/52.549 tok/s** and kernel
sum **7.356/15.800/77.821 s** at 512/1K/4K. Versus the prior post-matrix trace,
SWA duration falls **9.38%/9.00%/8.99%**, complete attention falls
**7.19%/6.18%/3.43%**, and kernel sum falls **0.95%/1.24%/1.20%**. Global is
flat within 0.27% and now dominates 4K attention at **16.823 s / 21.62%**;
perfect removal is a **1.276x** kernel-sum ceiling. The vendored AOTriton
runtime passes the GPU check and a head-dim-256 GQA control, but native V3 and
per-query-head V2 return `hipErrorInvalidValue` for Laguna head-dim-128 at
M128, and V3 also rejects a global-only M512 query tile. Direct AOTriton
adaptation is closed; the remaining global lane requires an in-tree
`KVLiveSpans`-aware head-dim-128 tiled causal kernel. [Post-qrow2 profile and
AOTriton screen](results/2026-07-23-gfx1151-laguna-post-qrow2-global-screen.json).

The first in-tree exact global query-row reuse candidate is rejected before
full-model timing. One 256-thread workgroup handled two adjacent query rows and
halved grid Y while preserving causal visibility, BF16 boundaries, reduction and
three-pass softmax order, and complete `KVLiveSpans`. All six M128/context4096
screens are byte-exact, but prior-context 0/128/384/896/1920/3968 regresses
**111.81%/87.70%/76.99%/79.35%/78.66%/78.15%**; the 4K leaf moves
**86.778 -> 154.590 ms** while VGPR rises **40 -> 48**. The doubled whole-context
score/shared state defeats the halved workgroup count. All candidate code is
removed; key tiling without duplicated full score rows remains untested.
[Rejected global qrow2](results/2026-07-23-gfx1151-laguna-global-qrow2-rejected.json).

The replacement online-softmax route is retained on gfx1151. One wave streams
BF16 K/V across two adjacent global query rows and carries online max,
denominator, and output state without whole-context score LDS. The production
M128/4K leaf moves **86.752 -> 14.807 ms (5.859x)**, local32/VGPR48 with zero
LDS/scratch. The clean repeated full-model gate improves 512/1K/4K
**69.751/64.756/52.584 -> 71.475/68.281/64.076 tok/s
(+2.472%/+5.444%/+21.854%)** at maximum KL **0.007589** and top-1 **9/9**.
The complete ten-prompt gate then improves weighted prefill **69.310 -> 69.529
tok/s (+0.315%)**, TTFT **+0.331%**, and h16/h32 E2E **+0.184%/+0.125%**;
every category is positive and decode is neutral. All 320 teacher-forced logits
are finite at maximum KL **0.030836**, top-1 is **317/320 (99.0625%)**, each
category is at least 96.875%, and the frozen Poolside oracle, deterministic
repeats, and lifecycle pass. gfx1151 now selects online global prefill by backend
capability; exact global prefill remains explicit rollback and the default on
unmeasured backends. [Retained global online gate](results/2026-07-23-gfx1151-laguna-global-qrow2-online-retained.json).

The post-promotion cached profile confirms that the retained route changes the
bottleneck rather than submission overhead. Relative to the post-SWA-qrow2
trace, global duration falls **79.49%/81.62%/82.53%** and kernel sum falls
**2.80%/5.33%/18.03%** at 512/1K/4K; synchronized prefill is
**71.456/68.307/64.071 tok/s**. At 4K, global is now only **2.939 s / 4.61%**
of kernel sum, SWA is **9.705 s / 15.21%**, and total attention is **19.82%**;
span-minus-sum remains only **0.144-0.208%**. Perfect global removal is just a
**1.048x** ceiling, so another high-register query-head-sharing route is
deferred. SWA's **1.179x** perfect-removal ceiling admits one bounded online
qrow2 screen before AR-O5 closes. This is attribution, not a new performance
headline. [Post-global-online profile](results/2026-07-23-gfx1151-laguna-post-global-online-all-family-profile.json).

The final SWA follow-up is retained on gfx1151. Its wave32 online-softmax qrow2
kernel replaces exact qrow2's two ring scans with one while preserving complete
`KVLiveSpans`, physical ring mapping, positions, evictions, and BF16 K/V
boundaries. Production M128/full-window improves **7.893 -> 2.552 ms (3.093x)**
and start508 wrap improves **8.676 -> 2.987 ms (2.904x)**; cached tracing names
the intended kernel at **2.559 ms**, local32/VGPR56/LDS0/scratch0. The repeated
full-model gate improves 512/1K/4K **71.354/68.156/63.995 ->
76.226/74.538/70.885 tok/s (+6.828%/+9.364%/+10.766%)** at maximum KL
**0.016558** and top-1 **9/9**.

The complete ten-prompt gate then improves weighted prefill **69.011 -> 69.761
tok/s (+1.086%)**, TTFT **+0.083%**, and h16/h32 E2E **+0.616%/+0.420%**;
every category is positive and decode is neutral. All 320 teacher-forced logits
are finite at maximum KL **0.042924**, top-1 is **316/320 (98.75%)**, every
category is at least 95.3125%, and the frozen Poolside oracle, deterministic
repeats, and lifecycle pass. gfx1151 now selects online SWA prefill by backend
capability; exact context-qualified qrow2 and wave32 remain explicit rollback,
and unmeasured backends retain their prior defaults.
[Retained SWA online gate](results/2026-07-23-gfx1151-laguna-swa-qrow2-online-retained.json).

The arithmetic-prefill campaign supersedes that 76.226 tok/s production row.
The promoted gfx1151 defaults combine same-byte D8 producer packing with a
128-column resident-T16 gate/up integer-dot tile, D4 Q4/Q6 down tiles,
row-scaled hipBLASLt source-F16 projections, and resident Q4/raw-Q6 64x16 WMMA
dense/shared consumers. A clean selector-unset three-repeat screen measures
pp512 **353.421/355.584/354.820 tok/s** (median **354.820**), 1K
**321.270/322.922/323.210**, and 4K **263.436/265.000/264.245 tok/s**.
The pp512 change is **76.226 -> 354.820 (+365.484%; 4.655x)** and exceeds the
qualified Vulkan row by **2.978%**.

The complete ten-prompt quality lane passes at max KL **0.040724836** and
**317/320 (99.0625%)** teacher-forced top-1, with every category above the 90%
floor, **2.615x** aggregate natural-prompt prefill, neutral h16/h32 decode,
deterministic repeats, the Poolside oracle, and exact allocation recovery. A
fail-closed publication step binds that quality revision and the package
capabilities to the clean selector-unset timing revision. It explicitly does
not reinterpret the old matrix-policy byte-equality screen: that screen remains
rejected across M128/M256/M512 because the admitted arithmetic is approximate;
same-policy repeated state is deterministic. Cached-only final tracing
independently measures **354.763 tok/s**, attributes **40.8%/19.4%/9.1%/4.9%**
of pp512 kernel sum to selected gate/up, selected down, source-F16, and
dense/shared quant respectively, and names all intended families with zero
scratch on the D8/D4 MMQ kernels.
[Production publication](results/2026-07-25-gfx1151-laguna-prefill-350-production.json) ·
[complete quality gate](results/2026-07-25-gfx1151-laguna-prefill-350-d8-category.json) ·
[production trace](results/2026-07-25-gfx1151-laguna-prefill-350-production-trace.json).

A read-only same-device/same-GGUF llama.cpp Vulkan pp512 profile now gives a
concrete external control. The user's unprofiled `c0bc8591e` build measures
**344.56 +/- 3.16 tok/s**; the instrumented Vulkan operation sum is **1.478897
s** (**346.20 tok/s implied**) and reproduces that wall within **0.48%**, while
perf-logger overhead lowers its own reported benchmark row to 316.13 tok/s.
Against the profiled pre-SWA-online hipEngine default at **71.456 tok/s /
7.150503 s kernel sum**, homologous selected gate/up, selected down, source-F16,
attention, and dense/shared families are
**5.687x/3.001x/3.192x/18.933x/10.179x** slower and explain **99.76%** of the
kernel-sum gap. The newly promoted SWA-online screen reaches **76.226 tok/s** but
remains **4.520x** behind the user's row. Source attribution shows expert-major
32x32 Q4/Q6 x Q8_1 integer-dot MMQ, M16xK64 cooperative-matrix Flash Attention,
and graph-pattern fusion. The transfer decision is not to copy unchecked Q8
numerics: hipEngine's prior Q8 gate/up route failed max KL at **0.171561**. With
the SWA quality decision complete, prioritize a new quality-safe expert-major
matrix path; submission-only work remains deferred at **0.144%** pp512 span
residual.
[llama.cpp Vulkan pp512 profile](results/2026-07-23-gfx1151-laguna-llamacpp-vulkan-pp512-profile.json).

The first Vulkan-transfer expert-major Q4/Q6 WMMA route proves that decoded
weight reuse survives gfx1151 scheduling but rejects all-layer promotion on
quality. Its clean M32/55/64/122/128/256/512 screen improves **1.197/1.345/
1.423/1.717/1.745/2.087/2.304x** and reaches **176.001 tok/s at M512**. A
shape-qualified M128+ route then moves uniformly-M128 category prefill **73.046
-> 130.557 tok/s (1.787x)**, TTFT **1.792x**, and h16/h32 E2E
**1.400x/1.261x**, with every category positive and neutral decode. The complete
320-step quality lane nevertheless reaches maximum KL **0.527791** (>0.05),
although top-1 is **314/320 (98.125%)**, each category remains above 96.875%,
and Poolside fallback, deterministic repeats, and lifecycle pass. Exact
grouped-small-M remains default; the registered leaf is retained only for a
full-suite component/layer-scope bisection.
[Rejected expert-major category gate](results/2026-07-24-gfx1151-laguna-expert-major-wmma-category-rejected.json).

The follow-up component bisection rejects both halves. Q4 gate/up-only reaches
**113.530 tok/s (1.545x)** but max KL **0.988050** at **312/320** top-1;
Q4/Q6 down-only reaches **80.418 tok/s (1.095x)** but max KL **1.183662** at
**311/320**. The combined route remains fastest and has lower KL than either
half, indicating partial numerical cancellation rather than one isolated bad
projection. Poolside fallback and lifecycle pass for every mode. No component
is retained; only an architecture-derived global-versus-SWA layer-family screen
remains before removing the temporary routes.
[Rejected expert-major component bisection](results/2026-07-24-gfx1151-laguna-expert-major-wmma-component-rejected.json).

The final architecture-derived layer-family screen also rejects both scopes.
Global-only (12 layers) reaches **82.020 tok/s (1.115x)** but max KL
**0.628301** at **310/320** top-1; SWA-only (36 layers) reaches **110.711 tok/s
(1.505x)** but max KL **1.205779** at **312/320**. Both are numerically worse
than all-layer KL **0.527791**. No arbitrary layer subset will be tuned from
prompt outcomes. Exact grouped-small-M remains default, all temporary runtime
and benchmark selectors are removed, and only the independently tested kernel
leaf/oracle/trace remain as diagnostic evidence.
[Rejected expert-major layer-family bisection](results/2026-07-24-gfx1151-laguna-expert-major-wmma-layer-family-rejected.json).

The successor LAP-0 control packet closes the stale-attribution gap without
changing defaults. A clean cached matrix512/attention128 trace measures
**73.757/76.381/74.766/71.025 tok/s** at 128/512/1K/4K. At M512, selected Q4
gate/up, selected Q4/Q6 down, source-F16, dense/shared quant, and attention own
**54.99%/16.45%/13.37%/9.59%/4.16%** of kernel sum; named non-`other` coverage
is **99.653%** and span-minus-sum is **0.151%**. Replacing those measured
families cumulatively with the unchanged Vulkan family times models
**139.5/174.3/217.1/293.4/339.9 tok/s** and explains **99.740%** of the
kernel-sum gap.

The complete all-exact versus shipping-control lane measures **53.596 -> 70.546
tok/s (1.31627x)** with max KL **0.0459275**, **319/320** top-1, neutral
decode, deterministic repeats, Poolside, and lifecycle pass. That leaves only
**0.0040725** KL headroom. Natural M512 routing has tile4/8/16/32 padding
factors **1.068/1.165/1.380/1.866x**, and deterministic post-layer BF16
proxies expose sparse late-depth outliers. LAP-1 therefore starts with the
source-faithful packed-dot gate/up body plus partial-tile handling; LAP-2 must
use block-local residual scaling and exact repair rather than another unchecked
one-plane Q8 route.
[LAP-0 control packet](results/2026-07-24-gfx1151-laguna-prefill-lap0-control.json).

The first LAP-1 packed-dot leaf is positive but not a runtime candidate yet.
On actual layer-1 Q4_K K3072/N1024 gate/up weights with natural routing
counts, the retained direct leaf versus producer-pack-inclusive raw MMQ32 moves
**26.612 -> 10.047 ms (2.649x)** at M256 and **52.522 -> 12.720 ms
(4.129x)** at M512. The current T16 WMMA diagnostic measures
**6.297/9.307 ms** but uses **888 MiB** for the pair versus raw Q4_K's
**864 MiB** and does not satisfy the campaign arithmetic contract. Synthetic
source-Q4_K x DS4-Q8_1 fixtures pass at max softmax KL **4.745e-5** and
**100%** top-1; cached resources are local128, VGPR120, LDS2048B, and
scratch0. At that stage, LAP-1 remained open pending smaller-row tail
scheduling, a lossless replacement-layout comparison, and an exact
fallback/decode path; no default changed.
[LAP-1 partial packed-dot leaf](results/2026-07-24-gfx1151-laguna-q4-k-mmq32-leaf.json).

The subsequent clean all-shape replay measures inclusive raw-MMQ32 speedups
**0.680/0.899/0.985/1.515/1.551/2.645/4.117x** at
M32/55/64/122/128/256/512. Global natural tile32 padding at those shapes is
**10.857/8.558/7.873/5.108/4.928/2.930/1.866x**, versus
**2.911/2.402/2.260/1.721/1.691/1.335/1.165x** for tile8. Thus literal
tile32 loses at M32–M64, misses the 2x premise at M122/M128, and passes at
M256/M512; LAP-1 next builds a smaller-row or mixed full32-plus-tail schedule.
[LAP-1 MMQ32 all-shape screen](results/2026-07-24-gfx1151-laguna-q4-k-mmq32-shape-screen.json).

The byte-neutral X8 replacement layout is the retained LAP-1 prefill-control
primitive. It is BF16-bit identical to raw MMQ32, uses the same
**905,969,664 bytes** for the actual layer-1 gate/up pair, and improves raw at
all seven natural shapes by **9.82–12.14%**. Producer-pack-inclusive X8 reaches
**0.766/1.011/1.105/1.693/1.735/2.957/4.554x** retained direct at
M32/55/64/122/128/256/512. Cached resources remain local128, VGPR120,
LDS2048B, and scratch0. That initial prefill-only result provisionally selected
X8 over raw/T16 but did not change the runtime default: M32 still lost and M128
was below the 2x LAP-1 gate. The exact-decode result below later supersedes the
resident-layout conclusion.
[LAP-1 retained X8 layout primitive](results/2026-07-24-gfx1151-laguna-q4-k-x8-mmq32-layout-retained.json).

The retained live-row schedule closes the LAP-1 natural-shape body gate without
a second tail geometry. It bypasses packed-dot accumulation for padded routes,
reducing the prior X8 time by **18.65–36.45%**. Producer-pack-inclusive X8 now
measures **3.309/4.064/4.283/5.211/5.331/6.515/9.330 ms**, or
**1.197/1.567/1.704/2.526/2.587/4.092/5.614x** retained direct at
M32/55/64/122/128/256/512. Raw/X8 checksums remain exact at every shape,
tracked temporary ownership returns to zero, and cached X8 resources are
local128, VGPR48, LDS2048B, scratch0. The explicit all-full synthetic control
regresses **0.3881 -> 0.4204 ms (+8.34%)**; natural routing remains the
promotion scope, and separate full/tail symbols are deferred unless an
integrated trace exposes that cost. Runtime defaults remain unchanged pending
the direct-T16 result below, arithmetic repair, and full-model gates.
[LAP-1 retained X8 live-row primitive](results/2026-07-24-gfx1151-laguna-q4-k-x8-mmq32-live-row-retained.json).

The clean exact-decode gate rejects X8 as the sole production representation.
The optimized local128 fallback dynamically constructs T16-shaped K256 tiles
in LDS and is BF16-bit exact at c1/c2/c4/c8, but retained T16 -> X8 moves c1
**0.157223 -> 0.174663 ms (+11.093%)** and c2
**0.351996 -> 0.362511 ms (+2.987%)**. X8 catches T16 at c4/c8, but c=1 is the
campaign target and the decode limit is 2%. The comparison records
**905,969,664/931,135,488 bytes** for X8/T16, peaks at
**1,837,482,624 tracked bytes** while both are temporarily resident, and
returns to zero. T16 remains the sole shipping expert layout; X8 remains only
the frozen MMQ ceiling. The direct-T16 result below closes the remaining LAP-1
leaf condition without a layout transpose or duplicate full-family sidecar.
[Rejected sole-resident X8 layout](results/2026-07-25-gfx1151-laguna-q4-k-x8-exact-decode-rejected.json).

The direct-T16 MMQ32 consumer closes LAP-1 without changing resident weights.
It loads the existing T16 metadata and Q4 payload directly into the proven MMQ
cache, produces BF16-bit-identical output to X8, and uses local128/VGPR48/
LDS2048B/scratch0 with packed-dot ISA. Clean producer-pack-inclusive T16 times
are **3.383/4.173/4.419/5.399/5.543/6.769/9.597 ms**, or
**1.174/1.528/1.662/2.464/2.502/3.959/5.502x** retained direct at
M32/55/64/122/128/256/512. The primary shapes remain only
**4.66%/4.05%/3.02%** behind X8, inside the 10% ceiling. Runtime defaults
remain unchanged while LAP-2 calibrates exact repair.
[Retained direct-T16 MMQ32 primitive](results/2026-07-25-gfx1151-laguna-q4-k-t16-mmq32-retained.json).

The earlier post-LPF all-family profile established the pre-AR-O1 bottleneck.
Three alternating non-profiled repetitions measure
**47.453/44.848/38.541 tok/s** at 512/1K/4K. One cached trace covers a 128-row
warmup and those three lengths with complete kernel-sum classification. Selected
Q4/Q6 experts own **56.78/53.23/50.48/43.45%** at 128/512/1K/4K; source-F16
owns **33.40/30.82/29.12/25.02%**; dense/shared quant projections own
**6.55/6.06/5.71/4.94%**; and global+SWA attention owns
**2.46/9.17/14.01/26.01%**. At 4K, global/SWA are **15.92/10.09%**, so the
promoted wave32 path makes global the later attention target. Every next ID,
final cursor, 511/512/513/wrap fixture, and lifecycle gate passes; unclassified
kernel time is below 0.001%, resources are recorded for all 26 symbols, and
kernel-span residual is only **0.28-0.34%**. This is an attribution baseline,
not a speedup claim, and ranks selected experts before source-F16 projections.
[Current all-family profile](results/2026-07-23-gfx1151-laguna-prefill-current-main-all-family-profile.json).

The first AR-O1 selected-expert screen rejects exact Q4T16 dual-SiLU fusion as
a production route. In one resident session with counterbalanced split/fused
order and three repetitions, all 36 next-token IDs match. Rows
16/32/55/64/122/128 move **46.380/48.917/49.088/50.558/51.081/51.412 ->
46.300/49.000/49.137/50.527/51.194/51.549 tok/s**. Aggregate measured wall is
only **0.129%** better, while rows 16 and 64 regress **0.172%/0.060%**, failing
the predeclared all-shape gate. Runtime selector code is removed; the existing
registered leaf and bit-exact kernel test remain. The next AR-O1 screen is the
inclusive Q8_1/dp4a gate/up path.
[Rejected fused-SiLU screen](results/2026-07-23-gfx1151-laguna-prefill-ar-o1-fused-silu-rejected.json).

The second AR-O1 screen finds a real selected-Q4 Q8_1/dp4a speedup but rejects
it on complete model quality. With producer-row activation quantization
included, rows 16/32/55/64/122/128 improve **2.51-4.17%** and aggregate wall
improves **3.773%**, with all 36 screen IDs agreeing. The three-repeat ten-prompt
category gate then improves weighted prefill **4.070%**, h16/h32 E2E
**2.650%/1.916%**, and every category while decode stays neutral. However,
teacher-forced split-vs-Q8 quality reaches maximum KL **0.17156** (>0.05), even
though top-1 is 315/320 and the frozen Poolside first-token gate passes at KL
`1.2837e-4`. Four prompts also have deterministic free-running ID differences.
The predeclared gate therefore removes the env/session selector, Q8 scratch,
runtime route, and harnesses; exact split remains the only production path, and
the independently tested leaf remains kernel-only evidence.
[Positive selected-Q8 screen](results/2026-07-23-gfx1151-laguna-prefill-ar-o1-q8-dp4a-screen.json).
[Rejected selected-Q8 category gate](results/2026-07-23-gfx1151-laguna-prefill-ar-o1-q8-dp4a-category-rejected.json).

The quality-preserving AR-O1 follow-up passes both clean gates and is the
backend-qualified gfx1151 default. One deterministic device pass compacts
active experts and routing metadata; exact C16xR4 Q4/Q6 down reuses each decoded
T16 tile across up to four packed rows, restores lane order before weighted
sum, and falls back to direct below 32 token rows. The shape screen improves
rows 32/55/64/122/128 by **2.63-6.92%** and aggregate wall by **5.461%** while
rows 16 remains on the exact direct fallback.

The one-load, three-repeat ten-prompt h16/h32 category gate moves weighted
prefill **50.193 -> 53.178 tok/s (+5.948%)**, median TTFT **1.627 -> 1.535 s
(-5.682%)**, and E2E **6.079/8.842 -> 6.312/9.087 output tok/s
(+3.835%/+2.762%)**. Code/general_en/general_ja/mixed_ja_en prefill improves
**5.77%/5.34%/5.96%/6.56%** and every category/horizon E2E improves. Decode is
neutral within **0.062%**. All 320 teacher-forced full-logit comparisons are
identical (`KL=0`, top-1 100%), all 30 free-running mode pairs are exact at
both horizons (60 checks), repeated trajectories are deterministic, the
Poolside oracle passes, and lifecycle returns to zero. gfx1100 and rows below
32 retain direct selected GEMV.
[Retained grouped-down shape gate](results/2026-07-23-gfx1151-laguna-prefill-grouped-down-ab.json).
[Retained grouped-down category gate](results/2026-07-23-gfx1151-laguna-prefill-grouped-down-category.json).

The exact grouped-combine follow-up removes the grouped path's final add launch
and selected-output round trip while preserving both BF16 boundaries and all ten
slot-order FMAs. Its clean production-shape micro improves GPU span
**1.249-1.313x** at every 32-128-row shape (**1.265x aggregate**). Five-repeat
complete-model wall is neutral at **0.999716x** with all 60 IDs exact. The
three-repeat ten-prompt gate is likewise non-regressive: prefill
**53.1880 -> 53.1840 tok/s (0.999924x)**, h16/h32 E2E
**0.999769/0.999960x**, and per-category prefill **0.998323-1.001962x**. All
320 teacher-forced comparisons have `KL=0`/100% top-1, all 30 free-running
pairs and repeats are exact, Poolside KL/top-1 remains `6.6214e-6/1.0`, and
lifecycle recovers. gfx1151 now defaults fused combine from 32 rows; explicit
unfused grouped and direct fallbacks remain. The quality-rejected M16 runtime
route/harness debt is removed while its kernel oracle remains. This is a
retained launch/traffic result, not a changed model-wall headline.
[Retained grouped-combine artifact](results/2026-07-23-gfx1151-laguna-prefill-grouped-combine-retained.json).

The follow-up untimed 256/512 routing replay closes the matrix-padding input.
At 256 rows, natural M2/M4/M8/M16/M32 factors are
**1.043/1.134/1.334/1.803/2.924x**; at 512 they are
**1.022/1.068/1.165/1.379/1.867x**. Deterministic Zipf is
**1.050/1.157/1.420/2.049/3.421x** and
**1.025/1.075/1.182/1.465/2.134x**, while the fixed top-10 hot control is 1.0x.
Complete natural per-layer/expert counts, selected-ID hashes, next IDs, build/
model identity, and lifecycle pass. This admits M16 for a measured 256/512
crossover screen; it does not admit blanket M32 or claim throughput.
[Routing-crossover diagnostic](results/2026-07-23-gfx1151-laguna-routing-256-512.json).

The admitted M16 control then passes its clean three-repeat 256/512 screen.
Retained grouped-small-M -> Q4T16/Q6T16 single-output WMMA moves median prefill
**54.591->59.773 tok/s (+9.493%)** and **51.754->56.596 tok/s (+9.356%)**,
with **+9.378%** aggregate synchronized wall. All six next IDs and final
full-logit top-1 values agree, maximum KL is `1.3054e-4`, and lifecycle returns
to zero. This is a positive diagnostic rather than a promotion claim; the full
ten-prompt category quality/E2E gate remains mandatory.
[Positive M16 down screen](results/2026-07-23-gfx1151-laguna-prefill-wmma16-down-screen.json).

That complete three-repeat gate **rejects M16 on quality despite a consistent
wall win**. Across all ten prompts at both 256/512 rows, grouped-small-M -> M16
moves weighted prefill **52.486 -> 57.421 tok/s (+9.404%)**, h16/h32 E2E
**+7.994%/+6.878%**, every shape/category prefill and E2E row positive, and
decode within 0.08%. But maximum final-logit KL is **1.10017** (>0.05), overall
top-1 is 90%, and code/mixed category top-1 is only **87.5%/75%**. Twenty-three
unique shape/prompt/horizon free-running mismatches reproduce in all three
repetitions. M32 shares the same reassociated arithmetic and has worse measured
natural/Zipf padding, so it is closed without another category run. Exact
adaptive grouped-small-M plus fused grouped combine remains the gfx1151
production default.
[Rejected M16 category gate](results/2026-07-23-gfx1151-laguna-prefill-wmma16-down-category-rejected.json).

AR-O2 now has a clean source-F16 matrix-library ceiling at every declared
M16/32/64/128/256/512 shape. The harness screens all seven hipBLASLt algorithms
returned for each production Q/K/V/gate/O M/K/N, then counterbalances exact,
rocBLAS, and selected hipBLASLt full/SWA family sequences. The inclusive library
controls conservatively pay BF16->F32->FP16 input casts before QKV/gate and O,
plus O's FP32->BF16 boundary. At M128, hipBLASLt moves full/SWA
**13.009/18.661 -> 1.132/1.293 ms (11.497x/14.431x)**; the synthetic 12-full/
36-SWA projection sum moves **827.901 -> 60.129 ms (13.769x)**. M16 through
M512 weighted inclusive speedups are **3.866/6.629/10.125/13.769/17.804/
23.263x**. A seeded nonzero mapping smoke passes, every selected algorithm uses
zero workspace, and tracked ownership returns to zero. This is a synthetic
matrix ceiling selecting custom WMMA work, not a runtime route, quality result,
or model-throughput change.
[Source-F16 library ceiling](results/2026-07-23-gfx1151-laguna-f16-library-ceiling.json).

The first clean direct-resident custom-WMMA screen passes every production
M16/32/64/128/256/512 full/SWA family. The candidate reads the existing
row-major F16 allocation, converts BF16 activations to F16 in registers, and
uses bounded LDS only to share activation/weight fragments; there is no
persistent sidecar or inference-time repack. At M128, exact -> WMMA moves
full/SWA **12.772/18.505 -> 1.800/2.474 ms (7.094x/7.479x)** and the synthetic
12-full/36-SWA sum **819.423 -> 110.681 ms (7.403x)**. Full/SWA speedups at
M16/32/64/256/512 are **1.803/1.364x**, **3.105/2.529x**,
**5.403/4.581x**, **9.904/10.507x**, and **9.895/11.138x**. It remains below
the inclusive hipBLASLt ceiling at every shape; M128 reaches **0.628/0.523x**
of the library-control throughput. The nonzero smoke has maximum KL
`3.60e-16`, top-1 100%, all output finite, and tracked ownership returns
exactly to zero. This admits the custom leaf to the complete model quality
lane; it does not change the runtime route or model-throughput headline.
[Source-F16 WMMA screen](results/2026-07-23-gfx1151-laguna-f16-wmma-screen.json).

The complete AR-O2 model gate rejects that direct-resident WMMA schedule despite
its wall win. Across three counterbalanced repetitions of all ten canonical
prompts, tiled -> WMMA moves weighted prefill **53.447 -> 73.637 tok/s
(+37.776%)**, TTFT **1.3651x** faster, and h16/h32 E2E **+21.289%/+14.439%**;
every category improves and decode stays within 0.04%. However, maximum
teacher-forced KL is **0.097062** (>0.05), even though suite top-1 is **317/320
(99.06%)** and each category remains above 90%. Three prompt trajectories differ
in all repetitions. The Poolside first-token oracle passes and tracked ownership
returns exactly to zero. The direct all-layer runtime/category selector is
therefore removed and that independently tested WMMA leaf remains diagnostic;
the compensated SWA-only successor below is a distinct reduction schedule.
[Rejected source-F16 WMMA category gate](results/2026-07-23-gfx1151-laguna-f16-wmma-category-rejected.json).

The compensated successor closes AR-O2's first production route. Each K16 WMMA
partial starts from zero and is Kahan-accumulated in FP32, while direct row-major
F16 residency and BF16-to-F16 register conversion remain unchanged. Its clean
M16-512 screen improves every full/SWA family; at M128 exact -> compensated is
**12.802/18.502 -> 2.190/2.893 ms (5.847x/6.396x)** and the synthetic 12-full/
36-SWA sum improves **819.707 -> 130.422 ms (6.285x)**. The all-layer schedule
still fails quality at KL **0.060389**, so the retained scope is only QKV/gate/O
on the 36 SWA layers from M16; all 12 full-attention layers, M2-15, and decode
stay exact.

Across the clean three-repeat ten-prompt category gate, tiled -> compensated
SWA-only WMMA moves weighted prefill **53.388 -> 69.037 tok/s (+29.313%)**,
median TTFT **1.529 -> 1.187 s (-22.377%)**, and h16/h32 E2E
**6.336/9.120 -> 7.413/10.183 output tok/s (+17.004%/+11.663%)**. Every
category improves prefill **28.206-30.729%** and E2E at both horizons; decode
is neutral. All 320 teacher-forced logits are finite, maximum KL is
**0.043888**, suite top-1 is **318/320 (99.375%)**, every category is at least
96.875%, the Poolside first-token oracle is exact top-1 at KL `4.2951e-5`,
repeats are deterministic, and tracked ownership returns exactly to zero.
`auto` now selects this route on gfx1151; explicit `tiled`/`gemv` remain exact
rollback paths, and unmeasured backends are unchanged.
[Retained compensated SWA source-F16 default](results/2026-07-23-gfx1151-laguna-f16-wmma-comp-swa-retained.json).

AR-O3 then promotes independent matrix and attention chunk policies. In a clean
two-repeat same-load screen, matrix128/attention128 -> matrix512/attention128
moves 512/1K/4K median prefill **64.997/60.385/49.540 ->
69.069/63.925/51.989 tok/s (+6.266%/+5.862%/+4.943%)**. Aggregate median wall
falls **107.516 -> 102.218 s (1.05183x)** and weighted throughput rises
**52.383 -> 55.098 tok/s (+5.183%)**. Every mode/repetition matches complete
logits, final/post-layer hidden, cursor, and all visible global/SWA K/V plus span
bytes; repeats and teardown are exact. The 411,953,168-byte M512 row/MoE scratch
remains below the existing 2-GiB admission floor. gfx1151 now defaults M512
matrix work while attention stays at 128; explicit overrides and unmeasured
backends retain M128. Canonical <=122-token category throughput is unchanged.
[Retained matrix-chunk default](results/2026-07-23-gfx1151-laguna-matrix-chunk-retained.json).

The cached post-matrix512 attribution shows where those retained changes move
the bottleneck. At 512/1K/4K, kernel sum is **7.426/15.998/78.763 s** and
span-minus-sum is only **0.140%/0.171%/0.159%**. Global+SWA attention rises to
**13.42%/19.93%/34.88%** of kernel sum; global alone grows
**3.05%/6.30%/21.34%**, SWA remains **10.36%/13.63%/13.54%**, and selected Q4
gate/up is still largest at **49.69%/46.19%/37.63%**. Relative to the same
pre-matrix 4K trace, source-F16 falls **26.575 -> 7.120 s (-73.21%)** and
selected down **16.461 -> 8.680 s (-47.27%)**, while global/SWA stays nearly
flat at **16.808/10.661 s**. This is attribution, not a new speedup claim; it
selects SWA query-group reuse and then tiled global attention while leaving host
submission deferred.
[Post-matrix512 all-family profile](results/2026-07-23-gfx1151-laguna-prefill-post-matrix512-all-family-profile.json).

The first exact AR-O5 SWA reuse candidate is rejected before a full-model run.
One wave32 handled all nine query heads sharing a KV head, preserving every
logical-slot reduction/softmax/output order and consuming complete
`KVLiveSpans`. Output is byte-exact at the 508..515 wrap and production
M128/full-window shapes, but VGPR rises **32 -> 104**. At M128, retained wave32
-> qgroup9 moves **9.179 -> 9.858 ms (+7.41%)**; at M8 wrap it moves
**1.054 -> 2.945 ms (+179.53%)**. All candidate code is removed. Continue with
query-row tiling/online softmax rather than serializing query heads.
[Rejected SWA qgroup9](results/2026-07-23-gfx1151-laguna-swa-qgroup9-rejected.json).

A matched Poolside llama.cpp `04b2b72c` control now uses the identical Laguna
Q4_K_M model hash, deterministic token stream, BF16 KV, and 128-row microbatch.
Three alternating native `prompt_ms` samples measure
**80.235/103.868/105.435/120.530 tok/s** at 128/512/1K/4K. Against hipEngine's
balanced 512/1K/4K results this is a diagnostic **2.189/2.351/3.127x**. Every
native prompt count is exact, all sampled IDs repeat, model/token hashes match,
and source/binary identity, clocks, GTT/RSS, and load exclusion are recorded.
This is not an eligible cross-engine speed claim: Poolside excludes sampling
and HTTP while hipEngine includes final argmax bookkeeping, and Poolside rounds
the requested 4,097-slot endpoint context to 4,352. It is nevertheless a
same-model lower-bound control showing that hipEngine's long-prefill path has
substantial implementation headroom.
[Matched Poolside control](results/2026-07-23-gfx1151-poolside-laguna-prefill-matched-control.json).

The exact adjacent-head global-attention follow-up is rejected: at the
3,968-prior + 128-current 4K leaf it moves **86.429 -> 125.319 ms (+45.00%)**
despite byte-exact output, so all candidate code was removed. LPF-6 submission/
graph work is deferred because the clean 4K kernel span minus sum is only
**0.302 s / 0.25%** across 35,233 dispatches; packed prefill is a separate c>N
serving scope. [Rejected global-pair artifact](results/2026-07-23-gfx1151-laguna-prefill-lpf5-global-pair-rejected.json).

The matched clean Poolside llama.cpp `04b2b72c` raw-token diagnostic reports
70.463/70.451 prompt tok/s and 19.063/18.882 native predicted tok/s at h16/h32,
with 29.658 s readiness, 76.10 GB sampled GTT, and 1.11 GB sampled RSS. It is
**not** a direct speed ratio: Poolside `predicted_ms` owns all output tokens,
hipEngine decode owns `horizon-1` post-TTFT forwards, and its HTTP wall differs
from the resident in-process boundary. Same-server Poolside output is also only
28/40 exact to hipEngine and 18/20 prompt/horizon groups repeat-deterministic;
the frozen fresh-process Poolside distribution remains the correctness oracle.
Artifacts: [retained LPF-1 hipEngine target AR](results/2026-07-23-gfx1151-laguna-prefill-lpf1-tiled.json),
[previous bulk-GEMV hipEngine row](results/2026-07-22-gfx1151-laguna-s21-target-ar-retained.json),
and [qualified Poolside baseline](results/2026-07-22-gfx1151-poolside-laguna-s21-target-ar-baseline.json).

#### Explicit public B4 DFlash correctness

The explicit-only public gate runs every canonical train/heldout prompt through
true AR, OpenAI blocking DFlash, and OpenAI live-streaming DFlash at the same
32-token limit. All **10/10 AR controls**, **10/10 blocking requests**, and
**10/10 streaming requests** have exact cumulative IDs; blocking/streaming text
also agrees, the EOT-24 case stops identically without leaking
`</assistant>`, and all four categories plus both splits pass. Every request
resets the retained target to position `-1` and the drafter to zero committed
context while preserving the same target/drafter/cycle owners. Closing a public
stream after its first emitted chunk resets both states, and final close releases
the cycle, drafter, target session, target weights, **79,817,890,405 peak tracked
bytes**, and all **1,883 peak allocations** back to zero.

All advertised capability checks pass: source/drafter hashes and revision, B4,
explicit-only/default-off policy, streaming support, target-corrected greedy
exactness, and the retained `0.9469x` fallback evidence/no-performance-claim
flag. The complete gate takes **233.401 s**; its route walls are diagnostic only
because first AR owns cold model load and this is a correctness protocol, not a
speed comparison. D5 therefore supports the pinned B4 path as an opt-in while AR
remains the default. [Public D5 artifact](results/2026-07-23-gfx1151-laguna-dflash-public-e2e.json).

#### Current post-prefill matched B4 DFlash economics

The tracked-clean merged-main `8f8baf9a1` confirmation alternates true AR and
pinned `b0486d1` B4 DFlash over all ten prompts, two repetitions, and 32 visible
outputs. All **20/20** pairs are exact/finite/state-aligned and both routes repeat
deterministically; the Poolside gate and lifecycle pass. Every weighted
throughput metric is within 0.32% of the original post-prefill packet, so the
small movement is confirmation-scale variance rather than a new kernel win.

| Scope | True AR decode tok/s | DFlash B4 decode tok/s | DFlash / AR | Draft acceptance | Target rows/output |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full 10 prompts | **16.384** | 15.527 | **0.9477x** | 50.48% | 1.6935 |
| Train, 6 prompts | 16.441 | 17.180 | 1.0450x | 58.33% | 1.5323 |
| Heldout, 4 prompts | 16.299 | 13.569 | 0.8325x | 41.15% | 1.9355 |
| `code` | 16.562 | **21.020** | **1.2692x** | 78.23% | 1.2500 |
| `general_en` | 16.707 | 12.690 | 0.7595x | 36.54% | 2.0968 |
| `general_ja` | 16.245 | 13.647 | 0.8401x | 40.63% | 1.9355 |
| `mixed_ja_en` | 15.873 | 13.372 | 0.8424x | 39.58% | 1.9355 |

Weighted prefill is **50.389 tok/s AR** versus **16.906 tok/s DFlash**. LPF-1/5
cut target verification **50.493 -> 32.644 s (-35.35%)** and improve DFlash
decode **10.715 -> 15.527 tok/s (+44.91%)** at unchanged 424/840 draft
acceptance. Full-suite decode still misses the >1.10x gate. Median TTFT moves
**1.620 -> 4.767 s** and E2E **8.872 -> 4.503 output tok/s (0.5075x)** because
DFlash prompt capture remains serial. Code wins, but heldout and all non-code
categories regress; AR remains default and automatic/performance promotion stays
deferred, while the exact explicit-only route is supported by the separate D5
public gate above. [Current-main compact confirmation](results/2026-07-23-gfx1151-laguna-dflash-current-main-confirmation.json); [original full diagnostic](results/2026-07-23-gfx1151-laguna-dflash-category-economics-post-prefill.json).

#### Pre-LPF-1 matched B4 DFlash economics (historical)

At the pre-LPF-1 source, the admitted Poolside revision `b0486d1` BF16 drafter
ran in one resident process against a true no-DFlash target path over the same
10 prompts, fixed 32
visible outputs, two repetitions, and alternating route order. Decode timing
starts after each synchronized first token and includes 31 visible outputs;
model load is excluded. All **20/20** AR/DFlash pairs are exact, all values are
finite, both routes are repeat-deterministic, target/drafter cursors satisfy one
of the two valid fixed-horizon commit boundaries, the frozen Poolside
first-token gate passes, and tracked ownership returns to zero.

| Scope | True AR decode tok/s | DFlash B4 decode tok/s | DFlash / AR | Draft acceptance | Accepted / output | Target rows / output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full 10 prompts | **16.388** | 10.715 | **0.6538x (-34.62%)** | 50.48% | 0.6839 | 1.6935 |
| Train, 6 prompts | 16.445 | 11.855 | 0.7209x | 58.33% | 0.7151 | 1.5323 |
| Heldout, 4 prompts | 16.303 | 9.365 | 0.5744x | 41.15% | 0.6371 | 1.9355 |
| `code` | 16.564 | 14.522 | 0.8768x | 78.23% | 0.7823 | 1.2500 |
| `general_en` | 16.714 | 8.751 | 0.5236x | 36.54% | 0.6129 | 2.0968 |
| `general_ja` | 16.248 | 9.405 | 0.5788x | 40.63% | 0.6290 | 1.9355 |
| `mixed_ja_en` | 15.879 | 9.234 | 0.5815x | 39.58% | 0.6129 | 1.9355 |

Across 210 cycles, DFlash accepts 424/840 proposed tokens; 18.10% of cycles
accept zero and 30.48% accept all four. The synchronized decode wall is 57.861
s: proposal is 6.721 s, target verification is **50.493 s (87.27%)**, and
post-verify/commit residual is 0.645 s. Median TTFT regresses
**3.478 -> 4.764 s (+36.98%)** because AR uses the retained bulk prefill while
DFlash still seeds target hidden captures serially. Fixed-horizon E2E moves
**5.724 -> 4.000 output tok/s (-30.12%)**. Combined resident ownership is
79,349,505,533 bytes (target 77,099,132,853; drafter 2,250,372,680), tracked
peak is 79,349,726,717 bytes, and teardown is exact.

The pre-LPF-1 >1.10x promotion gate failed decisively on decode before TTFT was
considered. LPF-1 now changes every B+1 target verifier with two or more rows,
so these economics are no longer a current promotion decision; the full suite
must be refreshed after the remaining prefill plan stabilizes. AR stays default
and DFlash remains off. The artifact preserves the complete historical timing
run plus an explicit offline reclassification of its derived fixed-horizon
state predicate; no measurement or acceptance value changed.
[Stale diagnostic artifact](results/2026-07-23-gfx1151-laguna-dflash-category-economics.json).

#### Cold startup

On the Radeon 8060S/gfx1151 UMA host, the versioned `laguna-repacked-v1`
artifact removes all runtime Q4T16/Q6T16/pack8 conversion.
The timed scope begins at GGUF/cache open and ends with all 814 weights resident;
cache construction and teardown are excluded. A run is cold-streamed only when
`/proc/self/io` reports physical reads of at least 80% of the 75,169,369,088
planned source bytes.

| Route | Cache state | Load samples (s) | Median | Change vs natural | Loader correctness / lifecycle |
| --- | --- | ---: | ---: | ---: | --- |
| Natural GGUF + NumPy replacement-layout repack | Cold-streamed | 227.510 | 227.510 s | baseline | KL `6.6214e-6`, top-1 100%; exact tracked recovery |
| **Versioned buffered repacked cache** | **Cold-streamed** | **48.812 / 47.951 / 48.202** | **48.202 s** | **-78.81%; 4.72x faster** | Same hipEngine IDs/logits; KL/top-1 gate and exact tracked recovery pass |

The retained cache contains 262 transformed entries / 70,718,767,104 bytes and
is source-bound by plan fingerprint, size/mtime, and the known GGUF SHA-256.
Each cold profiled run physically read 72.1-72.3 GB and recorded zero repack;
~40.4-41.6 s is now sequential reading and ~6.5-6.8 s is allocation/upload. A
28.655 s partially cached run is diagnostic only. Poolside's external
29.851-29.907 s `--no-repack --no-mmap` readiness is not a retained cross-engine
comparison because its physical-read state was not instrumented. The cache
reproduces the previously documented 29/32 natural Poolside greedy prefix,
31/32 teacher-forced top-1, repeat logits max-abs `0`, and finite taps; the
pre-existing low-margin token-30 branch means this is not exact Poolside
Greedy-32 parity. [Retained startup
artifact](results/2026-07-22-gfx1151-laguna-s21-repacked-cache-startup-retained.json).

### gfx1100 PARO context capacity and mixed-KV fidelity, 2026-07-14

**Status: 208 Ki BF16 is the recommended safe cap on a physical 24 GB card;
220 Ki is a validated edge, and neither all-layer INT8 nor native tail-four
mixed KV makes 256K a supported route.** Clean hipEngine `5a49b16d` ran
profile-aware BF16 sweeps on the W7900 and directly on this host's
25,753,026,560-byte (23.984 GiB) RX 7900 XTX. Every retained BF16 row uses the
current Qwen3.6 packed PARO snapshot, repeated token `9707`, 128 decode tokens,
full-run approximately 1 Hz whole-device monitoring, finite-output checks, and
a passing layout audit. Clean `d6504544` supplies the separate compact 256K
all-layer INT8 row; the native mixed-KV diagnostic records exact source hashes
and physical request-scratch probes in the July 14 artifact.

<!-- BEGIN TOPLINE:W7900_MEMORY_CAPACITY -->
| Route / profile | Hardware | Context/decode | Tracked peak | Observed device peak | Device/card margin | Capacity / quality status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| PARO BF16 KV reference | W7900, default chunks | 128K/128 | **22.124 GiB** | 21.107 GiB phase sample | n/a | Reference path |
| PARO BF16 KV, automatic 24 GB low-memory profile | RX 7900 XTX 24 GB | **208 Ki (212,992)/128** | **23.082 GiB** | **23.623 GiB** | **+0.361 GiB** | **Recommended practical safe cap** |
| PARO BF16 KV, automatic 24 GB low-memory profile | RX 7900 XTX 24 GB | 220 Ki (225,280)/128 | 23.369 GiB | **23.908 GiB** | **+0.076 GiB (~78 MiB)** | Physical pass, but **edge only—not safe cap** |
| PARO BF16 KV, default 48 GB-card profile | W7900 | 220 Ki (225,280)/128 | 24.090 GiB | at least 24.832 GiB | at most -0.848 GiB vs 24 GB card | Rejected for this larger-chunk profile |
| PARO INT8 per-token/head KV, FP16 scales | W7900 | 256K/128 | **22.971 GiB** | 21.041 GiB phase sample | +1.029 GiB tracked | **Rejected** by Qwen3.6 matched-context and task gates |
| PARO tail-four Hadamard-group32 mixed KV, BF16-oracle prefill | RX 7900 XTX 24 GB | 256 Ki (262,144)/128 request scratch | **23.469 GiB before failed allocation** | 22.566 GiB after clean OOM | 1.418 GiB free before request scratch; insufficient | **Rejected:** `HIP error 2` OOM and PARO fidelity failure; no segfault |
| PARO tail-four Hadamard-group32 mixed KV, direct-streaming control | RX 7900 XTX 24 GB | 256 Ki (262,144)/128 request scratch | **23.290 GiB** | **23.590 GiB** live sample | **+0.394 GiB** live | Allocation passes, but direct packed prefill is **correctness-rejected** |

The native explicit `tail4_hadamard_group32` layout keeps K/V for
full-attention layers `3,7,11,15,19,23` in BF16 and stores only layers
`27,31,35,39` as Hadamard-group32 INT8 with FP16 scales. At 262,400 retained
rows it uses `4,366,336,000` K/V bytes—**18.75% below BF16**—with no persistent
BF16 shadow. PARO's quality-preserving prefill uses a temporary BF16 oracle;
GGUF's post-quality layout audit reports zero persistent oracle/mirror buffers.
Native PARO still fails 1/11 prompts at 512/8 and 2/11 at 4K/16 (58.82%
worst-prompt top-1), and its 256 Ki quality-preserving request scratch OOMs.

The clean `c971262f` therock-7.15 GGUF-only closure passes all 11 prompts at
512/8 (max KL `0.007455`, top-1 100%) and 4K/16 (mean/max KL
`0.0001369/0.009926`, aggregate/minimum-prompt top-1 `99.47%/94.12%`) plus
bounded `mixed_v1` at 128K/16 (max KL `5.19e-5`, top-1 100%). At 128K,
persistent K/V is `2,185,297,920` bytes versus BF16 `2,689,597,440` bytes and
live owned memory falls `24.168 -> 23.698 GiB`. It still rejects promotion:
production 4K prefill/decode regress `0.67%/0.75%`, one-shot 128K decode
regresses `3.82%`, and production prefill allocates then frees
`1,075,838,976` bytes—byte-exact to four BF16 layer caches—raising allocator
high water `24.168 -> 24.700 GiB`. The transient attribution is inferred from
the exact bytes; it is not a persistent shadow. The policy remains explicit
and non-default. Evidence:
`benchmarks/results/2026-07-15-gfx1100-gguf-tail4-hadamard-clean-gate.json` and
`benchmarks/results/2026-07-14-gfx1100-native-tail4-hadamard-kv-outcome.json`.
<!-- END TOPLINE:W7900_MEMORY_CAPACITY -->

The physical-card result differs from the earlier W7900 220 Ki rejection
because the runtime is card-aware. W7900's 48 GB total selects
`1024/1024/4096/1024/1024` prefill chunks; a 24 GB card automatically selects
all-`768` via `low_memory_full_context_24gb`. The earlier W7900 result remains a
valid rejection of its default larger-chunk profile, but it cannot by itself
predict behavior under the actual low-memory profile.

The clean profile-aware sweep is:

| Hardware / prefill profile | Context | Tracked peak | Full-run device peak | Margin vs physical 24 GB bytes | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| W7900 default `1024/4096` | 176 Ki | 23.033 GiB | 23.779 GiB | +0.205 GiB | Fits byte envelope, limited margin |
| W7900 default `1024/4096` | 184 Ki | 23.226 GiB | 23.971 GiB | +0.013 GiB | Not safe |
| W7900 default `1024/4096` | 200 Ki | 23.610 GiB | 24.356 GiB | -0.371 GiB | Does not model a fitting 24 GB route |
| RX 7900 XTX automatic all-`768` | 176 Ki | 22.315 GiB | 22.857 GiB | +1.127 GiB | Direct physical-card pass |
| RX 7900 XTX automatic all-`768` | **208 Ki** | **23.082 GiB** | **23.623 GiB** | **+0.361 GiB** | **Recommended safe cap** |
| RX 7900 XTX automatic all-`768` | 220 Ki | 23.369 GiB | 23.908 GiB | +0.076 GiB | Direct pass, edge only |
| W7900 manual all-`768` screen | 232 Ki | 23.657 GiB | 24.163 GiB | -0.178 GiB | Rejected without risking physical-card OOM |

For this report, “safe” requires a directly tested point with at least 0.25 GiB
of observed whole-device headroom. Thus **208 Ki (212,992 tokens)** is the
practical cap; ordinary 200K or 200 Ki requests are below it. **220 Ki is the
largest physically validated point**, but only about 78 MiB remains, so it
should not be the configured maximum. The exact mathematical frontier is not
claimed: 209-219 Ki were not tested. No row segfaulted; the 232 Ki physical-card
run was intentionally skipped after its same-profile W7900 screen exceeded the
target byte capacity. Throughput is single-run/concurrent diagnostic data only,
not a performance claim.

The clean 256K/128 INT8 row retains 2,686,976,000 payload bytes plus 20,992,000
FP16 scale bytes across ten full-attention layers. The compact table is
16,793,600 bytes (`4,096 x 1,025` INT32 entries). Tracked peak falls
**25,723,838,504 -> 24,665,296,404 bytes** (**23.957 -> 22.971 GiB**, -0.986
GiB / -4.12%), increasing the 24 GiB margin from 0.043 to 1.029 GiB. One-shot
diagnostic throughput is effectively flat within run variance: prefill
632.837 -> 631.457 tok/s and decode 40.066 -> 40.008 tok/s.

The final BF16-reference-token matched 128K/16 gate is finite and passes the
no-shadow audit, but rejects at mean/max KL **0.85128/4.97382** and **41.18%**
top-1 agreement. This is the intrinsic comparison; the older 128K/128
independent-rollout KL/top-1 headline includes cascade after histories diverge.
Clipping, group16/32/64, K/V mixed formats, selective BF16 layers/heads, and
sink/recent residual windows all failed to clear both gates at 4K within the
reclaimed budget.

The clean `d0b56364` external-format screen adds matched-context top-k evidence
without changing support status. Its fixed 512/8 mixed-prompt S1 run completed
in **28.78 s** including setup (600 s budget):

| Emulated INT8 representation | Mean / max KL | Top-1 | Top-5 / top-10 overlap | 256K bytes / extra vs baseline | S1 decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Current per-token/head baseline | `0.36841 / 1.20200` | `66.67%` | `71.11% / 77.78%` | `2.520 GiB / 0` | Reject anchor |
| **Hadamard group32** | **`0.13342 / 0.45135`** | `77.78%` | `84.44% / 84.44%` | `2.656 / +0.137 GiB` | Lowest mean KL; transfer |
| KIVI-style K-per-channel/V-per-token-group | `0.16667 / 0.60739` | **`88.89%`** | **`86.67% / 88.89%`** | `2.698 / +0.178 GiB` | Better decision fidelity; higher primary KL |
| KVarN-informed eight-pass INT8 + BF16 sink/tail | `0.27125 / 1.25017` | **`88.89%`** | `77.78% / 80.00%` | `2.593 / +0.073 GiB` | Improve baseline only |

Only Hadamard group32 advanced to 4K/16. It passes top-1 at **94.12%** but
rejects mean/max KL at **`0.15512/1.14267`**; top-5/top-10 overlap is
`88.24%/84.71%`. The run stops before native kernels, 128K, or task benchmarks.
This is a representation diagnostic with `performance_claim=false`, not a new
supported cache format.

The same representation set was then isolated on hipEngine GGUF with identical
Q4_K_M weights, BF16 reference, fixed `mixed_v1` prompts, and teacher history.
Unlike PARO, every GGUF row passes the 512/8 screen by a wide margin; plain
symmetric group32 has the lowest mean KL:

| hipEngine GGUF representation | 512/8 mean / max KL | 512/8 top-1 | 4K/16 mean / max KL | 4K/16 top-1 | Transfer |
| --- | ---: | ---: | ---: | ---: | --- |
| Per-token/head max-abs | `0.0001646 / 0.0005551` | `100%` | **`0.12779 / 2.03039`** | `88.24%` | Reject |
| **Plain group32 (Q8_0 storage geometry)** | **`0.0000812 / 0.0003984`** | `100%` | `0.28106 / 4.39924` | `88.24%` | Reject |
| Hadamard group32 | `0.0000974 / 0.0003191` | `100%` | `0.25180 / 4.09533` | **`94.12%`** | Reject KL |
| KIVI-style INT8 | `0.0001753 / 0.0012793` | `100%` | `0.33306 / 5.43878` | `88.24%` | Reject |

The 4K failures are dominated by `decode_3`; Hadamard preserves 16/17 top-1
rows, while the other formats also miss `decode_4`. A reverse candidate-order
run reproduces every per-position KL and candidate top-1 exactly, ruling out
session-reset or ordering contamination. Compact evidence:
[`2026-07-13-w7900-gguf-int8-kv-external-format-screen.json`](results/2026-07-13-w7900-gguf-int8-kv-external-format-screen.json).

The exact native follow-up separates prompt content, Q8 K, Q8 V, and host
reconstruction on the same mixed token IDs:

| Engine / cache arithmetic | Repeated 4K/16 mean KL | Mixed 4K/16 mean / max KL | Mixed top-1 | Verdict |
| --- | ---: | ---: | ---: | --- |
| llama.cpp F16/F16 control | — | `0 / 0` | `100%` | Deterministic control |
| **llama.cpp native Q8_0 K/V** | **`0.00000619`** | **`0.075654 / 1.26009`** | **`94.12%`** | Reject KL |
| llama.cpp native Q8 K / F16 V | — | `0.096682 / 1.56852` | `94.12%` | Reject KL |
| llama.cpp native F16 K / Q8 V | — | `0.243219 / 3.99543` | `94.12%` | Reject KL; largest isolated error |
| hipEngine native per-head INT8 | `0.00000235` | `0.19038 / 2.99555` | `88.24%` | Reject both |

Mixed input raises llama.cpp full-Q8 mean KL 12,227x. F16/F16 is exactly zero,
and exact reruns reproduce both llama.cpp full-Q8 and hipEngine native
per-head results. Q8_0 has no F16 shadow: it writes FP32 K/V to INT8+FP16
block32 scales, quantizes Q to Q8_1 for integer K dots, and uses FP16 V/softmax
accumulation on RDNA3. V-only Q8 is worse than K-only, while full Q8 is better
than either, showing partial K/V error cancellation.

Direct arithmetic is not a universal repair. Native llama.cpp full-Q8 is 73.08%
lower mean KL than host group32 on mixed input, but still rejects. Native
hipEngine per-head is 48.98% worse than its host-reconstruction row
(`0.19038` versus `0.12779`). Therefore no group32/Hadamard native kernel or
128K gate follows. The prior llama.cpp repeated 128K/16 result
(`0.00521/0.08749`, 100% top-1) remains mechanically valid only as a saturation
control; it no longer establishes representative eight-bit fidelity.

The same-weight cross-engine BF16 bridge remains separate: hipEngine GGUF BF16
versus llama.cpp F16 at repeated 128K/16 rejects aggregate mean/max KL at
`0.26606/4.51481` because of prompt-final drift, while decode-only mean KL is
`0.000510` with 100% top-1. It does not isolate cache dtype.

The original free-generation task smoke remains `reference_unscorable` because
BF16 scored 0/5. A replacement restricted-choice probe provides partial bounded
functional evidence: at 4K, BF16 qualifies 2/5 and INT8 flips the qualified
multihop answer `D -> C` while retaining aggregation; at 32K, BF16 qualifies
3/5 and INT8 retains all three (multihop, aggregation, long-document). Thus high
KL can change a real decision but does not imply every answer changes. This is
not a full/free-generation quality claim and does not make 256K INT8 supported.

Capacity and outcome artifacts:
[profile-aware BF16 frontier](results/2026-07-13-gfx1100-paro-bf16-context-frontier.json),
[W7900 default-profile 220 Ki diagnostic](results/2026-07-13-w7900-paro-bf16-220ki-capacity.json), and
[INT8 accuracy outcome](results/2026-07-13-w7900-paro-int8-kv-accuracy-outcome.json).
Detailed diagnostics:
[llama.cpp Q8_0 repeated-token matched quality](results/2026-07-13-w7900-llamacpp-q8-kv-matched-quality.json),
[repeated/mixed prompt and native K/V arithmetic isolation](results/2026-07-13-w7900-gguf-q8-kv-protocol-arithmetic-isolation.json),
[same-weight GGUF bridge](results/2026-07-13-w7900-gguf-llamacpp-matched-parity.json),
[bounded functional check](results/2026-07-13-w7900-paro-int8-kv-functional-mc.json),
[matched baseline](results/2026-07-13-w7900-paro-int8-kv-fidelity-baseline.json),
[format screen](results/2026-07-13-w7900-paro-kv-format-ablation.json),
[PARO external-format KL/top-k screen](results/2026-07-13-w7900-paro-int8-kv-external-format-screen.json),
[same-weight GGUF external-format screen](results/2026-07-13-w7900-gguf-int8-kv-external-format-screen.json), and
[policy screen](results/2026-07-13-w7900-paro-kv-policy-ablation.json).

### W7900 PARO gfx1151 transfer gate, 2026-07-12

**Status: retained scoped-default validation and retained negative transfer
decision.** Clean detached `255e5aca` on W7900/GPU0 used the exact packed W4
PARO/BF16-KV model, repeated token `9707`, graph decode, TheRock HIP 7.15, two
discarded plus five measured runs per leg, and cached JIT. Because the first
off/on/off AOTriton sequence drifted with run order, the reverse on/off/on
sequence completes a balanced 15-sample comparison per mode.

| Workload | Same-stream prefill | Isolated prefill | Prefill delta | Total measured wall reduction |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | `2843.083` | `2889.650` | **+1.638%** | **1.653%** |
| 1K/128 | `2951.433` | `2966.051` | **+0.495%** | **0.127%** |
| 4K/128 | `2924.276` | `2929.897` | **+0.192%** | **0.562%** |

At 512, 1K, and 4K the isolated and same-stream legs match sampled seed, final
hidden, all 30 Conv/GDN state families, and all 10 live K/V families
byte-for-byte. This matrix was measured before queue isolation was narrowed by
query shape, so its isolated leg includes the 256-query 512/1K route as well as
the 4096-query 4K route. The merged runtime keeps 256-query AOTriton on the
caller stream and isolates query rows >=512. The 4K result therefore directly
validates the merged gfx1100 default; 512/1K remain supporting exact transfer
diagnostics rather than claims for the final route. No additional runtime
change is needed.

The architecture-specific chunk profile does not transfer. With AOTriton queue
mode held equally same-stream, linear/MoE-256 changes prefill by
`-7.723%/-8.782%/-6.398%` at 512/1K/4K. Its `0.58%-1.72%` tracked-memory
reduction does not offset disjoint, uniformly slower throughput ranges, so
`gfx1100` keeps the generic chunk policy.

Artifact:
[`2026-07-12-w7900-gfx1100-paro-gfx1151-transfer.json`](results/2026-07-12-w7900-gfx1100-paro-gfx1151-transfer.json).

### Superseded W7900 model sweep, 2026-07-07

**Status: superseded diagnostic.** This used one max-128K hipEngine session,
eager GGUF decode, one llama.cpp sample per phase, and no W7900-local state
oracle. The accepted clean 2026-07-12 table above replaces it. Historical
artifacts remain available from
[`2026-07-07...summary.json`](results/2026-07-07-w7900-gpu0-readme-refresh-20260707-104756-summary.json).

### Superseded gfx1151 model sweep, 2026-06-15

**Status: superseded diagnostic.** This was one measured run per shape with no
measured warmup, incomplete summary provenance, and unusable 512 MiB aperture
memory readings for llama.cpp. The accepted 2026-07-11 sweep above replaces
every public row with five-sample, clean-provenance evidence and proper GTT
sampling. Keep the old record only for history.

Artifacts: [old summary](results/2026-06-15-gfx1151-readme-udq4km-20260615-040438-summary.json),
[hipEngine PARO](results/2026-06-15-gfx1151-readme-udq4km-20260615-040438-hipengine-paro-packed-1run.json),
[hipEngine GGUF](results/2026-06-15-gfx1151-readme-udq4km-20260615-040438-hipengine-gguf-ud-q4km-1run.json),
[llama.cpp HIP](results/2026-06-15-gfx1151-readme-udq4km-20260615-040438-llamacpp-hip-ud-q4km-f16kv.json),
and [llama.cpp Vulkan](results/2026-06-15-gfx1151-readme-udq4km-20260615-040438-llamacpp-vulkan-ud-q4km-f16kv.json).

### W7900 direct GGUF concurrency, 2026-07-17

**Status: retained direct native-c4/c8 model-step throughput and retained real
OpenAI SSE arbitrary-C server scaling.** All rows use the same Qwen3.6-35B-A3B
`UD-Q4_K_M`, BF16 KV, greedy top-1, W7900/gfx1100, and TheRock HIP 7.15.
The two tables deliberately keep timing scopes separate: direct rows time
synchronized graph steps; server rows time complete concurrent TestClient SSE
cycles, including admission, prompt work, decode, delivery, and completion.

<!-- BEGIN TOPLINE:W7900_CONCURRENCY -->
| Direct route | Logical C | Native groups | Aggregate decode tok/s | Per-request tok/s | Aggregate / c1 | Aggregate / serial-c4 | TTFT p50 / p95 | Model-step ITL p50 / p95 | Tracked peak |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct c1 | 1 | 1x c1 | 85.469 | 85.469 | 1.000x | 1.009x | 0.209 / 0.209 s | 11.693 / 11.955 ms | 21.783 GiB |
| direct c2 | 2 | 1x c2 | 127.427 | 63.714 | 1.491x | 1.504x | 0.951 / 0.954 s | 15.765 / 16.023 ms | 22.394 GiB |
| direct c4 | 4 | 1x c4 | 184.575 | 46.144 | 2.160x | 2.178x | 2.020 / 2.023 s | 21.715 / 22.021 ms | 23.396 GiB |
| **direct c8** | **8** | **1x c8** | **246.872** | **30.859** | **2.888x** | **2.913x** | **3.475 / 3.479 s** | **32.414 / 32.749 ms** | **25.401 GiB** |
| chunked c8 control | 8 | 2x c4, serialized | 183.020 | 22.878 | 2.141x | 2.160x | 3.055 / 4.084 s | 43.767 / 44.281 ms | 26.069 GiB* |
| serial-c4 rate control | 4 | 4x c1, serialized | 84.738 | 21.185 | 0.991x | 1.000x | 0.548 / 0.877 s | 47.225 / 48.142 ms | 26.985 GiB* |

| Real OpenAI SSE route | Logical C | Physical execution | Aggregate generated tok/s | Per-request tok/s | Aggregate / logical-c1 | Aggregate / serial-c13 | Cycle wall p50 | Scheduler TTFT p50 / p95 | Scheduler ITL p50 / p95 | Cumulative tracked peak |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logical-c1 control | 1 | masked physical c8 | 25.583 | 25.583 | 1.000x | 0.807x | 5.003 s | 0.271 / 0.276 s | 36.499 / 39.809 ms | 29.312 GiB |
| physical c8 | 8 | 1x c8 | **136.122** | 17.015 | **5.321x** | 4.293x | 7.523 s | 1.751 / 2.088 s | 41.712 / 45.068 ms | 30.805 GiB* |
| grouped c9 | 9 | c8 + sparse c8 | 88.592 | 9.844 | 3.463x | 2.794x | 13.003 s | 1.709 / 2.235 s | 81.087 / 88.112 ms | 31.969 GiB* |
| **grouped c13** | **13** | **c8 + sparse c8** | **111.380** | **8.568** | **4.354x** | **3.513x** | **14.940 s** | **1.886 / 3.323 s** | **87.502 / 93.631 ms** | **32.869 GiB*** |
| serial-c13 bridge | 13 | 13x c1 serial | 31.708 | 2.439 | 1.239x | 1.000x | 52.479 s | 2.424 / 3.390 s | 382.821 / 396.004 ms | 32.869 GiB* |
<!-- END TOPLINE:W7900_CONCURRENCY -->

Direct protocol: prompt 512 per row, 128 decode transitions, one discarded
full-route warmup and median of three, one shared model load. Resident sessions
grow c1→c2→c4→c8; direct starred controls retain later allocations. Native c8
improves aggregate decode **188.84%** over c1 and **34.89%** over c4+c4, while
its lower per-request rate and higher ITL remain explicit. The marker slice is
**748 packed-native / 0 row-local / 0 copies**.

Server protocol: 512 exact round-tripped prompt IDs and 128 generated outputs
per request, 20 ms admission window, one discarded burst plus three measured
bursts per static route, one prepared 13-slot runner, and scheduler-owned
latency samples. Logical c1 is honestly a masked physical-c8 production control,
not relabelled native c1. C9/C13 are multiple declared groups, never native
widths. All **189/189** requests match actual resident prompt IDs, direct-c1
output IDs, OpenAI usage, and finish metadata; every static rate has <=1.299%
stdev/median. The controlled live trace observes physical c8 before admitting
five tail requests, reaches c8+masked-c8 C13, emits **1,664/1,664** exact IDs at
**107.284 aggregate tok/s**, and finishes with zero request/session ownership.
Optional compaction separately preserves nine moved survivors' state/KV and
resource identities and invalidates both pinned sparse graphs, but remains an
explicit diagnostic operation. Server memory is cumulative in fixed execution
order; starred rows are not isolated allocation deltas. gfx1151 is retained
separately in its E1/E3/F1 row.

Artifacts: [retained C4 throughput closure](results/2026-07-16-gfx1100-gguf-concurrency-c4-native-graph-scaling-closure.json),
[D4 OpenAI lifecycle closure](results/2026-07-16-gfx1100-gguf-concurrency-d4-openai-streaming-closure.json),
[D5 live observability](results/2026-07-17-gfx1100-gguf-concurrency-d5-live-observability-closure.json),
[E2 native-c8 correctness](results/2026-07-17-gfx1100-gguf-concurrency-e2-native-c8-correctness.json),
[E2 native-c8 scaling](results/2026-07-17-gfx1100-gguf-concurrency-e2-native-c8-scaling-closure.json),
[E3 arbitrary-C correctness](results/2026-07-17-gfx1100-gguf-concurrency-e3-arbitrary-c-correctness.json),
and [F1 real server scaling](results/2026-07-17-gfx1100-gguf-concurrency-f1-server-scaling-closure.json).
Historical mixed-quant/mixed-scope references remain linked in
[`HISTORY.md`](HISTORY.md) and the 2026-07-07 result directories.

### gfx1151 PARO exact shape/routing catalog, 2026-07-11

**Status: historical c1 and fail-closed production-routing anchor; explicit c2
is superseded by the retained 2026-07-18 G2 row.** P1 ran from clean detached
hipEngine `a18ff7bc`; P2 ran from clean detached `6f1910c9` on the same Radeon
8060S and exact model/fixture. P1 uses the same 512-token row at every width and
compares 137 generated IDs against true single-request sessions. P2 uses ragged
lengths 449 through 512 and checks every persistent state/KV family through
c8-to-c1 retirement.

No eligible native-batch timing row existed in this July 11 snapshot. The old
c2/c4/c8 candidates below are superseded by the exact G3 selected-batch widths
and G5 resident server promotion. c3/c5/c6/c7 remain rejected as native widths;
current production partitions them into certified c2/c4 plus c1 groups as shown
in the concurrency table above.

Protocol: Qwen3.6-35B-A3B PARO snapshot
`437eba06df05aad71a4dacdcaf3fff70ae1ee8a1`, W4 PARO, BF16 KV, 40 layers,
8 warmup decode steps, 128 measured decode steps, and greedy sampling. Exact
prompt-ID SHA-256 is `b162b2d0...2388`; model fingerprint is
`995a8c67...d917`. c1 is the median of three fresh processes
(`66.948/66.754/66.910 tok/s`). c2-c8 have one diagnostic native timing each;
correctness rejection makes more repetitions immaterial.

Rejected native diagnostics:

| c | Candidate shape | Equal prefix per row | Aggregate tok/s | Decision |
| ---: | --- | --- | ---: | --- |
| 2 | full native, selected-c1 MoE, batch-GEMV output | `2,2` | 78.525 | reject at index 2 (`17` vs `220`) |
| 3 | rowchunk2 full attention, selected-c1 MoE | `2,2,2` | 87.472 | reject at index 2 |
| 4 | rowchunk2 full attention, selected-c1 MoE | `2 x4` | 99.641 | reject at index 2 |
| 5 | rowchunk2 full attention, selected-c1 MoE | `2 x5` | 102.178 | reject at index 2 |
| 6 | selected-layer rowchunk2, selected-c1 MoE | `2 x6` | 109.806 | reject at index 2 |
| 7 | rowchunk2 full attention, selected-c1 MoE | `2 x7` | 109.580 | reject at index 2 |
| 8 | rowchunk2 full attention, selected-c1 MoE | `2 x8` | 115.508 | reject at index 2 |

The c8 teacher-forced bisect keeps packed-prefill hidden, recurrent state, and
full-attention KV bit-exact. On decode step 0, the selected-c1 route first
changes the input/state of linear layer 4; the visible token flips on the next
step. Grouped-compact produces the correct token at index 2 but fails the full
shrinking sequence at index 4, so it is not a replacement default.

The retained P2 lifecycle gate keeps physical slots sparse and un-compacted.
Slot 3 exits by EOS at c8; later explicit cancellation creates middle, tail,
and front holes while slot 4 survives to c1. Every generated sequence, all 30
linear Conv/GDN state pairs, and all 10 live K/V layer pairs are SHA-256 exact
against independent c1 at each row's retirement boundary. Ragged packed prefill
selects `per_segment_ragged_exact`; equal-length packed prefill is unchanged.

Run record:

| Field | Value |
| --- | --- |
| GPU/backend | AMD Ryzen AI MAX+ 395 / Radeon 8060S, detected gfx1151, target gfx1151 |
| Source/build | clean hipEngine `a18ff7bc428833a5f3d87ed422d04633abbf0b10`; Python 3.12.13; TheRock HIP `7.13.60980-c76140fa27`; detected/target gfx1151 |
| Timing scope | Direct resident backend decode wall; c1 median of 3; rejected native rows one run each |
| Correctness | In this historical snapshot, c1 endpoints repeat across 3/3 runs and c2-c8 candidates fail every row at index 2. The 2026-07-18 G3 selected-batch row supersedes c2/c4/c8; G5 supersedes the public serial-only route. The historical serial route passes ragged c8-to-c1 for 8/8 token/state/KV rows. |
| Production route | At this historical snapshot: `true_c1_graph` for c1 and `scheduler_true_c1_fallback` for public c2-c8. G5 now supersedes that route with package-default c2/c4/c8 resident groups. |
| Lifecycle route | `per_segment_ragged_exact` prefill plus true-c1 decode; EOS and front/middle/tail sparse cancellation are exact. No throughput claim is attached to the fallback. |
| Current artifacts | [`G3 direct c2/c4/c8`](results/2026-07-18-gfx1151-paro-g3-native-c248-direct-retained.json), [`G5 F1`](results/2026-07-18-gfx1151-paro-g5-f1-server-scaling.json), [`G5 SSE`](results/2026-07-18-gfx1151-paro-g5-sse-server-scaling.json), [`P1 exact catalog`](results/2026-07-11-sol-p1-gfx1151-paro-c1-c8-exact-catalog.json), [`P2 ragged lifecycle`](results/2026-07-11-sol-p2-gfx1151-paro-ragged-lifecycle.json) |
| Historical correction | [`2026-07-10...current-diagnostic-summary.json`](results/2026-07-10-gfx1151-paro-cn-current-diagnostic-summary.json), [`true-c1 shrink gate`](results/2026-07-10-gfx1151-paro-true-c1-shrinking-gates.json) |

Reproduce c2-c8 with `scripts/qwen35_batch_equality_matrix.py --batch-sizes
2,3,4,5,6,7,8`; use `scripts/qwen35_paro_bench.py --prompt-fixture ...
--prompt-row 0` for the exact c1 control. Commands and raw SHA-256 values are
embedded in the compact artifact. Reproduce P2 with
`scripts/qwen35_batch_shrinking_correctness.py --batch-size 8
--prompt-lengths 449,458,467,476,485,494,503,512 --steps-per-width 1
--survivor-slot 4 --eos-slot 3` and the same model/fixture.

### gfx1151 PARO DFlash S4 profile, 2026-07-11

**Status: retained diagnostic profile; no performance claim.** Clean detached
hipEngine `8eb27215` ran the curated 35B W4 PARO/BF16-KV target and 35B BF16
DFlash drafter on the first `code_promotion` fixture, B4 and 32 output tokens.
The exact/default replay route matches all AR IDs and finite-logit gates, but it
is decisively slower:

| Route | AR tok/s | DFlash tok/s | DFlash/AR | Exact | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| Canonical replay, graph auto | 65.266 | 9.676 | 0.148x | yes | S4 profile accepted; speed rejected |
| Branch-copy commit | 65.269 | 14.450 | 0.221x | no, first mismatch 1 | S5 correctness rejection |
| Canonical replay + fused target LM-head | 65.223 | 9.177 | 0.141x | yes | S7 performance rejection (-5.16%) |

The exact row accepts 1/114 proposed draft tokens and spends 5.6875 target rows
per output. Coarse attribution is 74.62% target verify and 25.21% draft; the
profiling-only synchronized companion identifies target linear layers (37.41%
of total wall), drafter decoder+LM-head (25.55%), and canonical replay plus
scratch canonicalization (20.80%) as the largest buckets. Commit scatter is
0.25%, drafter top-k/readback 0.41%, and accept readback 0.04%.

Exact replay records 30 validated verifier-graph misses and zero hits across
two shapes. Branch-copy records 27 hits after two captures, but inherits the
known non-canonical c>N state and fails output equality. S6 is therefore parked:
wider verification would amplify rejected work, and this c1 row shows no
multi-request draft group-cap bottleneck. Compact evidence:
[`2026-07-11-sol-s4-gfx1151-paro-dflash-profile.json`](results/2026-07-11-sol-s4-gfx1151-paro-dflash-profile.json).

### gfx1151 GGUF server automatic-route gate, 2026-07-11

**Status: diagnostic correctness rejection; no performance claim.** The first
post-E1/E2/E3 server matrix runs the committed ten-prompt category JSONL and its
documented four-prompt heldout directly. It records exact choice IDs, canonical
model/suite provenance, owned batch timing, and realized queue/backend groups.
The source tree had no staged or unstaged changes; 255 unrelated untracked
benchmark files are disclosed, so this diagnostic is not a clean retained
performance row.

| Client c | Realized groups (full suite) | AR median tok/s | Compatibility MTP median tok/s | MTP/AR | Exact vs c1 AR |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | ten c1 groups | 35.92 | 39.35 | 1.095x | fail `general_ja_explain` |
| 2 | five c2 groups | 56.84 | 58.50 | 1.029x | fail 3/10 prompts |
| 3 | c3+c3+c3+c1 | 60.83 | 61.47 | 1.011x | fail 2/10 prompts |
| 4 | c4+c4+c2 | 69.70 | 66.13 | 0.949x | fail 3/10 prompts |
| 8 | c4+c4+c2 (route cap 4) | 69.84 | 65.87 | 0.943x | fail 3/10 prompts |

Full-suite values are medians of three after one discarded route/shape warmup;
heldout values use five repetitions. The mixed client-c3 aggregate hides the
reason realized groups matter: isolated full-suite c3 groups are +1.71% for
MTP, but isolated heldout c3 groups are **-3.92%**, so c3 does not activate.
More importantly, the current server hook is the documented
`llama-compat` direct-commit/dp4a route and is not serial-prefix-equivalent.
Its apparent c1/c2 speed benefit cannot enter automatic/default routing.

True AR c1-c4 is exact across every repetition. One of three client-c8 AR runs
changes `general_ja_explain` even though its actual backend groups are c4+c4+c2;
that remains a separate SOL-G8 exact-concurrency blocker, not evidence for a
width-8 backend. SOL-S1 now makes automatic MTP fall back to the default AR
route until an exact/default hook exists; explicit opt-in keeps the
compatibility contract. The compact artifact is
[`2026-07-11-sol-s1-gfx1151-server-auto-route-gate.json`](results/2026-07-11-sol-s1-gfx1151-server-auto-route-gate.json).

### Radeon 8060S direct and server GGUF concurrency, 2026-07-19

**Status: retained direct native-c2/c4/c8 model steps and c1-preserving
occupancy-adaptive OpenAI SSE serving.** F2 keeps stable scheduler, state, and
KV ownership while mapping ephemeral execution rows into exact c1/c2/c4/c8
buckets. F3 now replaces the general segmented recurrence with an exact
singleton-indexed GDN only for packed gfx1151 AR: direct c2/c4/c8 improve
**+8.71%/+5.25%/+4.04%**, while c1 is structurally unchanged. The prior F2
server packet remains retained but was not remeasured for this direct-only F3
refresh. All clean direct summaries repeat exactly; prior **189** server rows and
C2→C8/C4→C8 transition traces remain exact. The c8 marker census is **748
packed-native / 0 row-local / 0 copies**, and diagnostic Conv/GDN time falls
**8.230 -> 4.038 ms (-50.94%)**. The active normal-HWS boot uses `amd_iommu=off`
and one HIP hardware queue; no causal IOMMU result is inferred.

<!-- BEGIN TOPLINE:GFX1151_CONCURRENCY -->
| Direct route | Logical C | Native groups | Aggregate decode tok/s | Per-request tok/s | Aggregate / c1 | Aggregate / retained serial-c4† | TTFT p50 / p95 | Model-step ITL p50 / p95 | Tracked peak |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct c1 | 1 | 1x c1 | 50.335 | 50.335 | 1.000x | 1.002x | 0.370 / 0.372 s | 19.874 / 20.160 ms | 21.783 GiB |
| direct c2 | 2 | 1x c2 | 78.552 | 39.276 | 1.561x | 1.564x | 2.177 / 2.177 s | 25.459 / 25.820 ms | 22.394 GiB |
| direct c4 | 4 | 1x c4 | 108.050 | 27.013 | 2.147x | 2.151x | 3.394 / 3.403 s | 37.026 / 37.399 ms | 23.396 GiB |
| **direct c8** | **8** | **1x c8** | **133.251** | **16.656** | **2.647x** | **2.653x** | **6.841 / 6.841 s** | **60.004 / 60.641 ms** | **25.401 GiB** |
| chunked c8 control† | 8 | 2x c4, serialized | 102.724 | 12.841 | 2.043x | 2.045x | 5.089 / 6.787 s | 77.902 / 78.467 ms | 26.069 GiB* |
| serial-c4 rate control† | 4 | 4x c1, serialized | 50.235 | 12.559 | 0.999x | 1.000x | 0.927 / 1.485 s | 79.643 / 80.637 ms | 26.985 GiB* |

† Controls are retained from clean pre-F3 `ef46ee8c`. The serial-c4 c1 path is
structurally unchanged and remains the ratio reference; chunked c8 is historical
because the F3 candidate would also change each physical c4 group.

| Real OpenAI SSE route | Logical C | Physical execution | Aggregate generated tok/s | Per-request tok/s | Aggregate / logical-c1 | Aggregate / serial-c13 | Cycle wall p50 | Scheduler TTFT p50 / p95 | Scheduler ITL p50 / p95 | Cumulative tracked peak |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| occupancy-adaptive c1 | 1 | 1x native c1 | 43.033 | 43.033 | 1.000x | 0.999x | 2.974 s | 0.417 / 0.417 s | 19.759 / 20.057 ms | 29.046 GiB |
| physical c8 | 8 | 1x c8 | **86.942** | 10.868 | **2.020x** | 2.019x | 11.778 s | 2.614 / 3.347 s | 64.949 / 68.260 ms | 30.805 GiB* |
| grouped c9 | 9 | c8 + c1 | 77.302 | 8.589 | 1.796x | 1.795x | 14.903 s | 2.734 / 3.783 s | 85.747 / 91.143 ms | 31.035 GiB* |
| **grouped c13** | **13** | **c8 + sparse c8** | **73.235** | **5.633** | **1.702x** | **1.701x** | **22.721 s** | **3.624 / 5.526 s** | **132.547 / 141.787 ms** | **32.386 GiB*** |
| serial-c13 bridge | 13 | 13x c1 serial | 43.066 | 3.313 | 1.001x | 1.000x | 38.638 s | 3.386 / 5.390 s | 257.799 / 271.455 ms | 32.386 GiB* |
<!-- END TOPLINE:GFX1151_CONCURRENCY -->

Direct protocol uses 128 decode transitions, one discarded warmup, and the
median of three. F3 direct c1/c2/c4/c8 is
**50.335/78.552/108.050/133.251 aggregate tok/s** with maximum rate
stdev/median **0.096%**; one physical c8 is **2.647x** c1 and **+23.32%** over
the current direct-c4 rate. The required c8 trace is **748 packed-native / 0
row-local / 0-copy**. Server protocol remains the clean F2 packet: 512 exact
prompt IDs and 128 generated outputs/request, a 20 ms admission window, one
discarded plus three measured bursts, and scheduler latency. Logical c1 executes
physical c1, while C9 uses c8+c1 and C13 remains two declared groups, never a
wider native claim. All **189/189** server requests match resident prompt IDs,
direct-c1 outputs, usage, and finish metadata; maximum static stdev/median is
**0.524%**. Grouped C13 is **1.702x** logical-c1 and **1.701x** serial
(**+70.05%**); one exact c8→c13 live trace emits **1,664/1,664** IDs at
**71.891 aggregate tok/s** and drains ownership to zero. Clean short C2→C8 and
C4→C8 traces emit **256/256** IDs each at **84.210/84.250 tok/s** and drain
ownership. Stable scheduler slots, session state, and KV never move; only
execution rows are dense. Starred server memory is cumulative in one prepared
process.

Artifacts: [F3 singleton-indexed GDN](results/2026-07-19-gfx1151-gguf-f3-singleton-gdn-retained.json),
[F2 occupancy-adaptive serving](results/2026-07-19-gfx1151-gguf-f2-occupancy-adaptive-serving.json),
[E1 direct correctness](results/2026-07-17-gfx1151-gguf-concurrency-e1-direct-correctness.json),
[retained E1 direct scaling](results/2026-07-17-gfx1151-gguf-concurrency-e1-native-c8-scaling-closure.json),
[E1 live-loop closure](results/2026-07-17-gfx1151-gguf-concurrency-e1-live-loop-closure.json),
[E3 arbitrary-C correctness](results/2026-07-18-gfx1151-gguf-concurrency-e3-arbitrary-c-correctness.json),
[F1 real server scaling](results/2026-07-18-gfx1151-gguf-concurrency-f1-server-scaling-closure.json),
and [current-main F0 recertification](results/2026-07-19-gfx1151-gguf-f0-current-main-recertification.json).

### gfx1151 historical cross-engine concurrency, 2026-06-15

**Status: stale diagnostic.** hipEngine uses PARO W4/BF16 KV; llama.cpp uses
Vulkan Q4_K_S/f16 KV. vLLM did not produce a healthy server. The summary lacks
the measured hipEngine commit, and the then-used per-run device properties could
report gfx1100 even though the run forced `HIPENGINE_HIP_ARCH=gfx1151`.

No eligible historical concurrency row; the `performance_claim=false` snapshot
remains linked below.

Protocol: prompt 512, decode 128, 8 warmup decode tokens, median of 3. Primitive
c>1 attention/KV checks passed. The generated-token field used the older
batch-shaped reference and is not independent-c1 evidence. Profiler, scaling,
and provenance gates also did not pass.

Artifacts: [combined summary](results/2026-06-15-gfx1151-readme-concurrency-20260615-213804-summary.json),
[hipEngine](results/2026-06-15-gfx1151-readme-concurrency-20260615-122207-hipengine-paro/summary.json),
[llama.cpp Vulkan](results/2026-06-15-gfx1151-readme-concurrency-20260615-213804-llamacpp-vulkan/summary.json), and
[vLLM blocker](results/2026-06-15-gfx1151-readme-concurrency-20260615-122207-vllm-gptq-int4-blocked.json).

## README Sweep Test Procedure

### W7900 model and concurrency refresh

Use a clean detached worktree. The wrapper fixes the GPU mapping, TheRock
environment, model paths, llama.cpp binaries, JIT cache policy, and output
layout.

```bash
RUN_TAG=$(date -u +%Y%m%d-%H%M%S)
WORKTREE="/tmp/hipengine-readme-w7900-${RUN_TAG}"
git worktree add --detach "$WORKTREE" HEAD

OUTDIR="$PWD/benchmarks/results" \
RUN_TAG="$RUN_TAG" \
REPO_ROOT="$WORKTREE" \
  "$WORKTREE/scripts/run_w7900_readme_refresh.sh" all
```

Subset commands:

```bash
scripts/run_w7900_readme_refresh.sh hipengine
scripts/run_w7900_readme_refresh.sh llamacpp
scripts/run_w7900_readme_refresh.sh concurrency
scripts/run_w7900_readme_refresh.sh vllm
```

Required W7900 settings:

| Surface | Settings |
| --- | --- |
| Device mapping | `HIP_VISIBLE_DEVICES=0`; W7900 is amdgpu `card1`; llama.cpp uses `ROCm0` and `Vulkan0` after masking |
| hipEngine environment | `/home/lhl/mambaforge/envs/therock/bin/python3.12`; hermetic TheRock root from `python -m rocm_sdk path --root`; `HSA_OVERRIDE_GFX_VERSION=11.0.0` |
| Model sweep | `512/128 1K/128 4K/128 32K/128 64K/128 128K/128`; 2 warmups; 5 measured; resident max-context session |
| PARO | snapshot `437eba06df05aad71a4dacdcaf3fff70ae1ee8a1`; `hip_gfx1100`; `packed_paro_w4`; BF16 KV; AOTriton threshold 512; graph replay decode |
| hipEngine GGUF | MTP-bearing Qwen3.6-35B-A3B `UD-Q4_K_M`; decode repack; WMMA bulk prefill; GEMV eager decode; BF16 KV |
| llama.cpp | Same GGUF; `-ngl 99 -fa 1 -ctk f16 -ctv f16`; split prefill/decode; one repetition per phase |
| Concurrency | prompt 512; decode 128; warmup 8; c=1,2,4,8; 3 repetitions; fixed token-id fixture |

Never add a combined summary with `performance_claim=false` to the current
topline table. Keep its artifact linked in the diagnostic section instead. A
retained refresh also needs the correctness and repetition gates from
[`docs/BENCHMARK.md`](../docs/BENCHMARK.md).

### gfx1151 model and concurrency refresh

The committed
[`run_gfx1151_readme_refresh.sh`](../scripts/run_gfx1151_readme_refresh.sh)
replaces the unreproducible 2026-06-15 `/tmp/run_gfx1151_readme_udq4km.sh`.
Run it from a clean detached worktree so component provenance observes no
tracked or untracked source changes:

```bash
RUN_TAG=$(date -u +%Y%m%d-%H%M%S)
WORKTREE="/tmp/hipengine-readme-gfx1151-${RUN_TAG}"
git worktree add --detach "$WORKTREE" HEAD

OUTDIR="$PWD/benchmarks/results" \
RUN_TAG="$RUN_TAG" \
REPO_ROOT="$WORKTREE" \
  "$WORKTREE/scripts/run_gfx1151_readme_refresh.sh" all
```

Subset commands are `... hipengine`, `... llamacpp`, and `... summary`. The runner fixes the
model identities, six standard shapes, native gfx1151 compiler target,
torch-free hermetic TheRock environment, PARO's two discarded plus five
measured runs, GGUF's calibrated one discarded plus three measured runs, and
five internal llama-bench repetitions. It records a
canonical provenance object in every component artifact. Each hipEngine shape
runs in its own process with a right-sized resident session, then the committed
merge gate verifies and preserves all samples in one compact rollup. This keeps
512/1K memory honest and avoids imposing a 128K allocation on every row.
Discarded runs warm the same kernels through eager submission; each measured
run captures and destroys a fresh state-bound graph after reset/prefill/warmup,
so no captured graph crosses a session reset. The summary phase verifies all
four component artifacts together and generates the Markdown tables only when
their provenance, model/build identity, correctness, return-code, variance,
and memory-scope gates pass.

gfx1151 is a UMA APU: sysfs reports only a 512 MiB visible-VRAM aperture while
the amdgpu GTT domain is 120 GiB and holds model allocations. The runner
therefore samples `mem_info_gtt_used` for llama.cpp HIP/Vulkan. The public
memory table must label that whole-device GTT scope and separately identify
hipEngine tracked or HIP phase-sampled peaks; it must not relabel the 512 MiB
aperture as total model memory.

Before updating the gfx1151 tables:

1. Detect and record `gfx1151` from the runtime/build output; do not fill the
   artifact from a CLI label alone.
2. Run PARO with 2 discarded warmups and 5 measured repetitions. Run GGUF with
   1 discarded warmup and 3 measured repetitions. Escalate GGUF to 5 only when
   a named variance, stability, or borderline-decision trigger fires; test
   lifecycle soak separately.
3. Run PARO concurrency for c=1 through c=8, including odd widths and dynamic
   c=8 to c=1 shrinking, with exact all-choice generated-token counts.
4. Keep comparison engines in separate columns when quant or timing scope
   differs. Bold may mark the raw row leader, but the nearby text must state
   that a cross-quant or cross-memory-scope maximum is descriptive rather than
   a controlled backend win.

The clean P1/P2 artifacts now satisfy the current c1-c8 independent-c1 and
ragged shrinking lifecycle gates. They retain c1 timing and classify c2-c8 as
exact width-1 production groups because every native candidate is
correctness-red. A future cross-engine concurrency-speed table still requires
one matched quant/timing protocol; do not republish the superseded 2026-06-15
native numbers as production throughput.

The lower-level hipEngine sweep command is:

```bash
PYTHONPATH=. \
HIPENGINE_HIP_ARCH=gfx1151 \
python3 scripts/qwen35_readme_sweep.py \
  --engine paro \
  --model /home/lhl/.cache/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-packed/snapshots/437eba06df05aad71a4dacdcaf3fff70ae1ee8a1 \
  --backend hip_gfx1151 \
  --shared-expert-format packed_paro_w4 \
  --token-id 9707 \
  --workloads 512/128 1K/128 4K/128 32K/128 64K/128 128K/128 \
  --warmup-runs 2 --measured-runs 5 --warmup-decode-tokens 4 \
  --attn-aotriton-min-tokens 512 --graph-replay-decode \
  --compiler-version-file /tmp/hipengine-hipcc-version-gfx1151.txt \
  --require-cached-build \
  --json benchmarks/results/<date>-gfx1151-hipengine-paro-readme-sweep.json
```

This lower-level command is not a complete refresh: use the committed wrapper
for GGUF, llama.cpp, environment capture, and artifact assembly. Concurrency
remains a separate gate because production c2-c8 is currently exact width-1
fallback, not the rejected native timing path.

### Speculative decode refresh

Exact/default GGUF MTP, fixed 10-cycle suite:

```bash
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 \
python3 scripts/gguf_ar_mtp_suite.py \
  --scope full \
  --mtp-route resident-b1-probe-block-direct-cap32k-minrows2-pmin05 \
  --record-cycle-stage-timings \
  --require-cached-build \
  --output benchmarks/results/<date>-ar-mtp-exact-full.json
```

`llama-compat` natural24 direct contract:

```bash
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 \
python3 scripts/gguf_ar_mtp_suite.py \
  --scope full \
  --mtp-route llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm-routerrow-draftdenseq8-draftonly-directcommit \
  --budgets 2 --cycles 24 --max-output-tokens 24 \
  --record-cycle-stage-timings \
  --require-cached-build \
  --output benchmarks/results/<date>-ar-mtp-llama-compat-natural24.json
```

llama.cpp B2 with 24 transition-matched decode steps. Request 25 outputs
because the first is sampled before llama.cpp starts `predicted_ms`:

```bash
python3 scripts/llamacpp_mtp_bench.py \
  --server-bin /path/to/llama-server \
  --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --ctx-size 8192 --concurrency 1 --gpu-layers 99 \
  --flash-attn on --cache-type-k f16 --cache-type-v f16 \
  --draft-max 2 --mode both --protocol natural \
  --prompts benchmarks/prompts/mtpbench-code-general-ja.jsonl \
  --max-tokens 25 --seed 12345 --temperature 0 \
  --top-k 1 --top-p 1 --min-p 0 \
  --server-extra-arg=--reasoning --server-extra-arg=off \
  --output benchmarks/results/<date>-llamacpp-natural25.json
```

Use `aggregate_decode_transition_per_second` for the cross-engine column;
retain `aggregate_decode_predicted_per_second` only as llama.cpp's native
self-report. See the [timing contract](#cross-engine-gguf-decode-timing-contract).

Dense DFlash B=4:

```bash
python3 scripts/dflash_chain_e2e_bench.py \
  --target-model /home/lhl/.cache/huggingface/hub/models--z-lab--Qwen3.6-27B-PARO/snapshots/84f86409151d4f2ec86dc0b6a096d5f6daa7f207 \
  --drafter-model /home/lhl/.cache/huggingface/hub/models--z-lab--Qwen3.6-27B-DFlash/snapshots/0919688658996800f86b895034249700e9481106 \
  --backend hip_gfx1100 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --max-prompts 9 --decode-tokens 64 --draft-budgets 4 \
  --draft-top-k 2 --whole-cycle-gate 0.90 \
  --verifier-mode native_bulk_bplus1 --verifier-graph auto \
  --full-attn-chain-mode batched --canonical-commit-mode branch_copy \
  --adaptive-budget off --hardware-gpu "AMD Radeon Pro W7900" \
  --json benchmarks/results/<date>-dflash-27b-b4.json
```

Use equivalent immutable local snapshots only when the recorded fingerprints
match these target and drafter revisions.

### HIP versus Vulkan microbenchmarks

Microbenchmark claims do not belong in the model-throughput tables. The v2
timing contract and exact bounded rerun commands are in
[`docs/HIP-vs-VULKAN.md`](../docs/HIP-vs-VULKAN.md) and
[`benchmarks/micro/README.md`](micro/README.md). Retained evidence is
[`gfx1100/W7900`](micro/results/gfx1100/w7900/2026-07-11-hip-vulkan-timing-v2-bounded.json)
and
[`gfx1151/Strix Halo`](results/2026-07-12-gfx1151-hip-vulkan-portable-q8.json).
The original stricter Q4/Q6 correctness misses are isolated in
[`2026-07-12-gfx1151-vulkan-q8-isolation-diagnostic.json`](results/2026-07-12-gfx1151-vulkan-q8-isolation-diagnostic.json): both Vulkan dot kernels pass
when given CPU q8_1 blocks, while stock packed-FP16 activation scales are
systematically one code below the CPU/HIP oracle. The retained portable shader
eliminates those scale mismatches; both the gfx1100-matched and current strict
gfx1151 matrices now pass 22/22 comparisons and all 232 burst rows.

The retained synthetic Laguna-shape IQ2_XS primitive packet is
[`2026-07-22-gpu1-iq2-xs-laguna-primitives.json`](results/2026-07-22-gpu1-iq2-xs-laguna-primitives.json).
At K=3072/N=128/E=16, rowbatch4 reduces event time by 7.24-58.82% versus
row-at-a-time grouping across 1-16 rows/expert and remains BF16-bit exact;
K=3072/N=1024 also passes exact selected/grouped gates. This is a kernel
schedule result, not Laguna model throughput or quality evidence.

The old/current narrow cross-format diagnostic is
[`2026-07-23-gpu1-iq-cross-format-prefill-current.json`](results/2026-07-23-gpu1-iq-cross-format-prefill-current.json).
At E16/K3072/N128 with equal 1/2/4/8/16 rows per expert, the fastest explicit
exact scalar/rowbatch IQ2 leaf moves
`20.452/22.584/25.977/48.428/97.458 -> 15.733/19.191/23.178/42.657/86.580 us`
(-10.78% to -23.07%) from pre-optimization source `8addd867` to the current
kernel tree. IQ3/IQ4 move by at most 2.22% between the two sessions, consistent
with run variance and no relevant kernel change. These are complete dual gate/up
dispatch latencies, not per-row or runtime-default timings, and remain a narrow
synthetic diagnostic rather than model throughput evidence.

The counterbalanced E256/K3072/N1024 tuning baseline is
[`2026-07-22-gpu1-iq2-xs-laguna-tuning-baseline.json`](results/2026-07-22-gpu1-iq2-xs-laguna-tuning-baseline.json).
Rotating-distinct top-10 selected single/fused-dual leaves are
`57.881/106.136 us`; repeated-expert cache-hot leaves are 20-22% faster and are
therefore diagnostic only. Rowbatch4 regresses balanced 16-token prefill by
6.30%, wins 8.35-29.92% at 32 tokens depending on routing skew, and wins
30.71-57.46% at 64-512 tokens. Full-shape fused decode and grouped prefill gates
are BF16-bit exact. These are synthetic primitive/policy results, not Laguna
model throughput or quality evidence.

The retained exact branchless IQ2 decoder is
[`2026-07-22-gpu1-iq2-xs-branchless-decode.json`](results/2026-07-22-gpu1-iq2-xs-branchless-decode.json).
Replacing divergent selector ternaries with the exact arithmetic map and sign-bit
OR moves rotating-distinct selected single/fused dual
`57.881/106.136 -> 49.200/78.784 us` (-15.00/-25.77%). All hot/repeated decode
controls improve 14.14-25.14%; all representative E256 prefill cases improve,
with scalar -21.21% to -32.26% and rowbatch4 -13.90% to -27.21%. Fused decode
and grouped prefill remain BF16-bit exact. The four kernels stay scratch-free;
their increased VGPR40/64/80/88 allocation is accepted because every measured
routing/size case wins. This remains primitive, not model-level, evidence.

The retained decode geometry extension is
[`2026-07-22-gpu1-iq2-xs-pair16-local64.json`](results/2026-07-22-gpu1-iq2-xs-pair16-local64.json).
Pairing adjacent shared-scale selectors and selecting local64 moves the
branchless-local256 rotating single/dual `49.200/78.784 -> 33.296/56.922 us`
(-32.33/-27.75%); hot/repeated controls improve 16.40-28.10%. The full E256
geometry sweep is BF16-bit exact to local256. Task32 regressed every matched
pair16 leaf by 10.46-31.20% and was removed. Pair16 grouped prefill was also
restored to group8 because short/sparse rows regressed up to 5.25%, despite
large populated-expert scalar wins. The retained decode leaves are local64,
VGPR64/96, LDS512B, and scratch0. This is synthetic primitive evidence.

The retained sparse-prefill policy is
[`2026-07-22-gpu1-iq2-xs-adaptive-rowbatch.json`](results/2026-07-22-gpu1-iq2-xs-adaptive-rowbatch.json).
For K3072 below four compact rows/expert on average, each block selects batch1,
batch2, or batch4 from its device-resident expert count; denser calls preserve
the original rowbatch4 symbol. Versus unconditional rowbatch4, all nine
16/32/64-token balanced/hot/Zipf leaves improve by 0.64-13.09%, including
balanced 16 `1.378 -> 1.198 ms` (-13.09%) and balanced 32
`2.179 -> 1.919 ms` (-11.91%). Outputs are BF16-bit exact; adaptive is
local256/VGPR88/LDS512B/scratch0. Standalone rowbatch2 never won. Rowbatch8 won
only the balanced five-row case (-4.15%) and regressed the other 14 leaves by
12.25-96.50%, so it was removed. This remains synthetic primitive evidence.

The retained selected-decode tile is
[`2026-07-22-gpu1-iq2-xs-output-tile2.json`](results/2026-07-22-gpu1-iq2-xs-output-tile2.json).
Sharing BF16 activation loads/conversions across two adjacent output columns
moves tile1 -> tile2 rotating selected single/dual
`33.569/57.176 -> 30.955/55.964 us` (-7.79/-2.12%); hot/repeated controls improve
4.04-8.82%, and an independent full-protocol repeat also wins all six leaves.
The full E256 output is BF16-bit exact. Tile2 is now default, while explicit
tile1 four-axis variants remain for rollback. Tile2 is local64/LDS512B/scratch0
at VGPR80/136. This remains synthetic primitive evidence.

The rejected Q8_1/`sudot4` decode experiment is
[`2026-07-23-gpu1-iq2-xs-q8-1-dp4a-rejected.json`](results/2026-07-23-gpu1-iq2-xs-q8-1-dp4a-rejected.json).
Llama.cpp-shaped packed-byte expansion plus tile2 made prequantized fused decode
1.47-4.83% faster, emitted `v_dot4_i32_iu8`, stayed scratch-free, and passed the
primitive quality gate (projection/fused KL mean `0.000330/0.006713`, top-1
`1.0`). Its 3.32-3.41 us activation quantizer nevertheless moves retained exact
-> inclusive fused decode `54.299 -> 55.532 us` (+2.27%) for rotating-distinct,
`51.338 -> 50.760 us` (-1.13%) for hot, and `47.461 -> 48.468 us` (+2.12%) for
repeated experts. The representative cold/repeated regressions reject the lane;
candidate code was removed and no Q8_1 sidecar or fusion is retained.

The retained explicit integer-prefill primitive is
[`2026-07-23-gpu1-iq2-xs-mmq32-prefill.json`](results/2026-07-23-gpu1-iq2-xs-mmq32-prefill.json).
At E256/K3072/N1024/top-10, raw IQ2 signed-byte fragments are expanded once per
32-column x K256 tile into LDS and reused across four 16x16 RDNA3 integer-WMMA
minitiles. Exact auto -> D4-quantizer-inclusive MMQ32 moves the 256-token
balanced/hot/Zipf cases `7.755/8.201/7.647 -> 5.528/5.842/5.927 ms`
(-28.72/-28.76/-22.49%) and the 512-token cases
`13.740/14.410/14.377 -> 6.889/7.726/7.902 ms`
(-49.86/-46.38/-45.03%). The populated-expert fixture passes max-relative
`<=0.05`; representative E256 quality has KL max `<=0.00453`, top-1
`>=0.98125`, and finite outputs. Rocprof records local128/VGPR104/LDS10240B and
scratch0; the D4 quantizer is local256/VGPR24/scratch0. Short padding remains a
hard blocker: 16-64 tokens regress 45.92-129.45%, and 128-token hot/Zipf regress
10.41-19.97%. The four-axis primitive and optional benchmark route are retained,
but runtime default promotion waits on Laguna all-layer quality and ownership of
Q8/tile scratch; exact adaptive/rowbatch remains unchanged. This is synthetic
primitive evidence, not Laguna model throughput or quality evidence.

## Update Checklist

1. Choose one protocol tuple and record the old artifact before running.
2. Create a clean detached worktree at the revision being measured.
3. Capture the canonical provenance block: GPU identity, configured/resolved
   backend, target arch, VBIOS, power/clock state, kernel, Python, ROCm/HIP
   compiler, Vulkan driver, comparison-engine commit, existing model
   fingerprint, exact argv/environment, and separate staged, unstaged, and
   untracked source state.
4. Run the named warmup, repetition, correctness, and memory protocol. Store raw
   logs outside git and a compact artifact under `benchmarks/results/`.
5. Reject artifacts with missing provenance or failed correctness. A diagnostic
   may be recorded, but it cannot replace a retained row.
6. Update the platform index, table, run record, artifact links, run date, and
   measured revision in this file.
7. Add the required entry to [`benchmarks/CHANGELOG.md`](CHANGELOG.md) and append
   the commands and decision to `WORKLOG.md`.
8. Run the root README sync and validation commands:

```bash
python3 scripts/sync_benchmark_readme.py --write
python3 scripts/sync_benchmark_readme.py --check
python3 -m json.tool benchmarks/results/<new-artifact>.json >/dev/null
git diff --check
```

Run `json.tool` once for each new or changed compact artifact. Do not scan
untracked experiment files as part of the rollup gate.

<a id="natural24-mtp-vs-ar-concurrency-diagnostic"></a>
<a id="blocked--diagnostic-benchmark-attempts"></a>

## Blocked and Diagnostic Benchmark Attempts

- **W7900 coding-agent A5 pressure/soak closure, GGUF Q4_K_M:** clean pushed
  `414d6d9e` runs the unchanged cache-off/native-sampler-off package defaults
  through all nine real-Uvicorn pressure workloads. The packet handles **122
  requests** as **108 exact completions / 12 exact retryable rejects / one
  two-ID disconnect / one deadline**, with **2,482 exact observed IDs**. The
  80-second soak is **40/40 exact at 11.151 SLO-goodput tok/s**; overload is
  **20 accepts / 12 rejects at 21.717 tok/s**. Queue depth reaches its declared
  16 cap, the slow-consumer stream queue peaks at 1/16, KV grows 3 -> 12 pages
  and drains, graph/workspace/memory recover, final ownership is zero, and 41
  KFD samples see no competing GPU0 process. Cache eviction is linked to the
  passing A2 p2048/p8192 lifecycle closure because cache off remains the
  performance-selected default. This is bounded correctness and absolute SLO
  evidence, not a tuning comparison or multi-day reliability claim. [A5
  artifact](results/2026-07-22-w7900-agentic-a5-pressure-soak-closure.json).
- **W7900 coding-agent A4 routing/SLO decision, GGUF Q4_K_M:** clean pushed
  `fb744f03` runs all **8 predeclared candidates x 3 balanced repetitions** over
  12 delayed mixed-shape requests each (**288 requests / 8,640 response-owned
  IDs**). No candidate passes all correctness and SLO gates. The zero-window
  package control stays exact but misses TTFT p95 once at **10.983 s > 10 s**;
  every apparently faster alternative changes the late `fixed-0011` p512/d48
  trajectory in at least one pass after 20-24 correct `9710` IDs (**9 mismatched
  rows** total). All requests reclaim, native routing and final ownership pass,
  but diagnostic median goodput gains up to **+63.81%** are invalid for
  promotion. C1/C2/C4/C8, strict-tool, and safety timing is skipped without
  inference; retain `protect_decode:256/burst-1` and the zero-ms package window.
  [Final A4
  decision](results/2026-07-22-w7900-agentic-a4-routing-decision.json).
- **W7900 coding-agent A3 native-sampler screen, GGUF Q4_K_M:** clean pushed
  `2f8f6bf1` stops at the real-Uvicorn C1 `small_repo` blocking-oracle
  prerequisite before any measured SSE. Fixed-seed host/native auto-tool rows
  repeat the valid first turn, but both reach the 64-token cap on turn 1 with
  `invalid_tool_call`; native correctly reports `gpu_sample` and zero logits D2H
  while host copies **63,569,920 bytes**. Two repeats of all **4/4** specific
  strict-tool turns are exact and valid, but every row is
  `host_logits_sample` / `native_gpu_unsupported_request`, totaling
  **198,656,000 bytes** of D2H across 200 tokens. Final ownership is zero. No
  route is both native-eligible and valid across the frozen tool workload, so no
  C1/C4/C8 timing, active-SSE, tool-ready, or speedup number is retained or
  inferred; keep GGUF native sampling default-off. [Blocked A3
  artifact](results/2026-07-22-w7900-agentic-a3-native-sampler-blocked.json).
- **W7900 coding-agent A2.4 final prefix decision, GGUF Q4_K_M:** the complete
  active-SSE-scoped funnel combines the A1 control, three-pair C1 rejection,
  protocol-complete C4/C8 skip, and passing lifecycle packet. Radix regresses
  both primary metrics for every family, fails every A1 C1 guard and two variance
  gates, and leaves medium-C4 unsatisfied because C1 never authorized wider
  timing. Keep cache off; radix remains explicit diagnostic-only.
  [A2.4 decision artifact](results/2026-07-21-w7900-agentic-a2-prefix-decision.json).
- **W7900 coding-agent A2.3 prefix lifecycle/pressure closure, GGUF Q4_K_M:**
  four real p2048/p8192 active-current/completed-source gates preserve exact
  response IDs and Conv/GDN/live-KV state (`KL=0`, top-1 100%), bound
  current/high-water pool bytes, and drain every ref after explicit eviction.
  Real-agentic final cache residency is bounded at **124,518,400 / 129,761,280 /
  255,590,400 bytes** for small/growing/medium with zero non-cache final owners;
  targeted LRU, COW/refcount/pin, cancellation, disconnect, slow-consumer,
  deadline, and admission rollback gates pass. Resident state fork/rollback is
  explicitly unsupported while app-local transcript copies remain isolated.
  This correctness closure does not promote the C1-regressive route.
  [A2.3 lifecycle artifact](results/2026-07-21-w7900-agentic-a2-prefix-lifecycle-closure.json).
- **W7900 coding-agent A2.1 C1 prefix decision, GGUF Q4_K_M:** clean pushed
  `fb9531b2` completed all **3 balanced off/radix pairs x 3 frozen families**
  after deterministic `processed_argmax` became eligible. All response IDs,
  tools, bounded cache ownership, and GPU0 exclusivity gates pass, but reuse is
  sparse: small has **0/12 hits**, growing **3/24**, and medium **3/18**. Radix
  versus paired off regresses median active-SSE goodput by **64.19% / 65.63% /
  26.64%** and raises buffered tool-ready p50 by **181.90% / 196.09% / 38.81%**
  for small/growing/medium. Candidate medians are only **4.727 / 4.216 / 2.838
  tok/s** with **5.247 / 6.097 / 6.632 s** tool-ready, all materially below A1;
  primary variance also fails on growing/medium. Keep radix default-off and do
  not run the gated C4/C8 promotion matrix. [Rejected complete A2.1 C1
  artifact](results/2026-07-21-w7900-agentic-a2-c1-prefix-rejected.json).
- **W7900 coding-agent A2 C1 prefix screen, GGUF Q4_K_M:** clean pushed
  `3a4024af` completed one warmed `small_repo` off/radix pair with all **4/4 +
  4/4 response-owned ID/tool/final-request-ownership gates exact**, but every
  strict-tool request resolved `sampler_mode=processed_argmax`; the scoped radix
  path explicitly returned `sampling_unsupported` on all four turns, producing
  **0 eligible turns / 0 lookups / 0 hits / 0 reused tokens**. The remaining
  pairs/workloads were stopped and the observed **13.268 vs 13.297 tok/s** and
  **1855.2 vs 1871.5 ms** tool-ready p50 are diagnostic-only, not an A/B result.
  [Blocked A2 artifact](results/2026-07-21-w7900-agentic-a2-c1-processed-argmax-blocked.json).
- **W7900 coding-agent A6 broad automatic-tool quality, GGUF Q4_K_M:** clean
  `878d07a9` completes two repeats of **24 externally scored turns** over
  repository, general-English, Japanese, and mixed Japanese/English families.
  Complete success is **10/48**; valid-call/correct-tool is **18/48**, exact
  arguments/external-oracle pass are **16/48**, safe patch success is **0/6**,
  and independent test success is **8/8**. Outcomes are **10 passed / 20
  invalid-tool-call / 10 no-tool-call / 6 content-alongside-tool-call / 2
  wrong-arguments**. All 24 repeat pairs match response IDs/outcomes, all 4,538
  IDs are response-owned, no raw markup leaks, GPU0 exclusivity and zero final
  ownership pass, and no timing fields are present. This synthetic packet
  supersedes the old four-turn A6 diagnostic but is not a public quality
  benchmark or performance row. [Broad A6 artifact](results/2026-07-22-w7900-agentic-a6-broad-quality.json).
- **W7900 coding-agent A1 capacity diagnostics, GGUF Q4_K_M:** clean `56c91f87`
  closes logical-C8 capacity for all frozen families at guarded
  **4K/4K/10,240** contexts and registry-capped c4 residency. Those one-run
  artifacts remain correctness/capacity-only and are superseded for performance
  by the retained repeated A1 row above. [C8 capacity matrix](results/2026-07-20-w7900-agentic-a1-c8-capacity-matrix.json),
  [repeated A1 baseline](results/2026-07-21-w7900-agentic-a1-repeated-baseline.json).
- **W7900 GGUF Q4_K_M:** the [2026-07-07 summary](results/2026-07-07-w7900-gpu0-readme-refresh-20260707-104756-summary.json) is the last measured path and
  has `performance_claim=false`. Repetition of token `9707` is confirmed as
  valid for the exact model by llama.cpp and the gfx1151 G1 oracle; the W7900
  measurement still needs its own current state/KV gate and repeated clean
  performance rerun before it can become a baseline.
- **OpenAI MTP server c=1/2/3/4/8:** the corrected
  [2026-07-11 route gate](results/2026-07-11-sol-s1-gfx1151-server-auto-route-gate.json)
  supersedes the pre-contract 2026-07-06/07 timing rows. It counts exact IDs,
  owns batch timing once, records canonical provenance, and separates client,
  queue, backend, and verifier widths. Compatibility MTP is diagnostically
  faster at c1/c2 but changes true-AR IDs, so no automatic-route performance
  claim is eligible; explicit opt-in remains separately labelled.
- **gfx1151 PARO c3-c8 and production attachment:** the retained G2 packet now
  accepts explicit direct selected-batch c2, superseding P1's old c2 candidate.
  P1 still rejects c3-c8, and P2 accepts the public true-c1 bridge through EOS
  and front/middle/tail sparse slots. No wider native width or public c2 route is
  eligible until it passes the same independent-c1 token/state/KV and shared-
  loop gates. The 2026-07-10 native timing artifact remains diagnostic.
- **gfx1151 model sweep:** the [committed summary](results/2026-06-15-gfx1151-readme-udq4km-20260615-040438-summary.json) omits source/build provenance
  and contains one measured repetition. Its values remain a dated diagnostic.
- **llama.cpp 24 GiB Q8_0 memory:** the former root README tables had no compact
  artifact, model fingerprint, llama.cpp revision, or run date. The numbers were
  removed; rerun before publishing another capacity table.

Rejected and superseded rows remain in JSON artifacts, `WORKLOG.md`,
[`benchmarks/CHANGELOG.md`](CHANGELOG.md), and
[`benchmarks/HISTORY.md`](HISTORY.md). Source-lineage targets and external
baselines in the archive are reference values, not hipEngine toplines.

## Table Conventions

- Workload format is `prompt_tokens/decode_tokens`.
- `tok/s` is reported separately for prefill, backend decode, and full request
  wall. Never compare those scopes without labeling them.
- Aggregate concurrency throughput is total generated tokens divided by the
  concurrent group wall. Per-sequence throughput is aggregate divided by live
  rows only when every row generates the same number of tokens.
- `Peak GiB` names the allocator or whole-card scope in the run record.
- Bold ratios in retained speculative rows identify speedup against the true
  same-protocol AR control. Plain maxima in diagnostic cross-engine tables are
  not promoted as wins.
