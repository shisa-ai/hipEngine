# Qwen3.8-27B GGUF gfx1151 Optimization Campaign

Status: **Campaign closed 2026-08-17 at merged commit `20e5106da`. G1
single-layout ownership, G2 prefill, G3 true AR, G4 exact MTP, and G5 memory
all pass on the one clean retained source. Clean closure rows are
379.398/376.554/369.755 prefill and 13.1014/12.9231/13.0953 AR tok/s at
512/1K/4K (above both frozen llama backends at every shape), exact native B3
at 23.85263 tok/s (1.7845x own AR, 21.48% above correctness-valid llama HIP
B3), and process GTT 15.275/15.710/15.727 GiB at the shapes and 15.899 GiB at
B3 (below every lower valid llama row). K0-K3 native INT8 K/V is closed below
K4 on model-level correctness; BF16 remains the supported route.** Post-closure
(2026-08-17) retention extends the byte-identical K_M-derived Q5 source-F16
prefill route to Q4_K_S: counterbalanced 512/1K prefill improves to
**396.091/387.648 tok/s** (+4.51%/+3.02%) and 4K to **380.305 tok/s** (+2.95%)
via a capacity-conditional scratch cap (4K grows to the 4,096-row plateau;
8K+ keeps 1,024-row chunks with memory flat). The working
performance set is `512/128`, `1024/128`, and `4096/128`. The current model
representation is Qwen3.8-27B Q4_K_S on Radeon 8060S / `gfx1151`.

The immediate objective is to beat current clean llama.cpp HIP and Vulkan at
all three working shapes for prompt processing, true autoregressive decode, and
valid exact MTP, while minimizing resident and whole-process memory. The
separate native INT8 K/V lane found no representation that transfers through
1K/8 under the quality contract; BF16 therefore remains the supported route. A
post-closure research agenda for additional INT8 / dtype directions (gfx1151
INT8 K/V revalidation, fp16 recurrent state, lm_head/embed_tokens int8, cheap
MTP drafts, int8 activations for prefill/concurrency, KVarN 4/2-bit K/V) is
recorded in Section 12; nothing there is retained until a full
production-profile gate passes.

The clean retained P4 publication is
`399.031/391.276/385.330` prefill tok/s at 512/1K/4K. It beats clean llama HIP
by **13.224%/7.363%/4.711%** and Vulkan by
**64.240%/58.021%/8.737%**, closing G2. A same-source three-run standard-Q6
control is `362.752/354.270/349.130`, so the causal latest-owner A/B remains
**+9.935%/+10.611%/+10.168%** with identical IDs, tracked peaks, and teardown.
Clean process GTT is **17.322/17.805/20.181 GiB**, down
**47.705%/47.030%/44.012%** from the opening snapshot but still
**9.741%/12.575%/26.104%** above the lower llama comparator, so G5 remains
open.

Bounded Q5 source-F16, rows>=512 four-wave shared standard-Q6 QKV, and shared
planar-Q6 FFN-down are retained; outer chunks, Q4 row128, planar-Q6 row80,
single-wave standard-Q6 48x64 tiling, the exact unequal-output Q4 QKV+gate
pair, 16-column dual-Q4 output subdivision, and the byte-neutral
Laguna-derived D8-MMQ transfer are rejected. The two shared-Q6 routes decode
one 48x256 slab into 24 KiB LDS while preserving four independent prior 48x64
arithmetic sequences; narrow V, root, rows<512, and peer backends retain their
exact fallbacks. AOTriton attention is active but nonmaterial, and the
remaining primitive add boundary is below the >=1% request gate.

P5 retains primary-plus-residual Q8_1 dp4a for rows1 dense gate/up, an exact
serial-c1 tile8 owner for the 48 Q5T16 recurrent outputs, and an exact four-wave
split-weight owner for the dense Q4 gate/up pair. That split-weight unit lowers
its family from 224 to 120 VGPR and improves fresh graph AR
1.154%/1.118%/0.964% at 512/1K/4K. Natural AR rises **12.1763 -> 12.3105
tok/s (+1.102%)**, with every full/train/heldout/category scope positive.

The latest P5 unit groups the 24 query heads by four K/V heads only for the
backend-qualified dense-H5120/L64 geometry from context 4096. Its rotating-K/V
leaf is BF16-bit exact and improves **0.549226 -> 0.116819 ms (4.7015x,
15/15)**; counterbalanced complete 4K graph AR improves same-source
**11.10932 -> 12.05960 tok/s (+8.554%)** with unchanged tracked peak and zero
teardown. This beats clean llama.cpp HIP by **4.792%** but remains **4.047%**
below Vulkan. All tested trajectories remain exact and bytes/peaks are
unchanged. Native rows and MTP retain their prior owners after the pre-scope B1
diagnostic regressed. At this grouped-GQA checkpoint, 512/1K, natural AR, and
Vulkan 4K remained open; the later fixed-H5120 norm row supersedes those
short-context development rates.

The compiler-clean post-grouped 4K rerank reconciles **80.781185 ms/token** of
kernel work (**95.921%** of profiled host decode) across exactly **934
launches/token**. Q4 split gate/up+SiLU now owns **38.168%**, Q4 singletons
**23.659%**, planar-Q6 BF16 **13.366%**, Q5 recurrent output **6.315%**,
standard-Q6 QKV **5.925%**, and Q6 root **5.703%**. Grouped attention plus
reduction has fallen to **2.097%**, so neither another generic attention variant
nor graph-width work is admissible without a new premise. Evidence:
[`post-grouped decode profile`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-post-grouped-decode-profile.json).

The next exact standard-Q6 screen is also closed. Splitting the rows1 recurrent-
QKV owner from 16 columns into two col8 blocks is BF16-bit exact over a
**123.047-MiB** actual-weight pool but measures **0.611306 -> 0.611347 ms
(0.99993x, 9/15 wins)** and projects effectively zero complete-wall saving.
The transient route is removed; output subdivision is not a path to the
required **1.20305x** family speedup. Evidence:
[`standard-Q6 c1 col8 rejection`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-q6-standard-c1-col8-rejected.json).

The changed-arithmetic QKV layout route is closed too. Moving recurrent QKV to
planar-qmicro and consuming one Q8_1 activation improves a three-weight family
**1.01080x**, but projects only **0.0633%** complete wall. A separately measured
precomputed-Q8 dot proves that even perfectly free quantization through a
hypothetical mixed Q6-QKV/Q4-gate producer reaches only **0.1262%**, still 7.9x
below admission. No mixed runtime package is warranted. Evidence:
[`planar-Q6 Q8 QKV rejection`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-q6-qkv-planar-q8-rejected.json).

The byte-neutral Q4 payload-transfer route is closed as well. The existing
BF16-input dual-interleaved T16 tile2 consumer regresses the current Q8_1x2
split-weight gate/up family **1.428673 -> 1.452220 ms (0.98379x, 7/45 wins)**
and projects **-0.6291%** selected wall. It also differs from the current
changed-association output at 1,476 BF16 positions across three actual layers,
although max KL is only **2.57e-11** and top-1 remains 3/3. No prefill/verifier
package or payload migration is warranted. Evidence:
[`dual-interleaved Q4 rejection`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-q4-dual-interleaved-rejected.json).

The only smaller existing Q4 payload still fails as an operation-complete
package. Its first cooperative metadata-expansion consumer regressed rows1
**1.423347 -> 1.486795 ms (0.95733x, 0/45 wins)**. A materially different
wave-shuffle consumer then realizes the compressed-byte ceiling at rows1:
**1.423808 -> 1.381499 ms (1.03063x, 45/45 wins)**, BF16-bit exact, with each
actual resident pair **103,055,360 -> 100,270,080 bytes (-2.70%)** and a
**1.1518%** selected-4K projection. It cannot be retained under sole ownership,
however. Exact native rows2/3/4 regress aggregate actual-layer wall
**27.78%/28.75%/32.03% (0/135 wins)**, and exact 512/1K/4K WMMA consumers
regress **0.643%/0.312%/0.034%**. Keeping only c1 would require a forbidden
standard-T16 sidecar, so all transient code is removed and standard-Q4T16
remains production. Evidence:
[`first qmicro Q4 split-weight rejection`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-q4-qmicro-split-weight-rejected.json) and
[`wave-shuffled qmicro operation rejection`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-q4-qmicro-wave-meta-rejected.json).

Cross-boundary Q8 production cannot rescue planar-Q6 FFN-down either. With
activation quantization performed entirely outside timing, the best existing
dp4a consumer reaches only **1.00300x (24/45 wins)** against the BF16 owner,
versus **1.08087x** required to save 1% of current 4K kernel wall. This
zero-cost-producer upper bound projects just **0.0400%**, so no fused
SiLU-to-Q8 producer is implemented. Evidence:
[`precomputed-Q8 Q6 bound`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-q6-precomputed-q8-bound-rejected.json).

One robust exact Q4 singleton subdivision survives the otherwise rejected broad
col4 policy. Serial-c1 full-attention K/V at K5120/N1024 improves a 67.5-MiB,
24-weight pool **16.962 -> 16.441 us/projection (1.03169x, 14/15 wins)**,
lowers traced VGPR **96 -> 56**, and changes no BF16 bits or bytes. The bounded
owner projects **11.821 us/token / 0.0146%** of post-grouped 4K kernel wall, so
no request-level result is inferred; exact verified-subwindow policy still
requires retention. Native rows/MTP, peers, and all other Q4 singleton shapes
stay on their prior owners. Evidence:
[`Q4T16 c1 col4 full-K/V`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-q4-single-col4-c1-decode.json).

Packed standard-Q4T16 coefficient publication does not improve the dominant
split-weight gate/up owner. Eight scale/min quartet broadcasts per K256 tile are
BF16-bit exact but change three actual layers **1.434625 -> 1.435730 ms
(0.99923x, 19/45 wins)** and project **-0.0294%** selected wall. Direct metadata
loads remain production. Evidence:
[`packed Q4 metadata rejection`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-q4-packed-meta-broadcast-rejected.json).

Producer-owned planar-Q6 root tile maxima are closed for serial AR. The
existing exact top-1 path preserves every FP32 logit and winner bit on the
994.629-MiB actual head, but changes **4.591176 -> 4.603906 ms (0.99723x,
4/15 wins)** and projects a **0.0158%** selected-wall regression. The direct
full-logit producer plus generic argmax remains production. Evidence:
[`Q6 root top-1 rejection`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-q6-root-top1-rejected.json).

Fixed-H5120 norm dataflow is retained. Caching 20 values/thread while preserving
the generic local256 FP32 tree improves all 128 actual norm leaves **1.23268 ->
0.35870 ms/token (3.4365x, 15/15)**. Complete 512/1K/4K graph AR improves
**1.458%/1.368%/1.380%** to **12.23245/12.06500/12.21721 tok/s**, and natural
AR improves **12.28760 -> 12.45494 (+1.362%)**. Every output/trajectory and
byte remains exact; rows>1, Q8, output norm, and peers stay generic. Evidence:
[`fixed-H5120 norm decode`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-fixed5120-norm-decode.json).

The clean post-commit publication is **399.836/390.793/384.712 prefill tok/s**
and **12.2099/12.0514/12.2095 AR tok/s** at 512/1K/4K. It confirms HIP AR
leads of **0.488%/6.095%** at 512/4K but places 1K **0.109%** behind; Vulkan
remains **2.854-5.254%** ahead. All CVs are below 0.046%, IDs/bytes remain
exact, and teardown is zero. Evidence:
[`post-norm clean publication`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-post-norm-publication.json).

The clean post-norm profile reconciles **79.30459 ms/token** selected kernel
wall against **82.72545 ms/token** profiled host decode at unchanged **934
launches/token**. Norms are now **0.376%**; Q4 dual/single own
**38.698%/23.975%**, and the residual is **3.42086 ms/token**. This supersedes
the post-grouped Amdahl ranking. Evidence:
[`post-norm decode profile`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-post-norm-decode-profile.json).

The first exact graph-gap contraction is retained for serial c1 FFN-down. The
32 Q4T16 and 32 planar-qmicro-Q6 producers fold only the rounded BF16 residual
into their direct store, adding no payload or scratch. The active 4K trace
removes exactly **64 launches/token (934 -> 870)** and improves profiled host
decode **82.46295 -> 82.31707 ms/token (-0.177%)** while kernel wall is flat
within **0.005%**. A counterbalanced 512/128 request gate improves
**12.23017 -> 12.25928 tok/s (+0.238%)**, with both candidate processes above
both controls, exact final IDs/logits, and byte-identical peaks. Rows>1, Q8,
shape/model/backend misses, and peer backends retain the primitive chain.
Evidence:
[`c1 down-residual graph contraction`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-c1-down-residual.json).

The next exact graph-gap contraction promotes the existing scalar Q5T16 GDN
dual-output producer on gfx1151. It preserves every FP32 recurrent output/state
bit while writing the exact RNE BF16 handoff consumed by `ssm_out`, eliminating
the standalone cast after 48 recurrent layers. The 4K trace removes another
**48 launches/token (870 -> 822)** and reduces GDN-start to Q5-start span
**1.36385 -> 1.20856 ms/token (-11.386%)**. Counterbalanced 512/128 graph AR
improves **12.26142 -> 12.28212 tok/s (+0.169%)**, with both candidate processes
above both controls, exact IDs/logits, unchanged peaks, and no bytes. The
verifier-chain alias remains excluded for P6. Evidence:
[`scalar GDN BF16 handoff`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-gdn-bf16-handoff.json).

A second architecture-local contraction promotes the retained exact dense-F32
alpha/beta pair only for gfx1151 rows1 K5120/N48+N48. The 4K trace removes
**48 launches/token (822 -> 774)** and improves alpha/beta-to-Conv span
**0.85573 -> 0.67959 ms/token (-20.583%)**, selected kernel wall **-0.134%**,
and profiled host decode **-0.242%**. Counterbalanced 512/128 graph AR improves
**12.25916 -> 12.27588 tok/s (+0.136%)** with both candidates above both
controls, exact outputs, unchanged peaks, and no bytes. The W7900 rejection and
rows2-4 native/MTP singleton route remain unchanged. Evidence:
[`dense-F32 alpha/beta pair`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-dense-f32-alpha-beta-pair.json).

The next serial contraction joins that pair with independent in-place Conv
channel blocks. A >64-MiB rotating proxy wins 15/15 before routing; the active
4K trace removes **48 launches/token (774 -> 726)** and improves complete
alpha/beta+Conv span **0.85271 -> 0.65549 ms/token (-23.128%)**, selected kernel
wall **-0.096%**, and profiled host decode **-0.173%**. Counterbalanced 512/128
graph AR improves **12.28434 -> 12.30966 tok/s (+0.206%)** with both candidates
above both controls, exact projection/Conv-state/final-logit results, unchanged
peaks, and no bytes. Verifier/native rows and peer backends keep the registered
pair plus Conv fallback. Evidence:
[`serial alpha/beta+Conv`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-alpha-beta-serial-conv.json).

Producer-owned add-RMSNorm -> Q8_1x2 publication is not admitted. The active
trace has 64 boundaries/token, but each Q8_1x2 pack exposes 160 independent
local32 workgroups while fixed-H5120 norm exposes one local256 block/eight
waves; fusion would serialize publication into 20 wave rounds. The directly
applicable exact gfx1151 screen already regressed **0.007578 -> 0.009251 ms
(+22.085%)** with only 96 packs/twelve rounds. No Qwen3.8 code or benchmark is
added; preserve the standalone pack's block parallelism and require a materially
different synchronization mechanism before reopening. Evidence:
[`prior exact rejection`](../benchmarks/results/2026-07-31-gfx1151-laguna-d9-q8-pack-fusion-rejected.json).

Concatenating serial full-attention Q and K projection blocks is rejected too.
The transient Q4T16 kernel preserved Q's eight-column and K's four-column
wave32 arithmetic exactly, but all 16 actual layer pairs over a 630.5-MB
rotating pool changed **0.189805 -> 0.191481 ms/layer (+0.883%, 1/15 wins)**.
That operation-complete timing already includes the removed sequential launch
boundary and projects a **0.026811-ms/token regression**, so all transient code
is removed and no graph gate is run. Evidence:
[`Q/K launch contraction rejection`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-q4-qk-launch-contraction-rejected.json).

The serial full-attention Q/K postprocess contraction is retained. One
local256 composite consumes packed BF16 Q/gate and BF16 K directly while
preserving the existing FP32 head RMSNorm/RoPE trees and gate bits. The 4K
trace removes **32 launches/token (726 -> 694)** and improves the complete
postprocess span **0.214228 -> 0.106899 ms/token (-50.100%)**, selected kernel
wall **-0.0027%**, and profiled host decode **-0.100%**. Counterbalanced
512/128 AR improves **12.29442 -> 12.30314 tok/s (+0.071%)**, with both
candidates above both controls, exact outputs/logits, unchanged peaks, and no
bytes. Shape/registry misses, native rows, prefill, and peer backends retain the
primitive chain. Evidence:
[`Q/K postprocess contraction`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-qk-postprocess-contraction.json).

The recurrent standard-Q6 QKV plus Q4 gate boundary is also retained as a
block-parallel mixed grid, not a serialized fusion. Four actual-weight layer
pairs independently beyond MALL improve **1.16623 -> 1.14551 ms (1.01809x,
14/15)** with both outputs BF16-bit exact. The 4K trace removes **24
launches/token (694 -> 670)**, improves complete pair span **6.91087 -> 6.74110
ms/token (-2.457%)**, selected kernel wall **-0.052%**, and profiled host decode
**-0.253%**. Counterbalanced 512/128 AR improves **12.31947 -> 12.33892 tok/s
(+0.158%)**, with both candidates above both controls, exact outputs/logits,
unchanged peaks, and no bytes. Native rows, prefill, misses, and peer backends
retain the two primitive projections. Evidence:
[`Q6/Q4 mixed grid`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-q6-q4-mixed-grid.json).

The homogeneous-Q4 recurrent analogue is rejected. Concatenating the unchanged
QKV and gate local32 block ranges preserves all BF16 bits but changes four
actual pairs **0.93259 -> 0.93888 ms (+0.675%, 2/15 wins)** over independent
115.625/69.375-MiB resident pools. Transient code is removed before any graph
or request gate. The adjacent Q5 `ssm_out` -> add-RMSNorm store seam is not a
conventional exact rounded-residual fusion either: normalization consumes the
unrounded FP32 sum while residual publication rounds separately to BF16. It
requires a materially different persistent/global-synchronization premise, not
the existing producer-store composite. Evidence:
[`Q4/Q4 recurrent-grid rejection`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-q4-q4-unequal-grid-rejected.json).

The existing split-parallel attention reducer also stays at its retained 32K
threshold. Forcing it at 4K adds **16 launches/token**, changes the complete
reducer **0.084902 -> 0.089330 ms/token (+5.216%)**, and changes the fixed-token
final logit **26.059303 -> 26.155388**. The unchanged grouped-GQA context kernel
is flat; one-run aggregate profiler variance is therefore not a retainable win,
and no request gate is run. Evidence:
[`4K parallel-reducer rejection`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-4k-parallel-attention-reducer-rejected.json).

The narrow full-attention K/V boundary is retained as two exact block-parallel
grids. Across all 16 target layers (eight Q4/Q4 and eight Q4/planar-Q6), a
102.19-MiB resident actual-weight pool improves **0.64858 -> 0.57993 ms/token
(1.11838x, 15/15)** with every BF16 output bit exact. The 4K trace removes
**16 launches/token (670 -> 654)** and improves the complete K/V span
**0.64748 -> 0.56195 ms/token (-13.209%)**. One-run aggregate profiler wall
moves **+0.059% kernel / +0.043% host** within noise, but the unprofiled
counterbalanced 512/128 authority improves **12.34127 -> 12.34844 tok/s
(+0.058%)**, with both candidates above both controls, exact IDs/logits,
unchanged peaks, and no bytes. Native rows/MTP, prefill, NextN, misses, and
peer backends retain the two qualified singleton projections. Evidence:
[`narrow K/V pair`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-narrow-kv-pair.json).

Laguna's exact adjacent-Q/paired-coefficient transport does not materially
transfer to the dominant Qwen3.8 split-weight gate/up owner. Reusing each
packed-Q byte plus aligned d/dmin and scale/min pairs is BF16-bit exact and
positive on all three 95.625-MiB actual layer pairs, but changes the family only
**1.431580 -> 1.425980 ms (1.00393x, 38/45 wins)**. The projected selected-4K
saving is **0.1200 ms/token / 0.1524%**, below the predeclared 1% operation-
complete gate. Transient code is removed and direct per-column loads remain.
Evidence:
[`Q4 pair-coefficient rejection`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-q4-paircoeff-rejected.json).

Extending sole qmicro-Q4 ownership from dense gate/up to the 32 Q4 FFN-down
weights is rejected despite saving another **44,564,480 bytes (42.5 MiB)**.
Actual-weight c1 and rows2-4 residual leaves are BF16-bit exact and improve
**1.0047-1.0124x**, and bounded shared-B/expanded-metadata WMMA makes the owner
operation-complete. The matched 512/128 A/B/A request gate nevertheless places
both candidate decode medians below the clean control: candidate mean
**12.44219** versus **12.45720 tok/s (-0.1205%)**; prefill similarly changes
**399.8383 -> 399.4079 tok/s (-0.1076%)**. Exact trajectories and lower bytes
do not authorize a throughput regression, so the transient owner is removed.
Evidence:
[`qmicro Q4 FFN-down rejection`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-q4-qmicro-down-rejected.json).

The corrected clean post-qmicro production profile explicitly enables the
persistent session, bulk attention, WMMA prefill, GEMV decode, and graph replay.
It reconciles **78.52745 ms/token (96.601%)** of an **81.29028-ms** profiled host
decode at exactly **654 launches/token**, superseding the discarded 719-launch
non-GEMV trace. Qmicro Q4 gate/up leads at **30.06950 ms / 38.292%**, followed
by planar-Q6 down **13.477%**, standard-Q4 singleton **11.985%**, standard-Q4
down **9.375%**, mixed Q6/Q4 **8.658%**, Q5 output **6.518%**, and Q6 root
**5.895%**. Graph width and generic attention remain closed; exact streaming
weight bytes and coalescing are the authority.
Evidence:
[`post-qmicro production decode profile`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-post-qmicro-decode-profile.json).

The profile's first new qmicro gate/up coalescing premise is rejected. Loading
one packed-Q byte once and explicitly publishing both adjacent output-nibble
dp4a packs preserves every BF16 bit, but three 95.625-MiB actual pairs regress
**1.385357 -> 1.454068 ms (+4.960%, 0/45 wins)**. Compiler/cache reuse in the
retained direct qloads is superior; the sibling is removed before graph or
request measurement.
Evidence:
[`qmicro paired-qload rejection`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-qmicro-paired-qload-rejected.json).

A matched clean process-level scheduling check also keeps the gfx1151
`GPU_MAX_HW_QUEUES=2` default. One queue changes 512/128 graph AR **12.40721 ->
12.40308 tok/s (-0.0333%)** and prefill **400.5497 -> 400.2541 (-0.0738%)**
with exact IDs, bytes, and teardown. Queue count is not a material remaining
serial-AR gap.
Evidence:
[`hardware-queue count rejection`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-hardware-queue-count-rejected.json).

The prior gfx1100 wide-weight-arena premise does not improve Qwen3.8 decode on
gfx1151. Raising the selective cutoff from 16 to 80 MiB collapses physical
weight owners **371 -> 3** with exact trajectories and tracked bytes, but matched
512/128 AR changes **12.41488 -> 12.39701 tok/s (-0.1439%)** and prefill
**402.319 -> 400.743 (-0.3918%)**. Task 22 keeps the 16-MiB arena; task 24 may
reopen only with matched whole-GTT evidence.
Evidence:
[`wide weight arena rejection`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-wide-weight-arena-rejected.json).

P5 closes through the fully qualified Q4_K_S model representation, not a
prompt-conditioned route. The 0.835-GiB smaller planned weight payload plus
three exact representation-independent policy transfers—qmicro Q4 gate/up+
SiLU, Q4 down+residual, and fixed-H5120 norms—move matched 512/128 AR
**12.42932 -> 13.06854 tok/s (+5.143%)**. Clean commit `3118943eb` publishes
512/1K/4K at **13.03883/12.86679/13.02544 tok/s**, respectively
**2.162%/1.156%/3.637%** above frozen clean llama Vulkan and further above
llama HIP; prefill also beats both backends at every required shape. Natural
true AR is **13.33276 tok/s**
with identical trajectories across all 30 requests and all categories, versus
same-file Q4_K_S llama HIP/Vulkan at **5.53853/7.51888 tok/s**. That
publication also exposed a native rows2-4 gate/up arithmetic mismatch on
`general_ja_plan` and `general_ja_explain`; task 23 subsequently repairs it with
the exact shared-weight Q8_1x2 rowtile documented under P6.
Evidence:
[`clean Q4_K_S publication`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-q4ks-clean-publication.json),
[`Q4_K_S true-AR policy retention`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-q4ks-decode-policies-retained.json), and
[`exact Q4_K_S native B3`](../benchmarks/results/2026-08-17-gfx1151-qwen38-27b-q4ks-exact-native-b3.json).

The campaign closes on 2026-08-17 at merged commit `20e5106da`. One clean
refresh per scope confirms every milestone on the final source: 512/1K/4K
prefill **379.398/376.554/369.755 tok/s** and AR **13.1014/12.9231/13.0953
tok/s** beat both frozen llama backends at every shape (+0.479-7.653% prefill,
+1.599-4.190% AR over the faster comparator); natural true AR is
**13.36641 tok/s**; exact native B3 is **23.85263 tok/s / 1.7845x own AR**
(train 1.8113x, heldout 1.7458x; every category above 1.52x), **21.48%** above
the correctness-valid llama HIP B3 row. The retained task-24 memory package
costs **-1.41%** B3 versus the pre-merge 24.19347 row while lifting AR and
keeping process GTT at **15.275/15.710/15.727 GiB** (shapes) and **15.899
GiB** (B3), below every lower valid llama row. All trajectories, acceptance
decisions, and teardowns are exact, and the milestone pytest tier passes with
only host-inapplicable gfx1100-only skips. Evidence:
[`G6 closure`](../benchmarks/results/2026-08-17-gfx1151-qwen38-27b-q4ks-g6-closure.json).

This document is the closed campaign record. Section 2 freezes the clean G0
snapshot used as the optimization denominator; it is not itself an optimization
claim.

Related authorities:

- [`QWEN35-08B-GFX1151-VULKAN-PARITY.md`](QWEN35-08B-GFX1151-VULKAN-PARITY.md)
  — recent dense gfx1151 campaign and its accepted/rejected owner ladder.
- [`QWEN36-27B-GGUF-CAMPAIGN.md`](QWEN36-27B-GGUF-CAMPAIGN.md) — historical
  W7900 dense-27B AR/MTP campaign.
- [`QWEN36-27B-GGUF-7900XTX.md`](QWEN36-27B-GGUF-7900XTX.md) — current
  single-layout gfx1100 package, memory ownership, and three-shape protocol.
- [`ROOFLINE-gfx1151.md`](ROOFLINE-gfx1151.md) and
  [`TUNING-gfx1151.md`](TUNING-gfx1151.md) — 40-CU, ~221-GB/s practical-read,
  cache, occupancy, and profiling rules.
- [`KVCACHE.md`](KVCACHE.md) — native INT8 K/V implementation and current
  dense-27B quality/runtime blockers.
- [`BENCHMARK.md`](BENCHMARK.md), [`TESTING.md`](TESTING.md), and
  [`KERNELS.md`](KERNELS.md) — evidence, anti-gaming, correctness, and kernel
  lineage contracts.

The 0.8B campaign left its automatic D08-T1 transfer blocked because Vulkan
parity did not close. The user's 2026-08-15 instruction is the explicit human
approval required to open this separate 27B campaign. It permits a fresh 27B
profile and plan; it does not make any 0.8B ratio transferable evidence.

---

## 1. Fixed target

### 1.1 Model and hardware identity

| Field | Value |
| --- | --- |
| Model | `/models/gguf/Qwen3.8-27B-Q4_K_M.gguf` |
| File bytes | `17,106,775,008` |
| Full SHA-256 | `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169` |
| Sampled fingerprint | `e7c9c4c1516de52b534407f0a4f22ad7392fcc796bc83873ca25fa3bf5fcbcb5` |
| Tensor inventory SHA-256 | `d32f336dc4f35a7ea84bac6c0c611b91c9918d05866638c8ceb30d1e891be69a` |
| GGUF tensors / type | `866` / `MOSTLY_Q4_K_M` |
| Geometry | hidden `5120`, FFN `17408`, 64 AR blocks + trailing NextN block |
| Layer mix | 48 linear-attention/GDN + 16 full-attention |
| Attention | 24 Q heads, 4 K/V heads, head dim 256 |
| Context declaration | 262,144 tokens |
| GPU | Radeon 8060S, `gfx1151`, 40 CUs, unified memory |
| Practical read roof | approximately 221 GB/s pending a campaign-local HIP refresh |

Qwen3.8 and Qwen3.6 files have equal execution geometry, tensor descriptors,
and tensor payload byte counts, but different weights and model semantics.
Qwen3.6 evidence can qualify a candidate design; it cannot replace a Qwen3.8
quality or complete-model gate.

### 1.2 Working set and two required input classes

The campaign optimizes exactly these context/decode shapes first:

- `512/128`
- `1024/128`
- `4096/128`

Each implementation gate uses both input classes:

1. **Exact shape control:** token ID `9707` repeated to the prompt length,
   greedy decode, EOS ignored. This gives matched context, deterministic state,
   and repeatable prefill/AR profiling.
2. **Natural semantic control:** all ten prompts in
   `benchmarks/prompts/mtpbench-code-general-ja.jsonl`, covering all four
   categories, six train prompts, and four heldouts. This is mandatory for MTP,
   cache quality, and any prompt-sensitive keep/revert decision.

Repeated-token MTP is a context/transaction stress only. It is never an
acceptance or speculative-speed headline.

### 1.3 Timing scopes

Keep these scopes separate:

- **Prefill:** prompt processing only, excluding model load and graph capture.
- **Shape AR:** 128 true scalar autoregressive transitions after each repeated-
  token prompt. Report capture-inclusive and steady replay fields separately.
- **Natural AR:** no-MTP generation from the same natural-suite harness used by
  MTP. It is the only valid MTP speed denominator.
- **MTP:** complete proposal + target verify + accept/commit + correction/
  scheduler wall for 24 timed transitions per prompt. Client/HTTP wall remains
  separate.
- **Memory:** hipEngine tracked ownership, process GTT delta, system-available
  delta, process RSS, transient high water, retained weight/KV bytes, and
  teardown. Do not add RSS and GTT as though they were disjoint physical pools
  on this APU.

---

## 2. G0 current snapshot and target gap

G0 uses clean hipEngine `943ec15f5`, clean llama.cpp HIP/Vulkan build 10438
`9d57ce456`, BF16 K/V, and Radeon 8060S/`gfx1151`. Binding shape rows use token
ID 9707 repeated exactly, one same-shape warmup, three measured runs, full
output hashes, and one right-sized process per engine/shape. `llama-bench`
rows are retained only as split-timing/profile diagnostics because it cannot
accept the explicit token arrays; the first such diagnostic also used F16 K/V
and is excluded from every binding denominator.

### 2.1 Matched right-sized shape rows

| Shape | hipEngine prefill | llama HIP | llama Vulkan | hipEngine AR | llama HIP | llama Vulkan | hipEngine GTT | Lower llama GTT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512/128 | `85.288` | **`352.426`** | `242.956` | `7.0257` | `12.1506` | **`12.7629`** | `33.125 GiB` | **`15.785 GiB`** |
| 1K/128 | `84.497` | **`364.443`** | `247.610` | `6.9592` | `12.0645` | **`12.7197`** | `33.613 GiB` | **`15.816 GiB`** |
| 4K/128 | `84.204` | **`367.993`** | `354.368` | `6.7144` | `11.5081` | **`12.5683`** | `36.046 GiB` | **`16.004 GiB`** |

All six llama rows and the common-capacity hipEngine control produce the same
129-token SHA-256 `5055191f…ee652`; hipEngine's three runs retain stable token
9707 and return tracked ownership to zero. Relative to the faster comparator,
hipEngine needs **+313.22%/+331.31%/+337.02%** prefill and
**+81.66%/+82.77%/+87.19%** AR lift. Meeting the lower matched llama GTT row
requires **52.35%/52.95%/55.60%** less hipEngine process-GTT delta.

### 2.2 Natural-suite AR and exact MTP

Ten prompts cover all four categories, six train prompts and four heldouts,
with three runs and 720 timed transitions per mode.

| Engine/mode | Throughput | Own-AR ratio | AR-equivalent output | Process GTT delta |
| --- | ---: | ---: | --- | ---: |
| hipEngine true AR | `7.10844` | `1.0000x` | deterministic reference | `34.555 GiB` |
| hipEngine exact B1 | `14.79394` | `2.0812x` | `30/30` | shared process |
| hipEngine exact B2 | `18.48249` | `2.6001x` | `30/30` | shared process |
| hipEngine exact B3 | **`19.72960`** | **`2.7755x`** | `30/30`; GPU accept = CPU | shared process |
| llama.cpp HIP AR | `12.06439` | `1.0000x` | deterministic reference | **`15.803 GiB`** |
| llama.cpp HIP B3 | `19.63473` | `1.6275x` | `30/30`; valid comparator | `16.358 GiB` |
| llama.cpp Vulkan AR | **`12.77754`** | `1.0000x` | nondeterministic on 2/10 prompts | `15.871 GiB` |
| llama.cpp Vulkan B3 | `26.10541` | `2.0431x` | invalid: `27/30`; stretch rate | `16.722 GiB` |

hipEngine needs **+79.75%** natural-AR lift. Its opening exact B3 already leads
the binding correctness-valid HIP row by **0.483%**, but the invalid Vulkan
rate remains a **+32.32%** stretch gap. Vulkan B1-B3 each match only 9/10
corresponding AR prompts per run, so none can be promoted as a correctness-valid
MTP target.

### 2.3 Opening ownership and ranked profile

The current gfx1151 materialization plan owns `33,127,663,616` weight bytes,
including `10,790,502,400` alternate-layout bytes; the qualified gfx1100
single-layout plan is `17,121,478,656` bytes with zero alternate bytes. The
same-stream pp512/pp1K/pp4K ledgers reconcile at least 97.66% of complete wall;
linear-attention FFN gate/up leads at 32.93-33.22%, followed by SSM output,
linear-attention FFN down/residual, full-attention FFN gate/up, and linear-
attention QKV/gate. The 512 eager AR ledger reconciles 99.55% of decode wall
and assigns 42.76% to FFN gate/up. This admits sole compressed ownership before
micro-tuning. Full provenance, samples, memory scopes, trace resources and raw
hashes are in
[`2026-08-15-gfx1151-qwen38-27b-p0-baseline.json`](../benchmarks/results/2026-08-15-gfx1151-qwen38-27b-p0-baseline.json).

---

## 3. Lessons carried forward

### 3.1 What the gfx1151 0.8B campaign established

| Lesson | Evidence | Qwen3.8-27B action |
| --- | --- | --- |
| Certify the actual route before kernel work. | The 0.8B opening rows had WMMA, GEMV, and graph disabled; route correction changed every gap. | M0 records requested and effective layout, prefill, graph, GDN, attention, and embedding owners. Fail closed on mismatches. |
| Re-profile after every structural keep. | Q5T16, GDN, and attention changes repeatedly changed Amdahl order. | Exactly one structural owner is active; M1 is repeated after each retained ownership or schedule change. |
| Existing gfx11 kernels can hide behind the wrong policy. | Routing the registered pack8 WMMA leaf improved Q4 pp512 by 35%; dense-BF16 WMMA added 27%. | Audit package capability misses before writing kernels. Current dense-H5120 compressed owners are explicitly disabled on gfx1151. |
| Sole compressed residency can improve speed and memory together. | 0.8B Q5T16 QKV/SSM-out and Q4T16 attention-Q removed expanded weights and improved complete wall. | Qualify the 27B sole-Q4, Q5, and planar-Q6 package before micro-tuning fallback pack8/dense paths. |
| gfx1151 GDN is geometry-sensitive. | 16K/16V cluster8 beat the underfilled 64-block LDS32 route; Q4 and Q8 needed independent gates. | Re-screen the 27B 16K/48V compact-peer chain. Do not copy the 0.8B 16K/16V selector. |
| Decode is device-critical when graph is already healthy. | 0.8B graph launch + Python was ~0.2%; launch count alone did not explain the Vulkan gap. | Profile current 27B graph/device gaps before submission work. Favor bytes, coalescing, row reuse, and exact fusion. |
| Microbench cache state can lie. | Small repeated buffers measured MALL; >64-MiB cycling pools exposed cold-stream behavior. | Every decode weight screen cycles more than 2x the 32-MiB MALL and uses actual role weights. |
| Public and core timing differ. | 0.8B public Q4 decode nearly reached Vulkan while core remained materially behind. | Report core/teacher-forced and public greedy boundaries; never subtract sampler from one engine only. |

Closed 0.8B directions stay closed unless the 27B profile supplies a new
premise: large-LDS pack8 WMMA was parity-or-worse on wave32; blanket
non-temporal weight loads lost end to end; generic wave64, blind tile sweeps,
and launch-count-only work did not close the gap.

### 3.2 What the gfx1100 dense-27B campaign established

The highest-leverage current transfer is **package ownership**, not a new
algorithm. `hip_gfx1151` currently disables all three dense-H5120 compressed
ownership switches that define the modern gfx1100 package:

| Capability | gfx1100 | gfx1151 opening state | Why it is first-class |
| --- | --- | --- | --- |
| `GGUF_DENSE_Q4_T16` | on | **off** | gfx1100 replaces the 13.037-GiB pack8 payload instead of retaining it beside a 10.049-GiB T16 sidecar. |
| `GGUF_DENSE_Q6_T16_QMICRO_PLANAR` | on | **off** | Wide Q6 projections/root become one byte-neutral planar owner; prior dense-BF16 residency fell by about 4.5 GiB. |
| `GGUF_DENSE_Q5_T16_SSM_OUT` | on | **off** | Replaces 48 expanded recurrent-output weights; gfx1100 saved 1.824 GiB and improved AR, verifier, and prefill. |
| `GGUF_GDN_PREFILL_AUTO_MODE` | compact peer | LDS32 direct | 27B 16K/48V Q/K should be materialized once per K head, not per V head. |

These bodies already live in-tree and compile as native gfx1151 code objects,
but policy enablement remains blocked on architecture-local actual-weight,
resource, complete-state, and complete-model gates. The campaign must not flip
all switches together.

Additional gfx1100 lessons:

- One physical payload per logical weight is the production contract. Do not
  restore pack8+T16 or raw+T16 duplication to rescue one operation.
- Operation completeness binds each layout: c1, verifier rows 2-4, prefill
  512/1K/4K plus tails, root top-1, graph capture, and unfused fallback.
- gfx1100 rocBLAS solution indices and row thresholds are device/compiler
  evidence, not gfx1151 defaults. Re-screen solution choice and chunk policy.
- Dense B3 became fast through native rows, row reuse, staged dependency-aware
  scheduling, compressed root scoring, state journals, and direct graph
  handoff. B4/B5 lost because their extra target-row cost exceeded break-even.
- Small exact sub-window wins are retainable, but profiler wall and aggregate
  suite noise must not be used to invent causality.

---

## 4. Definition of done

The BF16 campaign closes only when all conditions pass on the same clean
hipEngine source and current clean llama.cpp source:

1. **Prefill:** hipEngine is faster than both llama.cpp HIP and Vulkan at
   512/128, 1K/128, and 4K/128 under matched token IDs, BF16 K/V, right-sized
   context, full layer offload, warmup, and timing boundary.
2. **True AR:** hipEngine is faster than both backends at all three repeated-
   token shape controls and on natural true AR. Capture/setup ownership is
   disclosed and cannot be hidden asymmetrically.
3. **Exact B3 MTP:** the complete ten-prompt suite matches hipEngine true-AR
   greedy IDs and GPU acceptance matches CPU. Absolute B3 exceeds every
   correctness-valid llama.cpp backend and stays above own true AR.
4. **Memory:** for each matched shape and selected natural B3 route, hipEngine
   process GTT delta is no larger than the lower clean llama.cpp backend. Also
   report tracked ownership, system available delta, RSS, peak transient, and
   zero tracked bytes after close.
5. **Correctness:** all touched primitive CPU oracles pass; model-level KL is
   <=0.05 and top-1 agreement >=90%; logits/state/KV are finite and the
   route-specific exact transaction/trajectory gates pass.
6. **Ownership:** no duplicate/alternate payload, lazy shadow, or unbounded
   hot-path allocation appears. Every fused route keeps an unfused same-layout
   fallback.
7. **Durability:** retained rows update a compact artifact, immutable worklog,
   benchmark README/changelog, refactor ledger where needed, and package
   defaults in one validated atomic commit.

### 4.1 INT8 K/V closure is separate

An INT8 K/V route may close as an explicit supported capacity mode before it
becomes the default. It must pass all of the following:

- native matched-context quality on the full suite at 512/8, 1K/8, and 4K/16;
- mean KL <=0.05 and minimum per-prompt top-1 >=90%, finite logits, deterministic
  candidate repeats, and disclosed maximum KL/first mismatch;
- zero persistent BF16 K/V shadow and a bounded, audited prefill transient;
- graph-safe 512/1K/4K AR and correct MTP transaction/rollback when MTP is
  enabled;
- lower retained and peak memory than BF16 at each supported capacity;
- no throughput regression for default promotion. An explicit capacity mode may
  accept a small measured speed cost only when the BF16 route cannot satisfy the
  declared capacity, and must state that tradeoff.

A 128K/16 natural quality transfer and real 256K capacity row remain required
before advertising long-context INT8, even if the initial working set passes.

---

## 5. Measurement and profiling contract

### M0 — Freeze clean three-engine baselines

Run one engine at a time on an otherwise-idle GPU. Use one warmup and three
measured repetitions for development baselines; final closure uses five
counter-rotated blocks where engine lifecycle permits.

Required rows:

- hipEngine, llama.cpp HIP, and llama.cpp Vulkan at 512/128, 1K/128, 4K/128;
- natural true AR and B1-B3 for hipEngine and both llama backends;
- llama B4 only as a separately correctness-gated diagnostic;
- right-sized BF16 KV capacity for each shape, plus one common-capacity control
  to distinguish model/scratch ownership from KV growth.

Artifacts must contain model and binary hashes, source clean-state axes,
backend/target/device, ROCm/compiler/Mesa, commands/env, raw samples, effective
routes, output hashes, and all memory scopes. M0 also runs plan/live residency
censuses to quantify current pack8, T16 sidecar, dense-BF16 Q5/Q6, scratch,
state, graph, and K/V bytes.

### M1 — Build a complete semantic ledger

On gfx1151 ROCm 7.15, rocprof kernel timestamps can be zero. Use:

- `rocprofv3` for symbols, counts, grid/workgroup, VGPR/SGPR/LDS/scratch, and API
  census;
- same-stream `wall_clock64()` markers for semantic device-stage timing;
- unprofiled host wall for toplines;
- separately cached/profiling children so no measured process launches `hipcc`.

Profiles required before implementation:

1. pp512, pp1K, and pp4K prefill stage ledgers;
2. 128-transition production graph/direct AR ledger at short and 4K context;
3. one natural B3 final child using direct ROCTX windows, never the parent suite;
4. matching llama.cpp HIP and Vulkan operation ledgers where the backend tools
   support them.

Every kernel/node is assigned to embedding, norm, linear projections, conv,
GDN, full-attention projections, RoPE/KV, attention core, dense FFN, residual,
root/sampler, copy/API, or explicit residual. Stage sums must reconcile within
10% of complete wall, and preferably within 1%; otherwise fix the measurement
before choosing an owner.

### M2 — Re-profile after structural keeps

Sole-layout and GDN changes invalidate the opening profile. Repeat only the
changed shape/phase plus one complete control, then rerank. Do not work from a
stale Amdahl table.

---

## 6. Prioritized execution ladder

Only one implementation owner is active at a time. The stated order is the
opening hypothesis; each retained structural unit can change it.

### P0 — Baseline, census, and route certification

| ID | Work | Exit gate |
| --- | --- | --- |
| P0.1 | Run M0 three-engine 512/1K/4K and natural AR/B3 matrix. | Clean provenance, matched inputs/timing/KV, output verdicts, complete memory scopes. |
| P0.2 | Record effective route and physical weight census. | All requested owners are concrete; duplicate/alternate bytes and per-quant residency are explicit. |
| P0.3 | Run M1 semantic profiles. | Complete owner ledger and ranked whole-request ceilings. |

P0 completed on 2026-08-15 with the compact G0 artifact linked in Section 2.
No device code began before P0. A missing/incorrect registered route is fixed
as a route unit and followed by a profile refresh.

### P1 — Sole dense Q4T16 ownership

This is the first planned implementation because it is already the production
`gfx1100` representation and likely explains most of the 31.659-GiB opening
ownership.

1. Screen actual Qwen3.8 Q4 weights from a >64-MiB rotating pool for c1,
   verifier rows 2/3/4, M512/M1024/M4096, and tail rows.
2. Prove each operation uses the same T16 payload: single, dual+SiLU,
   down+residual/unfused add, attention projections, and graph capture.
3. Run the complete CPU/layout, full-state 512/1K/4K, and dense B1-B3
   transaction gates.
4. Run one package-default A/B against the P0-certified opening layout and
   confirm `alternate_layout_weight_bytes=0` and no lazy pack8 allocation.

Keep only if correctness passes, every required operation is available, memory
falls materially, and all three complete shapes are non-regressive. Do not
retain pack8 as a sidecar. A losing prefill operation must be repaired against
T16 or use a same-T16 unfused fallback.

### P2 — Sole planar Q6 and sole Q5T16

Run these as separate logical units, in the order selected by the post-P1
profile and byte census.

#### P2A — Planar qmicro Q6

Gate c1 BF16/F32 output, rows 2-4, wide FFN-down/QKV, narrow V, untied root
full-logit/top-1, M512/1K/4K prefill, NextN borrowing, and teardown. Do not copy
W7900 rocBLAS solution IDs. Start with native exact custom leaves, then screen at
most three gfx1151 prefill schedules if prefill is the blocking operation.

#### P2B — Q5T16 recurrent output

Gate the exact K6144/N5120 role across c1, rows 2-4, M512/1K/4K, GDN BF16
handoff, and actual rotating-cache behavior. The 0.8B Q5T16 result proves the
family can win on gfx1151, not that this larger shape wins. Preserve dense BF16
as an unfused numerical oracle, not a persistent shadow.

Each unit reruns M2. The expected combined outcome is single-layout-class
resident memory near the gfx1100 package rather than the current 31.7-GiB
class; only measurement can establish the actual gfx1151 value.

### P3 — Compact-peer 27B GDN and scratch ABI

Qualify `chain_compact_peer_wave32` for the exact 16-K-head/48-V-head/128x128
Qwen3.8 geometry:

- bit/state comparison against peer-wave and scalar-exact oracles;
- actual 512/1K/4K complete-chain timing and all 48 layer dispatches;
- normalized Q/K materialized once per K head;
- right-sized persistent scratch and zero stale view/lifetime overlap;
- AR/MTP state, graph, and category quality guards.

Do not use the 0.8B cluster8 16K/16V selector. Retain the current LDS32 direct
route as an explicit exact rollback until one release window closes.

### P4 — Prefill to parity and beyond

After P1-P3, select only the largest current prefill owner.

Candidate order:

1. **T16 projection schedule:** screen gfx1151-native WMMA/row policies for the
   actual Q4/Q5/Q6 roles. Treat W7900 source-F16/rocBLAS policies as design
   references; reselect solution and zero-workspace behavior locally.
2. **GDN/conv:** tune only if compact peer remains a leading bucket. Check grid
   sufficiency for 40 CUs, state traffic, VGPR, and scratch before another
   arithmetic variant.
3. **Chunking:** screen 128/256/512 rows at all three working shapes after the
   new owners land. The older gfx1151 all256 result is a starting point, not a
   universal answer.
4. **Full attention:** verify that AOTriton/native attention is actually active
   at each shape. Tune it only if the semantic ledger makes it material.
5. **Boundary fusion:** only operation-complete fusions that remove measured
   global traffic and retain the same sole payload can proceed.

A leaf normally needs >=1.10x and >=1% projected request saving to receive a
full-model A/B. Keep a smaller exact non-regressive win if already measured, but
close the package rather than opening an unbounded variant ladder.

The initial P4 closure on 2026-08-15 followed that bound. Q5 K6,144/N5,120
source-F16 survived while chunk128/256/512, Q4 dual row128, planar-Q6 row80,
standard-Q6 48x64, unequal-output Q4, dual-Q4 col16, and dense D8-MMQ all failed
their declared gates. The materially distinct shared-weight screen reopened P4:
one 128-thread/four-wave workgroup decodes each planar-Q6 48x256 slab once into
24 KiB LDS while preserving four independent retained 48x64 K16 WMMA sequences.
The universal route is rejected because narrow K5120/N1024 V reaches only
1.033x and 11/15 wins at 1K. The admitted rows>=512 K17408/N5120 FFN-down scope
is BF16-bit exact and improves **1.502x/1.421x/1.474x** at 512/1K/4K with
**45/45 wins**, projecting **5.132%/4.207%/4.606%** complete-request savings.

The first same-source full-model gate confirms **344.886/339.548/334.718 ->
363.521/354.231/349.204 tok/s (+5.403%/+4.325%/+4.328%)** with identical
preview IDs, byte-identical tracked peaks, and zero teardown ownership. Its
post-retain profile exposes the 24 standard-Q6 K5120/N10240 QKV calls at
**191.230 ms / 13.54%** of pp512 kernel wall. A second four-wave body applies
the same inter-wave decoded-slab sharing to standard T16 while preserving the
previous exact 48x64 sequence. Both immutable QKV weights improve
**2.961-3.548x** across 512/1K/4K with **90/90 wins**, zero BF16 mismatches, and
projected complete savings **8.998%/9.404%/9.668%**.

The cumulative full-model gate confirms **362.752/354.270/349.130 ->
398.792/391.861/384.628 tok/s (+9.935%/+10.611%/+10.168%)** with identical IDs,
byte-identical tracked peaks, and zero teardown ownership. The post-commit clean
publication on `a06589f34` then measures
**399.031/391.276/385.330 tok/s**, with all prefill CVs below 0.094%, stable
9707 final IDs, and zero tracked teardown. It beats clean llama HIP
**13.224%/7.363%/4.711%** and Vulkan **64.240%/58.021%/8.737%** at 512/1K/4K.
G2 and P4 are complete; G3 and G5 remain separate open gates.

### P5 — True-AR decode

Re-profile c1 after compressed ownership. Rank by complete graph-stage wall,
not by historical W7900 percentages.

Candidate classes:

1. actual Q4/Q5/Q6 GEMV bandwidth, coalescing, and thread geometry using
   >64-MiB cycling pools;
2. exact gate/up+SiLU, down+residual, and rounded next-RMSNorm boundaries from
   the gfx1100 family, each independently gated for H5120 gfx1151;
3. Conv/GDN producer+handoff and state-journal copy ownership;
4. 24Q/4KV/D256 full-attention crossover at contexts around 512, 1K, and 4K,
   with split count scaled for 40 CUs and complete `KVLiveSpans`;
5. root top-1 and token transport only if they own a measured positive gap;
6. graph/API work only if a fresh direct census finds >=1% removable host/copy
   wall. Capture-inclusive results remain mandatory.

A decode candidate must improve both repeated-shape AR and natural true AR or
have an explicitly bounded context-specific policy that leaves every other
shape unchanged.

P5's first retain is the rows1 H5120/N17408 dense gate/up Q8_1x2 dp4a+SiLU
owner. Its actual-weight family improves 1.03427x, clears the 1% projection
screen, improves all three graph-AR rows and every natural full/train/heldout/
category scope, and preserves all tested trajectories. Single-plane Q8_1 is
rejected because it changes heldout `general_ja_explain`; exact dual fusion,
Q6 residual fusion, local64, K coalescing, static-K, and packed quant loads are
rejected on performance. Same-Q4 QKV+gate Q8_1x2 reuse is also rejected: only
24/48 pairs are homogeneous Q4/Q4, and its 1.01892x actual-pair win projects to
just 0.121% of selected graph wall. Re-rank the remaining profile for a
materially larger operation-complete boundary; heterogeneous Q6-QKV/Q4-gate
requires an independent design rather than the same-Q4 dual body. Q8_1x2
thread geometry is closed too: t32 regresses 4.85%, while t128 improves the
family 1.01774x but projects only 0.676% complete wall and changes the reduction
boundary, below the 1% admission threshold for another full natural gate. An
exact tile8 split is closed as well: duplicated T16 metadata/activation work
regresses the Q8_1x2 family 5.25% and projects -2.04% complete wall. Extending
two-plane dp4a to single Q4 projections is rejected too: all six actual-weight
roles regress and call-weighted projection is -1.129% complete selected wall.
Existing planar-Q6 one-plane Q8_1/dp4a is closed at t64/t128/t256 too: every
actual FFN-down layer regresses, with best t256 projecting -0.103% wall. Exact
c1 output subdivision is also closed: the existing planar-qmicro col8 body is
bit-identical at rows1 but regresses FFN-down **0.996617 -> 1.002849 ms** and
the FP32 root **4.647579 -> 4.686152 ms**, projecting -0.130% selected wall.
The exact dense-Q5T16 coefficient-publication screen is closed as well: wave-uniform
metadata shuffles regress all five actual recurrent-output layers by 1.71-3.43%
and project -0.193% selected wall. The existing generic split-K3 attention
transfer is closed at the p512 decode window too: graph AR regresses both fresh
controls by 0.124-0.134% and changes the fixed-token final logit, far from the
48.071% attention-package saving required for a 1% request advance. One
sub-threshold exact micro-win is retained under the repository's non-regressive
micro-win rule: splitting each serial-c1 Q5T16 K6144/N5120 recurrent output
across two eight-column owners improves five cache-cold actual layers
**0.609500 -> 0.570542 ms (1.06828x, 74/75)**. Reverse-order p512 graph pairs
improve **+0.586%/+0.602%**, fresh 1K/4K controls improve **+0.692%/+0.531%**,
and natural true AR improves every full/train/heldout/category scope to
**12.157751 tok/s**. Outputs are BF16-bit exact, residency and tracked peaks are
unchanged, and the final policy excludes `native_batch_decode_session`, leaving
B1-B3 and all native rows on the direct owner. Evidence:
[`Q5T16 serial-c1 tile8`](../benchmarks/results/2026-08-15-gfx1151-qwen38-27b-q5-dense-tile8-decode.json).

A second exact micro-win splits the dominant residual-Q8_1x2 Q4T16 gate/up
consumer into independent two-wave gate and up owners. This halves each
thread's accumulator planes and lowers traced resources **224 -> 120 VGPR** and
**1,024 -> 512 B LDS** with zero scratch. Three cache-cold actual layers improve
**1.445515 -> 1.432400 ms (1.00916x, 39/45)**; although that projects only
0.352% selected wall, complete graph AR improves **1.154%/1.118%/0.964%** at
512/1K/4K and every candidate sample/process beats control. The ten-prompt
natural suite improves **12.176315 -> 12.310492 tok/s (+1.102%)** and its
minimum full/train/heldout/category ratio is 1.01078. BF16 outputs, all 30 AR
trajectories, all 30 diagnostic B1 trajectories, peaks, and teardown are exact.
Because pre-scope B1 regressed 0.290%, `native_batch_decode_session` explicitly
keeps the prior Q8_1x2 owner; the final policy changes serial true AR only.
Evidence:
[`Q4T16 split-weight decode`](../benchmarks/results/2026-08-15-gfx1151-qwen38-27b-q4-q8x2-split-weight-decode.json).

The 2026-08-16 production-numerics rebase supersedes that Q4_K_M serial-c1
owner selection without invalidating the historical measurement. Against the
current registered exact local32 dual+SiLU owner, the changed route passes a
clean 18-prompt/450-row strict-teacher gate (max KL `0.0008333`, `99.778%`
top-1, exact three-run repeats) but loses seven same-session p512/d128 pairs at
`0.998071x` with one win. Exact local32 is therefore restored as the Q4_K_M
serial-c1 package default; its native B1 retains the separately qualified
non-split Q8_1x2 owner. The selected Q4_K_S representation independently keeps
split-weight serial c1 and row-independent native rows2-4 under its exact gates.
Evidence: [`strict requalification`](../benchmarks/results/2026-08-16-gfx1151-qwen38-dense-pair-requalification.json)
and [`current A/B`](../benchmarks/results/2026-08-16-gfx1151-qwen38-dense-pair-strict-default.json).

Fresh graph/API census is closed as a P5 route. A compiler-clean selected-region
trace assigns 96.01% of measured wall to kernels, but widening the existing
exact device-feedback graph from one step to 2/4/8 steps reduces launches
**128 -> 64/32/16** while changing complete p512 AR only
**-0.018%/+0.032%/-0.005%**. IDs, peaks, graph ownership, and teardown remain
exact; the apparent trace residual is not removable launch/synchronize wall.
Keep one-step replay and require a new runtime/driver or host-wall premise before
reopening submission width. Evidence:
[`multistep graph rejection`](../benchmarks/results/2026-08-15-gfx1151-qwen38-27b-multistep-graph-replay-rejected.json).

The post-grouped rerank's first compressed-GEMV screen is also closed. A
byte-neutral Q8_1x2 producer interleaves each K4 primary/residual pair so the
split-weight consumer replaces two far-apart activation loads with one aligned
64-bit record. It is BF16-bit exact and improves three actual gate/up layer
medians **1.423306 -> 1.420789 ms (1.00177x)**, but wins only **24/45** pairs and
projects **7.122 ms / 0.0685%** selected wall. Transient code is removed; the
plane-major owner remains. Evidence:
[`interleaved Q8_1x2 rejection`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-q4-q8x2-interleaved-rejected.json).

The pre-existing exact BF16-input dual-interleaved T16 layout also fails against
the newer retained split-weight owner. Three 95.625-MiB actual gate/up pairs
measure **1.428673 -> 1.452220 ms (0.98379x, 7/45 wins)**, projecting
**-8.131 ms / -0.6291%** selected wall. Both payloads are exactly byte-neutral,
but the candidate is slower and its exact BF16 association differs from the
current Q8_1x2 route at 1,476 positions, so operation-complete integration stops.
Evidence:
[`dual-interleaved Q4 rejection`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-q4-dual-interleaved-rejected.json).

Qmicro closes the remaining existing Q4 resident-byte route only after two
bounded dataflows. Its K256/N16 tile is **2,304 instead of 2,368 bytes
(-2.70%)**. Cooperative per-block metadata expansion first regresses three
actual rows1 layers **1.423347 -> 1.486795 ms (0.95733x, 0/45 wins)**. Replacing
LDS expansion and barriers with eight-lane K32 metadata loads plus wave
exchanges reverses that result: the BF16-bit-exact rows1 family reaches
**1.423808 -> 1.381499 ms (1.03063x, 45/45 wins)** and projects **1.1518%**
selected 4K kernel-wall saving.

The admitted rows1 leaf nevertheless fails sole-payload operation completeness.
Exact qmicro row-reuse consumers move rows2/3/4 aggregate actual-layer wall
**1.430859 -> 1.828357 ms (0.78259x)**, **1.422347 -> 1.831333 ms
(0.77667x)**, and **1.425364 -> 1.881931 ms (0.75739x)** with **0/135** wins.
Exact rows512/1K/4K dual-WMMA consumers also regress the aggregate boundary
**0.643%/0.312%/0.034%**. The candidate is removed before materializer/runtime
integration because retaining the rows1 win would require a standard-T16
sidecar. Evidence:
[`first qmicro Q4 split-weight rejection`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-q4-qmicro-split-weight-rejected.json) and
[`wave-shuffled qmicro operation rejection`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-q4-qmicro-wave-meta-rejected.json).

A later materially different sole-payload dataflow resolves that rejection.
Direct compact-metadata consumers change actual c1 **1.423808 -> 1.381499 ms
(1.03063x, 45/45)** and rows2-4 aggregate **4.279782 -> 4.152054 ms
(1.03076x, 135/135)**, all BF16-bit exact. Direct dual WMMA wins at 512/1K;
at 4K, expanding only the compact coefficients once into the already-dead FFN
scratch plane changes three actual pairs **161.01965 -> 160.17212 ms (1.00529x,
32/45)**. The 128 gate/up weights therefore replace standard T16 with sole
qmicro, removing **178,257,920 bytes / 170 MiB** without a sidecar or new
workspace.

The complete-model gate is positive at every shape: AR changes
**12.36119/12.18945/12.34224 -> 12.48862/12.30596/12.46215 tok/s
(+1.031%/+0.956%/+0.972%)** and prefill changes
**400.234/391.159/385.116 -> 401.870/392.140/385.329 tok/s
(+0.409%/+0.251%/+0.055%)** at 512/1K/4K. Natural AR improves
**12.58125 -> 12.71731 (+1.082%)** with every full/train/heldout/category scope
positive. Native B1 improves **18.83154 -> 19.00472 (+0.920%)** with exact
30/30 AR and B1 trajectories, 339/393 accepted/proposed tokens, 786 target
rows, exact GPU/CPU decisions, and zero teardown. Clean commit `9c649e28d`
publishes **402.062/391.452/385.780 prefill tok/s**,
**12.4966/12.3186/12.4673 AR tok/s**, and **17.073/17.555/19.931-GiB** process
GTT at 512/1K/4K. AR beats clean llama HIP by **2.847%/2.106%/8.334%** and
remains **0.804-3.153%** behind Vulkan, so P5 remains open. Evidence:
[`sole qmicro gate/up retain`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-q4-qmicro-sole-retained.json)
and [`clean publication`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-post-qmicro-clean-publication.json).

A free-producer bound also closes cross-kernel Q6 activation fusion. The
existing planar-Q6 Q8_1/dp4a consumer is timed with Q8 quantization performed
once before all warmups and samples; best t256 is **0.993641 -> 0.990665 ms
(1.00300x, 24/45 wins)** across three actual FFN-down layers. The current family
needs **1.08087x**, so even this unrealizable upper bound projects only
**0.517 ms / 0.0400%** selected wall. No fused SiLU-to-Q8 package is warranted.
Evidence:
[`precomputed-Q8 Q6 bound`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-q6-precomputed-q8-bound-rejected.json).

The broad exact Q4 col4 singleton policy is rejected because FFN-down regresses
8.11% and projects a 0.772% selected-wall loss. The independent serial-c1
K5120/N1024 full-attention K/V shape is retained: 24 immutable actual weights
improve **16.962 -> 16.441 us/projection (1.03169x, 14/15 wins)** with BF16-bit
identity, zero new bytes, and traced resources of 56 VGPR / zero LDS / zero
scratch versus 96 VGPR control. Its complete projection is only **11.821
us/token / 0.0146%** of 4K kernel wall, below request-level timing resolution,
but it satisfies the exact non-regressive verified-subwindow rule. The
architecture-local shape map excludes native sessions, MTP, peers, and all
other Q4 singleton geometries. Evidence:
[`Q4T16 c1 col4 full-K/V`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-q4-single-col4-c1-decode.json).

The final same-payload metadata-publication screen is also closed. Unlike the
rejected Q5 FP32 coefficient path, this Q4 candidate packs four byte coefficients
per wave exchange: each eight-lane K32 group loads four scale and four min words
once instead of issuing 32 per-lane byte loads. It remains BF16-bit exact but
measures **1.434625 -> 1.435730 ms (0.99923x, 19/45 wins)** over layers 0/8/63;
layer 8 regresses **0.621%** and the post-grouped projection is **-0.0294%**.
Transient code is removed and the retained split-weight kernel keeps direct
standard-T16 metadata loads. Evidence:
[`packed Q4 metadata rejection`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-q4-packed-meta-broadcast-rejected.json).

The existing exact planar-Q6 producer-owned top-1 route also loses on the
Qwen3.8 root. Its one value/index pair per 16-logit tile removes the separate
full-logit argmax scan but adds one comparison and tile publication to every
root workgroup. On the actual 994.629-MiB K5,120/N248,320 head, the complete
boundary changes **4.591176 -> 4.603906 ms (0.99723x, 4/15 wins)** while all
FP32 logits, winner IDs, and winner-value bits remain identical. That projects
a **0.0158%** selected-wall regression, so serial AR retains the full-logit
producer plus generic argmax. Evidence:
[`Q6 root top-1 rejection`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-q6-root-top1-rejected.json).

The next distinct boundary transfers the retained fixed-hidden wave norm
premise from H1024 to H5120 without transferring its reduction association.
Each local256 thread caches 20 values, but the first three reduction levels are
reconstructed from immutable shared partials so the complete generic FP32 tree
remains exact; only the final five levels use wave32 exchanges. Across all 64
attention and 64 post-attention actual weights, standalone/add norm improve
**0.60341 -> 0.17357** and **0.63016 -> 0.18538 ms/token**; combined improves
**1.23268 -> 0.35870 ms/token (3.4365x, 15/15)** with zero BF16 mismatches.

The exact gfx1151 dense-H5120/Q4/rows1 policy improves complete graph AR
**12.05663 -> 12.23245 (+1.458%)** at 512, **11.90223 -> 12.06500 (+1.368%)**
at 1K, and **12.05091 -> 12.21721 (+1.380%)** at 4K. Natural AR improves
**12.28760 -> 12.45494 (+1.362%)** with every train/heldout/category scope
positive; native B1 is non-regressive with identical acceptance and target-row
counts. Prefill is positive at all shapes, tracked/process peaks are identical,
and no workspace or graph node is added. Generic rows>1, Q8, output-norm,
other-model, and peer-backend fallbacks remain registered. Development rows now
meet clean llama HIP across the repeated set and natural AR but remain
**2.52-5.15%** below Vulkan, so P5 remains open. Evidence:
[`fixed-H5120 norm decode`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-fixed5120-norm-decode.json).

The tracked-clean `15a2ca45b` repeat publishes
**399.836/390.793/384.712 prefill tok/s** and
**12.2099/12.0514/12.2095 AR tok/s** at 512/1K/4K. All CVs are below 0.046%,
all IDs and peaks are stable, and teardown is zero. Clean AR beats HIP by
**0.488%** at 512 and **6.095%** at 4K, but the noise-level development edge at
1K does not reproduce: clean 1K remains **0.109%** behind HIP. Vulkan remains
**4.333%/5.254%/2.854%** ahead, so P5 stays open. Evidence:
[`post-norm clean publication`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-post-norm-publication.json).

A clean selected-region rerank confirms the retained mechanism end to end:
selected kernel wall falls **80.78118 -> 79.30459 ms/token (-1.828%)** and
profiled host decode falls **84.21668 -> 82.72545 ms/token (-1.771%)**, with
exactly **934 launches/token** before and after. Fixed plus output norm now
costs only **0.29792 ms/token / 0.376%**. Q4 dual/single projections own
**62.673%** combined, while the kernel-to-host residual is **3.42086 ms/token**.
Q4 geometry, Q8 activation, and same-payload metadata publication are closed;
the rerank's requirement for a materially new sole-payload native/prefill
dataflow is satisfied by the later retained qmicro route above. Wider graph
replay is also closed. Refresh the selected-region profile after the qmicro
commit before choosing another residual-wall device-scheduling or fusion
premise. Evidence:
[`post-norm decode profile`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-post-norm-decode-profile.json).

### P6 — Exact B3 MTP

P6 is retained complete. A same-snapshot bisect localized the Q4_K_S native
trajectory difference to dense FFN gate/up+SiLU: rows2-4 used the close but
non-identical direct-BF16 qmicro rowtile, while scalar AR used the Q8_1x2
split-weight owner. Serializing that boundary made every one of 64 layer
outputs, state/KV journals, hidden seeds, and target top-1 bits exact.

The retained `dense_dual_q8_1x2_rowtile8_dp4a_bf16_bf16_out` owner preserves
the c1 thread ownership, dp4a/FMA order, wave tree, BF16 gate/up round trips,
and SiLU/store association for each row while sharing each compact weight
traversal. It is scoped only to Q4_K_S H5120/N17408 native rows2-4 and adds no
persistent or workspace bytes. Q4_K_M, rows1, rows5+, prefill, peers, and shape
misses retain their prior owners and primitive fallbacks.

The complete ten-prompt one-run gate measures true AR **13.27259**, B1
**19.92338**, B2 **23.36432**, and B3 **24.19347 tok/s**. B3 is **1.8228x**
own AR, **24.30803/24.02365** on train/heldout, and
**28.78902/20.36499/21.50307/24.04568** across code/general-English/
general-Japanese/mixed categories. All 40 mode/prompt trajectories match true
AR exactly and every GPU accept decision matches CPU. B3 exceeds the frozen
correctness-valid llama HIP row by **23.218%**; the faster Vulkan
**26.10541-tok/s** row remains correctness-invalid and is only the disclosed
stretch target. Tracked peak is **16.914 GiB** with zero teardown; task 24 owns
whole-GTT parity. Focused validation is 7/7 passed, and cache-only
`rocprofv3` names the expected rows3 kernel at 120 VGPR, 512-byte LDS, zero
scratch, and 0.462-0.465 ms on an actual layer-0 pair. Evidence:
[`exact Q4_K_S native B3`](../benchmarks/results/2026-08-17-gfx1151-qwen38-27b-q4ks-exact-native-b3.json).

B3 remains the production budget. Serial-exact B4 is a correctness route only:
on gfx1151 it measured 4.586 tok/s for Qwen3.8, 76.85% below graph-backed B3.
Do not reopen B4 until a native rows=5 graph/row schedule has a credible
full-suite break-even projection.

For B3:

1. census effective rows2-4 Q4/Q5/Q6, attention, GDN, journal, root, and graph
   symbols after P1-P5;
2. profile proposal, target verify, accept/commit, readback, and scheduler wall
   in a final child;
3. apply only dependency-aware batching/row reuse that preserves scalar logits,
   state/KV journals, reject/partial/full commit, rollback, correction, and
   dynamic-position reuse;
4. run true AR and B3 together over the complete category suite after every
   candidate; report train, heldout, categories, accepted/proposed, cycles,
   target rows, and memory.

The first closure target is exact llama.cpp HIP B3 (`19.655 tok/s` in the
opening diagnostic). The stretch target is the current Vulkan diagnostic rate
(`26.079`) while hipEngine remains exact. Acceptance cannot be changed through
prompt-, token-, or candidate-ID-specific policy.

### P7 — Residual memory and lifecycle

After compressed ownership, audit remaining bytes in this order:

1. private-c1 small-weight arena cutoff and owner count;
2. the 188-range dense decode-scratch arena;
3. dense SiLU gate-plane alias and split-GDN Conv/output lifetime reuse;
4. constrained 4K-row scratch recoloring and hidden-owner reuse;
5. graph/runtime and AOTriton workspace ownership;
6. token embedding placement only after shared AR/MTP requirements are explicit.

The gfx1100 arena thresholds are references, not copied values. On an APU,
measure tracked ownership, GTT delta, RSS, system available, transient high
water, and teardown. Reject a lower-live-memory route that raises peak or
regresses complete wall beyond the frozen guard.

The retained task-24 candidate closes that ladder without keeping any unsafe
alias. A Q4_K_S/H5120 private-c1 policy widens the immutable-weight arena to the
80-MiB inventory crossover (**849 logical allocations, two final physical
weight owners** after host embedding), enables the existing 188-range decode-
scratch owner, and maps an anonymous byte-for-byte **715,161,600-byte** Q4
embedding copy. Native B1-B3 allocates only the initial rollback state because
committed rows already live in the target session journal; serial-exact and B4
keep all row copies. At 4K only, every bulk-prefill field remains dedicated but
exact outer chunks cap scratch at 1,024 rows.

One right-sized process per shape with continuous GTT/RSS/system sampling
measures **15.275/15.710/15.727 GiB** process-GTT delta at 512/1K/4K versus the
lower llama HIP **15.785/15.816/16.004 GiB**. Tracked peak falls from the clean
Q4_K_S **15.729/16.147/18.465** to **15.063/15.481/15.494 GiB**. Sampled
prefill moves only **-0.338%/-0.306%/-0.248%**, AR moves
**+0.077%/+0.084%/+0.096%**, every final token remains 9707, and teardown is
zero. Exact B3 process GTT is **15.899 GiB** versus llama HIP's **16.358 GiB**;
all ten trajectories and GPU/CPU acceptance decisions remain exact. An adjacent
unsampled host/device embedding control is **+0.012% AR / -0.027% B3**, inside
the frozen 1% guard.

Both owner-slot and single-arena bulk aliases are rejected because they changed
the dedicated oracle logit **25.418247 -> 22-23** despite sometimes preserving
top-1. Priority recoloring, hidden reuse, direct file-backed mmap registration,
and short-session liveness are rejected for trajectory corruption. The exact
producer-folded journal is rejected for about 4% verifier-wall cost. Only the
safe owners above remain. Evidence:
[`retained G5 candidate`](../benchmarks/results/2026-08-17-gfx1151-qwen38-27b-q4ks-memory-parity-retained.json).

---

## 7. INT8 K/V lane

This lane can begin with diagnostics after P0, but runtime promotion waits until
the BF16 baseline and graph are stable.

### K0 — Re-test Qwen3.8, do not inherit Qwen3.6's verdict

Qwen3.8 has the same 24Q/4KV/D256 geometry, so the existing native writer and
split-K consumer are directly testable. It has different weights, so first run
without code changes:

- pure `int8_per_token_head`, FP32 scales, no BF16 mirror;
- complete natural/category/heldout suites at 512/8, 1K/8, and 4K/16;
- BF16 teacher forcing and full logits;
- layout, scale, retained-byte, prefill-oracle, and no-shadow audit;
- one cached kernel trace on gfx1151.

If any shape fails the per-prompt gate, pure per-head INT8 is rejected for this
model. A later 4K pass does not erase a 512/1K failure.

### K1 — Bound or eliminate the prefill oracle

Current GGUF chunk-outer execution retains one full-length BF16 K/V oracle pair
per INT8 layer until prefill completes. That erased the 9/7 map's peak savings
and projects to 7 GiB at 256K.

Prefer, in order:

1. direct INT8/hadamard attention during prefill over retained compact K/V;
2. a provably bounded shared/layer-local owner if execution order can reuse it
   without storing full-length per-layer pairs;
3. no route that keeps one full oracle pair per INT8 layer.

The old direct-streaming path's severe long-context throughput loss is a
profile target, not a reason to retain unbounded oracles. Measure 512/1K/4K
first and inspect attention tiling, scale loads, and 40-CU occupancy.

### K2 — Graph-safe cache ownership

The prior mixed-layer graph experiment page-faulted and was reverted. Add RED
coverage before a new attempt:

- cache storage/layout and frozen layer plan in the graph key;
- stable per-layer K/V/scale pointers and complete `KVLiveSpans` metadata;
- eager/graph full logits and all cache/state bytes at 512/1K/4K;
- reset/rearm, close/recreate, and cancellation/rollback;
- no graph replay of a BF16 kernel against INT8 pointers or vice versa.

A mixed route that remains eager-only is not performance-comparable with the
supported BF16 graph and cannot become default.

### K3 — Quality frontier without prompt overfit

If pure INT8 fails, use this bounded order:

1. **Fixed tail-four Hadamard group32.** This input-independent policy passed
   the 35B GGUF quality suite and has a native implementation. On a 16-full-
   attention-layer model it compresses four layers, so the maximum retained KV
   saving is about 12.5%; accept the modest ceiling explicitly.
2. **All-layer Hadamard group32 screen.** Run host/native format emulation first.
   Implement only if the full train-suite screen has a credible quality margin
   and projected bytes justify the added scale/transform work.
3. **Frozen mixed-layer map.** Select only from the six declared train prompts
   using worst-prompt quality, freeze the map, then evaluate the four untouched
   category heldouts and full suite. A map selected from the one failing prompt
   cannot be promoted.

K0-K3 closed on 2026-08-15 without a supported Qwen3.8 route. Pure per-token/
head INT8 passes native 512/8 but rejects at 1K/8; fixed tail-four Hadamard
rejects at 512/8 on one prompt. All-layer Hadamard passes its host screen and
native 512/8, then rejects native 1K/8. The only frozen map was selected from
six train prompts as BF16 ordinals `[3,6,8,9,10,12,13,14,15]` and Hadamard
INT8 ordinals `[0,1,2,4,5,7,11]`; its one allowed host transfer and native
512/8 pass, but native 1K/8 rejects at aggregate mean/max KL
**0.053384/5.173312** because `mixed_v1` reaches **0.586860/5.173312**, despite
100% top-1. Every native audit reports the exact fixed map, zero persistent
BF16 mirror bytes, and zero retained prefill-oracle buffers. A cached gfx1151
CPU-reference trace confirms the expected 24Q/4KV INT8 split-K producer and
reducer with zero scratch. The shape stop rule
therefore prevents every 4K, graph, MTP, capacity, and long-context promotion
gate. Temporary all-layer/frozen runtime surfaces were removed; BF16 remains
the only supported campaign route. Full evidence is in
[`2026-08-15-gfx1151-qwen38-27b-int8-kv-quality-rejected.json`](../benchmarks/results/2026-08-15-gfx1151-qwen38-27b-int8-kv-quality-rejected.json).

Do not reopen recent-token two-arena tails, key-only, block16, clipping, or
simple prefix masks without a materially new representation signal; those
families are already rejected in `KVCACHE.md`.

### K4 — Promotion gate

For an explicit supported 512/1K/4K INT8 mode require:

- all K0 quality rows pass;
- native writer, prefill, decode, and graph symbols execute on gfx1151;
- no persistent BF16 shadow and bounded transient peak below BF16;
- graph AR at all three shapes, deterministic candidate repeats, and clean
  lifecycle;
- full B3 transaction and candidate-MTP exactness versus its own candidate AR
  if MTP is advertised;
- retained and whole-GTT memory savings reported in bytes and percent;
- performance neutral or better for default promotion.

Then extend separately to 32K and 128K natural quality and a real 256K capacity
row before making a long-context claim.

---

## 8. Experiment bounds and stop rules

| Task class | Bound | Continue / keep | Stop / reject |
| --- | --- | --- | --- |
| Route/census | One clean route matrix plus one focused repair | Effective route and ownership become explicit; exact correction may be retained immediately. | Unknown route remains a blocker; no kernel tuning on it. |
| Existing-layout transfer | Actual weights, c1/rows2-4/M512/1K/4K, one full-model A/B | All operation classes pass; memory improves; complete wall is non-regressive. | Any missing operation, shadow allocation, correctness failure, or complete-shape regression. |
| Kernel leaf | At most 3 predeclared variants and one tuning dimension | >=1.10x leaf and >=1% request projection, or an already-measured exact small win. | Correctness/resource failure, <1% projected ceiling, or no robust paired win. |
| Full-model A/B | Best admitted leaf only; 1 warmup + 3 development or 5 final paired blocks | Correctness and all shape/quality/memory guards pass; promote exact non-regressive wins. | Do not rescue with an unplanned compound or favorable prompt subset. |
| MTP | Full train+heldout+category suite with same-harness true AR | Absolute B3 and every split improve or stay within frozen guard; IDs/accept exact. | Any heldout/category loss, invalid AR denominator, or prompt-conditioned policy. |
| INT8 quality | 512 -> 1K -> 4K transfer before long-context | Every prompt passes KL/top-1; no-shadow/bounded transient. | A failed earlier shape remains a rejection; do not average it away. |

Before coding, each worklog records current owner, time/share, byte ownership,
credible saved wall/bytes, experiment budget, correctness gate, accept threshold,
and revisit trigger. Ask before repeating any validation expected to exceed five
minutes when equivalent evidence already exists.

---

## 9. Anti-rabbit-hole list

Do not start or repeat these without a new measured complete-owner premise:

- tuning the current dual-layout fallback before P1 adjudicates sole T16;
- copying gfx1100 rocBLAS solution IDs, tile thresholds, or row policies without
  a gfx1151 screen;
- copying 0.8B 1024/3584 or 8Q/2KV capability keys to H5120/24Q/4KV;
- large-LDS pack8 WMMA, blanket non-temporal loads, generic wave64, or blind
  thread/tile sweeps;
- graph splitting/upload/parent-child composition merely because launch count is
  high;
- B4 serial-exact benchmarking as a performance candidate;
- fixed-prompt acceptance tuning, candidate-ID reranking, or any input-aware
  shortcut;
- duplicate resident layouts to make one width fast;
- temporal-tail INT8 K/V or prompt-selected layer masks;
- a long-context INT8 capacity claim based on repeated token 9707.

---

## 10. Milestones

| Milestone | Required result |
| --- | --- |
| **G0 Baseline frozen — complete 2026-08-15** | Clean three-engine 512/1K/4K + natural AR/B3 + matched memory, route census, and semantic ledgers. |
| **G1 Single-layout parity — complete 2026-08-15** | Q4/Q6/Q5 compressed ownership is independently gated; alternate/duplicate bytes are zero; complete shapes and B3 are correct. |
| **G2 Prefill win — complete 2026-08-16** | Clean `a06589f34` reaches 399.031/391.276/385.330 tok/s and beats both llama backends at all three shapes with retained correctness and zero teardown. |
| **G3 AR win — complete 2026-08-16** | Clean Q4_K_S beats both backends at all three shape-AR rows and natural true AR. |
| **G4 Exact MTP win — complete 2026-08-17** | Exact Q4_K_S full-suite B3 is 24.19347 tok/s / 1.8228x own AR, 23.218% above correctness-valid llama HIP, with all split/category disclosures and GPU/CPU acceptance agreement. |
| **G5 Memory win — complete 2026-08-17** | Clean post-commit refresh at `20e5106da` confirms process GTT **15.275/15.710/15.727 GiB** at 512/1K/4K and **15.899 GiB** at exact B3, below every lower valid llama row with exact outputs and zero teardown. |
| **K-G1 INT8 working-set support — blocked 2026-08-15** | No bounded representation passes the required native 512 -> 1K quality transfer; BF16 remains supported. |
| **K-G2 INT8 long support** | Independent 32K/128K natural quality and real 256K capacity/runtime gates pass. |
| **G6 Closure — complete 2026-08-17** | G2-G5 pass on the one clean retained source `20e5106da` with exact outputs, acceptance, and zero teardown; the full pytest tier passes with only host-inapplicable gfx1100-only skips; INT8 status is published accurately; campaign refactor debt is explicit seams with triggers (the shared 80-MiB arena env seam still awaits the pending gfx1100 cumulative confirmation); rollups are committed and pushed. Evidence: [`G6 closure`](../benchmarks/results/2026-08-17-gfx1151-qwen38-27b-q4ks-g6-closure.json). |

A milestone is blocked, not complete, if one column fails. Wins in prefill, AR,
MTP, or memory do not hide another failure.

---

## 11. Publication and update protocol

Every retained performance unit must include:

1. a unique immutable `worklog/entries/` file;
2. a compact JSON artifact under `benchmarks/results/` with canonical
   provenance and raw-sample hashes;
3. the current row and review date in `benchmarks/README.md`;
4. a dated `benchmarks/CHANGELOG.md` old -> new row with delta, reason, and
   artifact;
5. `docs/REFACTOR.md` for every temporary rollback/env/duplicate route;
6. this plan's status/scoreboard update;
7. explicit staging, staged diff review, atomic commit, and push.

Update [`PLAN.md`](PLAN.md) only if an architectural phase or invariant changes.
This campaign currently exercises existing backend/plugin, single-layout,
`KVLiveSpans`, and speculative-cycle contracts, so opening this plan does not
change the architectural source of truth.

---

## 12. Post-closure research agenda — additional INT8 / dtype (closed)

Status: **Closed research, nothing retained.** The G1-G6 campaign is closed on
BF16. This lane explores whether additional INT8 / dtype reduction makes sense
for hipEngine, prompted by the external `syv-ai/qwen38-27b-rtx3090` vLLM
reference (W4A16 AutoRound body, fp8 K/V, lm_head/embed_tokens int8, fp16
recurrent state, int8 activations, cheap MTP drafts, KVarN 4/2-bit K/V on an
RTX 3090). That stack is NVIDIA tensor-core / vLLM on GDDR6X (~936 GB/s); its
numbers are **not** directly comparable to ours. The only transferable signal
is where an idea reduces bytes moved on our memory-bandwidth-bound gfx1151
decode or on our compute-heavy prefill/concurrency paths. Every row below is a
T1/T2 production-profile arithmetic change, so none can ride the strict-parity
path; each needs the full profile gate, a registered strict fallback, and the
complete multi-prompt natural suite before any retention.

**All six R-rows are now assessed and nothing was retained.** R1 re-closed
native INT8 K/V on the Q4_K_S head (rejected at 512/8), R2's fp16 recurrent
state is quality-neutral but deferred for c=1 (GDN gate is 1.43% of decode
wall), R3's lm_head/embed int8 is a non-starter (GGUF already stores both below
int8), R4's cheaper MTP drafts are deferred (draft stage is 2.08% of decode
wall), R5's KVarN 4/2-bit K/V is not pursued (context not binding on this
hardware), and R6's int8-activation prefill is not pursued (no dispatchable
W8A8 dense prefill path; decode is already W8A8).

### R1 — gfx1151 INT8 K/V revalidation (start here)

Prior state: K0-K3 closed 2026-08-15 below K4 on model-level correctness — pure
`int8_per_token_head` passes native 512/8 but rejects at 1K/8; fixed tail-four
Hadamard rejects at 512/8; all-layer Hadamard passes its host screen and native
512/8 then rejects native 1K/8; the one frozen mixed map reaches
mean/max KL **0.053384/5.173312** at 1K/8 (`mixed_v1` 0.586860/5.173312)
despite 100% top-1. BF16 is the only supported route. Full evidence in
[`2026-08-15-gfx1151-qwen38-27b-int8-kv-quality-rejected.json`](../benchmarks/results/2026-08-15-gfx1151-qwen38-27b-int8-kv-quality-rejected.json).
Rejected families (temporal-tail two-arena, key-only, block16, clipping, simple
prefix masks) are listed in `KVCACHE.md` and are not reopened without a
materially new representation signal.

**Executed 2026-08-18 on the current Q4_K_S head** (merged `ff7293098`,
`scripts/qwen35_native_mixed_kv_suite.py`, `--require-no-bf16-mirror`, 512/8
shape stop rule): both natively-supported INT8 representations reject at the
first shape. Pure `int8_per_token_head` (fp32 scale, all 16 layers) is
`rejected_correctness` because the layout audit retains a persistent 48 MiB
BF16 mirror (no-mirror gate fails, logits byte-identical);
`tail4_hadamard_group32` (fp16 scale, layers 12-15) has zero mirror but top-1
agreement drops to **0.889** on `general_en_explain` (< 0.90 floor; max KL
only 2e-4). Native INT8 K/V on gfx1151 therefore stays closed on the current
head too. Evidence:
[`2026-08-18-gfx1151-qwen38-27b-r1-int8-kv-revalidation-512-8.json`](../benchmarks/results/2026-08-18-gfx1151-qwen38-27b-r1-int8-kv-revalidation-512-8.json).

The FP8-vs-INT8 scale-scheme angle (below) is the only path that could reopen
this lane; it requires building an FP8 (E4M3) K/V path on gfx1151, which does
not exist. Decode-traffic framing is moot until a quality-passing
representation exists.

Reopen only with a new angle, then re-run the K0-K3 shape stop rule (512 -> 1K
-> 4K transfer) before any 4K / graph / MTP / capacity claim:

1. **FP8 vs our INT8 scale scheme.** The external reference uses fp8 (E4M3)
   K/V; we use `int8_per_token_head` with FP32 scales or group32 Hadamard. A
   same-artifact comparison of scale scheme and error profile may be the new
   representation signal the previous closure lacked.
2. **Decode-traffic framing on gfx1151.** The Halo has 128 GB unified memory,
   so context capacity is not the win on this backend — the win would be
   reduced per-step K/V bytes on a memory-bound decode. Gate on decode tok/s
   delta plus KL/top-1, not on capacity alone.
3. **Reuse the gfx1100 template.** The gfx1100 INT8 admission (artifact-scoped
   by full-file SHA-256, FP32-scale, KL 1.04e-5 / 1.35e-4, 100% top-1, 126K
   context, W7900 reaches the 256K model limit) is the evidence and admission
   pattern to replicate, not the older gfx1151 rejection.

### R2 — fp16 recurrent state (worth testing; measure the quality tradeoff)

Today the DeltaNet recurrent state and conv state are allocated **FP32**
(`DType.FP32` over `time_step_rank x ssm_state_size x value_dim`, 48/64
layers, read + written every decode step). Halving to fp16 halves state
traffic on a memory-bound decode — the same lever that unblocked the external
stack's concurrency (their fp32 -> fp16 state was PPL-free; they deliberately
kept fp16's 10 mantissa bits, not bf16's 7).

This is a production-profile arithmetic change (state math is not byte-identical
in fp16), so it cannot be strict-parity. Measure strict-teacher mean/tail/max
KL + top-1 by category/shape/transition, the BF16-relative gate, and same-
schedule determinism, and register an FP32 strict fallback. Also decide fp16 vs
bf16 for our accumulators explicitly. If it clears, it applies to both gfx1151
and gfx1100 decode.

**Measured 2026-08-18 (host-side emulation, real Qwen3.8 layer-0 arrays).** The
state is `float*` in every GDN device kernel (the `fp16_lowp` variants change
only output dtype), so a full change spans the ~40-kernel GDN family. A
faithful host-side replay (`replay_gdn` math, real `ssm_norm`/`ssm_dt_bias`/
`ssm_a`, eps=1e-6) on 64 captured gfx1151 positions shows fp16-state
recurrent-output drift rms **4.72e-6** = **0.05x** the device's own
already-accepted bf16 output rounding (1.02e-4), while bf16-state drift is
**539x** (rms 5.50e-2) — validating the external's explicit fp16-not-bf16
choice. Decode-traffic framing: the GDN recurrent gate is **1.43% of decode
wall** (1.12/78.5 ms per token), so fp16 state bounds c=1 decode gain near
0.5-0.7% on gfx1151; the external's material win (concurrency via 150 MB/
request state) is outside our c=1 scope. Evidence:
[`2026-08-18-gfx1151-qwen38-27b-r2-fp16-recurrent-state-quality.json`](../benchmarks/results/2026-08-18-gfx1151-qwen38-27b-r2-fp16-recurrent-state-quality.json).

**Decision (c=1 lane):** do not invest in the full multi-kernel fp16-state device change
for the c=1 campaign; the quality tradeoff is favorable but the decode gain is
small and the real value is concurrency. Revisit (with a device fp16-state GDN
variant + full natural-suite KL/top-1 profile gate) only if a multi-request /
concurrency target is adopted. FP32 state remains the strict default.

**Built and audited 2026-08-20 (the c=N lane adopted the concurrency target).**
The state dtype is frozen per runner and all FP16 leaves are explicit registry
variants with FP32 strict-storage fallbacks. Decode covers scalar, segments,
indexed-singleton, and the gfx1151 rows>=8 shared-statecache24 owner. Packed
prefill now uses compact normalized-segments wave32-XOR: each indexed state
stays FP32-register-resident across its segment and converts at a chunk
boundary. The earlier per-token decode-order fallback failed c4/c8 prefill
scopes and is retained only as a generic diagnostic fallback.

The original 450-row artifact is a **c1 numerical capture**, not a complete
production-profile gate. Its actual global mean/p95/p99/max KL is
**5.28e-5/2.80e-4/5.81e-4/1.43e-3** with **449/450 = 99.78%** top-1; the
previous prose accidentally quoted the code-category slice. The repaired
packed c4/c8 gate covers **1,170** full-vocabulary rows including natural
static widths, category-diverse p512, sparse slots, neighbor substitution, row
permutation, and c8->c4->c2->c1 transitions. All hard scopes pass at global
mean/p95/p99/max KL **1.02e-4/3.19e-4/1.05e-3/3.2866e-2** and
**1168/1170 = 99.829%** top-1, with three exact repeats and exact isolation.
One long-c8 row requires and passes manual review: KL 0.03287 (<0.05), same
wide-margin top-1, and improved teacher NLL. These artifacts remain
`profile_qualification_claim=false`: runtime-manifest, BF16-relative/task,
serving SLO-goodput, and gfx1100 gates are still open. Evidence:
[`c1 capture`](../benchmarks/results/2026-08-20-gfx1151-qwen38-27b-r2-fp16-state-production-gate.json) and
[`packed gate`](../benchmarks/results/2026-08-20-gfx1151-qwen38-27b-r2-fp16-state-packed-production-gate.json).

**c=N roofline (2026-08-18 follow-up — the concurrency value is quantified, not
just "outside scope").** We do target c=N, so the multi-stream gain matters and
it grows with concurrency. State geometry (from the runner allocation):
recurrent state = 48 layers x 3.0 MiB = **144 MiB/slot** (time_step_rank 48 x
ssm_state_size 128 x value_dim 128 x 4 B), conv state = 48 x 0.16 MiB =
**7.5 MiB/slot** (linear_qkv_width 10240 x conv_kernel 4 x 4 B) => **151.5 MiB/
slot FP32**, fp16 => 79.5 MiB (saves **72 MiB/slot**). Per-step decode state
traffic = **0.296 GiB/slot** (every GDN recurrent layer reads+writes its state
per token, confirmed in `gdn.hip`). In batched decode the ~15.8 GiB of weights
are read **once per step and shared across rows**, but the state is per-slot, so
the state's share of bytes grows with c. Bandwidth-bound roofline (fp16 halves
state traffic): c1 0.8%, c2 1.6%, c4 3.1%, **c8 5.9%, c16 10.5%**, c32 17.4%,
c64 25.8% decode gain. Capacity is not a lever on gfx1151 (72 MiB/slot x 64 =
4.6 GB vs 128 GB unified) — the win is purely bandwidth, unlike the external's
24 GB VRAM story. The old **c1 prefill** roofline correctly treated state as
register-held, but it did not describe the former packed c>N fallback, which
loaded/stored state every token. The repaired packed compact-segments route is
register-held again; the clean bracket still measures FP16 aggregate prefill
wins of **+1.65% at c4 / +3.31% at c8** because packed chunk boundaries retain
per-slot state traffic. Caveats: the c>1 numbers are a **bandwidth-bound
roofline, not a measurement**; MLP/attention GEMM work grows xN with concurrency and may turn compute-bound
before c16-32 (no retained gfx1151 Qwen3.8 c>1 decode topline existed to pin the
crossover — a c8 `rocprofv3` decode profile was run to check it). **Measured c8
result (2026-08-18): the crossover question is currently unanswerable because
c8 dense decode is not on the GEMV path at all** — `gguf_packed_ar_rocprof.py`
c1/c8 shows the dense Q4 projections at rows=8 fall back to **WMMA prefill
kernels** (340 dispatches, 331.7 ms = 81.1% of the 408.9 ms step; native batch
owners cover rows 1-4 only), making the c8 step 5.52x slower than c1 and
aggregate throughput only 1.45x c1 (19.6 vs 13.5 tok/s) vs the ~8x a proper
batch path would give. `gdn_recurrent` scales cleanly (1.78 -> 8.56 ms = 4.82x
for 8x rows; ~11.1% of an uncontaminated c8 step, matching the 11.7% roofline
byte-share), so the fp16-state gain estimate is structurally consistent but
only harvestable after a native rows>=8 GEMV owner exists. That c8 decode-path
gap is itself a much larger win than fp16 state on this lane. Full
calculation and c8 profile: worklog entry
`20260818T120724.814158Z-lhl-qwen38-gfx1151-r2-cn-fp16-state-roofline-c6c76d.md`
and
[`2026-08-18-gfx1151-qwen38-27b-r2-cn-roofline.json`](../benchmarks/results/2026-08-18-gfx1151-qwen38-27b-r2-cn-roofline.json) +
[`2026-08-18-gfx1151-qwen38-27b-r2-c8-decode-profile.json`](../benchmarks/results/2026-08-18-gfx1151-qwen38-27b-r2-c8-decode-profile.json).

**Measured retained opt-in result (2026-08-20).** A clean
FP32-control-A -> FP16 -> FP32-control-B bracket on commit `d73c08a0e`
(p512/tok9707/64 decode steps, one warmup + three measurements per process)
uses the pooled six-sample FP32 median. Aggregate decode is
**12.9005 -> 13.0204 tok/s (+0.93%)** at c1,
**42.5856 -> 43.8535 (+2.98%)** at c4, and
**57.2905 -> 59.0106 (+3.00%)** at native c8. Per-request c4/c8 is
**10.6464 -> 10.9634** and **7.1613 -> 7.3763 tok/s**; these are now labeled
separately rather than presenting per-request values as aggregate throughput.
Aggregate prefill is **377.31 -> 379.09 (+0.47%)**, **307.16 -> 312.24
(+1.65%)**, and **246.26 -> 254.41 (+3.31%)** at c1/c4/c8. The eight-session
packet saves **1,234,698,240 tracked bytes (1.15 GiB)**. FP16 remains behind
the env flag and FP32 remains the strict/public default pending the open gates
above. Evidence:
[`repaired production bracket`](../benchmarks/results/2026-08-20-gfx1151-qwen38-27b-r2-fp16-state-repaired-production.json); the former
[`fair c=N artifact`](../benchmarks/results/2026-08-20-gfx1151-qwen38-27b-r2-fp16-fair-cn-production.json)
is superseded.

**Public-default serving decision (2026-08-20): rejected.** The tracked-clean,
predeclared localhost-Uvicorn static c1/c8 screen uses Q4_K_S identity and the
same compact-peer route. FP16 c1 exact/SLO-goodput is non-regressive
(**8.2121 -> 8.2434 tok/s, +0.38%**), but both modes fail the c8 ITL-p99 SLO
(**0.8532/0.8287 s > 0.5 s**) and exact c8 server throughput improves only
**13.0999 -> 13.2738 tok/s (+1.33%)**, below the 3% public-default target.
Completed responses remain exact, native routing/ownership/memory recovery
pass, and FP16 saves 1.08 GiB tracked peak. The complete canonical packet was
not run because neither binding screen can pass; relaxing the SLO afterward
would be benchmark gaming. FP16 remains opt-in and FP32 stays public default.
Evidence:
[`serving rejection`](../benchmarks/results/2026-08-20-gfx1151-qwen38-27b-fp16-state-serving-screen-rejected.json).

### R3 — lm_head int8 / embed_tokens int8 (see what it looks like)

Qwen3.8 has untied embeddings; the external repo requantizes the two ~2.5 GB
bf16 matrices (lm_head + embed_tokens) to int8 group-128 in place (~0.6%
round-trip, ~1 IFBench point on W4A16). For us the lm_head is a large decode
GEMM (K6144 -> 248k vocab), so int8 reduces per-step decode bytes.

Quality caveat: our Q4_K_S weights are already more lossy than their W4A16, so
the marginal cost of int8 lm_head may differ and must be measured against our
KL/top-1 gate — lm_head directly determines logits, so top-1 drift is the
primary risk. Measure decode tok/s delta plus the natural-suite KL/top-1 and
BF16-relative rows before deciding.

**Assessed 2026-08-18 — non-starter on the Q4_K_S GGUF path.** Our GGUF
already stores `output.weight` as **Q6_K** (6.5625 bits, 1.043 GB) and
`token_embd.weight` as **Q4_K** (4.5 bits, 0.715 GB) — both already *below*
int8 (8 bits, 1.271 GB) in size. The external's int8 premise was bf16 -> int8
(-50% on their 2.5 GB heads); for us int8 would *increase* bytes (lm_head
+22%, embed +78%), so it does not transfer. The registered
`w8a16_lm_head_argmax_rows_bf16` kernel is wired only for the PARO runner; the
GGUF decode path runs lm_head via `launch_gguf_linear` on the Q6_K root.
lm_head is a measurable decode cost (4.63 ms/token, ~5.9% of decode wall from
the AR16 rocprof trace), so the only direction that could cut it is a *lower*
head quant (Q4_K, ~0.72 GB) — a separate top-1-drift-risky experiment, not
this int8 premise. Evidence:
[`2026-08-18-gfx1151-qwen38-27b-r3-lmhead-embed-int8.json`](../benchmarks/results/2026-08-18-gfx1151-qwen38-27b-r3-lmhead-embed-int8.json).

### R4 — cheaper MTP drafts (exploring)

We are at MTP-3 **23.853 / 1.7845x AR** (topline; newest natural gate
24.500 / 1.831x). The external stack made 3 drafts pay off by cutting per-draft
cost (int8 drafter module + a 40k-token truncated draft-vocab head, ~1 ms vs
~3 ms per draft). Two separate ideas map to us: (a) a compressed/int8 drafter,
and (b) a truncated draft-vocab head. Known blocker to re-qualify: gfx1100 INT8
K/V previously broke MTP (B3 at only 0.6423x AR), so **any** dtype/KV change in
this lane must re-run the full B3 transaction / candidate-MTP exactness gate
versus its own candidate AR. Acceptance economics must be validated on the full
multi-prompt suite — never a single fixed prompt.

**Assessed 2026-08-18 — deferred on the Q4_K_S topline.** On the retained
exact B3 topline the entire draft/proposal stage is only **2.08% of decode
wall** (0.206 s of 9.920 s; 2.37 ms/cycle), while target verify is **97.41%**
(9.663 s) — so even a fully-free draft buys ~2% B3, and a truncated draft-vocab
head (which only touches the draft lm_head, ~57% of a ~2% draft stage) would
recover **well under 1%** of decode wall while risking acceptance loss that
could inflate the dominant verify term. The mechanism **already exists**
(`mtp_draft_vocab_cap` / `vocab_cap` in `Qwen35GGUFResidentMTPDraftRunner`,
retained at cap=65536 for Qwen3.6-PARO; `draft_vocab_cap` in `mtp_nextn.py`),
but the Qwen3.8 topline draft path is the dense `NextN` executor
(`Qwen35GGUFNextNExecutor`, no vocab cap, full 248320 vocab). Revisit only
under a concurrency/throughput target where the draft share grows. Evidence:
[`2026-08-18-gfx1151-qwen38-27b-r4-cheaper-mtp-drafts.json`](../benchmarks/results/2026-08-18-gfx1151-qwen38-27b-r4-cheaper-mtp-drafts.json).

### R5 — KVarN 4/2-bit K/V (mention; lower priority)

The external port (Huawei CSL KVarN: Hadamard + variance normalization + 4-bit
keys / 2-bit values per 128-token tile, ~840 B/token/layer) exists to fit 262k
context on a 24 GB card. On gfx1100 W7900 we already reach the 256K model
limit with INT8 K/V, so 4/2-bit buys little there; on the XTX it could extend
126K -> ~200K, and on gfx1151 context is not the constraint (128 GB). Not
something we necessarily want; if ever pursued, gate on the same KL/top-1 curve
plus bytes-vs-quality versus our own INT8, not on the external's capacity
headline.

**Decided 2026-08-18 — not pursued.** This is a mention-only item, and every
capacity motivation fails on this campaign's hardware. gfx1100 W7900 is already
at the **256K model limit** with INT8 K/V (model-bound, not memory-bound), so
4/2-bit adds zero headroom. gfx1151 (this campaign) is not context-constrained
(128 GB unified), and R1 just re-closed native INT8 K/V on the current Q4_K_S
head (both representations reject at 512/8; pure int8 keeps a 48 MiB BF16
mirror, tail4 Hadamard is mirror-free but top-1 0.889 < 0.90). KVarN 4/2-bit is
strictly *lower* precision than INT8, so it would face an even harder quality
gate with no capacity benefit here. The only backend with any pull is the XTX
(INT8 126K -> ~200K extension), which is outside this gfx1151 campaign. If ever
revisited, gate on the same KL/top-1 curve plus bytes-vs-quality versus our own
INT8, never on the external capacity headline, and only on a backend where
context capacity is actually binding. Evidence:
[`2026-08-18-gfx1151-qwen38-27b-r5-kvarn-4-2bit-kv.json`](../benchmarks/results/2026-08-18-gfx1151-qwen38-27b-r5-kvarn-4-2bit-kv.json).

### R6 — int8 activations / MLP (prefill + concurrency poke)

Rationale to test: we do see a perf boost on INT8 and it helps concurrency, and
prefill (rows > 1 GEMMs) is compute-heavy — so this may help prefill even
though our c=1 decode is memory-bound. The external stack's W4A8 Marlin path
(int8 tensor cores, with a sign-bug workaround) only paid at 40-64 concurrent
and bought nothing at batch size 1; PPL cost was +2.2% (whole MLP) to +3.7%
(all linear). On RDNA3 the int8 rate advantage differs from Ampere tensor
cores, and activation quantization violates arithmetic equality by design, so
this is a poke, not a retention: measure prefill tok/s + KL/top-1 for int8
MLP (gate/up, whole MLP, all-linear) on gfx1151 and retain only if a full
profile gate passes with no input-overfit.

**Assessed 2026-08-18 — not pursued: no dispatchable int8-activation prefill
path exists for the dense Q4_K_S MLP on gfx1151.** The dense MLP prefill
(rows > 1 GEMMs) dispatches `pack8_dual_wmma_prefill_bf16_bf16_out` — BF16
WMMA with Q4_K weights, **no activation quantization**. The only in-tree int8
machinery for the dense `linear_pair_silu` pair is **decode-side**: the Q8_1
activation quantizer (`gguf_q4_k_quantize_bf16_q8_1`) feeding the Q8_1x2 dp4a
GEMV owners (`dense_dual_q8_1x2_split_weight_dp4a` serial c1, `rowtile8_dp4a`
native rows 2-4) — so decode is *already* int8-activation (W8A8) and the
"perf boost on INT8" this item references is already captured on the memory-
bound path where it matters. The only W8A8 *prefill* prototype in-tree
(`gguf_q4_k_q8_1_selected_prefill`) is a gfx1100 MoE selected-expert
diagnostic microbench, not wired into dense dispatch; the sole registered
q8_1 prefill variant on gfx1151 (`selected_mmq_i128_j128_k256_q8_1_ds4_`
`prefill_compact`) is MoE-only (iq3_xxs/iq4_xs), never the dense Q4_K_S key.
Building a dense W4A8 prefill is a full new-kernel unit — a prefill-scale
(rows ≫ 4) activation quantizer plus a Q4_K→Q8_1 / dp4a dual gate/up prefill
kernel, registered against `linear_pair_silu`/`gguf_q4_k` with a strict
unfused fallback, and gated for production retention by strict-teacher
mean/tail/max KL + top-1 by category/shape/transition, BF16-relative and task
gates on the full multi-prompt suite (activation quantization violates
arithmetic equality by design). Payoff is real but secondary: prefill is
compute-bound (370-380 tok/s at 512/1K/4K, already beating llama.cpp
352-368), but the campaign's topline bottleneck is **decode** (~13 tok/s,
memory-bound) — which is already W8A8 — and the external's W4A8 only paid at
40-64 concurrency (+2.2-3.7% PPL), a regime outside this campaign's c=1-8
scope on a unified-memory part. Concrete blocker to ever revisiting: no dense
W4A8 prefill kernel and no prefill-scale activation quantizer, plus a binding
KL/top-1 gate on activation quantization. Evidence:
[`2026-08-18-gfx1151-qwen38-27b-r6-int8-activation-prefill.json`](../benchmarks/results/2026-08-18-gfx1151-qwen38-27b-r6-int8-activation-prefill.json).

### Gate discipline (applies to R1-R6)

- Production-profile T1/T2 arithmetic changes only; strict-parity paths stay
  at FP32/BF16 defaults with a registered strict fallback.
- Required per change: exact control/ownership, same-schedule determinism,
  strict-teacher mean/tail/max KL + top-1 by category/shape/transition,
  BF16-relative/task gates, and the full multi-prompt natural suite.
- Evidence policy: model + quant + workload shape + host + hardware + exact
  command + result + correctness gate; retention updates `benchmarks/README.md`
  and `benchmarks/CHANGELOG.md` with an artifact under `benchmarks/results/`.

### Working order

1. R1 — gfx1151 INT8 K/V revalidation (stated start).
2. R2 — fp16 recurrent state + quality tradeoff.
3. R3 — lm_head / embed_tokens int8.
4. R4 — cheaper MTP drafts (deferred, see assessed note above).
5. R6 — int8 activations poke (prefill / concurrency) (decided: not pursued — see assessed note above).
6. R5 — KVarN 4/2-bit K/V (decided: not pursued — see assessed note above).
