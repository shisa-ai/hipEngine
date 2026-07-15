# GGUF Prefill Optimization

Last updated: 2026-07-16.

Status: `SOL-R5` and the architecture-local gfx1100 optimization pass are
published. gfx1151's bounded `SOL-R6` production refresh is retained through
64K while repeated 128K remains blocked by the residual gfx11
scheduler/firmware lifecycle failure; the final W7900 defaults-only 1+3 rollup
is public through 128K.
`GPF-1` exact value-column tiling
and `GPF-2A` non-resident wave sharding are rejected. Register-resident
tree-reduced `GPF-2B` is fast but failed its historical natural greedy-
trajectory gate; it now requires fresh evaluation under the 2026-07-15
peer-aligned contract rather than retroactive relabeling. Register-resident
ordered `GPF-2C` retains byte identity but
is 12.98%-14.58% slower than fused at 512/1K/4K. Scalar-exact, LDS-resident
`GPF-2D` now passes its clean six-case byte-exact state matrix and balanced
512/4K wall gate. It improves clean prefill by 79.09%/68.44%, with all timed
tokens exact. The ten-prompt gate is also exact across all 250 checked logits
and every timed trajectory, with decode +0.023%. GPF-2D is accepted for a
gfx1151-scoped `auto` promotion and became that architecture's scoped default;
gfx1100 remained fused until the independent transfer gate below. A clean max-context six-shape
stress run confirms the automatic route from 512 through 128K. Exact Q4T16
shared-activation `GPF-3A` also passes its clean full-model gate: 512/1K/4K
prefill improves **747.764/804.150/687.676 -> 771.027/823.624/701.042 tok/s**
with byte-identical logits and trajectories and neutral aggregate decode.
`shared_x` became the gfx1151-scoped automatic route; gfx1100 stayed baseline
until its independent transfer gate.
A clean selector-unset four-run confirmation at promoted commit `431fe1e4`
reproduces **774.653/823.149/701.389 tok/s** with stable IDs; it is a focus
diagnostic, not the final right-sized memory rollup. `GPF-2E` removes
GDN Q/K/V scratch materialization and the eightfold duplicate Q/K norm work on
the production 4-K-head/32-V-head shape. Its clean current-default/direct A/B
improves 512/1K/4K prefill **776.428/825.319/700.824 ->
823.093/889.209/744.577 tok/s** (**+6.01%/+7.74%/+6.24%**). The six-case
full-model matrix and all 250 natural logit transitions are byte-exact; every
timed decode trajectory matches and weighted decode is **+0.075%**. GPF-2E is
now the gfx1151-scoped automatic route; gfx1100 remained fused until its
independent transfer gate. A clean
selector-unset focus confirmation at `b8949477` reproduces
**821.755/897.160/750.896 tok/s** at 512/1K/4K with stable IDs. That first
clean right-sized 1+3 publication window recorded
**819.641/893.266/752.308/640.096/540.850/387.334 tok/s**. Scoped two-wave
Q8T16 GPF-5A then refreshed 512-64K to
**889.904/919.598/762.940/648.948/546.296 tok/s** while restoring the
production wrapper at 128K. The post-GPF-5A llama.cpp parity tranche is complete. LCP-1/LCP-D1, LCP-2A,
scoped LCP-M2/LCP-3, and LCP-4A/LCP-4B now publish current selector-unset
512/1K/4K/32K/64K prefill of
**1294.885/1358.342/1365.720/1034.845/796.083 tok/s** and graph decode of
**49.041/51.623/52.422/43.572/37.622 tok/s**. All 15 measured IDs are `9707`,
tracked memory is unchanged, and maximum prefill/decode stdev over median is
**0.187%/0.049%**. Stream-ordered metadata stays automatic only through 4K;
LCP-3 remains scoped through 64K; and the faster 64-thread router-select
primitive remains rejected by full-model state.

Repeated 128K production is not published. The current automatic one-queue
route completes warmup at **509.708 tok/s** and then enters the same low-power
measured-pass-1 stall. Explicit metadata-off/router-512 and
`HSA_ENABLE_SDMA=0` full controls reproduce it, proving neither recent LCP
launch policy is causal nor SDMA disable a reliable fix. The earlier complete
`71e61524` row is historical and is not carried into the current topline.

The independent W7900 transfer gate at clean `bc5600e2` now admits GPF-2E
`chain_lds32_direct` and GPF-3A `shared_x` as gfx1100 automatic routes. GPF-2E
improves clean balanced 512/4K prefill **649.131/677.888 ->
1291.225/1401.330 tok/s** (**1.9892x/2.0672x**) and passes all **250/250**
natural logit transitions with exact timed trajectories and non-regressive
decode. The predeclared GPF-3A borderline repeat improves 512/4K
**640.876/672.866 -> 646.499/678.395 tok/s**, with byte-exact logits and
trajectories and aggregate decode wall -0.081%. GPF-5A improves focused 512/4K
prefill **645.901/676.444 -> 654.872/683.164 tok/s** and is conservatively
scoped through 4096 prompt tokens; the later 32K/64K screen regresses
1.62%/0.22%, so longer gfx1100 requests keep production Q8T16. The clean
combined transfer screen moves 512/4K from **648.512/682.172 -> 1352.908/1463.668 tok/s** with stable
IDs. Evidence is
[`2026-07-14-gfx1100-gguf-prefill-schedule-transfer-gate.json`](../benchmarks/results/2026-07-14-gfx1100-gguf-prefill-schedule-transfer-gate.json).

The clean post-transfer W7900 profile then invalidates the old LCP-1 hotspot
premise: 512/4K convolution is only **3.101/29.552 ms (0.87%/1.09%)**, while
exact GDN recurrence is **211.487/1652.114 ms (59.0%/61.1%)**. An exact
32-token/128-channel LDS convolution prototype passes six output/final-state
byte fixtures but is neutral at 512 and regresses full-model 4K **0.192%**; all
candidate code and routing were removed. Evidence is
[`2026-07-14-gfx1100-gguf-prefill-post-transfer-profile.json`](../benchmarks/results/2026-07-14-gfx1100-gguf-prefill-post-transfer-profile.json).

The independent `LCP-D1/D2` decode addendum is also complete on gfx1100. The
128K profile isolates **5.067 ms/token** grouped-GQA context plus
**1.621 ms/token** serial split reduction. Replacing only the reduction with a
parallel prepare and coalesced output stage moves the 513-split leaf
**194.881 -> 25.000 us (7.80x)**. Clean graph decode improves
**84.525 -> 85.561 tok/s (+1.23%)** at 32K, **72.446 -> 75.307 (+3.95%)** at
64K, and **56.927 -> 61.367 (+7.80%)** at 128K. The 64K logit gate records max
KL **1.904e-6**, top-1 100%, and exact generated IDs; memory is unchanged.
The gfx1100 backend selects the candidate from 32K onward while gfx1151 retains
serial reduction pending independent evidence. See
[`2026-07-14-gfx1100-gguf-decode-lcp-d2-parallel-reduce.json`](../benchmarks/results/2026-07-14-gfx1100-gguf-decode-lcp-d2-parallel-reduce.json).

The final clean W7900 defaults-only right-sized 1+3 publication records
**2716.648/3052.541/2953.101/2078.038/1559.878/1037.378 tok/s** prefill,
**92.833/98.148/100.522/88.240/76.691/62.669 tok/s** graph decode, and
**21.228/21.295/21.670/22.234/22.879/24.168 GiB** tracked memory at
512/1K/4K/32K/64K/128K. All 18 measured IDs are `9707`; maximum
prefill/decode stdev over median is **0.658%/0.223%**. This replaces the July
12 gfx1100 GGUF column and closes llama.cpp HIP parity; see the
[`final optimization rollup`](../benchmarks/results/2026-07-16-gfx1100-gguf-final-optimization-sweep.json).

Task #98 / `GPF-8` completed as a clean rejection: the eight-token FP32
chunkwise/WY recurrence passed primitive/resource gates but failed frozen
semantic, exact-trajectory, and 512 speed gates. Candidate code was removed;
the high-precision CPU algebra and full contract remain below as evidence.
Published defaults are unchanged.

On 2026-07-15, after reviewing the actual peer implementations, the numerical
contract changed prospectively. PARO K2 forms two wave32 partial reductions;
llama.cpp HIP uses register-sharded wave reductions; and llama.cpp Vulkan
`263cc04a5405` specializes the 128-state RADV path to eight lanes per value
column, 16 state rows per lane, and `subgroupClusteredAdd`. These schedules are
algebraically equivalent F32 recurrences but are not guaranteed bit-exact to a
scalar/decode-order contraction. llama.cpp's backend test uses CPU-reference
NMSE <= 1e-7 for F32 GDN rather than byte identity.

GGUF may therefore admit a reassociated GDN under the same class of contract:
CPU-reference primitive numerics; the complete 18-prompt category plus heldout
semantic suite at KL <= 0.05 and top-1 >= 90%; deterministic execution; decode
non-regression; and both 512/4K speed floors. Exact state bytes and exact free-
running trajectories remain diagnostics, not blockers. This is not a
retroactive acceptance: GPF-8 still fails KL <= 0.05 and the 512 floor. Existing
K2 and register-resident tree paths require fresh measurement under this
predeclared contract before promotion.

Scope: Qwen3.6-35B-A3B `UD-Q4_K_M`, BF16 KV, single-request bulk prefill on
`hip_gfx1100` and `hip_gfx1151`. LCP-D1 records the bounded 128K GGUF decode
follow-up selected by the parity audit; this is still not a general decode, MTP,
concurrency, or long-context memory plan.

## Decision

At the start of this tranche, the GGUF prefill gap was primarily a linear-
attention GDN recurrence problem. The production-exact fused kernel launches
one 128-thread block per
value head, keeps the prompt recurrence serial inside that block, and performs
each 128-element state contraction serially in one thread per value column. In
the retained W7900 512-token profile, its 30 launches consume **592.336 ms**, or
**79.51% of traced GPU time** and **72.11% of measured prefill wall**.

The first implementation therefore:

1. start from the correctness-certified raw-Q/K-plus-scale split path;
2. split independent value columns into wave-aligned blocks, initially 32 and
   64 columns per block;
3. preserve the current token order, eight-wide contraction order, state-update
   order, and separate post-recurrence RMSNorm+gate;
4. pass the existing six-case byte-exact GDN matrix; and
5. beat the fused production path in balanced full-prefill wall at both 512 and
   4096 rows.

Those were the tranche's original exact-schedule requirements. The 2026-07-15
peer review above explicitly supersedes byte identity as a universal promotion
requirement: a llama.cpp/PARO-style reassociated recurrence may instead pass
the peer-aligned numerical/semantic contract. Exact candidates still use the
stronger matrix, and prior failures remain historical evidence rather than
being relabeled after the fact.

`GPF-1` answered that question negatively on gfx1151. At 512/128, tile64
measured **388.300 tok/s** and tile32 **374.206 tok/s** versus the clean fused
control's **423.708 tok/s** (-8.36% and -11.68%). The cache-clean tile64 trace
also showed the recurrence itself regressing **794.120 -> 862.281 ms**, before
its additional **26.508 ms** prepare and **11.618 ms** RMSNorm+gate work. This
rules out `GPF-1B`: fusion can remove the latter overhead, but cannot recover a
recurrence kernel that is already 8.58% slower.

`GPF-2` identified the missing implementation property. Mapping one wave32 to
one value column is only fast when each lane loads its four recurrent-state
rows once, keeps them in registers across the complete serial token loop, and
writes them back once. The ordered-shuffle exact form and a tree-reduced form
that still round-tripped state through global memory both measured about
129 tok/s. The register-resident tree reaches **954.063/1031.350/847.981
tok/s** at 512/1K/4K versus the clean fused control's
**423.708/448.694/410.023** (+125.17%/+129.86%/+106.81%).

The promotion-grade same-session A/B confirms that the separate screens were
not a control/run-order artifact. With one warmup and four alternating measured
repetitions per mode, 512 moves **1212.462 -> 535.136 ms** (**422.281 ->
956.765 tok/s, 2.266x**) and 4096 moves **9977.239 -> 4848.216 ms**
(**410.534 -> 844.847 tok/s, 2.058x**). All 16 timed final IDs are `9707`,
provenance is clean at `31d4204d`, and the measured process took approximately
108 seconds including model/session setup. This completes the performance
portion of the promotion gate, not the numerical-contract decision.

Under the then-predeclared exact-trajectory contract, the decision was not to
relax the gate after observing a failure. On all ten prompts from the four-
category suite, only
**7/10** retain the first 25 fused samples and only **3/10** retain the complete
129-token (prefill sample plus 128 transitions) trajectory. First divergence is
transition 4 for `code_lru_cache`, 6 for `general_en_explain`, 18 for
`mixed_ja_en_review`, and 27/73/101/126 for four additional prompts. The first
three flips have KL **0.00922/0.01100/0.02851**, inside the scalar KL threshold,
but top-1 differs. This is a valid clean correctness rejection, not instability.
The balanced decode execution wall is flat (**53.316 vs 53.324 tok/s**), but
seven legs execute different token streams and therefore cannot support a
retained decode-performance comparison.

`GPF-2C` then applied the same state residency without changing the ordered
wave32 arithmetic. Plain and segmented primitive output/state remain byte-
exact, and all 46 focused correctness/routing tests pass. It recovers the
non-resident exact-wave row from **128.879 to 368.702 tok/s**, but still loses
to fused at every focused context: **368.702/383.292/354.672 tok/s** at
512/1K/4K, or **-12.98%/-14.58%/-13.50%**. The cache-clean 512 trace explains
the residual: ordered shuffle reconstruction leaves recurrence at
**928.006 ms / 30**, **16.86% slower** than fused recurrence, with 80 VGPR and
no spill. State residency was necessary, but ordered cross-lane reconstruction
is not an exact high-performance schedule.

`GPF-2D` removes the cross-lane reconstruction without relaxing arithmetic.
One thread still evaluates one value column's scalar 0..127 contraction and
update in the fused order, while a 32-thread block retains the 128x32 FP32
state tile in 16 KiB of row-major LDS across the serial token loop. Plain and
segment-aware tile32/tile64 fixtures are byte-exact. The first compiler build
was a useful rejection: forced row-loop unrolling generated **1,880 bytes of
scratch per thread** and limited 512 prefill to **401.732 tok/s**. Explicitly
leaving those loops rolled preserves order, removes all scratch, reduces VGPRs
from 96 to 64, and reaches **753.489/799.844/686.840 tok/s** at 512/1K/4K
(**+77.83%/+78.26%/+67.51%** versus fused). The cache-clean 512 recurrence is
**221.873 ms / 30**, 72.06% below fused recurrence, with 16 KiB LDS. Subsequent
clean exactness, balanced-wall, and natural-trajectory/decode gates all pass;
the backend-package policy now selects `chain_lds32` automatically on gfx1151
and keeps gfx1100 fused. The clean automatic six-shape stress gate records
**751.993/804.420/688.545/589.866/504.730/372.892 tok/s** prefill at
512/1K/4K/32K/64K/128K with stable final IDs. Its one max-128K session is not
the final right-sized memory/topline rollup.

That promotion changes the named 512 bottlenecks. The cache-clean default trace
records **221.873 ms** exact GDN recurrence, **156.474 ms** dense Q8T16 WMMA,
**116.075 ms** Q4T16 selected dual WMMA, and **56.181 ms** Q5T16 selected-down
WMMA out of **692.564 ms** traced kernels. This activates selected-MoE work,
not more generic GDN launch tuning.

The follow-up `GPF-2E` audit found that the retained exact prepare kernel still
copied raw Q/K/V from `conv_out` into three FP32 scratch arrays only for LDS32
to read them immediately. It also recomputed the same Q/K norms once per V head
even though eight V heads map to each K head. The explicit direct route keeps
the GPF-2D scalar recurrence unchanged, reads canonical Q/K/V directly, and
stores only per-V-head beta/decay plus compact per-K-head scales. Primitive
plain/segment output and state are byte-exact to materialized LDS32 and pass the
CPU-reference gate. The clean current-default/direct A/B wins all three focus
shapes, the ten-prompt natural gate is exact, and decode is non-regressive.
Backend policy therefore selects direct-conv on gfx1151 while leaving gfx1100
fused pending an independent transfer gate.

This tranche correctly did not start with AOTriton tuning, generic chunk
sweeps, graph capture, compiler flags, or another attempt to enable WMMA.
Full-attention prefill was only 0.54% of its starting 512-token GPU profile,
WMMA prefill was already enabled, and the existing exact split chain was
slower than fused. The later gfx1100 tranche correctly reprofiled the published
route rather than treating that historical 512 share as a permanent exclusion.

## Current Gap

The canonical numbers are the current eligible tables in
[`benchmarks/README.md`](../benchmarks/README.md). GGUF and llama.cpp use the
same sampled Q4_K_M file fingerprint
`936659d614707776d8e6ca1fb8595991159e78361bff2e3a3616aa91564c89fb`;
hipEngine uses BF16 KV and llama.cpp uses F16 KV. PARO uses a different W4
format, so it is a useful implementation/throughput reference rather than a
same-quant A/B. Vulkan is at least as fast as llama.cpp HIP at every listed
prefill shape, so the HIP column below is the conservative same-quant
comparator.

### W7900 / gfx1100

Clean 2026-07-14 hipEngine `ef3e97dd`, TheRock HIP 7.15; llama.cpp HIP
`1ebf790cd` build 9648 remains the matched reference. GGUF values are medians
from one discarded warmup and three measured repetitions in independent
right-sized processes. PARO remains the July 12 row.

| Workload | hipEngine GGUF | llama.cpp HIP | GGUF / llama HIP | hipEngine PARO | GGUF / PARO |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512/128 | 1290.246 | 2412.320 | 53.5% | 2917.732 | 44.2% |
| 1K/128 | 1395.244 | 2389.670 | 58.4% | 2995.876 | 46.6% |
| 4K/128 | 1401.632 | 2255.080 | 62.2% | 2943.038 | 47.6% |
| 32K/128 | 1221.716 | 1667.640 | 73.3% | 2108.868 | 57.9% |
| 64K/128 | 1021.693 | 1291.820 | 79.1% | 1584.131 | 64.5% |
| 128K/128 | 766.892 | 891.949 | 86.0% | 1056.252 | 72.6% |

### Radeon 8060S / gfx1151

The hipEngine GGUF column is the clean 2026-07-15 right-sized 1+3 production
refresh at `61a27d72`, TheRock HIP 7.15, kernel 7.1.3-2-cachyos, TuneD
`accelerator-performance`, and automatic one-queue process policy. llama.cpp
remains the clean July 11 matched reference; PARO is the separately retained
HIP 7.15 recovery. Repeated 128K is blocked and has no current ratio.

| Workload | hipEngine GGUF | llama.cpp HIP | GGUF / llama HIP | hipEngine PARO | GGUF / PARO |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512/128 | 1294.885 | 1061.260 | 122.0% | 1140.101 | 113.6% |
| 1K/128 | 1358.342 | 1043.230 | 130.2% | 1208.343 | 112.4% |
| 4K/128 | 1365.720 | 1009.240 | 135.3% | 1089.031 | 125.4% |
| 32K/128 | 1034.845 | 743.547 | 139.2% | 906.145 | 114.2% |
| 64K/128 | 796.083 | 573.611 | 138.8% | 716.775 | 111.1% |
| 128K/128 | blocked | 390.441 | — | 474.641 | — |

The published route does include the measured GDN improvement. Its additional
ratio narrowing at long context does not imply another context-dependent GDN
speedup: attention and other context-dependent work grow for every engine and
change the fraction of total wall.

### Decode Is The Control

Against llama.cpp HIP, GGUF decode is close or faster at short/mid context and
has a much smaller long-context deficit than prefill:

| GPU | Workload | hipEngine GGUF | llama.cpp HIP | GGUF delta |
| --- | --- | ---: | ---: | ---: |
| W7900 / gfx1100 | 512/128 | 92.833 | 80.756 | +15.0% |
| W7900 / gfx1100 | 4K/128 | 100.522 | 79.768 | +26.0% |
| W7900 / gfx1100 | 128K/128 | 62.669 | 60.933 | +2.8% |
| Radeon 8060S / gfx1151 | 512/128 | 49.041 | 50.939 | -3.7% |
| Radeon 8060S / gfx1151 | 4K/128 | 52.422 | 50.126 | +4.6% |
| Radeon 8060S / gfx1151 | 128K/128 | blocked | 32.114 | — |

That control makes a model-wide GGUF loader, quant, or HIP runtime explanation
unlikely. The retained 512-64K prefill deficit is closed against the listed llama.cpp HIP
reference. The missing current 128K row is a lifecycle/scheduler blocker, not a
measured kernel-throughput deficit.

## Evidence Timeline And Validity

The apparent June-to-July regression is real, but the old rate is not a valid
baseline to restore verbatim.

| Route | Evidence | Correctness status | How to use it |
| --- | --- | --- | --- |
| June normalized-Q/K split + K2 recurrence | W7900 512/1K/4K/32K/64K/128K prefill `2109.6/2331.3/2332.8/1799.8/1398.1/971.1 tok/s` | Rejected after the real llama.cpp greeting and token-serial/native state contract exposed a different recurrent result | Opportunity signal only; do not restore or benchmark-game toward it |
| Fused decode-order recurrence, selected by `937c13d1` | Current production path; July 7 W7900 refresh `654.0/664.6/668.1/635.3/578.7/490.3 tok/s` | Correctness-first baseline | Baseline to beat |
| Raw-Q/K-plus-scale exact split | [`SOL-G2 exact matrix`](../benchmarks/results/2026-07-11-sol-g2-gfx1151-gdn-prefill-exact-matrix.json): 6/6 greeting, 512, 1024/1025, and 4095/4096 cases | Byte-exact sampled token, hidden seed, resident Conv/GDN state; all-layer exact at greeting/512 | Unfused fallback, bisection oracle, and source for a new schedule |
| Existing exact split wall | [`SOL-G3 interleaved A/B`](../benchmarks/results/2026-07-11-sol-g3-gfx1151-gdn-prefill-interleaved-ab.json): chain is +5.19% at 512 and +6.70% at 4K | Correct but slower | Retain fused; do not retry the unchanged chain |
| Current gfx1151 fused M0 | HIP 7.15 clean control at 512/1K/4K: `423.708/448.694/410.023 tok/s`; cache-clean 512 trace: GDN `794.120/1227.335 ms`, 64.70% GPU-active | Five measured repetitions; timed token `9707` exact | Current same-session denominator |
| `GPF-1` tile64/tile32 | [`2026-07-13 rejected diagnostic`](../benchmarks/results/2026-07-13-gfx1151-gguf-prefill-gpf1-value-tiling-rejected.json): `388.300/374.206 tok/s` at 512, -8.36%/-11.68%; tile64 recurrence `862.281 ms` | Six primitive tile/segment cases byte-exact; 36 focused tests pass; decode flat | Rejected; keep only as short-lived diagnostic while GPF-2 is developed |
| `GPF-2A` non-resident wave32 | Ordered exact `128.879 tok/s`; tree-reduced `129.785 tok/s`; tree recurrence `3516.665 ms` | Ordered form byte-exact; tree primitive stays within numeric budget | Rejected: per-token global state traffic dominates |
| `GPF-2B` register-resident wave32 tree | [`candidate diagnostic`](../benchmarks/results/2026-07-13-gfx1151-gguf-prefill-gpf2-register-resident-candidate.json): `954.063/1031.350/847.981 tok/s` at 512/1K/4K; [`balanced A/B`](../benchmarks/results/2026-07-13-gfx1151-gguf-prefill-gpf2-balanced-ab.json): 2.266x/2.058x at 512/4K | Boundary KL/top-1 passes, but [`natural trajectory gate`](../benchmarks/results/2026-07-13-gfx1151-gguf-prefill-gpf2-trajectory-rejection.json) retains only 3/10 complete 128-step trajectories | Rejected for default; retain only as an explicit speed/numerical diagnostic |
| `GPF-2C` register-resident ordered wave32 | [`2026-07-13 rejected diagnostic`](../benchmarks/results/2026-07-13-gfx1151-gguf-prefill-gpf2c-ordered-resident-rejected.json): `368.702/383.292/354.672 tok/s` at 512/1K/4K, -12.98%/-14.58%/-13.50%; recurrence `928.006 ms` | Plain/segment output and FP32 state byte-exact; 46 focused tests pass; decode within -0.31%..-0.24% | Rejected: ordered shuffles remain slower than fused despite state residency |
| `GPF-2D` scalar-exact LDS32 residency | [`focus candidate`](../benchmarks/results/2026-07-13-gfx1151-gguf-prefill-gpf2d-lds32-focus-candidate.json): `753.489/799.844/686.840 tok/s` at 512/1K/4K; [`clean exact matrix`](../benchmarks/results/2026-07-13-gfx1151-gguf-prefill-gpf2d-exact-matrix.json): 6/6; [`balanced A/B`](../benchmarks/results/2026-07-13-gfx1151-gguf-prefill-gpf2d-balanced-ab.json): `420.959 -> 753.891` and `408.359 -> 687.831 tok/s` at 512/4K; [`automatic six-shape stress gate`](../benchmarks/results/2026-07-13-gfx1151-gguf-prefill-gpf2d-default-six-shape.json): `751.993/804.420/688.545/589.866/504.730/372.892 tok/s` | Sampled token, hidden seed, resident Conv/GDN state, and required layer outputs byte-exact; [`natural gate`](../benchmarks/results/2026-07-13-gfx1151-gguf-prefill-gpf2d-trajectory-decode-gate.json): 250/250 exact logits, all timed trajectories exact, decode +0.023%; default stress IDs stable | **Promoted on gfx1151** and superseded in the final rollup by GPF-3A/2E; gfx1100 stays fused pending transfer evidence |
| `GPF-3A` Q4T16 shared activation | [`exact fixture + real-model replay`](../benchmarks/results/2026-07-13-gfx1151-gguf-prefill-gpf3a-q4t16-shared-x-replay.json): Q4 gate/up `114.633 -> 97.082 ms` (-15.31%); [`clean full-model A/B`](../benchmarks/results/2026-07-13-gfx1151-gguf-prefill-gpf3a-full-model-ab.json): 512/1K/4K `747.764/804.150/687.676 -> 771.027/823.624/701.042 tok/s` (+3.11%/+2.42%/+1.94%); [`automatic focus confirmation`](../benchmarks/results/2026-07-13-gfx1151-gguf-prefill-gpf3a-default-focus.json): `774.653/823.149/701.389 tok/s` | BF16/FP16 fixture bytes and full-model logits byte-exact; every 128-step measured trajectory identical; aggregate decode wall -0.0031%; automatic IDs stable | **Promoted on gfx1151** and included in the final right-sized rollup; gfx1100 stays baseline pending transfer evidence |
| `GPF-2E` compact-scale direct-conv LDS32 | [`clean balanced A/B`](../benchmarks/results/2026-07-13-gfx1151-gguf-prefill-gpf2e-balanced-ab.json): current default `776.428/825.319/700.824 -> 823.093/889.209/744.577 tok/s` at 512/1K/4K (+6.01%/+7.74%/+6.24%); [`automatic focus`](../benchmarks/results/2026-07-13-gfx1151-gguf-prefill-gpf2e-default-focus.json): `821.755/897.160/750.896 tok/s`; [`right-sized 1+3 rollup`](../benchmarks/results/2026-07-13-gfx1151-gguf-prefill-gpf2e-right-sized-3run.json): `819.641/893.266/752.308/640.096/540.850/387.334 tok/s` | Plain/segment primitive and [`six-case full-model matrix`](../benchmarks/results/2026-07-13-gfx1151-gguf-prefill-gpf2e-exact-matrix.json) are byte-exact; [`natural gate`](../benchmarks/results/2026-07-13-gfx1151-gguf-prefill-gpf2e-trajectory-decode-gate.json) passes 250/250 exact logits and all timed trajectories with decode +0.075%; first-three serialized IDs are stable through 64K and the log-recovered 128K row links the stronger independent gates without inventing missing IDs | **Promoted and published on gfx1151**; gfx1100 stays fused; later 128K no-progress is a separate lifecycle soak |

The old route proves that substantially more parallel recurrence was possible;
it does not by itself prove that its normalized-Q/K materialization or
reduction tree passes the GGUF product gate. The useful recovery question is
how much of that parallelism passes the peer-aligned numerical and semantic
contract.

## Current Implementation Comparison

### hipEngine GGUF

The production path is in
[`qwen35_gguf_runner.py`](../hipengine/runtime/qwen35_gguf_runner.py), with GDN
kernels and registry bindings in
[`gdn.hip`](../hipengine/kernels/hip_gfx1100/linear_attn/gdn.hip) and
[`gdn.py`](../hipengine/kernels/hip_gfx1100/linear_attn/gdn.py). The gfx1151
backend reuses the registered gfx1100 source template under its own backend and
target-architecture identity.

Thirty linear-attention layers run bulk Q8 projections, convolution, GDN,
output projection, and MoE. Ten full-attention layers run bulk Q/K/V work,
RoPE/KV append, AOTriton attention, output projection, and MoE. The relevant
GDN choices are:

| Path | Launch geometry | Arithmetic | Current role |
| --- | --- | --- | --- |
| Fused decode-order | grid `num_v_heads`, block 128 | Serial tokens; one thread owns one value column and serially accumulates all 128 state rows; Q/K normalization and RMSNorm remain in the same head block | Production default |
| Exact split | prepare grid `tokens × num_v_heads`; recurrence grid `num_v_heads`, block 128; RMSNorm grid `tokens × num_v_heads` | Raw Q/K and scales stay separate; recurrence preserves the fused eight-wide multiplication/accumulation order | Exact fallback; 5.19%-6.70% slower in full wall |
| Fast K2 split | recurrence grid `num_v_heads × head_v_dim`, block 64 | Two wave32 shards reduce the 128 state rows | PARO path and first GPF-9 GGUF candidate under the peer-aligned contract |

The serial dependency across prompt tokens is real, but value columns are
independent until RMSNorm. The exact split already creates the synchronization
boundary needed to schedule those columns across more blocks without changing
the per-column recurrence.

### PARO

PARO's current implementation in
[`qwen35_paro.py`](../hipengine/runtime/qwen35_paro.py) uses prepare +
`qwen35_gdn_prefill_recurrent_k2_f32` + RMSNorm/gate. It assigns an independent
block to each value column and distributes the 128 state rows over two wave32
reductions. That is structurally much more parallel than the GGUF fused path. PARO's
quant/model contract is different, so its reduction order is implementation
evidence rather than a GGUF correctness oracle; the GGUF CPU and 18-prompt
gates still own admission.

PARO's retained 256-row gfx1151 chunks and isolated AOTriton stream are useful
scheduling precedents, not direct GDN fixes. They should transfer only after a
fresh GGUF profile names the same family and shape.

### llama.cpp HIP

The measured llama.cpp HIP binary is `1ebf790cd` build 9648. Its clean GDN
source file last changed at `e95dae18` and uses grid
`(H, n_seqs, ceil(S_v / 4))`, block `(physical_wave_size, 4)`: one wave owns a
value column, each lane keeps a shard of the state rows in registers, and wave
reductions form the contractions. The token loop remains serial. See
[`gated_delta_net.cu` at e95dae18](https://github.com/ggerganov/llama.cpp/blob/e95dae18d64ae4471d61a9dc87880a64e0e5c86e/ggml/src/ggml-cuda/gated_delta_net.cu).

This is important structural evidence: llama.cpp does not need a token-parallel
scan to reach the current reference rates. It exposes state/value-column
parallelism and uses a different reduction tree. Its source still contains a
TODO for a chunked prefill kernel, so the current llama.cpp number is not an
algorithmic ceiling.

The retained gfx1100 pp512 family trace confirms the fused HIP kernel is active
and attributes **15.534 ms** to GDN, versus hipEngine GGUF's current **211.487
ms** at 512. The hardware and full stacks are not a publication-grade matched
A/B, but source plus trace are decisive implementation evidence. See
[`2026-06-16-gpu1-llamacpp-hip-q4km-pp512-rocprof-diagnostic.json`](../benchmarks/results/2026-06-16-gpu1-llamacpp-hip-q4km-pp512-rocprof-diagnostic.json).

### llama.cpp Vulkan

Vulkan commit `263cc04a5405` uses the same serial token recurrence with a
different state-row reduction schedule. On a 64-lane RADV subgroup and
`S_v=128`, the pipeline selects eight lanes per value column, so one subgroup
processes eight columns concurrently. Each lane retains 16 state rows in
registers; `subgroupClusteredAdd(..., 8)` forms the K and Q contractions. The
backend-op oracle compares F32 output with the CPU backend at the default NMSE
<= 1e-7; it does not require byte identity. This independently confirms that
fast peer GDN is quality-tolerant F32 reassociation, not a hidden bit-exact
schedule.

## Current Bottleneck Attribution

The only retained family profile of the production-exact route is
[`2026-07-07-w7900-gpu0-gguf-q4km-current-gdn-audit.json`](../benchmarks/results/2026-07-07-w7900-gpu0-gguf-q4km-current-gdn-audit.json):
W7900/gfx1100, hipEngine `b891aa04`, TheRock HIP 7.13, Q4_K_M/BF16 KV,
512/0, WMMA prefill enabled, cached builds, and one synchronized measurement.
It measured 623.288 tok/s and 821.450 ms wall.

| Family | Dispatches | GPU ms | GPU share | Whole-wall share |
| --- | ---: | ---: | ---: | ---: |
| GDN fused decode-order prefill | 30 | 592.336 | 79.51% | 72.11% |
| Dense Q8_0 WMMA prefill | 250 | 53.387 | 7.17% | 6.50% |
| Selected Q4 dual-WMMA MoE | 40 | 39.391 | 5.29% | 4.80% |
| Selected Q5 WMMA MoE | 37 | 23.691 | 3.18% | 2.88% |
| Router | 120 | 12.627 | 1.70% | 1.54% |
| Full-attention prefill | 10 | 4.026 | 0.54% | 0.49% |
| All traced kernels | 1949 | 744.955 | 100.00% | 90.69% |

The current July 12 W7900 topline uses HIP 7.15 and measures 644.719 tok/s at
512, so this older profile is a strong route-local diagnosis but not a
stack-matched candidate baseline. `GPF-M0` below refreshes 512, 4K, and 128K
before any promotion claim.

Simple whole-wall Amdahl estimates from this profile are steering estimates,
not performance claims:

| Hypothetical GDN speedup | Predicted whole-prefill speedup |
| ---: | ---: |
| 2x | 1.56x |
| 4x | 2.18x |
| Infinite / remove all GDN wall | 3.59x |

Even eliminating the measured GDN wall would not quite close the current
W7900 512 same-quant gap by itself. A large GDN win is necessary, after which a
new profile should select selected-MoE, dense Q8, or the next actual bucket.

The older June split-route profiles remain useful only as a post-GDN ordering
hint. At 512 they attributed 27.39% to dense work, 30.41% to selected Q4/Q5
MoE, 22.66% to GDN, 6.36% to routing, and 1.80% to attention. At 4K the shares
were 22.51%, 28.99%, 25.69%, 7.03%, and 5.88%. Do not use those percentages to
select current code without a fresh profile.

## What We Have Ruled Out

| Hypothesis or easy win | Finding | Decision |
| --- | --- | --- |
| GGUF/Q4_K_M is inherently this slow | llama.cpp HIP uses the same Q4_K_M file and is 1.44x-3.74x faster across current shapes | Rejected as a sufficient explanation |
| The whole GGUF runtime is slow | Decode is within +21.0% to -13.2% of llama.cpp HIP at the sampled shapes | Focus on prefill-specific families |
| WMMA prefill is disabled | The retained command and artifact both report `effective_use_wmma_prefill=true` | No enablement work |
| Full attention/AOTriton is the 512 bottleneck | Ten full-attention launches are 0.54% of traced GPU time | Revisit only at 4K/128K if a fresh profile activates it |
| Select the existing exact chain | Balanced full wall loses 5.19% at 512 and 6.70% at 4K | Rejected unchanged; keep as fallback/oracle |
| Restore the old K2 chain without a fresh gate | It changes contraction order but is the retained PARO schedule and previously passed primitive numerics | Re-evaluate first under the 2026-07-15 peer-aligned full semantic and two-shape speed contract |
| Copy PARO's chunk sizes | PARO's wins affect different layer/quant paths; current GGUF GDN remains serial inside each layer launch | No broad threshold sweep |
| Prefill graph replay or host submission first | Traced kernels account for 90.69% of wall and GDN alone accounts for 72.11% | Kernel geometry first |
| Blind compiler flags or `__launch_bounds__` | No retained resource/ISA comparison identifies a compiler-only cause; source geometry exposes too few blocks directly | Collect VGPR/scratch/occupancy, then change a named constraint |
| Token-parallel prefix scan is required | llama.cpp's current faster kernel still loops tokens serially | Recover column/state parallelism first |

## Ranked Work Plan

| Order | ID | Work | Activation / exit |
| ---: | --- | --- | --- |
| 0 | `GPF-M0` | **Complete:** published-route 512/4K full traces, bounded 128K family sample, and matched llama.cpp 512/4K/128K traces | GPF-4 is selected from the measured residual; the full hipEngine 128K trace remains a rocprof lifecycle limitation, not missing selection evidence |
| 1 | `GPF-1` | **Rejected:** exact split recurrence with 64/32 value columns per block | Both full wall and tile64 recurrence regress; do not promote |
| 2 | `GPF-1B` | **Skipped:** fuse GPF-1 prepare/materialization only if recurrence wins | Tile64 recurrence itself loses 8.58%, so fusion cannot close this lane |
| 3 | `GPF-2B` | **Rejected under the historical exact-trajectory contract:** register-resident tree wins wall by 2.266x/2.058x but keeps only 3/10 complete natural 128-step trajectories | Eligible only for a fresh 18-prompt re-evaluation under the prospectively changed 2026-07-15 contract; do not retroactively relabel the old artifact |
| 4 | `GPF-2C` | **Rejected:** register-resident exact ordered-wave recurrence | Byte-exact, but focused 512/1K/4K prefill loses 12.98%-14.58% and recurrence loses 16.86% |
| 5 | `GPF-2D` | **Promoted on gfx1151:** scalar-exact value columns with recurrent state resident in a 32-column LDS tile | Automatic route passes the clean six-shape stress gate and final rollup; keep gfx1100 fused pending transfer evidence |
| 6 | `GPF-M1` | **Default profile complete:** exact GDN 221.873 ms, dense Q8T16 156.474 ms, Q4T16 selected 116.075 ms, Q5T16 selected 56.181 ms at 512 | These measured families select GPF-3A and its successors |
| 7 | `GPF-3A` | **Promoted on gfx1151:** share one Q4T16 activation fragment across the existing two 16-column WMMA accumulators | Clean 512/1K/4K full-model prefill +3.11%/+2.42%/+1.94%, exact logits/trajectories, aggregate decode -0.0031%; gfx1100 remains baseline |
| 8 | `GPF-2E` | **Promoted and published on gfx1151:** compact Q/K scales and direct `conv_out` Q/K/V reads for exact LDS32 | Clean 512/1K/4K prefill +6.01%/+7.74%/+6.24%, 250/250 natural logits exact, decode +0.075%; six right-sized rows retained; gfx1100 remains fused |
| 9 | `GPF-L1` | **Parked lifecycle diagnostic:** isolate intermittent 128K fresh-graph/session no-progress after the three-run timing window | Add phase markers and bounded lifecycle tests separately; do not lengthen the performance sweep or block the retained row |
| 10 | `GPF-4` | **Rejected as a default; retained explicit diagnostic:** event-link GGUF AOTriton to an isolated stream while pre/post math stays on the caller queue | Exact and often fast, but the required final gate exposed severe intermittent GPU-active stalls at 32K/128K; both gfx1151 and gfx1100 stay same-stream |
| 11 | `GPF-5` | **GPF-5A promoted on gfx1151 through 64K:** two exact 32-column Q8T16 waves share one activation tile | Final 512-64K components are +1.01%-8.57%; stable same-commit 128K is -2.59%, so request-scoped package metadata restores production there |
| 12 | `LCP-1` | **Promoted and published on gfx1151:** exact 32-token by 128-channel shared-memory convolution | 82/82 state parts exact; 4K body 954.134 -> 49.790 ms; six-shape prefill +1.10%..+24.04%; gfx1100 stays baseline |
| 13 | `LCP-D1` | **Retained long-context decode reduction:** cooperate only above 256 splits | 4,096 BF16 values exact; 128K reducer -16.30%; graph decode 27.753 -> 28.047 tok/s; shorter reducer unchanged |
| 14 | `LCP-2A` | **Promoted on gfx1151:** compiler-cacheable exact direct-LDS32 GDN state | Six-case state and 250/250 natural transitions exact; balanced 512/1K/4K prefill +34.76%/+36.63%/+36.58%; volatile GPF-2E remains rollback |
| 15 | `LCP-M2` | **Promoted on gfx1151 through 4K:** generate contiguous chunk metadata on-device instead of six synchronous H2D copies | Automatic-vs-explicit 512/1K/4K is 83/83 exact; balanced prefill +1.56%/+0.90%/+0.53%. Explicit 128K with one queue still enters the low-power no-progress state, so longer requests retain synchronous metadata |
| 16 | `LCP-3` | **Promoted on gfx1151 through 64K:** four exact Q8T16 waves share one activation tile | Clean 512/4K state is 83/83 exact; five-pair full-model prefill +0.53%/+1.57%; two-wave and production remain rollback paths |
| 17 | `LCP-4A` | **Promoted on gfx1151:** exact 256-thread BF16-hidden/F32-weight router logits | Clean 512/4K state is 83/83 exact; prefill +2.76%/+3.28%; graph decode exact/+0.071%; gfx1100 remains 512-thread |
| 18 | `LCP-4B` | **Promoted on gfx1151:** exact 128-thread bulk-prefill router selection | Fresh 4K family -70.17%; 512/4K full-model +0.34%/+0.36% with 83/83 exact state. Reject 64 threads because 4K full-model state differs; gfx1100 stays 512 and decode stays 256 |
| 19 | `LCP-PUB` | **Retained through 64K; 128K blocked:** final selector-unset production refresh | Five clean right-sized 1+3 rows improve prefill +25.11%..+46.10%, all 15 IDs exact, memory unchanged. One-queue, launch-rollback, and SDMA-disabled 128K full gates reproduce the firmware/scheduler stall; publish no stale current 128K number. |

### gfx1100 architecture-local continuation

| Original order | ID | Work | Activation / exit |
| ---: | --- | --- | --- |
| 12 | `GPF-6` | **Rejected:** wave/group register-resident direct-input schedules | Fast group4 misses the frozen semantic gate; exact group3 misses the speed floors; do not reopen reduction-width sweeps |
| 13 | `GPF-7` | **Rejected:** Atlas-inspired scalar-column register residency | Exact, but gfx1100 compiles at 256 VGPR with about 1 KiB scratch per thread; SM121 register capacity does not transfer |
| 14 | `GPF-8` | **Rejected and removed:** eight-token FP32 chunkwise/triangular-WY recurrence over direct-conv Q/K/V | Resource/family gate passes, but KL 0.056522 and the 512 llama.cpp floor fail; the later trajectory-policy change does not alter rejection |
| 15 | `GPF-9A/B` | **Rejected:** existing normalized-Q/K K2 and raw-Q/K-plus-scale register-resident wave32 tree | K2 fails KL `0.059031`; tree fails `0.068757`. Both pass top-1 and decode, but do not proceed to speed |
| 16 | `GPF-9C` | **Rejected on 512 speed:** `chain_peer_wave32` reproduces llama.cpp HIP's normalized-Q/K one-wave32 schedule | Semantics/decode pass; 4K beats llama.cpp by 11.45%, but 512 reaches `2210.729 tok/s`, 8.36% below the required floor |
| 17 | `GPF-9D` | **Rejected on strict decode:** `chain_peer_cluster8` reproduces llama.cpp Vulkan's eight-lane clustered S_v=128 schedule | KL `0.028689` and top-1 `444/450` pass, but decode regresses `0.001286%` with no predeclared tolerance; no speed gate |
| 18 | `LCP-3A` | **Rejected and removed:** four exact 32-column Q8T16 FP16-WMMA waves share one activation tile | Byte-exact, but width 4096 and short-K width 2048 regress 2.24%/8.87%; the pp512 mix projects about +0.3 ms. Stop independent-wave widening |
| 19 | `LCP-3B` | **Rejected and removed:** direct prequantized Q8_1 x Q8T16 integer-WMMA body | Primitive numerics/quality pass, but the body is 44.66% slower before quantization; quantization adds only 0.016 ms. Any continuation must reproduce the actual shared MMQ tile/decomposition, not route raw dp4a or retry direct T16 |
| 20 | `LCP-3C/3D` | **Audit complete; T16-backed MMQ128 rejected and removed:** reproduce llama.cpp's 128-output x128-token K256 tile over resident Q8T16 plus D4 Q8 activations | D4 bytes and primitive quality pass, but the T16 gather/transpose body is +118.81% before quantization. Source MMQ requires output-major packed weight bytes; do not retry over T16 |
| 21 | `LCP-3E` | **Rejected and removed:** source-compatible MMQ128 directly over output-major `block_q8_0` rows | Correctness passes and the final WGP body is spill-free with source-matched fragment/WMMA counts, but prequantized/included rows are +3.95%/+5.32% vs production. The frozen gate fails before profiler/runtime work; raw+T16 dual residency remains forbidden |
| 22 | `LCP-4A` | **Promoted:** capture-free normal FP32 Conv prefill with explicit sequential ISA multiply/add | Removes 20 private bytes/thread while preserving byte-exact Conv output/state; pp512 body -77.71%, clean production 512/4K prefill +1.44%/+1.86% |
| 23 | `LCP-5A` | **Promoted on gfx1100:** roll only the outer Q4/Q5/Q6 T16 quant-block loops and make quality-admitted peer GDN automatic | Removes HIP 7.2's Q5 256-VGPR/176-byte/75-spill code object while preserving byte-exact outputs; Q5 trace -42.08%, total peer kernels/span 184.513/194.886 ms. The clean 18-prompt contract passes and selector-unset 512/4K reaches 2588.231/2757.752 tok/s with 21.670 GiB tracked peak; gfx1151 and explicit exact rollback are unchanged |

There is no invented minimum full-model percentage. Under the project evidence
policy, every correctness-admitted, measured, non-regressive improvement is
retainable. The
512-and-4K requirement prevents selecting a shape-local regression; repetition
and variance gates decide whether a measured delta is real.

## GPF-1 Design

The existing exact recurrence maps one block to one value head:

```text
grid  = (num_v_heads)
block = (128)
value_idx = threadIdx.x
```

Change only the value-column mapping:

```text
grid  = (num_v_heads, ceil(head_v_dim / VALUE_TILE))
block = (VALUE_TILE)              # first candidates: 64, 32
value_idx = blockIdx.y * VALUE_TILE + threadIdx.x
```

Each live thread keeps the current code for every token and every state-row
term. It reads and updates a disjoint recurrent-state column, so no atomic or
cross-block recurrence reduction is needed. The existing separate
RMSNorm+gate kernel runs after all recurrence blocks complete on the stream and
retains the cross-column normalization boundary.

Why this is the lowest-risk high-impact candidate:

- the total arithmetic and state bytes do not change;
- the per-column FP operation order does not change;
- the number of schedulable recurrence blocks grows by 2x at tile 64 or 4x at
  tile 32;
- wave32-aligned blocks avoid partial-wave waste;
- the exact prepare and RMSNorm kernels already exist and are validated; and
- it directly tests whether coarse one-block-per-head occupancy is the loss.

Expected risks are duplicate/common Q/K reads across column tiles, extra block
scheduling, and the existing split prepare/RMSNorm/materialization overhead.
That is why selection is by recurrent-kernel trace plus end-to-end prefill wall,
not occupancy intuition alone.

If GPF-1 lowers the recurrence kernel but not full wall, GPF-1B should keep the
same tiled recurrence and remove unnecessary raw-Q/K/value scratch writes and
reads. It must not replace the exact Q/K scale tree or alter the contraction
expression.

### GPF-1 result (gfx1151, 2026-07-13)

All `(value_tile=128,64,32) x (plain,segments)` primitive cases preserved the
fused BF16 output and FP32 recurrent state byte-for-byte. The performance
hypothesis failed:

| Route | 512/128 prefill | Delta vs fused | Decode | GDN recurrence trace |
| --- | ---: | ---: | ---: | ---: |
| Fused control | 423.708 tok/s | control | 49.116 tok/s | 794.120 ms / 30 |
| Exact split tile64 | 388.300 tok/s | -8.36% | 49.104 tok/s | 862.281 ms / 30 |
| Exact split tile32 | 374.206 tok/s | -11.68% | 49.113 tok/s | not traced after wall rejection |

Tile64 lowers reported VGPRs from 96 to 40, but it does not add total waves:
the same 4,096 work-items become 64 two-wave blocks instead of 32 four-wave
blocks. It also gives up useful same-block sharing/locality. Occupancy intuition
was therefore insufficient, and the measured kernel result rejects both
`GPF-1` and its materialization-only `GPF-1B` follow-up.

## GPF-2 Result

`GPF-2A` first tested state-row sharding without changing state lifetime. The
ordered-shuffle kernel reconstructed the fused 0..127 contraction order and
was byte-exact, but 256 shuffles per token plus per-token global state traffic
collapsed 512/128 to **128.879 tok/s (-69.58%)**. Replacing the ordered
reconstruction with a wave tree barely changed the result:
**129.785 tok/s (-69.37%)**. Its cache-clean recurrence trace was
**3516.665 ms / 30**, 4.43x the fused recurrence.

The source comparison with llama.cpp exposed the missing property: its
`s_shard[rows_per_lane]` lives in registers across the token loop. `GPF-2B`
does the same, while caching normalized Q/K shards per token and storing final
state once. The recurrence falls to **61.411 ms / 30**, uses 40 VGPR, zero
scratch/LDS, and changes full-wall performance as follows:

| Route | 512/128 | 1K/128 | 4K/128 | Decode medians |
| --- | ---: | ---: | ---: | ---: |
| Clean fused control (5 runs) | 423.708 | 448.694 | 410.023 | 49.116 / 51.657 / 52.505 |
| Register-resident tree, 256 threads (3 runs) | **954.063** | **1031.350** | **847.981** | 49.020 / 51.600 / 52.414 |
| Delta | **+125.17%** | **+129.86%** | **+106.81%** | -0.20% / -0.11% / -0.17% |

The 128-thread/four-wave llama.cpp-shaped launch is rejected as shape-local:
it measures **957.756** at 512 (+0.39% versus 256 threads), but only
**1012.765/832.416** at 1K/4K (-1.80%/-1.84%). Keep 256 threads/eight waves.

The tree changes FP reduction order. Across greeting, 512, 1024/1025, and
4095/4096, fused and candidate sampled tokens are identical, top-1 agreement
is **100%**, and KL is **3.48e-6..5.39e-5**, well inside the repository gate.
It is not byte-exact: layer-0 recurrence diverges first and accumulated hidden
and state fingerprints differ. The clean balanced performance gate passes at
**2.266x/2.058x** for 512/4K with exact timed IDs. The subsequent natural gate
rejected it under the historical contract: only **3/10** prompts keep the full
fused 128-transition trajectory. `auto` therefore remained fused and
`chain_wave32_tree` remained an explicit diagnostic. The prospective 2026-07-15
contract change permits a fresh 18-prompt re-evaluation; it does not relabel
this old artifact.

`GPF-2C` changes only the exact ordered kernel's state lifetime: each lane
loads its four FP32 rows before the token loop, updates them in registers, and
stores them after the loop. It keeps the existing ordered shuffles, explicit
FMA sites, and output expression. The expected exactness holds, but performance
does not:

| Route | 512/128 | 1K/128 | 4K/128 | Decode medians |
| --- | ---: | ---: | ---: | ---: |
| Clean fused control (5 runs) | 423.708 | 448.694 | 410.023 | 49.116 / 51.657 / 52.505 |
| Register-resident ordered wave32 (3 runs) | 368.702 | 383.292 | 354.672 | 48.966 / 51.533 / 52.362 |
| Delta | -12.98% | -14.58% | -13.50% | -0.31% / -0.24% / -0.27% |

The cache-clean recurrence is **928.006 ms / 30** versus fused
**794.120 ms / 30** (+16.86%), with workgroup 256, 80 VGPR, and zero
scratch/LDS. This rejects ordered-shuffle residency as a default candidate.
The next exact schedule (`GPF-2D`) instead keeps the scalar one-thread-per-
value-column contraction and makes the state resident in block LDS. A 32-column
tile needs 16 KiB and a 64-column tile 32 KiB for the 128xvalue FP32 state;
both preserve scalar evaluation order without global state round trips or
cross-lane reconstruction.

### GPF-2D focus result (gfx1151, 2026-07-13)

The LDS32 implementation assigns one scalar-exact value column to each thread.
It loads the column's 128 FP32 state elements into a row-major 128x32 LDS tile,
keeps that tile resident for the complete token loop, and writes it back once.
No thread reads another thread's column, so the schedule needs no reduction,
atomic, or barrier and preserves the fused contraction/update expression.

| Route | 512/128 | 1K/128 | 4K/128 | Decode medians |
| --- | ---: | ---: | ---: | ---: |
| Clean fused control (5 runs) | 423.708 | 448.694 | 410.023 | 49.116 / 51.657 / 52.505 |
| Scalar-exact LDS32 (3 runs) | **753.489** | **799.844** | **686.840** | 49.069 / 51.674 / 52.522 |
| Delta | **+77.83%** | **+78.26%** | **+67.51%** | -0.10% / +0.03% / +0.03% |

The initial forced-unroll build is explicitly rejected: it measured only
**401.732 tok/s** at 512 and the trace exposed **1,880 bytes/thread scratch**.
Using `#pragma unroll 1` for the 128-row load/contraction/update/store loops
keeps the same iteration order and removes the spill. The final cache-clean
trace records **692.564 ms** total kernels and **221.873 ms / 30** recurrence
(32.04% of GPU-active time), workgroup 32, 64 VGPR, zero scratch, and 16 KiB
LDS. Prepare and RMSNorm+gate contribute **28.246/9.519 ms**. The full-model
screens are tightly clustered and every measured final ID is `9707`. The clean
six-case matrix, balanced 512/4K A/B, and ten-prompt gate described above then
clear the promotion contract. Backend capability metadata now makes LDS32 the
gfx1151 `auto` route while retaining fused and exact-chain rollback paths; the
clean automatic six-shape stress gate completes without ID instability.

### GPF-3A Q4T16 shared-activation result (gfx1151, 2026-07-13)

The production compact32 Q4T16 selected-dual kernel computes two serial
16-column output halves. Each half currently traverses K independently and
reloads the identical compact 16-row activation fragment. GPF-3A retains two
independent FP32 WMMA accumulators, moves the output-half loop inside the K
loop, and feeds both halves from one activation load. Each accumulator sees the
same K and WMMA order as baseline.

The BF16 and FP16 candidates are raw-byte exact on fixtures with uneven and
empty experts and multiple K blocks. `rocprofv3` records **44.725 -> 33.343
us (-25.45%)** on the tiny exact fixture; VGPR rises **48 -> 56**, while both
routes use zero scratch/LDS. On the resident 35B model's identical 40-layer,
512-token routing, Q4 gate/up replay falls **114.633 -> 97.082 ms (-15.31%)**
and all selected gate/up+down pairs fall **176.410 -> 158.535 ms (-10.13%)**.
The sampled token and routing are identical. The subsequent clean balanced
full-model gate at commit `95d484df` records baseline -> `shared_x` prefill of
**747.764 -> 771.027 tok/s (+3.11%)** at 512, **804.150 -> 823.624 (+2.42%)**
at 1K, and **687.676 -> 701.042 (+1.94%)** at 4K. All three 248,320-logit
comparisons are byte-identical (`KL=0`, top-1 100%), every measured 129-ID
prefill/decode trajectory matches, and aggregate decode wall is
**7527.985 -> 7527.750 ms (-0.0031%)**. GPF-3A is therefore promoted through
the gfx1151 backend capability; gfx1100 remains on baseline without transfer
evidence. A subsequent clean selector-unset production sweep at promoted
commit `431fe1e4` records four-run 512/1K/4K medians of
**774.653/823.149/701.389 tok/s** prefill and
**48.881/51.451/52.257 tok/s** eager decode, with all final IDs `9707`. Its one
max-4K session makes it a route confirmation rather than a canonical
right-sized publication row.

The executable full-model gate is
[`scripts/gguf_q4_t16_prefill_ab.py`](../scripts/gguf_q4_t16_prefill_ab.py).
It first requires the BF16 and FP16 byte-exact kernel fixture artifact, then
compares baseline and `shared_x` full-model prefill logits byte-for-byte at
512/1K/4K. In one resident session it discards at least one balanced warmup and
records at least four balanced measurements per mode and shape. Prefill wall is
synchronized around the production call. The following 128 graph-replay decode
steps are timed separately with graph capture and token readback excluded, and
every prefill sample plus decoded ID must match across every leg. Promotion
requires a lower candidate median prefill wall at all three shapes and no
increase in the sum of the three median decode walls. There is no percentage
threshold or full-model dilution rule.

### GPF-2E direct-conv promotion gates (gfx1151, 2026-07-13)

The retained GPF-2D prepare path writes prompt-sized raw Q, raw K, and V FP32
arrays, and its `[token, v_head]` scale layout repeats the same Q/K norm for
each V head sharing a K head. GPF-2E adds a separate diagnostic ABI: compact
prepare writes `[token, k_head]` Q/K scales plus `[token, v_head]` beta/decay,
then a direct LDS32 recurrence reads raw Q/K/V from the canonical `conv_out`.
The old materialized route is unchanged for a stable A/B and rollback.

RED failed collection before the compact-prepare and direct plain/segment
wrappers existed. GREEN passes 89 GDN/harness tests. On a production shared-
head fixture, both direct schedules are byte-identical to materialized LDS32
for BF16 output and FP32 final state and pass the CPU reference. A cached-only
`rocprofv3` trace observes both direct kernel names with workgroup 32, 64 VGPR,
16 KiB LDS, and zero scratch.

The first full-model screen used separate same-worktree max-4K sessions, one
discarded warmup, and three eager-decode measurements. It is intentionally
`performance_claim=false`, but every prefill distribution is separated:

| Route | 512/128 | 1K/128 | 4K/128 | Decode medians |
| --- | ---: | ---: | ---: | ---: |
| Current automatic GPF-2D + GPF-3A | 769.378 | 821.460 | 702.808 | 48.873 / 51.463 / 52.281 |
| Explicit compact/direct LDS32 | **817.004** | **903.229** | **755.077** | 48.802 / 51.374 / 52.224 |
| Prefill delta | **+6.19%** | **+9.95%** | **+7.44%** | separate-session diagnostic only |

All control and candidate final IDs are `9707`. At the focus-screen stage these
results retained only the explicit candidate; the later clean gates below are
the evidence that changes gfx1151 `auto`.

The clean detached `c3a065ee` exact matrix now passes all six required cases:
greeting, 512, 1024/1025, and 4095/4096. Fused and direct-conv produce the
same sampled token, FP32 hidden seed, and all 30 resident Conv/GDN state pairs
in every case; greeting and 512 also match every captured layer final row.
Both plain and segmented dispatch boundaries are covered. This is a
correctness-only gate; its diagnostic single-order walls are excluded from
performance claims.

The clean same-session A/B at `ffbcc4d9` compares the shipped materialized
LDS32 route directly with compact/direct LDS32, using one warmup and four
balanced measured repetitions per mode and context:

| Route | 512 | 1K | 4K |
| --- | ---: | ---: | ---: |
| Materialized LDS32 baseline | 776.428 | 825.319 | 700.824 |
| Compact/direct LDS32 | **823.093** | **889.209** | **744.577** |
| Throughput delta | **+6.01%** | **+7.74%** | **+6.24%** |

Every paired wall delta favors direct-conv and all 24 timed IDs are `9707`.
The retained command took approximately 2.5 minutes including model/session
setup. The subsequent clean ten-prompt gate at `5501aeb9` passes **250/250**
logit transitions with `KL=0`, top-1 100%, and exact tokens. Every balanced
128-token decode trajectory matches; weighted decode is **53.3282 -> 53.3684
tok/s (+0.075%)**. This satisfies the predeclared promotion contract.

gfx1151 backend capability now selects `chain_lds32_direct`; gfx1100 remains
fused pending independent evidence. Materialized LDS32 remains as an explicit
rollback/bisection route through one release window. A clean selector-unset
focus at promotion commit `b8949477` reproduces **821.755/897.160/750.896
tok/s** at 512/1K/4K and all 12 final IDs remain `9707`. It took about 2.3
minutes. Because it uses four measurements and one max-4K eager session, it is
a policy confirmation rather than the final right-sized fresh-graph rollup.
The automatic route is settled; the final rollup is recorded below.

### Final right-sized publication sweep (gfx1151, 2026-07-13)

The clean detached publication worktree is `28b45d38`. Each shape ran in its
own right-sized process under the hermetic TheRock HIP 7.15 boundary, with both
prefill selectors absent, cached builds required, repeated token `9707`, eager
warmups, and fresh state-bound graph decode for measured repetitions. The
original invocation requested two warmups and five measurements. After
reviewing the observed variance and cost, the retained window is the first
three measured repetitions; future gfx1151 GGUF sweeps use one warmup plus
three measurements.

| Shape | Prefill tok/s | Versus July 11 public GGUF | Versus max-128K GPF-2D stress | Graph decode tok/s | Tracked peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512/128 | **819.641** | +90.27% | +9.00% | 49.067 | 21.478 |
| 1K/128 | **893.266** | +104.19% | +11.04% | 51.644 | 21.710 |
| 4K/128 | **752.308** | +86.24% | +9.26% | 52.498 | 22.995 |
| 32K/128 | **640.096** | +73.03% | +8.52% | 43.550 | 23.559 |
| 64K/128 | **540.850** | +61.74% | +7.16% | 37.305 | 24.203 |
| 128K/128 | **387.334** | +43.14% | +3.87% | 27.753 | 25.493 |

The first-five-shape serialized components have finite logits and three stable
final IDs (`9707`) per row. The interrupted 128K process did not write its
final JSON, so its first three completed lines are recovered from the durable
log and no per-run token-ID field is invented. Each line is printed only after
the full repetition, including correctness and memory collection, returns.
The same clean automatic route is independently covered by the six-case byte-
exact state matrix and the ten-prompt gate with **250/250 exact logits** and
exact timed trajectories. Those stronger gates make the 128K performance row
eligible while preserving the missing-field disclosure.

Variance does not justify five measurements here. Across all six rows, the
largest prefill sample stdev/median is **0.132%** and the largest decode value
is **0.030%**, far below the 5% rejection guard. For every shape with five
serialized samples, the first-three median equals the five-sample median at
full precision. Compact retained evidence is
[`2026-07-13-gfx1151-gguf-prefill-gpf2e-right-sized-3run.json`](../benchmarks/results/2026-07-13-gfx1151-gguf-prefill-gpf2e-right-sized-3run.json).

The cost audit is explicit. For 512 through 64K, the original two-warmup/five-
measurement work accounts for **1583.236 s (26.387 min)**. One warmup plus the
first three measurements accounts for **1002.365 s (16.706 min)**. The second
warmup and measurements four/five therefore burned exactly **580.872 s
(9.681 min)** without changing any median. At 128K, continuing attempt 1 past
the 1+3 stop point cost about **27.4 min**, and the unnecessary full rerun cost
about **27.9 min**. Total avoidable time is approximately **65.0 minutes**.

The later no-progress behavior remains real but is a separate lifecycle soak.
Attempt 1 completed two warmups and four measurements, then stayed at 100% GPU
for 921 seconds during measurement 5. Attempt 2 completed two warmups and one
measurement, then repeated the state for 614 seconds during measurement 2.
No amdgpu fault/reset appeared; termination returned the GPU to idle and a
cached dispatch smoke passed. Current logging cannot distinguish prefill,
warmup decode, graph capture/replay/readback, or memory snapshot as the exact
phase. Track that with phase markers and bounded lifecycle tests rather than
lengthening every performance sweep. Evidence is
[`2026-07-13-gfx1151-gguf-prefill-gpf2e-lifecycle-soak.json`](../benchmarks/results/2026-07-13-gfx1151-gguf-prefill-gpf2e-lifecycle-soak.json).

## GPF-M2: Fresh Family Selection

The next tranche began with the required measured profile rather than another
GDN assumption. At clean published commit `81e2f4b8`, no-warmup one-pass
prefill-only rocprof runs used the automatic GPF-2D/3A/2E route, cached builds,
bulk attention, and WMMA prefill. The 512 and 4K traces are complete. A full
128K trace continued at 100% GPU without completing for approximately 15
minutes, so it was terminated and replaced with a bounded
`--collection-period 30:60:1` sample. That sampled process completed normally
with 333.598 seconds of host prefill. These are family-selection diagnostics,
not replacements for the published 1+3 throughput rows.

| Family | hipEngine 512 | hipEngine 4K | hipEngine 128K sample | llama.cpp 512 | llama.cpp 4K | llama.cpp 128K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact GDN recurrence | 206.899 ms / 33.39% | 1771.481 ms / 32.83% | 12158.943 ms / 29.29% | 44.180 ms / 9.85% | 330.488 ms / 8.85% | 12784.759 ms / 3.97% |
| Linear-attention convolution | 13.571 ms / 2.19% | **952.870 ms / 17.66%** | **7337.238 ms / 17.68%** | 4.260 ms / 0.95% | **32.980 ms / 0.88%** | 1088.971 ms / 0.34% |
| Dense Q8 | 158.982 ms / 25.66% | 844.670 ms / 15.65% | 5133.161 ms / 12.37% | 66.859 ms / 14.91% | 531.345 ms / 14.24% | 16942.774 ms / 5.26% |
| Selected/raw Q4 | 96.386 ms / 15.56% | 620.630 ms / 11.50% | 3845.752 ms / 9.26% | 141.704 ms / 31.60% | 1173.527 ms / 31.44% | 36479.808 ms / 11.32% |
| Selected/raw Q5 | 56.298 ms / 9.09% | 391.398 ms / 7.25% | 2542.025 ms / 6.12% | 70.947 ms / 15.82% | 585.011 ms / 15.67% | 18195.281 ms / 5.64% |
| Full/flash attention | 4.700 ms / 0.76% | 150.316 ms / 2.79% | 5695.609 ms / 13.72% | 6.137 ms / 1.37% | 204.083 ms / 5.47% | 208121.191 ms / 64.56% |

The 128K hipEngine percentages are only the bounded sample and must not be
compared as absolute milliseconds against llama.cpp's complete trace. Kernel
sums may also double count overlap across queues. The full 4K comparison is
nevertheless decisive: hipEngine convolution costs **952.870 ms** versus
llama.cpp's **32.980 ms**, a **28.89x** gap, while hipEngine's AOTriton core is
already faster in that trace (**150.316 vs 204.083 ms**). The AOTriton image
reports the same 2560-byte scratch footprint that triggered PARO's proven
queue-local downstream convolution cliff. GDN remains the largest raw family,
but it is the documented high-effort fallback; selected Q4/Q5 and attention
core are not the largest eligible residual.

GPF-4 is therefore activated: reuse the existing event-linked isolated
AOTriton stream policy for GGUF, leaving pre/post math on the caller stream.
Selection evidence is
[`2026-07-14-gfx1151-gguf-prefill-next-family-profile.json`](../benchmarks/results/2026-07-14-gfx1151-gguf-prefill-next-family-profile.json).
Promotion still requires exact 512/4K/128K off/on state and logit gates plus a
balanced full-wall A/B; profile evidence alone does not make the route default.

## GPF-4: GGUF AOTriton Queue Isolation Candidate

RED added three GGUF-specific unit requirements: an explicit/default policy,
an AOTriton bridge parameter on the full-attention runner, and lazy
stream/event reuse plus release on the resident session. All three failed on
the published route. GREEN reuses `AotritonPrefillStreamBridge`: Q/K/V and the
FP32-to-BF16 query cast remain on the caller stream, an input-ready event gates
only AOTriton's high-scratch launch on one lazy nonblocking stream, and an
output-ready event gates the existing post-attention BF16 gate on the caller
stream. Session close synchronizes and releases both events and the stream.

The implementation first landed default-off at `006306ac`. Explicit
`HIPENGINE_QWEN35_AOTRITON_ISOLATED_PREFILL_STREAM=1` selects it on either HIP
backend for testing. Clean promotion-screen evidence briefly admitted a
provisional gfx1151 package capability, but the final stability gate below
removed it. Both backends now remain same-stream by default.

Fresh-process differential correctness at repeated token `9707` is exact:

| Context | Sampled token | Compared parts | FP32 logits | FP32 hidden seed | 30 Conv/GDN pairs | 10 live K/V pairs |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 512 | 9707 / 9707 | 82 | byte-exact | byte-exact | all byte-exact | all byte-exact |
| 4K | 9707 / 9707 | 82 | byte-exact | byte-exact | all byte-exact | all byte-exact |

A fresh process is part of the timing contract. A same-session off/on
interleave is invalid for this candidate: once a baseline AOTriton launch has
contaminated the caller queue, later candidate legs cannot undo the persistent
queue-local state. That diagnostic measured flat and is deliberately excluded.
One discarded warmup plus three measured runs in a fresh process per mode give:

| Context | Same-stream off tok/s | Isolated on tok/s | Delta | Peak off/on GiB |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 819.834 | 827.678 | **+0.96%** | 21.478 / 21.478 |
| 4K | 743.990 | 912.357 | **+22.63%** | 22.995 / 22.995 |

These focus rows are not retained performance claims because the candidate was
uncommitted. They are strong implementation evidence. A no-warmup candidate
4K trace independently moves host prefill **5.475 -> 4.617 s**, kernel sum
**5396.575 -> 4543.829 ms (-15.80%)**, and convolution
**952.870 -> 88.839 ms (-90.68%)**, while AOTriton itself stays flat
(**150.316 -> 148.791 ms**). This directly confirms the selected queue-cliff
mechanism rather than merely shifting time to attention.

Compact focus evidence is
[`2026-07-14-gfx1151-gguf-prefill-gpf4-candidate-focus.json`](../benchmarks/results/2026-07-14-gfx1151-gguf-prefill-gpf4-candidate-focus.json).

The clean detached `006306ac` promotion gate reproduces every exact hash and
uses fresh processes for each timing leg:

| Context | Same-stream off tok/s | Isolated on tok/s | Delta | Decision |
| ---: | ---: | ---: | ---: | --- |
| 512 | 822.203 | 823.614 | **+0.17%** | Non-regressive |
| 4K | 747.721 | 902.928 | **+20.76%** | Retain |
| 128K screen | published 387.334 / sampled 392.904 | 432.403 | **+11.64% / +10.05%** | Promote, then confirm with final 1+3 |

The 128K screen is one no-warmup candidate run, not the final public row. It
completed in **303.125 s**, produced token `9707`, and held tracked peak at
**25.493 GiB**. Clean 512/4K exactness again compares all 82 parts with zero
mismatches. This was sufficient for a provisional architecture-scoped
promotion attempt; the final automatic-route six-shape 1+3 sweep still owned
publication. Promotion-screen evidence is
[`2026-07-14-gfx1151-gguf-prefill-gpf4-clean-promotion.json`](../benchmarks/results/2026-07-14-gfx1151-gguf-prefill-gpf4-clean-promotion.json).

The final gate **rejects the default**. Clean automatic-route 512/1K/4K/64K
components are stable and fast, but 32K contains one measured collapse to
**294.254 tok/s** between two approximately 761 tok/s runs. That satisfies the
documented variance trigger. Its fresh 1+5 replacement then failed to finish
even the warmup after 481 s process wall, versus approximately 43 s normally.
At 128K the warmup and measured run 1 were stable at **439.698/439.448 tok/s**,
but measured run 2 remained GPU-active for at least **1200 s**, versus roughly
298 s normally. Both attempts were bounded rather than converted into
unplanned lifecycle soaks. An explicit same-stream 32K control immediately
completed in **51.040 s / 642.003 tok/s**, isolating the instability to the
candidate route rather than the host.

Therefore `HIPENGINE_QWEN35_AOTRITON_ISOLATED_PREFILL_STREAM=1` remains an
explicit diagnostic only, no gfx1151 capability is exported, the published
GPF-2E row remains canonical, and GPF-5 is next. Final rejection evidence is
[`2026-07-14-gfx1151-gguf-prefill-gpf4-final-rejected.json`](../benchmarks/results/2026-07-14-gfx1151-gguf-prefill-gpf4-final-rejected.json).

## GPF-5: Dense Q8T16 WMMA Prefill

No new full-model profile is needed after GPF-4 rejection: the default route is
again the already-profiled GPF-M2 route, and GPF-4 did not change any default
kernel body. Excluding high-effort GPF-6 recurrence and rejected GPF-4 queue
scheduling, dense Q8T16 is the largest eligible family:

| Context | Dense Q8T16 | Selected Q4T16 | Selected Q5T16 | Router |
| ---: | ---: | ---: | ---: | ---: |
| 512 | **158.982 ms (25.66%)** | 96.386 ms (15.56%) | 56.298 ms (9.09%) | 30.173 ms (4.87%) |
| 4K | **844.670 ms (15.65%)** | 620.630 ms (11.50%) | 391.398 ms (7.25%) | 242.086 ms (4.49%) |

The matched llama.cpp Q8 MMQ buckets are 66.859/531.345 ms, making the family
duration ratios 2.38x/1.59x. These are selection ratios, not equivalent-kernel
speed claims. At 4K, the Q8T16 `32x32` body owns **671.736 ms / 380 launches
(79.53% of the family)** and reports 104 VGPR with no scratch.

A bounded six-tile recheck rules out a stale heuristic. On the dominant
`rows=1024,in=2048,out=8192` shape, `32x32` wins at **2.051 ms** versus
2.184/2.204/2.205/2.270 ms for `16x16`/`32x16`/`64x16`/`64x32`; at rows 4096
it narrowly wins **8.213 vs 8.254 ms** over `64x32`. Every tile is byte-exact.
Therefore GPF-5A does not retune tile dimensions. It pairs two independent,
order-preserving 32-column waves in one 64-column block and shares one
BF16-to-FP16 activation tile through bounded LDS. The first gate requires exact
bytes, no scratch, profiler-confirmed 64-thread/two-wave geometry, and a paired
real-shape micro win before any model routing.

The callable/default-off candidate passes that gate. Tail-row/output fixtures
are byte-exact to production `32x32`; a profiler smoke records **64 threads,
80 VGPR, 128 SGPR, 1 KiB LDS, and zero scratch**. Interleaved cycling-pool
microbench medians improve **2.008 -> 1.683 ms (-16.17%)** at 1K rows and
**8.004 -> 6.692 ms (-16.39%)** at 4K rows on `2048x8192`. Fresh-process
full-model 1+3 focus is also positive:

| Context | Default tok/s | GPF-5A tok/s | Delta | Peak off/on GiB |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 820.817 | 893.840 | **+8.90%** | 21.478 / 21.478 |
| 4K | 747.177 | 764.858 | **+2.37%** | 22.995 / 22.995 |

Every timed token is `9707`. Separate-process differential capture compares
FP32 logits/hidden, all 30 Conv/GDN state pairs, and all 10 live BF16 K/V
pairs: **82/82 parts are byte-exact** at both 512 and 4K. This dirty-worktree
focus is implementation evidence, not a retained performance claim. Candidate
evidence is
[`2026-07-14-gfx1151-gguf-prefill-gpf5a-candidate-focus.json`](../benchmarks/results/2026-07-14-gfx1151-gguf-prefill-gpf5a-candidate-focus.json).
The clean detached `4a1fff53` gate reproduces exactness and the speedup. At 512,
default **819.333** becomes **887.760 tok/s (+8.35%)**. The primary 4K baseline
leg contains one severe 25.015 tok/s outlier, satisfying the documented
variance trigger; the required fresh 1+5 replacement is stable:

| Context | Default samples tok/s | GPF-5A samples tok/s | Median delta |
| ---: | --- | --- | ---: |
| 4K | 747.651, 747.693, 745.914, 748.162, 746.572 | 765.959, 766.906, 766.606, 767.686, 766.537 | **+2.54%** |

Clean 512/4K differential state remains 82/82 parts byte-exact and memory is
unchanged. The gfx1151 backend therefore overrides only the two BF16/BF16
Q8T16 prefill registry aliases with an automatic wrapper; gfx1100 still maps
to production. The auto wrapper selects GPF-5A only for default TM32/TM64,
TN32, output width >=2048 shapes. Explicit `=0` is the rollback and `=1`
remains the cross-backend diagnostic. Clean evidence is
[`2026-07-14-gfx1151-gguf-prefill-gpf5a-clean-promotion.json`](../benchmarks/results/2026-07-14-gfx1151-gguf-prefill-gpf5a-clean-promotion.json).

The final automatic sweep adds one necessary scope gate. GPF-5A is stable and
positive through 64K, but automatic 128K measures **382.041 tok/s** while the
same-clean-commit explicit production control measures **392.219 tok/s** under
the identical 1+3, 128-decode-token protocol: **-2.59%** candidate prefill,
unchanged token IDs/memory, and +0.09% decode. This is a real, stable long-
context regression, not grounds to discard shorter exact wins or publish a
lower row. gfx1151 package metadata therefore sets
`GGUF_Q8_T16_PREFILL_TWO_WAVE_MAX_TOKENS=65536`; the resident prefill request
installs that policy session-locally, and explicit env `0|1` retains highest
precedence. 128K uses the production wrapper. Scope evidence is
[`2026-07-14-gfx1151-gguf-prefill-gpf5a-128k-scope.json`](../benchmarks/results/2026-07-14-gfx1151-gguf-prefill-gpf5a-128k-scope.json).
The final scoped `6418b278` automatic 128K retry confirms that the production
wrapper is selected: warmup is **386.098 tok/s** and measured run 1 is
**385.474 tok/s**, with **25.493 GiB** tracked peak. Measured run 2 then
reproduces the separately documented GPU-active later-pass lifecycle stall, so
it was bounded rather than converted into an unplanned soak. One completed
measurement cannot replace the accepted 1+3 row. Publication therefore
refreshes 512/1K/4K/32K/64K to
**889.904/919.598/762.940/648.948/546.296 tok/s** and carries forward the
unchanged production-wrapper 128K row at **387.334 tok/s**. All 15 refreshed
IDs are `9707`; tracked memory is unchanged; refreshed prefill/decode sample
stdev over median is at most **0.088%/0.038%**. Final evidence is
[`2026-07-14-gfx1151-gguf-prefill-gpf5a-right-sized-3run.json`](../benchmarks/results/2026-07-14-gfx1151-gguf-prefill-gpf5a-right-sized-3run.json).

Selection evidence is
[`2026-07-14-gfx1151-gguf-prefill-gpf5-family-selection.json`](../benchmarks/results/2026-07-14-gfx1151-gguf-prefill-gpf5-family-selection.json).
The external lineage checkout is absent in this environment, so the required
lineage command cannot resolve it; no external code is copied, and GPF-5A is an
in-tree schedule experiment over the catalogued T16 body.

## LCP-1: exact shared-token convolution

The post-GPF-5A llama.cpp parity audit selected linear-attention convolution as
the strongest transferable schedule: hipEngine's clean 4K caller-stream trace
spent **954.438 ms** in convolution versus llama.cpp's **32.980 ms**. LCP-1
therefore stages a 35-row input window for 32 output tokens by 128 channels in
17.5 KiB LDS. It preserves the production kernel's four separately rounded
FP32 products, serial add order, SiLU, and unchanged second state-update launch.
Both implementations are registered under `gguf_qwen35`; the production route
remains the exact fallback.

The first healthy-queue microbench was deliberately not hidden: the initial
candidate lost to production at both 512 and 4K rows. The bounded full-model
probe then measured **928.759 vs 756.961 tok/s (+22.70%)**, proving that the new
same-stream body avoids the post-AOTriton queue-local cliff that the isolated
microbench cannot reproduce. Replacing volatile product locals with an explicit
separately rounded `v_mul_f32_e32` removes scratch while preserving bytes. The
final cached 4K trace records **49.790 ms / 120** output launches versus
**954.134 ms / 120** for production, with 128 threads, 17.5 KiB LDS, and zero
scratch. Candidate evidence is
[`2026-07-14-gfx1151-gguf-prefill-lcp1-candidate-focus.json`](../benchmarks/results/2026-07-14-gfx1151-gguf-prefill-lcp1-candidate-focus.json).

Clean detached commit `3ff8e2d7` reproduces **82/82** exact FP32
logits/hidden, Conv/GDN, and live BF16 K/V parts at both 512 and 4K. Its
prescribed one-warmup/three-measurement focus is:

| Context | Production samples tok/s | LCP-1 samples tok/s | Median delta |
| ---: | --- | --- | ---: |
| 512 | 863.622, 891.776, 890.727 | 881.900, 907.469, 906.118 | **+1.73%** |
| 4K | 762.873, 761.546, 762.273 | 936.910, 937.202, 935.743 | **+22.91%** |

Tracked peak remains 21.478/22.995 GiB. gfx1151 therefore selects
`f32_tile32x128` automatically; gfx1100 remains on `f32_baseline` pending an
independent hardware transfer. Explicit
`HIPENGINE_GGUF_LINEAR_ATTN_CONV_PREFILL_MODE=baseline` is the rollback. Clean
evidence is
[`2026-07-14-gfx1151-gguf-prefill-lcp1-clean-promotion.json`](../benchmarks/results/2026-07-14-gfx1151-gguf-prefill-lcp1-clean-promotion.json).
The clean right-sized automatic sweep at `71e61524` is complete: prefill is
**906.979/929.724/946.366/778.371/636.330/433.811 tok/s** at
512/1K/4K/32K/64K/128K, or
**+1.92%/+1.10%/+24.04%/+19.94%/+16.48%/+12.00%** versus the prior public row.
Graph decode is **49.061/51.569/52.432/43.543/37.562/28.047 tok/s**, including
**+1.06%** at 128K. All 18 measured IDs are `9707`, tracked memory is
unchanged, and maximum prefill/decode sample stdev over median is only
**0.140%/0.113%**, so no five-run escalation is required. Compact evidence is
[`2026-07-14-gfx1151-gguf-lcp1-lcpd1-right-sized-3run.json`](../benchmarks/results/2026-07-14-gfx1151-gguf-lcp1-lcpd1-right-sized-3run.json).

## LCP-D1: exact long-split decode reduction

The bounded clean decode profiles at `631498dd` establish that the 512-token
Amdahl ordering does not transfer to 128K. At 512, dense Q8 is
**8.520 ms/token / 44.25%** and attention is **2.160 ms/token / 11.22%**. At
128K, attention grows to **17.882 ms/token / 50.95%**: the grouped-GQA context
body is **15.502 ms/token** and the gated split reducer is
**2.347 ms/token**.

An all-query-head register tile preserved output bytes but regressed the 128K
attention call **1.748 -> 2.878 ms (0.607x)** and was removed. LCP-D1 instead
keeps max selection, denominator summation, and final output accumulation in
the original serial split order. It parallelizes only independent exponentials
and normalization multiplies when `num_splits > 256`; the original serial body
remains byte-for-byte through 256 splits.

The 512-split microbench compares 4,096 BF16 outputs byte-for-byte and moves the
reducer **138.139 -> 101.350 us (1.363x)**; the 256-split control is neutral at
**76.184 -> 76.103 us**. The final clean `71e61524` 128K profile moves the
reducer **234.714 -> 196.466 us/call (-16.30%)**, attention
**17.882 -> 17.498 ms/token (-2.15%)**, total GPU time
**35.094 -> 34.668 ms/token (-1.22%)**, and profiled host wall
**36.860 -> 36.380 ms/token**, or **27.130 -> 27.488 tok/s (+1.32%)**. All 24
candidate tokens are exact; the reducer uses 16 VGPR and zero scratch. The
256/257 CPU contract, direct-gate/registry tests, cached BF16 GQA/GQA-state
smokes, and all seven CPU fixtures pass. Evidence is
[`2026-07-14-gfx1151-gguf-decode-lcpd1-clean-profile.json`](../benchmarks/results/2026-07-14-gfx1151-gguf-decode-lcpd1-clean-profile.json).

The graph-decode public row remains governed by the separate right-sized 1+3
component. Do not substitute the single eager marker-profile rate for that
median, and do not attribute shorter-context decode drift to LCP-D1: the new
branch is not executed at `num_splits <= 256`.

## LCP-2A: compiler-cacheable exact GDN state

The production-path audit reopened exact GDN work without relaxing the
recurrent-state contract. The first bounded idea, sharing scaled Q/K through
LDS across the 32 value columns, was byte-exact but nearly doubled the
isolated recurrence (`6.639 -> 13.014 ms` at 512 and
`58.455 -> 113.116 ms` at 4K); it was removed.

The second candidate changes no operation order. The existing rolled scalar
body is instantiated with a nonvolatile LDS pointer so LLVM may cache legal
state accesses; the original volatile plain/segment symbols remain unchanged
as rollback. On gfx1151, the isolated one-layer recurrence moves
**6.572 -> 1.763 ms (3.73x)** at 512 and
**58.613 -> 19.864 ms (2.95x)** at 4K. `rocprofv3` records the intended
nonvolatile body with **32 VGPR, 16 KiB LDS, and zero scratch**, versus 64 VGPR
for the production body.

The dirty-worktree correctness screen is exact in all six required full-model
cases: greeting, repeated 9707 at 512, 1024/1025, and 4095/4096. Sampled token,
FP32 hidden seed, all resident Conv/GDN states, and the greeting/512 all-layer
outputs match fused byte-for-byte. A one-warmup/four-interleaved-measurement
screen against `chain_lds32_direct` wins every pair and reduces median bulk
prefill wall by **26.33%/27.15%/27.15%** at 512/1K/4K. This dirty screen remains
`performance_claim=false`.

Clean detached candidate `53928aaf` reproduces all six exact cases. Its
one-warmup/four-interleaved-measurement A/B moves prefill
**900.814 -> 1213.912 tok/s (+34.76%)** at 512,
**940.736 -> 1285.266 tok/s (+36.63%)** at 1K, and
**941.462 -> 1285.888 tok/s (+36.58%)** at 4K; every pair wins and every timed
ID is exact. The ten-prompt gate passes **250/250** natural transitions with
`KL=0`, 100% top-1, exact timed decode trajectories, and weighted decode
**53.348 -> 53.359 tok/s (+0.021%)**. gfx1151 therefore selects
`chain_lds32_direct_nonvolatile` automatically; gfx1100 remains fused pending
its independent hardware gate. The volatile GPF-2E route remains the explicit
rollback. Clean evidence is
[`2026-07-14-gfx1151-gguf-gdn-lcp2a-clean-promotion.json`](../benchmarks/results/2026-07-14-gfx1151-gguf-gdn-lcp2a-clean-promotion.json); the earlier dirty candidate artifact remains implementation history.

## LCP-3: four-wave dense-Q8 prefill

GPF-5A shares one exact BF16 activation tile between two production-order
32-column waves. LCP-3 extends only that sharing scope: one 128-thread block
keeps four independent waves and their K traversal, WMMA order, accumulators,
and output mapping unchanged while sharing the same 1 KiB activation tile
across 128 output columns.

Tail fixtures are byte-exact. The cached candidate trace records
`gguf_q8_0_t16_prefill_wmma_nwave_kernel<32,4>` at 128 threads, 80 VGPR,
128 SGPR, 1 KiB LDS, and zero scratch. At 4K rows the dominant
`2048x8192` and `8192x2048` micros improve **7.50%** and **14.08%** over
GPF-5A.

Clean detached candidate `d34476da` reproduces **83/83** exact full-model parts
at both 512 and 4K. Five balanced same-session pairs move prefill
**1214.510 -> 1220.993 tok/s (+0.53%)** at 512 and
**1269.030 -> 1288.986 tok/s (+1.57%)** at 4K; all 20 timed IDs are `9707`.
The request-scope gate routes gfx1151 to four-wave through 65,536 prompt tokens
and restores production above it, conservatively inheriting the ceiling from
the predecessor two-wave schedule's measured 128K rejection. Explicit `HIPENGINE_GGUF_Q8_T16_PREFILL_4WAVE=0` selects
two-wave; `HIPENGINE_GGUF_Q8_T16_PREFILL_2WAVE=0` restores production. gfx1100
remains production pending independent hardware evidence. Clean evidence is
[`2026-07-15-gfx1151-gguf-q8-t16-four-wave-clean-promotion.json`](../benchmarks/results/2026-07-15-gfx1151-gguf-q8-t16-four-wave-clean-promotion.json).

## LCP-4A: exact F32-router launch geometry

The current F32-weight router body maps one eight-element hidden fragment per
lane. At the model's `hidden_size=2048`, only 256 of the former 512 lanes
receive useful work; the first reduction step adds zero partials. Launching the
unchanged token-tiled body with 256 threads preserves all meaningful additions
and every output byte.

Nine-pair primitive medians at `experts=256` improve
**0.683 -> 0.380 ms (-44.32%)** for 512 tokens and
**1.354 -> 0.756 ms (-44.17%)** for 1024. Clean detached candidate `3ef55ad4`
reproduces **83/83** exact full-model parts at 512 and 4K. Five balanced pairs
improve prefill **1218.536 -> 1252.147 tok/s (+2.76%)** and
**1290.923 -> 1333.229 tok/s (+3.28%)**, with all timed IDs exact. The separate
512/128 graph gate keeps the final token/logit exact and improves
**48.987 -> 49.021 tok/s (+0.071%)**.

A cached `rocprofv3` smoke records
`qwen35_router_logits_token_tile_kernel<unsigned short, float, 4>` at 256
threads, 32 VGPR, 128 SGPR, and zero scratch. gfx1151 therefore overrides only
`(router_logits, f32, bf16_hidden)` with the 256-thread wrapper; gfx1100 keeps
512 threads pending hardware-local evidence. Clean evidence is
[`2026-07-15-gfx1151-gguf-router-threads256-clean-promotion.json`](../benchmarks/results/2026-07-15-gfx1151-gguf-router-threads256-clean-promotion.json).

## GPF-8: Eight-Token Chunkwise/WY GDN

Status: rejected and removed on gfx1100. The CPU algebra/oracle remains as
rejected-design evidence; all HIP, registry, runtime-selector, and candidate-
specific test surfaces are gone, and no performance claim exists. The source
ideas are Atlas
`gated_delta_rule_wy{,64_prefill}.cu` at `8d187c7`/`37513bf` and the FLA-derived
vLLM `chunk.py`, `chunk_scaled_dot_kkt.py`, and `wy_fast.py` at their registered
file commits. hipEngine does not copy their CUDA/Triton bodies, BF16 matrix
contract, gate clamp, or Torch API. It independently applies the triangular
identity to the retained direct-conv FP32 Q/K/V and compact-scale ABI.

For a chunk starting at state `H0`, let `P[t]` be the product of decays from
chunk start through token `t`, and `R[t,s]` the product after source token `s`
through `t` (`R[t,t] = 1`). The exact real-number recurrence is:

```text
delta[t] = beta[t] * (
    value[t]
    - P[t] * key[t] @ H0
    - sum_{s<t} R[t,s] * (key[t] @ key[s]) * delta[s]
)

out[t] = P[t] * query[t] @ H0
         + sum_{s<=t} R[t,s] * (query[t] @ key[s]) * delta[s]

H1 = P[last] * H0
     + sum_s R[last,s] * outer(key[s], delta[s])
```

This is the direct lower-triangular/Woodbury-Young form. The committed
`gdn_prefill_chunkwise_wy_segments` oracle evaluates it in float64 and crosses
to float32 only at its output/state boundary. The hand-checked three-token
fixture and random packed-segment tests cover chunk sizes 1/2/3/8/16, an odd
17-token tail, slot remapping, and the current FP32 primitive budget.

### HIP schedule selected before implementation

- Keep the retained compact-scale prepare and read canonical FP32
  `conv_out` Q/K/V directly. Do not materialize prompt-sized normalized Q/K/V.
- Use `C=8`; one 256-thread block owns `(segment, v_head, value_tile32)` and
  keeps the same 128x32 FP32 state tile in 16 KiB LDS across the segment.
- Per chunk, stage normalized `Q[8,128]` and `K[8,128]` in LDS. Sixty-four
  threads compute the 8x8 K-K and Q-K coefficients; all 256 threads compute
  eight token/state projections in parallel; 32 value lanes solve the small
  triangular system; all threads update the chunk-final state tile.
- Target LDS is at most 32 KiB: 16 KiB state, 8 KiB Q/K, and bounded
  projection/delta/coefficient scratch. The first candidate uses FP32 scalar
  contractions, not low-precision WMMA, so the algebra change is isolated from
  a dtype change.
- Full eight-token chunks use WY algebra. A remainder is processed token-
  serially in the same block with the retained direct arithmetic; a pure
  one-token segment must be byte-exact to `chain_lds32_direct`.
- Register separate plain and segment-aware variants. `chain_lds32_direct`
  remains the registered unfused/default fallback throughout admission.

The schedule preserves the current number of GDN-stage launches (compact
prepare, recurrence, RMSNorm+gate), exposes eight prompt tokens to 256 threads,
and adds no `O(tokens * v_heads * C^2)` global coefficient arena. At the clean
512 trace, non-GDN kernels consume about 145.6 ms and llama.cpp's 2412.320 tok/s
floor implies about 212.2 ms total wall. Therefore the complete 30-layer GDN
family must fall from 210.501 ms to roughly 66 ms or less. Eight-way token
parallelism makes the design plausible; a serial WY implementation does not.

### Numerical and performance contract (predeclared)

1. **CPU algebra:** all committed chunk sizes and packed tails match the
   independent float64 serial recurrence within `atol=rtol=2e-6`; the
   production-shaped FP32 comparison must remain within `atol=3e-5,
   rtol=3e-4`. These thresholds are frozen before HIP work.
2. **Primitive HIP:** one-token/tail arithmetic is byte-exact. Full C=8 plain
   and segmented production-head fixtures must be finite and match the
   high-precision oracle within `atol=5e-4, rtol=5e-3` for recurrent output and
   final FP32 state. The ordinary project gate (KL <=0.05, top-1 >=90%) also
   applies; neither check substitutes for the model gates below.
3. **Resources and trace:** expected plain and segment symbols, workgroup 256,
   zero scratch, at most 128 VGPR, and at most 32 KiB LDS. The cached 512 trace
   must put the complete 30-layer GDN stage at <=66 ms before full-model
   promotion timing. Failure stops the lane without routing.
4. **Semantic gate:** all 18 frozen code/general-English/general-Japanese/mixed
   prompts in `gguf_gdn_semantic_gate.py` must be finite, have max KL <=0.05,
   and aggregate top-1 agreement >=99% on identical teacher-forced contexts.
   Thresholds may not be changed after seeing a result.
5. **Free-running trajectory:** every baseline/candidate repetition for the
   ten category-suite prompts must retain the same complete 128-transition
   generated-ID trajectory. The semantic harness currently reports this as a
   diagnostic; GPF-8 promotion treats `free_running_trajectories_exact` as
   required. Aggregate decode wall must be non-regressive.
6. **Full-model speed:** clean, fresh-process, balanced 512/4K timing must reach
   at least **2412.320/2255.080 tok/s**, the matched W7900 llama.cpp HIP floors,
   with the same model fingerprint, TheRock HIP 7.15, cached builds, and exact
   timed IDs. A win at only one shape is rejected.
7. **Publication:** only after all prior gates pass, rerun the defaults-only
   six-shape sweep, update the compact artifact/README/changelog, and promote
   through backend package metadata rather than an engine backend/quant branch.

### GPF-8 result (gfx1100, 2026-07-15)

The candidate passed every pre-model gate. Plain/segmented C=8 and 17-token-tail
fixtures met the frozen primitive bounds; cached traces reported **256 threads,
48 VGPR, zero scratch, and 28 KiB LDS**. A synthetic production-shape trace put
compact prepare + recurrence + RMSNorm/gate across 30 layers at **47.491 ms**,
below the predeclared 66 ms ceiling.

The clean W7900 model gates reject it. Across all 18 frozen prompts, teacher-
forced KL reaches **0.056522 > 0.05**, top-1 is **445/450 = 98.889% < 99%**,
and only **5/18** complete 128-transition free-running trajectories remain
exact. Aggregate decode wall is non-regressive (**-0.046%**) but cannot override
correctness. Clean 1+3 prefill measures **2003.399 tok/s at 512** and
**2280.244 tok/s at 4K**: 4K exceeds its llama.cpp HIP floor by 1.116%, while
512 misses by 16.951%, and both floors were required. The candidate kernels,
wrappers, registry entries, `chain_wy8` selector, and candidate tests were
removed. The float64 CPU oracle remains, and `chain_lds32_direct` stays the
production default. Evidence:
[`2026-07-15-gfx1100-gguf-gdn-chunkwise-wy8-rejected.json`](../benchmarks/results/2026-07-15-gfx1100-gguf-gdn-chunkwise-wy8-rejected.json).

### GPF-9 existing-route result (gfx1100, 2026-07-15)

The prospective peer-aligned gate was applied without changing its 0.05/0.90
thresholds. Existing normalized-Q/K `chain_k2` fails only KL: **0.059031** max,
**445/450 = 98.889%** top-1, and decode wall **-0.063%**. Existing raw-Q/K-
plus-scale register-resident `chain_wave32_tree` also fails KL: **0.068757**
max, **443/450 = 98.444%** top-1, and decode wall **-0.039%**. Free-running
trajectory differences are diagnostic and do not cause either rejection. No
speed gate was run.

The source audit narrows the missing candidate. On gfx1100 llama.cpp HIP uses
wave32, four state rows per lane resident across the serial token loop, and four
value columns per 128-thread block. K2 already has llama.cpp's materialized
normalized inputs but reduces over two waves; `chain_wave32_tree` has one-wave
register residency but applies raw Q/K scales inside the recurrence. GPF-9C's
explicit `chain_peer_wave32` now combines unit-normalized inputs with llama.cpp
HIP `1ebf790cda38`'s XOR reduction, post-reduction scalar decay/output scale,
and four columns per 128-thread block. Plain and segmented CPU-reference
fixtures pass; rocprof reports 40 VGPR and zero LDS/scratch. The clean W7900
18-prompt gate also passes: KL max **0.041737 <= 0.05**, aggregate top-1
**445/450 = 98.889%**, and decode median-wall sum **22524.379 -> 22513.178 ms
(-0.050%)**. Free-running trajectory mismatch remains diagnostic. Production
stays on `chain_lds32_direct`. The subsequent clean speed gate reaches
**2210.729 tok/s** at 512 (**-8.357%** versus llama.cpp HIP) and **2513.374
tok/s** at 4K (**+11.454%**). Because both floors are required, GPF-9C is
rejected for promotion. GPF-9D now tests Vulkan's distinct eight-lane clustered
schedule, whose higher column concurrency specifically targets the failed 512
shape. GPF-9D passes teacher-forced quality at KL max **0.028689** and top-1
**444/450 = 98.667%**, but aggregate decode wall is **22508.498 -> 22508.787
ms (+0.001286%)**. The prospectively frozen rule had zero tolerance, so no
speed gate follows and the peer-schedule lane closes without promotion. Evidence:
[`2026-07-15-gfx1100-gguf-gdn-peer-cluster8-rejected.json`](../benchmarks/results/2026-07-15-gfx1100-gguf-gdn-peer-cluster8-rejected.json).

The final GPF-9C residual profile changes what should be optimized next, without
changing that rejection. A same-W7900 current-stack pp512 trace puts
`chain_peer_wave32` at **207.253 ms / 1645 dispatches** of summed GPU work,
versus llama.cpp HIP at **203.301 ms / 2259 dispatches**. The peer recurrence
itself is effectively at parity (**17.134 versus 16.522 ms**); its separate
prepare + recurrence + RMSNorm/gate chain is only **3.689 ms** over llama.cpp's
fused-family bucket. The decisive trace-span difference is instead idle queue
time between launches: **28.061 versus 8.935 ms**, an excess **19.126 ms** that
explains **82.9%** of the first-to-last-kernel span delta despite hipEngine
launching 614 fewer kernels. The largest bubbles occur at copy-to-RMSNorm,
copy-to-copy, copy-to-selected-Q4, RMSNorm-to-dense-Q8, and copy-to-attention
boundaries. Dense Q8 is the largest positive kernel-family residual
(**+20.825 ms**), while selected Q4/Q5 is already **21.819 ms faster** and
cancels it. Further recurrence algebra, selected Q4/Q5, full attention, and
raw launch-count reduction are therefore not the measured pp512 targets. Keep
production on the exact route; use the explicit peer route only to test
copy-boundary queue-bubble reduction first and a genuinely new dense-Q8 path
second. Evidence:
[`2026-07-15-gfx1100-gguf-gdn-peer-wave32-residual-attribution.json`](../benchmarks/results/2026-07-15-gfx1100-gguf-gdn-peer-wave32-residual-attribution.json).

## Correctness And Promotion Contract

Before editing a kernel, read [`KERNELS.md`](KERNELS.md) and run the required
lineage check. Add the candidate as a registered variant; do not branch on a
backend or quant string in engine/dispatch code. Keep the exact split chain as
the required unfused fallback.

### GDN numerical contracts

Candidates claiming exactness use the six-case byte comparator: 17-token
greeting; repeated token `9707` at 512; 1024/1025 around the segment threshold;
and 4095/4096 around the retained chunk boundary. They require exact sampled
token, FP32 hidden seed, resident Conv/GDN state, and the named all-layer rows.

Algebraically equivalent reassociated candidates instead use the prospectively
adopted peer-aligned contract. The existing PARO schedule is selected explicitly
as `chain_k2`; `chain` remains the exact split. Candidates must pass CPU-
reference primitive numerics and
[`scripts/gguf_gdn_semantic_gate.py`](../scripts/gguf_gdn_semantic_gate.py)
on all 18 category and heldout prompts, with identical teacher-forced token
history, KL <= 0.05, aggregate top-1 >= 90%, finite deterministic execution,
and non-regressive decode. Free-running token equality and state bytes are
reported diagnostics, not blockers. Fused and the exact unfused chain remain
rollback/oracle paths. A historical candidate is never relabeled; promotion
requires a fresh artifact under this predeclared contract.

### Performance

Use one resident session, reset state before every leg, balance candidate/control
ordering, discard at least one warmup per context, and collect at least four
measured repetitions per mode at 512 and 4096. Exact candidates require exact
timed IDs; reassociated candidates require deterministic IDs plus the separate
semantic pass. Retain an admitted improvement if the distribution supports a
real win and neither primary context regresses; do not apply an arbitrary 5%
threshold.

Trace the candidate after a cache-only warmup and record kernel name, dispatch
count, duration, workgroup size, VGPR/SGPR, LDS, and scratch. A kernel-only win
does not promote the default if full-prefill wall loses.

After promotion:

1. repeat current profiles at 512, 4K, and 128K;
2. run the gfx1151 GGUF six-shape README sweep with one warmup and three
   measurements; escalate to five only for a named variance, stability, or
   borderline-decision trigger;
3. validate independently on gfx1100 and gfx1151 before a shared default, or
   register an architecture-specific default with the other architecture
   unchanged;
4. emit the compact result artifact and update `benchmarks/README.md`,
   `benchmarks/CHANGELOG.md`, and `WORKLOG.md`; and
5. add any temporary selector/duplicate path to `docs/REFACTOR.md` with its
   removal condition.

## Reproduction Commands

Use the hardware-local model path and backend:

```bash
export MODEL=/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
export BACKEND=hip_gfx1151              # or hip_gfx1100
export HIPENGINE_BACKEND="$BACKEND"
export HIPENGINE_HIP_ARCH=gfx1151       # or gfx1100
hipcc --version > /tmp/hipengine-hipcc-version.txt
```

Warm and verify the exact current route outside the profiler:

```bash
python3 scripts/qwen35_gguf_bench.py \
  --model "$MODEL" --quant gguf_q4_k_m \
  --prompt-length 512 --decode-tokens 0 --warmup-decode-tokens 0 \
  --warmup-runs 1 --measured-runs 1 --persistent-session \
  --force-bulk-prefill --bulk-prefill-attention-mode bulk \
  --use-wmma-prefill --use-gemv-decode --no-graph-replay-decode \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --json /tmp/gguf-prefill-512-warm.json
```

Then run the same command under `rocprofv3 --kernel-trace`, with warmups set to
zero and cached builds still required:

```bash
rocprofv3 --kernel-trace --output-format csv \
  --output-directory /tmp/gguf-prefill-profile \
  --output-file gguf-prefill-512 -- \
  python3 scripts/qwen35_gguf_bench.py \
    --model "$MODEL" --quant gguf_q4_k_m \
    --prompt-length 512 --decode-tokens 0 --warmup-decode-tokens 0 \
    --warmup-runs 0 --measured-runs 1 --persistent-session \
    --force-bulk-prefill --bulk-prefill-attention-mode bulk \
    --use-wmma-prefill --use-gemv-decode --no-graph-replay-decode \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build --json /tmp/gguf-prefill-512-profile.json
```

Summarize the emitted kernel CSV with:

```bash
python3 scripts/qwen35_gguf_rocprof_summary.py \
  --csv /tmp/gguf-prefill-profile/<kernel-trace.csv> \
  --tokens-prefill 512 --top 40 \
  --json /tmp/gguf-prefill-512-profile-summary.json
```

The existing G2 control command is:

```bash
python3 scripts/gguf_gdn_prefill_compare.py \
  --model "$MODEL" --backend "$BACKEND" --prompt-kind greeting \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --json /tmp/gpf-g2-greeting.json
```

Repeat it with `--prompt-kind repeated --prompt-length` at
`512,1024,1025,4095,4096`; use `--skip-layer-bisect` only for the longer cases
as documented in [`TESTING.md`](TESTING.md). Extend the driver to name the new
candidate rather than overloading `chain`.

The existing balanced G3 control is:

```bash
python3 scripts/gguf_gdn_prefill_ab.py \
  --model "$MODEL" --backend "$BACKEND" --contexts 512,4096 \
  --baseline-mode chain_lds32 --candidate-mode chain_lds32_direct \
  --prompt-token-id 9707 --expected-token-id 9707 \
  --warmups 1 --repetitions 4 --use-wmma-prefill \
  --correctness-artifact /tmp/gpf-g2-exact-matrix.json \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --json /tmp/gpf-g3-interleaved-ab.json
```

The candidate harness must preserve this timing contract. `fused` remains the
default timing baseline and byte-exact correctness oracle; an incremental
candidate must explicitly name the already-promoted baseline so the gate
measures the change under consideration rather than an older implementation.

For the retained six-shape result, run this command once per workload in a
clean detached worktree, changing `WORKLOAD` and the output name. Do **not**
pass all six shapes to one invocation: that creates a max-128K resident session
and invalidates the right-sized memory rows. Merge the six components with
`scripts/merge_readme_sweep_components.py --engine gguf`.

```bash
WORKLOAD=512/128
python3 scripts/qwen35_readme_sweep.py \
  --engine gguf --model "$MODEL" --quant gguf_q4_k_m \
  --backend "$BACKEND" \
  --workloads "$WORKLOAD" \
  --warmup-runs 1 --measured-runs 3 --warmup-decode-tokens 1 \
  --force-bulk-prefill --bulk-prefill-attention-mode bulk \
  --use-wmma-prefill --use-gemv-decode --graph-replay-decode \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --json "/tmp/gpf-${WORKLOAD//\//-}.json"
```

## Context-Compaction Handoff

This is the authoritative pickup state; do not reconstruct it from chat:

- gfx1151 automatic GGUF prefill selects exact
  `chain_lds32_direct_nonvolatile` GDN, Q4T16 `shared_x`, LCP-3 four-wave
  Q8T16 through 65,536 prompt tokens, LCP-4A 256-thread F32 router logits,
  LCP-4B 128-thread prefill router selection, and LCP-M2 stream-ordered
  contiguous metadata through 4,096 prompt tokens. Longer
  prompts restore synchronous metadata; prompts above 64K also restore the
  production Q8T16 wrapper while retaining LCP-4A.
  Before HIP loads, gfx1151 now also defaults to `GPU_MAX_HW_QUEUES=1`; explicit
  user values win, and gfx1100/mixed recognized arches are unchanged. The
  independent gfx1100 defaults are summarized below.
- Causal retained wins are the clean GPF-2D, GPF-3A, GPF-2E, scoped GPF-5A,
  LCP-1, LCP-D1, LCP-2A, scoped LCP-M2, scoped LCP-3, LCP-4A, and LCP-4B gates above. GPF-1, GPF-2A, GPF-2B,
  and GPF-2C are closed rejections; do not
  rerun them without a genuinely different algorithm or contract.
- Correctness is anchored by the six-case byte-exact matrices and the GPF-2E
  ten-prompt gate: 250/250 natural logits and every measured trajectory are
  exact. Do not weaken that contract after seeing a faster tree reduction.
- The current public gfx1151 GGUF prefill row is
  **1294.885/1358.342/1365.720/1034.845/796.083 tok/s** and graph decode is
  **49.041/51.623/52.422/43.572/37.622 tok/s** at
  512/1K/4K/32K/64K. Clean `61a27d72` supplies five independent right-sized
  components on TheRock HIP 7.15, kernel 7.1.3-2-cachyos, and TuneD
  `accelerator-performance`. Current 128K is blocked, not carried forward.
- The calibrated publication protocol remains one discarded warmup plus three
  measured repetitions. Five is a variance/stability/borderline escalation,
  not a default. The current claimed rows' largest prefill/decode stdev over
  median is **0.187%/0.049%**; all 15 measured IDs are `9707`.
- The 128K no-progress state is reduced but not eliminated by one hardware
  queue. The original matched A/B remains valid: four queues fail in warmup,
  while `GPU_MAX_HW_QUEUES=1` once completes warmup+3 at **499.755** and
  **500.210/500.873/500.687 prefill tok/s**, exact IDs, unchanged memory. The
  final current route later fails under one queue after a **509.708 tok/s**
  warmup; launch rollback and SDMA disable also fail their full gates. A clean
  HIP 7.13 versus 7.15 lifecycle matrix rejects downgrade as a fix: 7.13 passes
  two warmup+3 gates at **509.659/499.895 tok/s**, then a third stalls after
  measured pass 1; 7.15 stalls in both controls. Persistent telemetry is
  100%/2.9 GHz at **42-48 W** and every kernel journal is silent. Keep one queue
  as risk reduction, not lifecycle safety; the original findings are posted to
  ROCm#5107 and the cross-stack evidence is
  `benchmarks/results/2026-07-15-gfx1151-128k-hip713-vs-715-lifecycle.json`.
- GPF-4 remains rejected. LCP-3 supersedes GPF-5A through 64K with clean
  512/4K gains of **+0.53%/+1.57%** and 83/83 exact state. The predecessor
  two-wave schedule's same-commit 128K rejection remains **382.041 vs
  392.219 tok/s (-2.59%)**, so package policy conservatively restores production
  above 65,536 tokens.
- LCP-1 is retained on gfx1151 at all six shapes. Its clean 4K body falls
  **954.134 -> 49.790 ms**, its 512/4K focus is **+1.73%/+22.91%**, and the
  final right-sized prefill refresh is **+1.10%..+24.04%** through 64K plus
  **+12.00%** at 128K. gfx1100 independently rejected tiled convolution after
  its post-transfer profile made Conv non-material.
- LCP-2A is promoted on gfx1151: all six state cases and 250/250 natural
  transitions are exact; balanced 512/1K/4K prefill improves
  **+34.76%/+36.63%/+36.58%**, with weighted decode **+0.021%**. gfx1100
  independently progressed from direct LDS32 to its retained peer-wave route.
- LCP-M2 is promoted on gfx1151 through 4K. Clean balanced 512/1K/4K prefill
  improves **+1.56%/+0.90%/+0.53%** and automatic-vs-explicit state is 83/83
  exact at all three shapes. Explicit 128K under the one-queue process policy
  completes a 483.439 tok/s warmup but still enters the low-power no-progress
  state on measured pass 1. Package policy therefore retains synchronous
  metadata above 4K; env `0|1` remains the rollback/diagnostic override.
- LCP-D1 is retained for `num_splits > 256`; shorter reducers remain serial.
  The clean 128K reducer falls **234.714 -> 196.466 us/call (-16.30%)**, and
  right-sized graph decode improves **27.753 -> 28.047 tok/s (+1.06%)**.
- The parity audit and retained outcomes are complete in
  [`LLAMACPP-HIP-PARITY.md`](LLAMACPP-HIP-PARITY.md). LCP-2A closes the
  currently actionable exact-GDN source lane; LCP-3 closes the current
  dense-Q8 shared-layout screen; LCP-4A closes router-logit launch geometry;
  and the post-profile LCP-4B gate closes router selection without risky
  logits+top-k fusion. The fresh exact-decode tranche also closes launch-only
  grouped-GQA and dense-Q8 screens without a new promotion; graph replay remains
  admitted. Any future decode attempt needs a new exact algorithm/layout.
- No benchmark process is intentionally left running. The current 512-64K
  artifact, rollup, changelog, upstream correction, and root README export are
  synchronized. The 128K table cells are explicitly blocked until the same
  warmup+3 protocol completes on a fixed gfx11 stack or stronger production-
  quality workaround.

Keep GPF-4 explicit/default-off and LCP-3 request-scoped through 65,536 tokens,
with GPF-5A as its first rollback. Keep the exact production convolution
fallback while LCP-1's selector survives one release and gfx1100 transfer.
The post-LCP-4A/M2 profile and router-select closure are complete. Further
512-64K prefill work needs a new measured dominant family rather than
speculative fusion. Repeated 128K is owned by the gfx11 firmware/scheduler
blocker; do not retune exact kernels or the grouped-GQA decode body to hide a
prefill lifecycle failure.

- gfx1100 automatic GGUF prefill selects quality-admitted
  `chain_peer_wave32` GDN, Q4T16 `shared_x`, two-wave Q8T16 through 4,096
  prompt tokens, the liveness-aliased prefill arena, and the spill-free
  selected-Q4/Q5/Q6 source schedule. Decode selects the persistent cooperative
  router and LCP-D2 parallel split reduction from 32K; explicit
  `chain_lds32_direct` and serial reduction remain rollback paths.
- The final clean W7900 defaults-only publication is
  **2716.648/3052.541/2953.101/2078.038/1559.878/1037.378 tok/s** prefill,
  **92.833/98.148/100.522/88.240/76.691/62.669 tok/s** graph decode, and
  **21.228/21.295/21.670/22.234/22.879/24.168 GiB** tracked memory from
  512 through 128K. All 18 IDs are `9707`; this closes llama.cpp HIP parity.
- The peer semantic gate remains KL max **0.041737**, top-1 **445/450**, and
  decode wall **-0.103%**. Tail-four Hadamard-group32 INT8 KV passes quality
  and saves 18.75% persistent K/V but remains explicit because 4K/128K speed
  and a 1.002 GiB prefill transient fail promotion.

Keep GPF-4 explicit/default-off. LCP-3 owns the gfx1151 BF16/BF16 Q8T16
prefill aliases through 65,536 prompt tokens with GPF-5A as rollback; GPF-5A
owns gfx1100 only through 4,096 tokens. Request-scoped package policy restores
production above each architecture's bound. The long-context gfx1100 screen
confirms that cap. The gfx1151 512-64K refresh is final; investigate its 128K
lifecycle only with phase markers and bounded lifecycle coverage. On gfx1100,
do not revisit LCP-1 without a new hotspot profile. LCP-5A now makes
`chain_peer_wave32` automatic; `chain_lds32_direct` remains explicit scalar-
exact rollback and rejected GPF-9D remains diagnostic-only pending cleanup.
The post-LCP-4A exact/peer/llama.cpp profile shows that
GPF-9C removes the shipped first-order deficit: peer and llama.cpp summed GPU
work differ by only **0.507 ms**, with **2.564 ms** excess queue idle. Do not
reopen dense-Q8, selected-MoE, Conv, full-attention, or generic launch-count
experiments without genuinely new measured evidence. LCP-5A supplied that
evidence and is promoted on gfx1100. HIP 7.2 compiled the Q5T16 BF16 body at
256 VGPR, 176 private bytes, and 75 spills; rolling only the outer quant-block
loop gives 91 static VGPR and zero private bytes/spills while retaining the
inner K16 unroll and all operation order. Q4/Q5/Q6 BF16/FP16 CPU-oracle GPU
fixtures pass byte-exact. Matched real routing moves Q5 **50.863 -> 31.725 ms**,
Q6 **3.475 -> 2.630 ms**, and selected-MoE **98.538 -> 78.358 ms**. Cached
pp512 trace moves Q5 **51.009 -> 29.544 ms** and the complete peer route from
**203.808/215.307 -> 184.513/194.886 ms** kernels/span, now faster than the
matched llama.cpp HIP trace. The clean 18-prompt semantic contract remains KL
max **0.041737**, top-1 **445/450**, deterministic/finite, with aggregate decode
wall **-0.103%**. Clean selector-unset liveness-aliased 512/4K reaches
**2588.231/2757.752 tok/s**, clears both floors by **7.29%/22.29%**, and
restores the tracked peak from the dedicated peer diagnostic's 22.995 GiB
to **21.670 GiB**. gfx1100 package policy now selects `chain_peer_wave32`;
explicit `chain_lds32_direct` remains the scalar-exact rollback, and gfx1151
keeps its independently admitted `chain_lds32_direct_nonvolatile` default. Exact state and free-
running token identity remain diagnostics under the prospectively frozen peer
contract.

## Document Ownership

- [`SOL-OPTIMIZATION.md`](SOL-OPTIMIZATION.md) owns cross-project ordering;
  this document expands R5 only.
- [`GGUF.md`](GGUF.md) owns loader/runtime and correctness history.
- [`TUNING-gguf.md`](TUNING-gguf.md) is the historical tuning notebook; its
  June profiles are not current-route selection evidence.
- [`PREFILL.md`](PREFILL.md) owns the PARO native-prefill design.
- [`LLAMACPP-HIP-PARITY.md`](LLAMACPP-HIP-PARITY.md) owns the post-GPF-5A
  source/profile comparison and ranked AR parity work.
- [`MTP-LLAMACPP-PARITY.md`](MTP-LLAMACPP-PARITY.md) owns decode/MTP timing
  boundaries, not AR prefill optimization.
- [`PARO-GGUF-MTP-TRANSFER.md`](PARO-GGUF-MTP-TRANSFER.md) owns cross-path MTP
  transfer safety; quant-specific kernels do not transfer by analogy.
- [`BENCHMARK.md`](BENCHMARK.md) and [`TESTING.md`](TESTING.md) own evidence and
  promotion gates.
- [`ROOFLINE.md`](ROOFLINE.md) and
  [`ROOFLINE-gfx1151.md`](ROOFLINE-gfx1151.md) own architecture constraints;
  measured profiles still select the work.
