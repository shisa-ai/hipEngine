# GGUF Prefill Optimization

Last updated: 2026-07-15.

Status: `SOL-R5` plus the bounded GGUF half of `SOL-R6` are retained and
published on gfx1151.
`GPF-1` exact value-column tiling
and `GPF-2A` non-resident wave sharding are rejected. Register-resident
tree-reduced `GPF-2B` is fast but fails the predeclared natural greedy-
trajectory gate. Register-resident ordered `GPF-2C` retains byte identity but
is 12.98%-14.58% slower than fused at 512/1K/4K. Scalar-exact, LDS-resident
`GPF-2D` now passes its clean six-case byte-exact state matrix and balanced
512/4K wall gate. It improves clean prefill by 79.09%/68.44%, with all timed
tokens exact. The ten-prompt gate is also exact across all 250 checked logits
and every timed trajectory, with decode +0.023%. GPF-2D is accepted for a
gfx1151-scoped `auto` promotion and is now the scoped default; gfx1100 remains
fused pending an independent transfer gate. A clean max-context six-shape
stress run confirms the automatic route from 512 through 128K. Exact Q4T16
shared-activation `GPF-3A` also passes its clean full-model gate: 512/1K/4K
prefill improves **747.764/804.150/687.676 -> 771.027/823.624/701.042 tok/s**
with byte-identical logits and trajectories and neutral aggregate decode.
`shared_x` is now the gfx1151-scoped automatic route; gfx1100 stays baseline.
A clean selector-unset four-run confirmation at promoted commit `431fe1e4`
reproduces **774.653/823.149/701.389 tok/s** with stable IDs; it is a focus
diagnostic, not the final right-sized memory rollup. `GPF-2E` removes
GDN Q/K/V scratch materialization and the eightfold duplicate Q/K norm work on
the production 4-K-head/32-V-head shape. Its clean current-default/direct A/B
improves 512/1K/4K prefill **776.428/825.319/700.824 ->
823.093/889.209/744.577 tok/s** (**+6.01%/+7.74%/+6.24%**). The six-case
full-model matrix and all 250 natural logit transitions are byte-exact; every
timed decode trajectory matches and weighted decode is **+0.075%**. GPF-2E is
now the gfx1151-scoped automatic route; gfx1100 remains fused. A clean
selector-unset focus confirmation at `b8949477` reproduces
**821.755/897.160/750.896 tok/s** at 512/1K/4K with stable IDs. That first
clean right-sized 1+3 publication window recorded
**819.641/893.266/752.308/640.096/540.850/387.334 tok/s**. Scoped two-wave
Q8T16 GPF-5A then refreshed 512-64K to
**889.904/919.598/762.940/648.948/546.296 tok/s** while restoring the
production wrapper at 128K. The post-GPF-5A llama.cpp parity tranche is now
complete: exact tiled convolution LCP-1 and long-split reducer LCP-D1 publish
**906.979/929.724/946.366/778.371/636.330/433.811 prefill tok/s** and
**49.061/51.569/52.432/43.543/37.562/28.047 graph-decode tok/s** at
512/1K/4K/32K/64K/128K. All 18 measured IDs are `9707`; tracked memory is
unchanged; maximum prefill/decode stdev over median is **0.140%/0.113%**. The
separate 128K lifecycle-soak issue is not reproduced inside this calibrated
1+3 window. LCP-2A exact cacheable GDN, scoped LCP-3 four-wave Q8T16, and
LCP-4A's 256-thread F32 router are now promoted. The final LCP-M2 metadata gate
also retains the exact short-context win: stream-ordered metadata is automatic
through 4K, while longer requests keep synchronous metadata after the explicit
128K one-queue route still reproduced the low-power no-progress state. The
post-LCP-M2 4K profile closes the remaining router-select question with LCP-4B:
128 threads cuts that exact family 70.17% and improves 512/4K wall 0.34%/0.36%;
the faster 64-thread primitive is rejected by full-model state.

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

This is a scheduling/layout change, not a math relaxation. A llama.cpp-style
wave-reduced recurrence is the next diagnostic lane if simple exact tiling is
insufficient, but it is not promotion-eligible unless it also satisfies the
current exact state contract or that contract is explicitly changed.

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

The numerical-contract decision is now explicit: do not relax the gate after
observing a failure. On all ten prompts from the four-category suite, only
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
slower than fused. The next tranche must reprofile the published route rather
than treating that historical 512 share as a permanent exclusion.

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

Clean 2026-07-12 hipEngine `8116c453`, TheRock HIP 7.15; llama.cpp HIP
`1ebf790cd` build 9648. Values are medians after two discarded hipEngine
warmups and five measured repetitions per shape.

| Workload | hipEngine GGUF | llama.cpp HIP | GGUF / llama HIP | hipEngine PARO | GGUF / PARO |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512/128 | 644.719 | 2412.320 | 26.7% | 2917.732 | 22.1% |
| 1K/128 | 676.177 | 2389.670 | 28.3% | 2995.876 | 22.6% |
| 4K/128 | 677.618 | 2255.080 | 30.0% | 2943.038 | 23.0% |
| 32K/128 | 628.364 | 1667.640 | 37.7% | 2108.868 | 29.8% |
| 64K/128 | 572.612 | 1291.820 | 44.3% | 1584.131 | 36.1% |
| 128K/128 | 484.212 | 891.949 | 54.3% | 1056.252 | 45.8% |

### Radeon 8060S / gfx1151

The hipEngine GGUF column is the clean 2026-07-13 right-sized 1+3 rollup at
`28b45d38`, TheRock HIP 7.15, kernel 7.1.3-2-cachyos, and TuneD
`accelerator-performance`. llama.cpp remains the clean July 11 matched
reference; PARO is the separately retained HIP 7.15 recovery.

| Workload | hipEngine GGUF | llama.cpp HIP | GGUF / llama HIP | hipEngine PARO | GGUF / PARO |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512/128 | 819.641 | 1061.260 | 77.2% | 1140.101 | 71.9% |
| 1K/128 | 893.266 | 1043.230 | 85.6% | 1208.343 | 73.9% |
| 4K/128 | 752.308 | 1009.240 | 74.5% | 1089.031 | 69.1% |
| 32K/128 | 640.096 | 743.547 | 86.1% | 906.145 | 70.6% |
| 64K/128 | 540.850 | 573.611 | 94.3% | 716.775 | 75.5% |
| 128K/128 | 387.334 | 390.441 | 99.2% | 474.641 | 81.6% |

The published route does include the measured GDN improvement. Its additional
ratio narrowing at long context does not imply another context-dependent GDN
speedup: attention and other context-dependent work grow for every engine and
change the fraction of total wall.

### Decode Is The Control

Against llama.cpp HIP, GGUF decode is close or faster at short/mid context and
has a much smaller long-context deficit than prefill:

| GPU | Workload | hipEngine GGUF | llama.cpp HIP | GGUF delta |
| --- | --- | ---: | ---: | ---: |
| W7900 / gfx1100 | 512/128 | 89.873 | 80.756 | +11.3% |
| W7900 / gfx1100 | 4K/128 | 96.551 | 79.768 | +21.0% |
| W7900 / gfx1100 | 128K/128 | 56.745 | 60.933 | -6.9% |
| Radeon 8060S / gfx1151 | 512/128 | 49.067 | 50.939 | -3.7% |
| Radeon 8060S / gfx1151 | 4K/128 | 52.498 | 50.126 | +4.7% |
| Radeon 8060S / gfx1151 | 128K/128 | 27.753 | 32.114 | -13.6% |

That control makes a model-wide GGUF loader, quant, or HIP runtime explanation
unlikely. The remaining throughput deficit is specific to bulk prefill
execution.

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
it does not prove that its normalized-Q/K materialization or reduction tree is
acceptable. The useful recovery question is how much of that parallelism can
be recovered while keeping the production arithmetic contract.

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
| Fast K2 split | recurrence grid `num_v_heads × head_v_dim`, block 64 | Two wave32 shards reduce the 128 state rows | PARO path and historical GGUF idea; not GGUF target-exact |

The serial dependency across prompt tokens is real, but value columns are
independent until RMSNorm. The exact split already creates the synchronization
boundary needed to schedule those columns across more blocks without changing
the per-column recurrence.

### PARO

PARO's current implementation in
[`qwen35_paro.py`](../hipengine/runtime/qwen35_paro.py) uses prepare +
`qwen35_gdn_prefill_recurrent_k2_f32` + RMSNorm/gate. It assigns an independent
block to each value column and distributes the 128 state rows over two wave32
reductions. That is structurally much more parallel than the GGUF fused path,
but PARO's quant/model contract is different and its reduction order is not an
oracle for byte-exact GGUF state.

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

The local instrumentation tree defaults fused GDN on and the CUDA/HIP backend
supports the op, but the retained topline artifact is not a kernel trace. Before
using this source difference as causal proof, capture a llama.cpp HIP trace and
confirm the fused GDN kernel is active in the measured prefill phase.

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
| Restore the old K2 chain | It changes normalized-Q/K and contraction order and failed target/serial recurrent parity | Invalid under the current contract |
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
| 3 | `GPF-2B` | **Default rejected:** register-resident tree wins wall by 2.266x/2.058x but keeps only 3/10 complete natural 128-step trajectories | Keep as an explicit diagnostic; do not weaken the predeclared gate after failure |
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

There is no invented minimum full-model percentage. Under the project evidence
policy, every exact, measured, non-regressive improvement is retainable. The
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
rejects it: only **3/10** prompts keep the full fused 128-transition trajectory.
`auto` therefore remains fused, and `chain_wave32_tree` remains an explicit
diagnostic rather than a promotion candidate.

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

## Correctness And Promotion Contract

Before editing a kernel, read [`KERNELS.md`](KERNELS.md) and run the required
lineage check. Add the candidate as a registered variant; do not branch on a
backend or quant string in engine/dispatch code. Keep the exact split chain as
the required unfused fallback.

### GDN exactness

Extend the existing comparator so `fused`, `chain`, and the named candidate can
be selected without an untracked source edit. Require:

- 17-token real greeting;
- repeated token `9707` at 512;
- 1024 and 1025 around the segment threshold;
- 4095 and 4096 around the retained 1024-row chunk boundary;
- exact sampled token, FP32 hidden seed, and all resident Conv/GDN state for all
  six cases; and
- all-layer output identity at greeting and 512.

The repository-wide new-kernel floor remains KL <= 0.05 and top-1 >= 90%, but
that floor does not silently supersede the stronger current GGUF GDN state
contract. `GPF-2B` passes the floor but not byte identity. Changing the default
therefore requires an explicit decision recorded here, a multi-prompt greedy
trajectory/decode gate, and retention of fused plus the exact unfused chain as
rollback/oracle paths.

The executable promotion gate is
[`scripts/gguf_gdn_trajectory_gate.py`](../scripts/gguf_gdn_trajectory_gate.py).
It uses all prompts in `mtpbench-code-general-ja.jsonl`, covering
`code/general_en/general_ja/mixed_ja_en`. For each prompt it compares baseline
and candidate own-token greedy prefill samples plus 24 decoded transitions,
requiring exact IDs and the project KL/top-1 thresholds at every transition.
It then runs two balanced 128-transition production decode windows per mode and
prompt. `--baseline-mode` defaults to fused for historical gates; incremental
candidates name the shipped exact route explicitly. Candidate decode passes
only when all measured trajectories are exact and the sum of per-prompt
candidate median walls does not exceed baseline; there is no percentage
regression allowance.

### Performance

Use one resident session, reset state before every leg, balance candidate/control
ordering, discard at least one warmup per context, and collect at least four
measured repetitions per mode at 512 and 4096. Record tokens and require exact
timed IDs. Retain an exact improvement if the distribution supports a real win
and neither primary context regresses; do not apply an arbitrary 5% threshold.

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
  user values win, and gfx1100/mixed recognized arches are unchanged. gfx1100
  remains on its prior fused/baseline/production routes pending hardware-local
  transfer evidence.
- Causal retained wins are the clean GPF-2D, GPF-3A, GPF-2E, scoped GPF-5A,
  LCP-1, LCP-D1, LCP-2A, scoped LCP-M2, scoped LCP-3, LCP-4A, and LCP-4B gates above. GPF-1, GPF-2A, GPF-2B,
  and GPF-2C are closed rejections; do not
  rerun them without a genuinely different algorithm or contract.
- Correctness is anchored by the six-case byte-exact matrices and the GPF-2E
  ten-prompt gate: 250/250 natural logits and every measured trajectory are
  exact. Do not weaken that contract after seeing a faster tree reduction.
- The public gfx1151 GGUF prefill row is now
  **906.979/929.724/946.366/778.371/636.330/433.811 tok/s** and graph decode is
  **49.061/51.569/52.432/43.543/37.562/28.047 tok/s** at
  512/1K/4K/32K/64K/128K. Clean `71e61524` supplies all six independent
  right-sized components on TheRock HIP 7.15, kernel 7.1.3-2-cachyos, and
  TuneD `accelerator-performance`.
- The calibrated publication protocol remains one discarded warmup plus three
  measured repetitions. Five is a variance/stability/borderline escalation,
  not a default. The LCP sweep's largest prefill/decode stdev over median is
  only **0.140%/0.113%**; all 18 measured IDs are `9707`.
- The 128K no-progress state is now mitigated by a matched hardware-queue A/B.
  On clean current production, ROCm's default four queues enter the state in the
  first warmup at 100%/2.9 GHz but only 41-43 W; four host dumps remain in the
  same synchronous metadata H2D and the kernel journal has no fault. Changing
  only `GPU_MAX_HW_QUEUES=1` completes warmup+3 at **499.755 warmup** and
  **500.210/500.873/500.687 prefill tok/s**, with exact IDs, unchanged memory,
  and no low-power collapse. Treat this as a gfx11 scheduler/firmware workaround,
  not a kernel fix; evidence is posted to ROCm#5107.
- GPF-4 remains rejected. LCP-3 supersedes GPF-5A through 64K with clean
  512/4K gains of **+0.53%/+1.57%** and 83/83 exact state. The predecessor
  two-wave schedule's same-commit 128K rejection remains **382.041 vs
  392.219 tok/s (-2.59%)**, so package policy conservatively restores production
  above 65,536 tokens.
- LCP-1 is retained on gfx1151 at all six shapes. Its clean 4K body falls
  **954.134 -> 49.790 ms**, its 512/4K focus is **+1.73%/+22.91%**, and the
  final right-sized prefill refresh is **+1.10%..+24.04%** through 64K plus
  **+12.00%** at 128K. gfx1100 stays on the production convolution pending its
  hardware-local transfer gate.
- LCP-2A is promoted on gfx1151: all six state cases and 250/250 natural
  transitions are exact; balanced 512/1K/4K prefill improves
  **+34.76%/+36.63%/+36.58%**, with weighted decode **+0.021%**. gfx1100 stays
  fused pending its own transfer gate.
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
- No benchmark process is intentionally left running. The one-queue stability
  artifact, rollup, upstream comment, and root README export are complete. The
  public six-shape throughput table still carries the earlier right-sized row;
  its next refresh must use the new gfx1151 queue default and current LCP-4B
  selector-unset production path.

Keep GPF-4 explicit/default-off and LCP-3 request-scoped through 65,536 tokens,
with GPF-5A as its first rollback. Keep the exact production convolution
fallback while LCP-1's selector survives one release and gfx1100 transfer.
The post-LCP-4A/M2 profile and router-select closure are complete. Further
prefill work needs a new measured dominant family rather than speculative
fusion. Decode starts from the measured 128K grouped-GQA body, not from a
wholesale llama.cpp port or the rejected GDN tree.

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
