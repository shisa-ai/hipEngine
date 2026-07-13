# GGUF Prefill Optimization

Last updated: 2026-07-13.

Status: active `SOL-R5` implementation log. `GPF-1` exact value-column tiling
and `GPF-2A` non-resident wave sharding are rejected. Register-resident
tree-reduced `GPF-2B` is fast but fails the predeclared natural greedy-
trajectory gate. Register-resident ordered `GPF-2C` retains byte identity but
is 12.98%-14.58% slower than fused at 512/1K/4K. `auto` remains fused. The next
candidate is `GPF-2D`: retain each value column's scalar exact contraction while
keeping a 32- or 64-column state tile in LDS across the token loop, avoiding
both per-token global state traffic and ordered wave shuffles.

Scope: Qwen3.6-35B-A3B `UD-Q4_K_M`, BF16 KV, single-request bulk prefill on
`hip_gfx1100` and `hip_gfx1151`. This is not a general GGUF plan and does not
replace the separate decode, MTP, concurrency, or long-context memory plans.

## Decision

The current GGUF prefill gap is primarily a linear-attention GDN recurrence
problem. The production-exact fused kernel launches one 128-thread block per
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

For the relaxed-tree diagnostic only, that speedup changes the next bottleneck:
a cache-clean 512 trace attributes
only **61.411 ms (11.45%)** to the new recurrence, versus **158.223 ms** dense
Q8 WMMA and **173.023 ms** selected-MoE Q4+Q5 WMMA. Do not keep micro-tuning
GDN launch geometry: llama.cpp's four-wave/128-thread launch gains 0.39% at
512 but regresses 1K/4K by 1.80%/1.84%, so the balanced eight-wave/256-thread
schedule remains selected.

Do not start with AOTriton tuning, generic chunk sweeps, graph capture, compiler
flags, or another attempt to enable WMMA. Full-attention prefill is only 0.54%
of the current 512-token GPU profile, WMMA prefill is already enabled, and the
existing exact split chain is slower than fused.

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

Clean 2026-07-11 GGUF/llama.cpp refresh at hipEngine `d1231ee0`, TheRock HIP
7.13; current PARO values are the separately retained HIP 7.15 recovery.

| Workload | hipEngine GGUF | llama.cpp HIP | GGUF / llama HIP | hipEngine PARO | GGUF / PARO |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512/128 | 430.767 | 1061.260 | 40.6% | 1140.101 | 37.8% |
| 1K/128 | 437.467 | 1043.230 | 41.9% | 1208.343 | 36.2% |
| 4K/128 | 403.946 | 1009.240 | 40.0% | 1089.031 | 37.1% |
| 32K/128 | 369.942 | 743.547 | 49.8% | 906.145 | 40.8% |
| 64K/128 | 334.395 | 573.611 | 58.3% | 716.775 | 46.7% |
| 128K/128 | 270.601 | 390.441 | 69.3% | 474.641 | 57.0% |

The narrowing ratio at long context does not show that GDN has improved. The
GDN recurrence grows linearly while attention and other context-dependent work
grow for every engine, reducing its fraction of total wall.

### Decode Is The Control

Against llama.cpp HIP, GGUF decode is close or faster at short/mid context and
has a much smaller long-context deficit than prefill:

| GPU | Workload | hipEngine GGUF | llama.cpp HIP | GGUF delta |
| --- | --- | ---: | ---: | ---: |
| W7900 / gfx1100 | 512/128 | 89.873 | 80.756 | +11.3% |
| W7900 / gfx1100 | 4K/128 | 96.551 | 79.768 | +21.0% |
| W7900 / gfx1100 | 128K/128 | 56.745 | 60.933 | -6.9% |
| Radeon 8060S / gfx1151 | 512/128 | 49.536 | 50.939 | -2.8% |
| Radeon 8060S / gfx1151 | 4K/128 | 52.999 | 50.126 | +5.7% |
| Radeon 8060S / gfx1151 | 128K/128 | 27.862 | 32.114 | -13.2% |

That control makes a model-wide GGUF loader, quant, or HIP runtime explanation
unlikely. The large failure is specific to bulk prefill execution.

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
| 0 | `GPF-M0` | **In progress:** clean gfx1151 512 fused trace captured; 4K/128K and matched llama.cpp traces remain | Confirm GDN share and llama fused-GDN dispatch before attributing a candidate win |
| 1 | `GPF-1` | **Rejected:** exact split recurrence with 64/32 value columns per block | Both full wall and tile64 recurrence regress; do not promote |
| 2 | `GPF-1B` | **Skipped:** fuse GPF-1 prepare/materialization only if recurrence wins | Tile64 recurrence itself loses 8.58%, so fusion cannot close this lane |
| 3 | `GPF-2B` | **Default rejected:** register-resident tree wins wall by 2.266x/2.058x but keeps only 3/10 complete natural 128-step trajectories | Keep as an explicit diagnostic; do not weaken the predeclared gate after failure |
| 4 | `GPF-2C` | **Rejected:** register-resident exact ordered-wave recurrence | Byte-exact, but focused 512/1K/4K prefill loses 12.98%-14.58% and recurrence loses 16.86% |
| 5 | `GPF-2D` | **Next:** scalar-exact value columns with recurrent state resident in a 32- or 64-column LDS tile | RED/GREEN byte identity, then 512 screen and 1K/4K only if viable |
| 6 | `GPF-M1` | **Tree diagnostic complete:** relaxed GDN is 11.45%; default-path 4K/128K profiles remain | Select later buckets only after an exact/default GDN decision |
| 7 | `GPF-3` | Optimize dense-Q8 and/or selected-MoE WMMA named by the relaxed-tree profile | Family-local correctness A/B and full-wall win; retain decode non-regression |
| 8 | `GPF-4` | Revisit AOTriton queue isolation/query chunks at 4K-128K if attention becomes material | Same-shape exact A/B; no short-context regression |
| 9 | `GPF-5` | Router/glue/launch fusion or host submission work | Only after device-family residual is measured as material |
| 10 | `GPF-6` | Chunked/token-parallel GDN prefix algorithm | High-effort fallback only if column tiling and an approved reduction path leave material GDN wall |

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
`code/general_en/general_ja/mixed_ja_en`. For each prompt it compares the fused
and candidate own-token greedy prefill sample plus 24 decoded transitions,
requiring exact IDs and the project KL/top-1 thresholds at every transition.
It then runs two balanced 128-transition production decode windows per mode and
prompt. Candidate decode passes only when all measured trajectories are exact
and the sum of per-prompt candidate median walls does not exceed fused; there
is no percentage regression allowance.

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
2. run the six-shape README sweep with two warmups and five measurements;
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
  --prompt-token-id 9707 --expected-token-id 9707 \
  --warmups 1 --repetitions 4 --use-wmma-prefill \
  --correctness-artifact /tmp/gpf-g2-exact-matrix.json \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --json /tmp/gpf-g3-interleaved-ab.json
```

The candidate harness must preserve this timing contract and record all three
named modes until one is selected.

For the retained six-shape result, use the same boundary as the current
leaderboard:

```bash
python3 scripts/qwen35_readme_sweep.py \
  --engine gguf --model "$MODEL" --quant gguf_q4_k_m \
  --backend "$BACKEND" \
  --workloads 512/128 1K/128 4K/128 32K/128 64K/128 128K/128 \
  --warmup-runs 2 --measured-runs 5 --warmup-decode-tokens 1 \
  --force-bulk-prefill --bulk-prefill-attention-mode bulk \
  --use-wmma-prefill --use-gemv-decode --graph-replay-decode \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --json /tmp/gpf-readme-sweep.json
```

## Document Ownership

- [`SOL-OPTIMIZATION.md`](SOL-OPTIMIZATION.md) owns cross-project ordering;
  this document expands R5 only.
- [`GGUF.md`](GGUF.md) owns loader/runtime and correctness history.
- [`TUNING-gguf.md`](TUNING-gguf.md) is the historical tuning notebook; its
  June profiles are not current-route selection evidence.
- [`PREFILL.md`](PREFILL.md) owns the PARO native-prefill design.
- [`MTP-LLAMACPP-PARITY.md`](MTP-LLAMACPP-PARITY.md) owns decode/MTP timing
  boundaries, not AR prefill optimization.
- [`PARO-GGUF-MTP-TRANSFER.md`](PARO-GGUF-MTP-TRANSFER.md) owns cross-path MTP
  transfer safety; quant-specific kernels do not transfer by analogy.
- [`BENCHMARK.md`](BENCHMARK.md) and [`TESTING.md`](TESTING.md) own evidence and
  promotion gates.
- [`ROOFLINE.md`](ROOFLINE.md) and
  [`ROOFLINE-gfx1151.md`](ROOFLINE-gfx1151.md) own architecture constraints;
  measured profiles still select the work.
