# Laguna S 2.1 Decode Gap Analysis — W7900 / UD-Q2_K_XL

Status: diagnostic complete. No runtime default changes in this analysis.

Scope: resident batch-1 autoregressive decode of
`Laguna-S-2.1-UD-Q2_K_XL.gguf` on one AMD Radeon Pro W7900 (`gfx1100`). This
explains the measured gap between llama.cpp Vulkan and hipEngine, audits the
Qwen3.x optimization history for missed transfers, and ranks future work. It
does **not** replace the canonical benchmark or make an apples-to-apples
cross-engine throughput claim.

Compact evidence:
[`2026-07-24-gfx1100-laguna-q2-xl-decode-gap-analysis.json`](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-decode-gap-analysis.json).

## Executive answer

The supplied result is real and reproducible:

```text
llama.cpp Vulkan tg128: 93.67 +/- 0.16 tok/s
independent rerun:      94.513 +/- 0.141 tok/s
hipEngine D12 h32:      48.987 tok/s
```

`llama-bench tg128` and hipEngine's canonical h32 suite are not the same
protocol, so `94.513 / 48.987 = 1.929x` is diagnostic rather than a formal
product-throughput ratio. However, the gap is not explained by prompt depth,
sampling, Python, or one missing launch flag:

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

A formal cross-engine generated-token ratio would require a llama.cpp harness
using the same prompts, greedy sampler/readback, KV contract, and output
accounting. That is not necessary to establish the current bottleneck.

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

### 6.2 What the audit did find missing

There are open implementation-quality gaps, but they are format-specific:

1. **Raw IQ3_XXS selected-down schedule.** D12 still spends 2.131 ms/token here.
   Vulkan groups selected IQ3 down and route weighting in a materially shorter
   perturbed window. hipEngine has already fixed addressing, local size, and
   weighting, but has not performed a Vulkan/RADV-vs-HIP ISA and multi-output
   schedule port comparable to D12's Q5 work.
2. **Compound raw quant throughput.** IQ2, Q5 output, Q5 query/gate, and IQ3
   together dominate the device. Each needs actual-weight code-object comparison
   against the Vulkan shader; another generic fusion cannot substitute.
3. **Attention algorithm, not head-dimension remapping.** D4 was a large win,
   but SWA is still 2.153 ms at short depth and 13.1 ms at a full window.
   D10 token8 helped mechanically but failed the complete gate. A future route
   needs a new online/split algorithm or a declared quality-gated reassociation.
4. **Quality-gated Vulkan arithmetic.** Exact bit identity to old HIP boundaries
   prevents direct adoption of subgroup reduction/order and some post-op fusion.
   A separately registered quality lane may be required to approach Vulkan,
   subject to KL <= 0.05, top-1 >= 90%, and the complete category/heldout gate.
5. **Submission only after device work.** D12 has 3.08 ms between kernel sum and
   dispatch span, but deleting all of it still caps at 60.66 tok/s. A new
   scheduler is useful only after the kernel sum falls substantially.

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

This list is for a future AR reopening. It does not supersede the queued
Laguna DFlash/MTP campaign, and rejected D10–D17 code must not simply be
restored unchanged.

### P0 — Raw IQ3 selected-down Vulkan/HIP ISA and exact schedule audit

Why first:

- 2.131 ms/token, 12.92% of D12 kernel sum;
- current body is scratch-free and already has wave-uniform/local128 fixes, so
  the next premise must come from instruction/decode structure;
- Vulkan's perturbed selected-down group is the clearest remaining relative
  advantage, while its IQ2 group is not faster than hipEngine's.

Screen:

1. Extract HIP HSACO and RADV final ISA for the exact K1024/N3072/top-10 shape.
2. Compare codebook/grid decode, scalar vs vector address chains, load width,
   waitcnts, VGPR, subgroup reduction, and output-row ownership.
3. Prototype one source-backed multi-output/wave-owned exact sibling only if
   the ISA gives a concrete mechanism.
4. Require actual layer weights, cold/distinct routes, BF16-bit equality,
   scratch0, and a family win before a full-model run.

A 0.5-ms family saving would be useful for 50 tok/s, but cannot be extrapolated
to Vulkan parity.

### P1 — Quality-gated raw-IQ/Q5 subgroup lane

The exact lane has harvested most launch-preserving changes. Build a separate
registered candidate that follows Vulkan's reduction/FMA association and F16
or BF16 boundary intentionally, rather than claiming bit identity. Start with
one top family and compare logits against the CPU/source oracle.

Promotion requires:

- KL <= 0.05 and top-1 >= 90% on fixture inputs;
- full ten-prompt category plus heldout teacher-forced/free-running quality;
- no prompt-conditioned route or benchmark-specific logic;
- exact lifecycle/KV ownership even when arithmetic is quality-gated;
- retained exact D12 fallback.

### P2 — New SWA/global online attention algorithm

Do not repeat one-wave/two-wave head remaps or token8 unchanged. The next design
must change the algorithm:

- more independent slot batches with fewer block barriers;
- online/split softmax with bounded partials;
- exact `KVLiveSpans`, ring wrap, absolute positions, and eviction semantics;
- separate global and SWA crossover policies;
- attention-output oracle before model benchmarking.

At a full 512-token SWA window this is mandatory; near 4K the global scan is an
additional independent requirement.

### P3 — Quant metadata sidecar/repack only after ISA evidence

Qwen Q4 benefited from T16/repacked layouts, but Laguna's 39.68-GB model leaves
limited W7900 headroom and Vulkan is already fast on raw layouts. A sidecar is
justified only if P0 identifies repeatedly decoded scales/codebook metadata
that can be compacted without duplicating full weights. Record the complete
resident-byte cost and reject any route that causes paging or loses cold-route
performance.

### P4 — Device scheduler/fusion after kernel sum falls

A native graph or command-packet owner can attack the 3.08-ms short
span-minus-kernel window, but D8 and D16 invalidate the old mechanisms. Reopen
only with a new premise such as reusable dynamic command buffers or a
fused-kernel set whose body already wins independently. The acceptance target
must be measured queue gaps plus unchanged device kernels, not launch count
alone.

## 9. Do not chase without new evidence

- **Unchanged D8 graph replay:** measured regression and removed.
- **Unchanged D10/D13/D14/D15/D17 boundaries:** all have complete rejection
  artifacts; positive diagnostic rows do not waive category/TTFT failures.
- **More C-side packets:** D16 proved the visible gaps are queue spacing, not
  ctypes transition cost.
- **Q8_1/MMVQ/dp4a as a default premise:** both engines' controls reject it for
  this short Laguna path.
- **Wave64 as a blanket switch:** hipEngine's default is intentionally wave32;
  prior controlled wave64 probes were generally slower and Vulkan subgroup size
  alone is not a portable reason.
- **LDS staging by default:** the successful D12 change removed LDS/barriers;
  prior Qwen/Laguna evidence repeatedly shows occupancy and barrier cost can
  exceed reuse.
- **Launch-count-only fusion:** D11 and D13 are direct counterexamples.
- **Single-prompt promotion:** all future acceptance remains the complete
  category/heldout suite under the repository anti-gaming rule.

## 10. Evidence map

| Question | Evidence |
| --- | --- |
| Is 93.67 tok/s reproducible? | [`...llamacpp-vulkan-review.json`](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-llamacpp-vulkan-review.json) |
| What is the retained hipEngine row? | [`...d12-q5-wave32x2-retained.json`](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-d12-q5-wave32x2-retained.json) |
| What dominates D12? | D12 clean profile plus [`...d9-residual-profile.json`](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-d9-residual-profile.json) with D12 leaf replacements |
| What did D0–D17 retain/reject? | [`LAGUNA.md`](LAGUNA.md), “Laguna Q2 XL Decode Optimization Campaign” |
| Is IQ2 already tuned? | [`OPTIMIZE-KERNEL-IQ2_XS.md`](OPTIMIZE-KERNEL-IQ2_XS.md) |
| Why is Qwen competitive? | [`...gguf-final-optimization-sweep.json`](../benchmarks/results/2026-07-16-gfx1100-gguf-final-optimization-sweep.json) |
| Which Qwen tactics were considered? | [`OPTIMIZE.md`](OPTIMIZE.md), [`OPTIMIZE-DENSE.md`](OPTIMIZE-DENSE.md), Q3/Q4 artifacts, and `WORKLOG.md` |
| Compact conclusions and logger hashes | [`...decode-gap-analysis.json`](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-decode-gap-analysis.json) |

## Bottom line

The approximately 2x headline gap is genuine enough to guide engineering even
though the two public timing protocols differ. We are **not** losing 2x to
Python, sampling, graph replay, a missing compiler flag, or one unfused router.
The clean GPU trace proves otherwise.

hipEngine has already transferred the broad Qwen playbook and improved Laguna
from **19.596 to 48.987 tok/s**. The remaining route to llama.cpp-class speed is
narrower and harder: raw IQ3/IQ2/Q5 code generation, a stronger attention
algorithm, and eventually a scheduler that compounds independently faster
kernels. Matching Vulkan requires large device-work reductions across several
families; launch cleanup alone can move 49 toward 51, not toward 94.
