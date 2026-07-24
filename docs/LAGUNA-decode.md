# Laguna S 2.1 Decode Gap Analysis — W7900 / UD-Q2_K_XL

Status: expanded diagnostic plan complete; P0 exact-IQ3 ownership and P2.1
exact split attention are implemented and retained as gfx1100 defaults. Both
P1 IQ3, raw-Q5, and raw-IQ2 lanes plus P2.2 online FP32 partials are rejected.
The exact SWA tile16 score producer is retained as the gfx1100 default from live
count 257; P1 and P2 are closed. P3's bit-lossless Q5 T16 replacement screen is
rejected. P4.1's exact P2-derived split-reducer+softplus-gate body is retained
as the gfx1100 default at split thresholds; the matched target still requires a
genuinely new submission owner.

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
exact split attention, and P4.1 split-reducer+gate fusion move the current
counterbalanced h32 row to **51.825 tok/s / 19.296 ms/token**; the non-equivalent
`llama-bench` row still represents an **82.38%** diagnostic gap. However, the
gap is not explained by prompt depth, sampling, Python, or one missing launch
flag:

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

The final completion audit now supplies the closest retained cross-engine
boundary. It runs all 18 category+heldout prompts at context 4096, natural
greedy h16/h32, and two repetitions, then counts the same post-TTFT transitions:
hipEngine `decode_forward_calls/decode_seconds` versus llama.cpp Vulkan
`sum(predicted_n - 1) / sum(predicted_ms)`. Retained hipEngine measures
**51.839/51.432 tok/s** and Vulkan **64.213/64.336 tok/s**, so hipEngine is
**19.27%/20.06% slower** and needs **23.87%/25.09%** more throughput. This is a
much smaller and more actionable gap than the non-equivalent 94.513-tok/s
`llama-bench` diagnostic, but it still fails the Vulkan-beating objective.

Prompt IDs, natural sampling, horizons, context length, and transition ownership
match. One unavoidable arithmetic difference remains: hipEngine uses BF16
`KVLiveSpans`, while Vulkan uses F16 KV because the reported device capability
has no BF16 support. All 72 server-native prompt/predicted timing rows are valid.
The SSE Content-only path omits one or more returned token-array entries for 18
rows even though `predicted_n` and timings are complete, so returned-array
length is reported but not used as the timing gate. Cross-engine IDs are also
reported, not substituted for hipEngine's exact within-engine correctness and
lifecycle gates. Evidence:
[`...vulkan-matched-completion-audit.json`](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-vulkan-matched-completion-audit.json).

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
is **15.544 ms/transition**, while retained hipEngine is **19.443 ms** and its
clean short GPU kernel sum is about **15.516 ms**. Device work alone nearly
fills the Vulkan wall, but roughly 3.9 ms of queue/host span must also disappear.
Beating Vulkan therefore requires both a new independently winning fused body
and substantially lower submission spacing; neither half is optional.

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

The native scheduler is therefore the active next lane. It needs a genuinely
new mechanism—reusable dynamic command buffers or an AQL/native packet owner—not
D8 capture or D16 function-pointer packets. Its acceptance metric is matched
transition wall plus queue-gap attribution with unchanged kernel results, not
launch count. P4.1 raises the formal h32 row to 51.825 tok/s, but the matched
64.336-tok/s Vulkan target still requires **24.14%** more throughput.

## 9. Do not chase without new evidence

- **Unchanged D8 graph replay:** measured regression and removed.
- **Unchanged D10/D13/D14/D15/D17 boundaries:** all have complete rejection
  artifacts; positive diagnostic rows do not waive category/TTFT failures.
- **More C-side packets:** D16 proved the visible gaps are queue spacing, not
  ctypes transition cost.
- **Q8_1/MMVQ/dp4a as a default premise:** Vulkan's whole-path control and
  hipEngine's IQ2 experiment reject it broadly. An IQ3-only inclusive screen is
  allowed only because the llama.cpp HIP IQ3 family supplies new direct evidence.
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
| What is the retained hipEngine row? | [`...p4-split-gate-retained.json`](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-p4-split-gate-retained.json) |
| What dominates D12? | D12 clean profile plus [`...d9-residual-profile.json`](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-d9-residual-profile.json) with D12 leaf replacements |
| What did D0–D17 retain/reject? | [`LAGUNA.md`](LAGUNA.md), “Laguna Q2 XL Decode Optimization Campaign” |
| Is IQ2 already tuned? | [`OPTIMIZE-KERNEL-IQ2_XS.md`](OPTIMIZE-KERNEL-IQ2_XS.md) |
| Why is Qwen competitive? | [`...gguf-final-optimization-sweep.json`](../benchmarks/results/2026-07-16-gfx1100-gguf-final-optimization-sweep.json) |
| Which Qwen tactics were considered? | [`OPTIMIZE.md`](OPTIMIZE.md), [`OPTIMIZE-DENSE.md`](OPTIMIZE-DENSE.md), Q3/Q4 artifacts, and `WORKLOG.md` |
| Compact conclusions and logger hashes | [`...decode-gap-analysis.json`](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-decode-gap-analysis.json) |
| What does the corrected HIP/Vulkan history transfer? | [`HIP-vs-VULKAN.md`](HIP-vs-VULKAN.md), [`HIP-vs-VULKAN-HISTORY.md`](HIP-vs-VULKAN-HISTORY.md) |
| What does same-source llama.cpp HIP isolate? | [`...hip-vulkan-isa-attention-review.json`](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-hip-vulkan-isa-attention-review.json), `/tmp/laguna-llamacpp-hip-depth-profile-summary.json` hash therein |
| What is the raw-IQ3 ownership/ISA result? | Same review artifact plus `hipengine/kernels/hip_gfx1100/quant/gguf_iq_gemv.hip` and llama.cpp HIP `mmvq.cu`/`vecdotq.cuh` plus Vulkan `mul_mat_vec.comp`/`dequant_funcs_cm2.glsl` at `c0bc8591e` |
| What is the next attention algorithm? | Same review artifact; llama.cpp `fattn-tile.cuh`/`fattn-common.cuh` at `c0bc8591e`; in-tree `attention/paged_attn_decode.hip` split producer/reducer |
| Did a bit-lossless Q5 repack help? | [`...p3-q5-t16-repack-rejected.json`](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-p3-q5-t16-repack-rejected.json): exact generic/wave32x2 T16 both regress actual global/SWA layers and are not retained. |
| Does retained hipEngine beat Vulkan under matched natural completion? | No. The pre-P4 [`...vulkan-matched-completion-audit.json`](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-vulkan-matched-completion-audit.json) measures Vulkan **64.213/64.336 tok/s**; P4.1's category h32 is **51.825 tok/s**, still **24.14%** short. |

## Bottom line

The approximately 2x headline gap is genuine enough to guide engineering even
though the two public timing protocols differ. We are **not** losing 2x to
Python, sampling, graph replay, a missing compiler flag, or one unfused router.
The clean GPU trace proves otherwise.

hipEngine has already transferred the broad Qwen playbook and improved Laguna
from **19.596 to 51.825 tok/s**. The expanded review removes two tempting but
wrong shortcuts: neither a generic ACO/Clang upgrade nor a broad Q8_1 switch is
supported by the evidence.

The implementation loop remains ordered and falsifiable: P0 exact IQ3
route/output ownership and P2 exact attention split topology are retained; both
narrow P1 IQ3 lanes, P1 raw-Q5/raw-IQ2 row4, P2.2 online partials, and P3's
bit-lossless Q5 T16 replacement are rejected. The online-attention and raw-Q5
bodies prove tile-level ownership is mechanically valuable but violate the
frozen KL gate; IQ2 row4 and Q5 T16 instead fail the actual-weight performance
precondition. All are removed. The exact SWA tile16 score producer is retained
at live `>=257`; P1-P3 are closed. P4.1's exact gated split reducers are now
retained after independent-body, full-state, trace, and complete-category gates.
The matched completion audit replaces the non-equivalent 94.513-tok/s headline
with a formal **64.336 tok/s** h32 target, but retained hipEngine still needs
**24.14%** more throughput. The active P4 lane is now a genuinely new submission
owner; unchanged graph capture, host packets, or launch cleanup alone remain
non-retainable.
