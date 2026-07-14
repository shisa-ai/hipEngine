# gfx1151 hipEngine versus llama.cpp HIP parity audit

Status: **gfx1151 audit complete; gfx1100 HIP memory parity closed, compute parity open**
Date: **2026-07-14**
Machine-readable evidence:
[`gfx1151 parity audit`](../benchmarks/results/2026-07-14-gfx1151-llamacpp-hip-parity-audit.json),
[`gfx1100 post-transfer profile`](../benchmarks/results/2026-07-14-gfx1100-gguf-prefill-post-transfer-profile.json),
[`gfx1100 LCP-D2 gate`](../benchmarks/results/2026-07-14-gfx1100-gguf-decode-lcp-d2-parallel-reduce.json),
[`gfx1100 final rollup`](../benchmarks/results/2026-07-14-gfx1100-gguf-optimization-right-sized-3run.json), and
[`gfx1100 LCP-M1 memory gate`](../benchmarks/results/2026-07-14-gfx1100-gguf-lcp-m1-prefill-scratch-liveness.json).

This document answers a narrow question: after the retained GPF-5A work, what
still makes llama.cpp HIP faster than hipEngine GGUF on Radeon 8060S/gfx1151,
and which implementation ideas are actually worth transferring?

## Decision

Do **not** treat llama.cpp as a uniformly faster implementation and do not port
its generic HIP backend wholesale.

The current comparison has three distinct regimes:

1. **512-64K prefill:** hipEngine still trails, most sharply at 4K. The current
   512/4K traces attribute the actionable excess to exact GDN recurrence,
   long-token linear-attention convolution, and then dense Q8.
2. **128K prefill:** hipEngine is already within **0.80%** of llama.cpp HIP.
   This is not the place for another broad prefill rewrite.
3. **Decode:** hipEngine is within **-5.54% to +4.44%** through 64K and trails
   clearly only at 128K (**-13.58%**). That row needs a bounded matched decode
   profile and KV-dtype control before source differences are called causal.

The first implementation candidate is **LCP-1: an exact, same-stream,
long-token SSM-convolution kernel using llama.cpp's 32-token shared-memory
schedule**. It targets a measured 4K family gap of **954.438 versus 32.980 ms**
without importing llama.cpp's non-identical GDN reduction tree or re-enabling
the unstable isolated-AOTriton-stream policy.

## gfx1100 post-transfer update

The W7900 schedule-transfer work changes the ranking and closes LCP-1 on
`hip_gfx1100`. A clean, prefill-only `rocprofv3` profile at `16395fe5` records
**358.274/2701.741 ms** over **2009/5495** dispatches at 512/4K. Exact GDN
recurrence now owns **211.487/1652.114 ms (59.0%/61.1%)**, while convolution
falls to only **3.101/29.552 ms (0.87%/1.09%)**. The old 4K 954 ms convolution
queue cliff disappeared when the architecture-local direct-LDS32 GDN route was
promoted; it is not an independent current hotspot on gfx1100.

The planned 32-token by 128-channel LDS candidate was nevertheless implemented
as a bounded diagnostic. It preserves raw output and final-state bytes at token
lengths `4,31,32,33,512,4096`, but the normal-stream full-model screen is
neutral at 512 (**+0.043%**) and regresses 4K **1468.728 -> 1465.910 tok/s
(-0.192%)**. The candidate kernel, selector, and duplicate route were removed.
Do not tune LCP-1 further on gfx1100 without a new profile that restores a
material convolution bucket. The next first-order gfx1100 prefill problem is
exact GDN recurrence.

`LCP-D1` is now complete. A phase-marked eager profile at 128K attributes
**44.92%** of GPU time to full-attention core: grouped-GQA context costs
**5.067 ms/token** and the serial gated split reduction costs
**1.621 ms/token**. `LCP-D2` replaces only that reduction with a parallel
max/normalization prepare plus coalesced 32-dimension output reduction. At 513
splits, the leaf moves **194.881 -> 25.000 us (7.80x)**. The clean 64K
teacher-forced gate preserves all 17 generated IDs with max KL
**1.904e-6** and 100% top-1. Clean graph decode improves
**84.525 -> 85.561 tok/s (+1.23%)** at 32K, **72.446 -> 75.307 (+3.95%)**
at 64K, and **56.927 -> 61.367 (+7.80%)** at 128K, with unchanged tracked
memory and stable IDs. The 128K candidate is **0.71% above** the retained
llama.cpp HIP reference. gfx1100 therefore selects LCP-D2 from 32K onward;
gfx1151 remains on the serial reducer pending an independent gate.

The final residual gfx1100 prefill screens do not open a lower-effort GDN
route. Exact LDS16 is mixed (**-0.155% at 512, +0.434% at 4K**); a two-lane
VGPR-resident ordered schedule fails BF16 byte equality in both plain and
segmented fixtures before timing. Extending GPF-5A two-wave dense Q8 beyond its
4K W7900 cap also regresses 32K/64K **1.62%/0.22%**. All candidate code was
removed. See the
[`residual-screen artifact`](../benchmarks/results/2026-07-14-gfx1100-gguf-residual-prefill-screens.json).

The clean final gfx1100 defaults-only publication then records
**1290.246/1395.244/1401.632/1221.716/1021.693/766.892 tok/s** prefill and
**89.727/95.117/97.292/85.898/75.012/61.264 tok/s** graph decode from
512 through 128K. The prefill gap versus llama.cpp HIP narrows from **46.5% at
512 to 14.0% at 128K**; hipEngine decode is now ahead at every listed shape,
including **+0.54% at 128K**. All 18 measured final IDs are `9707`. Evidence:
[`gfx1100 final rollup`](../benchmarks/results/2026-07-14-gfx1100-gguf-optimization-right-sized-3run.json).

`LCP-M1` closes the retained HIP memory target without changing model math.
The gfx1100 production Qwen3.6 MoE/direct-LDS32 route now owns one aligned
phase-liveness arena instead of every attention/GDN/MoE intermediate at once.
A clean six-shape right-sized allocation census moves tracked memory from
**21.478/21.710/22.995/23.559/24.203/25.493** to
**21.204/21.256/21.544/22.108/22.752/24.041 GiB**. Every row is now
**0.048-0.402 GiB below** the retained llama.cpp HIP whole-device reading.
Residual memory versus Vulkan is only **0.036-0.266 GiB** from 1K through
128K, while 512 is 0.056 GiB lower. These remain different allocator scopes,
but the prior stronger condition—a narrower hipEngine count exceeding the
broader HIP count—is gone.

Correctness and wall gates are exact/non-regressive. A same-weight,
same-process 4K dedicated-versus-aliased A/B preserves all **248,320 FP32
logits byte-for-byte**. Clean 4K 1+3 prefill is
**1401.632 -> 1403.619 tok/s (+0.14%)**, and graph decode is
**97.292 -> 97.669 tok/s (+0.39%)**. Diagnostics, non-direct GDN modes,
non-MoE configs, and unvalidated backends retain dedicated buffers.

## gfx1100 parity continuation

Parity remains the target; the final optimization rollup is a new baseline, not
closure. Re-expressing the gaps as the uplift hipEngine still needs makes the
remaining work explicit:

- Prefill needs **+86.97%/+71.27%/+60.89%/+36.50%/+26.44%/+16.31%** to
  match llama.cpp HIP from 512 through 128K.
- Decode already exceeds llama.cpp HIP at every shape, but needs
  **+20.13%/+13.08%/+5.93%/+6.91%/+11.64%/+15.62%** to match llama.cpp
  Vulkan.
- `LCP-M1` has closed the retained llama.cpp HIP memory target at all six
  shapes. hipEngine now sits **0.402/0.362/0.130/0.108/0.143/0.048 GiB below**
  the HIP whole-device rows. Small cross-scope efficiency claims remain invalid;
  this result establishes capacity parity, not allocator equivalence.

The closed memory lane confirms the original attribution. At 4K+, the prior
**1.751-1.759 GiB** bulk-prefill bucket falls to **0.300-0.308 GiB**, below the
predeclared **0.35-0.45 GiB** ceiling. The reduction is approximately 1.45 GiB
at every 4K+ shape because context-dependent KV/state and metadata are unchanged.
No KV-format change is involved.

A cached-only current-tree 4K decode trace adds the missing middle-shape
attribution. Its final eight state-update/embedding-delimited exact decode
cycles contain about **708 dispatches/token** and **8.914 ms GPU time/token**. Dense Q8 T16 GEMV is
**39.28%**, selected-MoE T16 GEMV **20.30%**, full-attention core **9.04%**,
lm-head **7.19%**, router **6.65%**, and GDN decode **6.11%**. The 4K Vulkan
wall target is only a **5.93%** throughput uplift away; dense Q8 and selected
MoE are therefore the first decode families. Generic launch-count work is not
the explanation: hipEngine is already far below the roughly 1,600
dispatches/token in the retained llama.cpp HIP source/profile analysis.

The source-grounded Vulkan advantage remains c=1-specific: smaller
single-subgroup workgroups, no cross-wave LDS reduction, RADV/ACO scheduling,
q8_1 activation-load coalescing, and graph-level MoE/post-op fusion. It is not
Vulkan WMMA decode, a wider-than-dp4a instruction, or a generic attention
advantage. Existing hipEngine dp4a diagnostics also show that changing the dot
instruction alone is insufficient; any new decode candidate must improve the
actual dominant family and full-model wall.

Continuation order is now:

1. `LCP-2`: pursue exact chunked/prefix GDN or a separately predeclared
   quality-safe register-resident design; do not rerun rejected LDS16,
   two-lane, or tree defaults unchanged.
2. Profile-directed dense-Q8/selected-MoE decode work for Vulkan parity,
   retaining 4K first and escalating to the 512 and 128K endpoints.

`LCP-M1` is complete and promoted.

Machine-readable ratios, allocation buckets, the diagnostic 4K family trace,
and source boundary:
[`parity rebaseline`](../benchmarks/results/2026-07-14-gfx1100-gguf-parity-rebaseline.json).

## Evidence boundary

### What is matched

- AMD Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151.
- The same 22,663,387,424-byte Qwen3.6-35B-A3B `UD-Q4_K_M` file, sampled
  fingerprint
  `936659d614707776d8e6ca1fb8595991159e78361bff2e3a3616aa91564c89fb`.
- The same prompt/decode row counts: 512, 1K, 4K, 32K, 64K, and 128K with 128
  decode tokens for the public table.
- Clean retained hipEngine measurements and identified llama.cpp build/source
  lineage.

### What is not identical

- hipEngine stores BF16 KV; llama.cpp was run with F16 K/V.
- hipEngine's published GGUF protocol is one discarded warmup plus three
  measurements in an independent right-sized process. llama.cpp uses one
  internal warmup plus five `llama-bench` samples per split phase.
- hipEngine uses repeated token ID `9707`; `llama-bench` owns its synthetic
  workload generation. The row counts match, but MoE routing is not a
  token-for-token A/B.
- The retained llama.cpp topline predates the current hipEngine run and uses a
  different software revision.
- hipEngine peak memory is its tracked allocator high-water. llama.cpp peak is
  whole-device amdgpu GTT used. Cross-column memory differences are not direct
  allocator-efficiency measurements.
- The family traces are one-pass, no-warmup profiler diagnostics. Their
  throughput does not replace the public medians.

These limitations allow family selection and source-backed hypotheses. They do
not allow claims that two similarly named kernels implement identical math or
that KV dtype is irrelevant.

## Current topline gap

The public values come from [`benchmarks/README.md`](../benchmarks/README.md).
The wall gap is `N / hipEngine_tok_s - N / llama_tok_s`.

| Context | hipEngine prefill | llama.cpp HIP | hipEngine / llama | Prefill wall gap | Decode delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 889.904 | 1061.260 | 83.85% | 0.093 s | -3.87% |
| 1K | 919.598 | 1043.230 | 88.15% | 0.132 s | +1.33% |
| 4K | 762.940 | 1009.240 | 75.60% | **1.310 s** | +4.44% |
| 32K | 648.948 | 743.547 | 87.28% | 6.424 s | -1.69% |
| 64K | 546.296 | 573.611 | 95.24% | 5.713 s | -5.54% |
| 128K | 387.334 | 390.441 | **99.20%** | 2.693 s | **-13.58%** |

This is not a blanket GGUF deficit. The 4K prefill shape is the largest
relative miss; 128K prefill is effectively at parity under the current
cross-engine protocol.

## Post-GPF-5A family profiles

The clean detached hipEngine profile is commit
`2332756e32c04f61103be3aa5f0f72d00290ed3a`. It uses automatic gfx1151
GPF-2D/3A/2E/5A policy, cached builds, prefill only, and one no-warmup
`rocprofv3` pass. The matched llama.cpp family data are the complete 512/4K
traces already retained in
[`2026-07-14-gfx1151-gguf-prefill-next-family-profile.json`](../benchmarks/results/2026-07-14-gfx1151-gguf-prefill-next-family-profile.json).

### 512 rows

| Family | hipEngine | llama.cpp HIP | hip / llama | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Exact GDN recurrence | **205.570 ms / 36.13%** | 44.180 ms / 9.85% | **4.65x** | Largest hipEngine family and excess |
| Dense Q8 | 110.526 ms / 19.43% | 66.859 ms / 14.91% | 1.65x | Residual after GPF-5A |
| Selected Q4 | 95.946 ms / 16.86% | 141.704 ms / 31.60% | **0.68x** | hipEngine is faster; not a port target |
| Selected Q5 | 56.377 ms / 9.91% | 70.947 ms / 15.82% | **0.79x** | hipEngine is faster; not a port target |
| Linear-attention conv | 14.303 ms / 2.51% | 4.260 ms / 0.95% | 3.36x | Small absolute 512 opportunity |
| Full attention | 4.254 ms / 0.75% | 6.137 ms / 1.37% | **0.69x** | hipEngine is faster |

Total traced kernel time is **568.953 ms / 2,009 dispatches** for hipEngine
versus **448.373 ms / 2,490 dispatches** for llama.cpp.

### 4K rows

| Family | hipEngine | llama.cpp HIP | hip / llama | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Exact GDN recurrence | **1700.469 ms / 32.48%** | 330.488 ms / 8.85% | **5.15x** | Largest raw family, but exactness-constrained |
| Linear-attention conv | **954.438 ms / 18.23%** | 32.980 ms / 0.88% | **28.94x** | Best exact-plausible transfer |
| Dense Q8 | 749.444 ms / 14.32% | 531.345 ms / 14.24% | 1.41x | Third residual |
| Selected Q4 | 622.997 ms / 11.90% | 1173.527 ms / 31.44% | **0.53x** | hipEngine is faster |
| Selected Q5 | 391.637 ms / 7.48% | 585.011 ms / 15.67% | **0.67x** | hipEngine is faster |
| Full attention | 147.762 ms / 2.82% | 204.083 ms / 5.47% | **0.72x** | hipEngine is faster |

Total traced kernel time is **5235.029 ms / 5,495 dispatches** for hipEngine
versus **3732.516 ms / 17,789 dispatches** for llama.cpp.

Kernel sums can double-count queue overlap and are selection evidence, not a
host-wall equation. The ordering is nevertheless robust:

- GDN plus convolution contribute **2291.439 ms** more hipEngine kernel time at
  4K.
- hipEngine wins back **800.227 ms** in selected Q4/Q5 and full attention.
- llama.cpp launches 3.24x as many traced kernels at 4K, so fewer launches or a
  more generic graph runtime cannot explain its prefill lead.

### What GPF-5A changed

Against the prior default-route diagnostic profile, dense-Q8 family time moves:

| Context | Before GPF-5A | After GPF-5A | Diagnostic delta | Total-kernel delta |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 158.982 ms | 110.526 ms | -30.48% | 619.556 -> 568.953 ms (-8.17%) |
| 4K | 844.670 ms | 749.444 ms | -11.27% | 5396.575 -> 5235.029 ms (-2.99%) |

These are separate profiler runs. The clean focused A/Bs in
[`GGUF-PREFILL-OPTIMIZATION.md`](GGUF-PREFILL-OPTIMIZATION.md) remain the causal
GPF-5A evidence.

## Exact source lineage

The measured llama.cpp build reports local commit
`1ebf790cda38d827559548f67b0469189690cc8c`, build 9648. The committed
[instrumentation manifest](../benchmarks/llama.cpp/manifest.json) records
upstream base `6e9007ae61f4e994c27484759caac6ef2aa32b30`, seven local MTP
instrumentation commits, and the dirty-source patchset.

The performance-relevant backend files used below—GDN, SSM convolution, MMQ,
top-k MoE, CUDA/HIP dispatch/fusion, and FlashAttention—are byte-identical
between upstream base `6e9007ae6` and local measured head `1ebf790cd`. The
Qwen35MoE local committed change is inside the MTP block after the AR code cited
below. Upstream-base permalinks are therefore immutable references for the
measured AR implementation, while the manifest remains authoritative for the
local build lineage.

hipEngine links use retained commit
`2332756e32c04f61103be3aa5f0f72d00290ed3a`, the exact clean revision profiled
here.

## Source mapping

### 1. GDN: large advantage, not a direct port

llama.cpp's GDN maps a wave to one state/value column, loads each lane's state
rows into a register shard, keeps that shard live across the serial token loop,
and uses wave reductions for the two contractions:

- [register-resident shard and serial recurrence](https://github.com/ggml-org/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/ggml/src/ggml-cuda/gated_delta_net.cu#L4-L169)
- [four-wave launch geometry](https://github.com/ggml-org/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/ggml/src/ggml-cuda/gated_delta_net.cu#L181-L220)
- [Qwen35MoE linear-attention graph](https://github.com/ggml-org/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/src/models/qwen35moe.cpp#L360-L492)

hipEngine's current exact default instead keeps a `128 x 32` state tile in LDS;
one lane owns each value column and evaluates the canonical scalar contraction
order:

- [exact direct-conv LDS32 body and kernel](https://github.com/shisa-ai/hipEngine/blob/2332756e32c04f61103be3aa5f0f72d00290ed3a/hipengine/kernels/hip_gfx1100/linear_attn/gdn.hip#L1374-L1541)
- [32-thread launch](https://github.com/shisa-ai/hipEngine/blob/2332756e32c04f61103be3aa5f0f72d00290ed3a/hipengine/kernels/hip_gfx1100/linear_attn/gdn.hip#L4360-L4384)

The llama.cpp schedule explains the measured **4.65x/5.15x** family advantage,
but reduction order is part of hipEngine's current semantic contract. The
existing register/tree diagnostic already demonstrated the trade-off: it was
much faster but completed only 3/10 exact natural 128-step trajectories. The
ordered-wave exact attempt was slower.

**Decision:** do not copy the tree reduction under the default route. Keep
chunked/token-prefix GDN as high-effort research requiring the existing
six-case state and 250/250 natural-trajectory gates.

### 2. Convolution: strongest transferable schedule

llama.cpp uses a separate long-token kernel once `n_t > 32`. A 128-thread block
loads a **32-token by 128-channel** window into shared memory, keeps the four
convolution weights in registers, and computes each output from the shared
tile. Its wrapper fuses SiLU into the same launch:

- [long-token shared-memory kernel](https://github.com/ggml-org/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/ggml/src/ggml-cuda/ssm-conv.cu#L58-L125)
- [short/long selection and fused-SiLU launch](https://github.com/ggml-org/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/ggml/src/ggml-cuda/ssm-conv.cu#L127-L206)
- [graph fusion recognizes SSM-conv plus SiLU](https://github.com/ggml-org/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/ggml/src/ggml-cuda/ggml-cuda.cu#L3722-L3774)

hipEngine's prefill kernel maps a thread to one `(token, channel)` output and
reads the four-value window from global memory. A second launch updates the
persistent convolution state:

- [current output/SILU kernel](https://github.com/shisa-ai/hipEngine/blob/2332756e32c04f61103be3aa5f0f72d00290ed3a/hipengine/kernels/hip_gfx1100/linear_attn/conv.hip#L87-L147)
- [separate state-update kernel](https://github.com/shisa-ai/hipEngine/blob/2332756e32c04f61103be3aa5f0f72d00290ed3a/hipengine/kernels/hip_gfx1100/linear_attn/conv.hip#L227-L250)
- [two-launch wrapper](https://github.com/shisa-ai/hipEngine/blob/2332756e32c04f61103be3aa5f0f72d00290ed3a/hipengine/kernels/hip_gfx1100/linear_attn/conv.hip#L771-L819)

The 4K **28.94x** ratio includes the known same-stream queue cliff after
high-scratch AOTriton. However, queue isolation reduced hipEngine convolution
to **88.839 ms**, still **2.69x** llama.cpp's 32.980 ms. That confirms both an
external queue effect and an independent kernel-schedule opportunity.

**Decision:** implement the shared-token tile as a separately registered,
same-stream, exact candidate. Preserve product/add order per output, keep the
current kernel as fallback, and do not couple the experiment to GPF-4 stream
isolation.

### 3. Quantized matmul: copy ideas only for dense Q8

llama.cpp quantizes F32 activations to a Q8_1 MMQ layout designed for contiguous
shared-memory copies and bank-conflict padding. The format stores scales and
partial sums according to the weight quant:

- [Q8_1 MMQ layout](https://github.com/ggml-org/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/ggml/src/ggml-cuda/mmq.cuh#L20-L102)
- [activation quantization plus expert sorting](https://github.com/ggml-org/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/ggml/src/ggml-cuda/mmq.cu#L77-L224)
- [shared weight/activation tile processing](https://github.com/ggml-org/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/ggml/src/ggml-cuda/mmq.cuh#L3447-L3527)
- [RDNA conventional tiling rather than Stream-K](https://github.com/ggml-org/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/ggml/src/ggml-cuda/mmq.cuh#L3528-L3614)
- [shape/LDS tile-width selection](https://github.com/ggml-org/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/ggml/src/ggml-cuda/mmq.cuh#L3943-L4132)
- [RDNA3 high-expert MMQ policy](https://github.com/ggml-org/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/ggml/src/ggml-cuda/mmq.cu#L267-L371)

hipEngine materializes byte-lossless T16 replacement weights and consumes BF16
activations directly with WMMA. GPF-5A now pairs two 32-column waves and shares
one activation tile:

- [production and two-wave Q8T16 kernels](https://github.com/shisa-ai/hipEngine/blob/2332756e32c04f61103be3aa5f0f72d00290ed3a/hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_t16_prefill.hip#L113-L359)
- [gfx1151 scoped policy](https://github.com/shisa-ai/hipEngine/blob/2332756e32c04f61103be3aa5f0f72d00290ed3a/hipengine/kernels/hip_gfx1151/__init__.py#L29-L47)

The result is mixed, not a generic MMQ win:

- llama.cpp dense Q8 is still 1.65x/1.41x faster at 512/4K;
- hipEngine selected Q4 is 1.48x/1.88x faster; and
- hipEngine selected Q5 is 1.26x/1.49x faster.

**Decision:** retain T16 selected Q4/Q5. For Q8, borrow shared-layout and tile
selection ideas only if BF16 activation semantics and byte-exact outputs remain
intact. Q8_1 activation quantization is not presumed exact-equivalent.

### 4. MoE routing: optimize logits before top-k fusion

llama.cpp recognizes and fuses softmax/top-k/get-rows into one cooperative
kernel, then routes quantized `MUL_MAT_ID` through MMVQ or MMQ with expert
bounds:

- [fused top-k MoE kernel](https://github.com/ggml-org/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/ggml/src/ggml-cuda/topk-moe.cu#L72-L257)
- [launch and applicability rules](https://github.com/ggml-org/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/ggml/src/ggml-cuda/topk-moe.cu#L259-L402)
- [graph-pattern recognition and replacement](https://github.com/ggml-org/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/ggml/src/ggml-cuda/ggml-cuda.cu#L3447-L3540)
- [`MUL_MAT_ID` MMVQ/MMQ dispatch](https://github.com/ggml-org/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/ggml/src/ggml-cuda/ggml-cuda.cu#L2674-L2730)

hipEngine currently computes four tokens per block in a dedicated F32 router
matrix kernel and then launches cooperative select:

- [four-token router logits](https://github.com/shisa-ai/hipEngine/blob/2332756e32c04f61103be3aa5f0f72d00290ed3a/hipengine/kernels/hip_gfx1100/moe/router.hip#L81-L215)
- [cooperative select](https://github.com/shisa-ai/hipEngine/blob/2332756e32c04f61103be3aa5f0f72d00290ed3a/hipengine/kernels/hip_gfx1100/moe/router.hip#L234-L328)

At 4K, router logits cost **229.820 ms** while select costs **12.536 ms**.
Fusing select first can recover at most the small subwindow.

**Decision:** screen a matrix-oriented F32-router-logits kernel first. Fuse
selection only if a fresh profile still finds it material.

### 5. Full attention: llama.cpp is not the short/mid target

llama.cpp selects vector, tile, WMMA, or MMA FlashAttention according to head
size, GQA ratio, query rows, KV type, mask, and alignment:

- [kernel policy](https://github.com/ggml-org/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/ggml/src/ggml-cuda/fattn.cu#L340-L572)
- [dispatch](https://github.com/ggml-org/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/ggml/src/ggml-cuda/fattn.cu#L574-L597)

hipEngine's measured AOTriton family is already faster at 512 and 4K. The
prior 4K trace measured **147.762 versus 204.083 ms** after GPF-5A. At 128K,
llama.cpp FlashAttention dominates its own trace, while overall prefill is only
0.80% ahead and hipEngine's full 128K profile is unavailable due to the
profiler lifecycle issue.

**Decision:** no short/mid attention port. For the 128K decode gap, profile the
actual decode family and control KV dtype before choosing a kernel.

### 6. Graph, scheduling, and memory

llama.cpp's backend can fuse graph patterns, assign concurrent streams, and
capture/replay HIP graphs:

- [fusion and concurrent-event execution](https://github.com/ggml-org/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/ggml/src/ggml-cuda/ggml-cuda.cu#L4330-L4448)
- [graph instantiate/update/replay](https://github.com/ggml-org/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/ggml/src/ggml-cuda/ggml-cuda.cu#L4449-L4510)

That machinery is not the measured prefill advantage: llama.cpp launches more
kernels at both profile shapes, and the excess is inside named kernel families.
hipEngine's state-bound graph decode is already competitive through 64K.

llama.cpp does have a clear memory-planning design advantage. Its graph
allocator reuses eligible parent storage in place and frees intermediates when
the last child/view is consumed:

- [in-place parent reuse](https://github.com/ggml-org/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/ggml/src/ggml-alloc.c#L622-L689)
- [last-consumer allocation/free walk](https://github.com/ggml-org/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/ggml/src/ggml-alloc.c#L717-L804)

hipEngine now applies the same principle to the admitted gfx1100 production
route. `04b48b67` gives each scratch field an explicit route/phase lifetime,
places non-overlapping live ranges in one aligned arena, and retains dedicated
ownership for diagnostics and unvalidated routes. The six-shape census and
byte-exact 4K A/B above close the capacity gate without a speed regression.

**Decision:** `LCP-M1` is complete and promoted. Keep the explicit liveness
contract and fallback; reopen memory work only if a resident-allocation/KV/model
change makes hipEngine exceed the retained HIP capacity row again.

## Ranked parity work

| Rank | ID | Work | Why now | Exit gate |
| ---: | --- | --- | --- | --- |
| 1 | `LCP-1` | Exact 32-token shared-memory long-token convolution | **Closed on gfx1100 after post-transfer profile:** conv is 1.09% and exact candidate regresses 4K 0.192%; still untested as a gfx1151-local implementation | Do not revisit on gfx1100 without a new profile; gfx1151 retains the original gate if pursued independently |
| 2 | `LCP-D1/D2` | **Closed/promoted on gfx1100:** bounded 128K profile identified serial split reduction; parallel prepare/output reduction is the scoped default from 32K | 32K clean 1+3 +1.23%; 64K/128K clean confirmations +3.95%/+7.80%; long-context KL/top-1 gate passes | Keep serial rollback; gfx1151 requires independent transfer evidence |
| 3 | `LCP-2` | Exact chunked/prefix GDN research | Largest family and >4.6x gap; exact LDS16 is mixed and two-lane VGPR residency fails byte equality, while the direct tree port violates trajectory contract | High-effort only: six-case state matrix and 250/250 natural transitions before timing |
| 4 | `LCP-3` | Further dense-Q8 shared-layout/tile screen | Still 19.43%/14.32% and 1.65x/1.41x slower after GPF-5A | Byte-exact primitive, dominant-shape trace, 512/4K state/wall |
| 5 | `LCP-4` | Matrix-oriented F32 router logits; top-k fusion second | Logits are 94.8% of the measured 4K router bucket | Exact experts/weights and full state, then wall |
| 6 | `LCP-M1` | **Closed/promoted on gfx1100:** phase-liveness bulk-scratch arena | Tracked memory falls 0.274-1.452 GiB and clears the HIP capacity row at all shapes | 248,320 logits byte-exact; clean 4K prefill/decode non-regressive |

There is no invented minimum win. Exact, same-suite non-regressive
improvements remain retainable under the project evidence policy.

## Explicit non-targets

- Do not replace hipEngine selected Q4/Q5 with generic llama.cpp MMQ; the
  measured hipEngine families are already faster.
- Do not replace short/mid AOTriton with llama.cpp FlashAttention; the measured
  hipEngine family is already faster.
- Do not re-enable GPF-4 isolated AOTriton queues by default; its final 32K/128K
  stability gate remains rejected.
- Do not label llama.cpp's GDN tree reduction exact/default; it changes
  contraction order and the analogous hipEngine path failed trajectory parity.
- Do not prioritize generic graph or launch-count work for prefill; llama.cpp
  launches more kernels and wins inside specific families.
- Do not infer memory efficiency from the public cross-column peak values; the
  scopes differ.

## LCP-1 implementation gate

**gfx1100 result (2026-07-14): rejected and removed.** The candidate passed the
primitive byte gate but failed the normal-stream full-model wall gate after the
promoted schedule profile invalidated the old hotspot premise. The checklist
below remains the original gfx1151-local gate, not open gfx1100 work.

The first coding tranche should remain narrow:

1. Add a registered convolution variant; do not branch on backend or quant in
   engine/dispatch code.
2. Keep the current convolution output and state-update chain as the unfused
   fallback.
3. Tile 32 tokens by 128 channels in shared memory. Preserve each output's
   product/add order and final state bytes.
4. RED/GREEN around token lengths `4, 31, 32, 33, 512, 4096`, including initial
   and final convolution state.
5. Run the all-layer/full-state 512 and 4K differential gate.
6. Profile on the normal caller stream after AOTriton. The candidate must win
   without `HIPENGINE_QWEN35_AOTRITON_ISOLATED_PREFILL_STREAM=1`.
7. Run fresh-process one-warmup/three-measurement 512/4K A/B. Escalate only on
   the documented variance trigger.
8. If exact and non-regressive, promote through gfx1151 registry metadata and
   retain gfx1100 baseline until its own hardware gate.

This separates a source-backed kernel schedule from the rejected queue-policy
experiment and preserves every current correctness invariant.
