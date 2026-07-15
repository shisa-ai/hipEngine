# gfx1151 hipEngine versus llama.cpp HIP parity audit

Status: **audit complete; exact prefill tranche retained through scoped LCP-M2 on gfx1151**
Date: **2026-07-15**
Machine-readable evidence:

- [`2026-07-14-gfx1151-llamacpp-hip-parity-audit.json`](../benchmarks/results/2026-07-14-gfx1151-llamacpp-hip-parity-audit.json)
- [`2026-07-14-gfx1151-gguf-prefill-lcp1-clean-promotion.json`](../benchmarks/results/2026-07-14-gfx1151-gguf-prefill-lcp1-clean-promotion.json)
- [`2026-07-14-gfx1151-gguf-decode-lcpd1-clean-profile.json`](../benchmarks/results/2026-07-14-gfx1151-gguf-decode-lcpd1-clean-profile.json)
- [`2026-07-14-gfx1151-gguf-lcp1-lcpd1-right-sized-3run.json`](../benchmarks/results/2026-07-14-gfx1151-gguf-lcp1-lcpd1-right-sized-3run.json)
- [`2026-07-15-gfx1151-gguf-prefill-device-metadata-scoped-promotion.json`](../benchmarks/results/2026-07-15-gfx1151-gguf-prefill-device-metadata-scoped-promotion.json)
- [`2026-07-15-gfx1151-gguf-prefill-router-select-threads128-promotion.json`](../benchmarks/results/2026-07-15-gfx1151-gguf-prefill-router-select-threads128-promotion.json)

This document answers a narrow question: after the retained GPF-5A work, what
still makes llama.cpp HIP faster than hipEngine GGUF on Radeon 8060S/gfx1151,
and which implementation ideas are actually worth transferring?

## Decision

Do **not** treat llama.cpp as a uniformly faster implementation and do not port
its generic HIP backend wholesale.

The audit baseline had three distinct regimes:

1. **512-64K prefill:** hipEngine still trails, most sharply at 4K. The current
   512/4K traces attribute the actionable excess to exact GDN recurrence,
   long-token linear-attention convolution, and then dense Q8.
2. **128K prefill:** hipEngine is already within **0.80%** of llama.cpp HIP.
   This is not the place for another broad prefill rewrite.
3. **Decode:** hipEngine is within **-5.54% to +4.44%** through 64K and trails
   clearly only at 128K (**-13.58%**). At the audit point, that row needed a
   bounded matched profile before source differences could be called causal;
   LCP-D1 closes that attribution below for hipEngine BF16 KV.

That first candidate, **LCP-1**, is now retained: the exact same-stream
32-token shared-memory convolution reduces its 4K body from **954.134 to
49.790 ms** and improves the clean 4K focus by **22.91%** without importing
llama.cpp's non-identical GDN tree or re-enabling the unstable isolated-stream
policy. The bounded decode follow-up, **LCP-D1**, also retains an exact long-
split reducer improvement; its clean 128K eager profile moves
**27.130 -> 27.488 tok/s (+1.32%)**. Publication medians remain separate from
these marker-profile rates.

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

## Audit topline gap before LCP-1/LCP-D1

These values were the public GPF-5A baseline when the audit ran; the retained
LCP rollup below now supersedes the hipEngine column in
[`benchmarks/README.md`](../benchmarks/README.md). The wall gap is
`N / hipEngine_tok_s - N / llama_tok_s`.

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

hipEngine's bulk-prefill session allocates every named intermediate together:

- [scratch inventory and allocation](https://github.com/shisa-ai/hipEngine/blob/2332756e32c04f61103be3aa5f0f72d00290ed3a/hipengine/runtime/qwen35_gguf_runner.py#L12767-L13070)
- [session ownership](https://github.com/shisa-ai/hipEngine/blob/2332756e32c04f61103be3aa5f0f72d00290ed3a/hipengine/runtime/qwen35_gguf_runner.py#L7569-L7605)

**Decision:** liveness-based scratch aliasing is a capacity project, not a
current speed claim. Pursue it after the named compute residuals, with allocator
accounting and no-regression gates.

## Ranked parity work

| Rank | ID | Work | Why now | Exit gate |
| ---: | --- | --- | --- | --- |
| 1 | `LCP-1` | **Retained:** exact 32-token shared-memory long-token convolution | Clean 512/4K focus is +1.73%/+22.91%; 82/82 state parts are byte-exact and the 4K body falls 954.134 -> 49.790 ms | Complete on gfx1151; gfx1100 remains baseline pending W7900 evidence |
| 2 | `LCP-D1` | **Retained:** bounded 128K attribution plus exact long-split gated reduction | Attention is 50.95% at 128K; parallelizing only independent work above 256 splits cuts the reducer 234.714 -> 196.466 us/call | Complete for GGUF BF16 KV; PARO/KV-dtype work remains separate |
| 3 | `LCP-2A` | **Retained:** compiler-cacheable exact direct LDS32 GDN state | Clean balanced 512/1K/4K prefill is +34.76%/+36.63%/+36.58%; direct tree port remains invalid | Six-case state and 250/250 natural transitions pass byte-exactly; gfx1151 promoted |
| 4 | `LCP-3` | **Retained:** four exact Q8T16 waves share one activation tile | Clean 512/4K full-model prefill is +0.53%/+1.57%; dominant 4K shapes are 7.50%-14.08% faster than GPF-5A | Complete on gfx1151 through 64K; two-wave and production remain rollback paths |
| 5 | `LCP-4A` | **Retained:** remove idle half of exact F32 router reduction on gfx1151 | Clean 512/4K full-model prefill is +2.76%/+3.28%; graph decode is exact/+0.071% | 256-thread trace passes; refresh profile before deciding whether select fusion remains material |
| 6 | `LCP-M1` | Bulk-scratch liveness/alias plan | Capacity opportunity; not a current speed claim | Tracked allocation reduction, exact state, no perf regression |

There is no invented minimum win. Exact, same-suite non-regressive
improvements remain retainable under the project evidence policy.

## Retained implementation outcomes

### LCP-1: exact shared-token convolution

Clean detached commit `3ff8e2d7` passes the six-length primitive gate and the
512/4K 82-part full-state differential byte-for-byte. The spill-free body uses
128 threads, a 32-token by 128-channel tile, 17.5 KiB LDS, and zero scratch.
On the normal caller stream its 4K output family is **49.790 ms / 120 launches**
versus **954.134 ms** for the prior body. Fresh one-warmup/three-measurement
focus improves prefill **890.727 -> 906.118 tok/s (+1.73%)** at 512 and
**762.273 -> 936.910 tok/s (+22.91%)** at 4K, with unchanged memory. gfx1151
selects the tiled schedule; gfx1100 retains the baseline pending hardware-local
evidence.

### LCP-D1: exact long-split gated reduction

The clean attribution uses baseline `631498dd`; the retained candidate is
`71e61524`. Both use the same Q4_K_M model, BF16 KV, repeated token `9707`, four
eager-decode warmups, and 24 exact ROCTX-marked steps. At 512, dense Q8 remains
first at **8.520 ms/token / 44.25%**, while attention is only
**2.160 ms/token / 11.22%**. At 128K, attention becomes the context-local
majority at **17.882 ms/token / 50.95%**. The grouped-GQA context body is
**15.502 ms/token** and its gated split reducer is **2.347 ms/token**.

An all-eight-query-head register tile was byte-exact but decisively slower at
128K (**1.748 -> 2.878 ms/call, 0.607x**) and was removed. LCP-D1 instead keeps
max selection, denominator summation, and final output accumulation serial in
the original split order. Only independent exponentials and normalization
multiplies run cooperatively, and only when `num_splits > 256`; shorter contexts
execute the original serial body.

The 512-split reducer microbench is byte-exact for all 4,096 BF16 outputs and
moves **138.139 -> 101.350 us (1.363x)**. The 256-split control is neutral at
**76.184 -> 76.103 us**. In the clean 128K model trace, the reducer moves
**234.714 -> 196.466 us/call (-16.30%)**, attention moves
**17.882 -> 17.498 ms/token (-2.15%)**, total traced GPU time moves
**35.094 -> 34.668 ms/token (-1.22%)**, and profiled host wall moves
**36.860 -> 36.380 ms/token**, or **27.130 -> 27.488 tok/s (+1.32%)**.
All 24 candidate tokens are exact; the kernel uses 16 VGPR and zero scratch.
Evidence:
[`2026-07-14-gfx1151-gguf-decode-lcpd1-clean-profile.json`](../benchmarks/results/2026-07-14-gfx1151-gguf-decode-lcpd1-clean-profile.json).

### LCP-2A: compiler-cacheable exact GDN state

Clean detached candidate `53928aaf` preserves the six full-state cases and all
**250/250** natural transitions byte-for-byte. The compiler-cacheable direct
LDS32 body uses 32 VGPR, 16 KiB LDS, and zero scratch. Balanced 512/1K/4K
prefill improves **+34.76%/+36.63%/+36.58%**, while weighted decode is
**+0.021%**. gfx1151 selects it automatically; the volatile direct body remains
rollback and gfx1100 stays fused.

### LCP-3: exact four-wave dense Q8 prefill

Four independent production-order 32-column waves now share one 1 KiB BF16
activation tile. The named 128-thread kernel uses 80 VGPR and zero scratch.
Tail fixtures and clean detached 512/4K full-model captures are **83/83** exact.
Against automatic GPF-5A, five balanced pairs improve
**1214.510 -> 1220.993 tok/s (+0.53%)** at 512 and
**1269.030 -> 1288.986 tok/s (+1.57%)** at 4K; every timed ID is `9707`.
gfx1151 selects four-wave under the inherited 65,536-token ceiling, then
restores production. `HIPENGINE_GGUF_Q8_T16_PREFILL_4WAVE=0` is the two-wave
rollback; gfx1100 remains production. Evidence:
[`2026-07-15-gfx1151-gguf-q8-t16-four-wave-clean-promotion.json`](../benchmarks/results/2026-07-15-gfx1151-gguf-q8-t16-four-wave-clean-promotion.json).

### LCP-4A: exact 256-thread F32 router logits

The existing token-tiled router used 512 threads for `hidden_size=2048`, while
only the first 256 lanes received one eight-element dot fragment. Its first
reduction step therefore added only zeros. Reusing the unchanged HIP body with
256 threads preserves the meaningful reduction tree byte-for-byte and cuts the
isolated 512/1024-token `2048x256` router by **44.32%/44.17%**.

Clean detached candidate `3ef55ad4` is **83/83** exact at 512 and 4K. Five
balanced pairs improve prefill **1218.536 -> 1252.147 tok/s (+2.76%)** and
**1290.923 -> 1333.229 tok/s (+3.28%)**. A separate clean 512/128 graph gate is
exact and moves **48.987 -> 49.021 tok/s (+0.071%)**. `rocprofv3` confirms the
named token-tile body at 256 threads, 32 VGPR, and zero scratch. gfx1151 now
selects the 256-thread wrapper through the registry; gfx1100 remains at 512
pending independent evidence. Evidence:
[`2026-07-15-gfx1151-gguf-router-threads256-clean-promotion.json`](../benchmarks/results/2026-07-15-gfx1151-gguf-router-threads256-clean-promotion.json).

### Right-sized publication gate

The clean automatic one-warmup/three-measurement sweep is complete at
`71e61524`. Prefill is
**906.979/929.724/946.366/778.371/636.330/433.811 tok/s** at
512/1K/4K/32K/64K/128K, improving the prior public rows by
**1.92%/1.10%/24.04%/19.94%/16.48%/12.00%**. Graph decode is
**49.061/51.569/52.432/43.543/37.562/28.047 tok/s**; the 128K row improves
**1.06%**. All 18 measured IDs are `9707`, tracked memory is unchanged, and
maximum prefill/decode sample stdev over median is only **0.140%/0.113%**, so no
five-sample escalation is justified. At 32K, 64K, and 128K, hipEngine prefill
now exceeds the retained llama.cpp HIP rows by **4.68%/10.93%/11.11%**. The
single eager marker-profile rate above remains kernel evidence; this 1+3 graph-
decode sweep supplies the publication medians.

### gfx1151 hardware-queue stability gate

A clean current-production 128K rerun with ROCm's documented default four
hardware queues entered the no-progress state during its first warmup: active
power fell from roughly 122-127 W to 41-43 W while utilization/SCLK remained
100%/2.9 GHz. Four seven-minute host dumps stayed in synchronous metadata
`hipMemcpy`, the kernel journal recorded no amdgpu/KFD fault, and terminating
the process restored idle without reset. Changing only
`GPU_MAX_HW_QUEUES=1` completed warmup+3 at **499.755 warmup** and
**500.210/500.873/500.687 measured prefill tok/s**, with exact token `9707`,
unchanged memory, and normal active power. Clean 512/4K checks are also
non-regressive at **+0.35%/+0.46% prefill** and **+0.066%/+0.072% decode**.
hipEngine therefore applies one queue before `libamdhip64` loads when gfx1151 is
the only recognized visible HIP backend; explicit values are preserved and
gfx1100 is unchanged. Evidence:
[`2026-07-15-gfx1151-hip-one-queue-stability-promotion.json`](../benchmarks/results/2026-07-15-gfx1151-hip-one-queue-stability-promotion.json)
and the [ROCm#5107 comment](https://github.com/ROCm/ROCm/issues/5107#issuecomment-4976739824).

### Scoped LCP-M2 metadata closure

Under the promoted one-queue policy, the stream-ordered contiguous metadata
kernel remains full-model exact and wins at the short/mid shapes: clean five-
pair 512/1K/4K prefill improves **1261.643/1333.877/1356.934 ->
1281.323/1345.928/1364.103 tok/s (+1.56%/+0.90%/+0.53%)**. Clean automatic-
vs-explicit state is **83/83 exact** at all three shapes. The explicit 128K
route still enters the low-power no-progress state on its first measured pass
under one queue, so gfx1151 selects device metadata only through 4K and retains
the synchronous path above it. This is a scoped exact win, not evidence that
the queue workaround fixed every scheduling trigger.

### Post-profile LCP-4B router-select closure

The required final 4K profile measures router select at **12.539 ms / 130
launches (0.41%)**. That does not justify replacing LCP-4A's dominant exact
logits geometry with cross-block logits+top-k fusion. A launch-only screen is
safer and retained: 128 threads cuts the named family to **3.741 ms (-70.17%)**
and improves clean balanced 512/4K prefill **+0.34%/+0.36%**, with 83/83
full-model state parts exact. The faster 64-thread primitive is rejected because
4K full-model logits/hidden, Conv/GDN, and KV state differ.

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

## Next parity targets

LCP-1, LCP-D1, LCP-2A, scoped LCP-M2, LCP-3, LCP-4A, and LCP-4B close the
currently actionable exact convolution, GDN, metadata, dense-Q8, router-logit,
and router-select prefill bodies. Use the promoted gfx1151 one-queue process
default for subsequent production/profile runs. The required 4K refresh is
complete and does not justify router-select fusion; do not disturb the already-
faster selected Q4/Q5 families. For decode, the clean 128K trace still
leaves the grouped-GQA context body at **15.502 ms/token** and dense Q8 at
**8.546 ms/token**. Any follow-up must target one of those measured families and
preserve the current BF16-KV, `KVLiveSpans`, and exact state/token contracts.
