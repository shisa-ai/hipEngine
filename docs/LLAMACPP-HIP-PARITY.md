# gfx1151 hipEngine versus llama.cpp HIP parity audit

Status: **gfx1151 audit complete and post-merge transfer queued; gfx1100 llama.cpp HIP prefill/decode parity closed, Vulkan decode and 128K prefill remain**
Date: **2026-07-16**
Machine-readable evidence:
[`gfx1151 parity audit`](../benchmarks/results/2026-07-14-gfx1151-llamacpp-hip-parity-audit.json),
[`gfx1100 final optimization sweep`](../benchmarks/results/2026-07-16-gfx1100-gguf-final-optimization-sweep.json),
[`gfx1100 peer-GDN promotion`](../benchmarks/results/2026-07-15-gfx1100-gguf-prefill-lcp5a-spill-free-peer-promotion.json),
[`gfx1100 decode attribution`](../benchmarks/results/2026-07-15-gfx1100-gguf-decode-lcpd3-attribution.json),
[`gfx1100 LCP-D2 gate`](../benchmarks/results/2026-07-14-gfx1100-gguf-decode-lcp-d2-parallel-reduce.json), and
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

The original audit's first implementation candidate was **LCP-1: an exact,
same-stream, long-token SSM-convolution kernel using llama.cpp's 32-token
shared-memory schedule**. It targets a measured 4K family gap of **954.438
versus 32.980 ms** without re-enabling the unstable isolated-AOTriton-stream
policy. The post-merge plan below now gates the already-written peer GDN
schedules before reopening that implementation work.

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

The July 14 defaults-only publication is superseded by the clean final
right-sized 1+3 sweep at `28b37356` on the complete therock HIP 7.15 stack.
Package defaults now include the admitted phase-liveness arena, persistent
cooperative router, peer-wave GDN, spill-free selected prefill, and the scoped
long-context parallel reducer. The final sweep records
**2716.648/3052.541/2953.101/2078.038/1559.878/1037.378 tok/s** prefill and
**92.833/98.148/100.522/88.240/76.691/62.669 tok/s** graph decode from 512
through 128K. All 18 final IDs are `9707`; maximum prefill/decode stdev over
median is **0.658%/0.223%**.

This closes llama.cpp HIP compute parity: prefill is **12.62-30.95% faster**
at all six shapes and decode is **2.85-26.02% faster** everywhere. Prefill also
beats llama.cpp Vulkan by **3.37-17.10%** from 512 through 64K; only 128K
Vulkan remains ahead by **3.88%**. Vulkan decode remains ahead at every shape,
with the smallest gap now **2.47% at 4K**.

Final tracked right-sized memory is
**21.228/21.295/21.670/22.234/22.879/24.168 GiB**. Against the broader
llama.cpp HIP whole-device readings, differences are only **-0.378/-0.323/
-0.004/+0.018/-0.016/+0.079 GiB**. This is practical capacity parity, not an
allocator-efficiency claim: the scopes still differ, and 32K/128K sit slightly
above HIP by 18/79 MiB. The peer route's additional Q/K/V live interval explains
the small increase from the post-LCP-M1 direct-GDN census without reopening a
material memory lane.

Evidence:
[`gfx1100 final optimization sweep`](../benchmarks/results/2026-07-16-gfx1100-gguf-final-optimization-sweep.json).

## gfx1100 residuals after closure

The final pass closes the llama.cpp HIP target and leaves two concrete reference
residuals rather than a broad GGUF deficit:

- Vulkan prefill remains **3.88% ahead only at 128K**. hipEngine wins 512-64K.
- Vulkan graph decode remains **13.87/8.75/2.47/3.91/8.42/11.52% ahead** from
  512 through 128K. The 4K row is the nearest promotion boundary.
- Tracked memory is within **-0.378 to +0.079 GiB** of llama.cpp HIP's broader
  whole-device reading. Do not chase the remaining cross-scope MiB as a
  performance optimization without a same-scope capacity failure.

The final clean therock-7.15 4K marked trace contains **8.652 ms GPU/token**.
Dense Q8 T16 GEMV is **39.45%**, selected-MoE T16 GEMV **21.41%**, attention
**9.06%**, lm-head **7.25%**, GDN **6.08%**, and the persistent router
**4.81%**. Exactly one new selected-Q4 pressure candidate reduced static VGPR
`195 -> 114` but regressed canonical graph decode **2.79%** and was removed.
Existing dense-Q8, selected-MoE dp4a/layout, and broad geometry screens are
closed.

At 32K, attention rises to **17.36%** of **9.724 ms GPU/token**, but the retained
prepare+parallel reducer is only **90.878 us/token (5.38% of attention)**. The
remaining **1.569 ms/token** is the already grouped-GQA split-K context scan.
Any further c=1 decode work therefore needs a genuinely new context-scan or
matrix-family algorithm; generic launch reduction, serial-reducer retuning, and
unchanged dp4a probes are not justified by the current evidence.

The source-grounded Vulkan advantage remains c=1-specific: smaller
single-subgroup workgroups, no cross-wave LDS reduction, RADV/ACO scheduling,
q8_1 activation-load coalescing, and graph-level MoE/post-op fusion. It is not
Vulkan WMMA decode, a wider-than-dp4a instruction, or a generic attention
advantage. Existing hipEngine dp4a diagnostics also show that changing the dot
instruction alone is insufficient; any new decode candidate must improve the
actual dominant family and full-model wall.

The post-GPF-9C pp512 residual trace now closes the missing short-shape
attribution. On the same W7900, current `chain_peer_wave32` and llama.cpp HIP
sum to **207.253 versus 203.301 ms** of GPU kernels. Peer GDN recurrence is only
**17.134 versus 16.522 ms**; recurrence algebra is no longer the candidate's
short-shape blocker. hipEngine instead leaves **28.061 ms** of idle gaps between
kernels versus llama.cpp's **8.935 ms**. That **+19.126 ms** queue starvation is
**82.9%** of the first-to-last-kernel span delta, despite hipEngine issuing
**1645 versus 2259** dispatches. Copy-to-RMSNorm and copy-to-copy boundaries
alone contribute **12.344 ms** of the hipEngine gaps. Dense Q8 is the largest
positive kernel residual (**+20.825 ms**), but selected Q4/Q5 is already
**21.819 ms faster** and offsets it. `LCP-2A` has now removed the first measured
submission defect: request/chunk metadata was synchronously uploaded before all
40 layers at pp512. Retained `e03e5a34` removes exactly **240 copies**, reducing
matched dispatches **1645 -> 1405**, queue idle **27.956 -> 15.163 ms
(-45.76%)**, and trace span **235.907 -> 224.511 ms (-4.83%)**. Clean pp512
improves **2210.729 -> 2292.186 tok/s (+3.68%)** with stable IDs and unchanged
memory, but remains **4.98% below** the required floor. `LCP-2B` then removes
the remaining 40 compact-WMMA `wmma_total` D2H reads with a tight
routing-independent tile bound. Matched dispatches fall **1405 -> 1365**, queue
idle falls **15.163 -> 11.634 ms (-23.27%)**, and clean pp512 improves
**2292.186 -> 2334.451 tok/s (+1.84%)** with stable IDs, unchanged memory, and
decode within -0.053%; 4K prefill improves +0.70%. The cumulative queue work
moves pp512 **2210.729 -> 2334.451 tok/s (+5.60%)**, but remains **3.23% below**
the required floor. Production therefore remains on exact direct-LDS32.

Continuation order is now:

1. `LCP-3A` is closed as a rejection: extending exact activation sharing from
   two to four independent FP16-WMMA waves is flat at width 8192 (-0.185%) but
   regresses width 4096 (+2.24%) and short-K width 2048 (+8.87%). The measured
   pp512 family mix projects about a 0.3 ms loss, so all candidate code was
   removed before model routing.
2. `LCP-3B` is also closed as a rejection: the direct prequantized Q8_1 x
   Q8T16 integer-WMMA body passes its primitive numerical gate but is **44.66%
   slower** than production before quantization; quantization adds only 0.016
   ms. The candidate was removed before profiler/model routing.
3. `LCP-3C` is complete. llama.cpp's measured Q8_0 MMQ uses a **256-thread,
   128-output x 128-token** workgroup, stages K256 weights plus one D4-Q8 K128
   activation half in **57,856 B** dynamic LDS, and issues two signed-int WMMA
   K16 calls per 32-K scale interval. The measured kernel is **232 VGPR, zero
   scratch**. Its pp512 width totals are 11.542/4.559/8.625/5.700 ms at
   8192/4096/2048/512, exactly resolving the ~30.4 ms family. This is materially
   different from LCP-3B's direct 64-output x32-token body. `LCP-3D` reproduced
   that tile over byte-lossless resident T16 and passed D4 bytes plus primitive
   quality, but failed the primary body gate **0.523062 -> 1.144524 ms
   (+118.81%)**; D4 packing adds only 0.007061 ms. T16's K-major 16-column
   bytes require four-byte gather/packing to fill llama.cpp's output-major
   shared fragments, so the candidate is removed. `LCP-3E` then tested the
   source-compatible output-major raw layout directly. Its final spill-free WGP
   body matched the source fragment/WMMA instruction counts and passed primitive
   correctness, but still lost the frozen primary gate: **0.521823 -> 0.542442
   ms (+3.95%)** prequantized and **0.549562 ms (+5.32%)** including D4 packing.
   It was removed before profiler/model routing. A raw sidecar is independently
   forbidden: the traced dense-Q8 weights are about **1.390 GiB**. Close
   dense-Q8 prefill tile/layout experiments pending genuinely new evidence; do
   not retune selected Q4/Q5, short full attention, or generic launch count.
4. `LCP-4A` is promoted. The apparent post-Q8 convolution residual was mostly
   the normal prefill body's **20 private bytes/thread**, forced by `volatile`
   products used to prevent FP32 contraction. A capture-free body emits explicit
   sequential `v_mul_f32_e32`/`v_add_f32_e32`, keeps output/final-state bytes
   exact, and uses zero scratch. Cached pp512 Conv falls **8.496 -> 1.894 ms / 30
   (-77.71%)**; clean production direct-LDS32 prefill improves **+1.44%/+1.86%**
   at 512/4K with stable IDs. The clean follow-up identifies the production gap
   precisely: exact/peer/llama.cpp total kernels are **369.285/203.808/203.301
   ms**, and their GDN families are **199.030/20.840/16.522 ms**. Peer versus
   llama.cpp has only **3.071 ms** of trace-span residual, but the clean peer
   512/4K speed gate reaches **2385.677/2585.343 tok/s** and still misses the
   frozen 512 floor by **1.104%**. Keep exact production; target only that final
   peer queue/span residual or new exact-parallel GDN evidence.
5. Continue profile-directed dense-Q8/selected-MoE **decode** work for Vulkan
   parity independently, retaining 4K first and escalating to the 512 and 128K
   endpoints.

`LCP-M1` is complete and promoted. New chunked/prefix GDN research is deferred
until the measured peer-route integration and dense-Q8 residuals are exhausted;
do not rerun rejected LDS16, two-lane, tree, or WY8 designs unchanged.

Machine-readable evidence:
[`parity rebaseline`](../benchmarks/results/2026-07-14-gfx1100-gguf-parity-rebaseline.json),
[`GPF-9C residual attribution`](../benchmarks/results/2026-07-15-gfx1100-gguf-gdn-peer-wave32-residual-attribution.json),
[`LCP-2A metadata reuse`](../benchmarks/results/2026-07-15-gfx1100-gguf-prefill-chunk-metadata-reuse.json),
[`LCP-2B tight no-read`](../benchmarks/results/2026-07-15-gfx1100-gguf-compact-wmma-tight-no-read.json),
[`LCP-3E raw MMQ rejection`](../benchmarks/results/2026-07-15-gfx1100-gguf-raw-q8-mmq128-rejected.json),
[`LCP-4A no-scratch Conv`](../benchmarks/results/2026-07-15-gfx1100-gguf-conv-no-scratch.json),
and
[`post-LCP-4A residual attribution`](../benchmarks/results/2026-07-15-gfx1100-gguf-post-conv-residual-attribution.json).

## gfx1151 post-merge transfer plan

The current gfx1100 branch is intentionally cheap to bring back to gfx1151 at
the **source** level. `hip_gfx1151` aliases the complete registered gfx1100
kernel key space and JIT-compiles the shared HIP bodies as native gfx1151 code
objects. The new peer-wave GDN, clustered GDN, long-context parallel reducer,
and exact cooperative-router kernels therefore require no second source port.
That does **not** transfer performance evidence or automatic policy: shared
source, architecture-local admission remains the rule.

| Work from the gfx1100 pass | What transfers after merge | gfx1151 policy before a hardware gate |
| --- | --- | --- |
| GPF-2E direct LDS32 GDN, GPF-3A shared-X selected Q4, GPF-5A two-wave dense Q8 | Already originated, passed, and is enabled on gfx1151 | Reproduce as the clean baseline; do not claim it again as new work |
| GPF-9C peer wave32/XOR GDN and GPF-9D clustered8 GDN | Kernel bodies, registry keys, explicit modes, primitive tests, and the 18-prompt gate harness | Diagnostic only; `auto` remains `chain_lds32_direct` |
| LCP-D2 parallel long-context split reduction | Native gfx1151 alias and explicit environment selector | Disabled; serial reduction remains automatic |
| LCP-M1 phase-liveness scratch arena | Architecture-neutral host allocator and lifetime contract | Disabled; dedicated scratch remains automatic |
| Cooperative F32-weight decode router and persistent completion counter | Shared kernels and current runtime selection path | Re-measure immediately because the selectors are currently global rather than backend-package-scoped |
| Rejected GPF-6/7/8/9A/9B and rejected Q4/Q6/Q8 indexing experiments | Historical evidence and tests only | Do not rerun unchanged |
| LCP-1 shared-token convolution | Source idea and exact fixture/gate design; the gfx1100 candidate was removed | Reprofile first, then reimplement only if convolution remains material on gfx1151 |

The post-merge execution order is:

1. **Establish a clean merged-main control.** Build every shared kernel with
   `HIPENGINE_HIP_ARCH=gfx1151`, run the focused registry/primitive bundle, and
   record defaults-only 512/1K/4K/32K/64K/128K prefill plus graph decode. A/B
   the cooperative router and persistent counter first; if either regresses,
   move its automatic selection behind gfx1151 package metadata before any
   other tuning.
2. **Gate the already-written peer GDN schedules.** Run `chain_peer_wave32`
   first, then `chain_peer_cluster8`, using the same prospectively frozen
   contract as gfx1100: CPU-reference primitive correctness, the full
   18-prompt category plus heldout suite at KL <= 0.05 and top-1 >= 90%,
   deterministic execution, strict decode non-regression, and both matched
   llama.cpp HIP 512/4K speed floors. The gfx1100 outcomes are useful priors,
   not gfx1151 verdicts: wave32 passed semantics but missed the W7900 512 floor;
   clustered8 passed quality but missed strict decode by 0.00129%.
3. **Reprofile the winning/default route at 512 and 4K.** The retained gfx1151
   trace attributed 205.570/1700.469 ms to GDN, 14.303/954.438 ms to
   convolution, and 110.526/749.444 ms to dense Q8. Do not assume that ranking
   survives a different GDN schedule. If convolution remains material,
   reopen LCP-1 on gfx1151 with the original exact 32-token shared-memory gate;
   otherwise advance to dense Q8. Selected Q4/Q5 and short/mid full attention
   remain non-targets unless the fresh profile reverses their measured lead.
4. **Transfer the orthogonal gfx1100 wins independently.** Gate LCP-D2 at
   32K/64K/128K with the long-context logit/KL check and graph-decode A/B; this
   directly targets gfx1151's retained 128K decode deficit. Gate LCP-M1 with
   the 4K byte-exact logit A/B, six-shape allocation census, and prefill/decode
   non-regression. Promote each through gfx1151 package metadata only after its
   own evidence passes.
5. **Publish one final defaults-only rollup.** Repeat the six-shape 1+3 protocol,
   full semantic/correctness gates, and matched llama.cpp comparison; update
   the benchmark artifact, scoreboard, changelog, and worklog together.

This ordering avoids spending a second implementation cycle on code already
available through the registry alias, while preserving the lesson from the
first transfer: even within gfx11, launch geometry, occupancy, queue behavior,
and context thresholds remain hardware-specific.

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

The llama.cpp schedule explains the measured **4.65x/5.15x** family advantage.
Reduction order is no longer a universal admission requirement for an
algebraically equivalent peer schedule: exact state bytes and free-running
trajectory identity are diagnostics, while the CPU-reference primitive and
18-prompt KL/top-1/determinism/decode gates own quality. The generic K2 and raw
wave-tree routes still fail that current KL gate. The later peer-exact geometry
is mixed on gfx1100: GPF-9C passes quality but misses the 512 speed floor, while
GPF-9D passes quality but misses strict decode by 0.00129%.

**Decision:** do not make a reassociated tree automatic without the full product
and architecture-local speed gate. Reuse the explicit GPF-9C/9D modes for the
independent gfx1151 transfer before opening new chunked/token-prefix research.

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

## Ranked parity outcome and next decisions

| Rank | ID | Outcome / next work | Current evidence | Reopen gate |
| ---: | --- | --- | --- | --- |
| 1 | `LCP-5A` | **Closed/promoted on gfx1100:** spill-free selected prefill admits peer-wave GDN as package default | Final prefill beats llama.cpp HIP at all six shapes and Vulkan through 64K | Reopen gfx1100 prefill only for the remaining 128K Vulkan gap with a new long-context algorithm |
| 2 | `LCP-D1/D2/D3` | **Closed for this pass:** parallel reducer and persistent router retained; selected-Q4 pressure candidate rejected | Final decode beats llama.cpp HIP everywhere; Vulkan gap is 2.47-13.87% | Require new dominant-family/context-scan evidence; do not rerun unchanged dp4a/tile/launch-count candidates |
| 3 | `LCP-M1` | **Closed/promoted:** phase-liveness arena plus peer Q/K/V lifetimes | Final tracked memory is within -0.378..+0.079 GiB of HIP whole-device rows | Reopen only for a same-scope capacity failure, not small cross-scope MiB |
| 4 | `tail4_hadamard_group32` | **Quality/storage pass; default rejected:** keep explicit | 18.75% persistent K/V saving, but 4K/128K speed regresses and prefill has a 1.002 GiB transient | Remove transient and improve long-context group32 attention, then repeat full suite |
| 5 | `gfx1151 transfer` | Separate post-merge hardware lane | Shared source is ready; architecture-local policy is not assumed | Follow the transfer plan below with clean gfx1151 correctness/profile/sweep evidence |

There is no invented minimum win. Exact, same-suite non-regressive
improvements remain retainable under the project evidence policy.

## Explicit non-targets

- Do not replace hipEngine selected Q4/Q5 with generic llama.cpp MMQ; the
  measured hipEngine families are already faster.
- Do not replace short/mid AOTriton with llama.cpp FlashAttention; the measured
  hipEngine family is already faster.
- Do not re-enable GPF-4 isolated AOTriton queues by default; its final 32K/128K
  stability gate remains rejected.
- Do not label llama.cpp's GDN tree reduction exact or automatic without the
  numerical/semantic, decode, and architecture-local speed gates.
- Do not prioritize generic graph or raw launch-count reduction for prefill;
  llama.cpp launches more kernels. The measured exception is specific
  copy-boundary queue starvation, not dispatch count itself.
- Do not infer memory efficiency from the public cross-column peak values; the
  scopes differ.

## LCP-1 implementation gate

**gfx1100 result (2026-07-14): rejected and removed.** The candidate passed the
primitive byte gate but failed the normal-stream full-model wall gate after the
promoted schedule profile invalidated the old hotspot premise. The checklist
below remains the original gfx1151-local gate, not open gfx1100 work.

If the post-GDN gfx1151 profile reopens LCP-1, its coding tranche should remain
narrow:

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
