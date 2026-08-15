# hipEngine Topline Benchmarks

Last updated: **2026-08-15**

This file is the current benchmark scoreboard. It intentionally contains only
current user-facing results, compact protocol/status notes, and links to the
authoritative evidence. It is not an optimization journal.

## Root README performance summary

The root README exports this compact retained summary verbatim.

<!-- BEGIN TOPLINE:README_HIGHLIGHTS -->
### Radeon Pro W7900 (`gfx1100`)

| Model and format | Test | Prompt processing (tok/s) | Text generation (tok/s) |
| --- | --- | ---: | ---: |
| Qwen3.6-35B-A3B ParoQuant W4 | 512 input tokens, 128 output tokens | **2917.732** | **115.599** |
| Qwen3.6-35B-A3B GGUF `Q4_K_M` | 512 input tokens, 128 output tokens | **2716.648** | **92.833** |
| Qwen3.6-27B Dense GGUF `Q4_K_M` | 512 input tokens, 128 output tokens | **865.179** | **28.368** |
| Laguna S 2.1 GGUF `UD-Q2_K_XL` | 4,096 input tokens; prompt processing only | **440.893** | — |

#### Multiple requests

Each value is the total tokens per second across all active requests:

| Model and interface | 1 request | 2 requests | 4 requests | 8 requests | 9 requests | 13 requests |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B GGUF `UD-Q4_K_M` (engine) | **98.263** | **148.944** | **209.304** | **266.479** | — | — |
| Qwen3.6-35B-A3B GGUF `UD-Q4_K_M` (server) | **72.169** | — | — | **158.542** | **137.001** | **129.507** |

#### MTP

| Model and mode | Text generation | Speed compared with AR |
| --- | ---: | ---: |
| Qwen3.6-27B Dense GGUF `Q4_K_M` — MTP-3 | **60.875 tok/s** | **2.9672x** |
| Qwen3.6-35B-A3B GGUF `UD-Q4_K_M` — MTP-2 | **122.67 tok/s** | **1.2679x** |

### Radeon RX 7900 XTX (`gfx1100`)

| Model and format | Test | Prompt processing (tok/s) | Text generation (tok/s) |
| --- | --- | ---: | ---: |
| Qwen3.6-27B Dense GGUF `Q4_K_M` | 512 input tokens, 128 output tokens | **973.457** | **33.521** |

### Strix Halo / Radeon 8060S (`gfx1151`)

| Model and format | Test | Prompt processing (tok/s) | Text generation (tok/s) |
| --- | --- | ---: | ---: |
| Qwen3.6-35B-A3B GGUF `UD-Q4_K_M` | 512 input tokens, 128 output tokens | **1369.489** | **54.330** |
| Laguna S 2.1 GGUF `Q4_K_M` | 512 input tokens, 128 output tokens | **654.249** | **23.221** |
| Maple-Preview 2-bit | 512-token prompt test; varied prompts for generation | **754.458** | **153.201** |

#### Multiple requests

| Model and interface | 1 request | 2 requests | 4 requests | 8 requests |
| --- | ---: | ---: | ---: | ---: |
| Maple-Preview 2-bit (engine) | **123.131** | **165.697** | **202.038** | **214.788** |

#### MTP

| Model and mode | Text generation | Speed compared with AR |
| --- | ---: | ---: |
| Qwen3.6-35B-A3B GGUF `UD-Q4_K_M` — MTP-2 | **80.10 tok/s** | **1.4282x** |

### RTX PRO 6000 Blackwell (`sm_120a`)

| Model and format | Test | Prompt processing (tok/s) | Text generation (tok/s) |
| --- | --- | ---: | ---: |
| Maple-Preview 2-bit | 512-token prompt test; varied prompts for generation | **1917.492** | **402.361** |

These rows use different models and tests. Compare results only when their
protocols match. MTP-2 and MTP-3 use two and three draft tokens per cycle. The
35B-A3B MTP-2 path matches llama.cpp's MTP output on the validated prompt suite.
It remains opt-in because that output can differ from normal autoregressive
generation.
<!-- END TOPLINE:README_HIGHLIGHTS -->

## Where detailed evidence lives

| Need | Source |
| --- | --- |
| Exact commands, revisions, model fingerprints, correctness gates, samples, profiler summaries | [`benchmarks/results/`](results/) compact JSON artifacts |
| Reverse-chronological benchmark changes | [`CHANGELOG.md`](CHANGELOG.md) |
| Superseded benchmark notebook through 2026-07-10 | [`HISTORY.md`](HISTORY.md) |
| Benchmark rules and reproduction procedures | [`docs/BENCHMARK.md`](../docs/BENCHMARK.md) |
| MTP-specific protocols and terminology | [`MTP.md`](MTP.md) and [`docs/MTP-LLAMACPP-PARITY.md`](../docs/MTP-LLAMACPP-PARITY.md) |
| Kernel and implementation decisions | [`worklog/entries/`](../worklog/entries/) and [`WORKLOG-LEGACY.md`](../WORKLOG-LEGACY.md) |
| Hardware-specific RX 7900 XTX report | [`7900XTX.md`](7900XTX.md) |

The post-2026-07-10 intermediate narratives removed by this cleanup remain in
the linked JSON artifacts and changelog. The complete pre-cleanup Markdown is
also recoverable from Git without keeping a 1.1 MB notebook in the live
scoreboard:

```bash
git show 6a8d38ae70b9e2c4244df10d8621db83da6c8112:benchmarks/README.md
```

## Evidence status

| Status | Meaning | Eligible for a current numeric table? |
| --- | --- | --- |
| **Retained** | Correctness, provenance, repetition, and protocol gates passed for the named scope. | Yes. |
| **Current snapshot** | Clean current-production measurement used to describe the shipped route, but not itself a new optimization claim. | Yes, with that label. |
| **Diagnostic** | Useful attribution or comparison with a known limitation. | No; keep it in its artifact/changelog unless it explains a current blocker. |
| **Stale / superseded** | A newer route, dependency, or evidence contract replaced it. | No. |
| **Blocked / rejected** | The protocol could not complete or the candidate failed a gate. | No numeric topline. |

A row is scoped by platform, GPU, model fingerprint, quantization, KV type,
backend, workload, concurrency, speculative policy, and timing window. A newer
diagnostic never replaces a retained row.

## Current single-request scoreboards

### Radeon Pro W7900: Qwen3.6-35B-A3B

These are the current six-shape hipEngine publication sweeps. PARO and GGUF use
different weight formats and should not be treated as a same-math A/B. `Peak`
is hipEngine tracked allocator high-water.

| Workload | PARO prefill | PARO decode | PARO peak | GGUF prefill | GGUF decode | GGUF peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512/128 | **2917.732** | **115.599** | 18.144 GiB | **2716.648** | **92.833** | 21.228 GiB |
| 1K/128 | **2995.876** | **103.238** | 18.367 GiB | **3052.541** | **98.148** | 21.295 GiB |
| 4K/128 | **2943.038** | **105.943** | 19.161 GiB | **2953.101** | **100.522** | 21.670 GiB |
| 32K/128 | **2108.868** | **92.438** | 19.864 GiB | **2078.038** | **88.240** | 22.234 GiB |
| 64K/128 | **1584.131** | **78.260** | 20.403 GiB | **1559.878** | **76.691** | 22.879 GiB |
| 128K/128 | **1056.252** | **60.663** | 22.124 GiB | **1037.378** | **62.669** | 24.168 GiB |

Evidence: [`PARO five-run sweep`](results/2026-07-12-w7900-v030-8116c453-hipengine-paro-packed-5run.json)
and [`GGUF final optimization sweep`](results/2026-07-16-gfx1100-gguf-final-optimization-sweep.json).
The matched llama.cpp HIP/Vulkan comparison columns and their differing memory
scopes remain in the linked artifacts and the archived rollup rather than being
repeated here.

### Radeon Pro W7900: Qwen3.6-27B Dense GGUF

| Workload | Prefill | Autoregressive decode | Tracked peak | Status |
| --- | ---: | ---: | ---: | --- |
| 512/128 | **865.179 tok/s** | **28.368 tok/s** | 15.605 GiB | Pair-only Q6-QKV/Q4 gate + pair-produced full Q |
| 1024/128 | **890.634 tok/s** | **28.851 tok/s** | 15.720 GiB | Pair-only Q6-QKV/Q4 gate + pair-produced full Q |
| 4096/128 | **865.653 tok/s** | **26.332 tok/s** | 16.368 GiB | Exact pair-only/full-Q fallback + packed producers |

These rows use one discarded warmup plus three measured PM4 resets, executed
strictly serially across boards. The ordered pair-only Q6-QKV/Q4-gate route
improves binding W7900 full prefill **+0.45%/+0.74%** at M512/M1024 (11/14
paired wins); M4096 is an identical-owner exact fallback. Complete quality is
330/330 top-1 at max KL 0.014671, tracked peaks are byte-identical, and
decode/MTP ownership is unchanged. The latest absolute 1K/4K rows are
0.33%/0.30% below the prior publication under monotonic run drift, while the
same-session route wins both binding rows. The exact natural-octet Q5 source-F16
producer improves binding counterbalanced W7900
full prefill **+1.49%/+0.78%/+0.61%** at 512/1K/4K (20/21 wins), with exact
trajectories and byte-identical tracked peaks. Independent 512/1K absolute rows
are 0.47%/0.15% below the prior publication under run-to-run spread, while the
binding same-session route wins every row. The selective pair-owned Q4
source-F16 producer improves binding counterbalanced
W7900 full prefill **+1.12%/+0.67%/+0.59%** at 512/1K/4K (20/21 wins), with
exact trajectories and byte-identical tracked peaks. Its independent 4K absolute
row is 0.31% below the prior publication under run-to-run spread, while the
same-session candidate wins 6/7. The exact record-owned planar-Q6 producer
improves counterbalanced W7900 full prefill
**+3.31%/+1.71%/+1.30%** at 512/1K/4K (21/21 wins) with exact trajectories and
byte-identical tracked peaks. The exact
unequal-output Q4 pair improves counterbalanced W7900 full prefill
**+1.39%/+1.53%/+1.17%** at 512/1K/4K; the isolated 4K confirmation wins 7/7,
while tracked residency and decode ownership are unchanged. The Q5
recurrent-output extension improves counterbalanced W7900 full prefill
**+4.91%/+5.07%/+4.82%** at 512/1K/4K (21/21 wins) without changing decode
ownership or tracked residency. Exact zero-workspace selected rocBLAS FP16 GEMMs
also improve the earlier counterbalanced W7900 package **+0.17%/+1.32%** at
512/4K; no selected prior-shape 1K solution exists.
Against the same-commit diagnostic dual-layout rollback, the earlier shared
package default improves
prefill **152.61-184.82%**, decode **17.58-18.74%**, and whole-device peak delta
**45.50-47.03%**, with exact outputs and clean teardown. The current exact
natural suite is true AR **20.516 tok/s** and B3 **60.875 tok/s / 2.9672x**;
its lower absolute rate than the historical **61.147** row is protocol/code
drift, not a single-layout regression. Evidence:
[`same-commit W7900 non-regression`](results/2026-08-12-qwen36-27b-w7900-single-layout-non-regression.json).
The superseded dual-layout publication remains in the
[`latest-Vulkan parity exhaustion audit`](results/2026-08-07-qwen36-27b-latest-vulkan-parity-exhaustion-audit.json).

### Radeon RX 7900 XTX: Qwen3.6-27B Dense GGUF — blocked cross-engine closure

The pre-campaign dual-layout hipEngine path could not admit this model on the
23.984-GiB XTX. The package-default sole-T16 route now fits and is stable; its
model-qualified prefill-only Q4/Q5/Q6 source-F16 arithmetic passes the complete
category quality gate and same-commit W7900 safeguard. The compact peer-GDN
route closes all three frozen prefill targets, its Q/K scratch is sized to the
compact 16-K-head ABI rather than the 48-V-head fallback ABI, dense SiLU reuses
its dead BF16 gate plane as the down-projection input, and private-c1 decode
scratch uses one physical owner instead of 188. The campaign still does **not**
meet complete cross-engine acceptance: Vulkan sets
the lower memory floor at every shape, 4K AR decode remains below HIP, and
Vulkan wins selected MTP. The selector-unset 512/128, 1024/128, 4096/128, and
8192/128 hipEngine matrix is retained as the current partial result. Clean
same-commit llama.cpp `c8e03ce81` HIP and Vulkan establish the frozen speed and
whole-device VRAM targets:

| Workload | HIP prefill | Vulkan prefill | HIP decode | Vulkan decode | Lower peak delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512/128 | **964.606** | 870.872 | **33.025** | 13.391 | **15.690 GiB** (Vulkan) |
| 1024/128 | **981.040** | 836.898 | **32.924** | 13.379 | **15.700 GiB** (Vulkan) |
| 4096/128 | **946.733** | 835.765 | **32.560** | 13.309 | **15.912 GiB** (Vulkan) |
| 8192/128 diagnostic | **906.648** | 829.630 | 32.779 | **37.669** | **16.166 GiB** (Vulkan) |

Prefill is llama-bench `avg_ts` over five internal repetitions after one
warmup; 512-4K decode is 128 context-matched server transitions. The fresh 8K
decode row is standardized `llama-bench`, not a context-matched server result.
On the complete ten-prompt
natural suite, HIP selects B2 at **46.863 tok/s / 1.4841x AR / 16.940 GiB**
peak delta, while Vulkan selects B4 at **81.952 tok/s / 6.1223x AR / 16.673
GiB**. The frozen hipEngine gates add a 1% speed margin and require no more than
the lower Vulkan memory row.

Current compact peer-GDN plus constrained long-row recoloring plus split-GDN
lifetime reuse plus dense-SiLU alias plus single-owner decode-scratch hipEngine
matrix (one warmup plus three strictly serial, selector-unset persistent-session
reset/replays per shape; 5-ms whole-device sampling around each process):

| Workload | Prefill | Decode | Tracked peak | Whole-device peak delta | Gate status |
| --- | ---: | ---: | ---: | ---: | --- |
| 512/128 | **974.481 tok/s** | **33.516 tok/s** | **15.587 GiB** | **16.029 GiB** | **prefill pass** / **decode pass** / memory fail |
| 1024/128 | **1006.756 tok/s** | **34.494 tok/s** | **15.668 GiB** | **16.182 GiB** | **prefill pass** / **decode pass** / memory fail |
| 4096/128 | **987.193 tok/s** | **31.414 tok/s** | **16.150 GiB** | **16.813 GiB** | **prefill pass** / decode fail / memory fail |
| 8192/128 diagnostic | **820.281 tok/s** | **29.377 tok/s** | **16.408 GiB** | **17.068 GiB** | standardized speed below llama.cpp / memory fail |

The latest retained planner first computes the established size-first layout,
then at rows >=4096 moves only eight fields with total declared live duration
>=5 ahead of that order and accepts the candidate only when it is strictly
smaller. This keeps `attn_out` at **48.0625 MiB**, shrinks the 4K-row arena
**401.0625 -> 372.375 MiB (-28.6875 MiB, -7.15%)**, and reduces 4K/8K tracked
peak by exactly **28.6875 MiB** and whole-device peak by
**31.480/31.465 MiB**. The rows<4096 gate is required: unrestricted recoloring
reproducibly made the second short-row NextN target-logit row NaN, while the
explicit size-first control and the final B1-B3 transaction pass. All four
prefill/decode deltas are within **0.172%**, the 8K root+128 oracle is exact,
and the W7900 4K safeguard is clean. Direct-output variants that saved more
memory remain removed after regressions up to **1.256%**. The remaining gaps to
the lower Vulkan memory floors are **0.340/0.482/0.901/0.902 GiB**, so memory
parity remains open. Evidence:
[`retained constrained recoloring`](results/2026-08-14-qwen36-27b-constrained-liveness-recoloring-retained.json),
[`retained split-GDN lifetime reuse`](results/2026-08-14-qwen36-27b-split-gdn-conv-lifetime-retained.json),
[`retained single-owner decode-scratch arena`](results/2026-08-14-qwen36-27b-private-c1-decode-scratch-arena-retained.json),
[`rejected GDN value/recurrent alias`](results/2026-08-14-qwen36-27b-gdn-value-recurrent-alias-rejected.json), and
[`retained compact Q/K scratch`](results/2026-08-14-qwen36-27b-compact-gdn-qk-scratch-retained.json).

The bounded sole-T16 Q4/Q6-to-F16/rocBLAS owners improve prefill over the prior
Q6-only matrix by **2.62%/2.40%/2.43%** at 512/1K/4K while decode changes
**-0.21%/-0.03%/-0.02%**, tracked residency is unchanged, and the complete
10-prompt category suite passes with minimum per-prompt top-1 **96.97%** and max
KL **0.03176**. Evidence: [`retained Q4 FFN-down F16 prefill`](results/2026-08-13-qwen36-27b-q4-ffn-down-f16-rocblas-prefill-retained.json).
A subsequent byte-neutral extension admits only 1,024-output Q4 full-attention
K/V. Counterbalanced full-prefill A/B improves XTX **+0.326%/+0.172%/+0.079%**
and W7900 **+0.123%/+0.203%/+0.315%** at 512/1K/4K; complete quality improves
to max KL **0.01563**. Because independent-run thermal drift exceeds this small
increment, the higher FFN-only absolute matrix above remains the conservative
public rollup. Evidence: [`retained narrow-attention F16`](results/2026-08-13-qwen36-27b-q4-narrow-attention-f16-prefill-retained.json).
The next byte-neutral extension admits full-attention output and improves the
same counterbalanced XTX rows **+0.350%/+0.306%/+0.074%** and W7900 rows
**+0.096%/+0.299%/+0.301%**, with complete max KL **0.01744**. It likewise does
not replace the conservative independent-run topline. Evidence:
[`retained attention-output F16`](results/2026-08-13-qwen36-27b-q4-attention-output-f16-prefill-retained.json).

The earlier prefill exhaustion audit is superseded by exactly its stated reopen
condition: a materially new operation-complete, byte-neutral sole-T16 Q4
dataflow. One four-wave kernel reuses each BF16 activation fragment across both
FFN gate/up weights, rounds both projections at the existing BF16 boundary, and
applies SiLU without global intermediates. It is BF16-bit exact and improves
counterbalanced full prefill on W7900 **+5.43%/+5.78%/+5.43%** and XTX
**+5.80%/+6.21%/+6.02%**, all 42/42 pairs winning. The independent XTX matrix
is now **852.668/914.600/901.068 tok/s**, still **11.60%/6.77%/4.82% below**
clean llama.cpp HIP and requiring **+13.13%/+7.26%/+5.07%** more throughput for
raw parity. The objective therefore remains active; select the next lane from a
fresh retained-path profile rather than the obsolete exhaustion profile.
A subsequent exact rocBLAS solution-index policy improves counterbalanced XTX
full prefill **+0.43%/+1.33%** at 512/4K and W7900 **+0.17%/+1.32%**, with
unqualified shapes and rocBLAS versions falling back to standard dispatch. The
independent XTX deficit is now **11.26%/6.45%/3.63%** at 512/1K/4K. A final
exact dataflow cleanup shares the existing F16 activation plane across each
admitted full-attention K/V pair, removing **16** M512 casts/launches and cutting
total profiled kernel sum **0.246%**; aggregate full-prefill movement is noise,
so no topline number is changed. Evidence:
[`shared F16 activation cast`](results/2026-08-13-qwen36-27b-shared-f16-activation-cast-retained.json),
[`rocBLAS solution indices`](results/2026-08-13-qwen36-27b-rocblas-solution-indices-retained.json),
[`dual-WMMA SiLU prefill`](results/2026-08-13-qwen36-27b-q4-dual-wmma-silu-prefill-retained.json),
and the now-historical
[`prefill exhaustion audit`](results/2026-08-13-qwen36-27b-prefill-target-exhaustion-audit.json).
The Q5T16 diagnostic is now promoted for only the recurrent-output K6,144/N5,120
bulk-prefill shape. Complete quality passes 330 transitions at minimum
per-prompt top-1 **96.97%** and max KL **0.00934** with 480 asserted candidate
dispatches. Counterbalanced full prefill improves XTX
**+4.74%/+4.69%/+4.66%** and W7900 **+4.91%/+5.07%/+4.82%** at 512/1K/4K,
all **42/42** pairs winning with byte-identical tracked peaks. The independent
XTX matrix moves **855.960/917.774/912.359 -> 892.123/963.237/956.770 tok/s**;
4K is **+1.060%** versus llama.cpp HIP and clears its frozen +1% gate by
**0.060%**, while 512/1K remain **7.514%/1.815% below** HIP. Decode and natural
MTP rows remain exact Q5T16 fallbacks. Evidence:
[`retained Q5 recurrent prefill`](results/2026-08-13-qwen36-27b-q5t16-f16-rocblas-prefill-retained.json).
A later packed-column producer leaf follows the Q5T16 payload rather than
reopening integer MMQ: adjacent-pair ownership improves the scalar producer
**1.490-1.538x**, while natural-octet ownership improves it **1.738-1.827x** on
both gfx1100 boards and both 1024/1280-column production tiles. All **248/248**
leaf pairs win with byte-exact source-F16 output and no new residency/workspace.
The octet owner now also wins every binding complete-engine cell: XTX
**+1.01%/+0.42%/+0.37%** and W7900 **+1.49%/+0.78%/+0.61%** at 512/1K/4K,
with **41/42** pairs, exact trajectories, unchanged peaks, and complete category
and llama-compatible safeguards. Selector-unset XTX is
**952.759/990.403/982.619 tok/s**. Evidence:
[`retained Q5 packed-column producers`](results/2026-08-13-qwen36-27b-q5-packed-column-f16-producers-retained.json)
and [`retained natural-octet Q5 producer`](results/2026-08-13-qwen36-27b-q5-octet-producer-engine-retained.json).
The subsequent exact unequal Q4/Q4 linear-attention owner reuses each BF16 K16
fragment across QKV and gate for their common 6,144 output columns, then computes
the QKV-only tail in the retained singleton geometry. The actual-weight leaf is
BF16-bit exact and improves XTX **1.285x/1.164x/1.128x** and W7900
**1.296x/1.167x/1.138x** at M512/1K/4K, all 90/90 component pairs winning.
Counterbalanced full prefill improves XTX **+1.50%/+0.90%/+0.97%** and W7900
**+1.39%/+1.53%/+1.17%**; complete quality is **330/330 top-1 / KL 0**, tracked
peaks are byte-identical, and decode/MTP rows retain exact singleton owners.
At that checkpoint, the independent XTX matrix became
**912.509/969.550/956.213 tok/s** and 4K was threshold-flat. The later exact Q6
producer result below supersedes those absolute rows.
Evidence: [`retained unequal Q4 pair prefill`](results/2026-08-13-qwen36-27b-q4-unequal-dual-prefill-retained.json).
The subsequent exact planar-Q6 producer assigns one thread to each contiguous
12-byte qmicro record and emits K4xN4 values, avoiding four separate column
owners re-addressing the same record. It improves counterbalanced full prefill
on XTX **+2.54%/+1.29%/+1.00%** and W7900 **+3.31%/+1.71%/+1.30%**, all 42/42
pairs winning with exact trajectories and unchanged tracked peaks. At that
checkpoint, selector-unset XTX reached **939.535/985.387/975.862 tok/s**.
Evidence: [`retained direct Q6 F16 producer`](results/2026-08-13-qwen36-27b-q6-direct-f16-producer-engine-retained.json).
The exact Q4T16 pair-owned source-F16 producer then moved from diagnostic leaf
to a fail-closed quant/shape/row policy. It improves binding counterbalanced
full prefill on XTX **+0.96%/+0.46%/+0.36%** and W7900
**+1.12%/+0.67%/+0.59%**, with **41/42** paired wins, exact trajectories, and
byte-identical tracked peaks. Production tracing observes **264** pair launches
and zero scalar Q4 producers at M512; the full llama-compatible B1-B3 safeguard
is output/acceptance exact. Selector-unset XTX is now
**945.796/987.169/977.479 tok/s** at that checkpoint. The later Q5 octet owner
supersedes those absolute rows. Evidence:
[`retained selective Q4 pair producer`](results/2026-08-13-qwen36-27b-q4-pair-selective-engine-retained.json).
The later bounded full-attention-Q extension uses that exact pair producer only
at M512-M2047. Binding XTX/W7900 M512/M1024 improves
**+0.780%/+0.429%** and **+0.777%/+0.688%** with **26/28** admitted pairs
winning; M2048/4K remains exact. The next ordered pair-only route admits Q4
gates only behind the 24 already-admitted Q6-QKV peers, improving binding XTX
**+0.801%/+0.571%** and W7900 **+0.447%/+0.738%** with **25/28** wins.
Complete quality is **330/330 top-1**, max KL **0.014671**, and production
tracing observes **432 pair / 0 scalar** Q4 producers, exactly 72 more than the
prior package. The strictly serial selector-unset XTX matrix is
**965.209/1003.206/983.082 tok/s**, reaching raw HIP parity at 512 and clearing
the frozen HIP+1% prefill gates at both 1K and 4K. Evidence:
[`retained pair-only Q6-QKV/Q4-gate route`](results/2026-08-13-qwen36-27b-q6-qkv-q4-gate-pair-only-engine-retained.json).
The remaining exact Q4/Q4 operation was the final fresh-profile-selected lane.
Its source-F16 replacement passes complete quality at **320/330 top-1 / max KL
0.005645** and improves binding XTX **+0.201%/+0.081%** at M512/M1024, but
canonical W7900 M512 regresses **0.037%** with only **1/7** wins. Runtime
ownership is removed, the exact unequal-Q4 owner remains, and the residual
prefill lane is exhausted with raw-HIP parity at all shapes but the frozen
HIP+1% 512 gate still **0.928%** short. Evidence:
[`final residual rejection`](results/2026-08-13-qwen36-27b-q4-unequal-pair-source-f16-engine-rejected.json)
and [`final residual audit`](results/2026-08-13-qwen36-27b-prefill-residual-exhaustion-audit.json).
The later compact peer-GDN route supersedes that projection-only exhaustion:
it is bit-exact at the complete chain and wins all **42/42** cross-board engine
pairs. Its first fresh selector-unset XTX matrix reached
**977.397/1012.309/987.809 tok/s**. The subsequent route-shaped scratch keep
preserves all prefill gates at **974.814/1009.979/988.405 tok/s** while reducing
tracked peaks by **9.8125/21.25/107.375 MiB** at 512/1K/4K; 8K saves
**107.375 MiB** too. Evidence:
[`right-sized compact Q/K scratch`](results/2026-08-14-qwen36-27b-compact-gdn-qk-scratch-retained.json) and
[`independent compact peer-GDN XTX matrix`](results/2026-08-14-qwen36-27b-gdn-compact-peer-independent-xtx.json).

The complete ten-prompt llama-compatible natural suite selects B3:

| Mode | Transition decode | vs true AR | Draft acceptance | Whole-device peak delta |
| --- | ---: | ---: | ---: | ---: |
| True AR | **20.782 tok/s** | 1.0000x | — | — |
| B1 | **53.222 tok/s** | 2.5609x | 91.27% | — |
| B2 | **66.350 tok/s** | 3.1926x | 82.97% | — |
| B3 | **72.887 tok/s** | **3.5071x** | 77.17% | **17.183 GiB** |

All B1-B3 outputs match true AR across all ten prompts, four categories, six
train prompts, and four heldouts. B3 is **55.53% faster** than clean llama.cpp
HIP B2 but **11.06% slower** than Vulkan B4 and exceeds Vulkan's lower memory
floor by **0.509 GiB**. One independent full suite is complete. A second suite
cannot reverse the binding speed/memory failures, so the stop rule closes the
campaign without spending another expensive variance run; no XTX MTP row is
promoted to the compact root topline.

The candidate uses one T16 payload for each of all 288 rank-2 Q4 tensors and a
sole 715,161,600-byte device-visible mapped GGUF mmap for the root Q4 token
table, with no VRAM shadow. The initial 512 screen measured PM4 at **33.424
vs 32.897 tok/s (+1.601%)** across three rearmed 128-transition runs. A stricter
same-session, counterbalanced transport matrix confirms PM4 over HIP graph at
all campaign contexts: **33.494 vs 33.027 (+1.412%)** at 512, **34.445 vs
33.854 (+1.747%)** at 1K, and **31.320 vs 30.830 tok/s (+1.588%)** at 4K.
PM4 won all **15/15** paired samples with exact recorded outputs, zero native
fallbacks or unretired submissions, and clean teardown. It remains the
narrowly-scoped default for this model at private-c1 horizons of at least 128.
The mapping cuts tracked residency **16.749 -> 16.083 GiB** and
same-workload sampled peak delta **17.347 -> 16.679 GiB**; the standard
512/128 PM4 row peaks at **16.712 GiB** and full graph/session teardown returns
tracked bytes to zero. A model-scoped dense phase-liveness arena then removes
unused MoE fields and aliases mutually exclusive linear/full-attention/FFN
scratch: physical bulk scratch falls **0.589 -> 0.111 GiB**, tracked peak falls
**16.083 -> 15.605 GiB**, and sampled peak delta falls **16.712 -> 16.214
GiB** with exact eager/PM4 final logits across three resets. A model-scoped
small-weight arena then packs 481 <=16-MiB allocations into one owner, reducing
physical weight owners **850 -> 370** and sampled peak delta another **44.000
MiB** to **16.171 GiB**; the 645,120-byte tracked alignment cost leaves tracked
peak at 15.605 GiB. Storing the shared Q4T16 K256 slab output-major then turns
scalar LDS fragment traffic into vector loads, cuts the traced 288-call Q4
family **433.351 -> 421.447 ms (-2.747%)**, and improves exact 512/128 prefill
**719.232 -> 730.589 tok/s (+1.579%)** without changing decode or residency.
The model-scoped arena cutoff then widens from 16 to the first complete 80-MiB
inventory crossover: **849** immutable allocations share one owner, only the
994.6-MiB untied head remains dedicated, physical weight owners fall **370 ->
2**, and standard process peak delta falls **16.171 -> 16.095 GiB (-77.840
MiB)** with neutral exact 512/128 behavior. The subsequent model-qualified
bounded Q4/Q5/Q6 F16/rocBLAS owners, exact unequal-Q4 pair, exact record-owned planar-Q6 and pair-owned Q4 producers,
plus the natural-octet-owned Q5 producer, bounded pair-produced full-Q route,
ordered pair-only Q6-QKV/Q4-gate route, compact peer-GDN route, and its
right-sized Q/K scratch produce the current **974.814/33.522**,
**1009.979/34.530**, **988.405/31.401**, and **820.061/29.381 tok/s** matrix at
512/1K/4K/8K while keeping one persistent T16 weight layout per tensor. Tracked
peaks are **15.596/15.699/16.263/16.521 GiB**. Every shape fits and is
deterministic; 512/1K decode pass and all three frozen prefill rows clear their
HIP+1% gates. All memory rows and 4K decode remain below their frozen
cross-engine gates. This is a retained current snapshot, not complete
cross-engine closure.

BF16 K/V remains the only supported dense-27B cache route. The new native
24-query/4-KV-head INT8 split-K consumer is CPU-reference gated and traced, but
pure FP32-scale INT8 is quality-rejected: its complete 512/8 suite passes 10/11
prompts and falls to **77.78%** minimum-prompt top-1, even though 4K/16 passes.
A deterministic 9-BF16/7-INT8 layer map passes complete 512/8 and 4K/16 suites
plus bounded mixed 8K/16K/32K rows, but is not supportable yet. At 32K it saves
**0.434 GiB** live while raising tracked peak **0.448 GiB**; seven prefill-
oracle pairs project to **7 GiB** at 256K, graph admission faulted and was
reverted, and eager 4K/128 decode is **10.52%** below same-capacity BF16 graph
decode. The earlier host screen also found no reason to prefer a recent-token
BF16 tail. Production defaults are unchanged.
Evidence: [`initial temporal-tail blocker`](results/2026-08-13-qwen36-27b-int8-kv-temporal-tail-screen-blocked.json)
and [`native FP32-scale/mixed-layer diagnostic`](results/2026-08-13-qwen36-27b-int8-kv-fp32-mixed-layer-diagnostic.json).

The final live target+NextN census proves **zero duplicate-payload and zero
alternate-layout bytes** across 870 references / 866 physical ranges; mapped
embedding and untied output head are shared exactly once and all tracked
ownership closes to zero. Stability evidence adds three AR plus three natural
MTP cold process cycles, then **100 mixed 512/1K/4K reset/rearms / 400 PM4
submissions** in one resident process with exact tokens/logits, constant
post-warm ownership, zero fallback/unretired work, and fully retired graph,
executable, and context children. Deep 512/1K/4K eager oracles match fresh
serial-prefix token, FP32 hidden, every Conv/GDN state, and all live BF16 K/V
bytes at all 12 transitions. PM4 matches those eager states plus full logits and
complete `KVLiveSpans` at the common pre-execution boundary, again with zero
fallback/unretired submissions and clean retirement. Real dense B1-B3
reject/partial/full/rollback/reseed transactions, proposal-time cancellation,
and the public AR/repeated-MTP/HTTP torch-free owner-reuse lifecycle also pass;
tracked public teardown is zero and final VRAM is within 0.8 MiB of baseline.
Public dense MTP now shares public AR's bulk prompt admission and catches NextN up from shifted device-resident target hidden rows. A fixed 601-second soak completes **204 cycles / 408 requests** across all ten prompts/four categories with every AR/MTP pair exact, stable per-prompt hashes, one reused target/draft owner, zero live-byte spread, no torch, tracked close zero, **18.209-GiB** peak VRAM delta, **62/83/94 C** max edge/junction/memory temperatures, and final VRAM **+0.79 MiB**. Public verification uses correctness-first serial-exact; native verification after bulk prefill remains a documented blocker.

Evidence: [`clean comparator floors`](results/2026-08-12-qwen36-27b-xtx-clean-llamacpp-floors.json),
[`pre-single-layout blocker`](results/2026-08-12-qwen36-27b-xtx-pre-single-layout-blocked.json),
[`sole-T16 first fit`](results/2026-08-12-qwen36-27b-xtx-sole-t16-first-fit.json),
[`mapped-host/PM4 partial pass`](results/2026-08-12-qwen36-27b-xtx-mapped-host-embedding.json),
[`dense scratch liveness`](results/2026-08-12-qwen36-27b-xtx-dense-prefill-scratch-liveness.json),
[`small-weight arena`](results/2026-08-12-qwen36-27b-xtx-small-weight-arena.json),
[`Q4T16 output-major LDS`](results/2026-08-12-qwen36-27b-xtx-q4-t16-output-major-lds.json),
[`wide weight arena`](results/2026-08-12-qwen36-27b-xtx-wide-weight-arena.json),
[`complete engine AR matrix`](results/2026-08-12-qwen36-27b-xtx-engine-ar-matrix.json),
[`llama-compatible natural MTP matrix`](results/2026-08-12-qwen36-27b-xtx-llama-compatible-mtp.json),
[`correctness/runtime residency`](results/2026-08-12-qwen36-27b-xtx-correctness-residency.json),
[`cold/warm PM4 lifecycle`](results/2026-08-12-qwen36-27b-xtx-lifecycle.json),
[`public AR/MTP bulk-admission soak`](results/2026-08-12-qwen36-27b-xtx-public-ar-mtp-soak.json),
[`PM4 versus HIP graph at all contexts`](results/2026-08-12-qwen36-27b-xtx-pm4-all-contexts.json),
and [`same-commit W7900 non-regression`](results/2026-08-12-qwen36-27b-w7900-single-layout-non-regression.json).

The shared gfx1100 default also passes its original-card safeguard. Against a
same-commit W7900 rollback that reconstructs rank-2 Q4 pack8+T16 residency, the
single-layout route improves 512/1K/4K prefill by **152.61-184.82%**, decode by
**17.58-18.74%**, natural true AR by **2.61%**, and B3 by **0.24%**, while
cutting natural-suite peak delta **31.680 -> 17.183 GiB (-45.76%)**. All outputs
and acceptance ledgers remain exact and teardown returns tracked ownership to
zero.

### Radeon 8060S: Qwen3.6-35B-A3B GGUF

This is the latest clean, exact one-queue production snapshot. The artifact is a
campaign completion gate, not a claim that its final step improved every row.

| Workload | Prefill | Decode | Tracked peak | Whole-device GTT peak |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **1369.489 tok/s** | **54.330 tok/s** | 20.566 GiB | 21.000 GiB |
| 4K/128 | **1430.215 tok/s** | **54.798 tok/s** | 20.951 GiB | 21.499 GiB |
| 32K/128 | **1144.713 tok/s** | **46.405 tok/s** | 21.597 GiB | 22.152 GiB |
| 64K/128 | **936.218 tok/s** | **40.180 tok/s** | 22.336 GiB | 22.890 GiB |
| 128K/128 | — | — | — | — |

Repeated 128K remains blocked by the documented later-pass lifecycle stall; no
numeric 128K row is carried forward. Evidence:
[`SH14-C1 completion gate`](results/2026-08-06-gfx1151-gguf-sh14-c1-cumulative-completion-gate.json).

### Laguna S 2.1

| Platform / format | Workload | Prefill | Decode | Evidence |
| --- | --- | ---: | ---: | --- |
| W7900 / `UD-Q2_K_XL` | 4096 prompt, prefill only | **440.893 tok/s** | — | [`H8B production`](results/2026-08-03-gfx1100-laguna-q2-xl-scoped-activation-pack-reuse-production.json) |
| Radeon 8060S / `Q4_K_M` | 512/128 | **654.249 tok/s** | **23.221 tok/s** | [`prefill production`](results/2026-07-27-gfx1151-laguna-attention-packed-query-producer-candidate.json), [`decode production`](results/2026-08-01-gfx1151-laguna-registry-resolution-cache-retained.json) |

The W7900 Laguna decode campaign and rejected H7/H8 ladders are implementation
history, not scoreboard content; follow the production artifact, changelog, and
[`docs/LAGUNA-PARITY-STATUS.md`](../docs/LAGUNA-PARITY-STATUS.md).

## Current concurrency scoreboards

All values are aggregate generated tokens per second. Direct rows time the
resident model path; server rows include the named OpenAI serving protocol and
must not be compared as the same timing scope.

### W7900 Qwen3.6-35B-A3B GGUF `UD-Q4_K_M`

| Interface | c1 | c2 | c4 | c8 | c9 | c13 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct engine | **98.263** | **148.944** | **209.304** | **266.479** | — | — |
| OpenAI SSE | **72.169** | — | — | **158.542** | **137.001** | **129.507** |

Direct c1 uses HIP graph; the admitted c2/c4/c8 rows use the exact scoped PM4
transport. Server c9/c13 are declared grouped execution, not native widths.
All 189 server request rows and 24,192 generated IDs pass the exact gate.
Evidence: [`context-scoped C8 server refresh`](results/2026-08-08-gfx1100-context-scoped-c8-server-refresh.json).

### Maple-Preview 2-bit on Radeon 8060S

| Interface | c1 | c2 | c4 | c8 | Scope |
| --- | ---: | ---: | ---: | ---: | --- |
| Public engine generation64 | **123.131** | **165.697** | **202.038** | **214.788** | Admission, prefill, generation, reclaim |
| Fixed helper decode64 | — | **250.481** | **346.365** | **428.063** | Decode helper only; excludes public scheduling |

Evidence: [`public P4`](results/2026-08-08-gfx1151-maple-p4-long-prefill-public-batch-retained.json)
and [`D1 helper`](results/2026-08-08-gfx1151-maple-d1-batched-affine4-rowreuse-retained.json).

## Current speculative decode scoreboards

| Platform / model | Contract | True AR | MTP | MTP / AR | Status and evidence |
| --- | --- | ---: | ---: | ---: | --- |
| W7900 / Qwen3.6-27B Dense `Q4_K_M` | Current exact natural25 B3 | 20.516 | **60.875** | **2.9672x** | Current sole-T16 package snapshot; exact and faster than same-commit dual-layout control, but still below selected llama.cpp Vulkan B4. [`artifact`](results/2026-08-12-qwen36-27b-w7900-single-layout-non-regression.json) |
| W7900 / Qwen3.6-35B-A3B `UD-Q4_K_M` | `llama-compat` MTP-2 natural suite | 96.75 | **122.67** | **1.2679x** | Retained explicit opt-in; accuracy-traded versus normal AR. [`artifact`](results/2026-07-19-w7900-llama-compat-reusable-native-cycle.json) |
| Radeon 8060S / Qwen3.6-35B-A3B `UD-Q4_K_M` | `llama-compat` MTP-2 natural suite | 56.09 | **80.10** | **1.4282x** | Retained explicit opt-in; accuracy-traded versus normal AR. [`artifact`](results/2026-07-19-gfx1151-llama-compat-native-cycle-transfer.json) |

MTP ratios always use a true no-MTP AR path from the same protocol. Verifier
`off`/`B0` diagnostics are not speedup denominators. The full category suite,
heldouts, and anti-gaming rules are mandatory; see
[`docs/BENCHMARK.md`](../docs/BENCHMARK.md#anti-gaming).

## Maple-Preview retained backend comparison

These are same-model retained rows, but CUDA and HIP run on different hardware.

| Platform | Workload | Current throughput | Exactness / scope | Artifact |
| --- | --- | ---: | --- | --- |
| Radeon 8060S | Native prefill 128/320/512 | **750.854 / 741.890 / 754.458 tok/s** | 18/18 states, 90/90 positions, KL 0 | [`P4`](results/2026-08-08-gfx1151-maple-p4-long-prefill-public-batch-retained.json) |
| Radeon 8060S | c1 natural+heldout continuation | **153.201 tok/s** | 18 prompts, 1,152 timing pairs, exact state/head | [`D0`](results/2026-08-08-gfx1151-maple-d0-selector-snapshot-retained.json) |
| RTX PRO 6000 Blackwell | Native prefill 128/320/512 | **1953.820 / 1852.124 / 1917.492 tok/s** | 18/18 states, 90/90 positions, KL 0 | [`CUDA prefill`](results/2026-08-08-cuda-sm120a-maple-native-prefill-retained.json) |
| RTX PRO 6000 Blackwell | c1 natural+heldout continuation | **402.361 tok/s** | 1,152/1,152 paired wins; 1,296/1,296 positions exact | [`CUDA split-K`](results/2026-08-09-cuda-sm120a-maple-splitk-global-decode-retained.json) |

CUDA resident batching and serving are not claimed by these c1 rows.

## Current PM4 transport diagnostic

This is a transport diagnostic for the W7900 Qwen3.6-35B-A3B Q4_K_M one-step
replay path, not a separate model-throughput topline.

| Transport | Replay | Capture | Status |
| --- | ---: | ---: | --- |
| HIP graph | **10.726350 ms/token / 93.228 tok/s** | 34.993 ms | Portable oracle and fallback |
| Stateful PM4, global acquire | **10.052766 / 99.475** | 79.296 ms warm-metadata | Exact comparison path |
| Stateful PM4, local-cache acquire | **9.964358 / 100.358** | 132.858 ms cold | Scoped default when the measured replay window amortizes capture |

Evidence: [`local-cache capture/replay`](results/2026-08-08-gfx1100-pm4-setup-local-cache-clean.json)
and [`scoped default`](results/2026-08-08-gfx1100-pm4-scoped-default.json).

## Reading the tables

- Workload format is `prompt_tokens/decode_tokens`.
- Prefill, backend decode, full request wall, server wall, and component timing
  are different scopes. Compare only like with like.
- Aggregate concurrency throughput is total generated tokens divided by group
  wall. Per-request throughput is lower when multiple requests share the GPU.
- PARO, GGUF, and llama.cpp rows may use different quantization or KV formats;
  raw leaders are descriptive unless the artifact establishes a matched A/B.
- hipEngine tracked memory and whole-device VRAM/GTT are different scopes.
- A bold value identifies the reported row, not a universal claim across
  mismatched hardware, models, or protocols.

## Maintenance contract

Keep this file compact:

1. Replace the current row for a protocol tuple; do not append an optimization
   diary beneath it.
2. Put exact commands, samples, deltas, profiler data, correctness details, and
   candidate decisions in the compact JSON artifact.
3. Put the one-line old-to-new transition in [`CHANGELOG.md`](CHANGELOG.md) and
   substantial implementation decisions in a new immutable worklog entry.
4. Mention a blocked/rejected run here only when it removes a current numeric
   row or defines a user-visible limitation. Link one artifact and one rerun
   condition; keep candidate ladders out of the scoreboard.
5. Keep superseded tables in [`HISTORY.md`](HISTORY.md), artifacts, or Git
   history rather than copying them forward.
6. Update `Last updated`, then synchronize the public block:

```bash
python3 scripts/sync_benchmark_readme.py --write
python3 scripts/sync_benchmark_readme.py --check
git diff --check
```

The full evidence and artifact requirements remain authoritative in
[`docs/BENCHMARK.md`](../docs/BENCHMARK.md).
