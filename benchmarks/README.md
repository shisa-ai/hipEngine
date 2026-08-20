# hipEngine Topline Benchmarks

Last updated: **2026-08-20**

This file is the current benchmark scoreboard. It intentionally contains only
current user-facing results, compact protocol/status notes, and links to the
authoritative evidence. It is not an optimization journal.

## Root README performance summary

The root README exports this compact retained summary verbatim.

<!-- BEGIN TOPLINE:README_HIGHLIGHTS -->
### Radeon Pro W7900 (`gfx1100`)

| Model and format | Test | Prompt processing (tok/s) | Text generation (tok/s) |
| --- | --- | ---: | ---: |
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

### RX 7900 XTX (`gfx1100`) — Qwen3.8-27B `Q4_K_M` prefill

| Workload | hipEngine | llama.cpp HIP | HE vs HIP | llama.cpp Vulkan | HE vs Vulkan |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 | **959.4** | 965.0 | -0.6% | 865.7 | +10.8% |
| 1K | **999.7** | 979.5 | +2.1% | 832.6 | +20.1% |
| 4K | **981.8** | 945.7 | +3.8% | 836.5 | +17.4% |

#### Dedicated-server context
| KV route | Server shape | Measured context | Peak / headroom |
| --- | --- | ---: | ---: |
| BF16 default | c1 operational | **32K** | 21.869 / 2.115 GiB |
| Pure INT8 explicit | c1 repeated natural soak | **112K** | 23.323 / 0.661 GiB |
| Pure INT8 explicit | c1 one-request physical ceiling | **126K** | 23.963 / 0.022 GiB |

#### Decode / MTP

| Metric | hipEngine | llama.cpp HIP | HE vs HIP | llama.cpp Vulkan | HE vs Vulkan |
| --- | ---: | ---: | ---: | ---: | ---: |
| AR decode 512 | **34.06** | 32.86 | +3.6% | 13.39 | 2.54x |
| AR decode 1K | **34.91** | 32.75 | +6.6% | 13.38 | 2.61x |
| AR decode 4K | **31.79** | 32.41 | -1.9% | 13.31 | 2.39x |
| MTP natural | **62.44 B3** | 44.33 B2 | +40.9% | 73.33 B2 | -14.8% |

### Strix Halo / Radeon 8060S (`gfx1151`)

| Model and format | Test | Prompt processing (tok/s) | Text generation (tok/s) |
| --- | --- | ---: | ---: |
| Qwen3.6-35B-A3B GGUF `UD-Q4_K_M` | 512 input tokens, 128 output tokens | **1369.489** | **54.330** |
| Qwen3.8-27B Dense GGUF `Q4_K_S` | 512 input tokens, 128 output tokens | **396.091** | **13.069** |
| Laguna S 2.1 GGUF `Q4_K_M` | 512 input tokens, 128 output tokens | **654.249** | **23.221** |
| Maple-Preview 2-bit | 512-token prompt test; varied prompts for generation | **754.458** | **153.201** |

#### Multiple requests

| Model and interface | 1 request | 2 requests | 4 requests | 8 requests |
| --- | ---: | ---: | ---: | ---: |
| Maple-Preview 2-bit (engine) | **123.131** | **165.697** | **202.038** | **214.788** |

#### MTP

| Model and mode | Text generation | Speed compared with AR |
| --- | ---: | ---: |
| Qwen3.8-27B Dense GGUF `Q4_K_S` — MTP-3 | **23.853 tok/s** | **1.7845x** |
| Qwen3.6-35B-A3B GGUF `UD-Q4_K_M` — MTP-2 | **80.10 tok/s** | **1.4282x** |

### RTX PRO 6000 Blackwell (`sm_120a`)

| Model and format | Test | Prompt processing (tok/s) | Text generation (tok/s) |
| --- | --- | ---: | ---: |
| Maple-Preview 2-bit | 512-token prompt test; varied prompts for generation | **1917.492** | **402.361** |

Rows use different models and tests; compare only matching protocols. The RX 7900 XTX cross-engine rows use the same Qwen3.8 file and timing boundary.
llama.cpp Vulkan MTP is speed-only because its ledger differs from Vulkan AR;
hipEngine and llama.cpp HIP match their controls. MTP-2/MTP-3 use two/three
draft tokens. The 35B-A3B MTP-2 path matches llama.cpp MTP on the validated suite and remains opt-in because it can differ from normal AR.
<!-- END TOPLINE:README_HIGHLIGHTS -->

## Current opt-in packet

On Strix Halo/gfx1151, Qwen3.8-27B `Q4_K_S` FP16 recurrent state remains an
explicit opt-in rather than the public default. A clean p512 FP32/FP16 bracket
measures aggregate c1/c4/c8 decode **12.9005/42.5856/57.2905 ->
13.0204/43.8535/59.0106 tok/s** and aggregate prefill
**377.31/307.16/246.26 -> 379.09/312.24/254.41 tok/s**, while the complete
packed numerical/isolation hard gate passes. The serving screen is also
non-regressive: c1 improves **+0.38%** and exact c8 server throughput improves
**+1.33%**. Both modes share an absolute c8 ITL-SLO failure, which remains a
serving-path issue rather than a candidate regression. The former fixed 3%
threshold is removed; scoped default promotion is tracked in the
[`candidate artifact`](results/2026-08-20-gfx1151-qwen38-27b-r2-fp16-state-repaired-production.json)
and [`serving evidence`](results/2026-08-20-gfx1151-qwen38-27b-fp16-state-serving-screen-rejected.json).

## Where detailed evidence lives

| Need | Source |
| --- | --- |
| Exact commands, revisions, model fingerprints, correctness gates, samples, profiler summaries | [`benchmarks/results/`](results/) compact JSON artifacts |
| Reverse-chronological benchmark changes | [`CHANGELOG.md`](CHANGELOG.md) |
| Superseded benchmark notebook through 2026-07-10 | [`HISTORY.md`](HISTORY.md) |
| Benchmark rules and reproduction procedures | [`docs/BENCHMARK.md`](../docs/BENCHMARK.md) |
| Execution-profile numerical calibration | [`2026-08-16 ZBook-local policy artifact`](results/2026-08-16-execution-profile-threshold-calibration.json) and [`docs/EXECUTION-PROFILES.md`](../docs/EXECUTION-PROFILES.md) |
| MTP-specific protocols and terminology | [`MTP.md`](MTP.md) and [`docs/MTP-LLAMACPP-PARITY.md`](../docs/MTP-LLAMACPP-PARITY.md) |
| Quantization-quality protocols and current tables | [`quant/README.md`](quant/README.md) |
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

## Current Qwen3.6-35B quantization quality

The current gate scores 90 full-vocabulary BF16-teacher positions across all ten
code/English/Japanese/mixed prompts. Every row uses the exact local artifacts
and identical teacher contexts; no historical or unmatched-artifact rows are
mixed into this table. This is a cross-runtime distribution gate, not
held-out-corpus PPL.

| Exact local artifact | Size / BPW | Evidence scope | Mean KL vs BF16 ↓ | Top-1 agreement ↑ | Status |
| --- | ---: | --- | ---: | ---: | --- |
| GGUF `UD-Q4_K_M` | 21.107 GiB / 5.180 | exact-artifact, ROCmFPX HIP | **0.013713** | 92.222% | Matched-runtime quality baseline |
| ROCmFP4 STRIX_LEAN | **17.739 GiB / 4.354** | exact-artifact, ROCmFPX HIP | 0.045984 | **97.778%** | Quality-traded: KL/category margin fails |
| PARO full8192 packed | 19.068 GiB / 4.680 | exact-artifact, hipEngine HIP | 0.027038 | 92.222% | Quality-traded; runtime-correct and deterministic |

ROCmFP4 is 15.96% smaller than local Q4_K_M and retains more BF16 greedy
argmaxes, but fails the paired KL/category margin. PARO is runtime-correct and
deterministic after the packed-layout repair, yet remains quality-traded versus
hipEngine Q4_K_M. See the [`quality artifact`](results/2026-08-16-zbook-qwen36-quant-quality.json)
and [`protocol`](quant/README.md).

Current package decisions are compactly separated by execution profile:

- Packed PARO retains exact SiLU+down-rotation (**1.371x leaf, 69 fewer c8/L4
  launches**) with neutral aggregate wall; unsafe math is rejected.
- ZBook strict c1 retains the exact cooperative router (**30.438 -> 33.219
  tok/s, 18/18 wins**). Physical c4/c8 retain exact Q8T16 rowtiling while c2
  remains direct.
- The combined c1/cN package is exact over **1,050/1,050 rows** and remains the
  implementation default, but is not a public `production` profile: the
  60-second server soak completed 87/120 requests and rejected 33 as overloaded.

Evidence: [`PARO boundary`](results/2026-08-16-qwen36-35b-gfx1151-rocmfpx-opp3-silu-rotate-retained.json),
[`c1 router`](results/2026-08-16-zbook-qwen36-c1-router-retained.json),
[`c4/c8 rowtile`](results/2026-08-16-gfx1151-q8t16-batch-route-retained.json),
[`package decision`](results/2026-08-16-zbook-qwen36-production-profile-cn-blocked.json), and the
[`ROCmFPX transfer report`](quant/ROCMFPX-TRANSFER.md).

Current Qwen3.5-0.8B gfx1151 status remains **Vulkan parity blocked**, while
the exact D08-X package is retained. Its clean three-way exact-core pp512 is
hipEngine / llama HIP / Vulkan **4896/4848/5510 Q4** and **4997/4640/5704 Q8
tok/s**. Retained operation-complete units include Q4 pack8 gate+up+SiLU,
dense-BF16 down+residual, and Q8T16 alpha/beta dual WMMA; the final natural and
category gate is **1794/1800 top-1, max KL 0.005930**, with all **72/72** graph
trajectories exact. Rejected projection, GDN-broadcast, and pack8-residual
candidates remain in artifacts and the changelog. Evidence:
[`three-way snapshot`](results/2026-08-15-gfx1151-qwen35-08b-post-x3-current-exact-three-way.json),
[`alpha/beta WMMA`](results/2026-08-15-gfx1151-qwen35-08b-q8t16-alpha-beta-dual-wmma-prefill.json),
[`cumulative gate`](results/2026-08-15-gfx1151-qwen35-08b-cumulative-semantic.json), and the
[`campaign`](../docs/QWEN35-08B-GFX1151-VULKAN-PARITY.md).

## Current single-request scoreboards

### Radeon Pro W7900: Qwen3.6-35B-A3B

The current valid six-shape publication is GGUF. `Peak` is hipEngine tracked
allocator high-water.

| Workload | GGUF prefill | GGUF decode | GGUF peak |
| --- | ---: | ---: | ---: |
| 512/128 | **2716.648** | **92.833** | 21.228 GiB |
| 1K/128 | **3052.541** | **98.148** | 21.295 GiB |
| 4K/128 | **2953.101** | **100.522** | 21.670 GiB |
| 32K/128 | **2078.038** | **88.240** | 22.234 GiB |
| 64K/128 | **1559.878** | **76.691** | 22.879 GiB |
| 128K/128 | **1037.378** | **62.669** | 24.168 GiB |

The former PARO speed row is withdrawn: its sweep predates the grouped-V-head
runtime fix and therefore followed the wrong generated trajectory. It remains a
[`pre-fix diagnostic`](results/2026-07-12-w7900-v030-8116c453-hipengine-paro-packed-5run.json),
not current performance evidence; a fresh repaired-runtime sweep is required.
GGUF evidence: [`final optimization sweep`](results/2026-07-16-gfx1100-gguf-final-optimization-sweep.json).
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

### RX 7900 XTX: Qwen3.8-27B Dense GGUF cross-engine comparison

#### Prefill

| Workload | hipEngine | llama.cpp HIP | HE vs HIP | llama.cpp Vulkan | HE vs Vulkan |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 | **959.4** | 965.0 | -0.6% | 865.7 | +10.8% |
| 1K | **999.7** | 979.5 | +2.1% | 832.6 | +20.1% |
| 4K | **981.8** | 945.7 | +3.8% | 836.5 | +17.4% |

#### Decode / MTP

| Metric | hipEngine | llama.cpp HIP | HE vs HIP | llama.cpp Vulkan | HE vs Vulkan |
| --- | ---: | ---: | ---: | ---: | ---: |
| AR decode 512 | **34.06** | 32.86 | +3.6% | 13.39 | 2.54x |
| AR decode 1K | **34.91** | 32.75 | +6.6% | 13.38 | 2.61x |
| AR decode 4K | **31.79** | 32.41 | -1.9% | 13.31 | 2.39x |
| MTP natural | **62.44 B3** | 44.33 B2 | +40.9% | 73.33 B2 | -14.8% |

Fixed-shape rows use token 9707 and 128 timed transitions. Prefill is a
three-run hipEngine median versus llama-bench's five-sample mean. MTP uses all
ten category prompts plus the heldout split; B1-B5 were swept independently
for each llama.cpp backend. hipEngine B3 is exact, and llama.cpp HIP B2 matches
its AR output ledger. llama.cpp Vulkan B2 is a labeled speed-only comparator
because its output ledger differs from Vulkan AR. Evidence:
[`Qwen3.8 cross-engine comparison`](results/2026-08-15-qwen38-27b-xtx-hip-vulkan-comparison.json) and
[`clean-idle hipEngine correction`](results/2026-08-15-qwen38-27b-xtx-clean-idle-performance-correction.json).

The real single-request BF16 server roof is now bracketed separately. The
operational **32K** row peaks at **21.869 GiB** with **2.115 GiB** free. A
pre-sized contiguous KV pool reaches an observed **52K** ceiling
(53,248 total tokens) at **23.959 GiB**, leaving only **0.025 GiB**; **53K**
(54,272 tokens) and all larger controls through 64K abort out of resources.
The 52K row is a physical-roof diagnostic, not a reliable serving setting, so
32K remains recommended. Evidence: [`Qwen3.8 BF16 context roof`](results/2026-08-15-qwen38-27b-xtx-bf16-context-roof.json).

The pure FP32-scale INT8 route now has a bounded exact-prefill owner. Reusing
one BF16 K/V oracle pair instead of 16 lowers 32K-capacity tracked peak
**18.943 -> 17.330 GiB**, which is also **0.590 GiB below BF16**. On matched
4K/128 runs, prefill is within **-0.050%** of BF16 graph with overlapping
sample ranges, while eager INT8 decode is **6.50% faster**. Complete 512/8 and
4K/16 quality pass, and the bounded W7900 `mixed_v1` gate now reaches
**129,024 prompt tokens / 16 teacher-forced steps** with mean/max KL
**0.0000104/0.0001354**, **100% top-1**, finite logits, all 16 full-attention
layers INT8, and zero BF16 mirror bytes. This is a bounded one-prompt long row,
not a complete 129K category suite.

Guard-free page-aligned XTX probes close the actual c1 boundary. **129,024 total
tokens (126K, 504 pages)** completes one exact natural request at
**23.962624 GiB** peak with **0.021751 GiB** sampled headroom and clean
ownership. The next page, **129,280 tokens**, stalls raw warmup beyond 600
seconds; **130,048 tokens (127K)** returns startup HIP OOM. Separately, four
different natural **112K** requests pass at **23.322876/0.661499 GiB**
peak/headroom, so 112K remains the strongest repeated server evidence while
126K is the measured one-request physical ceiling. No arbitrary reserve
reclassifies a completed row. Long pure INT8 still uses the unverified gate,
graph capture rejects it, exact natural B3 is only **0.6423x** true AR, and BF16
remains supported/default.

Do not transfer the c1 ceiling to concurrent serving. The native-packed
short-context INT8 route retains a BF16 mirror and is not a compact memory route;
the later C1 compact route is serial physical-c1 only. This pre-C1 XTX table
reports observed mirrored-route execution and actual failures:

| Offered HTTP clients | Physical residency | Highest observed working row | Strongest repetition at/near frontier | First actual failure observed | Peak / headroom at highest pass |
| ---: | ---: | ---: | --- | --- | ---: |
| 1 | 1 | **126K/request** | 112K: 4/4 different natural requests | **126.25K:** startup stall; **127K:** startup HIP OOM | 23.963 / 0.022 GiB |
| 2 | 2 | **6K/request** | 5.5K: 8/8 multi-turn; 4.5K: 16/16 | **8K:** 0/2, HTTP 500 HIP OOM; 8.25K also has an unsupported-route startup failure | 23.972 / 0.012 GiB |
| 4 | 4 | **1.5K/request** | 16/16 multi-turn | **None tested** | 23.821 / 0.163 GiB |
| 8 | queue groups capped at physical c4 | **2K/request** | 1.5K: 8/8 shape-aware; 1.25K: 32/32 multi-turn | **None tested** | 23.974 / 0.011 GiB |

The c2 interval between the 6K pass and 8K OOM is untested. The offered-c8 2K
pass covers one burst and predates authoritative response-shape capture; it is
still an execution pass, not a failure. No exact XTX c4 or c8 failure ceiling
was measured.

The same route on W7900 reaches the model-native **256K c1** limit with one
exact request at **29.441/15.543 GiB** peak/headroom. More VRAM removes the
short mirrored-route OOM but not its software boundary or scheduler limitation:

| Offered HTTP clients | Highest pass | Observed physical groups | Requests | First failure | Peak / headroom |
| ---: | ---: | --- | ---: | --- | ---: |
| 1 | **256K/request** | c1 | 1/1 | None; model context limit reached | 29.441 / 15.543 GiB |
| 2 | **8K/request** | c2 | 2/2 | 8.25K resident-prepare `NotImplementedError` | 25.658 / 19.326 GiB |
| 4 | **8K/request** | c2 + c2 | 4/4 | 8.25K resident-prepare `NotImplementedError` | 30.684 / 14.301 GiB |
| 8 | **8K/request** | c1 + c3 + c4 | 8/8 | 8.25K resident-prepare `NotImplementedError` | 30.707 / 14.277 GiB |

This capacity soak used non-streaming blocking requests. Its c2/c4/c8 shapes
therefore describe the compatible-sampling HTTP coalescer, not live resident
membership; offered width does not equal one submitted physical width. The raw
`/ready` field also reported `continuous_decode=false` because that endpoint
used a hardcoded default rather than the loaded engine capability.

A separate W7900 controlled-SSE gate at 512 prompt + 24 decode tokens qualifies
the short mirrored route's resident lifecycle. With c1 already observed in
decode, three more requests joined; sampled occupancy moved
`0 -> 1 -> 4 -> 3 -> 2 -> 1 -> 0`, all four rows exactly matched independent c1
token oracles, admitted/reclaimed deltas were `4/4`, and final
active/occupied/refcounted ownership was zero. Cancellation, elastic-pool and
overload gates remain open, and this does **not** turn the mirrored <=8K route
into compact INT8 KV.

Evidence:
[`initial INT8 frontier`](results/2026-08-15-qwen38-27b-int8-kv-quality-frontier-runtime-blocked.json),
[`bounded INT8 serving qualification`](results/2026-08-15-qwen38-27b-bounded-int8-kv-serving-qualification.json),
[`dedicated-XTX context soak`](results/2026-08-15-qwen38-27b-dedicated-xtx-context-soak.json), and
[`actual context, W7900 quality, and concurrency frontier`](results/2026-08-16-qwen38-27b-actual-context-quality-w7900.json).

Runtime admission now binds this explicit route to the artifact's full SHA-256,
backend/target, weight quant, KV layout, and scale contract. The distinct gfx1151
file retains its rejected decision, and unknown/mismatched combinations fall
back to BF16 unless an explicit non-promotable diagnostic override is set. Server
readiness and capability manifests expose the evidence and effective storage;
this provenance hardening changes no benchmark metric.

The RX 7900 XTX compact correctness milestone now admits logical **c2/c4** for
the exact gfx1100 artifact with no persistent BF16 payload or mirror. Varied
p512/d24 warmup, measured, and live rows are all exact versus fresh c1 oracles;
a shifted request is byte-exact for full-vocabulary logits, 48 Conv/GDN state
pairs, and 16 INT8 K/V plus FP32-scale planes. Staggered p512/d128 SSE reaches
c4, disconnects one member, keeps all three survivors exact, records
admitted/reclaimed/cancelled **4/4/1**, and drains ownership to zero. Every c>N
model step still executes as serial physical c1, so this is **not** a native-c>N
or throughput result. Evidence:
[`compact serial c4 qualification`](results/2026-08-16-qwen38-27b-int8-kv-compact-serial-c4.json).

### Radeon 8060S: Qwen3.8-27B Dense GGUF retained campaign state

Qwen3.8 uses `Q4_K_S` with BF16 K/V. The campaign is closed at merged commit
`20e5106da`; the Q5 source-F16 prefill retention (2026-08-17) raises 512/1K/4K
prefill on gfx1151 via the byte-identical K_M-derived Q5T16 recurrent-output
route (counterbalanced +4.51%/+3.02% at 512/1K, +2.95% at 4K with a
capacity-conditional scratch cap that keeps 8K+ memory flat). Prefill and true
AR beat both clean llama backends at every working shape, exact native B3 beats
the correctness-valid llama HIP row, and process GTT stays below the lower
valid llama row at 512/1K/8K+ (4K peak grows a fixed +2.30 GiB to enable the
4K source-F16 win).

| Shape | Clean prefill | Clean AR | Retained process GTT | Lower valid llama GTT |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **396.091** | **13.069** | **15.275 GiB** | 15.785 GiB |
| 1K/128 | **387.648** | **12.894** | **15.710 GiB** | 15.816 GiB |
| 4K/128 | **380.305** | **13.038** | **17.863 GiB** | 16.004 GiB |

Exact native B3 is **23.85263 tok/s / 1.7845x AR** with all ten prompt
trajectories and GPU/CPU acceptance decisions exact; retained process GTT is
**15.899 GiB** versus valid llama HIP's **16.358 GiB**. Natural true AR is
**13.36641 tok/s** versus same-file llama Q4_K_S HIP/Vulkan at
**5.53853/7.51888 tok/s**. Rejected aliases and direct file mapping remain
recorded—not discarded—in the linked evidence.
Evidence: [`clean Q4_K_S`](results/2026-08-16-gfx1151-qwen38-27b-q4ks-clean-publication.json),
[`exact B3`](results/2026-08-17-gfx1151-qwen38-27b-q4ks-exact-native-b3.json),
[`memory package`](results/2026-08-17-gfx1151-qwen38-27b-q4ks-memory-parity-retained.json),
[`G6 closure`](results/2026-08-17-gfx1151-qwen38-27b-q4ks-g6-closure.json), and the
[`campaign plan`](../docs/QWEN38-27B-GFX1151-CAMPAIGN.md).

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
| W7900 / Qwen3.6-27B Dense `Q4_K_M` | Exact/default natural25 B3 | 22.926 | **61.147** | **2.6671x** | Retained exact natural25 control; all greedy outputs and GPU/CPU acceptance agree. [`artifact`](results/2026-08-07-qwen36-27b-latest-vulkan-parity-exhaustion-audit.json) |
| RX 7900 XTX / Qwen3.8-27B Dense `Q4_K_M` | Exact/default natural25 B3 | 35.287 | **62.440** | **1.7695x** | Clean idle-card correction; exact greedy and GPU/CPU acceptance, retained fusion improves matched AR 3.764% and B3 0.439% with every category non-regressive. [`artifact`](results/2026-08-15-qwen38-27b-xtx-clean-idle-performance-correction.json) |
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
