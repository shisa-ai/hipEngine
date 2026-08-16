# hipEngine Topline Benchmarks

Last updated: **2026-08-16**

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
| Qwen3.8-27B Dense GGUF `Q4_K_M` | 512 input tokens, 128 output tokens | **399.836** | **12.210** |
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

Current campaign diagnostic: Qwen3.5-0.8B on Radeon 8060S/`gfx1151` ran the
full Vulkan-parity campaign (D08) to a blocked closure, then D08-X retained
Q8_0 cluster8 GDN, pack8-WMMA bulk, dense-BF16 WMMA, and operation-complete Q4
pack8 gate+up+SiLU. That X3 route measures **4344 -> 4944 exact-core pp512
(+13.81%)** over five paired blocks. A fresh six-block clean-HEAD external
packet now measures hipEngine / same-source llama HIP / Vulkan exact-core
pp512 at **4896 / 4848 / 5510 Q4** and **4997 / 4640 / 5704 Q8 tok/s**.
hipEngine therefore reaches **1.010x llama HIP / 0.889x Vulkan Q4** and
**1.077x / 0.876x Q8**. The Q4 core gap to Vulkan falls from **21.458 to 11.657
ms**; public prefill remains 0.867x Vulkan. hipEngine still beats llama HIP
decode, while public decode is **0.959x Vulkan Q4 / 1.047x Vulkan Q8**. All 36
children are finite/deterministic/cross-engine exact and every metric CV is
below 5%. Core Vulkan parity stays open. Evidence: [`post-X3 three-way`](results/2026-08-15-gfx1151-qwen35-08b-post-x3-current-exact-three-way.json)
and [`operation-complete Q4 prefill`](results/2026-08-15-gfx1151-qwen35-08b-pack8-dual-wmma-silu-prefill.json).
Full per-package evidence and history live in
[`benchmarks/HISTORY.md`](HISTORY.md), the D08/D08-X artifacts under
[`results/`](results/), and
[`docs/QWEN35-08B-GFX1151-VULKAN-PARITY.md`](../docs/QWEN35-08B-GFX1151-VULKAN-PARITY.md).
The post-X3 Q4 marker rerank reconciles **100.08%** of its 106.338-ms wall.
That profile's owners are dense FFN **34.007 ms**, linear-attention projections
**31.815 ms**, and GDN **23.617 ms**; normalized historical-attribution gaps
rerank linear projections first at **13.460 ms**. Its explicit 17.628-ms
Q5-QKV + Q4-gate fallback screen was byte-exact but only **1.0059x / 0.114 ms**
faster across all 18 pairs. The next GDN cluster8 wave-broadcast screen was
also exact but regressed **0.775 -> 1.195 ms (0.648x)**. Both transient routes
were removed. The next dense-FFN package retained exact WMMA down+residual:
its 12-owner leaf is **9.959 -> 9.804 ms (1.0158x)**, while five paired Q4
blocks observe **+3.09% core / +1.68% public pp512** with all guards passing.
The causal retained claim is the 0.155-ms leaf saving. Applying the same exact
rounded-residual store to the remaining 12 pack8-Q4 down owners instead loses
both paired screens at **-4.22% core / -4.80% public pp512**; that candidate is
removed and the pack8 projection+add chain remains current. Evidence:
[`post-X3 prefill rerank`](results/2026-08-15-gfx1151-qwen35-08b-post-x3-prefill-rerank.json),
[`projection rejection`](results/2026-08-15-gfx1151-qwen35-08b-q5t16-q4pack8-qkv-gate-rejected.json),
[`GDN rejection`](results/2026-08-15-gfx1151-qwen35-08b-gdn-cluster8-broadcast-rejected.json),
[`dense residual retention`](results/2026-08-15-gfx1151-qwen35-08b-dense-bf16-wmma-residual-prefill.json),
and [`pack8 residual rejection`](results/2026-08-15-gfx1151-qwen35-08b-q4-pack8-wmma-residual-rejected.json).
The next same-resident operation-complete owner shares each exact activation
tile across the 18 Q8T16 alpha/beta pairs. Its byte-exact leaf is **2.113 ->
0.422 ms (5.010x)**; Q4/Q8 paired core pp512 improves **3.64%/2.45%** and
public pp512 **5.05%/2.13%**, all with 5/5 and 3/3 wins. Evidence:
[`Q8T16 alpha/beta dual WMMA`](results/2026-08-15-gfx1151-qwen35-08b-q8t16-alpha-beta-dual-wmma-prefill.json).
The prior same-session graph replay census, whose decode route is unchanged by
X3, leaves only **0.114/0.127 ms/token Q4/Q8** outside device stages. The clean
fresh-process p16-p4096 threshold diagnostic then completes all **187** children
with finite logits and exact final IDs: Q4 current/pre-X2 is **1.764x only at
p512** and **0.997x-1.032x elsewhere**, while automatic GDN beats strict X2 at
every Q4/Q8 length. This keeps the p512-only WMMA scope and current GDN policies;
see [`2026-08-15-gfx1151-qwen35-08b-prompt-threshold-sweep.json`](results/2026-08-15-gfx1151-qwen35-08b-prompt-threshold-sweep.json).
The final natural+category-p512 cumulative packet then passes at **1794/1800
current top-1 (99.667%), max KL 0.005930**, deterministic finite state, and
**72/72 exact eager/recorded-graph trajectories**; this closes post-review
validation without changing the blocked Vulkan-parity status. The post-X6
rerun preserves the same **1794/1800** current top-1 and max KL, keeps all
**72/72** recorded-graph pairs exact, and matches candidate/rollback teacher and
state digests on **36/36** Q4 prompt/profile pairs. Evidence:
[`cumulative baseline`](results/2026-08-15-gfx1151-qwen35-08b-cumulative-semantic.json)
and [`dense residual retention`](results/2026-08-15-gfx1151-qwen35-08b-dense-bf16-wmma-residual-prefill.json).

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

### Radeon 8060S: Qwen3.8-27B Dense GGUF campaign opening

This is the clean G0 **current snapshot**, not an optimization claim. hipEngine
uses `943ec15f5`; llama.cpp HIP and Vulkan use build 10438 `9d57ce456`.
Binding shape rows use Qwen3.8-27B Q4_K_M, BF16 K/V, explicit repeated token
9707 arrays, one same-shape warmup, three measured runs, full output hashes,
and one right-sized process per engine/shape. GTT is whole-device process-run
delta on the unified-memory APU.

| Shape | hipEngine prefill | llama HIP | llama Vulkan | hipEngine AR | llama HIP | llama Vulkan | hipEngine GTT | Lower llama GTT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512/128 | **85.288** | 352.426 | 242.956 | **7.0257** | 12.1506 | 12.7629 | **33.125 GiB** | 15.785 GiB |
| 1K/128 | **84.497** | 364.443 | 247.610 | **6.9592** | 12.0645 | 12.7197 | **33.613 GiB** | 15.816 GiB |
| 4K/128 | **84.204** | 367.993 | 354.368 | **6.7144** | 11.5081 | 12.5683 | **36.046 GiB** | 16.004 GiB |

Every llama row has the same stable 129-token hash as the common-capacity
hipEngine control. hipEngine tracked ownership returns to zero after close and
peaks at 31.659/32.093/34.497 GiB, but its opening plan still carries
10,790,502,400 alternate-layout bytes. Synthetic-input llama-bench rows and the
first F16 profile are retained as diagnostics only, not mixed into this table.

Natural exact suite (ten prompts, six train/four heldout, four categories,
three runs and 720 timed transitions per mode):

| Engine/mode | Throughput | AR ratio | AR-equivalent output | GTT delta |
| --- | ---: | ---: | --- | ---: |
| hipEngine true AR | **7.10844** | 1.0000x | deterministic reference | **34.555 GiB** |
| hipEngine exact B1 | **14.79394** | 2.0812x | 30/30 | shared process |
| hipEngine exact B2 | **18.48249** | 2.6001x | 30/30 | shared process |
| hipEngine exact B3 | **19.72960** | **2.7755x** | 30/30; GPU accept = CPU | shared process |
| llama.cpp HIP AR | 12.06439 | 1.0000x | deterministic reference | 15.803 GiB |
| llama.cpp HIP B3 | 19.63473 | 1.6275x | 30/30; valid comparator | 16.358 GiB |
| llama.cpp Vulkan AR | 12.77754 | 1.0000x | nondeterministic on 2/10 prompts | 15.871 GiB |
| llama.cpp Vulkan B3 | 26.10541 | 2.0431x | invalid: 27/30; stretch rate | 16.722 GiB |

The opening exact B3 leads the binding correctness-valid HIP comparator by
0.483%, while prefill, AR, memory and the invalid Vulkan stretch rate remain
open campaign gaps. Evidence:
[`G0 baseline`](results/2026-08-15-gfx1151-qwen38-27b-p0-baseline.json) and
[`campaign plan`](../docs/QWEN38-27B-GFX1151-CAMPAIGN.md).

#### Retained clean state — G2 complete, P5 open

Clean commit `15a2ca45b` includes the shared-Q6 prefill owners and the retained
P5 decode package through fixed-H5120 norms. One right-sized process per shape,
one warmup and three measured runs produce:

| Shape | hipEngine prefill | llama HIP | llama Vulkan | hipEngine AR | llama HIP | llama Vulkan | hipEngine GTT | Lower llama GTT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512/128 | **399.836** | 352.426 | 242.956 | **12.2099** | 12.1506 | 12.7629 | **17.322 GiB** | 15.785 GiB |
| 1K/128 | **390.793** | 364.443 | 247.610 | **12.0514** | 12.0645 | 12.7197 | **17.805 GiB** | 15.816 GiB |
| 4K/128 | **384.712** | 367.993 | 354.368 | **12.2095** | 11.5081 | 12.5683 | **20.181 GiB** | 16.004 GiB |

Prefill beats clean llama.cpp HIP by **13.452%/7.230%/4.543%** and Vulkan by
**64.571%/57.826%/8.563%** at 512/1K/4K, preserving G2. Every throughput CV is
below 0.046%, all nine outputs finish at token 9707, and tracked teardown is
zero. Relative to the frozen opening snapshot, prefill is
**368.807%/362.494%/356.881%** faster and AR is
**73.788%/73.172%/81.841%** faster. Process GTT remains
**47.705%/47.030%/44.012%** below opening but still
**9.741%/12.575%/26.104%** above the lower llama row. Clean AR now beats llama
HIP by **0.488%** at 512 and **6.095%** at 4K, but remains **0.109%** behind at
1K and **2.854-5.254%** behind Vulkan, so P5 and G5 remain open. Evidence:
[`post-norm clean publication`](results/2026-08-16-gfx1151-qwen38-27b-post-norm-publication.json).

The causal same-source standard-Q6 A/B remains
**362.752/354.270/349.130 -> 398.792/391.861/384.628 tok/s
(+9.935%/+10.611%/+10.168%)**. Both actual standard-Q6 QKV weights improve
2.961-3.548x with 90/90 wins; the planar leaf improves 1.421-1.502x with 45/45
wins. All are BF16-bit exact, while short rows, narrow V, root, shape misses,
and peer backends retain exact fallbacks. Evidence:
[`clean G2 publication`](results/2026-08-16-gfx1151-qwen38-27b-prefill-publication.json),
[`shared standard-Q6 prefill`](results/2026-08-15-gfx1151-qwen38-27b-q6-standard-shared4-prefill.json),
and [`shared planar-Q6 prefill`](results/2026-08-15-gfx1151-qwen38-27b-q6-shared4-prefill.json).

P5 retains rows1 Q4T16 Q8_1x2 dp4a gate/up, the exact serial-c1 Q5T16 tile8
and Q4 split-weight owners, plus exact fixed-H5120 standalone/add norm kernels.
The latest same-source generic-norm -> fixed-norm graph rows are:

| Shape | Same-source control | Current AR | Fixed-norm delta | llama HIP | llama Vulkan |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512/128 | 12.05663 | **12.23245** | **+1.458%** | 12.1506 | 12.7629 |
| 1K/128 | 11.90223 | **12.06500** | **+1.368%** | 12.0645 | 12.7197 |
| 4K/128 | 12.05091 | **12.21721** | **+1.380%** | 11.5081 | 12.5683 |

The 128 actual norm leaves are BF16-bit exact and improve **1.23268 -> 0.35870
ms/token (3.4365x, 15/15)**. Natural true AR moves **12.28760 -> 12.45494
tok/s (+1.362%)**, with every full/train/heldout/category scope positive and
all 30 trajectories exact. Native B1 is non-regressive **18.85366 -> 18.86475
(+0.059%)** with identical trajectories, 339/393 accepted/proposed tokens, and
786 target rows. Prefill improves 0.031-0.155%; tracked peaks, process GTT,
graph nodes, and resident bytes are unchanged. The development A/B edges clean
llama HIP at all three shapes and leads it on natural AR; the post-commit clean
publication confirms the 512/4K leads but places 1K **0.109%** behind HIP.
Vulkan remains **2.52-5.25%** ahead, so P5 stays open. Evidence:
[`post-norm clean publication`](results/2026-08-16-gfx1151-qwen38-27b-post-norm-publication.json),
[`fixed-H5120 norm decode`](results/2026-08-16-gfx1151-qwen38-27b-fixed5120-norm-decode.json),
[`Q8_1x2 dp4a retain`](results/2026-08-15-gfx1151-qwen38-27b-q4-q8x2-dp4a.json),
[`Q5T16 serial-c1 tile8`](results/2026-08-15-gfx1151-qwen38-27b-q5-dense-tile8-decode.json),
and [`Q4T16 split-weight decode`](results/2026-08-15-gfx1151-qwen38-27b-q4-q8x2-split-weight-decode.json).

The clean post-norm selected-region profile reconciles **79.30459 ms/token**
of kernel work against **82.72545 ms/token** profiled host decode at unchanged
**934 launches/token**. Relative to the post-grouped checkpoint, kernel and
host wall fall **1.828%/1.771%**. Norms are now only **0.376%**; Q4 dual and
single projections own **38.698%/23.975%**, and the profiler-to-host residual
is **3.42086 ms/token**. This supersedes the prior Amdahl ranking. Evidence:
[`post-norm decode profile`](results/2026-08-16-gfx1151-qwen38-27b-post-norm-decode-profile.json).

The retained exact c1 FFN-down graph contraction folds the rounded BF16
residual into the same-resident direct Q4T16/Q6-qmicro producer stores. The
active 4K trace removes **64 launches/token (934 -> 870)** and improves
profiled host decode **82.46295 -> 82.31707 ms/token (-0.177%)** with selected
kernel wall flat within 0.005%. A counterbalanced 512/128 gate improves
**12.23017 -> 12.25928 tok/s (+0.238%)** with both candidate processes above
both controls, exact IDs/logits, unchanged peaks, and no resident/workspace
bytes. This is retained development evidence, not a clean topline refresh.
Evidence:
[`c1 down-residual graph contraction`](results/2026-08-16-gfx1151-qwen38-27b-c1-down-residual.json).

The independently promoted scalar Q5T16 GDN dual-output handoff removes the
standalone FP32-to-BF16 cast after all 48 recurrent layers. The post-contraction
trace removes another **48 launches/token (870 -> 822)** and reduces the
GDN-start to Q5-start operation span **1.36385 -> 1.20856 ms/token (-11.386%)**.
Counterbalanced 512/128 graph AR improves **12.26142 -> 12.28212 tok/s
(+0.169%)**, with both candidate processes above both controls, exact
IDs/logits, unchanged **16.753218-GiB** tracked peaks, and no new bytes. The
verifier-chain key remains excluded for the separate MTP gate. This is retained
development evidence, not a clean topline refresh. Evidence:
[`scalar GDN BF16 handoff`](results/2026-08-16-gfx1151-qwen38-27b-gdn-bf16-handoff.json).

At 4K, the gfx1151 package now selects a 24Q/4KV grouped-GQA BF16 split
producer. A 15-pair rotating-K/V leaf is BF16-bit exact and improves
**0.549226 -> 0.116819 ms (4.7015x, 15/15)**. Counterbalanced complete graph
processes improve same-source AR **11.10932 -> 12.05960 tok/s (+8.554%)** with
all 12 fixed-token IDs identical, finite logits, byte-identical
**19.529302-GiB** tracked peaks, and zero teardown. The current row beats clean
llama.cpp HIP **11.50812 tok/s by 4.792%**, but remains **4.047%** below Vulkan
**12.56830 tok/s** at that checkpoint. The later fixed-H5120 norm route raises
the current 4K development row to **12.21721 tok/s**, **6.162%** above HIP and
**2.793%** below Vulkan. This is retained development evidence, not a clean
publication topline. Evidence:
[`grouped-GQA split attention`](results/2026-08-16-gfx1151-qwen38-27b-grouped-gqa-split-attention.json).

One exact post-grouped sub-window also remains default despite being below
request-level timing resolution:

| Retained P5 owner | Actual-weight control | Retained | Speedup / wins | Projected 4K saving |
| --- | ---: | ---: | ---: | ---: |
| Q4T16 serial-c1 full-K/V K5120/N1024 col4 | 16.962 us/projection | **16.441 us/projection** | **1.03169x / 14 of 15** | **11.821 us/token / 0.0146% kernel wall** |

The shape-scoped owner is BF16-bit exact across 24 immutable weights, changes no
resident/workspace bytes, and lowers traced VGPR 96 to 56. Native rows, MTP,
peer backends, and every other Q4 singleton shape retain their previous owner;
no topline row is inferred from this verified sub-window. Evidence:
[`Q4T16 c1 col4 full-K/V`](results/2026-08-16-gfx1151-qwen38-27b-q4-single-col4-c1-decode.json).

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
