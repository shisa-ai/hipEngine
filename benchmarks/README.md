# hipEngine Topline Benchmarks

Last updated: **2026-09-01**

This file is the current benchmark scoreboard. It intentionally contains only
current user-facing results, compact protocol/status notes, and links to the
authoritative evidence. It is not an optimization journal.

## Root README performance summary

The root README exports this compact retained summary verbatim.

<!-- BEGIN TOPLINE:README_HIGHLIGHTS -->
### Radeon Pro W7900 (`gfx1100`)

| Model and format | Test | Prompt processing (tok/s) | Text generation (tok/s) |
| --- | --- | ---: | ---: |
| Qwen3.6-35B-A3B ParoQuant W4 | 512 input tokens, 128 output tokens | **2852.100** | **115.804** |
| Qwen3.6-35B-A3B GGUF `Q4_K_M` | 512 input tokens, 128 output tokens | **2763.590** | **94.603** |
| Qwen3.6-27B Dense GGUF `Q4_K_M` | 512 input tokens, 128 output tokens | **875.364** | **28.681** |
| Laguna S 2.1 GGUF `UD-Q2_K_XL` | 4,096 input tokens; prompt processing only | **440.893** | — |

#### Multiple requests (total tok/s across all active requests)

| Model and interface | 1 request | 2 requests | 4 requests | 8 requests | 9 requests | 13 requests |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B GGUF `UD-Q4_K_M` (engine) | **98.263** | **148.944** | **209.304** | **266.479** | — | — |
| Qwen3.6-35B-A3B GGUF `UD-Q4_K_M` (server) | **72.169** | — | — | **158.542** | **137.001** | **129.507** |

#### MTP
| Model and mode | Text generation | Speed compared with AR |
| --- | ---: | ---: |
| Qwen3.6-27B Dense GGUF `Q4_K_M` — Generation-2 C1/K3 D24 | **32.076 tok/s** | **1.4382x** |
| Qwen3.6-27B Dense GGUF `Q4_K_M` — Generation-2 production C2/K2 D24 automatic | **34.341 tok/s** | **1.1173x** |
| Qwen3.8-27B Dense GGUF `Q4_K_M` — Generation-2 production C2/K2 D24 automatic | **36.726 tok/s** | **1.1970x** |
| Qwen3.8-27B Dense GGUF `Q4_K_M` — Generation-2 production C2/K3 D24 explicit | **51.769 tok/s** | **1.3376x** |
| Qwen3.6-35B-A3B GGUF `UD-Q4_K_M` — Generation-2 production C2/K2 D24 automatic | **93.644 tok/s public** / **98.505 tok/s three-run** | **1.1565x** / **1.1368x** |
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
| Qwen3.6-35B-A3B GGUF `UD-Q4_K_M` (GEMV lib hoist) | sync'd eager, per-token | — | **38.9** |
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
| Qwen3.8-27B Dense GGUF `Q4_K_M` — normal-owner C1 MTP-3 automatic | **15.609 tok/s** | **1.5916x** |
| Qwen3.8-27B Dense GGUF `Q4_K_M` — production C2 MTP-3 automatic | **17.031 tok/s** | **1.1441x** |
| Qwen3.8-27B Dense GGUF `Q4_K_M` — production C1 MTP-3 c68-128 explicit | **13.088 tok/s** | **1.3998x** |
| Qwen3.6-35B-A3B GGUF `UD-Q4_K_M` — MTP-2 | **80.10 tok/s** | **1.4282x** |

### RTX PRO 6000 Blackwell (`sm_120a`)

| Model and format | Test | Prompt processing (tok/s) | Text generation (tok/s) |
| --- | --- | ---: | ---: |
| Maple-Preview 2-bit | 512-token prompt test; varied prompts for generation | **1917.492** | **402.361** |

Rows use different models and tests; compare only matching protocols. The RX 7900 XTX cross-engine rows use the same Qwen3.8 file and timing boundary. llama.cpp Vulkan MTP is speed-only because its ledger differs from Vulkan AR; hipEngine and llama.cpp HIP match their controls. MTP-2/MTP-3 use two/three draft tokens. The 35B-A3B MTP-2 path matches llama.cpp MTP on the validated suite and remains opt-in because it can differ from normal AR.
<!-- END TOPLINE:README_HIGHLIGHTS -->

## Current default notes

W7900 Qwen3.6 automatic MTP is exact-scope only: 35B MoE C1/K2 and production
C2/K2/D24, plus 27B Dense C1/K3 and production C2/K2/D24. Physical C2 requires
resident capacity 2; other keys use K0. [`Final audit`](results/2026-08-28-w7900-dual-model-physical-c2-campaign-final.json).

W7900 Qwen3.8 explicit physical C3/C4 now pads verifier groups to rows6
multiples and reuses the qualified rows6 owners. K2 improves C3 **21.549 ->
32.776 tok/s (+52.10%)** and C4 **24.314 -> 36.141 tok/s (+48.64%)**, but
remains **0.9079x/0.9196x true AR**, so automatic C3/C4 stays K0.
[`Rows6 multiples`](results/2026-08-29-w7900-qwen38-q4km-rows6-multiple-rowtiles-retained.json).
The explicit physical-C2 production owner is correctness-qualified for
standard `Q4_K_M`, BF16 KV, D24, and context <=95; longer requests fail closed
to K0. P9 selects K2/R6 at **1.1902x AR** with a **155.70 ms** 1.10x physical-
cycle budget. P10 removes seven proposal and four accept global synchronizations
per four-cycle profile while preserving 252/252 bit-exact logits; complete-suite
explicit C2 remains **1.1932x AR**. P11 passes strict controls, SSE/cancel/
overload, bounded resources, negative K0 keys, and final trace. P12 promotes the
exact cap2/C2/K2/context4-95/D24 key at **36.726 vs 30.720 tok/s (1.1970x
AR)**; every category is ≥1.1363x. C3/C4 remain K0.
[`P8 closure`](results/2026-08-29-w7900-qwen38-q4km-p8-c2-correctness-closure.json) ·
[`P9 attribution`](results/2026-08-29-w7900-qwen38-q4km-p9-cycle-attribution.json) ·
[`P10 sync wins`](results/2026-08-30-w7900-qwen38-q4km-p10-sync-wins.json) ·
[`P11 integrated`](results/2026-08-30-w7900-qwen38-q4km-p11-integrated-explicit-c2.json) ·
[`P12 promotion`](results/2026-08-30-w7900-qwen38-q4km-p12-c2-automatic-promotion.json).

W7900 standardized Qwen3.8 `Q4_K_M` C1-C8 complete-wall results (total tok/s), separated by workload:

**True AR decode**

| Engine | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hipEngine | **22.854** | **37.903** | **52.296** | **63.613** | **70.994** | **77.046** | **81.256** | **83.939** |
| llama.cpp current HIP | 21.657 | 34.649 | 30.367 | 27.748 | 36.248 | 45.343 | 51.757 | 57.687 |
| llama.cpp Laurent HIP | 21.298 | 34.151 | 30.635 | 27.850 | 36.695 | 46.091 | 52.681 | 58.840 |

**Explicit K3 MTP decode diagnostic**

| Engine | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hipEngine | **34.623** | **51.769** | **54.590** | **55.780** | **61.052** | **68.531** | **66.106** | **69.717** |
| llama.cpp current HIP | 32.553 | **41.042** | 45.324 | 49.977 | 59.644 | 72.195 | 75.354 | 94.735 |
| llama.cpp Laurent HIP | **32.733** | 40.808 | 45.947 | 51.054 | **61.013** | **74.628** | **78.281** | **101.072** |

**Prefill**

| Engine | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hipEngine | **189.074** | **289.870** | **370.318** | **405.343** | **440.200** | **457.406** | **469.752** | **473.754** |
| llama.cpp current HIP | 179.035 | 206.603 | 207.636 | 258.750 | 308.960 | 330.059 | 360.231 | 405.406 |
| llama.cpp Laurent HIP | 184.024 | 225.730 | 221.227 | 259.929 | 316.711 | 357.259 | 365.592 | 404.240 |

The current AR rows are arithmetic means from a counterbalanced two-run, full-
C1-C8 same-host D1/D24 repeat. hipEngine leads the strongest peer at every
width. C2 closes the stale 9.94% deficit at **37.903 vs 34.649 tok/s (+9.39%)**;
the slower hipEngine run still beats the faster peer run by **7.34%**, and every
category plus heldout scope leads. Decomposition finds both C2 admission
(**340.62 vs 388.05 ms**) and marginal decode (**40.25 vs 43.37 ms/step**) ahead.
All 720 hipEngine repeat rows match, all peer cells are content-exact, C2-C8
form full native groups, and both hipEngine packets drain. A clean cached-only
C2 trace records a **38.26 ms** marker wall, **23.19 ms** kernel interval union,
**15.07 ms** uncovered wall, zero memory-copy-trace operations, and device
argmax with one i32-vector readback; it is attribution only, not a throughput sample. No
runtime or kernel change is attributed to this refresh.

The explicit K3 row is a separate engine diagnostic. Its current C1/C2 cells
come from the counterbalanced exact-R8 gate: C1 is the no-op R6 control and
refreshes the accumulated current runtime at **34.623 tok/s**, while C2 omits
four inactive verifier rows and improves padded R12 **42.350→51.769 tok/s
(+22.24%)**. C1/C2 now lead the strongest peer by **5.77%/26.14%**; every C2
category plus heldout scope gains **21.18%-22.69%**, the true-AR ratio rises
**1.0909x→1.3376x**, all 240 cross-packet sequences match, and all processes
drain. Target telemetry changes only R12→R8 and the non-overlapping outer stage
sum falls **18.89%**. C5-C8 use the current one-group physical owner; C4-C8
trail our own AR. All current peer K3 cells and 78/80 Laurent cells are content-exact;
Laurent's two C8 differences are deterministic and pass the anti-repetition
guards. Automatic capacity-8 requests remain K0, distinct from the promoted
capacity-2/C2/K2 product key. Prefill uses the peer one-token protocol (10
prompts x C1-C8, content-exact).

The canonical hipEngine C1-C3 rows are arithmetic means from the counterbalanced
same-build exact fused-Q4-retile gate; C4-C8 remain from the counterbalanced full-
width same-host repeat. hipEngine is **189.074/289.870/370.318/405.343/440.200/
457.406/469.752/473.754 prompt tok/s**; current HIP is **179.035/206.603/
207.636/258.750/308.960/330.059/360.231/405.406** and Laurent HIP is
**184.024/225.730/221.227/259.929/316.711/357.259/365.592/404.240**. hipEngine
forms full native groups in 10/10 cells at every C2-C8 width and now leads the
strongest peer at every width: **+2.74%/+28.41%/+67.39%** at C1/C2/C3. The
exact retiles improve their same-build AR means by **+12.02%/+8.23%/+4.01%**;
all combined category and heldout scopes are positive in both exact arms, all
paired and repeated generated rows match, and every process drains. The formerly
published C5/C6/C8 deficits remain closed at **+38.99%/+28.03%/+16.86%**. The
earlier exact planar-Q6 FFN-down retile remains independently established by
its same-build rollback: **+6.70%/+2.45%/+2.24%** at C1/C2/C3 with every
category and heldout scope positive. An older two-run explicit-K3 packet bounded then-current K3
repeat spread at **0.01%-1.06%** through C7; it is superseded for AR rates by the
full repeat above ([`older repeat`](results/2026-08-30-w7900-q4km-c1c8-parity-refresh-repeat-pair.json)).
The K3 arm is forced: with `--mtp-request-mode automatic` on this host and model,
MTP is declined at **every** capacity-8 width, so what ships in that scope is AR
and the K3 row measures the engine, not the product. [`automatic route gating`](results/2026-08-30-w7900-q4km-automatic-mtp-route-gating.json)
[`W7900 matrix`](results/2026-08-30-w7900-qwen38-q4km-c1c8-cross-engine.json) ·
[`pre-grouping refresh + submodules (superseded for current rates/acceptance)`](results/2026-08-30-w7900-q4km-c1c8-hipengine-refresh-post-promotions.json) ·
[`prefill row`](results/2026-08-30-w7900-q4km-c1c8-hipengine-prefill-row.json) ·
[`prefill row, grouped`](results/2026-08-30-w7900-q4km-c1c8-hipengine-prefill-row-grouped.json) ·
[`grouped-prefill promotion`](results/2026-08-30-w7900-q4km-c1c8-hipengine-grouped-prefill-promotion.json) ·
[`admission/decode decomposition, post-grouping`](results/2026-08-30-w7900-q4km-c1c8-admission-decomposition-post-grouping.json) ·
[`exact planar-Q6 prefill retention`](results/2026-08-31-w7900-q4km-planar-q6-prefill-retained.json) ·
[`exact fused-Q4 prefill retiles (current C1-C3 rows)`](results/2026-08-31-w7900-q4km-fused-q4-prefill-retiles-retained.json) ·
[`exact unpadded-R8 C2/K3 retention (current K3 C1/C2 rows)`](results/2026-08-31-w7900-q4km-k3-c2-unpadded-r8-retained.json) ·
[`full C1-C8 prefill peer repeat (current C4-C8 rows)`](results/2026-08-31-w7900-q4km-c1c8-prefill-peer-repeat.json) ·
[`full C1-C8 AR peer repeat and C2 attribution (current rows)`](results/2026-08-31-w7900-q4km-c1c8-ar-peer-repeat-attribution.json) ·
[`admission/decode decomposition, pre-grouping (superseded AR arm)`](results/2026-08-30-w7900-q4km-c1c8-submodule-decomposition.json) ·
[`single-wave exact route, counterbalanced speed confirmed`](results/2026-08-30-w7900-q4km-t16-single-wave-rows-accepted.json) ·
[`single-wave shape extension, band corrected`](results/2026-08-30-w7900-q4km-t16-single-wave-shapes-accepted.json) ·
[`counterbalanced T16 correction`](results/2026-08-31-w7900-q4km-t16-single-wave-counterbalanced-band-correction.json) ·
[`current C1-C3 server attribution`](results/2026-08-31-w7900-q4km-c1c3-current-server-prefill-attribution.json) ·
[`current K3 physical-group attribution`](results/2026-08-31-w7900-q4km-current-post-grouping-k3-attribution.json) ·
[`production-sidecar small-M rejection`](results/2026-08-31-w7900-q4km-t16-production-sidecar-smallm-rejected.json) ·
[`exact row64 down-projection retention`](results/2026-08-31-w7900-q4km-t16-downproj-row64-retained.json) ·
[`runner-only packed-prefill probe`](results/2026-08-31-w7900-q4km-packed-prefill-runner-probe.json) ·
[`default-AR ready-cohort retention`](results/2026-08-31-w7900-q4km-default-ar-ready-cohort-retained.json) ·
[`initial C8 physical-group retention`](results/2026-08-31-w7900-q4km-c8-physical-group-retained.json) ·
[`final C5/C8 physical-group closure`](results/2026-08-31-w7900-q4km-c5c8-physical-group-closure.json) ·
[`P13 final audit and C6/C7 refresh`](results/2026-08-31-w7900-q4km-p13-final-audit.json) ·
[`one-group K3 C6-C8 cycle attribution`](results/2026-08-31-w7900-q4km-one-group-k3-c6c8-attribution.json).

The explicit K3 C5/C8 cells now use one production physical group instead of
serial `[4,1]`/`[4,4]` C4 groups. The final tracked-clean, same-commit C5/C8
D1/D24 default/rollback gate improves D24 **49.227→57.345 tok/s (+16.49%)** at
C5 and **56.414→61.785 (+9.52%)** at C8, reducing complete MTP wall by
**14.16%/8.69%**. Every prompt and category improves. All four packets pass;
all 520 candidate/rollback generated rows match and every process drains to zero
tracked allocations. D1 stays K0 without allocating an MTP owner. D24 telemetry
records C5 proposal/target C5/R24 and C8/R36 with `[8,5120]`, versus rollback
C1+C4/R18 and C4/R18 with `[4,5120]`. The published C5 cell moves
**47.960→57.345 (+19.57%)**; the final non-best-of C8 repeat replaces
**62.985→61.785 (-1.91% repeat drift)** and remains **+10.61%** versus the
pre-task 55.860 cell. These are retained explicit engine-path results; automatic
C5-C8 remains K0 pending separate economics/profile gates. The final audit found
that this same default route had also made the published C6/C7 split-group cells
stale. Its clean completion pair moves C6 `[4,2]→[6]`,
**50.421→66.042 tok/s (+30.98%)**, and C7 `[4,3]→[7]`,
**55.983→62.719 (+12.03%)**. Every prompt/category improves, all 260
candidate/rollback rows match, and both packets drain. Canonical C6/C7 therefore
move **49.020→66.042 (+34.73%)** and **55.225→62.719 (+13.57%)**.
A cached-only one-group C6-C8 profile localizes the residual K3 gap to the target
and GPU rather than copied accept/commit state. Target+accept+commit consumes
**88.0-88.8%** of cycle wall and kernel interval union consumes **80.0-84.1%**.
From C6→C8, **94.52%** of the **84.72 ms** cycle growth is device-busy;
**90.92%** is target-composite growth and Q4 target kernels account for
**42.80 ms (57.25%)** of target kernel-sum growth. The fused Q4 gate/up owner is
absent at C6/C7 but costs **87.61 ms per R36 C8 cycle**. All profiled cycles have
zero memory-copy operations, exact AR/MTP generated IDs, one physical group,
and clean drain. A subsequent actual-weight R36 leaf screen rejects the exact
unfused chain: it is bit-exact but **17.49% slower** (1.5247 vs 1.2977 ms) and
loses all 20 counterbalanced samples. The fused owner stays; further work must
improve that body or another measured target family, not revisit accept/commit
D2H or rejected small-M. A subsequent Q4-only mixed-R8 composition was exact
and faster in isolated actual-weight leaves, but the binding counterbalanced
C5-C8 suite rejects it: aggregate complete-wall deltas are **−0.56%/−0.23%/
−0.28%/+0.006%** at C5/C6/C7/C8. The default-off scope and launch maps were
removed rather than retained as dead runtime selection. The next measured
family, planar Q6, does survive the binding gate: exact mixed-R8/R6 partitions
improve aggregate C5/C6/C7/C8 complete throughput **57.827→58.117 (+0.50%)**,
**64.706→65.069 (+0.56%)**, **62.704→63.055 (+0.56%)**, and
**65.953→66.325 tok/s (+0.56%)**. Every category and heldout scope improves;
target-enqueue+blocking-accept operation-complete time falls **1.26–1.48%**,
all 520 cross-arm MTP rows match, physical ownership remains one group, and all
processes drain. The mixed Q6 route is now default for measured R24/R30/R36
roles; R18, shape misses, strict, and `...Q6_MIXED_TARGET_ROWTILES=0` keep the
repeated-R6 fallback. [`mixed-R8 planar-Q6 retention`](results/2026-08-31-w7900-q4km-k3-c5c8-mixed-r8-q6-chunks-retained.json) The analogous exact Q5 composition is rejected and removed: aggregate C5/C6/C7/C8 deltas are **−0.07%/−0.38%/+0.44%/+0.31%**, so its C7-C8 gains cannot satisfy the all-width rule. [`mixed-R8 Q5 rejection`](results/2026-08-31-w7900-q4km-k3-c5c8-mixed-r8-q5-chunks-rejected.json) A model-bound 131,072-row CJK-aware Q6 proposal head now reduces draft scoring without changing target authority: counterbalanced C5/C6/C7/C8 improve **58.333→60.429 (+3.59%)**, **65.202→67.974 (+4.25%)**, **63.187→65.795 (+4.13%)**, and **66.378→69.360 tok/s (+4.49%)**. Proposal wall falls 29.19–33.44%; acceptance is identical by width/category/scope, both pairs win all 80 prompt cells, all 520 cross-arm rows match, and all processes drain. Strict and `HIPENGINE_GGUF_MTP_HOT_VOCAB=0` retain the full head. [`selected CJK-128K proposal head`](results/2026-09-01-w7900-q4km-k3-c5c8-selected-cjk128k-draft-head-retained.json) The standard-Q4 R6 target family now uses an exact two-wave/16-column owner at five measured shapes: counterbalanced C5/C6/C7/C8 improve **60.418→60.962 (+0.90%)**, **67.775→68.349 (+0.85%)**, **65.661→65.973 (+0.47%)**, and **69.274→69.601 tok/s (+0.47%)**. All 16 width×category cells improve, all 520 cross-arm rows and acceptance counts match, and a marked trace records 1,560 candidate launches with combined Q4 rowtile duration **473.47→463.65 ms (−2.07%)**. Strict and `HIPENGINE_GGUF_Q4_T16_ROWTILE16_W2=0` retain the WG32 parent. [`exact two-wave Q4 R6 retention`](results/2026-09-01-w7900-q4km-k3-c5c8-q4-row6-two-wave-retained.json) The exact segmented-GDN wave reduction then improves the current C5/C6/C7/C8 rows **60.611→61.052 (+0.73%)**, **68.167→68.531 (+0.53%)**, **65.860→66.106 (+0.37%)**, and **69.551→69.717 tok/s (+0.24%)**. Every category and heldout scope is positive; all 520 cross-arm generated rows and acceptance counts match, every process drains, and the actual-shape four-token leaf improves **1.020–1.136x**. Strict, FP16-state, peer, non-physical, shape misses, and `HIPENGINE_GGUF_GDN_STATE_ROWS_WAVE_REDUCE=0` retain the local128 parent. [`exact segmented GDN wave reduction`](results/2026-09-01-w7900-q4km-k3-c5c8-segmented-gdn-wave-reduce-retained.json)

**2026-08-31 audit note:** do not use the pre-grouping refresh's 31 tok/s K3 plateau or width-dependent acceptance as current evidence; pre-#30 grouped acceptance was 0.7889 at every width, while the current one-group C8 packet records 0.7850 with the same accepted-token count as its rollback. Task #25 repaired the fixed-order T16 protocol: tracked-clean forward/reverse repeats confirm the `(5120,17408)` and `(5120,10240)` row-128 defaults, every measured losing shape, and ULP-0/finite output; `(5120,12288)` narrows from row 128 to row 112 after a five-pair repeat measured rows 120/124/128 at 0.9943x/0.9949x/0.9992x. W7900 has 96 CUs; the down projection's 107 blocks are 428 wave32s (~4.46 waves/CU), not 107 blocks against 512 CUs. A current full-suite repeat pair showed that “grouped prefill” was not a binary route fact: explicit AR queued each C2/C3 request independently and formed a full resident group only 5/20 times at each width. Task #29 now closes that race with one EngineService admission command for every compatible cohort already inside the frontend batch window, while preserving per-request handles and lifecycle. The same-build rollback/default packet moves full native groups from 2/10→10/10 at C2 and 0/10→10/10 at C3; explicit AR moves 158.868→157.774, 178.504→261.748, and 223.643→346.923 prompt tok/s at C1/C2/C3. All 120 cross-packet generated rows match and both packets finish with zero active allocations. Task #11 is now complete: an exact row64 sibling reduces the `(17408,5120)` owner's 256-row padding and owns rows 33-192, with five-pair leaf speedups of 1.009x-1.855x; row193+ keeps the parent after a sharp 0.897x crossover. The same-build full-suite pair improves C1 prompt throughput by 4.63%/4.51% in both exact arms; the stable full-group control is +0.62% at C2 and flat-positive at C3, so no canonical row is replaced. A fresh pre-#30 K3 D1/D24 diagnostic confirmed that each prompt-local accept sequence and aggregate 0.788944724 acceptance were identical at C1/C3/C5/C8. The then-current capacity-8 owner was physically capped at four requests, producing `[1]`, `[3]`, `[4,1]`, and `[4,4]`; C8 paid two stable size-4 stage sums, with accept/commit/blocking-readback at 70.9% of the named-stage sum. Task #30 has now removed that ceiling for gfx1100 production: server admission, frontier, proposal, target, accept, and cycle owners resolve C8 while strict and rollback remain C4. The final same-commit C5/C8 D1/D24 gate measures **49.227→57.345 tok/s (+16.49%)** at C5 and **56.414→61.785 (+9.52%)** at C8, with every prompt/category positive, 520/520 candidate/rollback generated rows equal, clean D1 K0 ownership, clean D24 drain, proposal C5/C8, and target rows through R24/R36. The recovery audit closes the omitted changed cells: C6 improves **50.421→66.042 (+30.98%)** and C7 **55.983→62.719 (+12.03%)**, with 260/260 generated rows equal. Automatic C5-C8 is still K0. A production-materializer sidecar sweep separately closes the leaf question: small-M is strict-exact in 35/35 Q4 role/row cells but is 2.52-14.84x slower than rowtile by HIP events and 2.51-13.53x operation-complete, so the existing rowtile owner remains default.

Strix Halo Qwen3.8 `Q4_K_M`: [strict C1/B3 automatic at cap1 or cap4 singleton](results/2026-08-27-gfx1151-qwen38-dynamic-admission-d7-closure.json) is **15.609 vs 9.807 tok/s (1.5916x)**; [production c68-128 explicit](results/2026-08-27-gfx1151-qwen38-c68-c128-production-explicit.json) remains available. Exact C2 verifier [Q6](results/2026-08-28-gfx1151-qwen38-c2-q6-verifier-rowtiles-retained.json) and [Q5](results/2026-08-28-gfx1151-qwen38-c2-q5-verifier-rowtile-retained.json), followed by [production-profile Q4 rowtiles](results/2026-08-28-gfx1151-qwen38-c2-production-q4-rowtile-retained.json), lift K3 **11.724→17.031 tok/s (+45.27%)** and **0.8170x→1.1441x true AR**. Independently qualified [C3 R6/R9/R12 rowtiles](results/2026-08-28-gfx1151-qwen38-c3-production-rowtiles-retained.json) improve C3/K3 **19.070→19.934 tok/s (+4.53%)**, but remain **0.9589x AR**; production C2/K3 is automatic only for context1-128/D24, while C3-C8 and scope misses remain K0.

The [Qwen3.8 external reproduction survey](results/2026-08-28-gfx1151-qwen38-external-reproduction-survey.json) separates source-claim reproductions from a matched standard-`Q4_K_M` C1-C8 comparison. `q38rocm` strict MTP K4 reproduces **38.85 decode tok/s** under its source protocol, but requires custom FP4 and exactly one slot. In the matched matrix, Laurent is the strongest broad alternate llama.cpp route; hipEngine leads AR at C3-C7, while its MTP route beats its own AR only at C2. Laurent adaptive DFlash2 remains rejected because cross-request state contamination produced invalid output.

`Q4_K_S` uses FP16 recurrent state with FP32 rollback. Its exact W8192 DMS
sidecar stays default-off. [`DMS`](../docs/DMS.md).

### Agentic quality (quality-only; no speed claim)

| Model | Overall | Development | Sealed heldout | Code / instruction / repository / tool | Valid calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B `UD-Q4_K_M` (reference) | 44/68 (64.71%) | 20/34 | 24/34 | 14/16 · 4/16 · 10/16 · 16/20 | 56/64 |
| Qwen3.8-27B `Q4_K_M` | **50/68 (73.53%)** | **22/34** | **28/34** | 14/16 · **12/16** · 10/16 · 14/20 | **64/64** |
| Ornith-1.5-35B-A3B `Q4_K_M` | 42/68 (61.76%) | 16/34 | 26/34 | 14/16 · 4/16 · 10/16 · 14/20 | 60/64 |

All repeat/control/ownership gates pass; failures are model-owned, so no
implementation is retained. [`Final`](results/2026-08-26-zbook-agentic-quality2-campaign-final.json).

## Where detailed evidence lives

See result artifacts, [`CHANGELOG.md`](CHANGELOG.md), and [`BENCHMARK.md`](../docs/BENCHMARK.md).

## Benchmark harness catalog

There is no single "run everything" benchmark. Different questions are answered
by different harnesses, each with a specific timing scope, numerical contract,
and shape. The table below is the map: it says what each harness measures so a
result is only compared against like-for-like rows. A ✓ marks what a harness
owns and reports; a column left blank means that axis is not measured by that
harness (not that it is zero). Always run hipEngine rows through the hermetic
thecrock wrapper for the target architecture (see `docs/BENCHMARK.md`).

Legend: **AR** = true no-MTP autoregressive decode · **MTP** = speculative
multi-token-prediction decode (with a true-AR denominator where a ratio is
reported) · **Prefill** = prompt-processing tok/s · **Decode** = generation
tok/s · **Mem** = tracked/HIP/GTT memory usage · **Conc** = per-concurrency
(c=1..8) sweep.

| Harness (`scripts/`) | What it answers | AR | MTP | Prefill | Decode | Mem | Conc | Canonical entrypoint |
| --- | --- | :-: | :-: | :-: | :-: | :-: | :-: | --- |
| `qwen35_readme_sweep.py` | Single-request prefill/decode/memory per shape (llama-bench-style), one resident session, per-shape reset | ✓ | | ✓ | ✓ | ✓ | | `--engine gguf --model <model> --backend hip_gfx1151 --workloads 512/128 1K/128 ...` |
| `qwen35_gguf_bench.py` | GGUF c=1 AR prefill/decode, fresh resident session per run, HIP-graph decode | ✓ | | ✓ | ✓ | ✓ | | `--model <model> --prompt-length 512 --decode-tokens 128` |
| `gguf_true_ar_category_bench.py` | True no-MTP AR baseline over the mtp-bench category suite (the legitimate MTP speed denominator) | ✓ | | ✓ | ✓ | | | `--model <model> --prompts benchmarks/prompts/mtpbench-code-general-ja.jsonl` |
| `gguf_mtp_category_bench.py` | MTP category matrix over budgets 1..8 with guarded objective extraction; attach a true-AR baseline for ratios | | ✓ | | ✓ | | | `--budgets 1,3,5 --objective-budget b5` |
| `gguf_mtp_long_context_gate.py` | Eager-native MTP correctness vs serial-exact teacher across context/page/budget/acceptance boundaries; optional real host-proposal AR-ID gate (no speed claim) | ✓ | ✓ | | | | | `--cycle-ends 1016-1032,4K --candidate-budgets 1,2,3 --fail-on-fail` |
| `gguf_ar_mtp_suite.py` | One-command AR-vs-MTP decode ratio over the category suite under one enforced decode config | ✓ | ✓ | | ✓ | | | `--scope partial --output <json>` |
| `specdec2_perf_bridge.py` | Current-source Generation-2 true AR vs staged SPECDEC2 plus C1 direct control; complete/decode timing, ownership stages, physical C/K, exact IDs, and ROCTX leaf mode | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | `--backend hip_gfx1151 --concurrency 1 --budgets 1,2,3 ...` then separate `--concurrency 2,4 --budgets 2 ...` |
| `qwen35_batch_retained_bench.py` | **PARO-path** compact c>N batch decode; aggregate + per-request tok/s, equality vs c1, optional MTP draft depth | ✓ | ✓ | | ✓ | ✓ | ✓ | `--batch-size 8 --decode-tokens 128` |
| `qwen35_batch_gguf_diagnostic.py` | GGUF c>N generated-token **correctness** equality vs independent c1 (no throughput claim) | ✓ | | | | | ✓ | `--rows 8 --execute` |
| `server_f1_concurrency_bench.py` | Matched gfx1151 F1 HTTP concurrency through c32; profile-aware throughput, SLOs, routes, control, and memory | ✓ | | | ✓ | ✓ | ✓ | `--engine hipengine --model <model> --concurrencies 1,2,4,8,17,32` |
| `gguf_concurrency_baseline.py` | GGUF c1 + explicit serial c2/c4 timing controls (Phase-A route baseline) | ✓ | | ✓ | ✓ | | ✓ | `--model <model> --concurrencies 1,2,4` |
| `mtp-bench.py` | llama.cpp-compatible MTP prompt-suite benchmark (server economics); can wrap hipEngine verifier economics | ✓ | ✓ | | ✓ | | | `--mode hipengine-current` |
| `exact_token_generation.py` | Direct/HTTP generated-token identity gate (correctness, not throughput) | ✓ | ✓ | | | | | `direct --model-path ...` then `http --oracle ...` |
| `benchmark_matrix.py` | Join exact-token direct/server rows into a validated matrix report | ✓ | ✓ | | | | | `build --manifest ...` |

The two rows that most closely produce the README **concurrency scoreboards**
are `qwen35_batch_retained_bench.py` (direct engine) and `server_f1_concurrency_bench.py`
(OpenAI server). The **single-request** tables come from `qwen35_readme_sweep.py`
(GGUF/PARO) and `qwen35_gguf_bench.py`. The **speculative-decode** tables come
from `gguf_ar_mtp_suite.py` / `gguf_mtp_category_bench.py` with a
`gguf_true_ar_category_bench.py` true-AR denominator.

This catalog is maintained alongside the harnesses themselves: when a harness
learns a new axis (for example MTP added to a previously AR-only bench), update
this table in the same unit.

## Evidence status

| Status | Meaning | Eligible for a current numeric table? |
| --- | --- | --- |
| **Retained** | Correctness, provenance, repetition, and protocol gates passed for the named scope. | Yes. |
| **Current snapshot** | Clean current-production measurement used to describe the shipped route, but not itself a new optimization claim. | Yes, with that label. |
| **Diagnostic** | Useful attribution or comparison with a known limitation. | No; keep it in its artifact/changelog unless it explains a current blocker. |
| **Stale / superseded** | A newer route, dependency, or evidence contract replaced it. | No. |
| **Blocked / rejected** | The protocol could not complete or the candidate failed a gate. | No numeric topline. |

A row is scoped by platform, model/quant/KV, workload, concurrency, policy, and
timing window. A newer diagnostic never replaces a retained row.

## Current Generation-2 qualification

W7900 Qwen3.6-35B-A3B `UD-Q4_K_M`, BF16 KV, p128/d8, token-budget
scheduling, and same-loaded-server c1 oracles:

| Logical concurrency | 1 | 4 | 8 | 17 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Aggregate HTTP tok/s | **27.443** | **43.337** | **46.158** | **45.797** | **44.320** |
| Exact rows | 1/1 | 4/4 | 8/8 | 17/17 | 32/32 |

The canonical W7900 packet retains physical c1/c2/c4/c8 and logical c1-c32:
all nine fixed/ragged/load/cancel/overload/recovery/soak workloads pass 210/210
correctness-accounted rows, bounded overload, complete admission/reclaim, and
zero final ownership or tracked-memory delta. Exact Qwen3.8 physical c1-c8 and
its planar-Q6 row8 kernel are also retained; detailed rows remain in the
benchmark changelog and result artifacts.

On Radeon 8060S/gfx1151, the final Qwen3.8 `Q4_K_S` package retains queue2,
exact physical c1-c8/logical c1-c32 mechanics, packed prefill, direct resident
state, Q4 row8 two-wave, and scoped Q5 col8. The 130-row width, 2,100-request
load, context/graph/prefix/pressure, and lifecycle packets pass. Product closure
remains blocked at c32: **10.590 tok/s**, **18.617 s TTFT p95**, **2.125 s ITL
p99**, **24.171 s E2E p95**, and **0/3 SLO runs**; C2 64K and heavy-load SLOs
also remain blocked. [`gfx1151 campaign final`](results/2026-08-24-gfx1151-qwen38-concurrency2-campaign-final.json).

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

Current Qwen3.5-0.8B gfx1151 remains **Vulkan parity blocked** while the exact
D08-X package is retained: the final gate is **1794/1800 top-1, max KL
0.005930**, with **72/72** graph trajectories exact. [`Campaign`](../docs/QWEN35-08B-GFX1151-VULKAN-PARITY.md).

## Current single-request scoreboards

### Radeon Pro W7900: Qwen3.6-35B-A3B

The repaired-runtime publication uses two warmups and five measured resets per
right-sized session. `Peak` is hipEngine tracked allocator high-water.

| Workload | PARO prefill | PARO decode | PARO peak | GGUF prefill | GGUF decode | GGUF peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512/128 | **2852.100** | **115.804** | **18.144 GiB** | 2763.590 | 94.603 | 21.073 GiB |
| 1K/128 | 2965.063 | **103.113** | **18.367 GiB** | **3198.957** | 99.728 | 21.133 GiB |
| 4K/128 | 2927.519 | **106.020** | **19.161 GiB** | **3177.565** | 101.917 | 21.468 GiB |
| 32K/128 | 2085.511 | **92.422** | **19.851 GiB** | **2154.871** | 89.432 | 22.060 GiB |
| 64K/128 | 1559.680 | **79.098** | **20.344 GiB** | **1600.734** | 78.021 | 22.736 GiB |
| 128K/128 | 1049.467 | 61.804 | **21.881 GiB** | **1058.075** | **63.177** | 24.088 GiB |

PARO leads short-context generation and memory; GGUF leads prefill from 1K and
128K generation. IDs, variance gates, and clean provenance pass. Evidence:
[`PARO sweep`](results/2026-08-23-w7900-current-default-hipengine-paro-packed-5run.json), [`GGUF sweep`](results/2026-08-23-w7900-current-default-hipengine-gguf-q4km-5run.json).

### Radeon Pro W7900: Qwen3.6-27B Dense GGUF

This `Q4_K_M`/BF16-KV snapshot uses one warmup and three measured resident
resets per shape with state-bound PM4 graph decode.

| Workload | Prefill | Decode | Tracked peak |
| --- | ---: | ---: | ---: |
| 512/128 | **875.364 tok/s** | **28.681 tok/s** | 15.587 GiB |
| 1K/128 | **911.658 tok/s** | **29.383 tok/s** | 15.681 GiB |
| 4K/128 | **878.721 tok/s** | **26.747 tok/s** | 16.204 GiB |

All nine IDs are stable/finite; prefill/decode CV is at most 0.733%/0.475%, and
the ten-prompt gate is exact. [`Current-default evidence`](results/2026-08-23-w7900-qwen36-27b-current-default-publication.json). Qwen3.8 details remain in the XTX tables above.

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

Explicit gfx1151 Laguna DFlash remains non-default and uses the tile1 target
verifier. The attempted tile4 transfer was trajectory-identical to tile1 but
failed the shared full-suite true-AR gate and did not improve complete E2E wall;
see the [`tile4 rejection`](results/2026-08-20-gfx1151-laguna-dflash-iq3-tile4-rejected.json).

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
| W7900 / Qwen3.6-27B Dense `Q4_K_M` | Exact/default natural25 B3 | 29.457 | **60.929** | **2.0684x** | Current clean snapshot; all ten prompts, greedy outputs, and GPU/CPU acceptance agree. The ratio replaces stale historical denominators. [`artifact`](results/2026-08-23-w7900-qwen36-27b-current-default-publication.json) |
| RX 7900 XTX / Qwen3.8-27B Dense `Q4_K_M` | Exact/default natural25 B3 | 35.287 | **62.440** | **1.7695x** | Clean idle-card correction; exact greedy and GPU/CPU acceptance, retained fusion improves matched AR 3.764% and B3 0.439% with every category non-regressive. [`artifact`](results/2026-08-15-qwen38-27b-xtx-clean-idle-performance-correction.json) |
| Radeon 8060S / Qwen3.8-27B Dense `Q4_K_M` | Exact natural25 B3 | 11.692 | **21.158** | **1.8095x** | Clean current-main direct-leaf snapshot; all ten prompts and 30 MTP comparisons are exact, GPU/CPU acceptance agrees, and cached profiling confirms the qualified scalar-C1 and native Q4 rows4/2 owners. [`artifact`](results/2026-08-26-gfx1151-qwen38-current-main-ar-mtp.json) |
| W7900 / Qwen3.6-27B Dense `Q4_K_M` | Public production/BF16 resident-C2 K2 D24, automatic | 30.736 | **34.341** | **1.1173x** | Latest-source 10/10 engaged/exact; all categories non-regressive; blocking/SSE/static-intent/cancel/drain pass. [`artifact`](results/2026-08-28-w7900-dual-model-physical-c2-campaign-final.json) |
| W7900 / Qwen3.6-35B-A3B `UD-Q4_K_M` | Public production/BF16 resident-C2 K2 D24, automatic | 80.973 | **93.644** | **1.1565x** | Latest-source 10/10 engaged and MTP self-exact; three-run ratio 1.1368x; all categories non-regressive; strict-teacher, blocking/SSE/cancel/drain pass. [`artifact`](results/2026-08-28-w7900-dual-model-physical-c2-campaign-final.json) |
| Radeon 8060S / Qwen3.8-27B Dense `Q4_K_M` | Public strict/BF16 normal-cap4 realized-C1 B3, automatic | 9.807 | **15.609** | **1.5916x** | Current-source 10/10 >1.10x; all categories positive; 78.57% acceptance; C2-C8 group at normal AR width and select pure K0. [`artifact`](results/2026-08-27-gfx1151-qwen38-dynamic-admission-d7-closure.json) |
| Radeon 8060S / Qwen3.8-27B Dense `Q4_K_M` | Public production/BF16 C1 B3, c68-128/h24, explicit | 9.350 | **13.088** | **1.3998x** | 10/10 >1.10x; all slices positive; 87.63% acceptance; numerics/blocking/SSE pass. c129+/auto K0. [`artifact`](results/2026-08-27-gfx1151-qwen38-c68-c128-production-explicit.json) |
| Radeon 8060S / Qwen3.8-27B Dense `Q4_K_M` | Production/BF16 C3 K3 D24, explicit diagnostic | 20.788 | **19.934** | **0.9589x** | Scoped R6/R9/R12 reuse improves MTP 4.53%; 10/10 exact and 1,296 numerical rows pass, but mixed/aggregate trail AR, so automatic C3 remains K0. [`artifact`](results/2026-08-28-gfx1151-qwen38-c3-production-rowtiles-retained.json) |
| W7900 / Qwen3.6-35B-A3B packed PARO W4A16+MTP BF16 | Production/default B1 fast, raw D24 | 110.830 | **115.770** | **1.0446x** | Exact `720/720`; complete 10-prompt numerical/repeat/task/state gate passes. Fast improves strict MTP 10.33% overall and every category. [`artifact`](results/2026-08-24-w7900-paro-fast-d24-3run-default.json) |
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

## Reading the tables

Workloads use `prompt_tokens/decode_tokens`. Compare only matching timing,
model/quant/KV, concurrency, and memory scopes; bold identifies the reported
row, not a universal leader.

## Maintenance contract

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
   history rather than copying them forward (`git show 6a8d38ae70b9e2c4244df10d8621db83da6c8112:benchmarks/README.md`).
6. Update `Last updated`, then synchronize the public block:

```bash
python3 scripts/sync_benchmark_readme.py --write
python3 scripts/sync_benchmark_readme.py --check
git diff --check
```

The full evidence and artifact requirements remain authoritative in
[`docs/BENCHMARK.md`](../docs/BENCHMARK.md).
