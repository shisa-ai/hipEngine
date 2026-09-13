# hipEngine Topline Benchmarks

Last updated: **2026-09-13**

Surya OCR 2 on Strix Halo gfx1151, hipEngine HIP lane vs transformers fp32 on
the same host iGPU, over 12 pages covering layout-JSON and markup output,
Japanese and mixed script, dense small text, a ruled table, a blank page, a
degraded scan, and long block-heavy pages: decode 54.6-57.0 tok/s vs
24.6-28.1, a median 2.19x faster (2.03x-2.23x across every page), and
end-to-end 0.20-8.85 s vs 0.29-14.38 s. Every row passes its declared gate and
each decode stage repeats identically. A lane is reported PASS only when a
reference is present, the output matches it, it terminated the way the
reference did, the isolated decode repeats agree, the timed decode stage and the
end-to-end run agree on both ids and termination, and the declared output format
holds; `gate_failures` names every condition that did not. Eleven pages gate on
the full greedy id chain. The degraded scan gates on the layout skeleton — box
count, labels, reading order and per-box `count` — because under this suite's
ad-hoc prompt both lanes free-run into the same rigid 10-box template whose bbox
digits track neither the page nor each other: the drawn line widths are 470-554
of 1000, torch reports 474-566, and hipEngine a flat 564, so the drift is
recorded as `bbox_max_delta` (90 of 1000) rather than gated. Every lane times
the same region — page image plus prompt in, greedy token ids out — with
checkpoint load and runner construction reported separately as `init_s`. Decode
is weight-bandwidth-bound: one token streams 2342 MB of fp32 weights at 136-143
GB/s, and 88% of decode kernel time is the projection GEMMs.
[Suite run](results/2026-09-13-gfx1151-surya-suite-recut.json),
[rect re-qualification](results/2026-09-12-gfx1151-surya-rect-reference.json),
[phase-attributed profile](results/2026-09-12-gfx1151-surya-phase-attributed-profile.json).
Each lane is measured in its own process and the two runs merged: running both
lanes in one process is not reliable on this suite — the same 12 cases measured
hipEngine-only pass every gate, while a combined run returns the last three
hipEngine rows degenerate from the first token with every stage time unchanged.
An earlier run of this suite recorded the torch lane 13-26% lower on every page;
that artifact is superseded and the 2.25x ratio it implied is not comparable
with this one.
Earlier Surya rows here reported decode trailing torch at 55.5 vs 68.2 tok/s;
that comparison came from a harness defect in which the torch prefill stage was
timed before its synchronization and decode was derived by subtracting two
differently-measured runs. Decode is now timed directly.

Full-page transcription is gated against the text drawn on each page, not only
against a captured oracle. Using the checkpoint's real full-page HTML prompt,
eight documents (Japanese, mixed script, dense small text, a ruled table, a
blank page, a degraded scan, a 25-line block-heavy page, and a 300-DPI A4 page)
all reach a natural EOS with line recall 1.000, line exact rate 1.000, character
error rate 0.0000, and zero reading-order violations; the ruled table reads as
4x8 with 32/32 cells and a matching header, and a deliberately starved budget
is reported as truncation with omissions rather than passing a prefix. The A4
page is the page-scale case at 2480x3508 (220x156 patch grid, 8580 image
tokens): 2108 tokens to a natural EOS, vision 47.7 s, prefill 8.0 s, decode
58.4 s (27.7 ms/token, 36.1 tok/s), 114.3 s end to end; its ground truth is
paragraph-level because its body is wrapped prose. One caveat is recorded: on
the degraded scan the HIP lane differs from torch fp32 on 21 of 616 ids, all
bbox coordinate digits, with identical labels and text and a worst coordinate
delta of 4 of 1000 — and torch bf16 differs from torch fp32 on 15 ids there and
changes the decoded text, so the drift is a property of that page's coordinates
rather than of the HIP route.
[Transcription acceptance](results/2026-09-12-gfx1151-surya-transcription-acceptance.json).

Surya OCR 2 is also reachable through hipEngine's OpenAI-compatible server.
`scripts/surya_http_e2e.py` serves the A4 page over `/v1/chat/completions` and
asserts equality with the direct-call row above: 2108 completion tokens, 8701
prompt tokens (121 text plus 8580 image), 13 blocks, 12/12 units, CER 0.0000,
32/32 table cells, `stop`, 119.9 s wall against 114.3 s direct. Vision bounds
and media form are engine declarations, so a page-scale model is admitted
without editing the server.
[HTTP serving e2e](results/2026-09-12-gfx1151-surya-http-serving-e2e.json).

This file is the current benchmark scoreboard. It intentionally contains only
current user-facing results, compact protocol/status notes, and links to the
authoritative evidence. It is not an optimization journal.
Superseded campaigns, incremental optimization deltas, and the long-form per-campaign narratives live in [`HISTORY.md`](HISTORY.md).

## Root README performance summary

The root README exports this compact retained summary verbatim.

<!-- BEGIN TOPLINE:README_HIGHLIGHTS -->
Every number below is measured on the named hardware and links to a
reproducible artifact. **Prompt processing** is how fast hipEngine reads your
input; **text generation** is how fast it writes new tokens. **With MTP** is
speculative decoding, which is enabled only where it is qualified for that
model and shape. Rows use different models and protocols — compare within a
row, not across them.

### Performance

#### Radeon Pro W7900 — 48 GB (`gfx1100`)

| Model | Quant | Prompt processing | Text generation | With MTP | Max context |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | ParoQuant W4 | **2852.1** | **115.8** | **115.8** | — |
| Qwen3.6-35B-A3B | GGUF `Q4_K_M` | **2763.6** | **94.6** | 122.7 (opt-in) | — |
| Qwen3.6-27B Dense | GGUF `Q4_K_M` | **875.4** | **28.7** | **32.1** | — |
| Qwen3.8-27B Dense | GGUF `Q4_K_M` | **678.8** | **29.6** | — | — |
| Laguna S 2.1 | GGUF `UD-Q2_K_XL` | **440.9** (4K) | — | — | — |

#### Strix Halo / Radeon 8060S — 120 GB (`gfx1151`)

| Model | Quant | Prompt processing | Text generation | With MTP | Max context |
| --- | --- | ---: | ---: | ---: | ---: |
| Maple-Preview | 2-bit | **754.5** | **153.2** | — | — |
| Qwen3.6-35B-A3B | GGUF `UD-Q4_K_M` | **1369.5** | **54.3** | 80.1 (opt-in) | — |
| Laguna S 2.1 | GGUF `Q4_K_M` | **654.2** | **23.2** | — | — |
| Qwen3.8-27B Dense | GGUF `Q4_K_S` | **396.1** | **13.1** | **23.9** | — |
| Qwen3.8-27B Dense | GGUF `Q4_K_M` | — | — | **15.6** | — |

**Time-series forecasting (TimesFM 2.5 200M).** hipEngine decodes batch=8,
context 8192, horizon 512 forecasts in **0.082 s** on this GPU — 8.6x the
official torch reference on the same hardware and 111x the NumPy CPU
reference. The FP16 production path is within 0.86% max error of the FP32
oracle (gate: 2%); a strict FP32 parity path is one flag away.

#### NVIDIA RTX PRO 6000 Blackwell — 96 GB (`sm_120a`)

| Model | Quant | Prompt processing | Text generation | With MTP | Max context |
| --- | --- | ---: | ---: | ---: | ---: |
| Maple-Preview | 2-bit | **1917.5** | **402.4** | — | — |

### Serving several requests at once

hipEngine is very strong at multi-concurrency vs llama.cpp (or even vLLM).
Aggregate tokens per second across all active requests, Qwen3.8-27B `Q4_K_M`
on the W7900 under one server protocol; the peers use F16 KV where hipEngine
uses BF16.

| Requests | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hipEngine | **23.6** | **39.1** | **53.1** | **63.9** | **72.8** | **79.5** | **83.2** | **85.9** |
| llama.cpp HIP | 21.0 | 34.4 | 30.6 | 27.7 | 36.7 | 46.4 | 52.1 | 58.4 |
| hipEngine advantage | +12% | +14% | +74% | +130% | +99% | +71% | +60% | **+47%** |

Direct engine route on the same card and model, 512-token prompts and 128
generated tokens per request, showing what each added request costs in memory:

| Requests | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Text generation (total) | 29.6 | 54.0 | 75.2 | 92.3 | 105.9 | 117.6 | 123.9 | **131.3** |
| Prompt processing (total) | **678.8** | 368.9 | 362.6 | 380.0 | 378.3 | 403.6 | 385.3 | 376.6 |
| Peak memory (GiB) | 19.4 | 20.3 | 21.1 | 22.0 | 22.8 | 23.7 | 24.5 | 25.4 |

Eight concurrent requests need about 25 GiB, so this shape wants a 32 GB or
larger card. What the same model tolerates on a 24 GB card is not qualified yet.

On Strix Halo, Maple-Preview 2-bit scales to **214.788** tok/s across eight
requests (123.131 at one, 165.697 at two, 202.038 at four). Where speculative
decoding runs automatically in production it is scoped to a qualified shape:
Qwen3.6-35B-A3B GGUF reaches **93.644 tok/s public** — 1.1565x its own AR — at
two concurrent requests on the W7900.
<!-- END TOPLINE:README_HIGHLIGHTS -->

## Current default notes

W7900 automatic MTP is deliberately narrow. Qwen3.6 enables only its qualified
C1 and resident-capacity-2 C2 keys; all other scopes use K0. Qwen3.8 retains
production/BF16 C2/K2 at resident capacity 2, context 4-95, and D24. Every
other scope miss remains K0. Evidence links are in the model sections below.

**Qwen3.8-27B `Q4_K_M` no longer uses speculative decoding on this backend.** A
full depth sweep on 2026-09-06 measured 20 cells across C2-C8 and K1-K3, each
against an AR arm in the same process: every cell ran below its own AR, the
closest being C8/K3 at 0.9902x. Both Qwen3.8 rows are therefore no longer
automatic and the route selects AR at every width; explicit opt-in still works.
Earlier published C2/K2 and C8/K3 speedups are withdrawn
([artifact](results/2026-09-06-gfx1100-qwen38-mtp-ck-matrix.json)).

Strix Halo `Q4_K_M`: strict C1/K3 automatic at **18.191 tok/s (1.6445x AR)**; production explicit/K0. Production C8/K3 is **52.103 vs 52.025 AR tok/s**. Detailed gfx1151 evidence remains in result artifacts.

TimesFM 2.5 200M GPU decode (batch 8, context 8192, horizon 512): **0.082 s**
with the double correctness gate described above. Attention runs as fused
WMMA flash kernels (long-prefill and split-kv short-query variants) and the
FP16 GEMMs use shape-keyed rocBLAS solution autotuning
([artifact](results/gfx1151-timesfm-quadtile-flash-2026-09-09.json)).

TimesFM 3.0 500M GPU decode (batch 8, 3 variates, context 8192, horizon 512,
one non-autoregressive pass): **0.319 s median, 4.17x the torch fp32 reference
on the same GPU**. The same double gate on three oracle fixtures (multivariate,
unaligned/univariate, covariate-mask + 640-horizon); the fp16 production path
fuses the variate-attention QK norms in-kernel with a strict unfused fp32
fallback. Torch comparison protocol in the artifact
([artifact](results/gfx1151-timesfm3-varnorm-fusion-2026-09-10.json)).

EVIE-4.5B multimodal retrieval encode (8 pages 448x336 + 8 queries): the
batched segment-isolated runtime (encode_documents/encode_queries, ragged
no-padding) runs the fp16 production path in **1.13 s** (doc 1.02 s, query
0.11 s; sequential path 1.81 s). Under the matched protocol (identical
real-processor inputs, preprocessing outside both timers, full 8x8 scoring;
reproduce with scripts/evie_matched_protocol_{torch,hip}.py): **hip fp16
1.17 s vs torch bf16 1.29 s — hipEngine leads 1.10x** (doc 1.04x, query
1.61x). Retrieval quality vs the torch fp32 teacher: hip fp16 preserves

### W7900 Qwen3.8 `Q4_K_M` C1-C8

These are aggregate tokens per second across all active requests under the
standardized complete-wall server protocol.

**True AR decode**

| Engine | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hipEngine | **23.636** | **39.064** | **53.141** | **63.889** | **72.821** | **79.508** | **83.197** | **85.891** |
| llama.cpp current HIP | 20.997 | 34.361 | 30.595 | 27.737 | 36.662 | 46.351 | 52.132 | 58.429 |
| llama.cpp Laurent HIP | 20.913 | 35.273 | 31.042 | 27.852 | 37.031 | 47.309 | 53.235 | 59.378 |

**Explicit K3 MTP decode diagnostic**

| Engine | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hipEngine | **36.872** | **57.700** | **65.922** | **68.106** | **70.541** | **80.714** | **85.263** | 94.080 |
| llama.cpp current HIP | 30.658 | 39.435 | 45.065 | 48.014 | 59.269 | 68.046 | 73.500 | 92.345 |
| llama.cpp Laurent HIP | 31.446 | 39.172 | 45.457 | 45.946 | 60.427 | 74.465 | 77.420 | **95.830** |
| hipEngine K3 / published AR | 1.5599x | 1.4771x | 1.2405x | 1.0660x | 0.9687x | 1.0152x | 1.0248x | 1.0953x |

**Prefill**

| Engine | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hipEngine | 188.124 | **290.890** | **366.704** | **408.710** | **439.451** | **457.617** | **473.896** | **475.561** |
| llama.cpp current HIP | **192.834** | 232.181 | 246.710 | 268.236 | 313.218 | 351.989 | 362.168 | 408.393 |
| llama.cpp Laurent HIP | 192.229 | 232.667 | 247.092 | 268.799 | 317.570 | 351.067 | 367.582 | 395.080 |

The ten-prompt suite includes four heldouts and uses raw greedy sampling, a
20 ms batch window, capacity 8, and a 1,024-token session limit. Prefill is D1;
AR and K3 are D24. Every row is a counterbalanced two-run mean from the
[2026-09-04 P8 final-closure recapture](results/2026-09-04-w7900-q4km-k3-c8-p8-final-closure-matrix.json)
on one physical host. Its binding two-order D24 suite puts the C8 candidate at
**95.240 tok/s mean**: above the published (94.735) and fresh (92.345)
current-llama.cpp exact-peer rows, below the published (101.072) and fresh
(95.830) Laurent strongest-peer rows. hipEngine uses BF16 KV; the peers use
F16 KV. A later gate promoted a capacity-8 C8/K3 key, but the 2026-09-06 depth
sweep withdrew it: MTP measured below AR at every width.
The direct packed-AR decode route uses the singleton-indexed GDN recurrence
(2026-09-05): 512/128 graph decode improves **c2 +7.40%, c4 +6.06%, native C8
+5.91%**, exact ([artifact](results/2026-09-05-gfx1100-gdn-singleton-retained.json)).
[`C8 automatic promotion`](results/2026-09-05-w7900-q4km-k3-c8-automatic-promotion.json);
[`dedicated campaign`](../docs/QWEN38-GFX1100-C8-K3-CAMPAIGN.md).

On Strix Halo, the current automatic Qwen3.8 `Q4_K_M` route is strict C1/K3;
production C1-C8 remains explicit/K0 as summarized above. `Q4_K_S` uses FP16
recurrent state with FP32 rollback; its exact W8192 DMS sidecar remains
default-off.

### W7900 Qwen3.8 `Q4_K_M` direct engine c1-c8

Direct engine packed-AR route, 512-token prompts and 128 generated tokens per
request, one warmup and three measured runs per width. This is a different
protocol from the server-protocol peer tables above and must not be compared
with them cell by cell. It is the first full-width sweep taken after the
2026-09-05/06 gfx1100 audit campaign.

| Width | c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prefill (total tok/s) | **678.785** | 368.906 | 362.560 | 379.976 | 378.273 | 403.582 | 385.332 | 376.638 |
| Decode (total tok/s) | 29.632 | 53.965 | 75.186 | 92.254 | 105.870 | 117.632 | 123.911 | **131.299** |
| Decode (per request) | **29.632** | 26.983 | 25.062 | 23.063 | 21.174 | 19.605 | 17.702 | 16.412 |
| TTFT (s) | **0.754** | 2.776 | 4.237 | 5.390 | 6.768 | 7.612 | 9.301 | 10.875 |
| Tracked peak (GiB) | **19.414** | 20.264 | 21.115 | 21.965 | 22.815 | 23.666 | 24.516 | 25.366 |

### Agentic quality (quality-only; no speed claim)

| Model | Overall | Development | Sealed heldout | Code / instruction / repository / tool | Valid calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B `UD-Q4_K_M` (reference) | 44/68 (64.71%) | 20/34 | 24/34 | 14/16 · 4/16 · 10/16 · 16/20 | 56/64 |
| Qwen3.8-27B `Q4_K_M` | **50/68 (73.53%)** | **22/34** | **28/34** | 14/16 · **12/16** · 10/16 · 14/20 | **64/64** |
| Ornith-1.5-35B-A3B `Q4_K_M` | 42/68 (61.76%) | 16/34 | 26/34 | 14/16 · 4/16 · 10/16 · 14/20 | 60/64 |

## Where detailed evidence lives

See result artifacts, [`CHANGELOG.md`](CHANGELOG.md), the
[`harness catalog`](HARNESSES.md), and [`BENCHMARK.md`](../docs/BENCHMARK.md).
Optimization history lives there, not in this current-row scoreboard.

Qwen3.8 runs plain AR on this backend; its speculative rows are retained for
explicit opt-in and re-measurement only.

Superseded campaigns, optimization deltas, and the per-campaign narratives that this current-row scoreboard no longer carries are archived verbatim in [`HISTORY.md`](HISTORY.md).

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

### Radeon Pro W7900: Qwen3.6-27B Dense GGUF

This `Q4_K_M`/BF16-KV snapshot uses one warmup and three measured resident
resets per shape with state-bound PM4 graph decode.

| Workload | Prefill | Decode | Tracked peak |
| --- | ---: | ---: | ---: |
| 512/128 | **875.364 tok/s** | **28.681 tok/s** | 15.587 GiB |
| 1K/128 | **911.658 tok/s** | **29.383 tok/s** | 15.681 GiB |
| 4K/128 | **878.721 tok/s** | **26.747 tok/s** | 16.204 GiB |

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
| W7900 / Qwen3.6-35B-A3B `UD-Q4_K_M` | Public production/BF16 resident-C2 K2 D24, automatic | 80.973 | **93.644** | **1.1565x** | Latest-source 10/10 engaged and MTP self-exact; three-run ratio 1.1368x; all categories non-regressive; strict-teacher, blocking/SSE/cancel/drain pass. Shares the artifact linked in the row above. |
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

## Reading the tables

Workloads use `prompt_tokens/decode_tokens`. Compare only matching timing,
model/quant/KV, concurrency, and memory scopes; bold identifies the reported
row, not a universal leader. A blank cell is a shape not yet measured, not a
failure, and Max context is published only where a dedicated ceiling run exists.

## Maintenance contract

1. Keep this file inside the compact-scoreboard budget that
   `tests/test_benchmark_readme_sync.py` enforces (< 500 lines, < 40 000 bytes,
   < 50 `results/` links); archive superseded material in `HISTORY.md` instead.
2. Replace the current row for a protocol tuple; do not append an optimization
   diary beneath it.
3. Put exact commands, samples, deltas, profiler data, correctness details, and
   candidate decisions in the compact JSON artifact.
4. Put the one-line old-to-new transition in [`CHANGELOG.md`](CHANGELOG.md) and
   substantial implementation decisions in a new immutable worklog entry.
5. Mention a blocked/rejected run here only when it removes a current numeric
   row or defines a user-visible limitation. Link one artifact and one rerun
   condition; keep candidate ladders out of the scoreboard.
6. Keep superseded tables in [`HISTORY.md`](HISTORY.md), artifacts, or Git
   history rather than copying them forward (`git show 6a8d38ae70b9e2c4244df10d8621db83da6c8112:benchmarks/README.md`).
7. Update `Last updated`, then synchronize the public block:

```bash
python3 scripts/sync_benchmark_readme.py --write
python3 scripts/sync_benchmark_readme.py --check
git diff --check
```

The full evidence and artifact requirements remain authoritative in
[`docs/BENCHMARK.md`](../docs/BENCHMARK.md).
