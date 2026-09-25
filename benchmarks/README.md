# hipEngine Topline Benchmarks

Last updated: **2026-09-26**

A **96-case BFCL quality diagnostic** on the same Qwen3.6 target and speculative
configurations scores hipEngine **75/96**, llama.cpp **76/96**, and gufo **73/96**.
All cases complete. Correcting hipEngine's omitted `parallel_tool_calls` default
from disabled to enabled raises its score from 52/96, with 23 newly passing
parallel-call cases and no regressions. The unchanged llama.cpp/gufo outputs
are reused. This small diagnostic is not the full accuracy gate or an
accuracy-parity finding. Performance rows below predate the default repair.
[Quality results and protocol](mlperf-edge/README.md#parallel-call-default-repair-2026-09-26) ·
[Artifact](results/2026-09-26-gfx1151-bfcl96-parallel-default.json).

MLPerf Edge Agentic **speculative 140-turn screen**, physical host **gfx1151 /
Radeon 8060S**, Qwen3.6-27B Q4_K_M target tensors, 32K context and concurrency 1:

| Engine | Completed | Mean turn latency | Measured phase | Inline command IoU |
| --- | ---: | ---: | ---: | ---: |
| hipEngine production + Qwen3.6 MTP | 140/140 | 5.665 s | 793.15 s | 0.5586 |
| llama.cpp HIP + Qwen3.6 MTP | 140/140 | 5.908 s | 827.10 s | 0.6423 |
| Patched gufo + Qwen3.8 DFlash2 | 140/140 | 7.439 s | 1,041.49 s | 0.6387 |

All three execute speculation and return valid tool calls. MTP arms use a
separate GGUF with all original target tensor bytes preserved plus NextN
weights; gufo uses the original target file. hipEngine/llama.cpp target KV is
BF16, gufo target KV is FP16. Effective cache-inclusive prefill is 4,744 / 6,089 /
approximately 8,023 tok/s; client-visible TTFT averages 2.535 / 2.872 / 7.039 s.
Different outputs, tokenization and timing windows preclude a kernel-throughput
or accuracy-parity claim. Inline IoU is not task correctness; see the separate
BFCL diagnostic above.
[Protocol and accounting](mlperf-edge/README.md#speculative-140-turn-comparison-2026-09-25) ·
[Paired MTP artifact](results/2026-09-25-gfx1151-edge-speculative-140.json).

Earlier **autoregressive 140-turn screen**, physical host **gfx1151 / Radeon
8060S**, same Qwen3.6-27B Q4_K_M, BF16 KV, 32K window, concurrency 1,
reasoning off and shipped cache policies:

| Engine | Completed | Mean turn latency | Measured phase | Inline command IoU |
| --- | ---: | ---: | ---: | ---: |
| hipEngine production, AR host sampler | 140/140 | 8.925 s | 1,249.47 s | 0.5586 |
| llama.cpp HIP, AR | 140/140 | 9.157 s | 1,282.06 s | 0.6420 |

One fresh-server pass each, with different outputs: not a stable speedup or
quality-parity finding. Both return valid calls without empty/missing/error
responses. Harness TPOT has a post-first-chunk tokenization defect and is not
valid decode-rate evidence. BFCL and full performance gates have not run.
[Protocol, commands and breakdown](mlperf-edge/README.md) ·
[Artifact](results/2026-09-25-gfx1151-mlperf-edge-140.json).

A compatibility-patched **gufo + Qwen3.8 DFlash2 draft** completes the same
Qwen3.6 target workload with **FP16 KV**, active speculation and prefix caching:

| Engine | Completed | Mean turn latency | Measured phase | Inline command IoU |
| --- | ---: | ---: | ---: | ---: |
| Patched gufo, Qwen3.6 target + Qwen3.8 DFlash2 | 140/140 | 7.439 s | 1,041.49 s | 0.6387 |

Draft acceptance is 15.79%; prefix token reuse is 96.55%. Effective prefill is
approximately 8,023 tok/s (full prompt tokens / pooled prefill seconds, durations
reconstructed from rounded server rates). Client-visible TTFT averages 7.039 s:
132/140 responses buffer tool output until essentially completion. The prior
pair below the speculative table is AR. Different KV/output settings
and no gufo AR control prevent a speculative-speedup or quality-parity claim.
The full BFCL gate has not run. [Timing definitions and checks](mlperf-edge/README.md#patched-gufo-with-cross-version-dflash2-2026-09-25) ·
[Artifact](results/2026-09-25-gfx1151-gufo-dflash-edge-140.json).


Qwen3.8-27B Q4_K_M sampling on **zbook / Radeon 8060S (gfx1151)**:
full-vocabulary GPU sampling achieves **11.61 decode tok/s and 9.63 engine
end-to-end tok/s**, 1.34% and 1.63% below the fresh greedy control.
Fast CPU ordering achieves 9.99 decode and 8.53 end-to-end tok/s.

The protocol uses BF16 KV, strict model arithmetic with the sampler algorithm
recorded separately, one active request in a capacity-4 service, MTP/cache off,
all 10 category prompts plus 8 heldouts at natural lengths, 16 output tokens,
temperature 0.7, top-p 0.95, no top-k cap, and two repetitions.

| Path | Selection ms/token | Decode ms/token | Decode tok/s | Decode loss vs greedy | End-to-end tok/s | End-to-end loss vs greedy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Greedy, fresh control | — | 84.95 | 11.77 | — | 9.79 | — |
| Original CPU | 49.17 | 135.59 | 7.38 | 37.5% | 6.55 | 33.3% |
| CPU with redundant sort removed | 25.08 | 111.38 | 8.98 | 23.9% | 7.81 | 20.5% |
| CPU with numeric sorting | 13.17 | 100.11 | 9.99 | 15.1% | 8.53 | 12.8% |
| Original GPU | 48.39 | 132.40 | 7.55 | 36.0% | 6.71 | 31.7% |
| GPU full-vocabulary sort/scan | 0.95 | 86.10 | 11.61 | 1.34% | 9.63 | 1.63% |

Decode rates pool steady transitions, excluding the first two transitions per
request. End-to-end rates divide all output tokens by total `generate_detailed()`
wall time, including prefill and every decode step but excluding model loading,
preparation and tokenization. Neither rate is HTTP throughput or a long-context
measurement. No graph replay occurs at this short horizon.

Original CPU/GPU and redundant-sort-removal rows are earlier same-host runs with
the same protocol; their loss percentages use the original greedy control
(11.80 decode / 9.82 end-to-end tok/s). Numeric CPU and sorted GPU share the fresh
interleaved control. Versus the originals, CPU decode/end-to-end improve
35.4%/30.2%; GPU improves 53.8%/43.5%.

All 144 requests complete and all 72 repeat pairs match. CPU generated tokens
match the original in all 36 sampled requests. GPU numerical fixtures and the
full-vocabulary c1/c4 lifecycle/state/ownership gate pass; GPU versus CPU or old
GPU sampled-token identity is not required. The short output suite is not a
separate task-quality certification. Supported, available GGUF native sampling
is default-on; `HIPENGINE_QWEN35_NATIVE_SAMPLER=0` selects host rollback. This does
not widen supported request shapes or claim gfx1100 hardware validation.
[Commands, correctness checks, and per-request evidence](results/2026-09-21-zbook-fast-cpu-gpu-sampling.json).

September 20 W7900 C1 refresh on physical host **epyc**, GPU0
`0xe282895b62c2b295`: BF16 KV, 512/128 and 4096/128, one warmup and
three measured repetitions. Both GGUF models pass 18-prompt graph/eager
identity checks; PARO has repetition-stable IDs, not a newly run independent
arithmetic gate. These are standalone resident-harness snapshots, not HTTP
throughput or new arithmetic promotions. MTP, INT8 KV, and capacity were not
re-measured. [Commands and samples](results/2026-09-20-w7900-gfx1100-v060-headline-refresh.json).

HTTP prefix caching defaults to radix. On W7900, the BF16 layout repair passes
all four categories and four heldouts with active and completed sources:
2,048 teacher-forced rows have zero KL and exact state against private
same-schedule recomputation. The serial-suffix arithmetic comparison is reported
separately; this packet does not validate INT8 prefix reuse.
[Correctness and diagnosis](results/2026-09-20-w7900-prefix-bf16-layout-repair.json).

The separate no-mirror INT8/FP32 prefix reader repair passes the same 16
category/lifecycle scopes on **epyc GPU1, Radeon RX 7900 XTX**, using
Qwen3.8-27B `Q4_K_M`: 2,048 teacher-forced rows have zero KL and exact
retained payload, scales, and state against private same-schedule recomputation.
This is not a W7900 transfer or cross-schedule output-identity claim.
[INT8 ownership evidence](results/2026-09-20-rx7900xtx-prefix-int8-ownership-repair.json).


September 20 C1 refresh on **Framework Desktop / Radeon 8060S**, physical
host `gfx1151`: Qwen3.6-35B-A3B `UD_Q4_K_M` measures 1,418.1 prefill and
56.7 decode tok/s at 512/128; Qwen3.8-27B `Q4_K_M` measures 404.5 and
12.15. Each has three timing samples and an 18-prompt graph/eager identity
check. Qwen3.8 production MTP completes the ten-prompt natural25 category
suite, including four heldouts, at 22.37 versus 11.30 tok/s true AR
(1.979x), with all ten cells exact, engaged and budget-conformed.
These are current snapshots, not an isolated optimization A/B.
The Qwen3.6 packed c2/c4 byte-exact diagnostic fails and is not certified
by these C1 results. Other model rows retain their stated measurement dates.
[Commands, provenance, samples, and limitations](results/2026-09-20-gfx1151-v060-headline-refresh.json).

Merged-tree working baseline on physical host `gfx1151` (Radeon 8060S): the
upstream prefix-cache A/B reproduces the merge screening rows (35B MoE all lanes
28.19 → 33.88 tok/s, 16/27 hits; dense 27B code lanes 5.64 → 7.35 tok/s, 9/18
hits). On the served multi-turn protocol with an in-load true-AR control, rows
that used MTP decode at **25.34 against 12.27 tok/s per request (2.07x)** with
median ITL halved (82.8 → 40.1 ms) at c=1, that per-request advantage is gone at
c=4 (9.59 against 9.30), and the c=4 speculative lane loses coverage to
activation (2 of 6 rows, 635 `no_provider` events). Warm prefix hits now
speculate too: a hit row restores the draft provider's checkpoint at its reused
boundary and streams its suffix, measuring **82.6 → 35-41 ms median ITL and
1.63-1.82x lower end-to-end latency** on the served c=1 protocol with
byte-identical output. This is a working baseline, not a promotion or an
aggregate MTP throughput claim.
[Baseline artifact](results/2026-09-20-gfx1151-post-merge-baseline-cache-and-mtp-arms.json).

Merge qualification on physical host `gfx1151` (Radeon 8060S): the local
MTP/containment branch and upstream prefix-cache branch pass the CPU and
serving integration checks, live cache-hit/lifecycle checks, and sampled-MTP
repeatability smoke. The dense27B serial-oracle numerical probe has a
pre-existing failure reproduced on untouched upstream; this is not full
production numerical qualification or a new performance claim.
[Qualification and limitations](results/2026-09-20-gfx1151-prefix-mtp-merge-qualification.json).

Surya OCR 2 fp32 on **zbook, Ryzen AI MAX+ PRO 395 / Radeon 8060S (gfx1151)**,
12 pages covering layout/markup, Japanese and mixed script, dense text, tables,
blank and degraded pages, and longer layouts. Both lanes explicitly execute
preprocessing, vision, prefill, and a manual greedy loop; loading is excluded.
Three measured repetitions follow a discarded warmup, in separate lane processes.

| Metric across 12 pages | hipEngine HIP fp32 | transformers torch HIP fp32 |
| --- | ---: | ---: |
| Decode rate | 55.4–57.0 tok/s | 25.1–30.2 tok/s |
| Complete request | 0.199–8.874 s | 0.273–13.732 s |

Median per-page speedup: **2.16x decode, 1.65x end to end**. hipEngine wins
end to end on 11/12 pages; blank is 0.438 s versus torch's 0.401 s.
All 24 rows pass reference, termination, repeatability, and output-format gates.
Eleven pages require exact generated IDs; the degraded scan gates its layout
structure and records bbox drift (90/1000) under the benchmark's ad-hoc prompt.
Full-page text accuracy is qualified separately with the real training prompt.
The final runtime passes 415 Surya tests with zero skips. Removing redundant
upload synchronization is near-flat in the paired run: decode rate +0.18%,
request latency +0.22%; no end-to-end speedup is claimed for that cleanup.
[Final per-page results](results/2026-09-14-gfx1151-surya-final-lanes.json),
[cleanup A/B](results/2026-09-14-gfx1151-surya-upload-cleanup-ab.json),
[protocol and environment](../docs/model-cards/MODEL-SURYA.md#final-lane-comparison-2026-09-14).
Separate processes follow the established protocol; combined-process qualification
is tracked in `docs/REFACTOR.md`.

Full-page transcription is gated against the text drawn on each page, not only
against a captured oracle. Using the checkpoint's real full-page HTML prompt,
eight documents (Japanese, mixed script, dense small text, a ruled table, a
blank page, a degraded scan, a 25-line block-heavy page, and a 300-DPI A4 page)
all reach a natural EOS with line recall 1.000, line exact rate 1.000, character
error rate 0.0000, and zero reading-order violations; the ruled table reads as
4x8 with 32/32 cells and a matching header, and a deliberately starved budget
is reported as truncation with omissions rather than passing a prefix. The A4
page is the page-scale case at 2480x3508 (220x156 patch grid, 8580 image
tokens): 2108 tokens to a natural EOS, vision 43.297 s, prefill 7.860 s, decode
40.504 s (52.04 tok/s), 91.891 s end to end (one request); its ground truth is
paragraph-level because its body is wrapped prose. One caveat is recorded: on
the degraded scan the HIP lane differs from torch fp32 on 21 of 616 ids, all
bbox coordinate digits, with identical labels and text and a worst coordinate
delta of 4 of 1000 — and torch bf16 differs from torch fp32 on 15 ids there and
changes the decoded text, so the drift is a property of that page's coordinates
rather than of the HIP route.
[Final transcription acceptance](results/2026-09-14-gfx1151-surya-final-transcription.json).

Surya OCR 2 is also reachable through hipEngine's OpenAI-compatible server.
`scripts/surya_http_e2e.py` serves the A4 page over `/v1/chat/completions` and
verified output equality with a direct call on 2026-09-12: 2108 completion tokens, 8701
prompt tokens (121 text plus 8580 image), 13 blocks, 12/12 units, CER 0.0000,
32/32 table cells, `stop`. Its historical 119.9 s HTTP timing is not a final
latency measurement. Vision bounds
and media form are engine declarations, so a page-scale model is admitted
without editing the server.
[HTTP serving e2e](results/2026-09-12-gfx1151-surya-http-serving-e2e.json).

Text-prefill score tiles take the byte budget's own width and vision tiles are
bounded by a shape envelope (`max(128, rows/32)` rows): the budget's widest tile
is within 1.1% of the text optimum at every measured length, while a dense vision tile costs 19-43% ([text](results/2026-09-13-gfx1151-surya-text-prefill-postfix.json), [vision](results/2026-09-13-gfx1151-surya-vision-tiling-postfix.json)).

This file is the current benchmark scoreboard. It intentionally contains only
current user-facing results, compact protocol/status notes, and links to the
authoritative evidence. It is not an optimization journal.

UD/main integration validation is documented in the
[integration report](../docs/campaigns/UD-MAIN-INTEGRATION.md); its working-tree timings
do not replace the published performance rows.

## Root README performance summary

The root README exports this compact retained summary verbatim.

<!-- BEGIN TOPLINE:README_HIGHLIGHTS -->
Tokens/s on each named host: **prompt processing** is input, **text generation**
is output. **MTP** is speculative decoding in qualified scopes. Dashes are
unmeasured; context limits are capacity tests.

### Performance

#### Radeon Pro W7900 — 48 GB (`gfx1100`)

| Model | Quant | Prompt processing | Text generation | With MTP | Max context |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | ParoQuant W4 | **2879.4** | **113.3** | 115.8 | — |
| Qwen3.6-35B-A3B | GGUF `UD-Q4_K_M` | **2924.5** | **95.0** | 122.7 | — |
| Qwen3.8-27B Dense | GGUF `Q4_K_M` (BF16 KV) | **898.6** | **30.8** | 39.7 | — |
| Qwen3.8-27B Dense | GGUF `Q4_K_M` (INT8 KV) | 868.6 | 27.9 | — | **176,128** |
| Laguna S 2.1 | GGUF `UD-Q2_K_XL` | **440.9** | — | — | — |

BF16 rows: September 20, `epyc`, 512/128 tokens, warmup + three runs.
INT8: September 13. MTP/capacity not refreshed. Laguna: 4K prompts.
35B GGUF MTP is opt-in; 27B MTP is **1.63x** its matched 24.36 tok/s AR,
not either decode column.

#### Strix Halo / Radeon 8060S — 120 GB (`gfx1151`)

| Model | Quant | Prompt processing | Text generation | With MTP | AR measured |
| --- | --- | ---: | ---: | ---: | --- |
| Maple-Preview | 2-bit | **754.5** | **153.2** | — | 2026-08-08 |
| Qwen3.6-35B-A3B | GGUF `UD-Q4_K_M` | **1418.1** | **56.7** | 80.1 | 2026-09-20 |
| Laguna S 2.1 | GGUF `Q4_K_M` | **654.2** | **23.2** | — | 2026-07/08 |
| Qwen3.8-27B Dense | GGUF `Q4_K_S` | **396.1** | **13.1** | **23.9** | 2026-08-17 |
| Qwen3.8-27B Dense | GGUF `Q4_K_M` | **404.5** | **12.15** | **22.4** | 2026-09-20 |
| Qwen3.8-27B Dense | GGUF `Q4_K_M` (INT8 KV) | — | **10.84** | **19.3** | 2026-09-23 |

September 20: Framework Desktop. Qwen3.8 `Q4_K_M` MTP: **1.98x** its matched
11.30 tok/s AR, 25 output tokens. A short-context measurement. 35B MTP is
the July 19 opt-in result. INT8-KV row: **1.78x** AR at c1, **2.39x** at c2,
**3.06x** at c4, documented override.
[Measurements](https://github.com/shisa-ai/hipEngine/blob/main/benchmarks/results/2026-09-20-gfx1151-v060-headline-refresh.json).

**Time-series forecasting (TimesFM 2.5 200M).** hipEngine decodes batch=8,
context 8192, horizon 512 forecasts in **0.082 s** on the HP ZBook Strix Halo
host (8.6x the official torch reference) and **0.062 s** on a Framework Desktop
host — the same GPU on two machines, so the gap is thermal headroom.

**VibeVoice-ASR 9B** is torch-free on Strix Halo. Q4_K_M beats bf16 — **1.52x**
prefill, **1.55x** decode — at 6.1 vs 16.7 GB, WER 2.01% vs 2.81%; **RTF 0.35**.
[Results](https://github.com/shisa-ai/hipEngine/blob/main/benchmarks/results/2026-09-14-gfx1151-vibevoice-q4-e2e-rtf.json).

#### NVIDIA RTX PRO 6000 Blackwell — 96 GB (`sm_120a`)

| Model | Quant | Prompt processing | Text generation | With MTP | Max context |
| --- | --- | ---: | ---: | ---: | ---: |
| Maple-Preview | 2-bit | **1917.5** | **402.4** | — | — |

### Long context on a 24 GB GPU

Qwen3.8-27B `Q4_K_M` can hold up to 232K tokens on a 24 GB `gfx1100` GPU
using [DMS](https://arxiv.org/abs/2506.05345), a trained KV eviction policy.
In these tests, DMS INT8 agreed more closely with BF16 than the direct-INT8 route:

| KV configuration | Max context | Top-1 agreement vs BF16 | Mean row-KL |
| --- | ---: | ---: | ---: |
| BF16 KV | 40,960 | — | — |
| DMS BF16 | 73,728 | — | — |
| Direct-INT8 KV | 131,072 | 91.4% | 0.188 |
| DMS INT8 | 232,448 | 100% | 0.001 |

Direct-INT8 failed 9 of 11 quality prompts and is not a default.
[Capacity evidence](https://github.com/shisa-ai/hipEngine/blob/main/benchmarks/results/2026-09-09-rx7900xtx-gguf-int8-direct-prefill-capacity.json)

### Prefix caching

Multi-turn conversations resend the whole transcript, so hipEngine reuses the
KV pages and hybrid state of any 256-token-aligned prefix a later request
repeats. Radix is the default for both the direct engine and HTTP serving;
`--prefix-cache off` rolls it back. Different prefill routes can change generated tokens.
Qwen3.6-35B-A3B `UD-Q4_K_M` on Strix Halo (`gfx1151`), 14 multi-turn lanes of
three turns, reusing 42,496 of 87,582 prompt tokens:

| Lane | Cache off | Cache on | Wall time |
| --- | ---: | ---: | ---: |
| Coding, transcript resent | 10.09 tok/s | **16.49** tok/s | **-38.8%** |
| Coding, transcript rebuilt | 10.13 tok/s | **12.67** tok/s | **-20.0%** |
| Chat (ShareGPT) | 29.50 tok/s | 29.15 tok/s | +1.2% |
| All lanes | 16.93 tok/s | **20.81** tok/s | **-18.6%** |

### Serving several requests at once

Aggregate tokens per second across all active requests, Qwen3.8-27B `Q4_K_M`
on the W7900, September 4, 2026. Peers use F16 KV where hipEngine uses BF16.

| Requests | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hipEngine | **23.6** | **39.1** | **53.1** | **63.9** | **72.8** | **79.5** | **83.2** | **85.9** |
| llama.cpp HIP | 21.0 | 34.4 | 30.6 | 27.7 | 36.7 | 46.4 | 52.1 | 58.4 |
| hipEngine advantage | +12% | +14% | +74% | +130% | +99% | +71% | +60% | **+47%** |

On Strix Halo, Maple-Preview 2-bit scales to **214.788** tok/s across eight
requests (123.131 at one, 202.038 at four). Where speculative
decoding runs automatically in production it is scoped to a qualified shape:
Qwen3.6-35B-A3B GGUF reaches **93.644 tok/s public** — 1.1565x its own AR — at
two concurrent requests on the W7900.
<!-- END TOPLINE:README_HIGHLIGHTS -->

## RX 7900 XTX Same-Model Comparison

Native C1 verification for dense `Q4_K_M` now uses allocated cache capacity,
not the historical 95-position workaround. Longer-context correctness checks:

| Model / hardware | Workload | Result |
| --- | --- | --- |
| Qwen3.8-27B `Q4_K_M` / RX 7900 XTX | 4K/8K/16K target verification, B1-B3; 8K prompt + 33 outputs, C1 B3, all ten category prompts | 192 exact state/logit/rollback cases; 10/10 end-to-end AR matches, 94 native graph calls, zero serial fallback |

[Reproduction commands and context diagnostics](results/2026-09-08-rx7900xtx-native-context-correctness.json).
These tested lengths are regression coverage, not a context ceiling.

Same Qwen3.8-27B `Q4_K_M` file on one RX 7900 XTX, HIP and BF16 KV.
PP/TG use 8192 prompt tokens and 128 decode transitions. C1 MTP uses the
full ten-prompt category suite, 25 visible outputs, explicit B3 and three
repetitions. Rates are tok/s; output agreement is against each engine's AR.
hipEngine uses the direct resident API; external rates use server phase timers.

| Engine | PP8192 | TG128 at 8192 | C1 MTP, 25 outputs | AR-ID matches |
| --- | ---: | ---: | ---: | ---: |
| hipEngine | 777.23 | 29.82 | **63.48** | 30/30 |
| nasone32 default | **943.14** | **35.70** | 58.56 | 27/30 |
| nasone32, sequential GDN | - | - | 58.65 | 30/30 |
| strix-llama.cpp | 901.73 | 29.55 | 47.69 | 30/30 |

Each engine completed a 112K BF16 prompt. With compact KV, hipEngine completed
128K using pure INT8/FP32 scales; nasone32 and strix-llama.cpp completed 192K
using Q8_0. These are synthetic execution-capacity observations, not serving
reserve recommendations.

Single-request capacity ladder, Qwen3.8-27B `Q4_K_M`. **RX 7900 XTX**: BF16
KV server 40,960; DMS BF16 73,728; INT8 KV direct engine 131,072; single
hidden plane 155,648; DMS INT8 merged lane 172,288; direct INT8 prefill or
DMS INT8 + single hidden plane 232,448.
[Direct-route capacity](results/2026-09-09-rx7900xtx-gguf-int8-direct-prefill-capacity.json),
[DMS alias ladder](results/2026-09-08-rx7900xtx-dms-int8-hidden-alias-ladder.json),
[speed evidence](results/2026-09-09-rx7900xtx-gguf-int8-direct-prefill-wmma-oracle-control-8192.json).
**W7900, INT8 KV server**: contexts up to **155,648 tokens** (verified at
65,536 / 98,304 / 131,072 / 155,648 with full prefills, decodes, and clean
shutdown at each point; ~2.9x the previous 54,272-token limit after the
route's memory repair)
([65,536](results/2026-09-10-w7900-server-alloc-probe-64k.json),
[131,072](results/2026-09-10-w7900-server-alloc-probe-128k.json),
[155,648](results/2026-09-10-w7900-server-alloc-probe-152k.json)).
W7900 default-route prefill at 8,192 tokens: 766.0 tok/s, decode 33.9;
the INT8 KV route runs the same workload at 677 tok/s
([gate](results/2026-09-10-w7900-int8-slot-local-aotriton-gate-corrected.json),
[paired repetitions](results/2026-09-10-w7900-p7-c1-speed-parity-paired-reps.json)).

Automatic resident context, W7900 (48 GiB), `hipengine serve --model <gguf>` with
no sizing flags. After weights load the server prices the per-request context
against free HIP memory and grows a shared KV pool on demand within its memory
budget. Four requests may be in flight by default; concurrency does not reserve
one full-context KV plane per request:

The serving defaults are INT8 KV (`int8_per_token_head`), fp32 scales, and four
resident request slots. Measured on the W7900 (48 GiB), one short chat request, no sizing
or KV flags, `Qwen3.8-27B-Q4_K_M.gguf` (SHA-256 `7b2aec3b…c89f1b`):

| KV storage | Slots | Selected context | Bytes/token | Usable | Whole-card use |
| --- | ---: | ---: | ---: | ---: | ---: |
| INT8, fp32 scales | 1 | **176,128** | 142,944 B | 24.62 GiB | 32.5 GiB |
| BF16 | 1 | 121,344 | 207,456 B | 24.62 GiB | 34.4 GiB |
| BF16 | 4 | 40,960 | 600,963 B | 24.62 GiB | 40.7 GiB |

The selection follows each model's own KV growth rate, so models differ
by their attention geometry rather than by a per-model table. The 3 GiB reserve
covers 2.66 GiB of device memory the capacity model does not price at all — HIP
context, JIT kernel modules, AOTriton, and KV pool pointer tables — which is
allocated after the free-memory reading the model prices against. The resident
growth rate is validated against prediction: two Tier-1 probes at 16,384 and
45,568 give a measured slope of 263,156 B/token against a predicted 262,979, a
**+0.067%** error

The selection follows each model's own KV growth rate, so the two models differ
by their attention geometry rather than by a per-model table. The 3 GiB reserve
covers 2.66 GiB of device memory the capacity model does not price at all — HIP
context, JIT kernel modules, AOTriton, and KV pool pointer tables — which is
allocated after the free-memory reading the model prices against. The resident
growth rate is validated against prediction: two Tier-1 probes at 16,384 and
45,568 give a measured slope of 263,156 B/token against a predicted 262,979, a
**+0.067%** error
([16,384](results/2026-09-12-w7900-27b-resident-slope-16384-bf16-c4.json),
[45,568](results/2026-09-12-w7900-27b-resident-slope-45568-bf16-c4.json)).
`--max-context-tokens` and `HIPENGINE_MAX_CONTEXT_TOKENS` still force a cap; a
context that cannot be allocated — automatic or requested — backs off with a
logged warning naming the size that was asked for, and
`HIPENGINE_GGUF_AUTO_CONTEXT=0` disables both the sizing and the backoff. The
24 GB-class arithmetic is the one that matters most for that class of card: at 4
resident slots with BF16 storage a 27B `Q4_K_M` is weight-bound and leaves room
for only about 3.3K tokens, and both the storage class and the slot count are
levers — each slot takes a full-context KV plane, so one slot carries roughly
three times the context of four.

[Full comparison and source review](results/2026-09-08-rx7900xtx-engine-comparison.md)
and [commands, samples and checks](results/2026-09-08-rx7900xtx-engine-comparison.json).

### DMS INT8 offline evaluation — RX 7900 XTX

| Qwen3.8-27B `Q4_K_M`, W8192 | BF16-DMS | INT8-DMS, FP32 scales | Reduction |
| --- | ---: | ---: | ---: |
| 16,384/32 device-store bytes, including workspaces | 829,473,104 | 432,046,928 | 397,426,176 (47.9%) |
| C1 execution fit, 73,728 prompt / eight decode; tracked / whole-card peak | Not paired | 23.186 / 23.938 GiB; 47.4 MiB sampled headroom | Not compared |

Four long-context categories pass (maximum KL 0.014937592; top-1 100%).
All ten canonical short prompts, including four category heldouts, pass dense-BF16
and BF16-DMS-relative checks (64 steps; dense-relative top-1 649/650).
64K C1/interleaved-C2 replay is byte-exact; the 73,728-token point passes finite-logit/drain checks, not a replay gate.
These are not calibrated production-profile or free-running task gates; heldouts
are not proven sidecar-train-disjoint. Packed/larger-C execution and full-session
speculative rollback are unqualified. BF16 fallback stays; no serving promotion,
whole-process memory percentage, throughput gain or proven context maximum.
[Numerical/lifecycle evidence](results/2026-09-07-rx7900xtx-dms-int8-postfix-audit.json);
[72K pass / 74K-75K OOM evidence](results/2026-09-07-rx7900xtx-dms-int8-requested-backoff.json); [dense context measurements](results/2026-09-07-rx7900xtx-int8-repair-capacity-audit.json).

## Current default notes

**Prefix caching is on by default.** Both `hipengine serve` and the in-process
resident engine loop default to `radix`; `--prefix-cache off` (or
`HIPENGINE_PREFIX_CACHE=off`) rolls it back. Radix reuses completed prompt
prefixes at 256-token granularity. Measured on `zbook`/gfx1151 with
Qwen3.6-35B-A3B `UD-Q4_K_M`, 14
multi-turn lanes and three turns each through the in-process resident loop:
radix resolves 16 of 27 lookups and reuses 42,496 of 87,582 prompt tokens, and
now runs **18.6% faster in wall time (351.5 s to 286.0 s, output rate 16.93 to
20.81 tok/s, +22.9%)**. The coding lanes carry the win — cumulative coding
**-38.8% wall** (10.09 to 16.49 tok/s) and fixture coding **-20.0% wall** (10.13
to 12.67 tok/s) — while the ShareGPT lanes are flat (+1.2%) because their
prompts are 30-685 tokens and most turns return `prompt_too_short` before any
lookup. The same command and lanes previously measured **+6.4% wall**, so the
sign of the result changed rather than its magnitude drifting: a reused request
used to prefill its unmatched suffix one token at a time at **34.0 ms/token**,
and now prefills it batched with the rest of the prompt.

W7900 BF16 cache-hit fidelity is tested independently: all 16 category/lifecycle
scopes pass the calibrated envelope with exact logits and state against private
batched recomputation. Gapped prefix pages now use the registered gather so their
attention route matches contiguous pages. Serial suffix replay uses different
arithmetic and its KL disagreement also occurs without cache reuse; both
comparisons are preserved in the
[layout-repair packet](results/2026-09-20-w7900-prefix-bf16-layout-repair.json).
Request eligibility fallbacks report `prefix_fallback_reason`; they do not
detect numerical divergence at runtime.
On gfx1151 the same-route packed-reference packet passes all eight suites at
1,024 teacher-forced rows with 2 top-1 flips and 0 state mismatches
([artifact](results/2026-09-20-gfx1151-prefix-gate-packed-reference-1024rows.json)).

Cache-hit fidelity is compared with private recomputation using the same
prefill schedule. Different chunk boundaries can change arithmetic and
generated tokens. In particular, a no-mirror INT8 continuation reads retained
quantized pages, while a fresh full prefill uses a transient BF16 oracle;
token-for-token identity across those routes is not guaranteed. The exact
same-schedule results above must not be read as cross-schedule identity.

Placement no longer decides which prefill route a cache hit gets. A gapped
device-KV placement used to fall to the packed paged route at roughly 6.5
ms/token against 0.39 ms/token for the slot-local prefill it displaced; a
gapped BF16 slot now keeps the slot-local AOTriton prefill by swapping its
identity spans for its real block table and gathering its pages into dense
head-major buffers with the existing block-table copy, so every hit pays
contiguous cost at any suffix length (a forced-gapped 35B suffix prefill
measures 8.3 s against 8.2 s contiguous; the packed paged route measured
134.5 s for the same shape). `HIPENGINE_GGUF_PREFIX_GAPPED_SUFFIX_MAX` still
defaults to 512 but binds only where the gather route is unavailable - the
`HIPENGINE_GGUF_GAPPED_GATHER` kill-switch, a backend without head-major KV
(gfx1100 today), or a context beyond the validated 64K head-major allocation
class. The wide retained working set
(`HIPENGINE_GGUF_PREFIX_RETAINED_SNAPSHOTS=16`) is therefore the strongest
configuration measured rather than a guarded opt-in. On the merged main
(including the post-merge private packed-workspace KV budgeting fix) the
same protocol re-measures it at **-18.4% wall against -17.8% for the default
retention** (20.82 vs 20.68 tok/s against a 17.00 tok/s no-cache baseline;
18/27 vs 16/27 lookups hit, 46,592 vs 42,496 reused tokens, zero declines,
worst single-turn prefill 15.8 s), with the pre-merge best at -20.1%/-18.4%
the day before — the same deterministic hit pattern, within same-host
day-to-day variance.
Current merged-main measurements:
[retained-16](results/2026-09-20-gfx1151-qwen36-gguf-prefix-cache-multiturn-ab-merged-retained16.json);
[default retention](results/2026-09-20-gfx1151-qwen36-gguf-prefix-cache-multiturn-ab-merged-default.json);
[off baseline](results/2026-09-20-gfx1151-qwen36-gguf-prefix-cache-multiturn-ab-merged-baseline-off.json).
Pre-merge gather-route evidence:
[retained-16](results/2026-09-19-gfx1151-qwen36-gguf-prefix-cache-multiturn-ab-gather-retained16.json);
[default retention](results/2026-09-19-gfx1151-qwen36-gguf-prefix-cache-multiturn-ab-gather-default.json);
[off baseline](results/2026-09-19-gfx1151-qwen36-gguf-prefix-cache-multiturn-ab-gather-baseline-off.json).

A replay adapted from one recorded coding-agent session measures 32
consecutive requests over one tool-using conversation, each re-sending the
cumulative history, with prompts growing 740 to 13,762 tokens. On zbook /
Radeon 8060S, Qwen3.6-35B-A3B `UD_Q4_K_M` runs through the in-process
resident loop with the server chat renderer (one process per arm).
Radix with 16 retained snapshots
reuses **222,720 of 231,530 theoretically shareable tokens (96.2%)**, hits 30
of 32 lookups, and completes the lane in **208.0 s against 581.3 s uncached
(-64.2% wall, 2.79x; 19.69 vs 7.05 tok/s)**, cutting prefill work by 86%
(428.4 s to 60.4 s). Default retention measures **-44.1% wall (19/32
hits)**: it keeps only two resident cache entries, so each new completion
evicts the previous request's snapshot and misses the reuse a cumulative
client depends on. A coding agent's resend pattern — every tool round-trip
re-sending the entire transcript — is the prefix cache's best case, and the
default-vs-retained-16 gap is 20 wall points here. This is one sequential
repetition, not a general performance or correctness qualification.
The replay substitutes system policy and tool schemas, omits thinking,
caps tool results, and uses recorded rather than newly generated continuations.
Outputs were not compared across arms. The artifacts record the uncommitted
benchmark changes used for the measurement.
Session-replay artifacts:
[retained-16](results/2026-09-20-gfx1151-qwen36-gguf-prefix-cache-session-replay-retained16.json);
[default retention](results/2026-09-20-gfx1151-qwen36-gguf-prefix-cache-session-replay-default.json);
[off baseline](results/2026-09-20-gfx1151-qwen36-gguf-prefix-cache-session-replay-baseline-off.json);
fixture: [agentic-session-replay-v1](prompts/agentic-session-replay-v1.json).

**Tuning candidate for coding-agent traffic:** on hosts with more than 24 GB
of graphics memory and sufficient free memory, evaluate
`HIPENGINE_GGUF_PREFIX_RETAINED_SNAPSHOTS=16` against the default on your workload.
The default retention count is `max(1, resident capacity)` — two resident
entries at capacity one — which evicts a long transcript's previous-turn
boundary while the conversation is still growing; the session-replay lane
missed 13 of 32 resends that way. Each retained entry pins one ~64 MiB
Conv/GDN state clone plus its conversation's KV pages (~5 MiB per 256-token
block, shared along a chain), so the measured 32-turn / 13,762-token session
pins **1.4 GB at 16 entries against 0.4 GB at the default** — and at the
default 1 GiB `HIPENGINE_GGUF_PREFIX_RETAINED_STATE_BYTES` ceiling the clone
budget is exactly full at 16 entries for a 35B checkpoint, so the count knob
and the byte ceiling meet there. Memory-tight hosts keep the default; both
knobs bound the pinned set. The wide default has not been measured under
concurrent multi-conversation serving, where 16 slots spread across
conversations instead of down one chain.

W7900 Qwen3.6 enables automatic MTP only for its qualified single-request and
capacity-2/two-request keys. **On W7900, Qwen3.8-27B `Q4_K_M` uses ordinary AR by default
at every width.** Explicit MTP is available there with production/BF16 KV, context
4-95 and 24 generated tokens: two active requests at resident capacity 2 with
K2/K3, or eight at capacity 8 with K3. A single active request at capacity 8
with K2/K3 currently engages through the legacy singleton target route (exact
against AR, but not the qualified packed target); the packed single-request
route remains a diagnostic-only test path until its serving evidence is
registered. K denotes maximum draft candidates per request; other keys fall
back to AR.

The 2026-09-06 `epyc`/W7900 snapshots use all ten category/heldout prompts,
greedy sampling, a 20 ms batch window and same-process true AR. All rows engage
and are token-exact on 10/10 prompts; these single runs do not qualify latency
or supply repeated-pair confidence estimates.

| Active requests / resident capacity | Requested depth | AR tok/s | MTP tok/s | MTP / AR |
| --- | --- | ---: | ---: | ---: |
| 2 / 2 | K2 | 42.20 | 42.41 | 1.005x |
| 8 / 8 | K3 | 92.67 | 97.35 | 1.051x |

A separate explicit C2/K3 qualification measured 44.69 versus 41.87 AR tok/s
(1.067x). Depths beyond K3 are not offered. [Measurements](results/2026-09-06-w7900-q4km-mtp-packet6-grid-and-c2k3.json).

**Single-request (C1) explicit MTP, legacy target route.** Three independent
runs per depth on one `epyc`/W7900 host; each run pairs every one of the ten
canonical/heldout prompts against same-process true AR with alternating arm
order, greedy, 24 generated tokens, 20 ms batch window, and a 10/10
token-exact gate against the AR output. Automatic selection stays AR.

| Requested depth | AR tok/s | MTP tok/s | MTP / AR (median, range) |
| --- | ---: | ---: | ---: |
| K0 (control, AR) | 24.42 | — | 0.9995x [0.9993, 0.9996] |
| K2 (legacy route) | 24.37 | 38.59 | 1.5844x [1.5831, 1.5862] |
| K3 (legacy route) | 24.36 | 39.70 | 1.6329x [1.6296, 1.6334] |

Source: [September 7 legacy-route measurements](results/2026-09-07-w7900-packed-c1-k0-k3-economics.json).

The K1 screening cell measured 1.3916x on the same legacy route
([1.3911, 1.3928]). These rates do not describe the packed target route
(measured below); a K7 refusal probe confirms a requested K7 runs AR
(engaged 0/10).

**Single-request (C1) packed target route — measured, not viable.** Under the
fail-closed diagnostic harness (injected one-request `packed_c1_target`
evidence row, legacy verifier forbidden process-wide, packed frontier calls
counted per run), the repaired packed target is exact but carries heavy
per-cycle overhead at one active request: three balanced pairs per depth
measured **K2 0.9908x** [0.9872, 0.9922] (a net loss against AR) and **K3
1.1140x** [1.1120, 1.1174] against the same-run K0 control 0.9986x — versus
1.5844x/1.6329x on the legacy route. The backend physical policy admits
packed C1 only at K2/K3; K1 and K4-K7 have no packed cell (a K1 harness
attempt is refused at the adapter and the legacy-forbidden guard converts
that into a hard 500; the subsequent shutdown-command timeout is teardown
noise after the aborted first request, not an idle hang). The packed
single-request route also survives K0↔MTP switching: a 40-leg sequence
(alternating explicit-MTP and automatic-K0 requests on one resident owner,
both switch directions, 130 packed frontier calls, legacy verifier never
invoked) kept every leg token-exact, engaged only the MTP legs, and exited
cleanly
([economics](results/2026-09-07-w7900-packed-route-c1-k0-k3-economics.json),
[switch proof](results/2026-09-07-w7900-packed-route-c1-k0-mtp-switch-proof.json)).
Reducing the packed one-row frontier overhead gates any packed-route
product registration. Until then the legacy route is the only
measured single-request MTP path.

Strix Halo `Q4_K_M`: the production profile uses AR, including when MTP is
explicitly requested. The current FP32-state manifest has no matching MTP
certificate. The supported strict/BF16 C1/K3 cell measures **20.985 tok/s**
versus **11.150 AR tok/s (1.882x)** across three complete ten-prompt runs.
Every category improves in every run, and all 30 prompt/run cells are exact.
Select `--execution-profile strict --speculative-candidate-budget 3` with
resident capacity 1 or 4, a declared session limit of at most 1,024 tokens
(`--max-context-tokens 1024`), 1-67 prompt tokens, a 25-token horizon, and ordinary
greedy/EOS handling. The benchmark used capacity 4. K4, `ignore_eos`, and
unsupported contexts or horizons select AR. The old FP16 production C8
measurement is historical evidence, not current-profile admission.
[Current measurements](results/2026-09-12-gfx1151-qwen38-final-headline-refresh.json)
and [serving gates](results/2026-09-12-gfx1151-qwen38-serving-mtp-closure.json).

Long-context verification inside that cell now keeps full-attention rows in the
staged chain across the 1,024-token split boundary instead of dropping to the
per-row scalar owner. Each row attends on the leaf the scalar owner would use at
its own position, so the batched projections, head norm/rotary, KV writes, and
output projection stay batched across rows on both sides of the boundary. On the
10-case eager long-context packet (cycle ends 1024/1025 x B1/B2/B3 plus four
controlled B3 acceptance cases, three runs per arm on this host) the packet's own
cycle wall measures **8.129 s -> 7.893 s median (-2.9%, mean -3.6%, 7 of 10 cases
faster)** at unchanged 16/32 split-K calls per case and with every correctness
flag identical against the serial-exact teacher. The same per-row leaves stay
exact at 2K, 3.6K, and 8K spans, including 32 splits per row at 8K against a
33-split workspace. The same code under the captured target graph at cycle end
1032 passes with the same 32/48/64 split-K calls. This is a verifier-route
change: the serving limits above are unchanged.
[Eager long-context verifier staged chain](results/2026-09-18-gfx1151-qwen38-long-context-eager-verifier-staged-chain.json).

**Serving status (2026-09-18):** that staged-verifier change also reached
short-context dense verifier rows, because it replaced the gate that declined the
staged route whenever the session's context limit was above the 1,024-token
split threshold. On the affected tree the automatic MTP route stopped
reproducing the autoregressive arm's greedy output: the two agreed for 8 output
tokens and diverged at token 7 for 16 or more, on an ordinary prompt, under both
the strict and the production execution profile, and the divergent text was
incoherent where the autoregressive arm's was correct. The executed route was
localized with a temporary trace of the session, layer router, and staged chain:
the commit's per-row branch launched the backend's short-context batch attention
leaf, which takes the row's position cap as a host scalar, where the plain span
leaf reads the row's own live count from the spans. The staged chain now takes
that substitution only from a caller that staged the metadata it reads back, so
the captured target graph keeps the span leaf. On the retained ladder protocol
the single-prompt arms return to exact at 8, 16, and 32 tokens, and the matched
c=1 lane returns to **23.46 tok/s against true AR's 11.74 (2.00x)**, with 99.7%
of output tokens from speculative cycles and 2.04 accepted drafts per cycle
against **9.37 against 11.75 (0.80x)** before the fix. The longer-context
verifier keeps the commit's win: the retained eager packet still moves 8.129 s ->
7.893 s with every correctness flag identical against the serial-exact teacher.
That lane's control tree is not exact in general: its two arms are byte-identical
on five of seven natural prompts and diverge at tokens 47 and 155 on the other
two, and on a raw completions prompt a 32-token run differs by one repetition
token on the control tree and on the fixed tree identically, so that residual
late divergence is a property of this verifier route that the fix restores rather
than removes and it still needs the production numerical envelope before it is
called acceptable. [Verifier AR
identity fix](results/2026-09-18-gfx1151-qwen38-mtp-verifier-ar-identity-fix.json);
[exactness regression and exact-tree
control](results/2026-09-18-gfx1151-qwen38-mtp-exactness-regression.json);
[matched serving baseline](results/2026-09-18-gfx1151-qwen38-sharegpt-natural-serving-baseline.json);
[concurrent coupling arm with an in-run AR control](results/2026-09-18-gfx1151-qwen38-mtp-serving-coupling-route-mix.json).

**Concurrency (2026-09-18):** on the fixed tree the same coupling protocol
measures **24.01 tok/s for the automatic route against 20.14 for true AR
(+19.2%) at c=2** with every speculative row at 99%+ coverage, and **12.06
against 12.14 (-0.7%) at c=4** with every speculative row at realized width 1.
The c=4 lane is AR-favorable on this model, quant, and host: a speculative cycle
there costs about 2.2 autoregressive steps (median ITL 208.4 against 141.1 ms)
while emitting about 2.1 tokens, so pairing rows cannot flip the sign, and the
retained measurement of the wider cell after the faster AR rebase is 1.0517x with
one category below 1.0. Wider realized groups therefore stay closed, and the
width-2 automatic promotion is rejected on this evidence. [Wider realized groups
rejected](results/2026-09-18-gfx1151-qwen38-mtp-wider-realized-groups-rejected.json).

**Refused groups (2026-09-18):** decomposing a due group the provider refuses
into one provider cycle per capable row is **not** a way to recover coverage. On
the protocol the review named -- a row that crosses the provider's 1023-token
window mid-stream beside a co-resident that never crosses -- the split raises MTP
output share from **1.8% to 69.3% at c=2 and 2.1% to 73.9% at c=4** while decode
falls **17.08 -> 14.17** and **18.63 -> 14.65 tok/s** end to end, and **21.33 ->
17.51** / **24.68 -> 16.94** excluding TTFT, against AR controls of 20.28 and
22.08. Acceptance is not the limit: the healthy row accepted depth 3 on all 162
of its cycles (99.8% coverage) and still cost **111.1 ms per token against 92.1
ms** when the group stayed autoregressive. A refused group therefore keeps
decoding autoregressively in one batch, and the split route stays behind a
default-off flag. The separate cost of the route itself is prefill: the refused
speculative arms spend 3.4x the AR arm's TTFT (11.9-13.0 against 3.5 s on
~840-token prompts) and then decode at the AR rate. [Refused-group split
rejected](results/2026-09-18-gfx1151-qwen38-mtp-refused-group-split-rejected.json).

**Sampled acceptance (2026-09-19):** temperature sampling is the largest
remaining reason a real request cannot use MTP at all -- every processor except an
EOS gate makes the row autoregressive, because the verifier's argmax accept emits
`argmax(p)` where the request asked for a draw from `p`. The sampled rule that
closes that gap (accept the drafted token with `min(1, p/q)`, otherwise resample
from `normalize(max(0, p - q))`) is implemented, is what the device now commits,
and **passes its arithmetic gate
on real gfx1151 rows**: across three prompts and eight decode steps each,
**824 of 824 induced-law comparisons are exact** (max total variation 4.5e-16,
max KL 9.0e-16, top-1 agreement 1.0 against a 1e-9 tolerance) over supports up to
248,320 tokens, **144 of 144** autoregressive-law agreements hold, and a
Monte-Carlo arm that drives the real accept/resample walk 4,000 times per sampler
config reports top-1 agreement 1.0 with no cell outside its calibrated limit.

Serving that route now **wins**. With the coupled accept committed on the
device, the protocol that measured **0.47x** at c=1 (**6.37 against 13.56 tok/s**,
median TTFT 5,628 against 1,735 ms) measures **26.05 against 13.73 tok/s
(1.90x)** with median ITL **36.4 against 132.1 ms** and **253 of 256** output
tokens from speculative cycles; acceptance is unchanged, so the gain is the
cycle's own cost and not draft quality. The limit was architectural: the native
target graph commits the argmax accept on device, so a sampled row used to run
without graph replay and with no device proposal, and the accept decision was
computed on the host from one full-vocabulary logits row per verified prefix
(528.5 ms per cycle against 98.9 ms for the identical draft chain at
temperature 0). A sampled row now keeps the N2 target graph through a captured
sampled accept/commit variant that walks the drafted chain with the row's own
staged draws, so the row's sampler stream stays aligned with the autoregressive
route. The production cap4 c1 evidence row advertises the sampled mode, so a
temperature request at that key is served by this route by default; c>=2 lanes
still inherit the one-row policy cell and are not covered. [Sampled-acceptance
distribution
gate](results/2026-09-18-gfx1151-qwen38-mtp-sampled-accept-distribution-gate.json);
[serving measurement
(accepted)](results/2026-09-19-gfx1151-qwen38-mtp-sampled-acceptance-serving-accepted.json);
[rejected](results/2026-09-18-gfx1151-qwen38-mtp-sampled-acceptance-serving-rejected.json).

**Historical prefix-cache hits (2026-09-18, before batched suffix prefill):**
a hit was a measured loss on a realistic
multi-turn load, and the reason is the prefill route rather than the cache. Four
ShareGPT conversations replayed as four growing turns each (272-1,098-token
prompts, 128 output tokens, the true-AR control and the MTP arm in one load)
hit **12 of 16** lookups -- every turn after the first, from
`completed_snapshot`, at 256-512 matched tokens and 158,859,264 state-clone
bytes -- and the hit rows prefill the reused-prefix suffix **one token at a
time**: **82.1 ms per executed token against 3.3 ms on the miss rows (24.9x)**, a
**33,475 ms median TTFT against 1,160 ms**, with the suffix split across 1-3
chunks in `prefill_ms`. The same refusals that block the suffix also block the
draft provider, so every hit row ran 100% autoregressive (0 cycles, one
`no_provider` event per output token) against 126 of 128 tokens from speculation
on the matching miss rows. End to end the same load decodes **3.47 tok/s with
`--prefix-cache radix` against 46.52 with it off**, so that version kept the
cache off by default. The current batched-suffix implementation and default
are described in the prefix-caching section above; these historical rates do
not describe it. MTP provider checkpoints for cache hits are still unavailable.
The earlier 11.8x prefix win was a
p256+s1 packet, where the per-token loop costs exactly one step. [Prefix hits
rejected](results/2026-09-18-gfx1151-qwen38-prefix-hit-multiturn-rejected.json);
source: [worklog entry](../worklog/entries/20260918T180419.254833Z-lhl-prefix-hit-multiturn-462873.md).

**c=1 engagement (2026-09-18):** with the server's default `auto` policy and the
production profile, Qwen3.8-27B `Q4_K_M` on gfx1151 speculates at one active
request: **20.0 tok/s against a matched 11.90 tok/s AR baseline (1.68x)** at a
517-token prompt and **17.8 against 11.75 (1.52x)** at 945, in three runs per
shape with a CV at or below 0.07% and identical generated text over the
client's first 600 characters per row; a 3,530-token prompt declines to AR at
the adapter's 1,023-token window.
This corrects the earlier public statement that production-profile requests use
AR. [c=1 engagement](results/2026-09-18-gfx1151-qwen38-mtp-c1-engagement.json).

**c=1 engagement re-verified (2026-09-19):** the same client, protocol and host
at `b25740a50`, with one `auto`-policy server load serving all three arms,
measure **20.08 tok/s for the explicit request against 11.98 for the in-load AR
control at a 517-token prompt (1.676x)** and **17.98 against 11.83 (1.521x)** at
945, with the no-field default within 0.05% of the explicit request and every CV
at or below 0.09%. A non-streaming route probe confirms engagement rather than
inferring it from the rate: the 517- and 945-token shapes report
`effective_route=speculative_mtp` with 55 and 23 cycles at 0.442 and 0.818
acceptance, while 3,530 tokens still declines inside the adapter's 1,023-token
window (selected `speculative_mtp`, effective `default`,
`decision_reason=backend_k0_fallback`). The arms' first 600 generated characters
are byte-identical on all three shapes. The topline row above is unchanged
because this run reproduces it.
[c=1 engagement recheck](results/2026-09-19-gfx1151-qwen38-mtp-c1-engagement-recheck.json).

**Realized groups and the width cells (2026-09-19):** the width policy's C5-C8
question is answered by what the server actually executes. Across 36 measured
points -- 8-prompt ShareGPT loads with four 17-34-token and four 696-841-token
prompts, at concurrency 2 through 8, with burst, 400-ms-staggered, and 50-ms
dispatch-window arrivals, each against a true-AR arm in the same load -- **every
one of the 24 speculative points ran one-row speculative groups**. No wider group
formed in any arrival pattern, so the wide cells are not a throughput lever on
real traffic and concurrency never batches speculative rows. What MTP buys at
concurrency is interleaving: with staggered arrivals it holds 6-7 of 8 rows and
about half the output tokens at every concurrency, and decodes **18.92 against
true AR's 15.22 tok/s at c=2 (+24%), 23.74 against 18.84 at c=4 (+26%), and 25.20
against 18.87 at c=8 (+34%)**, while true AR itself saturates near 19 tok/s from
c=4 up. Burst arrival is not that regime: the provider's single prompt activation
is contested, so only 1 of 8 rows engages at c=4 and c=7 and none at c=8, and
those points measure contention rather than the route.

The engaging wide cell does not survive re-measurement. On the same instrument
that qualified it (one synchronized 8-row batch per cell, 10 mtp-bench prompts,
24-token horizon, true no-MTP AR baseline in the same run) the gfx1151 production
`(8,3)` cell measures **46.88 tok/s against 50.10 AR (0.936x)** with **0 of 10
cells reproducing the AR output**, against the retained 2026-09-05 qualification
of 52.10 against 52.03 (1.0015x) with 40 of 40 cells AR-equal. The failure is not
numerical drift: every one of the 80 rows reports
`effective_route=speculative_mtp` with 7-9 draft cycles and a
`physical_streaming_category_rejected` event, because the backend's prompt-streaming
policy admits widths 1-4, so the provider opens without prompt state while the
route still runs. The generated ids agree with AR for 3-5 tokens and then land in
the same `<|im_end|>`/`<|im_start|>` loop on all 10 prompts -- the MTP arm's
output does not depend on the prompt. The cell is reachable through production
admission, not just the diagnostic resolver: the same protocol run without
`--generation2-diagnostic` engages it on all 10 cells and reproduces the identical
degenerate output. The C5-C7 refusal cost did improve to -6%/-7%/-7% from
-15%/-18%/-18%. The gfx1151 production `(8,3)` cell is therefore withdrawn: it is
removed from the backend's width/depth policy and its serving-evidence row is
deleted, so an explicit capacity-8 MTP request takes the registered strict
fallback and returns the AR output. Re-qualifying a gfx1151 wide cell requires
admitting widths 5-8 to the prompt-streaming policy first, and the current
width-8 cycle is 0.936x of AR. The gfx1100 cell is unaffected: that backend's
prompt-streaming policy admits width 8. [Width census and
C8/K3
withdrawal](results/2026-09-19-gfx1151-qwen38-mtp-width-census-and-c8-k3-withdrawal.json).

**Packed workspace lease (2026-09-19):** the eager packed-execution KV
workspace on gfx1151 now leases one slot per serving capacity instead of a fixed
eight. The slot term read a `max_batch_size` attribute the resident model runner
does not have, so every server pinned an eight-slot workspace whatever
`--max-active-requests` said. On the 27B `Q4_K_M` / BF16 KV shape with
8192-token sessions the lease falls from **256 pages (4.00 GiB) to 32 pages
(0.50 GiB) at one active request** and from 256 to **128 pages (2.00 GiB) at
four**, with device memory after a 5,469-token prompt at **33.74 -> 30.24 GiB**
and 34.55 -> 32.55 GiB. With the default automatic context the one-request lease
falls from 4096 pages (64 GiB) to **1024 pages (16 GiB)** while the resolver
spends the freed memory on context: it selects 262,144 tokens where the
8-slot lease left it 131,072, so the four-request arm leases the same 4096 pages
at twice the context. The slot term is the serving capacity and not one more:
an MTP verify group is K+1 rows *inside* one slot, and across 24 geometry
requests at one active request (two short prompts, an explicit MTP request, and
the 5,469-token prefill) the largest slot count any layout asked for was 1. No
workspace allocation failed at either capacity and every request completed.
[Workspace lease artifact](results/2026-09-19-gfx1151-qwen38-packed-workspace-lease-capacity.json);
[per-run rows](results/2026-09-19-gfx1151-qwen38-packed-workspace-lease-capacity-rows.json);
source: [worklog entry](../worklog/entries/20260918T205735.370215Z-lhl-packed-workspace-lease-capacity-be3814.md).

TimesFM 2.5 200M GPU decode (batch 8, context 8192, horizon 512) — **two
physical Strix Halo `gfx1151` hosts, recorded as separate lanes**: **0.082 s**
on the power-limited **HP ZBook Ultra G1a** and **0.062 s** on the **Framework
Desktop** (median of five independent process invocations each). Both pass the
double correctness gate described above. The 25% gap is host power/thermal
headroom, not a code difference, and is not an old→new delta. Attention runs as
fused WMMA flash kernels (long-prefill and split-kv short-query variants) and
the FP16 GEMMs use shape-keyed rocBLAS solution autotuning
([zbook artifact](results/gfx1151-timesfm-quadtile-flash-2026-09-09.json),
[Framework artifact](results/2026-09-11-framework-desktop-timesfm-2p5-decode-lane.json)).

TimesFM 3.0 500M GPU decode (batch 8, 3 variates, context 8192, horizon 512,
one non-autoregressive pass) — same two lanes: **0.319 s median** on the
power-limited **HP ZBook Ultra G1a** (4.17x the torch fp32 reference on that
host) and **0.220 s median** on the **Framework Desktop**. The same double gate
on three oracle fixtures (multivariate, unaligned/univariate, covariate-mask +
640-horizon); the fp16 production path fuses the variate-attention QK norms
in-kernel with a strict unfused fp32 fallback. Torch comparison protocol in the
artifact; the Framework lane has no torch install, so no torch ratio is claimed
there. The 32% gap is host power/thermal headroom, not a code difference
([zbook artifact](results/gfx1151-timesfm3-varnorm-fusion-2026-09-10.json),
[Framework artifact](results/2026-09-11-framework-desktop-timesfm3-decode-lane.json)).

EVIE-4.5B multimodal retrieval encode (8 pages 448x336 + 8 queries): the
batched segment-isolated runtime (encode_documents/encode_queries, ragged
no-padding) runs the fp16 production path in **1.13 s** (doc 1.02 s, query
0.11 s; sequential path 1.81 s). Under the matched protocol (identical
real-processor inputs, preprocessing outside both timers, full 8x8 scoring;
reproduce with scripts/evie_matched_protocol_{torch,hip}.py): **hip fp16
1.17 s vs torch bf16 1.29 s — hipEngine leads 1.10x** (doc 1.04x, query
1.61x). Retrieval quality vs the torch fp32 teacher: hip fp16 preserves
the argmax ranking 8/8 (torch bf16 5/8), max rel 0.24% (bf16 3.5%).
The fp16 path is fixture-gated against the torch fp32 oracle (query
embedding cos 0.9999986, MaxSim within 0.01%); a strict fp32 path passes
the same oracle at 1e-5 tolerances, with a registered strict-elementwise
fallback (HIPENGINE_EVIE_STRICT_ELEMENTWISE). Full optimization history
(15.62 s -> 1.13 s)
in the artifacts
([sprint](results/gfx1151-evie-4p5b-fp16-perf-sprint-2026-09-09.json),
[cluster8](results/gfx1151-evie-4p5b-cluster8-recurrence-2026-09-10.json)).

### XTX (RX 7900 XTX) Qwen3.8 `UD-Q4_K_M` / `UD-Q4_K_S` dynamic-routing stack

No valid final W7900 parity row exists: the four-arm A/B below was measured on
the XTX, and the W7900 paired MTP run is withdrawn as invalid (see the dated
retraction in [`CHANGELOG.md`](CHANGELOG.md)).

2026-09-11 four-arm A/B (hipEngine direct, p512/d128, graph-replay decode,
medians of 3, all four arms same-conditions): the UD artifacts vs their plain
counterparts. Prefill is tokens/s.

| Model | Prefill plain | Prefill UD | Ratio | Decode plain | Decode UD | Decode ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `UD-Q4_K_M` | 986.0 | **905.6** | **0.918x** | 35.37 | **29.98** | **0.848x** |
| `UD-Q4_K_S` | 946.0 | **893.7** | **0.944x** | 36.66 | **29.49** | **0.804x** |

2026-09-11 decode-parity campaign final re-measure (same conditions; the
plain arms also gained from the shared Q5 local32 owner — plain K_M
34.12 -> 35.37, plain K_S 30.55 -> 36.66): the four levers — IQ4_XS
gate/up dual+SiLU, IQ4_NL local32, the split/scale IQ (IQ3_S/IQ3_XXS/
IQ2_S/IQ2_XS) local32 family, and the Q5 dense local32 single — took UD
decode 26.20 -> **29.98** (K_M, +14.4%) and 25.53 -> **29.49** (K_S,
+15.5%), pure-kernel 35.23 -> 30.26 and 36.16 -> 30.02 ms/tok, 898 ->
831 and 918 -> 839 launches/tok. Both artifacts pass the tokenized
category gate on the final stack (K_M max 1.484e-2, K_S max 2.014e-2,
top-1 99.91%); K_M's local32-family tail is pinned at its admitted
state (all four IQ3_S decode slots strict). Remaining decode levers:
the Q5 MoE selected-expert path (2.22 ms/tok K_M on the direct GEMV),
Q3_K strict decode (1.43/2.43 ms/tok), and the Q5 gate/up dual.
([final rollup](results/final-decode-campaign-2026-09-11/).)

MTP for these artifacts is now **admitted**: the U6 certification records are
complete, `_UD_MTP_PRESET_FINGERPRINTS` carries both fingerprints, and the
artifacts derive the `mtp` scope alongside `ar`. The admitted scope is width c1
and the 4-95 token context bucket, and it is the measured scope — c2 K2 engages
and holds `ar_exact` but reaches only 1.0418x, c4 has no production physical
width cell, c8 runs out of memory at capacity 8 on a 24 GB card, and above 1023
context the adapter refuses MTP while the packed verifier stops batching. An
admission check with no request field, no diagnostic flag and no in-process
grant measures `UD-Q4_K_M` AR 14.23 -> MTP **34.32** tok/s (**2.4126x**) and
`UD-Q4_K_S` AR 15.19 -> MTP **35.71** tok/s (**2.3517x**), 10/10 engaged and
10/10 `ar_exact`, on the server-path diagnostic denominator
([admission check](results/2026-09-12-ud-gfx1100-mtp-automatic-admission.json),
[width census](results/2026-09-12-ud-gfx1100-mtp-width-cells.json)). That
denominator is not the resident graph-replay leaf used below, so it does not
replace these rates.

The retained paired measurement (c1, natural25, B3, ten prompts, two repeats,
recorded production graph replay for the true-AR arm), taken on a clean
worktree at `92c7e3dc4`:

| Tier | UD AR | Plain AR | UD / plain AR | UD MTP B3 | Plain MTP B3 | UD / plain MTP | UD MTP / AR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Q4_K_M` | 32.922 | 38.373 | **0.858x** | 50.231 | 65.347 | **0.769x** | **1.5258x** |
| `Q4_K_S` | 32.311 | 41.135 | **0.785x** | 49.356 | 70.092 | **0.704x** | **1.5275x** |

All four arms have complete 20-row-per-group evidence, deterministic repeats,
GPU/CPU acceptance agreement, and a positive MTP/AR ratio. The `Q4_K_M` arms
are generated-ID exact and both carry `speed_claim_eligible: true` from a clean
worktree. `Q4_K_S` carries the recorded `general_ja_plan` exactness divergence
(2/20 rows) described below, which the suite scores as `correctness_failed` and
so marks `speed_claim_eligible: false`; the `Q4_K_S` row is published as a
measured rate with that caveat, not as an eligible speed claim. The remaining
gap is concentrated in MTP compute: UD trails plain by 14.2%/21.5% in AR but
23.1%/29.6% in MTP.
([paired artifact](results/2026-09-13-ud-gfx1100-paired-clean-provenance.json),
[predecessor, superseded](results/2026-09-13-ud-gfx1100-rounded-norm-fixed5120.json),
[provenance audit](../worklog/entries/20260913T114312.816781Z-lhl-ud-claim-evidence-audit-c22a97.md).)

Those rates follow the rows 2-4 local32 IQ verifier sibling
(`2ac44d7a7`), the Q5_K gate/up pair row gate, the Q8_0 attention K/V
rowtile route, the Q5T16 single-wave verifier rowtile, and the rounded
fixed-5120 norm leaf (all `2026-09-13`), which together moved UD MTP B3
35.541 -> **50.231** (K_M, +41.3%) at flat AR; the K_S progression
33.061 -> **49.356** carries the same exactness divergence at both ends.

**Provenance of the per-lever deltas.** The incremental before/after figures
quoted in the per-lever paragraphs below were measured on worktrees that the
suite records as `dirty: true`, so their own provenance gate marks them
`speed_claim_eligible: false`. They remain valid as kernel- and
verifier-level diagnostics, and the verifier sub-window A/Bs they rest on are
separate measurements, but only the retained table above carries
`speed_claim_eligible: true`. Re-deriving each lever's end-to-end delta on a
clean tree is tracked in the provenance audit entry.

The rows 2-4 local32 IQ verifier sibling gives each block
several prompt rows over the same local32 IQ decode geometry, with every row
bit-identical to the rows == 1 owner's output for that row; at kernel level it
is 2.0-4.3x the strict per-row GEMV at rows 2-4 and covers 94.1% (K_M) / 85.9%
(K_S) of the rows 2-7 dead-zone MACs. The block verifier also had to bind the
dense-IQ execution-owner session, without which its raw-IQ rows 2-4 kept the
strict owner. All four section-6.1 arms still pass with at least 24x margin on
the binding mean limit and 100% top-1 overall and per category, and a
fresh-process reproduction matches all sixteen arm numbers to the last digit.
([kernel headroom](results/2026-09-12-ud-gfx1100-phase4-local32-rows-headroom.json),
[numerics gate](results/2026-09-12-ud-gfx1100-phase4-ar-verify-numerics.json),
[gate reproduction](results/2026-09-12-ud-gfx1100-phase5-ar-verify-numerics.json),
[implementation entry](../worklog/entries/20260912T092947.074218Z-ud-phase4-lane-ud-phase4-local32-rows-sibling-103062.md).)

The Q8_0 attention K/V route removes the last prefill-sized owner from the
verifier. `gguf_q8_0_t16_prefill_wmma` ran the ten `(5120, 1024)` Q8_0
`attn_k`/`attn_v` projections at verifier rows as one wave32 per block with 32
blocks total — 122.6 us per call, 45 GB/s on 5.6 MB tensors — for 0.858 ms/step,
or 2.0% of verifier kernel time. Admitting those shapes to the 128-thread
`q8_0_t16_rowtile_gemv` owner, which reads each tile once for four rows, runs
the same 84 calls per step at 51.8 us each for 0.363 ms/step: **-0.495 ms/step
on the Q8_0 family (-57.7%)** at an unchanged call count. The generated token
sequence is unchanged, and the section-6.1 gate improves on every metric —
mean/p95/p99/max KL **3.02e-05 / 1.23e-04 / 2.81e-04 / 3.66e-04** against
3.92e-05 / 1.88e-04 / 4.06e-04 / 4.06e-04 before, top-1 1.0000, two
deterministic repeats. The paired arm measures UD-Q4_K_M 45.973 -> **46.192**
tok/s (+0.48%) against a plain-control spread of -0.31%/+0.18%, and UD-Q4_K_S
46.990 -> **47.023** (+0.07%), which tracks the shape count: K_M carries ten of
these projections and K_S three.
([census](results/2026-09-13-ud-gfx1100-q8-rowtile-attn-kv-census.json),
[numerics gate](results/2026-09-13-ud-gfx1100-q8-rowtile-attn-kv-ar-verify.json).)

The rounded add+rmsnorm leaf now has a fixed-5120 owner too. It was the last
norm-family item in the verifier without one, and it carried the same defect the
plain generic owner had: one block per row, but a runtime-trip-count loop over
`hidden_size` with no register cache, so it read the residual and the addend
twice and paid the nine-barrier tree. `gguf_norm_fixed5120_wave256_kernel` gains
a `kRoundSum` template parameter that makes both the reduction and the residual
output consume `bf16(bf16(x) + bf16(add))`, the `add+rmsnorm` layer gets a
`rounded_bf16_out_fixed5120_wave256` variant of it, and gfx1100 declares
`GGUF_ROUNDED_NORM_RESIDUAL_DECODE_POLICIES` for `2 <= rows <= 8`. The
interleaved A/B (three baseline runs, two candidate runs) puts the family at
**12.640 -> 5.171 ms over 12 steps** (**1.053 -> 0.431 ms/step**, 13.38 ->
5.47 us per call, -59.1%) and the whole verifier at **36.760 -> 36.321
ms/step (-1.19%)** at an unchanged 985 calls/step, with the two bands not
overlapping. The leaves are **byte-identical to their generic counterparts at
rows 2/3/4/5/6/8**, so the section-6.1 gate is **identical to the last digit**
to the parent commit's run (mean/p95/p99/max KL **2.503e-05 / 1.654e-04 /
2.930e-04 / 2.971e-04**, top-1 **1.0000** overall and per category and per
budget). The paired suite follows: MTP B3 rises **49.371 -> 50.131 tok/s**
(UD-Q4_K_M, +1.54%) and **48.294 -> 48.634** (UD-Q4_K_S, +0.70%) while the
true-AR denominator stays flat (32.750 -> 32.705 and 32.098 -> 32.081), which is
what a verifier-only change must do. All four paired arms are `binding_passed`
with `timing_evidence_valid`. Rollback:
`HIPENGINE_GGUF_ROUNDED_NORM_FIXED5120=0`.
([artifact](results/2026-09-13-ud-gfx1100-rounded-norm-fixed5120.json),
[worklog](../worklog/entries/20260913T090609.044291Z-lhl-ud-rounded-norm-fixed5120-fa5016.md).)

The fixed-5120 norm leaf now reaches the verifier row slab. The kernel and both
of its registry keys already existed on this backend, but only the gfx1151
package declared `GGUF_NORM_RESIDUAL_DECODE_POLICIES` and its table admits
`rows == 1` alone, so every W7900 call fell through to the generic
local256 owner - which has a runtime-trip-count loop and no register cache, so
it reloads `x` and `add` for its second pass and pays nine tree barriers. That
covers the rows-3 verifier slab and the rows-1 decode alike. Both fixed-5120 entry
points now accept rows 1-8 and launch one block per row, and the gfx1100 package
declares the policy for rows 1-8 under both the plain and the
UD-preset-extended keys. One block owns one row, so every per-thread partial,
the reduction tree, the `rsqrtf` argument and the epilogue are unchanged, and
the leaves are **byte-identical to their generic counterparts at rows
1/2/3/4/6/8**. The verifier A/B puts the `add_rmsnorm` leaf at **0.770 ->
0.329 ms/step** (770.07 -> 328.56 us/step, 64 calls, **12.03 -> 5.13 us** per
call, -57.3%) and the whole verifier at **37.067 -> 36.6206 ms/step (-1.21%)** at
an unchanged 985 calls/step. Because the change is bit-identical, the movement it
produces is a pure speed effect: the same 64 norm launches per step at 16.9 ->
5.9 us each is about -0.7 ms on a 30.5 ms single-row step, and the paired suite
measures true-AR decode at
**31.904 -> 32.750 tok/s** (UD-Q4_K_M, +2.65%) and **31.303 -> 32.098**
(UD-Q4_K_S, +2.54%). The MTP B3 step is roughly twice as long, so the same
absolute saving is about half the percentage and sits inside the
acceptance-driven spread (UD-Q4_K_M 49.408 -> 49.371, UD-Q4_K_S 48.548 ->
48.294). The section-6.1 teacher-forced gate passes on both repeats at
mean/p95/p99/max KL **2.503e-05 / 1.654e-04 / 2.930e-04 / 2.971e-04** with top-1
**1.0000 overall and in every category and at every budget**, and all four
paired arms are `binding_passed` with `timing_evidence_valid`.
([artifact](results/2026-09-13-ud-gfx1100-norm-fixed5120-row-slab.json),
[worklog](../worklog/entries/20260913T073710.202482Z-lhl-ud-norm-fixed5120-rowslab-8c2632.md).)

The strict dense IQ row slab now covers the verifier's row count instead of
sitting below it. `gguf_iq_dense_strict_kernel` is the exactness-contract owner
for Q3_K/IQ3_S/IQ3_XXS/IQ2_S/IQ2_XS, and `grid.y` is `ceil(rows/R)`, so at the
3-row native verifier the old largest-slab-at-or-below rule picked `R=2` and
read every weight slice **twice**. The smallest slab at or above the row count
makes `grid.y` one. An interleaved A/B (three old-rule runs, two new-rule runs)
puts the strict family at **4.264 -> 3.179 ms/step (-25.4%)** and the whole
verifier at **38.275 -> 37.039 ms/step (-3.23%)**, with no overlap between the
two bands. The change is arithmetic-neutral by the kernel's declared contract -
`R` only changes which prompt rows share a block, padding rows load `0.0f` and
are never stored - and the section-6.1 gate confirms it directly by reporting
**identical numbers to the last digit** under both rules on UD-Q4_K_M
(mean/p95/p99/max **3.435e-05 / 2.112e-04 / 2.904e-04 / 2.952e-04**, top-1
1.0000, two deterministic repeats); UD-Q4_K_S passes at
1.871e-05 / 1.115e-04 / 3.704e-04 / 4.865e-04. The paired suite does not
separate this change from run-to-run spread (UD-Q4_K_M flat, UD-Q4_K_S +3.39%
against an AR denominator that moved +2.9%), so the retention rests on the
verifier sub-window A/B and the exact gate identity.
([artifact](results/2026-09-13-ud-gfx1100-iq-dense-strict-row-slab-cover.json),
[worklog](../worklog/entries/20260913T063400.608406Z-lhl-ud-iq-dense-row-slab-cover-5bb456.md).)

The Q5T16 single-wave route retires the four-wave WG128 geometry from the
verifier rows-2-8 rowtile. The four-wave owner runs four wave32 waves and sums
their partial vectors through shared memory; the single-wave owner runs one
wave32 per output block over eight columns with each lane owning eight
contiguous `k` inside the 256-element block, so the subblock `d`/`dmin` decode
hoists out of the inner loop and the block needs no cross-wave exchange or
`__syncthreads()`. On the verifier the Q5_K family falls **11.75 -> 7.98
ms/step (-32%)** across 126 in-window calls per step, at 1.05x-1.58x per call
on all six Q5_K shapes, and the whole-kernel step falls 40.9 -> 38.4 ms/step.
The paired arm measures UD-Q4_K_M MTP B3 46.192 -> **49.409** tok/s (+6.96%)
at an unchanged AR denominator (31.909 -> 31.904), so MTP/AR rises
1.4476 -> **1.5487x**. The plain control arms carry no Q5_K weights and move
+1.16%/+0.41%, which bounds the session spread. The four-wave entry point
stays registered as the parent-parity owner that the grouped rows6/rows8
variants are bit-identical to, and the single-wave owner is selected by policy
with `HIPENGINE_GGUF_Q5_T16_ROWTILE_SINGLE_WAVE=0` as the rollback.
([artifact](results/2026-09-13-ud-gfx1100-q5t16-single-wave-rowtile.json),
[worklog](../worklog/entries/20260913T041334.891199Z-lhl-ud-q5t16-single-wave-rowtile-06039f.md).)

The 2026-09-11 baseline this compares against measured UD-Q4_K_M 31.993 AR /
35.541 MTP B3 (1.1109x) and UD-Q4_K_S 31.311 / 33.061 (1.0559x), with plain
controls at 37.186 / 63.465 and 39.565 / 64.094. Across the runs since, both UD
AR arms have stayed within 0.3% and the plain controls within 3.2%, which
bounds the run-to-run spread.
([baseline artifact](results/paired-ud-plain-mtp-c1-natural25-b3-graph-xtx.json),
[invalidated eager-denominator attempt](results/paired-ud-plain-mtp-c1-natural25-b3-xtx.json).)

Earlier revisions of this section published a W7900/XTX census comparison
(46,913 vs 30,263 µs/token pure, localised to two local32 GEMV kernels at 3.84x
and 3.33x). **That is withdrawn**: it read a contended GPU0 census as a clean
device property.

Both artifacts pass the tokenized 18-prompt category/heldout screen
(`scripts/gguf_ud_combined_stack_gate.py --category-heldout`, real tokenizer
+ chat template, incumbent-extended 512-token prompts, 1170 teacher-forced
positions, seeds 7/11/23): K_S with the full shipped stack (max 5.1e-3,
top-1 99.83%); K_M with the Q3_K W4A16 prefill route pinned strict for
`layers.0.ffn_up` only - the per-tensor bisection showed that earliest Q3_K
tensor carries ~92% of the route's max-row KL tail (its 1-ULP
accumulation-order flips get the most downstream amplification), so pinning
it recovers the envelope (max 2.8e-2, p99 1.9e-3, top-1 99.74%) at ~3.4%
prefill cost. Everything else - IQ4_XS dual, coop64/coop32 owners, IQ2
additions, local32 decode - is bit-exact vs the incumbent once that slot
reverts. Next lever: a precision-preserving Q3_K owner (strict accumulation
association with shared decoded weights, or a higher-precision split
representation); cooperative geometry alone cannot help.
([gate artifacts](results/combined-stack-gate2/perslot-2026-09-11/),
[four-arm artifacts + bisection](results/final-four-arm-perslot-2026-09-11/),
[pre-pin record](results/final-four-arm-2026-09-10/).)

### W7900 Qwen3.8 `Q4_K_M` C1-C8

#### Server performance on the INT8 KV route

The server matches the direct engine on every axis we can measure, at a
fraction of the memory cost of the previous release:

| Metric (single request, 27B `Q4_K_M`, 16K context) | Server route | Direct engine |
| --- | ---: | ---: |
| Prefill throughput, 2K-8K-token prompts (tok/s) | 677-732 | 685-756 |
| Decode latency (ms/token) | 35.2 | 35.1 |
| Time to first token, 2K-token prompt (s) | 2.8 | - |
| Per-request transient memory at 32K context (GiB) | ~1.6 | - |
| Packed-prefill oracle peak, 2K-8K-token prompts (GiB) | -0.438 vs the previous default | — |
| Maximum context on 24 GB card (tokens) | 155,648 | — |

Prefill logits are identical between the routes. The context row reflects
complete prefill-and-decode request cycles at declared 65K/98K/131K/152K
contexts, each with a 2,048-row prompt (~2.9x the previous release's
54,272-token limit); the declared context is what was bracketed, not a
full-length prefill at that depth. Since 2026-09-11 the
default packed prefill is layer-outer, which shares one BF16 oracle pair per
session instead of one per INT8-retained full-attention layer: measured 0.438 GiB
lower tracked peak at 2K/4K/8K rows and 0.000 at 1K rows, where a single chunk
means there is no oracle to share, with identical generated IDs at 1K-8K rows.
On wall time the A/B shows no regression, but read that as one observed
comparison rather than a qualified parity result: the 1.5% reference is a single
spread from one same-code pair at 1,024 rows (where the executor does not
engage), each length was measured once on a shared host, and the memory figures
are the runtime's own tracked allocator peak rather than whole-device VRAM.
[Speed repetitions](results/2026-09-10-w7900-p7-c1-speed-parity-paired-reps.json),
[layer-outer promotion A/B](results/2026-09-11-w7900-p3-promotion-ab-layerouter-8192.json),
[server route check](results/2026-09-11-w7900-p3-promotion-server-c1-on.json),
[capacity bracket](results/2026-09-10-w7900-server-alloc-probe-152k.json),
[memory](results/2026-09-10-w7900-server-alloc-probe-16k-p4-lease-removed.json),
[streaming](results/2026-09-10-w7900-p7-http-transport-budget.json).

#### Several requests at once on the INT8 KV route

Until 2026-09-11 the INT8 KV route served C>1 as physical C1 with one serial
row per request, so extra concurrency bought no aggregate throughput. The
row-batched direct INT8 decode consumer is now qualified to four rows on this
artifact, and concurrent requests share one packed model step.

Measured with `scripts/gguf_server_cwidth_probe.py --widths 1,2,4` on the W7900:
each width runs N concurrent completions and then the same N requests one at a
time against the same server, and the ratio is concurrent aggregate
complete-request throughput over that measured serial rate.

| Concurrent requests | 1 | 2 | 4 |
| --- | ---: | ---: | ---: |
| Serial control (tok/s) | 428.9 | 424.9 | 419.3 |
| Concurrent (tok/s) | 426.5 | **529.8** | **597.0** |
| Concurrent / serial | 0.99x | **1.25x** | **1.42x** |

2048-token prompts and 64 generated tokens per request, greedy sampling, 16K
declared context, INT8 per-token/head KV with fp32 scales. The route counters
show the reason for the change: before promotion every decode step fell back to
serial rows (`packed_decode_width_unqualified`, 62/64 steps at C2/C4) and no
packed steps ran; after promotion there are zero serial fallbacks and 62/63
packed steps. Per-request latency still grows with width (4.95 s at C1, 7.97 s
at C2, 14.1 s at C4) and is reported separately from aggregate throughput.

[Promoted measurement](results/2026-09-11-w7900-ikv-c2-cwidth-promoted-c4.json),
[pre-promotion baseline](results/2026-09-11-w7900-ikv-c2-cwidth-baseline-pre-promotion.json),
[packed-transition correctness](results/2026-09-11-w7900-ikv-c2-packed-transition-gate.json),
[width-4 model gate](results/2026-09-11-w7900-ikv-c2-batch-decode-gate.json),
[kernel ownership trace](results/2026-09-11-w7900-ikv-c2-batch-decode-ownership-trace.json).

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

The C1 column of the K3 table and the single-request legacy-route
re-measurement (three balanced pairs: K3 39.70 vs 24.36 AR tok/s,
1.6329x median) both describe the legacy singleton target route. The
packed single-request target route is measured separately and is much
slower (below); the C2-C8 columns are unaffected.

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
[`dedicated campaign`](../docs/campaigns/QWEN38-GFX1100-C8-K3-CAMPAIGN.md).

On Strix Halo, Qwen3.8 `Q4_K_M` speculates at one active request with the
default candidate budget and bf16 KV, and declines to autoregressive decoding
above the speculative head's 1,023-token context window. The qualified C1/K3
rows below are the strict-profile measurements behind that admission. `Q4_K_S`
uses FP16
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

Aggregate prefill drops from c1 to c2 and then stays flat because one request
prefills its 512 rows in a single slab while wider groups split into slot-fair
bounded rounds against the 256-row prefill scratch. Tracked peak grows about
0.85 GiB per added request, so the c8 shape does not fit a 24 GB card.
c1 re-measured 2026-09-09 on current `main`: 680.4 / 29.7 tok/s, stdev
0.3% ([refresh artifact](results/2026-09-09-w7900-qwen38-q4km-c1-refresh.json)).
Single-request long-context ceilings on a 24 GB RX 7900 XTX are measured in
the root README's long-context section; no concurrent-request long-context
setting is qualified. The 2026-09-06
startup probe served one request at 3,072 context tokens and failed to start at
4,096, with no measured BF16-versus-INT8 difference, but it sampled memory
outside the prefill and decode peaks and never checked the live context length,
so those points are not a published ceiling. Rerun condition: a repaired
[`gguf_context_ceiling_probe.py`](../scripts/gguf_context_ceiling_probe.py)
([`capacity notes`](../docs/campaigns/QWEN38-27B-GFX1100-24GB-CAPACITY.md),
[`probe artifact`](results/2026-09-06-rx7900xtx-qwen38-c1-context-ceiling.json)).
Evidence: [`direct c1-c8 sweep`](results/2026-09-06-gfx1100-qwen38-q4km-direct-c1c8-sweep.json).

### Agentic quality (quality-only; no speed claim)

| Model | Overall | Development | Sealed heldout | Code / instruction / repository / tool | Valid calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B `UD-Q4_K_M` (reference) | 44/68 (64.71%) | 20/34 | 24/34 | 14/16 · 4/16 · 10/16 · 16/20 | 56/64 |
| Qwen3.8-27B `Q4_K_M` | **50/68 (73.53%)** | **22/34** | **28/34** | 14/16 · **12/16** · 10/16 · 14/20 | **64/64** |
| Ornith-1.5-35B-A3B `Q4_K_M` | 42/68 (61.76%) | 16/34 | 26/34 | 14/16 · 4/16 · 10/16 · 14/20 | 60/64 |

All repeat/control/ownership gates pass; failures are model-owned, so no
implementation is retained. [`Final`](results/2026-08-26-zbook-agentic-quality2-campaign-final.json).
Strix Halo Qwen3.8 `Q4_K_M` automatic MTP is restricted to its verified
strict/BF16/C1/B3/raw-greedy key; other scopes use K0/AR.
[`Serving closure`](results/2026-08-26-gfx1151-qwen38-q4km-mtp-serving-s5-closure.json).
Qwen3.8 `Q4_K_S` defaults to FP16 recurrent state with FP32 rollback; its exact
DMS sidecar remains default-off pending serving gates. [`DMS`](../docs/reference/DMS.md).

Agentic quality is quality-only: Qwen3.8-27B `Q4_K_M` scores **50/68 (73.53%)**
with 64/64 valid calls; no runtime mechanism is retained.
[`Final`](results/2026-08-26-zbook-agentic-quality2-campaign-final.json).
Generation-2 automatic serving remains K0: gfx1151 P9 is exact 540/540 but c2/c4
are 0.6975x/0.5843x AR; gfx1100 exact speculative cells remain behind direct.
[`Closure`](results/2026-08-26-gfx1151-specdec2-perf-campaign-closure.json) ·
[`Recovery`](../docs/campaigns/MTP-CONCURRENCY2-RECOVERY.md).

## Where detailed evidence lives

See result artifacts, [`CHANGELOG.md`](CHANGELOG.md), the
[`harness catalog`](HARNESSES.md), and [`BENCHMARK.md`](../docs/BENCHMARK.md).
Optimization history lives there, not in this current-row scoreboard.

Qwen3.8 runs plain AR on this backend; its speculative rows are retained for
explicit opt-in and re-measurement only.
## Benchmark harness catalog

Compare only matching harness scopes. ✓ marks a reported axis; blanks are
unmeasured. **AR/MTP/Prefill/Decode/Mem/Conc** mean true autoregressive,
speculative, prompt, generation, memory, and concurrency respectively. Use the
hermetic target-architecture wrapper; see `docs/BENCHMARK.md`.

| Harness (`scripts/`) | What it answers | AR | MTP | Prefill | Decode | Mem | Conc | Canonical entrypoint |
| --- | --- | :-: | :-: | :-: | :-: | :-: | :-: | --- |
| `qwen35_readme_sweep.py` | Single-request prefill/decode/memory per shape (llama-bench-style), one resident session, per-shape reset | ✓ | | ✓ | ✓ | ✓ | | `--engine gguf --model <model> --backend hip_gfx1151 --workloads 512/128 1K/128 ...` |
| `qwen4exp_canonical_ar_bench.py` | Exact-token Qwen4Exp cross-engine p512/p1024/p4096 prefill plus context-conditioned tg128, output hashes, and comparison artifact | ✓ | | ✓ | ✓ | | | `hipengine --model-root <model>` or `llamacpp --server-bin <binary> --model <part1>` |
| `qwen4exp_profile_gap.py` | Exact-fixture prefill ROCTX roles, launch/copy/allocation census, lifecycle, and separate selected-expert telemetry | | | ✓ | | ✓ | | `--mode prefill --case-id code-p512 --profile --role-markers` |
| `qwen4exp_context_decode_profile.py` | Restored exact live-context transition roles, complete mutable-state hashes, per-bucket lifecycle, and allocation census | ✓ | | | ✓ | ✓ | | `--live-count 513 1025 4097 --repetitions 3 --profile --role-markers` |
| `qwen4exp_llamacpp_exact_profile.py` | Exact-token pinned llama.cpp prefill and cached single-transition decode under direct rocprof, selected by monotonic bounds | ✓ | | ✓ | ✓ | | | `--case-id code-p512 --case-id code-p1024 --case-id code-p4096` |
| `qwen4exp_mtp_head_profile.py` | Isolated Qwen4Exp MTP full-Q8 draft/head/D2H timing and selected-head Amdahl ceiling | | ✓ | | ✓ | ✓ | | `--output <json>` |
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

Keep this catalog synchronized whenever a harness gains a measured axis.

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

## Qwen3.8-Flash-Next on Framework (gfx1151)

Fresh combined-default Framework `gfx1151` baseline, UD-Q4_K_XL/BF16 KV,
four categories, p512/p1024/p4096 + tg128:

| Engine | p512 PP / TG | p1024 PP / TG | p4096 PP / TG |
| --- | ---: | ---: | ---: |
| hipEngine | 153.96 / 19.38 | 152.30 / 18.75 | 142.03 / 14.42 |
| halo-box Vulkan | 316.28 / 25.27 | 391.68 / 25.30 | 425.72 / 24.51 |
| halo-box HIP | 282.76 / 21.08 | 368.33 / 20.56 | 351.08 / 18.83 |

Weighted tok/s,36 samples per engine, all outputs repeat, clean teardown.
All lanes exceed2% per-case CV somewhere; these are sequential screening
comparisons, not statistical parity.
[Frozen baseline evidence](results/2026-09-05-framework-qwen4exp-refreshed-baselines.json).

## Qwen3.8-27B UD short-context diagnostic

On zbook / Radeon 8060S, both published UD files match 162/162 next-token
choices against tokenwise llama.cpp HIP using the same artifact, 18
category/heldout prompts (39–71 tokens) and nine forced logit positions per
prompt. Both engines process the prompt serially in this comparison.

| UD file | Default mean/max KL vs llama.cpp | Q5/Q6 candidate allocated weight bytes | Status |
| --- | ---: | ---: | --- |
| `UD-Q4_K_M` | 0.000192 / 0.002185 | 26,396,502,016 → 21,526,849,536 (−18.45%) | Diagnostic only |
| `UD-Q4_K_S` | 0.000230 / 0.002833 | 21,125,912,576 → 18,599,622,656 (−11.96%) | Diagnostic only |

The candidate changes Q5/Q6 residency, matches all 162 baseline top-1 choices
per file, and is not enabled by default. Maximum tokenwise-teacher KL is
0.002724 (K_M) and 0.004469 (K_S). Batched llama.cpp instead matches 161/162
choices per file, so these results depend on the execution schedule; they do
not establish batch invariance. Weight-buffer sums exclude load peak, scratch
and KV. No speed, BF16-model quality, long-context or serving qualification is
claimed. [Tokenwise protocol and category results](results/2026-09-07-zbook-ud-tokenwise-teacher.json);
[batched-teacher protocol and memory results](results/2026-09-07-zbook-ud-c1-residency-logits-diagnostic.json).

### Raw IQ dense prefill (gfx1151)

The published UD files reached none of hipEngine's optimized kernels: a
model-wide veto stripped the repacked layouts from every rank-2 tensor whenever
any raw-IQ tensor was present, so 0% of their weight bytes were in a layout the
optimized families can consume, against 94.8% for the plain `Q4_K_S` file of the
same model. Making that veto per-tensor, widening the dense T16 role coverage,
and routing raw IQ prefill through the integer-MMQ kernel took `UD-Q4_K_M`
prefill from 22.1 to 150.3 tok/s on the published sweep protocol (512/128,
zbook / Radeon 8060S), choosing the route that is admissible under the
production envelope over a faster one that is not.

| Stage | Optimized weight bytes | Prefill | Decode |
| --- | ---: | ---: | ---: |
| Start | 0.0% | 22.1 tok/s | 6.21 tok/s |
| Per-tensor repack eligibility | 44.2% | 29.9 | 7.24 |
| Dense T16 role coverage | 61.3% | 41.5 | 7.62 |
| Dense IQ integer MMQ | 61.3% | 127.2 | 7.63 |
| Four-quant integer MMQ | 61.3% | 171.9 | 7.71 |
| Four-quant W4A16 (current) | 61.3% | **150.3** | 7.69 |

The same host and protocol run the plain `Qwen3.8-27B-Q4_K_S` file at 294.8
tok/s, so the gap between the two files narrowed from 13.3x to 1.96x.

**Accuracy is ranked against hipEngine strict, not against an external engine.**
`docs/EXECUTION-PROFILES.md` section 6 defines the production gate as strict
versus candidate production on the same artifact, and section 7.1 is explicit
that another engine is a comparison oracle rather than the definition of strict
bytes. Scored that way, over 162 teacher-forced rows:

| Route | Mean | p95 | p99 | Max | Rows over the 5e-2 ceiling |
| --- | ---: | ---: | ---: | ---: | ---: |
| Two-quant MMQ | 0.001038 | 0.003479 | 0.014385 | 0.049730 | 0 |
| Four-quant MMQ (current) | 0.002110 | 0.006797 | 0.024044 | **0.170390** | **1** |
| Four-quant W4A16 | **0.000827** | 0.004547 | 0.012475 | **0.023513** | 0 |

The integer-MMQ route **exceeds the binding 5e-2 absolute maximum-row ceiling
at one position**, so it is not admissible under that envelope. **W4A16 is
therefore the default**, passing every threshold at a 12% throughput cost; the
integer route stays registered. See
[`gate reference`](results/2026-09-09-zbook-ud-production-gate-reference.json).

Every rate here is gross, from a power- and thermal-limited laptop whose
plain-file result is 294.8 tok/s against a published 396.1 for the same
model/quant on a desktop part. Treat the absolute numbers as provisional
pending that re-measure; the ratios and the accuracy ranking are same-host.

The older raw-IQ2_XS residency experiment is recorded in
[benchmark history](HISTORY.md#ud-raw-iq2-xs-residency-diagnostic).

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
0.005930**, with **72/72** graph trajectories exact. [`Campaign`](../docs/campaigns/QWEN35-08B-GFX1151-VULKAN-PARITY.md).

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
the ten-prompt gate is exact. Qwen3.8 W7900 detail is in the direct engine
c1-c8 table above.

### Radeon 8060S: Qwen3.8-27B Dense GGUF

For standard `Q4_K_M` with BF16 KV, short 33-48-token prefills use the
row48 gate/up kernel. Six paired same-host measurements improve complete
prefill throughput by **9.2-9.6%** versus the previous row64 kernel;
all 18 category and heldout prompt trajectories match exactly.
[Short-prefill evidence](results/2026-09-12-gfx1151-qwen38-row48-prefill-retained.json).

Qwen3.8 uses `Q4_K_S` with BF16 K/V. A byte-identical `Q_K_M`-derived Q5T16
route feeds prefill; decode runs the retained `Q4_K_S` path. hipEngine
prefill and decode both beat the llama.cpp backends at every working shape
(4K needs 2.30 GiB more memory to do so):

| Shape | Clean prefill | Clean AR | Retained process GTT | Lower valid llama GTT |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **396.091** | **13.069** | **15.275 GiB** | 15.785 GiB |
| 1K/128 | **387.648** | **12.894** | **15.710 GiB** | 15.816 GiB |
| 4K/128 | **380.305** | **13.038** | **17.863 GiB** | 16.004 GiB |

Speculative decode and single-request decode versus the same GGUF on llama.cpp:

| Mode | hipEngine | llama HIP | llama Vulkan |
| --- | ---: | ---: | ---: |
| True AR decode (tok/s) | 13.36641 | 5.53853 | 7.51888 |
| Exact native B3 spec-decode (tok/s, speedup) | 23.85263 (1.7845x) | — | — |
| Process memory (GiB) | 15.899 | 16.358 | — |
Evidence: [`clean Q4_K_S`](results/2026-08-16-gfx1151-qwen38-27b-q4ks-clean-publication.json),
[`Q5 source-F16 prefill retention`](results/2026-08-17-gfx1151-qwen38-27b-q4ks-q5-source-f16-prefill-retention.json)
(the source of the prefill and AR columns above),
[`G6 closure`](results/2026-08-17-gfx1151-qwen38-27b-q4ks-g6-closure.json)
(the source of the true-AR and B3 rows above),
[`exact B3`](results/2026-08-17-gfx1151-qwen38-27b-q4ks-exact-native-b3.json)
(the earlier task-23 retention measurement, 24.19347 tok/s at 1.82281x on a
source that predates `fix: make Qwen3.8 Q4_K_S native B3 exact`),
[`memory package`](results/2026-08-17-gfx1151-qwen38-27b-q4ks-memory-parity-retained.json), and the
[`campaign plan`](../docs/campaigns/QWEN38-27B-GFX1151-CAMPAIGN.md).

The standard (non-UD) `Q4_K_M` file is a separate lane from `Q4_K_S`.
Current measurements use the public production profile, FP32 recurrent state,
BF16 KV, two hardware queues, and a prepared 8,192-token session. Each shape
has one discarded warmup and three measured resets; decode uses the
backend-selected HIP graph path.

| Shape | Prefill | AR decode | Tracked peak |
| --- | ---: | ---: | ---: |
| 512/128 | **404.474 tok/s** | **12.150 tok/s** | 37.428 GiB |
| 1K/128 | **397.164 tok/s** | **11.843 tok/s** | 37.428 GiB |
| 4K/128 | **376.558 tok/s** | **12.118 tok/s** | 37.428 GiB |

Every prefill/decode CV is below 0.40%. All timed final logits are finite,
and the graph/eager preflight matches every generated ID, final full logits,
and state fingerprint on 18 category/heldout prompts. The public packed
numerical gate covers 8,716 rows at KL0/top-1 100%; real-socket blocking/SSE,
cancellation, refill and clean-drain checks pass. Tracked peak includes
public session pools and the graph preflight; session-owned peak is 19.810 GiB.
These memory scopes differ from the older right-sized low-level-session
measurements. The numerical/serving packet above is September 12 evidence;
the September 20 refresh reruns graph/eager correctness, not that full packet.
[Current profile measurements](results/2026-09-20-gfx1151-v060-headline-refresh.json).

The following earlier September 12 same-file comparison used the then-current
low-level hipEngine session. Both engines ran back to back on one host with
BF16 K/V. It is a historical comparator, not a fresh comparison against the
public-profile measurements above:

| Shape | hipEngine prefill | llama.cpp HIP prefill | hipEngine AR | llama.cpp HIP AR |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | 380.132 tok/s | **383.191 tok/s** | 12.222 tok/s | **12.230 tok/s** |
| 1K/128 | 378.444 tok/s | **385.116 tok/s** | 11.989 tok/s | **12.137 tok/s** |
| 4K/128 | 361.622 tok/s | **374.692 tok/s** | **12.141 tok/s** | 11.592 tok/s |

In that earlier comparison hipEngine and llama.cpp HIP were within about
3.5%, with neither ahead everywhere. HIP led prefill, while hipEngine led
4K decode. The Vulkan comparison likewise belongs to that older session:
hipEngine led short prefill but trailed Vulkan decode by 4.1-6.4%.
Those differences must not be read as results for the refreshed profile.

The llama.cpp columns above come from `llama-bench`, which generates its own
prompts and cannot take the repeated-token prompt hipEngine uses, so that tier
is a split-timing comparator rather than a token-exact one. A stricter
explicit-token-array tier on the same file, gated on a uniform token hash, puts
hipEngine prefill 7.9% ahead of llama.cpp HIP at 512/128 and 3.8% ahead at
1K/128 but 1.7% behind at 4K/128, with decode 0.6% behind at 512/128, 0.6%
behind at 1K/128, and 5.5% ahead at 4K/128. The tier changes the prefill
verdict, so both are recorded. hipEngine `Tracked peak` counts hipEngine
allocator ownership and the llama.cpp peak is an external whole-process GTT
delta; the two are different scopes and are not comparable as a memory result.
[`same-file llama.cpp comparator`](results/2026-09-12-gfx1151-qwen38-27b-q4km-same-file-llama-comparator.json).

Two dense Q4T16 rowtile variants that this backend previously inherited from
`gfx1100` only as exclusions are now admitted: the narrow col4 rowtile and the
fused down+residual rowtile. Both are bit-identical to the retained owners at
every reachable shape and row band, and each is a small win (**1.005-1.257x**
and **1.007-1.011x** per call over 21 counterbalanced pairs per cell). They are
reachable only at rows 2-4, and together they cover about 0.15% of decode, so no
number in the tables above moves.
[`dense rowtile qualification`](results/2026-09-12-gfx1151-qwen38-27b-q4km-dense-rowtile-withheld-variants-qualified.json).

#### Long-context speculation

Speculative decoding on this file was bounded twice. The draft adapter refuses
MTP at its 1,023-token context window, and the verifier independently dropped
its 48 dense linear-attention layers to a per-row scalar route once the span
crossed the 1,024-token split-K threshold, which cost more than the drafts
saved. The verifier's bound is gone: the batched staged chain is byte-exact
against the serial-exact teacher on the eager route at every straddle band
measured - 19 cases at cycle ends 1,020-1,028, 13 at 3,528-3,532 and 3 at 8,192 -
with the staged owner confirmed by route counters (48 staged calls, 0 row-wise
calls per case), and the matched row-wise control is exact as well (13 of 13 at
3,528-3,532 with 192 row-wise calls). It is now the default on `gfx1151` for
plain `Q4_K_M`-stamped artifacts; other file types, preset (UD) artifacts and
`gfx1100` keep the row-wise strict route, which stays reachable everywhere
through `HIPENGINE_GGUF_STAGED_LINEAR_ROWS_LONG=0`.

One server process per arm, a true no-MTP control selected by the request's own
`speculative_mtp` field, BF16 KV, candidate budget 3, and the adapter's context
window raised to 8,192 tokens (see below):

| Prompt tokens | True AR | MTP, staged route | MTP, row-wise route |
| ---: | ---: | ---: | ---: |
| 672 | 11.99 | **24.22 (2.019x)** | 24.20 (2.016x) |
| 877 | 11.93 | **16.33 (1.369x)** | 16.33 (1.366x) |
| 3,055 | 11.21 | **18.25 (1.630x)** | 8.00 (0.713x) |

The first two shapes stay below the split threshold, where both routes run the
same arithmetic, so they are the control that the arms differ only in route. At
3,055 tokens the staged route is **2.28x** the row-wise route and turns a loss
into a win. The full mtp-bench category suite and the category heldouts at the
same long shape pass their true-AR identity gate - **10 of 10** canonical prompts
across `code`/`general_en`/`general_ja`/`mixed_ja_en` and **8 of 8** heldouts
match the AR arm's generated ids token for token over the 32-token probe - at
**1.42x-2.55x (median 1.864x)** and **1.35x-2.21x (median 1.828x)** respectively,
with 98.4-99.2% of output tokens from speculative cycles. The adapter's 1,023-token window is a
separate, still-unpromoted bound: it refuses long-context speculation by
default. The 1,023-token MTP admission window that used to refuse them was removed on 2026-09-20, so these long-shape rows now reach the speculative route directly.
[Evidence](results/2026-09-19-gfx1151-qwen38-staged-linear-rows-long-verifier-route.json).

#### ShareGPT serving at one, four and eight concurrent requests

A real workload through the OpenAI server: vLLM's ShareGPT loader and pruning
criteria, the second turn as the expected length, capped at 128 output tokens,
greedy decoding, default server configuration (BF16 K/V, 16,384-token context,
automatic speculative MTP). `vllm bench serve --backend openai-chat` drove the
load and a recorder read each response's `hipengine` block, so the routing
column is measured per request rather than inferred from the rate.

| Concurrent requests | Output tok/s | Mean TPOT | Mean TTFT | Requests that ran MTP |
| --- | ---: | ---: | ---: | ---: |
| 1 | 20.69 | 48.7 ms | 3.0 ms | 24 of 24 (99.1% of tokens) |
| 4 | **28.39** | 140.2 ms | 118.7 ms | 5 of 24 |
| 8 | 27.23 | 277.0 ms | 552.7 ms | 11 of 24 |

Speculation carries the whole single-request workload and about a fifth to a
half of the concurrent ones. The rest run autoregressive decoding for a reason
recorded per request: the serving evidence for this cell covers a one-request
realized group, so a request admitted while it is alone can be batched into a
wider decode group where the route is not yet qualified, and a group mixing rows
with and without a draft provider fails closed. Concurrent requests therefore add
throughput without adding speculation, and the aggregate rate is the number to
read - a per-window rate from a client that cannot keep up with the stream is a
delivery burst, not decode.

At this 128-token budget most replies are still inside a thinking block when the
budget ends (20 of 24 requests emitted no answer text), so the table measures
decode throughput on a real prompt distribution, not answer quality. Time to the
first generated token is 847 ms at one request and 1,150 ms on a cold server,
against the 5,993 ms median time to the first *answer* token after the reasoning
block. [Routing artifact](results/2026-09-17-gfx1151-qwen38-sharegpt-mtp-routing.json).

##### Attributing the split, not just adding it up

`ar_output_tokens` is defined as `completion_tokens - mtp_output_tokens`, so a
response whose split adds up has proved nothing about which tokens came from
which execution mode. The response therefore also reports the committed cycle
records that bound the speculative output
(`mtp_output_tokens_explained_by_cycles`, `unexplained_mtp_output_tokens`) and
names every failed check in `reconciled_reasons`, and with
`HIPENGINE_MTP2_OUTPUT_SPANS=1` the backend records one committed output span per
speculative cycle and per autoregressive step - execution mode, planner reason,
emitted-token position, token count - which must tile the committed output.

The same c=1 protocol with spans on:

| Quantity | Value |
| --- | ---: |
| Requests / failures | 24 / 0 |
| MTP output share | 99.06% |
| Committed spans | 1,046 |
| Spanned tokens (all speculative output) | 3,067 (3,043 MTP + 24 AR) |
| Unspanned tokens (AR emitted outside the plan) | 5 |
| Completion tokens | 3,072 |
| Refusal events (`no_provider`, one per request) | 24 |
| Requests with a complete attribution proof | 24 of 24 |

The unspanned tokens are exactly `ar_output_tokens - ar_output_tokens_in_cycles`
(5 = 29 - 24) and no speculative token is unspanned, so the split is traceable to
committed work. Note that the 24 refusal events are *events*: the same requests
emitted 29 autoregressive tokens, 24 from the depth-zero `no_provider` cycle that
opens each request and 5 from the decode that retires it. Event counts and token
counts are reported under separate names for that reason.
[Attribution artifact](results/2026-09-17-gfx1151-qwen38-mtp-attribution-span-proof.json).

### Radeon 8060S: Qwen3.6-35B-A3B GGUF

September 20 C1 measurements use two hardware queues, graph decode, and one
4,226-token low-level BF16 session for both shapes. Each shape has one
discarded warmup and three measurements; all 18 category/heldout graph/eager
checks pass. The longer rows below remain the August 6 one-queue measurements.
The separate packed c2/c4 byte-exact gate fails on the refreshed tree; these
C1 timings do not qualify cross-batch arithmetic.

| Workload | Prefill | Decode | Tracked peak | Whole-device GTT peak |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **1418.062 tok/s** | **56.736 tok/s** | 21.453 GiB | Not sampled |
| 4K/128 | **1462.354 tok/s** | **56.340 tok/s** | 21.453 GiB | Not sampled |
| 32K/128 | **1144.713 tok/s** | **46.405 tok/s** | 21.597 GiB | 22.152 GiB |
| 64K/128 | **936.218 tok/s** | **40.180 tok/s** | 22.336 GiB | 22.890 GiB |
| 128K/128 | — | — | — | — |

Repeated 128K remains blocked by the documented later-pass lifecycle stall; no
numeric 128K row is carried forward. Evidence:
[September 20 C1 refresh](results/2026-09-20-gfx1151-v060-headline-refresh.json);
[`SH14-C1 completion gate`](results/2026-08-06-gfx1151-gguf-sh14-c1-cumulative-completion-gate.json).

### Laguna S 2.1

| Platform / format | Workload | Prefill | Decode | Evidence |
| --- | --- | ---: | ---: | --- |
| W7900 / `UD-Q2_K_XL` | 4096 prompt, prefill only | **440.893 tok/s** | — | [`H8B production`](results/2026-08-03-gfx1100-laguna-q2-xl-scoped-activation-pack-reuse-production.json) |
| Radeon 8060S / `Q4_K_M` | 512/128 | **654.249 tok/s** | **23.221 tok/s** | [`prefill production`](results/2026-07-27-gfx1151-laguna-attention-packed-query-producer-candidate.json), [`decode production`](results/2026-08-01-gfx1151-laguna-registry-resolution-cache-retained.json) |

The W7900 Laguna decode campaign and rejected H7/H8 ladders are implementation
history, not scoreboard content; follow the production artifact, changelog, and
[`docs/archive/LAGUNA-PARITY-STATUS.md`](../docs/archive/LAGUNA-PARITY-STATUS.md).

Explicit gfx1151 Laguna DFlash remains non-default and uses the tile1 target
verifier. The attempted tile4 transfer was trajectory-identical to tile1 but
failed the shared full-suite true-AR gate and did not improve complete E2E wall;
see the [`tile4 rejection`](results/2026-08-20-gfx1151-laguna-dflash-iq3-tile4-rejected.json).

### Radeon 8060S: VibeVoice-TTS 1.5B session

The generation session on the pinned single-speaker request: a 121-token prompt
that generates 27 tokens, 25 diffusion frames at 20 solver steps and CFG 1.3,
producing 3.333 s of audio at 24 kHz. `RTF` is wall time over that output
duration. The torch lane is the same request, on the same host, in the pinned
torch oracle venv.

| Lane | Pooled RTF | Warm | Time to first audio | LM | Diffusion | Decode | Semantic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hipEngine | **1.201** | **4.005 s** | 0.373 s | 1.450 s | 1.298 s | 0.301 s | **0.781 s** |
| torch (oracle venv) | 1.336 | 4.454 s | — | — | — | — | — |

Both lanes are timed on **the same request with the same random operands**. The
torch lane previously drew its own voice-prompt VAE noise and its own diffusion
noise; it now serves the fixture's recorded operands (`encode_draw0`,
`encode_draw1` and one `callN_initial_noise` per diffusion call), matched by shape
so an unrecognised draw falls through to the normal RNG. The cold run consumes
exactly the recorded set — 1, 1 and 25 draws with nothing unmatched — and
reproduces the fixture's step-0 `eps` to `max_abs_diff` 0.0, so the two rows
describe one request rather than two draws from the same distribution.

Four hipEngine runs of this protocol measured pooled RTF 1.315, 1.333, 1.337 and
1.355 on this host (median 1.337) before the low-row WMMA column tile landed, which
takes the semantic stage from 1.108 s to 0.781 s and pooled RTF to 1.201. Three torch
runs measured 1.325, 1.336 and 1.336 (median 1.336), so the HIP lane is now faster on
this request. Decode is stable across the hipEngine runs (0.301-0.327 s).

A later verification re-run of the same protocol at HEAD measured pooled RTF **1.228**
(warm 4.092 s, diffusion 1.230 s, semantic 0.837 s), so the retained 1.201 is a
favorable run and the post-tile margin over torch is **~9%** rather than the ~11% the
headline alone implies. The HIP lane is still faster on this request, but read 1.201
as the top of the observed band, not as a reproducible figure. Artifact:
[session verification](results/2026-09-15-gfx1151-vibevoice-tts-session-verify.json).
Generation budget comes from the request's declared `max_audio_seconds` (4.333 s,
a 35-token cap) in both lanes, and both reach EOS at 27 tokens.

The diffusion solver now runs on device. `sample_speech_tokens` keeps the latent,
the CFG-combined eps and the previous x0 device-resident for the whole 20-step
schedule, so the per-step eps readback is gone. In a same-session interleaved A/B
on this request (one process, one machine state, arm order device/host/device/host)
the diffusion stage falls from 1.521 s to 1.295 s, **-14.9%**, and warm synthesis
from 8.515 s to 8.216 s, **-3.5%**. The generated chain is unchanged: the device
loop reproduces the host loop's per-step `eps`, CFG-combined eps and post-step
latent bit-for-bit over all 20 steps, and the 10-seed generated-audio quality suite
returns the identical 58/60 with the same two failing request-runs.

Those A/B runs were taken at host load average 13.3, so their absolute RTF (~2.5)
is roughly twice the recorded 1.201 and is not comparable to it. Only the
same-session device/host delta is quoted; the absolute headline above is unchanged.
Artifact: [device solver loop](results/2026-09-16-gfx1151-vibevoice-tts-device-solver-loop.json).

The diffusion head caches its invariant work. The timestep MLP is keyed by the
bf16-rounded timestep (the 20-step schedule repeats across the 25 frames), the
condition projection is skipped when the condition bytes are unchanged, and
`silu(c)` is hoisted above the layer loop. Forcing the two weight-read caches to
miss in a same-session interleaved A/B (one process, one machine state, arm order
cached/uncached/cached/uncached) takes the diffusion stage from 1.237 s to 1.086 s,
**-12.2%** (1.139x), and pooled RTF from 1.256 to 1.213, **-3.4%**. The uncached arm
disables the two weight-read caches only; the `silu(c)` hoist is active in both arms,
so the delta is a lower bound on the full change. That run was on an **idle** host --
the cached arm's RTF of 1.204/1.222 sits in the retained 1.201 band -- so unlike the
device-solver A/B above, these absolute figures are comparable to the headline. The
two arms produce bit-identical per-step `eps` and `speech` over all 25 recorded calls
(0 of 500 steps differ, max absolute difference 0.0), and every frozen
fixture-parity margin is unchanged (pre-onset eps 0.00847 against its 0.025 gate,
whole-trajectory eps 0.0393 against 0.08). Artifact:
[invariant cache](results/2026-09-16-gfx1151-vibevoice-tts-diffusion-invariant-cache.json).

#### Diffusion arithmetic review (2026-09-16)

Production arithmetic and the timing figures above are unchanged. The
oracle-injected two-speaker branch diagnostic now explicitly selects the
256-thread reduction used when its 0.15 bound was calibrated, with a fresh head
so caches cannot carry production arithmetic into the reference check.
The production 64-thread path measures **0.15524** in that diagnostic; this
change does not reduce that drift or claim it satisfies the reference bound.
Default-path per-step numerical checks and unassisted audio-quality checks remain
separate gates.

| Diffusion reduction | Reference diagnostic | Generated-audio finding |
| --- | --- | --- |
| 64 threads (production, unchanged) | 0.15524 | Existing 58/60 baseline |
| 128 threads (rejected) | Passes 0.15 bound | New `single-numbers` failure; Whisper WER 8.3% -> 25% |
| 256-order shared-row candidate (rejected) | Passes 0.15 bound | 58/60 hides an additional intelligibility failure; Whisper WER 0% -> 12.5% |

The 128-thread run was stopped after 33 of 60 generated requests were scored,
once the new regression was independently confirmed. The 256-order run scored
all 60. Neither candidate is a production optimization or a quality-neutral fix.
[128-thread rejection](results/2026-09-16-gfx1151-vibevoice-tts-diffusion-128-rejected.json),
[256-order rejection and reproduction sources](results/2026-09-16-gfx1151-vibevoice-tts-diffusion-reduction-repair.json),
[256-order per-request quality](results/2026-09-16-gfx1151-vibevoice-tts-reduction-repair-quality.json),
[generic-schedule diagnostic](results/2026-09-16-gfx1151-vibevoice-tts-diffusion-schedules.json).

### Radeon 8060S: VibeVoice-TTS 1.5B generated-audio quality

Six held-out scripts (one to four turns, 7 to 40 words) over the same two voice
sets, run with nothing injected: the session draws its own voice-prompt VAE
noise, its own diffusion noise and its own negative-LM branch, and each request's
generation budget comes from its declared `max_audio_seconds` rather than from the
oracle's token count. Ten seeds per request, 60 request-runs in total.

| Request | Turns | WER (10 seeds) | Duration ratio | Speaker runs |
| --- | ---: | --- | ---: | --- |
| `single-short` | 1 | 0.0 | 0.76-1.14 | n/a |
| `single-medium` | 1 | 0.0 | 0.67-1.04 | n/a |
| `single-numbers` | 1 | 0.0-0.083 | 0.83-1.03 | n/a |
| `two-2turn` | 2 | 0.0-0.118, 1 above gate | 0.72-1.12 | `[0,1]` 10/10 |
| `two-4turn` | 4 | 0.0-0.043 | 0.94-1.20 | `[0,1,0,1]` 10/10 |
| `two-long` | 2 | 0.025-0.075 | 0.68-0.88 | `[0,1]` 9/10 |

58 of 60 request-runs pass. Every request reaches EOS, no output clips, the longest
internal gap is under 1 s, and no repeated 4-gram appears. Every chunk boundary is
checked for a sample-count discontinuity and for a click: across all 60 runs the
chunk sizes sum to the output length, every chunk is a whole 3200-sample codec
frame, and the largest sample-to-sample step at a boundary stays below 0.64x the
signal's own interior 99.9th-percentile step.

The two failures are both at the edge of their gates and neither is truncation or
silence. `two-2turn` on one seed measures word error rate 0.118 against a 0.10
gate, where its other nine seeds measure 0.0 to 0.059; the extra error is a single
word. `two-long` on one seed misassigns one 1 s window inside the first turn, so
the encoder-based attribution sees a spurious speaker flip; its other nine seeds
reproduce `[0,1]` cleanly, and the transcript on that seed is unaffected.

This suite is the gate that rejects the semantic encoder's one-row GEMV dispatch.
That dispatch is 3.00x faster on the stage in isolation (55.22 -> 18.42 ms per
chunk) and would take pooled RTF from 1.337 to 1.146, but over the same ten seeds
it drops the suite to 54 of 60 and reproduces a word-error-rate 0.647 runaway with
repeated speech and an 11.73 s output where the script expects about 7 s. The
perturbation at the boundary is roughly one bf16 ulp, so this is trajectory
sensitivity rather than an arithmetic defect, and neither the correctness suite
nor the exact-token-chain gate detects it. The dispatch is not landed.

Speaker attribution is measured on the acoustic encoder's own window assignment,
not on the ASR's speaker labels, which report a single speaker for two-speaker
scripts that contain both voices. The four-turn script reproduces its whole
`[0,1,0,1]` sequence on all ten seeds, so the check covers the intermediate
speaker changes rather than only the endpoints. The instrument does **not**
resolve absolute voice identity: classifying a single-voice request against both
references sits at the noise floor (mean cosines 0.424 against 0.413 for one
request) and assigns every window to the wrong reference for another, so identity
is recorded per request as a diagnostic and is not gated.

WER is measured by `microsoft/VibeVoice-ASR-HF` running through hipEngine itself, so
the suite also transcribes the reference implementation's own PCM for the pinned
requests and records that as an **evaluator floor**. It measures 0.0 on both, which
bounds the ASR lane's systematic error but does not make every non-zero WER
tributable to the TTS lane: the evaluator can still mishear audio whose phonetics
differ from the reference. Separating those cases needs an independent ASR on the
same waveform, which this suite does not run. A cross-check with
`openai/whisper-large-v3-turbo` on the same waveforms showed the largest failure
this suite has recorded (a two-speaker first turn at 0.353 WER on one seed, before
the decoder change below) was a genuine synthesis failure — whisper dropped that turn
entirely rather than substituting words — while the smaller 0.059-0.118 values are
within the evaluator's own reach on this lane's audio.

Evidence: [quality suite](results/2026-09-15-gfx1151-vibevoice-tts-quality-suite.json).

### Radeon 8060S: YuE2 3B AR replay

The autoregressive stage of the pinned `m-a-p/YuE2-3B` checkpoint over the frozen
18-case replay matrix: 128/512/2048-token prefixes, CFG on and off, unequal-prefix
branches, scored against the pinned upstream oracle on the same host. Pooled over
112 recorded full-vocabulary rows and 576 decode-step rows. `Prefill total` is the
wall time to prefill all 18 prefixes, so it includes every case's 4-10 rows.

| Prefill route | Mean KL | Max KL | Top-1 | Top-8 recall | Prefill total | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `strict` (row-by-row) | 1.43e-3 | 3.27e-2 | **95.83%** | 95.88% | 522.5 s | [`strict replay`](results/yue2_ar_replay_strict_20260916.json) |
| `hipblaslt` (batched, default) | **1.16e-3** | **7.10e-3** | 95.49% | **95.88%** | **77.6 s** | [`batched replay`](results/yue2_m7_gemm_algorithm_20260917.json) |

Re-running the batched route after the paired-branch lm head landed
(`dense_gemv_bf16_f32_out_rowtile2` plus the runtime's paired `logits`) reproduces
these numbers: pooled mean KL 1.1579e-3 over the same 576 rows and 18 cases, top-8
recall 0.9588, status pass, and a `--repeat 1` run whose aggregate is byte-identical
([`paired head`](results/yue2_ar_replay_paired_head_20260917.json),
[`repeat`](results/yue2_ar_replay_paired_head_repeat_20260917.json)). That change is
bit-identical per row to the single-row head, so the replay matrix is expected to be
unchanged and is; the session-level validation is in the greedy session gate below.

Both routes replay the first case bit-for-bit after all 18 cases have run on the
same runtime, and both stay inside the broad floor (mean KL ≤ 0.05, top-1 ≥ 90%).
The batched route selects its hipBLASLt algorithm by measured index rather than by
hipBLASLt's own first entry (`scripts/hipblaslt_algo_scan.py`); that choice took
its pooled mean KL from 2.11e-3 to 1.16e-3, its worst case's max KL from 9.10e-2
(`abc-nocfg-L128-s5678`, 5 rows) to 7.10e-3, and its prefill total from 91.3 s to
77.6 s, at a top-1 of 95.49% against the strict route's 95.83%. The strict arm was
not re-run in the session that measured the 77.6 s, so only the batched route's
own 91.3 s → 77.6 s is a same-protocol pair; the earlier 522.5 s → 91.3 s pair came
from one session and is why the strict route's absolute wall time should be read as
a band rather than a fixed figure.

### Radeon 8060S: YuE2 3B greedy session gate

The torch-free staged session (`Yue2ArSession`) generating freely at temperature
zero from the pinned `m-a-p/YuE2-3B` checkpoint, against the pinned upstream
oracle's own greedy trajectories on the same host. `Prefix` is the assembled
positive-branch prefix checked against the reference's own ABC IDs, so a
trajectory divergence cannot be read as a prompt-assembly bug. `Forced` is the
per-step greedy agreement when the runtime is driven along the reference's
tokens, so both sides score the same context at every step; `Free` is the
unassisted trajectory's first divergence from the reference, and the semantic
phase is budget-truncated in every case.

| Case | Prefix | ABC forced | Semantic forced | Free first divergence | Truncation (ABC / semantic) | Evidence |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `english-melody-s1234` | identical | 99.7% | 97.1% | 27 | natural / budget | [`session gate`](results/yue2_session_gate_paired_head_20260917.json) |
| `english-full-s1234` | identical | 99.1% | 98.2% | 2 | natural / budget | [`session gate`](results/yue2_session_gate_paired_head_20260917.json) |
| `mandarin-off-s1234` | identical | no ABC stage | 96.7% | 23 | no ABC stage / budget | [`session gate`](results/yue2_session_gate_paired_head_20260917.json) |

Teacher-forced agreement is at least 96.7% on every phase of every case; the symbolic ABC phases agree on 99.7% and 99.1% of steps. Every recorded mismatch sits on a near-tie: the first semantic mismatch is 1.2e-01 / 1.2e-01 / 6.2e-02 below my own top-1 score for melody / full / off, inside the BF16 logit noise the replay matrix above already quantifies (mean KL 1.2e-3, top-1 95.5%). The unassisted trajectories track the reference until their first flip and are chaotic afterwards, so the forced columns, not the free agreement rates, carry the fidelity claim. Free comparison is context-aligned only where the ABC stage itself matched (`english-melody-s1234`) or is absent (`mandarin-off-s1234`); `english-full-s1234` diverged during ABC planning, so its free semantic row compares continuations of different prefixes and is diagnostic only. Unassisted wall clock: `melody` 34.8 s, `full` 36.7 s, `off` 29.0 s.

These numbers are the post-M7 state. The `english-melody-s1234` ABC phase was 100.0% and `mandarin-off-s1234` semantic was 96.1% before the hipBLASLt algorithm-selection and attention changes landed; every case then moved to the values above, which is recorded as the `session_gate_after_change` block of [`the M7 artifact`](results/yue2_m7_gemm_algorithm_20260917.json). Re-running the gate after the paired-branch lm head landed reproduced those post-M7 values exactly (99.7% / 99.1% ABC, 98.2% / 97.1% / 96.7% semantic), so that change is invisible at the session level.

The paired-branch head itself is validated on the session loop rather than on the gate alone: [`pairing validation`](results/yue2_ar_pairing_validation_20260917.json) drives `generate_tokens` at temperature 1.0 on two cases with a serial-head subclass as the control and compares the emitted tokens, the RNG state, the CFG score rows, the sampler's masked score row and the softmax distribution. All agree exactly (logit max abs 0.0, distribution KL 0.0, identical top-1, identical RNG state), and a lifecycle pass that runs case A, then case B, then case A again in one process with a `runtime.reset()` between reproduces each case's tokens exactly.

### Radeon 8060S: YuE2 3B acoustic flow-matching solver

The torch-free NAR runtime's midpoint ODE solve against the pinned upstream
reference on the same host, on a recorded 34-row / 545-token-AR chunk with a
4-step schedule. `Forced` evaluates the native velocity at each *recorded*
state, which isolates a velocity defect from a trajectory defect; the end-to-end
row runs the native solver from the recorded noise and compares its own latents.
Every row is relative L2 against the reference, whose final latent norm is
72.743.

| Quantity | rel_l2 | cosine | max abs | Evidence |
| --- | ---: | ---: | ---: | --- |
| Forced velocity, step 0 (`t=1.0`) | 0.01327 | 0.999912 | 0.0859 | [`NAR gate`](results/yue2_nar_gate_20260916.json) |
| Forced velocity, step 1 (`t=0.75`) | 0.01495 | 0.999888 | 0.0625 | [`NAR gate`](results/yue2_nar_gate_20260916.json) |
| Forced velocity, step 2 (`t=0.5`) | 0.01892 | 0.999821 | 0.1218 | [`NAR gate`](results/yue2_nar_gate_20260916.json) |
| Forced velocity, step 3 (`t=0.25`) | 0.01221 | 0.999926 | 0.0781 | [`NAR gate`](results/yue2_nar_gate_20260916.json) |
| 4-step solve, final latents | **0.01225** | **0.999926** | 0.0781 | [`NAR gate`](results/yue2_nar_gate_20260916.json) |

The solver is the reference's own scheme: `mid = state - v(state, t) * h/2`, then
`state = state - v(mid, t - h/2) * h`, with each update rounded through BF16
exactly as the reference does. The 1.2% residual is BF16 rounding noise, not
structural drift - 16.2% of the produced latent entries are BF16-identical to
the reference and the produced norm is 72.855 against 72.743. Condition 1.00 s
and forced-velocity plus solve 0.80 s on this chunk.

Two further recorded cases cover the rest of the solver's contract. A song that
crosses a real chunk boundary (96 frames at a reduced chunking context, so the
chunks are 26 / 26 / 26 / 18 frames) and a restricted-visibility case
(`nar_cond_end = 128`, where the NAR sees only the first 128 AR keys):

| Case | Solved latents rel_l2 | cosine | Worst forced velocity | Evidence |
| --- | ---: | ---: | ---: | --- |
| `nar_cond_end = 128` | 0.00885 | 0.999962 | 0.01443 | [`cond_end gate`](results/yue2_nar_gate_condend_20260916.json) |
| Multi-chunk, chunk 0 (26 frames) | 0.00878 | 0.999964 | 0.02141 | [`multi-chunk gate`](results/yue2_nar_gate_multichunk_20260916.json) |
| Multi-chunk, chunk 1 (26 frames) | 0.01026 | 0.999948 | 0.02141 | [`multi-chunk gate`](results/yue2_nar_gate_multichunk_20260916.json) |
| Multi-chunk, chunk 2 (26 frames) | 0.00876 | 0.999963 | 0.02141 | [`multi-chunk gate`](results/yue2_nar_gate_multichunk_20260916.json) |
| Multi-chunk, chunk 3 (18 frames) | 0.01071 | 0.999944 | 0.02141 | [`multi-chunk gate`](results/yue2_nar_gate_multichunk_20260916.json) |

These cases are why the gate exists: they exposed two defects that the single
release chunk could not. The solver's midpoint buffer was only partially
re-initialized, so a chunk shorter than the one before it evaluated its midpoint
against stale boundary rows (0.16667 before the fix, 0.01071 after), and the
hipBLASLt problem cache was built only for the largest chunk's tile, so a shorter
later chunk launched an unprepared row count. Both are fixed and both cases pass
every chunk.

### Radeon 8060S: YuE2 3B VAE decoder

The torch-free FP32 Oobleck decoder against the pinned upstream reference on the
same host. Weight normalization is folded once at load time; the conv,
transposed-conv and SnakeBeta kernels accumulate in FP32 with the reference's own
loop order. Error is measured against the reference waveform, whose own tiled and
full decodes already differ by 1.33e-06 - so these figures sit at the reference's
own FP32 noise floor rather than at a modelling difference.

| Case | Samples | max abs | rel L2 | Peak | Full decode | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 3-frame latent | 5 696 | 1.27e-06 | 1.99e-06 | 0.483541 | 0.04 s | [`VAE gate`](results/yue2_vae_gate_20260916.json) |
| 64-frame latent | 122 816 | 2.99e-06 | 1.88e-06 | 0.991600 | 0.68 s | [`VAE gate`](results/yue2_vae_gate_20260916.json) |
| 64-frame latent, tiled 16/16 | 122 816 | 2.79e-06 | - | - | 1.51 s | [`VAE gate`](results/yue2_vae_gate_20260916.json) |

Tiling is exact rather than approximate: with a 16-frame core and 16 frames of
halo the tiled waveform is **bit-identical** to the same runtime's full decode,
and it matches the reference's own tiled output to 2.79e-06. There is no
crossfade, no boundary smoothing, and no zero padding of the final audio; the
last tile's crop is shorter than its own output because the natural length is
`1920 * frames - 64`. The decoder's own dependency interval gives a required halo
of 12 frames for a 16-frame core, which the gate asserts against the recorded
value. Tiled decoding is slower than the full decode here (1.51 s vs 0.68 s at
64 frames) because every tile re-decodes its halo context; its purpose is
bounding device residency, not wall clock.

### Radeon 8060S: YuE2 3B end-to-end product path

The complete staged pipeline - AR plan and semantic decode, the acoustic
flow-matching solve, and the FP32 Oobleck decode - driven through the registered
`yue2` model plugin with no torch in the process. Each committed case fixture
stores the reference pipeline's own assembled prefix and semantic codes, so the
gate replays the reference's exact conditioning and checks the product path
against the audio the reference produced for it: every case reproduces the
recorded latent frame count and the recorded sample count exactly.

| Case | Latent frames | Samples | Peak | Replay |
| --- | ---: | ---: | ---: | ---: |
| `english-full-s1234` | 1,871 | 3,592,256 | 0.9246 | 301.8 s |
| `english-full-s5678` | 1,704 | 3,271,616 | 1.0354 | 252.8 s |
| `english-melody-s1234` | 2,061 | 3,957,056 | 1.0009 | 356.8 s |
| `english-melody-s5678` | 1,499 | 2,878,016 | 0.8217 | 202.9 s |
| `english-off-s1234` | 1,517 | 2,912,576 | 0.9596 | 188.9 s |
| `english-off-s5678` | 2,192 | 4,208,576 | 1.0196 | 370.4 s |
| `mandarin-full-s1234` | 1,920 | 3,686,336 | 0.9781 | 356.0 s |
| `mandarin-full-s5678` | 1,692 | 3,248,576 | 0.8506 | 287.7 s |
| `mandarin-melody-s1234` | 1,807 | 3,469,376 | 0.8589 | 307.3 s |
| `mandarin-melody-s5678` | 1,943 | 3,730,496 | 0.8815 | 354.0 s |
| `mandarin-off-s1234` | 1,297 | 2,490,176 | 0.9463 | 148.4 s |
| `mandarin-off-s5678` | 1,497 | 2,874,176 | 0.7696 | 193.2 s |

Across all twelve cases that is **21,000 latent frames** and **40,319,232
samples** of audio, all finite and all non-silent, at 4 ODE steps rather than the
product's 32 (the replay takes 55 min, from
148 s for the shortest case to 370 s for
the longest; the reference's own recorded runs take 150-370 s per case). Solver
arithmetic parity is the M4 gate's claim, so step count does not affect what this
one measures. Replaying `english-full-s1234` twice gives **bit-identical**
latents. The per-case replay times in the table are the figures this gate produced
at the revision that established the frame and sample counts; the same twelve cases
at 2 ODE steps replay in **527.4 s** in total with the attention and projection work
that followed.

Measured against the reference's own recorded runs on the same host, per stage,
across all twelve cases at 2 ODE steps:

| Stage | hipEngine | Reference | Ratio |
| --- | ---: | ---: | ---: |
| FP32 Oobleck decode | 331.76 s | 2 049.23 s | **6.18x faster** |
| Acoustic solver, raw | 126.92 s | 537.52 s | 2.81x faster (fewer steps) |
| Acoustic solver, per ODE step | 63.46 s | 16.80 s | **3.78x slower** |

Both rows above compare against the reference's *recorded product* runs, and the
decode row is not a decode comparison: on this host the reference's first decode in a
process costs 8x its steady state (163.1 s against 19.6 s for the same 1 297 frames),
and its recorded product decode carries that first-use cost plus a per-request decoder
transfer. The matched comparison is in "YuE2 3B stage-by-stage matched timing" below.
The solver row is not like for like either, because the reference always solves at its
own 32 steps; only the per-step row is matched, and that row was **26.29x** before the
attention and projection work described in the next section, **5.79x** after the scalar
attention work, and **3.78x** now. `scripts/yue2_reference_comparison.py` produces both
tables. Evidence:
[`comparison`](results/yue2_reference_comparison_geometry_20260917.json),
[`previous geometry`](results/yue2_reference_comparison_wmma_20260917.json),
[`scalar attention`](results/yue2_reference_comparison_20260917.json),
[`pre-fix comparison`](results/yue2_reference_comparison_pre_attention_fix_20260917.json).

A separate live run exercises the whole session rather than recorded
conditioning: a greedy request plans and decodes 96 semantic
tokens, solves them to 96 latent frames and decodes
3.84 s of audio in 16.4 s. Running it
twice returns identical latent and audio identities. `torch` is not imported by
either run; the gate fails if it ever is. `scripts/yue2_e2e_gate.py` is the
reproducing command. Evidence:
[`e2e replay`](results/yue2_e2e_gate_20260916.json),
[`e2e live`](results/yue2_e2e_live_gate_20260916.json).

The stage comparison above replays recorded conditioning, so it never times the AR
stage. `scripts/yue2_case_timing.py` drives one recorded case through the complete
product path at the product's 32 ODE steps - plan, semantic decode, solve, decode -
and prints each stage against the reference's own recorded timing for that request
(`mandarin-off-s1234`). Per unit of work, which is the like-for-like reading:

| Stage | hipEngine | Reference | Ratio |
| --- | ---: | ---: | ---: |
| AR semantic decode | 57.3 ms per token | 31.3 ms per token | **1.83x slower** |
| Acoustic solver, 32 steps | 22.3 ms per frame | 21.1 ms per frame | 1.06x slower |
| FP32 Oobleck decode | 16.3 ms per frame | 53.3 ms per frame | **3.27x faster** |

Those are medians of three runs at current HEAD. The harness caps the semantic phase at
the reference recording's own token count, so both sides now do the same work: 1 298
frames against the reference's 1 297, and 51.92 s of audio against 51.88 s. The wall
clocks are therefore comparable for the first time, and they say the product path is
**ahead**:

| Whole path, identical request and work | Elapsed | Audio | RTF |
| --- | ---: | ---: | ---: |
| hipEngine at current HEAD (median of 3) | 124.54 s | 51.92 s | **2.40** |
| hipEngine with the pre-change AR paths (median of 3) | 155.22 s | 51.92 s | 2.99 |
| torch reference, pinned product run | 137.80 s | 51.88 s | 2.66 |
| torch reference, fresh product runs on this host | 235.60-260.19 s | 51.88 s | 4.54-5.02 |

The two hipEngine arms differ only in the AR paths, and every run of both arms produced
the same 1 298 tokens, the same 2 492 096 samples and the same PCG64 state digest. The
paired elapsed ratio is 1.236 (range 1.213-1.246 over the three pairs), so the AR change
is worth **19% of whole-path time**; the AR stage itself went from 79.2 to 57.3 ms per
token, a paired ratio of 1.380. The NAR and VAE rows move together across the three
repetitions in both arms (about +20% and +10% over the session), which is machine drift,
not an arm effect - the arms track each other, so the paired ratios are the stable
quantity and the absolute seconds are not.

The reference's own product path is not reproducible here. Two fresh runs today take
235.60 s and 260.19 s against its pinned 137.80 s, and the entire difference is its VAE
stage: 170.09 s and 186.35 s against the pinned 69.09 s, while its AR (41.96 and 43.55 s
against 40.60 s) and NAR (20.16 and 26.76 s against 27.32 s) agree within noise. Its warm
tiled decode of these same latents is 20.15 s, measured in the same session as everything
else above, so even its pinned product run pays 3.5x its own warm decoder and today's runs
pay 8.5x. That is its per-request decoder handling rather than the machine. All three
reference rows are reported as measured, and the pinned one is what the per-stage table
compares against, because its stages are the ones its recorded timings describe.

So the product path is ahead of the reference in both states - 124.54 s against 137.80 s
pinned, and against 235.60-260.19 s today - for a reason that is worth stating precisely:
this runtime's decoder stays warm and pays 16.3 ms per frame, while the reference's
re-does per-request decoder setup. Its steady-state stages are still the faster ones, which
is why the matched table above is the implementation comparison and this table is the
product one.

The AR row is the entire remaining gap, and the stage is 60% of our elapsed time. A
fixed-token comparison of the two AR decode paths pins it down, with both sides driving
the same recorded trajectory for the same 1 297 steps over the same 98-token positive and
12-token negative prefixes and the same two CFG branches, timed alternately in one
session (`scripts/yue2_ar_matched_timing.py`, `scripts/yue2_reference_ar_timing.py`):
**54.94 ms per decode step against 30.88**, so the reference is **1.78x faster** on
matched work. Sampling is excluded there by design, because the token is recorded.

The cause is traffic, and it is now measured on both sides. A branch reads ~3.575 GB per
step, of which 2.819 GB is the 28 layers of q/k/v/o/gate/up/down and 756 MB the output
head, from the checkpoint's own tensor sizes. This path forwards the two CFG branches
separately, so it reads the layers twice, and it now reads only the semantic phase's
window of the head (134 MB rather than 756 MB): **5.77 GB per token at 57.3 ms, about
101 GB/s**. The reference batches both branches into one forward per step
(`GraphAR(model, [prefix, negative], ...)` in its `yue2/sampling.py`), so it reads the
layers once: **3.58 GB per token at 30.9 ms, about 116 GB/s**. The two rates are within
15% of each other, so the stage is not at a bandwidth ceiling - this host's LPDDR5X peak
is ~256 GB/s - and what separates them is that we move 1.6x the bytes. Removing the
duplicated branch traffic would put the same rate at ~2.95 GB per token, or about 29 ms
- the reference's own step - but that is a projection from measured traffic, not a
measurement. Two-row kernels are the route: the head already has one
(`dense_gemv_bf16_f32_out_rowtile2`, 112 GB/s to 199 GB/s at the production shape), and
the layer GEMVs are the same family, documented at 20-28% of peak on this host
(`scripts/gguf_q8_0_dense_bw_microbench.py`).

Host sampling is the smaller lever and it is now small: **7.67 ms per token over the
full row against 0.71 ms windowed** (`scripts/yue2_ar_sampling_cost.py`, CPU-only,
single-threaded, 30 medians per call; the window is `semantic`'s 32 769 rows of 184 704,
and the windowed arm is what the loop runs). The full-row figure is 2.60 ms of CFG
combine, 3.89 ms of repetition penalty plus top-k/top-p, 0.22 ms of softmax and 0.94 ms
of the categorical draw. So the product's 57.3 ms per token is model work plus about a
millisecond of sampling, and the step itself is the lever - not the sampler.
Evidence: [`matched AR timing`](results/yue2_ar_matched_timing_20260917.json),
[`AR sampling cost`](results/yue2_ar_sampling_cost_20260917.json),
[`product case timing`](results/yue2_product_case_timing_20260917.json).

### Radeon 8060S: YuE2 3B stage-by-stage matched timing

Every earlier stage comparison in this file put one side's warm measurement against the
other side's recorded product run, which mixes in per-request overheads and, for the
decoder, an 8x first-use cost. These three harnesses give both sides identical inputs
and compare steady-state passes, timed alternately in one session on
`mandarin-off-s1234` at the product's 32 ODE steps:

| Stage | Inputs shared | hipEngine | torch reference | Reading |
| --- | --- | ---: | ---: | --- |
| AR decode, 1 297 steps | recorded token trajectory, both prefixes, 2 CFG branches | 54.94 ms/step | 30.88 ms/step | **reference 1.78x faster** |
| Acoustic solver, 32 steps | prefix, codes, seed and the reference's own noise | 27.35 s | 21.81 s | **reference 1.25x faster** |
| FP32 Oobleck decode, 1 297 frames | the case's recorded latents, tiled 1024/16 on both sides | 20.08 s | 20.15 s | reference 1.00x faster (level) |

Every row above is a same-session re-measurement of both sides at current HEAD, on
`mandarin-off-s1234` at the product's 32 ODE steps. The AR row is the fixed-token replay:
both sides drive the same recorded trajectory, so sampling is excluded by design (the
token is recorded) and the number is model work only. The replay uses the full-vocabulary
head, which is what the reference's own rows are, so the windowed head and the
window-relative sampler do not appear in it; the product loop below is where they show up.

The AR step was 58.30 ms/step before the paired-branch head (see the next section): the lm
head runs once for both branches through a two-row F32-output GEMV, which is bit-identical
per row, leaves the run's trajectory digest unchanged, and took the step to 54.88, against
54.94 on the re-measurement above.

Each harness refuses to report when the two sides did not run the same work: the AR
harness compares prefix, negative-prefix and trajectory digests, and the solver harness
compares the initial-noise digest after being handed the reference's own whole-song
draw (the two implementations' seeded draws differ by design, and the first version of
this comparison silently compared two different trajectories). The solver's latent norm
is 287.400 against 287.113 on that shared noise, which is the arithmetic-parity
cross-check the M4 gate makes on recorded fixtures. Our tiled decode is bit-identical to
our own full decode (max abs 0.0) while the reference's differ by 1.4e-06, its own FP32
noise floor.

So on matched, steady-state work this runtime is behind on two stages and level on the
third. What the product path does differently is overhead: the reference moves its
decoder off the device after every request and back on for the next one, and its
recorded product decode of these same latents is 69.09 s against its own warm 19.60 s,
while this runtime pays about 8 ms per token of host sampling that the reference does
with device ops. Evidence:
[`matched AR timing`](results/yue2_ar_matched_timing_20260917.json),
[`matched solver timing`](results/yue2_nar_matched_timing_20260917.json),
[`matched decoder timing`](results/yue2_vae_matched_timing_20260917.json).

### Radeon 8060S: YuE2 3B paired-branch projections

The AR decode runs two CFG branches over one set of weights, and the profile of a step is
79% projection GEMVs. The lm head alone streams 756 MB of BF16 weight per call, once per
branch. `dense_gemv_bf16_f32_out_rowtile2` puts two rows in one block so the weight is
issued once for both branches, keeping each row's K traversal and reduction order, which
makes a row bit-identical to the single-row kernel.

| Shape (in x out) | Serial, 2 calls | Two-row, 1 call | Speedup | Serial GB/s | Two-row GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| LM head 2048 x 184704 | 6.723 ms | **3.798 ms** | 1.77x | 112.5 | 199.2 |
| q 2048 x 2048 | 0.060 ms | 0.033 ms | 1.78x | 140.7 | 250.6 |
| o 2048 x 2048 | 0.054 ms | 0.031 ms | 1.77x | 154.3 | 272.7 |
| v 2048 x 1024 | 0.041 ms | 0.028 ms | 1.46x | 102.7 | 150.3 |
| k 2048 x 1024 | 0.028 ms | 0.022 ms | 1.30x | 149.0 | 193.1 |

Per-call latency with a synchronize per call, best of 30, after 5 warmups. In the runtime
only the head is paired so far, which is where the traffic is: the step goes from
**58.30 ms to 54.88 ms** (1.06x) on `mandarin-off-s1234` with the trajectory digest
unchanged at `6a308700b3b949ac`, so the paired head reproduces the serial head exactly.
Each branch keeps its own KV spans and its own positions (this case's positive prefix is
98 tokens against the negative branch's 12); only the weight read is shared. Evidence:
[`kernel A/B`](results/yue2_ar_rowtile_bench_20260917.json),
[`step timing`](results/yue2_ar_matched_timing_20260917.json),
gate `tests/test_unit_yue2_ar_gemv_rowtile2.py`.

### Radeon 8060S: YuE2 3B phase-windowed output head and sampler

`distribution` masks every row outside a phase's domain and the phase's end token to
`-inf` before any other arithmetic, so a phase can only ever select inside its own
window: 32 769 rows for `semantic` (the 32 768 codec tokens plus `MUSIC_END`, 17.7% of the
vocabulary) and 151 849 for `abc`. The head projects just that slice, and the sampler runs
in window coordinates, so mask, repetition penalty, top-k, top-p, softmax and the
draw all work over the window instead of the full vocabulary.

| Phase | Head, full 184 704 rows | Head, windowed | Speedup | Rows | Weight read |
| --- | ---: | ---: | ---: | ---: | ---: |
| `semantic` | 3.84 ms | **0.82 ms** | **4.68x** | 32 769 | 722 MiB → 128 MiB |
| `abc` | 4.13 ms | 3.44 ms | 1.20x | 151 849 | 722 MiB → 593 MiB |

Both head rows are medians of 15 interleaved pairs, with the two arms measured inside one
repetition and the order flipped each repetition; the paired ratio spans 4.41-5.01 for
`semantic` and 1.18-1.39 for `abc`. A separate interleaved run gave 4.53x and 1.20x.

| Sampler stage, `semantic` | Full row | Window | Speedup |
| --- | ---: | ---: | ---: |
| mask, penalty, top-k, top-p, softmax, draw | 5.86 ms | **0.43 ms** | **13.5x** |
| the same for `abc` (82.2% of the vocabulary) | 3.37 ms | 2.56 ms | 1.32x |

The production session loop with two-branch CFG, temperature 1.0, top-p 0.95, top-k 100,
repetition penalty 1.2 over a 50-token window on `mandarin-off-s1234`, three arms measured
inside one repetition with the order rotated, 5 repetitions per arm:

| Arm | ms/token | Saved vs baseline |
| --- | ---: | ---: |
| full head, full-row sampler | 61.11 | — |
| windowed head, full-row sampler | 55.14 | 5.97 ms (1.11x) |
| windowed head, windowed sampler | **43.80** | **17.32 ms (1.40x)** |

A second interleaved run measured the same comparison as 60.84 → 43.75 ms/token, 17.10 ms
saved, within 1.3% of the first. The saving exceeds the two head calls' own 6.0 ms because
the device-to-host row and the host-side conversion shrink from 739 KB to 131 KB per
branch as well. All 15 runs produced **identical tokens and an identical PCG64 state
digest**.

The windowed row is bit-identical to the full projection inside the window and `-inf`
outside it for both phases, and the windowed sampler returns bit-identical scores and
softmax probabilities to the full-row sampler, with the same top-1 and the same drawn
token, over 432 cases (both phases x 3 sampling settings x both arithmetic modes) on rows
recorded from the pinned reference. Windowed execution is covered by those checks and by
the session gate below; the replay matrix keeps using the unwindowed call, so it validates
that fallback and the paired head rather than the window itself. Evidence:
[`head window validation`](results/yue2_ar_head_window_20260917.json),
[`sampler window validation`](results/yue2_sampler_window_20260917.json),
[`session gate`](results/yue2_session_gate_windowed_sampler_20260917.json).

### Radeon 8060S: YuE2 3B NAR attention packing and projection selection

The NAR attention kernel walked the key/value cache once per (query row, query
head) pair, so a 1 299-frame song re-read 27.9 GB of K/V per call. Blocks now own
eight query rows (`YUE2_NAR_ROWS_PER_BLOCK`) and reuse every K and V element they
load across those rows; the K row and the query row are read eight and four
elements per instruction instead of one, which is what a lane owning one key had
been asking the L1 for 32 sectors at a time; and the V walk steps by pointer
instead of multiplying and branching per key. `rocprofv3 --pmc` on the previous
kernel measures why the packing mattered more than the traffic: 3.5e8 cycles
carried 3.7e9 VALU instructions at **19% occupancy**, so it was stalled rather
than bandwidth-bound, and the same run after the change reaches **93%**. The
projection path had a second, larger defect: both YuE2 runtimes took the first
hipBLASLt algorithm hipBLASLt returned, and that entry is 2.9-4.2× slower on these
shapes than the measured best, which the same wrapper already documented as its
fast heuristic. Both now select by measured index.

| Measurement | Before | After | Ratio |
| --- | ---: | ---: | ---: |
| `yue2_nar_attention_kernel`, 1 299 rows / 2 695 keys, per call | 177.22 ms | **19.33 ms** | **9.17x** |
| The same kernel, tile maximum vectorized (paired, same session) | 32.82 ms | **31.34 ms** | **1.05x** |
| The same kernel, softmax weights stored key-major (paired, same session) | 27.40 ms | **26.38 ms** | **1.04x** |
| The same kernel, dot-product row loop specialized and walks unrolled (paired) | 26.27 ms | **19.33 ms** | **1.36x** |
| `yue2_nar_attention_wmma_kernel`, same shape, per call | 20.1 ms | **3.85 ms** | **5.1x** |
| The same kernel, block widened to 384 threads and the key batch to 32 (paired) | 3.85 ms | **2.13 ms** | **1.84x** |
| NAR projections, one 2-step solve, 800 GEMMs | 3.89 s | **0.94 s** | **4.14x** |
| NAR solve of `mandarin-off-s1234`, 2 ODE steps | 27.76 s | **5.14 s** | **5.40x** |
| NAR solve of the same case, product 32 ODE steps | 443.4 s | **26.07 s** | **17.01x** |
| Full replay of the same case, 32 steps (solve + decode) | 465.9 s | **46.16 s** | **10.09x** |

All rows are gfx1151 (zbook) measurements. The kernel row is a median of six
batches of five calls; every replay row was run on the same host with the gate's
own JSON as the source, and the 32-step pair uses the protocol the reference's own
27.32 s figure was measured under, which puts the solver 16.2× behind the pinned
upstream before this work and **1.05× ahead** of it now. Eight rows per block is a
measured optimum: four rows lands at 22.53 ms, sixteen at 25.39 ms and thirty-two
at 88.50 ms, where shared-memory and register pressure take over.

Per-call times on this host drift about 10% between batches — the same kernel
source measures 27.4-28.5 ms in one batch and 32.0-33.4 ms in another — so the last
three rows are quoted as paired same-session ratios, alternating the two sources
within one batch. What they change: the per-tile maximum walked the tile's
shared-memory scores one float at a time, 128 requests per row in a single
dependent `fmax` chain, and it now reads four entries per instruction into four
independent accumulators; the softmax weights were stored row-major, so one key's
weights for eight rows cost eight index computations against the runtime block size
and eight shared-memory requests, and they are now stored key-major, eight
consecutive floats per key; and the row loop inside the dot product was rolled at 46
instructions per (row, eight dimensions) against eight multiply-adds, because its
bound is only known at run time — a full row set now takes an unrolled path, and
both accumulation walks carry running pointers and a 32-bit trip count. All three
changes are bit-exact — a maximum over a set is exact and order-independent for
finite scores, the sums still walk each row key-ascending, and the dot products keep
their order — and all three are confirmed against the recorded parent-bits fixture.
The third is the largest: it removes **38% of the kernel's instructions**
(12 581 to 7 816 per lane-tile, with vector ALU down 36%, scalar ALU 45% and global
loads 64%), measured with `rocprofv3 --pmc`, which is deterministic where the wall
clock is not.

What is left is a kernel-design change rather than more loop surgery. Profiling the
pinned upstream on the same case puts its own attention kernel at **4.14 s** for the
whole 32-step solve (1 792 calls at 2.31 ms, 12.4 TFLOP/s); this kernel's share is
about 39 s, so it is **9.4×** off a tensor-core flash attention. The scalar FP32
ceiling on 40 CUs is 25.6 TFLOP/s and this kernel keeps roughly a quarter of its
instructions as useful arithmetic after the softmax reductions, the weight exchange
and the loads, so even a perfect scalar kernel lands near 6 TFLOP/s — about 8.5 s
here. Closing the rest needs matrix cores, which is what the reference uses.

`yue2_nar_attention_wmma_kernel` is that tensor-core path: a second attention for the
production head geometry (16 query heads over 8 key/value heads at head_dim 128) with
f16 WMMA operands, f32 score accumulation, a 16-key tile per lane-half reduction, K
and V sharing one staged buffer, and an f16 output accumulator rescaled by the
online softmax — the same arithmetic class as the reference. Its fragment contracts
follow the in-tree Laguna flash attention, which preserves llama.cpp's
`fattn-mma-f16` design. It runs the production shape in **2.13 ms per call against
the scalar kernel's 19.0 ms (9.3×)**, faster than the reference's own 2.31 ms
dispatch, and takes the product's 32-step solve to **26.07 s against the reference's
27.32 s (1.05× faster)**. It changes arithmetic by design, so the scalar kernel stays
registered as the strict fallback behind `HIPENGINE_YUE2_NAR_ATTENTION=scalar` and
this variant is held to the production gates instead of the parent-bits fixture: all
three M4 solver gates pass (chunk0 latents rel L2 0.01099 / cosine 0.999941,
restricted visibility 0.00894 / 0.999961, multi-chunk 0.01090 / 0.999943 against
0.05 / 0.999), the full twelve-case replay passes with every case reproducing its
exact frame and sample counts in 458.7 s against 527.4 s, and its own unit test
checks seven shapes against an independent FP64 reference. Evidence:
[`M7 tensor-core attention`](results/yue2_m7_attention_wmma_20260917.json),
[`M7 block geometry`](results/yue2_m7_attention_wmma_geometry_20260917.json).

The geometry is where most of that came from, and the measurement that found it was
an instruction count rather than a timing. At the original 128-thread block the
kernel was at 7.5 TFLOP/s, 15% of the device's fp16 peak, while the reference's
kernel reached 12.4. A static ISA count of the built object showed 2 902
instructions per key batch of which only 64 are WMMA, and 1 312 waves over 40 CUs ×
4 SIMD against 11.2M cycles puts it at roughly **11 cycles per instruction** —
latency-bound at two waves per SIMD, not issue-bound or WMMA-bound. Three
instruction-level attempts came back as measured no-ops: vectorizing the K-fragment
load produced a **byte-identical ISA** (the compiler had already done it) and
tightening `__launch_bounds__` to 3 and 4 blocks per CU changed nothing, because the
shared-memory footprint already limits the block count to one per CU. Widening the
block from 4 waves to 12 (384 threads, 96 rows) with a 32-key batch took the kernel
to **2.13 ms** while leaving the lane-to-column mapping untouched. The lane mapping
is what made that a small change: 16 columns per wave means 8 query rows × 2 query
heads, so the fragment code is independent of how many waves the block has.

At 32 steps the solver is now ~5.3 s of attention, ~14.7 s of projections and ~6 s of
conditioning prefill and other work, so the projections are the largest single item
and run at hipBLASLt's rate.

`scripts/yue2_solver_stage_profile.py` measures that split directly rather than by
differencing two solve lengths. It times one velocity evaluation with HIP events and
splits it by suppressing one launch at a time, so the differences stay pipeline-clean;
inserting a synchronize around each kernel does not work, because with 200 projection
launches per evaluation the sync cost alone exceeds the kernel time being measured. The
result for the production case at 28 layers, 1 299 rows and 2 695 keys:

| Stage | Per evaluation (old geometry) | Per evaluation (now) | At 32 steps | Share |
| --- | ---: | ---: | ---: | ---: |
| Projections (200 launches, 3 662 GFLOP) | 229 ms | 229 ms | **14.7 s** | 56% |
| NAR attention (28 launches) | 136 ms | 83 ms | **5.3 s** | 20% |
| Conditioning prefill (paid once) | — | — | **3.44 s** | 13% |
| Everything else (norms, RoPE, casts, state) | 53 ms | 53 ms | **3.4 s** | 13% |

The projection and attention columns are the same measurement at the two attention
geometries; the solve total is 26.07 s.

The projections are the remaining lever, and they are close to a ceiling. They run at
16.0 TFLOP/s where the same shapes launched back to back by hipBLASLt measure
19.4 TFLOP/s, so three quarters of the library's own in-place rate is already reached;
the runtime already selects the fastest zero-workspace algorithm, and a 4 096³ square
GEMM also measures 21 TFLOP/s, so ~20 TFLOP/s is the library's practical ceiling on
this device against a ~51 TFLOP/s fp16 peak for 40 CUs. The solver is device-bound
rather than host-bound — host enqueue for one evaluation is 28 ms against 419 ms of
device time. Evidence:
[`solver stage profile`](results/yue2_solver_stage_profile_20260917.json).

An earlier version of this section quoted a synchronize-wrapped split of the 7.95 s
two-step solve (3.31 s of attention, 0.94 s of projections, 3.73 s of conditioning
prefill, RoPE and norms). Those numbers are superseded: the syncs drained the pipeline,
which understated the GEMMs and overstated the elementwise remainder.

The attention rewrite is bit-exact rather than merely close: the parity fixture
`tests/fixtures/yue2/operators/nar_attention_parent.npz` holds the parent kernel's
own output bits and the kernel test compares against them exactly. The projection
change cannot be bit-exact (a different GEMM accumulation order is the point), so it
is held to the production gates instead: all three M4 solver gates pass (chunk0
latents rel L2 0.01248 / cosine 0.999923 against 0.01225 / 0.999926, multi-chunk
0.01071 / 0.999944 unchanged, restricted visibility 0.00905 / 0.999960 against
0.00885 / 0.999962), the 18-case AR replay stays inside its floor while improving,
and the three-case greedy session gate passes with every case slightly better
(semantic teacher-forced 98.2% / 97.1% / 96.7%). The kernel keeps a scalar K walk
for head dimensions that are not a multiple of eight, where the rows are not
16-byte aligned; that fallback has its own test, and perturbing it fails the new
`head_dim=100` case while `head_dim=128` still passes. Evidence:
[`M7 attention packing`](results/yue2_m7_attention_packed_20260917.json),
[`M7 projection selection`](results/yue2_m7_gemm_algorithm_20260917.json),
[`M7 row blocking`](results/yue2_m7_attention_20260917.json).

The decoder is faster than the reference's (22.47 s against 69.09 s) and the AR
stage runs at 19.35 tokens/s against 31.97.

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

Two exact width-8 owners joined it on 2026-09-05, each measured here at 512/128
against the route before it: a Q8T16 pair rowtile (+2.51%) and a Q4 pair-reuse
owner (+4.97%), both in [`CHANGELOG.md`](CHANGELOG.md). The table above predates
them.

### Maple-Preview 2-bit on Radeon 8060S

| Interface | c1 | c2 | c4 | c8 | Scope |
| --- | ---: | ---: | ---: | ---: | --- |
| Public engine generation64 | **123.131** | **165.697** | **202.038** | **214.788** | Admission, prefill, generation, reclaim |
| Fixed helper decode64 | — | **250.481** | **346.365** | **428.063** | Decode helper only; excludes public scheduling |

Evidence: [`public P4`](results/2026-08-08-gfx1151-maple-p4-long-prefill-public-batch-retained.json)
and [`D1 helper`](results/2026-08-08-gfx1151-maple-d1-batched-affine4-rowreuse-retained.json).

## Current speculative decode scoreboards

Rows labeled historical are older protocol snapshots, not admission evidence
for the current gfx1151 FP32 production profile.

| Platform / model | Contract | True AR | MTP | MTP / AR | Status and evidence |
| --- | --- | ---: | ---: | ---: | --- |
| RX 7900 XTX / Qwen3.8-27B Dense `Q4_K_M` | Exact/default natural25 B3 | 38.373 | **65.347** | **1.7029x** | Current `ud-quants` paired control on a clean worktree at `92c7e3dc4`; `complete_exact`, all ten prompts exact across two runs with identical token IDs, GPU/CPU acceptance agree, `speed_claim_eligible: true`, and the true-AR arm uses recorded production graph replay. [`artifact`](results/2026-09-13-ud-gfx1100-paired-clean-provenance.json) |
| Radeon 8060S / Qwen3.8-27B Dense `Q4_K_M` | Merged-tree served baseline: multi-turn ShareGPT 256–1,200-token prompts, greedy, cache off, in-load AR control | 12.27 | 25.34 | **2.07x** | Working baseline on `19f2dd45e`, not admission evidence. Per-request decode rate (median of `completion/(e2e−ttft)` over rows); median ITL **82.8 → 40.1 ms** and the matched per-turn medians favour the speculative arm at every turn. At c=4 the same protocol is **1.03x** (9.59 against 9.30) while the load aggregate rises to 21.65 tok/s, so concurrency buys the aggregate and the route adds nothing per request there. Warm prefix hits use no MTP. A different workload from the natural-short rows above; rates are not comparable across them. [`artifact`](results/2026-09-20-gfx1151-post-merge-baseline-cache-and-mtp-arms.json) |
| Radeon 8060S / Qwen3.8-27B Dense `Q4_K_M` | Natural ShareGPT prompts, c=1, matched prompts, in-load AR control | 12.02 | 23.54 | **1.958x** | Promoted staged route; the route choice is inert below the 1,023-token graph cap (the row-wise route measures 23.62, **1.964x**). Above the cap on natural long prompts the staged route measures **15.39 against 9.33 row-wise and 8.75 AR (1.76x against 1.07x)**, same prompts and tokens, acceptance 0.688 either way. At c=4 the same short prompts are **0.85x** AR on both routes. `serial_exact` verification is token-identical to AR on 9/9 prompts at 8.07 tok/s against 23.55 for the default `native` mode. [`artifact`](results/2026-09-19-gfx1151-qwen38-natural-sharegpt-mtp-route-comparison.json) |
| Radeon 8060S / Qwen3.8-27B Dense `Q4_K_M` | Historical direct-leaf natural25 B3 | 11.692 | 21.158 | 1.8095x | August 26 direct-leaf protocol, not the public-server headline. [`artifact`](results/2026-08-26-gfx1151-qwen38-current-main-ar-mtp.json) |
| W7900 / Qwen3.6-35B-A3B `UD-Q4_K_M` | Public production/BF16 resident-C2 K2 D24, automatic | 80.973 | **93.644** | **1.1565x** | Latest-source 10/10 engaged and MTP self-exact; three-run ratio 1.1368x; all categories non-regressive; strict-teacher, blocking/SSE/cancel/drain pass. Shares the artifact linked in the row above. |
| Radeon 8060S / Qwen3.8-27B Dense `Q4_K_M` | Public strict/BF16 cap4 realized-C1 K3, natural25 | 11.150 | **20.985** | **1.882x** | Three full-suite runs; all 30 cells exact, engaged and budget-conformed; every category faster. Blocking/SSE and cancellation/refill pass. The production default was AR under that protocol; automatic C1 admission followed on 2026-09-18. [`artifact`](results/2026-09-12-gfx1151-qwen38-final-headline-refresh.json) |
| Radeon 8060S / Qwen3.8-27B Dense `Q4_K_M` | Historical FP16 production C1 B3, c68-128/h24 | 9.350 | 13.088 | 1.3998x | Older profile certificate; not current FP32-production admission. [`artifact`](results/2026-08-27-gfx1151-qwen38-c68-c128-production-explicit.json) |
| Radeon 8060S / Qwen3.8-27B Dense `Q4_K_M` | Historical FP16 production C3 K3 D24 diagnostic | 20.788 | 19.934 | 0.9589x | Older implementation comparison, slower than its AR baseline; not current-profile admission. [`artifact`](results/2026-08-28-gfx1151-qwen38-c3-production-rowtiles-retained.json) |
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

Workloads use `prompt_tokens/decode_tokens`; compare matching timing, model/quant/KV, concurrency and memory scopes.
Bold marks the reported row, not a universal leader. Blank cells are unmeasured, not failures; Max context requires a dedicated ceiling run.

## Maintenance contract

Replace protocol rows rather than adding optimization diaries. Keep commands,
samples, deltas, profiler/correctness details and decisions in compact artifacts;
record transitions in [`CHANGELOG.md`](CHANGELOG.md), decisions in immutable
worklog entries and old tables in [`HISTORY.md`](HISTORY.md) or Git history (`git show 6a8d38ae70b9e2c4244df10d8621db83da6c8112:benchmarks/README.md`).
Blocked/rejected runs belong here only for withdrawn rows or user-visible limits,
with an artifact and rerun condition. [`docs/BENCHMARK.md`](../docs/BENCHMARK.md)
defines the full evidence contract. Update `Last updated`, run
`python3 scripts/sync_benchmark_readme.py --write`, then `--check` and
`git diff --check`.
