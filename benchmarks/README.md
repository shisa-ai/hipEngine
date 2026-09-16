# hipEngine Topline Benchmarks

Last updated: **2026-09-16**
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
[protocol and environment](../docs/MODEL-SURYA.md#final-lane-comparison-2026-09-14).
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
[integration report](../docs/UD-MAIN-INTEGRATION.md); its working-tree timings
do not replace the published performance rows.

## Root README performance summary

The root README exports this compact retained summary verbatim.

<!-- BEGIN TOPLINE:README_HIGHLIGHTS -->
Measured tokens/s on each named host. **Prompt processing** measures input;
**text generation** measures output. **MTP** is speculative decoding within
qualified scopes. Dashes indicate unmeasured results; context limits come
from separate capacity tests.

### Performance

#### Radeon Pro W7900 — 48 GB (`gfx1100`)

| Model | Quant | Prompt processing | Text generation | With MTP | Max context |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | ParoQuant W4 | **2852.1** | **115.8** | **115.8** | — |
| Qwen3.6-35B-A3B | GGUF `Q4_K_M` | **2763.6** | **94.6** | 122.7 | — |
| Qwen3.8-27B Dense | GGUF `Q4_K_M` | **868.6** | **27.9** | 39.7 | **176,128** |
| Laguna S 2.1 | GGUF `UD-Q2_K_XL` | **440.9** | — | — | — |

Laguna: 4K prompts. 35B-A3B GGUF MTP: explicitly enabled.
Qwen3.8 MTP: legacy BF16/K3, one request, 24 outputs;
**1.63x versus its matched 24.36 tok/s AR**, not the INT8 column.

#### Strix Halo / Radeon 8060S — 120 GB (`gfx1151`)

| Model | Quant | Prompt processing | Text generation | With MTP | Max context |
| --- | --- | ---: | ---: | ---: | ---: |
| Maple-Preview | 2-bit | **754.5** | **153.2** | — | — |
| Qwen3.6-35B-A3B | GGUF `UD-Q4_K_M` | **1369.5** | **54.3** | 80.1 | — |
| Laguna S 2.1 | GGUF `Q4_K_M` | **654.2** | **23.2** | — | — |
| Qwen3.8-27B Dense | GGUF `Q4_K_S` | **396.1** | **13.1** | **23.9** | — |
| Qwen3.8-27B Dense | GGUF `Q4_K_M` | **404.5** | **12.2** | 21.0 | — |

Qwen3.8 `Q4_K_M` defaults to production AR. MTP uses strict/K3 in one
qualified single-request scope: **1.88x its matched 11.15 tok/s AR baseline**,
not the 512/128 generation column.
[Measurements](https://github.com/shisa-ai/hipEngine/blob/main/benchmarks/results/2026-09-12-gfx1151-qwen38-final-headline-refresh.json).

**Time-series forecasting (TimesFM 2.5 200M).** hipEngine decodes batch=8,
context 8192, horizon 512 forecasts in **0.082 s** on the HP ZBook Strix Halo
host (8.6x the official torch reference there) and **0.062 s** on a Framework
Desktop host — the same `gfx1151` GPU on two physical machines, so the gap is
host power/thermal headroom, not a code change.

**VibeVoice-ASR 9B** is torch-free on Strix Halo. Q4_K_M beats bf16 — **1.52x** prefill,
**1.55x** decode — at 6.1 vs 16.7 GB, WER 2.01% vs torch 2.81%; **RTF 0.35**
(decode 21.4 tok/s).
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

The full 262,144-token context needs a predicted 24.8 GiB and does not fit.
The direct-INT8 route shown here failed 9 of 11 quality prompts and is not a
default. [Capacity evidence](https://github.com/shisa-ai/hipEngine/blob/main/benchmarks/results/2026-09-09-rx7900xtx-gguf-int8-direct-prefill-capacity.json)

### Serving several requests at once

Aggregate tokens per second across all active requests, Qwen3.8-27B `Q4_K_M`
on the W7900, measured September 4, 2026 under one server protocol.
The peers use F16 KV where hipEngine uses BF16.

| Requests | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hipEngine | **23.6** | **39.1** | **53.1** | **63.9** | **72.8** | **79.5** | **83.2** | **85.9** |
| llama.cpp HIP | 21.0 | 34.4 | 30.6 | 27.7 | 36.7 | 46.4 | 52.1 | 58.4 |
| hipEngine advantage | +12% | +14% | +74% | +130% | +99% | +71% | +60% | **+47%** |

Direct engine measurements, September 6, 2026, same card and model,
512-token prompts and 128 outputs per request. They precede the shared-pool
changes; memory figures are not current serving estimates:

| Requests | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Text generation (total) | 29.6 | 54.0 | 75.2 | 92.3 | 105.9 | 117.6 | 123.9 | **131.3** |
| Prompt processing (total) | **678.8** | 368.9 | 362.6 | 380.0 | 378.3 | 403.6 | 385.3 | 376.6 |
| Peak memory (GiB) | 19.4 | 20.3 | 21.1 | 22.0 | 22.8 | 23.7 | 24.5 | 25.4 |

Eight requests need about 25 GiB (32 GB or larger card); 24 GB shapes
are not qualified for concurrency yet.
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

INT8 KV is used only when the loaded artifact is qualified for it; the gate is
keyed on the exact artifact SHA-256, size, backend, target, weight quant, layout,
and scale dtype. An unqualified artifact falls back to BF16 with the reason
recorded in `/ready`, so the row above is specific to the qualified artifact.

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
[`dedicated campaign`](../docs/QWEN38-GFX1100-C8-K3-CAMPAIGN.md).

On Strix Halo, Qwen3.8 `Q4_K_M` defaults to production AR. Strict C1/K3
MTP requires the qualified settings described above; production MTP requests
fall back to AR. `Q4_K_S` uses FP16
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
([`capacity notes`](../docs/QWEN38-27B-GFX1100-24GB-CAPACITY.md),
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
DMS sidecar remains default-off pending serving gates. [`DMS`](../docs/DMS.md).

Agentic quality is quality-only: Qwen3.8-27B `Q4_K_M` scores **50/68 (73.53%)**
with 64/64 valid calls; no runtime mechanism is retained.
[`Final`](results/2026-08-26-zbook-agentic-quality2-campaign-final.json).
Generation-2 automatic serving remains K0: gfx1151 P9 is exact 540/540 but c2/c4
are 0.6975x/0.5843x AR; gfx1100 exact speculative cells remain behind direct.
[`Closure`](results/2026-08-26-gfx1151-specdec2-perf-campaign-closure.json) ·
[`Recovery`](../docs/MTP-CONCURRENCY2-RECOVERY.md).

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
[`campaign plan`](../docs/QWEN38-27B-GFX1151-CAMPAIGN.md).

The standard (non-UD) `Q4_K_M` file is a separate lane from `Q4_K_S`.
Current measurements use the public production profile, FP32 recurrent state,
BF16 KV, two hardware queues, and a prepared 8,192-token session. Each shape
has one discarded warmup and three measured resets; decode uses the
backend-selected HIP graph path.

| Shape | Prefill | AR decode | Tracked peak |
| --- | ---: | ---: | ---: |
| 512/128 | **404.487 tok/s** | **12.226 tok/s** | 24.153 GiB |
| 1K/128 | **395.000 tok/s** | **11.997 tok/s** | 24.153 GiB |
| 4K/128 | **373.218 tok/s** | **12.153 tok/s** | 24.153 GiB |

Every prefill/decode CV is below 0.22%. All timed final logits are finite,
and the graph/eager preflight matches every generated ID, final full logits,
and state fingerprint on 18 category/heldout prompts. The public packed
numerical gate covers 8,716 rows at KL0/top-1 100%; real-socket blocking/SSE,
cancellation, refill and clean-drain checks pass. Tracked peak includes
public session pools and the graph preflight; session-owned peak is 19.810 GiB.
These memory scopes differ from the older right-sized low-level-session
measurements. [Current profile measurements](results/2026-09-12-gfx1151-qwen38-final-headline-refresh.json).

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
| Radeon 8060S / Qwen3.8-27B Dense `Q4_K_M` | Historical direct-leaf natural25 B3 | 11.692 | 21.158 | 1.8095x | August 26 direct-leaf protocol, not the public-server headline. [`artifact`](results/2026-08-26-gfx1151-qwen38-current-main-ar-mtp.json) |
| W7900 / Qwen3.6-35B-A3B `UD-Q4_K_M` | Public production/BF16 resident-C2 K2 D24, automatic | 80.973 | **93.644** | **1.1565x** | Latest-source 10/10 engaged and MTP self-exact; three-run ratio 1.1368x; all categories non-regressive; strict-teacher, blocking/SSE/cancel/drain pass. Shares the artifact linked in the row above. |
| Radeon 8060S / Qwen3.8-27B Dense `Q4_K_M` | Public strict/BF16 cap4 realized-C1 K3, natural25 | 11.150 | **20.985** | **1.882x** | Three full-suite runs; all 30 cells exact, engaged and budget-conformed; every category faster. Blocking/SSE and cancellation/refill pass. Production default is AR. [`artifact`](results/2026-09-12-gfx1151-qwen38-final-headline-refresh.json) |
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
