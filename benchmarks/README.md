# hipEngine Topline Benchmarks

Last updated: **2026-09-10**
This file is the current benchmark scoreboard. It intentionally contains only
current user-facing results, compact protocol/status notes, and links to the
authoritative evidence. It is not an optimization journal.

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
| Qwen3.8-27B Dense | GGUF `Q4_K_M` | **680.4** | **29.7** | — | — |
| Laguna S 2.1 | GGUF `UD-Q2_K_XL` | **440.9** (4K) | — | — | — |

#### Strix Halo / Radeon 8060S — 120 GB (`gfx1151`)

| Model | Quant | Prompt processing | Text generation | With MTP | Max context |
| --- | --- | ---: | ---: | ---: | ---: |
| Maple-Preview | 2-bit | **754.5** | **153.2** | — | — |
| Qwen3.6-35B-A3B | GGUF `UD-Q4_K_M` | **1369.5** | **54.3** | 80.1 (opt-in) | — |
| Laguna S 2.1 | GGUF `Q4_K_M` | **654.2** | **23.2** | — | — |
| Qwen3.8-27B Dense | GGUF `Q4_K_S` | **396.1** | **13.1** | **23.9** | — |
| Qwen3.8-27B Dense | GGUF `Q4_K_M` | — | — | **15.6** | — |

**Time-series forecasting (TimesFM 2.5 200M).** batch=8/context 8192/
horizon 512 forecasts in **0.082 s** - 8.6x the official torch reference on
the same GPU; FP16 production within 0.86% max error of the FP32 oracle.

#### NVIDIA RTX PRO 6000 Blackwell — 96 GB (`sm_120a`)

| Model | Quant | Prompt processing | Text generation | With MTP | Max context |
| --- | --- | ---: | ---: | ---: | ---: |
| Maple-Preview | 2-bit | **1917.5** | **402.4** | — | — |

Blank cells are shapes we have not measured yet, not failures. Max context is
published only where a dedicated ceiling run exists.

- **Qwen3.8-27B `Q4_K_M` context ceilings on 24 GB `gfx1100`:** BF16 KV
  server 40,960; DMS BF16 73,728; INT8 KV direct engine 131,072; DMS INT8
  merged lane 172,288; direct-INT8 prefill or DMS INT8 + single hidden
  plane 232,448 (the model's full 262,144 context needs a predicted
  24.8 GiB and does not fit). With
  [DMS](https://arxiv.org/abs/2506.05345), a trained eviction policy
  compacts the KV cache: the DMS INT8 route is significantly more accurate
  than direct-INT8 KV (mean row-KL 0.001 vs 0.188, top-1 agreement 100% vs
  91.4% against the BF16 teacher). [Capacity
  evidence](https://github.com/shisa-ai/hipEngine/blob/main/benchmarks/results/2026-09-09-rx7900xtx-gguf-int8-direct-prefill-capacity.json)

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
larger card; concurrent-request shapes on 24 GB are not qualified yet.

On Strix Halo, Maple-Preview 2-bit scales to **214.788** tok/s across eight
requests (123.131 at one, 165.697 at two, 202.038 at four). Where speculative
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
capacity-2/two-request keys. **Qwen3.8-27B `Q4_K_M` uses ordinary AR by default
at every width.** Explicit MTP is available with production/BF16 KV, context
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

A separate explicit C2/K3 run measured 44.69 versus 41.87 AR tok/s (1.067x).
The 56-cell screen mixes capacities and does not establish N=1 support by
itself. The single-request (C1) explicit route engages the legacy singleton
target verifier today: the backend physical policy admits packed C1 only at
K2/K3, and the measured packed route is far slower than the legacy route
(below), so the legacy route remains the only viable single-request path and
its rates are what a single-request user gets. Deeper single-request depths
(K1, K4-K7) refuse before mutation on both routes: the physical policy lists
no packed cell for them and every retained evidence row qualifies at most
K3. [Measurements](results/2026-09-06-w7900-q4km-mtp-packet6-grid-and-c2k3.json);
[qualification work](../docs/QWEN38-27B-GFX1100-CONCURRENCY2-BETTER-MTP.md).

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
the argmax ranking 8/8 (torch bf16 5/8), max rel 0.24% (bf16 3.5%).
The fp16 path is fixture-gated against the torch fp32 oracle (query
embedding cos 0.9999986, MaxSim within 0.01%); a strict fp32 path passes
the same oracle at 1e-5 tolerances, with a registered strict-elementwise
fallback (HIPENGINE_EVIE_STRICT_ELEMENTWISE). Full optimization history
(15.62 s -> 1.13 s)
in the artifacts
([sprint](results/gfx1151-evie-4p5b-fp16-perf-sprint-2026-09-09.json),
[cluster8](results/gfx1151-evie-4p5b-cluster8-recurrence-2026-09-10.json)).

### W7900 Qwen3.8 `Q4_K_M` C1-C8

#### Server performance on the INT8 KV route (2026-09-10)

The INT8 KV server route was repaired this release: long-prompt prefill
output is now exact, and the route's memory overhead was reduced by
removing a duplicate context-sized reservation. Measured on W7900,
27B `Q4_K_M`, single request, 16K context:

- **Prefill speed** matches the direct engine: 2K-8K-token prompts run at
  677-732 tok/s through the server route versus 685-756 directly, with
  bitwise-identical logits
  ([repetitions](results/2026-09-10-w7900-p7-c1-speed-parity-paired-reps.json)).
- **Decode speed** is the same compute path either way: 35.2 ms/token
  through the server entry versus 35.1 ms directly
  ([ladder](results/2026-09-10-w7900-decode-boundary-ladder-16k.json)).
- **Memory**: the server now pays exactly the canonical 32.5 KiB per
  layer-token of KV (no server-only context-sized overhead on this
  route); per-request transient scratch fell ~75% at 32K context
  ([measurement](results/2026-09-10-w7900-server-alloc-probe-16k-p4-lease-removed.json)).
- **Context capacity**: the INT8 server route now accepts up to
  155,648-token contexts on a 24 GB card (~2.9x the previous 54,272-token
  limit), verified with complete prefill-and-decode request cycles at
  65K/98K/131K/152K contexts
  ([bracket](results/2026-09-10-w7900-server-alloc-probe-152k.json)).
- **Streaming**: first token arrives as soon as prefill completes
  (~2.8 s on a 2K-token prompt); subsequent tokens stream at the decode
  rate
  ([observation](results/2026-09-10-w7900-p7-http-transport-budget.json)).

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

## Qwen3.8-Flash-Next implementation-first status

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

The completed same-host family alignment covers six prompts and both phases.
It identifies FFN/linear/GR as the largest prefill gaps and QSA as the largest
decode gap; HIP kernel sums and Vulkan query intervals remain diagnostic.
[Generated family evidence](results/2026-09-05-framework-qwen4exp-family-alignment.json).
The post-Q8 current-HE refresh keeps that ranking, with linear5.551s,
GR4.362s and FFN14.445s at p4096; the Vulkan profile is explicitly reused,
not remeasured. [Updated family packet](results/2026-09-05-framework-qwen4exp-post-q8-family.json).

Standalone GR up+sigmoid+mean screen, actual weights, rows512, no memory
preconditioning:

| Weight | Parent (ms) | Candidate (ms) | Speedup |
| --- | ---: | ---: | ---: |
| Layer0 attention up | 3.338 | 3.234 | 1.032x |
| Layer0 FFN up | 3.331 | 3.207 | 1.039x |
| Layer4 attention up | 3.320 | 3.204 | 1.036x |
| Layer4 FFN up | 3.313 | 3.209 | 1.032x |

F32 gate/mixed bits exact,20 tests pass, both order strata positive at512;
small-row order reversals are disclosed. Model admission pending.
[GR screen](results/2026-09-06-framework-qwen4exp-gr-wave-scale.json).

Latest retained Q8 wave-scale production A/B, normal model execution:

| Prompt | Parent prefill | Wave-scale prefill | Gain |
| --- | ---: | ---: | ---: |
| 512 | 154.29 | 155.45 | +0.75% |
| 1024 | 151.67 | 153.43 | +1.17% |
| 4096 | 141.58 | 143.41 | +1.29% |

Tok/s on Framework, UD-Q4_K_XL/BF16 KV. All72 trajectories exact; all12 cases
improve prefill and complete-request wall, with zero final allocations.
Decode drift remains: no intrinsic decode improvement or new external
parity claim. Full A/B33m53s, without synthetic memory preconditioning.
[Q8 production](results/2026-09-05-framework-qwen4exp-q8-wave-scale-production.json).
The earlier microbench's order reversal remains recorded in its
[separate evidence](results/2026-09-05-framework-qwen4exp-q8-wave-scale.json).

The larger Q4 row-batch screen is rejected: exact RB16/32 variants lose
against RB8 in all12 final actual-weight, synthetic-routing cells across
tokens512/1024/2048. Production is unchanged.
[Screen evidence](results/2026-09-05-framework-qwen4exp-q4-rowbatch-rejected.json).

The preceding Q4 output-pair retention remains part of the baseline:

| Shape | Parent prefill | Q4 pair prefill | Gain | Decode before -> after |
| --- | ---: | ---: | ---: | ---: |
| p512 | 145.21 | **153.78** | **+5.90%** | 19.385 -> 19.383 |
| p1024 | 142.82 | **151.21** | **+5.87%** | 18.556 -> 18.512 |
| p4096 | 134.01 | **141.30** | **+5.44%** | 13.113 -> 12.899 |

All rates are tok/s on Framework `gfx1151`, UD-Q4_K_XL/BF16 KV, four
categories and tg128. This is a same-residency incremental A/B with all72
trajectories exact, all12 prefill cases positive, and complete-request wall
speedups of1.67-3.90% per case. Decode losses of0.23%/1.63% at p1024/p4096
are retained under the owner's prefill-first direction and remain open work.
Both arms drift; this is not a stable absolute decode comparison or external
parity claim. Full A/B elapsed34m53s; teardown is zero.
[Full timing and promotion evidence](results/2026-09-05-framework-qwen4exp-q4-pair-production.json).

The preceding serial-GDN admission improved prefill9.50-10.55% and remains
part of this baseline. Its separate decode tradeoff is preserved in its
[GDN evidence](results/2026-09-05-framework-qwen4exp-gdn-register-production.json).

The preceding Q5_1 pair2 admission improved prefill3.73-4.17% and remains
part of this measured baseline.
[Pair2 evidence](results/2026-09-05-framework-qwen4exp-q51-pair-production.json).

The prior promoted combination remeasured at121.14 pp/s on the code-p4096
diagnostic (not an all-category refresh). The original standalone Q5_1 output-
pair candidate reduces two actual down banks at tokens512 from about54.5
to43.5 ms (1.253x), exact outputs; the full model result is above.
[Combined profile and Q5_1 screen](results/2026-09-05-framework-qwen4exp-q51-pair.json).

Production retains exact page256 QSA and bundled-Q4 prefill under the
2026-09-05 owner decision to take the prefill gains and optimize decode next.
Strict keeps the prior owners. Separate component A/B measurements are:

| Component | Shape | Prefill before -> after (tok/s) | Gain | Decode change |
| --- | --- | ---: | ---: | ---: |
| Bundled Q4 | p512 | 123.34 -> 127.62 | +3.47% | -0.23% |
| Bundled Q4 | p1024 | 121.47 -> 125.33 | +3.18% | -0.06% |
| Bundled Q4 | p4096 | 97.34 -> 99.72 | +2.45% | -0.64% |
| H256 wave QSA | p4096 | 97.83 -> 116.70 | +19.29% | -0.48% |

These are separate same-residency component runs, not additive gains or
combined throughput. The measured hot-decode tradeoffs remain open work.
[Promotion and evidence](results/2026-09-05-framework-qwen4exp-prefill-promotion.json).

The original standalone exact Q4 gate/up bundled publication screen reduces the actual-weight
gate/up+SiLU boundary at tokens512 from30.39 to24.97 ms (1.217x), all pairs
exact. Timing variability and whole-model admission remain open; this is
not a production gain.
[Q4 bundle](results/2026-09-05-framework-qwen4exp-q4-bundle.json).

The current same-host comparator screen ran on the Framework Desktop (physical
host `gfx1151`, machine ID `55ea6c509d0b49eea8de7094a1023668`, Ryzen AI Max+
395 / Radeon 8060S) with the verified four-part Unsloth `UD-Q4_K_XL` artifact,
BF16 K/V, one warmup, and three measured requests per canonical case:

| Engine | p512 pp/tg128 | p1024 pp/tg128 | p4096 pp/tg128 | Repeatability |
| --- | ---: | ---: | ---: | --- |
| hipEngine production + exact Q5_K row4, HIP (later same-residency A/B) | **122.57 / 19.97** | **121.47 / 19.25** | **97.85 / 15.29** | **12/12**, cross-arm exact |
| hipEngine production `c0cfdc3ef`, HIP | 118.44 / 19.92 | 117.79 / 19.22 | 95.14 / 15.21 | **12/12** |
| Upstream llama.cpp `4d9176092`, HIP | 283.85 / 21.06 | 367.97 / 20.79 | 395.02 / 19.63 | **11/12** |
| Upstream llama.cpp `4d9176092`, Vulkan | 230.35 / 24.94 | 305.47 / 24.59 | 357.44 / 23.53 | **11/12** |
| halo-box master `b212548e0`, HIP | 265.69 / 21.02 | 368.90 / 20.50 | 356.62 / 18.63 | **12/12** |
| halo-box master `b212548e0`, Vulkan | **298.97 / 24.92** | **369.72 / 24.52** | **402.46 / 23.47** | **12/12** |

The two upstream lanes vary on `mixed_ja_en-p4096`. Every external lane also
exceeds 2% maximum per-case coefficient of variation on at least one metric,
so this is a screening result, not a frozen closure target. Do not compare
these rates as old-to-new deltas against `zbook`. Exact commands, binary and
model hashes, per-sample rates, and output hashes are in the
[Framework comparator packet](results/2026-09-05-framework-gfx1151-qwen38-flash-next-current-comparators.json).

The row4 row is a later same-host internal A/B, not a paired rerun against
the external lanes. Its own parent rates are 118.92/117.72/95.42 pp/s:
prefill improves 3.07%/3.19%/2.55%, all 72 measured trajectories are exact,
max per-case prefill CV is 0.23%, and teardown is clean.
[Production evidence](results/2026-09-05-framework-qwen4exp-row4-production.json).

The original H256 sparse-attention standalone screen at
24Q/2KV/D256 and selected stride2051, rows512 attention measures
171.93→28.02 ms (6.14x), with exact parent output bits. This leaf ratio is
not a whole-model gain.
[QSA candidate](results/2026-09-05-framework-qwen4exp-qsa-h256-wave.json).

Its full-suite internal A/B measures p4096 prefill 97.83→116.70 tok/s
(+19.29%), but decode 15.27→15.20 (-0.48%). All 72 trajectories are exact;
the candidate initially stayed default-off pending focused decode followup. An English
128-step probe also matches full logits and complete K/V, but is not a
replacement for the original timing protocol.
[Full-suite audit](results/2026-09-05-framework-qwen4exp-qsa-fullsuite-audit.json).

The isolated English rerun has flat decode, but its subset initially reversed
the original arm order. That diagnostic does not waive the full-suite finding;
the harness now preserves original fixture indices for focused comparisons.
[Subset-order audit](results/2026-09-05-framework-qwen4exp-qsa-subset-order-audit.json).

The corrected-order three-case rerun still finds mixed-language decode
down 0.64%; English/Japanese are effectively flat and all 18 trajectories
are exact. A 2880 MHz clock snapshot during the affected section is a
lead, not a causal diagnosis. The later owner decision accepts the measured tradeoff.
[Corrected followup](results/2026-09-05-framework-qwen4exp-qsa-corrected-followup.json).

The standalone page256-addressing sibling reduces rows512 attention from
28.13→27.56 ms versus the generic wave candidate, with both arms exact.
This does not establish a power benefit or clear the whole-model decode gate.
[Page256 screen](results/2026-09-05-framework-qwen4exp-qsa-page256.json).

External phase telemetry reproduces page256's late mixed decode loss and
records lower after-arm clocks (about 2880–2882 versus 2898 MHz). It is
correlation, not yet a causal proof or promotion.
[Phase clocks](results/2026-09-05-framework-qwen4exp-qsa-phase-clocks.json).

A separate mixed p4096 control holds every sampled phase at2700 MHz:
prefill93.01→112.25 tok/s (+20.69%), decode14.800→14.798 (−0.017%), exact
outputs. This supports an operating-point effect but is not promotion at
the original setting; 2900/2900 high policy was restored.
[Fixed-clock control](results/2026-09-05-framework-qwen4exp-qsa-fixed-clock-control.json).

The subsequent Framework code-case owner diagnostic at `cf9c55920` has 100%
role coverage and traced/unprofiled final-logit equality:

| Diagnostic | code-p512 | code-p4096 |
| --- | ---: | ---: |
| Unprofiled prefill wall (s) | 4.306 | 43.046 |
| Routed MoE device time (s) | 2.402 | 19.024 |
| QSA device time (s) | 0.042 | 8.983 |
| GDN device time (s) | 0.489 | 3.906 |

No optimization or new paired comparator verdict is claimed.
[Owner refresh](results/2026-09-05-framework-gfx1151-qwen38-flash-next-owner-refresh.json).

Framework standalone Q5_K grouped-row4 gate/up, including its device map,
screens at 30.022→24.391 ms (64 rows) and 239.491→107.911 ms (512 rows),
with bit-exact paired outputs. This is a development-tree primitive result,
not a runtime promotion or whole-model gain.
[Candidate evidence](results/2026-09-05-framework-qwen4exp-q5k-grouped-row4.json).

Earlier implementation-calibration evidence was collected on physical host
`zbook` (Ryzen AI Max+ Pro 395 / Radeon 8060S, `gfx1151`). The pinned artifact
runs through public `LLM.generate()` under the strict c1/greedy text scope.
Frozen same-artifact llama.cpp PR #27742 full logits over all 10 canonical
code/general-English/general-Japanese/mixed prompts measured:

| Artifact | Context scope | Mean / p95 / p99 / max KL ↓ | Top-1 | Tracked peak / after close |
| --- | --- | ---: | ---: | ---: |
| Qwen3.8-Flash-Next `UD-Q4_K_XL` | real ≤2,051-token canonical text gate | **0.01406 / 0.04154 / 0.04776 / 0.04931** | **10/10** | 82.718 GB / **0 B** |
| Qwen3.8-Flash-Next `UD-Q4_K_XL` | predeclared eight category heldouts, matched BF16 K/V | **0.00987 / 0.02331 / 0.02766 / 0.02874** | **8/8** | same residency / **0 B** |

The pinned 111.335-GB/four-hash artifact owns one 28.800-GB sparse-mmap PLE
table and 82.523 GB hot weights. Exact batching passes 687/687 rows; the strict
prefill default chunk is now 512 (PLE staging capacity plumbed to the chunk;
previously silently capped at 256): same-session counterbalanced sweeps give
p508 **8.458→8.270 s (-2.22%)** and p1012 **17.062→16.751 s (-1.82%)** with
identical logits SHAs, and natural 16K improves to **341.177 s / 47.989 tok/s**
with the full gate passing (prior chunk-256 steady rows were p508 58.466 and
p1006 55.046 tok/s).

The earlier `zbook` exact-token screen fed all engines the same four category
prompts at p512/p1024/p4096 and measured 128 decode transitions after each
prefix:

| Engine | p512 pp/tg128 | p1024 pp/tg128 | p4096 pp/tg128 | Repeatability |
| --- | ---: | ---: | ---: | --- |
| hipEngine current production (PF-5 GDN tile-16 after arm) | **89.87 / 14.81** | **88.97 / 14.77** | **72.93 / 12.39** | 12/12 deterministic; one-residency cross-mode exact; 12/12 prefill-positive |
| hipEngine current production (PF-1/PF-3 after arm) | **89.34 / 14.84** | **88.54 / 14.79** | **72.58 / 12.40** | 12/12 deterministic; one-residency cross-mode exact |
| hipEngine pre-PF baseline (`37d59564…`, HB-1 retained arm) | 83.37 / 14.32 | 82.91 / 14.27 | 69.20 / 12.18 | 12/12 deterministic; cross-arm exact; historical |
| Upstream Vulkan `f1793c1c4`, queue/repack/fit-off | 200.01 / 24.39 | 241.84 / 21.33 | 266.58 / 18.98 | 12/12 exact; noisy p512/p1024 rows |
| Patched-upstream HIP `f1793c1c4` | 235.89 / 17.75 | 306.51 / 16.99 | 283.73 / 14.89 | 12/12 deterministic; cross-arm exact; non-stock loader |
| Halo-box base `6c84c7d5` + loader patches | 223.89 / 17.66 | 308.28 / 16.88 | 301.62 / 14.90 | 12/12 deterministic; cross-arm exact; short-shape drift |
| Halo-box PR11 `a7ad7b7f` + loader patches, fresh matched-BF16 screen | **240.11 / 18.04** | **324.64 / 17.24** | **349.49 / 15.10** | 12/12 deterministic; p512/p1024 prefill unstable; p4096 stable |
| EngramHalo HIP `1423f689` | 234.84 / 17.44 | 314.98 / 17.04 | 381.17 / 15.99 | p512/p1024 exact; p4096 fails |
| Nathan Vulkan `ad914eb`, queue/repack/fit-off | 360.23 / 24.34 | 357.61 / 21.10 | 351.85 / 19.01 | diagnostic: 0/12 exact |
| apepojken Vulkan `843d575` | 291.73 / 23.21 | 375.23 / 22.42 | 397.43 / 22.25 | diagnostic: 8/12 exact |

Nathan produced 16 different outputs from 16 identical-prompt requests;
apepojken varies on four canonical cases; EngramHalo varies on one p4096 case.
Their affected rates remain diagnostics rather than correctness-valid targets.
Pristine upstream HIP did not finish loading in two 1,800-second attempts, so the
measured patched-upstream lane is explicitly non-stock. The fresh halo-box
screen exactly matches the HB-1 BF16 configuration and binary; its maximum
per-case prefill CV is **10.0%/9.7%/1.26%** at p512/p1024/p4096, so only the
p4096 row satisfies the ≤2% stability rule. The previous retained halo-box arm
was 246.55/343.48/354.21 pp/s; the fresh −2.61%/−5.49%/−1.33% shift confirms
that short-shape absolute comparisons need counterbalanced thermal pairs.

This remains a screening refresh, not section-6 closure: five same-thermal
competitor pairs and 4K MTP remain open. Current production's maximum per-case
CV in the one-residency packet is **1.64% prefill / 1.07% decode**. The Vulkan
rows were refreshed with their entitled graphics queue, repack, and fit-off
configuration; upstream p512 prefill/decode and p1024 prefill also remain too
noisy to freeze the closure target.
[`PF-1/PF-3 production refresh`](results/2026-09-04-gfx1151-qwen38-flash-next-halo-pf13-production-refresh.json),
[`halo-box HB-1 comparison`](results/2026-09-02-gfx1151-qwen38-flash-next-halo-box-hb1.json),
[`current P12 packet`](results/2026-09-02-gfx1151-qwen38-flash-next-p12-validation-packet.json),
[`generated report`](results/2026-09-02-gfx1151-qwen38-flash-next-p12-validation-report.md),
[`canonical AR screening`](results/2026-08-30-gfx1151-qwen38-flash-next-canonical-ar-screening.json),
[`entitled Vulkan refresh`](results/2026-09-02-gfx1151-qwen38-flash-next-entitled-vulkan-canonical-refresh.json).

Frozen halo-box HB-base/PR11 exact profiles confirm that the PR activates its
Q4/Q8 MMQ retunes, prompt top-10 compaction, weighted top-10 sum, 32-warp GDN,
and selected elementwise/recurrent specializations on the binding Q4 payload.
Routed-compact/J48/J64, shared-mul-add, and Q8-KV attention paths are inactive.
The largest isolated trace change is p4096 GDN core kernel sum
**2,139.377→647.976 ms**; it is diagnostic pending HB-3 operation-complete
matched pairs. [`halo-box HB-2 census`](results/2026-09-02-gfx1151-qwen38-flash-next-halo-box-hb2.json).
HB-3 stops before timing: pinned operation harnesses pass their exposed cases,
but **0/5 active families** currently share an identical cross-engine fixture,
dtype/layout contract, and operation boundary. No mechanism ratio or candidate is
reported. [`halo-box HB-3 blocker`](results/2026-09-02-gfx1151-qwen38-flash-next-halo-box-hb3-blocked.json).

The 2026-09-04 remediation resolves the PF-1/PF-3 review blocker. One committed
harness kept a single generator resident and toggled both exact routes in ABBA
orders reversed across adjacent cases. Weighted prefill improves
**86.62→89.34 (+3.13%)**, **85.88→88.54 (+3.09%)**, and
**70.80→72.58 tok/s (+2.51%)** at p512/p1024/p4096; all 12 cases improve and
all 72 measured trajectories are exact across modes. Decode changes
−0.15%/+0.13%/−0.07%. Production again selects PF-3 Q5_1 M1 and PF-1 grouped
Q8_0 down; strict retains the preceding registered owners. PF-4's fused-combine
whole-model rejection remains provisional, and the Q4_K M1/PF-5 w32 kernel
losses remain valid.
[`Production refresh`](results/2026-09-04-gfx1151-qwen38-flash-next-halo-pf13-production-refresh.json),
[`Review plan`](../worklog/entries/20260904T100046.831998Z-lhl-qwen4exp-halo-box-campaign-review-511155.md).

The 2026-09-05 PF-5 closure promotes the exact GDN token-tile-16 prefill owner
(binding Hk=16/Hv=48/D=128) as the production default inside the colwarps gate
after a fail-closed, engagement-verified one-residency A/B: weighted prefill
**89.435→89.873 (+0.49%)**, **88.553→88.966 (+0.47%)**, and
**72.661→72.929 tok/s (+0.37%)** at p512/p1024/p4096 with all 12 cases
non-negative, 72/72 cross-mode exact outputs, and per-case prefill CV ≤1.6%.
The binding-shape leaf wins 23.3%/29.1%/35.3% at rows 16/64/512, bit-exact in
outputs and final FP32 state. The columnwarp parent stays registered for
non-envelope shapes and the `HIPENGINE_QWEN4_EXP_GDN_TILE16_PREFILL=0` opt-out;
serial strict remains the registered fallback. The first same-day A/B was
invalidated as a no-op (the runner bypassed the replaced registry key, so both
arms ran the parent) and is superseded. Scaling the loop's 0.3037 screening
baseline by the measured code-only geomean infers ~0.3045; the frozen-halo
screening metric itself is refreshed at the next stable-clock closure verify.
[`PF-5 tile-16 promotion A/B`](results/2026-09-05-gfx1151-qwen38-flash-next-pf5-gdn-tiled16-whole-model-ab.json),
[`Hv48 correction`](../worklog/entries/20260905T022638.146026Z-lhl-qwen4exp-pf5-gdn-tiled16-hv48-correction-c178a9.md).

The frozen p508 role/API profile still puts hipEngine versus llama HIP device
kernels at **5.959 vs 1.625 s (3.67×)** and decode at **48.63 vs 38.90
ms/output (1.25×)**. The main p508 owners are MoE **3.161 s** (layers 0–26:
**2.526 s**), GDN **634.94 ms**, and QSA **110.49 ms**. The largest single miss
is layer-2 Q5_K gate/up at **301.47 vs 15.38 ms**. Decode submits **1,195
direct kernels plus 48 MoE graphs/token**; 625 additional rows/token are
graph-expanded nodes. A strict, layer-local stateful graph diagnostic now
captures one complete 34-kernel GDN+MoE physical layer: output and all request
state owners remain exact through four replays, while synchronized layer wall
falls **4.051→1.258 ms (3.22x)**. A chained layers-0..2 rung is likewise exact
and contracts **9.801→3.896 ms (2.52x)**. A fixed-position layers-0..3 mixed
GDN/QSA diagnostic remains device-state/output exact at **12.160→4.955 ms
(2.45x)**. Its advancing-position successor passes positions 8–11 across
position/context, K/V, QSA index, GDN state, and output at **13.882→4.974 ms
(2.79x)**. An eight-layer successor adds active PLE and a second QSA owner and
remains exact at **26.739→10.112 ms (2.64x)**. The all-physical-layer rung
covers active PLE, all 48 layers, all 12 QSA owners, and 136 device-state owners
at **154.346→57.900 ms (2.67x)** without reproducing third-replay corruption.
The complete host-staged transition then adds generated-token PLE publication,
embedding, final full-vocabulary head, device argmax, and token feedback: the
changing-token trajectory and 138 owners are exact; reset→replay and forced-
eager→graph resumption also pass. The probe-local 194.758-ms eager arm disables
the shipped per-layer MoE cache and adds a script loop, so its 3.15x ratio is
retired. The follow-up strict runner measures **68.855 ms** with shipped MoE
graphs and **143.989 ms** without them, but imports the **61.910-ms** graph row
from the separate probe process. Therefore the derived **1.112x/6.945 ms is not
a named-production A/B** and P8 remains admission-pending. Device argmax versus
host full-logit D2H differs by at most 0.35 ms with a sign flip; steady
allocation growth and teardown pass.
[`named denominator`](results/2026-09-01-gfx1151-qwen38-flash-next-p8-production-denominator.json),
[`stateful layer graph`](results/2026-09-01-gfx1151-qwen38-flash-next-p8-stateful-layer-graph.json),
[`three-layer segment`](results/2026-09-01-gfx1151-qwen38-flash-next-p8-gdn-segment3-graph.json),
[`mixed fixed-position segment`](results/2026-09-01-gfx1151-qwen38-flash-next-p8-mixed-segment4-graph.json),
[`advancing mixed segment`](results/2026-09-01-gfx1151-qwen38-flash-next-p8-advancing-mixed-segment4-graph.json),
[`eight-layer segment`](results/2026-09-01-gfx1151-qwen38-flash-next-p8-advancing-segment8-graph.json),
[`all 48 physical layers`](results/2026-09-01-gfx1151-qwen38-flash-next-p8-all48-graph.json),
[`full host-staged transition`](results/2026-09-01-gfx1151-qwen38-flash-next-p8-full-transition-graph.json).

The current exact-token impact packet fully attributes p512/p1024/p4096 and
live-513/1025/4097. hipEngine versus patched llama HIP device sums are
**5.972/11.196/54.762 s vs 1.926/2.990/10.838 s** prefill and
**51.062/53.262/87.732 ms vs 41.009/41.110/44.005 ms** decode. At p4096,
QSA alone is **10.229 s vs 0.526 s** prefill and **35.587 vs 0.591 ms** decode.
Every hipEngine role window is 100% attributed, both family ledgers have zero
generic remainder, exact decode state/lifecycle gates pass, and median active
experts are 333/327/325 of 512. The old 41.6% remainder was incomplete table
coverage. The patched llama CSVs flushed and hashed, but rocprof required forced
exit after flush, so comparator subwindows remain diagnostic attribution.
[`canonical impact profile`](results/2026-09-01-gfx1151-qwen38-flash-next-canonical-impact-profile.json).

The first retained impact-ranked unit replaces the serialized H256 p4096 QSA
attention owner with an exact three-pass path: parallel incumbent-order QK
scores, one global selected-order online-softmax coefficient recurrence, and
independent output-column weighted-V recurrences. Across four canonical p4096
categories and 12 counterbalanced tg128 pairs, complete decode improves
**93.912→80.061 ms/token (1.173x)**; every pair wins, the aggregate 95% ratio
interval is **1.170–1.176**, and full logits/IDs remain exact. The named trace
records all three expected kernels, reduces the QSA role **36.304→20.913
ms/token**, attributes 100% of device time, allocates nothing in the measured
windows, and tears down to zero. This retained row does not replace the
five-pair section-6 closure baseline.
[`exact ordered QSA decode`](results/2026-09-02-gfx1151-qwen38-flash-next-p6-qsa-ordered-decode.json).

A durable isolated-route recheck reopens the layer-2 grouped-WMMA candidate:
the p508 trace cuts layer-2 MoE **371.10→88.13 ms (4.21×)** and Q5_K gate/up
**279.86→16.66 ms**. Same-process p508 improves **90.25→95.06 tok/s
(+5.34%)**; all 20 category-balanced p512 pairs improve, with per-category
means **+4.83% to +5.20%** and every five-pair 95% CI above 1.0. Each route is
repeat-exact and keeps the same final top-1 token, but full logits differ. The
complete 450-row gate then **rejects** the T2 candidate: overall mean/p95/max KL
`5.03e-4/2.65e-3/0.01238` and 446/450 top-1 pass, as do every category,
repeat/state, and lifecycle checks, but the binding prefill-last/prefill-to-c1
mean KL is **0.001179 > 0.001**. The route remains default-off; c2 and depth
promotion gates were not run because they cannot compensate for this failure.
[`P1 layer-2 rejection`](results/2026-08-31-gfx1151-qwen38-flash-next-p1-layer2-grouped-profile-rejected.json).

The fresh P2 split keeps current production default-off for that candidate and
profiles layers 0–26 at **2.366 s**: exact Q4/Q5_K gate/up **1.200 s**, exact
Q5_1/Q8 down **1.152 s**, and activation plus routing/shared tails only
**13.25 ms**. Layers 3–26 alone retain **1.849 s**; active experts span 166–298
with median 9 rows per active expert. The next exact/T1 work therefore targets
multi-row weight reuse/output tiling in both projection halves, not the <0.6%
tail. Telemetry was collected separately and its D2H wall is excluded.
[`P2 early-MoE profile`](results/2026-08-31-gfx1151-qwen38-flash-next-p2-early-moe-profile.json).

The P3 split names another **1.670 s** of primary p508 roles outside routed MoE:
GR projection/read **709.32 ms**, Q8 `attn_qkv+attn_gate` **532.36 ms**, router
**181.91 ms**, `ssm_out` **137.84 ms**, and shared projections **121.61 ms**.
The first operation-complete target is the 36-layer qkv+gate boundary; it must
preserve current qkv-MMQ and exact-gate arithmetic or qualify a declared T1
pair, with both singleton routes retained as fallbacks. The first extension—Q8
MMQ on the omitted K2560/N6144 gate—wins **1.0352x** p508 and passes all
numerical scopes, but is rejected because candidate state repeat 1 differs from
repeats 2–3 on the first prompt. Ignoring the first same-schedule run as warmup
is not a valid production rule; exact coltile remains default. The next P3
subunit fuses GR sigmoid materialization with gated mean for rows <=256. It
removes one launch per GR read and improves clean counterbalanced
p508+128-step decode **14.162→15.111 tok/s (1.0670x, 95% CI
1.0543–1.0797)**. The complete T0 gate
is exact: **450/450 logits, 18/18 state/task prompts, three repeats, and clean
teardown**. Rows >256 remain unfused after a rows508 primitive loss. The
multirow F32 router projection also reuses each weight row across four prompt
rows while preserving dense arithmetic: clean p508 improves **89.689→91.121
tok/s (1.0160x, 95% CI 1.0143–1.0177)**, with 450/450 logits and 18/18 state/task
prompts exact. c1 remains on the dense owner. The rows>256 GR-up composite also
preserves the exact Q8 reduction while emitting sigmoid gates and branch mean:
clean p508 improves **91.158→91.600 tok/s (1.00484x)** and code-p1024
**88.754→89.239 tok/s (1.00547x)**, with 450/450 logits and 18/18 state/task
prompts exact. P4 also promotes the exact fixed256/precomputed-offset/vector2
QSA dense owner: the real primitive improves **6.846→2.485 ms (2.755x)**,
clean p508 **91.529→92.442 tok/s**, and code-p1024 **89.150→90.634 tok/s**, with
the complete exact/state/task gate passing. P5 moves normal greedy top-1 to the
device: Python-visible D2H falls from **993,280 to 8 bytes/token (124,160x)**
with 450/450 logits, 18/18 generated task sequences, compact state, physical-c2
outputs, and lifecycle exact. Resident-token chaining and normal-AR hidden-copy
elision then reduce the ledger from **28 to 26 blocking copies/token** while
preserving 12 async copies. The p508+128-step wall ratio is neutral at
**1.00343x (95% CI 0.98776–1.01909)**; this is a transfer-boundary retention,
not a wall-speed claim. The current P12 canonical p512/p1024/p4096 production snapshot is
**83.35/82.93/69.20 pp/s** and **14.18/14.16/12.16 tg/s**, all 36 measured
samples deterministic with zero teardown. Named strict is
**61.05/60.32/52.56 pp/s** and **13.52/13.43/9.47 tg/s**; its three-repeat
variance prevents a closure-rate claim.
[`P3 Q8-gate rejection`](results/2026-08-31-gfx1151-qwen38-flash-next-p3-q8-mmq-attn-gate-rejected.json).
[`P3 fused GR`](results/2026-08-31-gfx1151-qwen38-flash-next-p3-gr-sigmoid-mean.json).
[`P3 F32 router tile4`](results/2026-08-31-gfx1151-qwen38-flash-next-p3-router-f32-tile4.json).
[`P3 GR up+sigmoid+mean`](results/2026-08-31-gfx1151-qwen38-flash-next-p3-gr-up-sigmoid-mean.json).
[`P4 QSA dense fixed256`](results/2026-08-31-gfx1151-qwen38-flash-next-p4-qsa-dense-fixed256.json).
[`P5 device argmax`](results/2026-08-31-gfx1151-qwen38-flash-next-p5-device-argmax.json).
[`P5 current canonical AR`](results/2026-08-31-gfx1151-qwen38-flash-next-p5-current-canonical-ar.json).

P6 localizes the long-context cliff to indexed QSA activation. Identical
transition medians at live counts 2,051/2,052/4,097 are **66.61/95.88/96.02
ms**. The boundary adds **30.77 ms** of profiled kernel time; sparse attention
alone adds **27.47 ms**, while score/top-k adds **0.92 ms**. The nearly flat
2,052→4,097 result points to the fixed ~2K selected-attention budget rather than
continued context growth. [`P6 context profile`](results/2026-08-31-gfx1151-qwen38-flash-next-p6-context-transition-profile.json).

A same-weight external-fork refresh built EngramHalo HIP `1423f689` and
Nathan Vulkan `ad914eb` locally. BF16-KV p508/p1012/tg32 shape rows are
**296.12/362.72/17.62** and **413.04/396.25/23.85 tok/s**; Nathan's local
build agrees with its v0.7.2 payload within 1%. These are historical
`llama-bench` shape diagnostics, not exact-prompt or source-only A/B rows. Nathan
lazy-on/off averages **413.04/329.23 p508 (1.255x)** but converges by p1012;
an Engram MTP diagnostic is 1.128x complete-wall at 94.55% acceptance but only
**9/10** AR-message exact, so it is not a valid speed target.
[`external fork refresh`](results/2026-08-30-gfx1151-qwen38-flash-next-external-fork-refresh.json).

The cross-engine survey adds a 160-row, full-vocabulary, same-GGUF packet.
Current upstream HIP is 160/160 top-1 and effectively identical to frozen
#27742 HIP. EngramHalo is 159/160 with mean/max KL **9.85e-4/0.01431**.
Upstream Vulkan, Nathan, and apepojken are each 159/160 versus frozen HIP, but
Nathan is effectively identical to upstream Vulkan (160/160, mean KL about
**2e-10**) and apepojken remains 160/160 versus upstream Vulkan at mean/max KL
**0.00109/0.01576**. This localizes Nathan's failure to multi-step execution
rather than broad static math. Short Q8-KV apepojken MTP is **1.807x**
complete-wall at 92.8% acceptance but only **9/10** AR-message exact, matching
EngramHalo's failing prompt. Nathan MTP is provisionally **1.161x** at 95.45%
acceptance, but AR and MTP each self-repeat only 9/10 and just **8/10** prompts
match across both repeats of both modes. All affected speed rows are invalid as
targets. The survey also compares upstream/fork test coverage and
absolute-quality evidence.
[`Strix Halo survey artifact`](results/2026-08-31-gfx1151-qwen38-flash-next-strix-halo-survey.json).

The previous GDN decode-all claim is **invalid**: its selector was unreachable,
so the packet compared the strict owner to itself; the 16.2 tok/s helper also
used all-layer DP4A rather than admitted safe43. Wiring the actual candidate
costs **6.832 ms/token plus a 0.117-ms tail**, versus **2.454 ms/token** for the
retained GDN owner, and lowers full decode. Commit `15a436766` clears the dead
binder route. Prefill colwarps 27–47 remains certified. Current
production/strict manifests are `9e27fec0…` / `42509601…`; omitted routes stay
strict.
The certified
compact-WMMA MoE suffix (layers 27–47: Q4_K dual gate/up + Q5_1 down on the
f16-WMMA matrix-core kernels, tile 16×16; replaces the ds4-MMQ suffixes and
strict owners on those layers; `HIPENGINE_QWEN4_EXP_PRODUCTION_MOE_PREFILL=1`)
passes the complete 450-row/three-repeat packet at KL
mean/p95/p99/max `2.79e-4/1.53e-3/3.49e-3/5.98e-3`, **446/450 top-1** (all
scopes ≥ 98.67%), exact repeat/state, 18/18 repeat-exact free generation
(4 task-valid divergences), exact c2 with zero teardown, and improves paired
p508/p1012 **6.572→6.287 s (-4.34%, 80.82 tok/s)** /
**13.398→12.694 s (-5.26%, 79.73 tok/s)**. The layer-27 boundary is the
maximal envelope-admissible suffix (full-layer WMMA screens at mean 5.9e-3).
The certified GDN column-warp suffix (llama gated_delta_net layout, layers
27–47; 4.58× per launch, −17.1%/−15.7% paired p508/p1012, supersedes
peer-GDN) and the iu8-WMMA gate/up suffix (layers 35–47 within the WMMA-MoE27
route; exact Q4_K q values + 3 residual activation planes + min-offset
ds-trick) passes the complete packet at KL mean/p95/p99/max
`2.62e-4/2.20e-3/4.34e-3/5.52e-3`, **446/450 top-1**, zero scope failures,
exact repeat/state, 18/18 repeat-exact free generation (15/18 strict-exact),
exact c2 with zero teardown, and improves paired p508/p1012
**7.430→6.650 s (-10.5%)** / **15.260→13.469 s (-11.7%)** over the f16
production stack under matched conditions; the binder selects it via
`HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL=1` (layers 35–47). Current natural 16K improves **946.999→341.177 s (-63.96%, 47.989 tok/s; chunk-512 gate**
re-passed with retrieval/oracle/transactional/teardown exact) with every
binding control exact; 64K historical evidence is retained but not rerun because
47.989<100 tok/s. 262K is capacity-only (91.126 GB tracked), not inference. Q8 MTP is exact on 10/10
prompts but remains opt-in at **0.955x AR**. A PR-#303 feasibility diagnostic
measures its 675.4-MB full Q8_0 draft head at **3.153 ms (41.3% of a 7.639-ms
draft step)**, but even a free head projects only **0.964x AR** on the retained
suite; target verification and host draft outputs remain first. [`hot-head diagnostic`](results/2026-09-01-gfx1151-qwen38-flash-next-mtp-hot-head-feasibility.json).
<=1K image/video/PNG chat and
request-owned c2 blocking/SSE pass with zero teardown; packed c-aware speed,
remote media, multimodal SSE, and 128K+/262K inference are not claimed.
Evidence: [`gap`](results/2026-08-28-gfx1151-qwen38-flash-next-llamacpp-matched-baseline.json) · [`MoE graph`](results/2026-08-29-gfx1151-qwen38-flash-next-exact-moe-graph-decode.json) · [`production`](results/2026-08-29-gfx1151-qwen38-flash-next-moe27-q8-32-production.json) · [`chunk512`](results/2026-08-29-gfx1151-qwen38-flash-next-prefill-chunk512.json) · [`Q8 MMQ`](results/2026-08-29-gfx1151-qwen38-flash-next-q8-mmq-prefill-production.json) · [`Q5_1 MMQ`](results/2026-08-29-gfx1151-qwen38-flash-next-q5-1-mmq-suffix32-production.json) · [`Q4_K MMQ`](results/2026-08-29-gfx1151-qwen38-flash-next-q4-k-mmq-suffix35-production.json) · [`MMQ+DP4A stack`](results/2026-08-29-gfx1151-qwen38-flash-next-production-mmq-prefill-dp4a43-stack.json) · [`profile manifest`](results/2026-08-29-gfx1151-qwen38-flash-next-production-mmq-profile-manifest.json) · [`peer GDN`](results/2026-08-29-gfx1151-qwen38-flash-next-production-gdn-peer35.json) · [`final campaign`](results/2026-08-29-gfx1151-qwen38-flash-next-prefill-mmq-campaign-final.json) · [`master re-baseline`](results/2026-08-29-gfx1151-qwen38-flash-next-llamacpp-master-rebaseline.json) · [`WMMA MoE27`](results/2026-08-29-gfx1151-qwen38-flash-next-wmma-moe27-production.json) · [`iu8 gate35`](results/2026-08-30-gfx1151-qwen38-flash-next-iu8-wmma-gate35-production.json) · [`GDN colwarps27`](results/2026-08-30-gfx1151-qwen38-flash-next-gdn-colwarps27-production.json) · [`QSA flash (key-parallel 35-47)`](results/2026-08-30-gfx1151-qwen38-flash-next-qsa-flash31-production.json) · [`fresh full profile`](results/2026-08-30-gfx1151-qwen38-flash-next-fresh-full-profile.json) · [`invalid GDN decode correction`](results/2026-08-30-gfx1151-qwen38-flash-next-gdn-colwarps-decode-all.json).

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
