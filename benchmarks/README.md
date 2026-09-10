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

#### NVIDIA RTX PRO 6000 Blackwell — 96 GB (`sm_120a`)

| Model | Quant | Prompt processing | Text generation | With MTP | Max context |
| --- | --- | ---: | ---: | ---: | ---: |
| Maple-Preview | 2-bit | **1917.5** | **402.4** | — | — |

Blank cells are shapes we have not measured yet, not failures. Max context is
published only where a dedicated ceiling run exists.

- **Qwen3.8-27B `Q4_K_M` holds 232,448 tokens of context on a 24 GB `gfx1100` card.** The ceiling depends on the KV format and route — six measured configurations, 40,960 to 232,448; see Long context with DMS for what each applies to. The model's full 262,144 context needs a predicted 24.8 GiB and does not fit. [Capacity evidence](https://github.com/shisa-ai/hipEngine/blob/main/benchmarks/results/2026-09-09-rx7900xtx-gguf-int8-direct-prefill-capacity.json)

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
**W7900, INT8 KV server (2026-09-10, repaired route — lease removed;
layer-outer executor via env flag)**: tier-1 allocation-validity bracket
**passes 65,536 / 98,304 / 131,072 / 155,648 DECLARED contexts** with clean
teardown
([65,536](results/2026-09-10-w7900-server-alloc-probe-64k.json),
[131,072](results/2026-09-10-w7900-server-alloc-probe-128k.json),
[155,648](results/2026-09-10-w7900-server-alloc-probe-152k.json)) — each
point is one 2,048-row request at that declared context, NOT a
full-length-prompt completion. Tier-1 in the capacity protocol's sense:
every allocation succeeding plus one multi-slab prefill and decode with
finite logits. No cross-card comparison to the RX 7900 XTX direct ceiling
(131,072) is claimed - that number is a different card's route.
Admission-safety qualification under load (allocation failure with a live
survivor, operational reserve) remains open, as does a full-length tier-2
confirmation. The old unqualified 54,272 was measured pre-repair with the
duplicate reservation pinned. W7900
default-route prefill at 8,192 tokens: 766.0 tok/s, decode 33.9;
direct-INT8 wmma prefill 735.6 (96%); repaired packed slot-local server
route 677 tok/s — a diagnostic gate run with intermittent competing GPU
work, not a certified throughput claim
([gate](results/2026-09-10-w7900-int8-slot-local-aotriton-gate-corrected.json));
the paired-repetition qualification lives in the W7900 parity section
below.

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

### W7900 Qwen3.8 `Q4_K_M` C1-C8

#### Server/direct parity result (2026-09-10, roadmap P0-P7)

The [server-direct parity campaign](../../docs/SERVER-DIRECT-PARITY-ROADMAP.md)
section 1.1 has the full stage table and status levels. The diagnostic
headline on the repaired C1 INT8 route (slot-local AOTriton prefill;
layer-outer packed executor available via env flag, default OFF pending
packet gates; duplicate packed-KV reservation removed):

- **Speed** (layer-outer executor enabled via its env flag — default OFF
  pending the remaining packet gates): packed prefill 677-732 tok/s at
  2K-8K rows vs the direct scalar control's 685-756 (paired 8,192-row
  repetitions: ratio median 1.004, mean 0.978, min 0.890 under shared-host
  contention; all bitwise-identical logits and IDs)
  ([paired reps](results/2026-09-10-w7900-p7-c1-speed-parity-paired-reps.json)).
- **Decode boundary** (private-session executor-entry comparison, not
  server pool/scheduler execution): R0 raw session 35.15 ms/token median;
  R1 the same session class through the packed entry 35.22 ms (ratio
  1.002). The public LLM arm's whole-request wall (5.00 s incl. prefill)
  is consistent with a small service cost but does not isolate it
  ([ladder](results/2026-09-10-w7900-decode-boundary-ladder-16k.json)).
- **Memory**: canonical 32.5 KiB/L-token KV exactly (2 x 16 x 4 x (256+4));
  zero server-only linear context term on the scoped route (lease 0,
  one shared oracle pair, 0.0625-0.59 GiB from 16K-152K declared)
  ([probe](results/2026-09-10-w7900-server-alloc-probe-16k-p4-lease-removed.json));
  route transients at 32K declared fell ~6.4 GiB -> ~1.6 GiB (-75%).
- **Capacity**: fresh C1 ceiling 155,648+ declared contexts (2.87x the
  pre-repair 54,272; exceeds the plain direct-engine INT8 ceiling of
  131,072).
- **Trace identity**: AOTriton `attn_fwd` for >=512-row rounds, native paged
  prefill for <512-row tails - two kernel identities, as predicted
  ([trace](results/2026-09-10-w7900-layer-outer-trace-identity-1500.csv.gz)).
- **HTTP observation**: blocking 3.895 s vs SSE 3.839 s whole-request walls;
  SSE first chunk at 2.76 s (prefill-bound) and chunk-gap median 34.7 ms.
  These measure delivery cadence and whole-request difference, NOT an
  isolated per-token transport cost (chunks need not map one-to-one to
  tokens); no sub-ms/token transport claim is made. Kernel-family
  attribution: FFN/WMMA 72%, native tail attention 18%, AOTriton round 1%
  ([SSE](results/2026-09-10-w7900-p7-http-transport-budget.json),
  [attribution](results/2026-09-10-w7900-p5-kernel-family-attribution-1500.json)).

Status levels: the layer-outer executor is implementation-landed with
parity/wall/trace/probe diagnostics passed (enable:
`HIPENGINE_GGUF_PACKED_LAYER_OUTER=1`); its packet gates
(layer-boundary/state comparison, exact KV/control fixtures,
shifted/ragged GPU coverage, aliasing, cancellation cleanup) are
outstanding, so it is not yet a qualified default. The lease removal is
default on for the C1/prefix-off/MTP-off route
(`HIPENGINE_GGUF_PACKED_KV_LEASE=1` rolls back). Remaining open packets:
C>1 INT8 (IKV-C2), resumable prefill (P6), and their gated acceptance
items.

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

## Where detailed evidence lives

See result artifacts, [`CHANGELOG.md`](CHANGELOG.md), the
[`harness catalog`](HARNESSES.md), and [`BENCHMARK.md`](../docs/BENCHMARK.md).
Optimization history lives there, not in this current-row scoreboard.

Qwen3.8 runs plain AR on this backend; its speculative rows are retained for
explicit opt-in and re-measurement only.

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
