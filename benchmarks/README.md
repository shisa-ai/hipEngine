# hipEngine Topline Benchmarks

Last updated: **2026-10-23 UTC**

## 16K INT8/MTP with and without DMS — RX 7900 XTX

Qwen3.8-27B `Q4_K_M`, **epyc / RX 7900 XTX**, production INT8 KV/FP32 scales, W8192/2x DMS, MTP budget 3.
Ten category prompts × two passes, one discarded warmup per arm, one active request.
Each request: 16,384 inputs, 63–256 greedy outputs, prefix cache off, no allocation tracing.

| Policy | Prefill tok/s | Decode tok/s | Median time to first text |
| --- | ---: | ---: | ---: |
| INT8 + MTP, DMS off | 645.9 | 28.55 | 25.36 s |
| INT8 + MTP, DMS on | 608.1 | 19.18 | 26.93 s |

Pooled prefill includes MTP priming and first-token overhead; decode divides
output tokens minus one by the first-to-last-text window. DMS lowers prefill **5.8%** and decode **32.8%** relative to DMS off.
All 40 requests used MTP and repeat exact IDs within policy, matching that policy's pre-optimization IDs; 6/10 prompts match across policies.
This is not a task-quality score or an MTP-versus-AR comparison.
[Commands, samples, baseline comparison, route checks, and provenance](results/2026-10-21-xtx-dms-speed-recovery.json).
[Capacity and prefix checks](results/2026-10-21-xtx-dms-capacity-prefix.json): default prefix caching and MTP complete 131,039 inputs plus 32 outputs at configured context 131,072; not a maximum search.

## Gemma 4 on Radeon Pro W7900 — October 16

Gemma 4 26B-A4B `UD-Q4_K_XL`, host **epyc / Radeon Pro W7900**, production
profile, BF16 KV, 512-token prompts and 128 fixed greedy outputs, one warmup
and three samples, measured through `scripts/gemma4_campaign_bench.py`. The
512-token row and MTP result were rerun on October 16; the longer prompt rows
remain from October 15. The MTP result uses the Q8_0 assistant head, budget 4,
18 prompts × 2 samples, and whole-call throughput including prompt processing.

| Prompt tokens | Prefill tok/s | Decode tok/s | llama.cpp prefill | llama.cpp decode | Prefill ratio | Decode ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | **3805.6** | **75.76** | 3165.2 | 69.25 | 1.20x | 1.09x |
| 2048 | 3665.4 | 74.30 | 3687.3 | 65.25 | 0.99x | 1.14x |
| 8192 | 3336.1 | 70.04 | 3557.7 | 62.25 | 0.94x | 1.13x |
| 32768 | 2646.9 | 58.80 | 2502.8 | 52.77 | 1.06x | 1.11x |
| 65536 | 2086.3 | 48.69 | 1800.2 | 44.89 | 1.16x | 1.08x |
| 131072 | 1467.5 | 36.02 | 1139.8 | 34.89 | 1.29x | 1.03x |

Decode leads at every length; prefill trails at 2048 and 8192 and leads
elsewhere. The 32768-131072 decode figures were re-measured after the global
decode step started cutting its key range into one slice per 128 keys past
8192 keys (41.67 to 48.69 tok/s at 65536, 32.79 to 36.02 at 131072, generated
ids equal); the other cells are the paired session. llama.cpp `45451c3` is no faster
than the earlier `8cfc315` comparator on this model and device (3190.5 against
3199.6 prefill and 69.6 against 71.6 decode tok/s at 512).

With the `Q8_0` assistant head and a draft budget of 4, over 18 prompts and
128 tokens, the fresh MTP rerun reaches **120.2 tok/s against 76.9 plain
(1.56x)**. Draft acceptance is 74.1% and all 36 cases pass the production
envelope check. The result includes the short prompt-processing phase.
Artifacts: [plain 512-token rerun](results/2026-10-16-gemma4-gfx1100-plain-512-rerun.json), [MTP rerun](results/2026-10-16-gemma4-gfx1100-mtp-b4-rerun.json), [previous comparison](results/2026-10-15-gemma4-gfx1100-llamacpp-latest-comparison.json), [long-row decode slicing](results/2026-10-15-gemma4-gfx1100-flash-long-rows.json), [expert chunk8](results/2026-10-15-gemma4-gfx1100-expert-chunk8.json), [expert down](results/2026-10-10-gemma4-gfx1100-expert-down-wmma-promotion.json), [prefill attention](results/2026-10-09-gemma4-gfx1100-wmma-promotion-packet.json), [dense f16 weights](results/2026-10-10-gemma4-gfx1100-dense-f16w-prefill.json), [decode graph on this device](results/2026-10-12-gemma4-gfx1100-origin-merge-validation.json), [short-row flash decode](results/2026-10-12-gemma4-gfx1100-flash-decode-short-row.json).

## Gemma 4 staged sliding prefill — October 6

Gemma 4 26B-A4B `UD-Q4_K_XL`, host **epyc / Radeon Pro W7900**, production
profile, BF16 KV, 8192 prompt / 128 greedy outputs. Production now prefers the
staged multi-row attention candidate for one geometry: a sliding prefill block
the tiled and AOTriton routes refuse (a sliding layer whose presented key count
exceeds its window), which the native exact kernel serves by walking all
presented keys from column 0. The candidate keeps the native arithmetic and adds
the per-row window trim, so the substitution is byte-identical on the exercised
matrix.

A same-process paired A/B, where the only difference is
`runner.exact_prefill_attention_variants`, measures:

| Arm | Prefill | Decode | Wall |
| --- | ---: | ---: | ---: |
| native (`gemma4_plain`) | 9.4851 s / 863.7 tok/s | 1.9089 s / 66.53 tok/s | 11.3942 s |
| staged (`gemma4_staged`) | 8.3245 s / 984.1 tok/s | 2.0291 s / 62.59 tok/s | 10.3258 s |

Prefill improves 12.24% and wall 9.38%; **decode is 6.29% slower in the same
pair**. This is a scoped prefill/wall win, not a universal non-regression. The
shipping public path resolves to the staged scope natively and measures prefill
8.2560 s (992.24 tok/s), decode 1.9643 s (64.65 tok/s), wall 10.2359 s and public
wall 10.2485 s, with `public_path_parity` true. An observer records 25 and 175
staged calls at 2049 and 8192 prompt tokens through the public surface with no
launcher replacement. Direct GPU parity (40 BF16/F32 cases, byte-identical
across native / staged-trimmed / staged-untrimmed) and 4 shared-scratch GPU
cases pass, plus 65 CPU integration cases. No MTP or whole-profile quality claim
is made. Full commands, provenance and hashes are in the
[artifact](results/2026-10-06-gfx1100-gemma4-staged-sliding-prefill.json).

## gfx1100 merged baseline and transfer screen — October 4

Gemma 4 26B-A4B `UD-Q4_K_XL`, host **epyc / Radeon Pro W7900**, production
profile, BF16 KV, capacity 8448, 32 greedy outputs: one warmup and three
resident-harness samples per prompt length. Decode rates include full-logit host
collection; public-generation wall time is a separate one-request observation.

| Prompt tokens | Prefill tok/s | Decode tok/s | Public generation wall |
| --- | ---: | ---: | ---: |
| 513 | 1285.1 | 70.13 | 0.837 s |
| 2053 | 1047.3 | 66.13 | 2.486 s |
| 8191 | 808.5 | 66.27 | 10.632 s |

Repeated and public output IDs agree. These are merged-tree snapshots, **not
pre-merge speedups**. First-request timings are saved separately with their
compilation/cache conditions. Six additional public requests confirm native
attention routes and finite, bit-identical full logits between separate and
shared attention owners.

The bounded transfer screen finds two useful Qwen routing candidates on the
**separate RX 7900 XTX lane**: true 16-row Q5 T16 col8 is 2.36–4.10x faster than
the direct leaf across seven shapes with exact parent outputs, and local32 is
1.285x faster for the narrow C1 K5120/N1024 projection with exact outputs.
Those are leaf screens, not public-model gains. The attn_q CPU-reference top-1
floor fails for both control and candidate; no complete promotion gate is
claimed. Six other C1 shapes favor existing local32 over donor tile8. The
narrow 2/4/8-row shape also has faster alternatives, unlike the wider shapes.
All four additional mixed-quant fusions lose to gfx1100's shipping chain; two
IQ-side fusions miss their declared exact contract.

On W7900, transferred staged-attention changes preserve exact outputs in their
explicit diagnostic A/B. Native production does not request staged attention:
compared with native attention, the staged diagnostic has slower short-context
prefill/decode and faster 8191-token prefill. No default changes follow from this
screen. Host rotary reuse saves about 0.8% prefill at the single-block 513-token
prompt; multiblock requests have no effective cross-request hit with the
single-range cache. Full commands, samples, route observations, numerical
findings and archived diagnostic drivers are in the
[transfer artifact](results/2026-10-04-gfx1100-transfer-screen.json).

### Public DMS sidecar diagnostic

On physical host **gfx1151 / Radeon 8060S**, Qwen3.8-27B Q4_K_M,
production profile and INT8 KV, both W256 and W8192 sidecars complete the
public-library comparison but are slower end-to-end than dense KV.
At 12,288 input tokens and 32 greedy output tokens:

| Policy | Mean public wall (s) | Mean backend prefill (s) | Post-prefill inclusive tok/s | Achieved logical/live KV ratio |
| --- | ---: | ---: | ---: | ---: |
| Dense INT8 | 35.751 | 32.741 | 10.810 | 1.000x |
| Local W256 / historical target 2x | 58.005 | 42.188 | 10.847 | 1.954x |
| Published W8192 / historical target 2x | 58.146 | 42.038 | 10.642 | 1.199x |

Four category prompts, one warmup and two measured repeats each; singleton,
MTP/prefix cache off, FP32 scales. Post-prefill rates use 31 tokens over
backend request-minus-prefill, not public wall time. All arms also complete
128- and 2,048-token inputs with exact same-arm repeatability. Both sidecars
pass the bounded dense-teacher numerical diagnostic (100% top-1 across tested
categories); repeated background and short outputs do not certify realistic
long-context task quality. Policies differ in weights and thresholds as well
as window size. [Commands, provenance, memory and category results](results/2026-09-26-gfx1151-dms-public-sidecars.json).

Latest results (2026-09-26). **Halo-box Vulkan MTP** (`strix-llama.cpp@03895887abe6`) completes the
Qwen3.6 Q4_K_M 140-turn replay on Radeon 8060S in **643.08 s** (mean **4.593 s**) and passes
**77/96** frozen BFCL diagnostic cases; saved hipEngine/llama.cpp HIP MTP times are 793.15/827.10 s
with corrected quality 75/96 and 76/96, and gufo is 1,041.49 s and 73/96. The paired **96-case BFCL
quality diagnostic** scores hipEngine **75/96**, llama.cpp **76/96**, and gufo **73/96**. Different
backends and outputs preclude an isolated kernel-throughput or accuracy-parity claim.
[Four-engine table and protocol](mlperf-edge/README.md#latest-halo-box-vulkan-comparison-2026-09-26)
· [Speed artifact](results/2026-09-26-gfx1151-halo-edge140-bfcl96.json)
· [Quality artifact](results/2026-09-26-gfx1151-bfcl96-parallel-default.json)
· [MLPerf Edge detail](mlperf-edge/README.md).

The dated measurement feed that previously led this file — MLPerf 140-turn screens, GGUF native
sampling, the September 20 W7900/gfx1151 C1 refreshes, prefix-cache and merge-qualification packets,
and the Surya OCR lane comparisons — is archived in
[`HISTORY.md`](HISTORY.md#moved-from-the-compact-scoreboard-2026-10-06).

Superseded checkpoint ladders, dated engagement notes and the detailed
cross-engine prefill screens live in [`HISTORY.md`](HISTORY.md); this file
keeps the current rows and their evidence links.

This file is the current benchmark scoreboard. It intentionally contains only current user-facing
results, compact protocol/status notes, and links to the authoritative evidence.
It is not an optimization journal.

UD/main integration validation is documented in the
[integration report](../docs/campaigns/UD-MAIN-INTEGRATION.md); its working-tree timings do not
replace the published performance rows.

## Qwen3.8 Flash Next HTTP prefill and decode

On Framework Desktop / Radeon 8060S (`gfx1151`), Qwen3.8-Flash-Next
`UD-Q4_K_XL` with a `Q8_0` MTP sidecar, production profile and BF16 KV,
the October 22 HTTP comparison isolates batch-decode attention and selection
changes. Both arms use one active request, fresh prompts, prefix cache off,
64 greedy outputs, MTP budget 3 and up to 4096-token idle prefill chunks.
Each cell is the median of four categories and three measured repetitions
after one warmup per case and arm.

| Fresh prompt | AR prefill before → after | MTP prefill before → after | AR decode before → after | MTP decode before → after |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 1319.3 → 1314.3 | 1100.8 → 1091.2 | 15.315 → 17.082 (+11.54%) | 37.986 → 36.799 (−3.13%) |
| 4,096 | 1654.4 → 1644.6 | 1325.5 → 1305.5 | 12.153 → 14.676 (+20.76%) | 37.440 → 36.783 (−1.75%) |

All rates are tok/s. MTP prefill includes predictor priming; decode measures
63 first-to-last-output transitions, not whole-request throughput. All 208
canonical and category/heldout outputs match their baselines. A counterbalanced
old/new MTP rerun did not reproduce the table's slowdown: median paired change
−0.001%, with all eight cases between −0.194% and +0.016%. No MTP gain is claimed.
These are diagnostic timings: inherited strict-teacher numerical findings remain open,
and this exact change does not establish a new numerical-quality verdict.
[Protocol, samples and correctness checks](results/2026-10-22-gfx1151-flashnext-batch-decode.json).

An October 23 comparison on the same host, model and execution profile reduces
MTP partial-acceptance restoration overhead by combining cursor uploads.
The synchronized diagnostic covers 56 partial restores across nine requests
at 1,024, 1,537 and 4,096 prompt tokens, each generating 32 tokens.

| Partial-acceptance restore | Before | After |
| --- | ---: | ---: |
| Median restore window | 1.503 ms | 1.211 ms (−19.4%) |
| Cursor uploads per restore | 24 | 1 |

The separate uninstrumented 208-request comparison preserves every output and
shows no per-case median decode regression. This is a restore-window improvement,
not an isolated end-to-end speedup claim.
[Commands, state checks and all timing samples](results/2026-10-23-gfx1151-flashnext-mtp-cursors.json).

A subsequent same-host comparison enables a request-local subset of the MTP
output head: a fixed token prefix plus special tokens and committed-history
rows. Target verification is unchanged. Against the cursor-upload baseline,
median MTP decode improves **6.10–6.79%** across the eight 1K/4K canonical
cases using the same protocol above. The 18 category/heldout cases use natural
prompt lengths and 32 outputs (31 decode transitions), with the same warmup,
repetition and alternating-arm schedule. In those cases, code
and English improve **6.59–7.48%**, Japanese changes by −0.03% to +0.19%, and
mixed-language cases improve 0.08–3.11%. Japanese requests mostly select the
full-head fallback because of recent vocabulary misses. AR changes by less
than 0.1% in either direction.

All 208 output sequences and acceptance rates match the previous baseline.
Greedy and sampled public controls, repeatability and cancellation/reuse checks
pass. These are native diagnostic timings, not a new cross-engine comparison
or a resolution of the inherited numerical findings above.
[Source, commands, route checks and per-case samples](results/2026-10-23-gfx1151-flashnext-mtp-subset-head.json).

## Qwen3.8 Flash Next current production rates

On Framework Desktop / Radeon 8060S/gfx1151, Qwen3.8-Flash-Next UD-Q4_K_XL,
`production` profile (the default), BF16 KV, chunk 4096, greedy decode (compact
capture), canonical 12-case protocol with one warmup and three repetitions.
The 512/1024/4096 rows were refreshed on 2026-10-19 UTC at clean committed
source; the 8192/16384 rows are the October 14 measurements.

| Prompt length | Prefill tok/s | Decode (128) tok/s |
| ---: | ---: | ---: |
| 512 | **1036.2** | **23.68** |
| 1024 | **1323.9** | **22.72** |
| 4096 | **1650.7** | **23.24** |
| 8192 | **1621.5** | **23.18** |
| 16384 | **1576.1** | **23.04** |

The 8192- and 16384-token rows come from a second 12-case fixture (1024 /
8192 / 16384 tokens) run under the same protocol; its 1024-token cases are
token-identical to the canonical ones and measured 1321.6.

The production plan runs the routed experts, the dense projections at every
layer, the hyper-connection, indexer and router projections and the prefill
attention on F16-operand matrix kernels with F32 accumulation, and a
row-split GDN recurrence at every GDN layer; every route keeps a registered
strict fallback. The hyper-connection read runs as three launches that share
one F16 image of its output with the projections that follow. Decode uses a
row-split GDN step and warp-per-expert routed-expert kernels, and runs the
one-row hyper-connection read in three launches; decode measured 20.88 /
20.12 / 20.53 tok/s before those changes the same day. The same host
and protocol measured 779.4 / 977.9 / 1190.3 prefill at 512 / 1024 / 4096
tokens on 2026-10-13, before the hyper-connection, GDN and expert-glue
kernel work, and 305.6 / 331.6 / 335.5 for the plan without the F16 routes.

Against the `strict` teacher on the refreshed 18-prompt full-vocabulary gate
(450 teacher-forced rows): mean KL 6.9e-4, p95 4.4e-3, max 0.0146, top-1
99.1%, bit-identical repeats. All 18 strict and all 18 production free-run
hashes match the previously reviewed 13/18 strict-exact task packet. On the
October 14 18-prompt cross-engine battery: 13/18 strict-exact, top-1 agreement
1.00, top-20 mean KL 0.077 (the plan without the F16 routes: 15/18 and
0.0052). On four prompts of about 2,000 to 2,600 tokens the plan measures mean KL 2.1e-3
against `strict`; long prompts are gated against what a single rounding flip
does on the same prompts (floor 5.0e-3), plus retrieval answers at 8K and
16K tokens, and the plan passes (16/16 answers, the same as `strict`).
[Current rates and fresh correctness gate](results/2026-10-19-gfx1151-flashnext-topline/summary.json)
· [protocol and samples](results/2026-10-19-gfx1151-flashnext-topline/README.md)
· [October 14 rates](results/2026-10-14-qwen4exp-prefill-select-fusion/README.md)
· [long-prompt gate](results/2026-10-14-qwen4exp-long-context-gate/README.md)
· [gates and decode changes](results/2026-10-14-qwen4exp-decode-batch/README.md)
· [prefill changes and kernel attribution](results/2026-10-14-qwen4exp-prefill-hc-gdn-batch/README.md)
· [long-context isolation](results/2026-10-14-qwen4exp-production-f16-composition/README.md)
· [attention kernel](results/2026-10-14-qwen4exp-fast-qsa-wmma/README.md)
· [GDN norm/gate kernel](results/2026-10-14-qwen4exp-gdn-norm-gate-waves/README.md)
· [indexer and router GEMM](results/2026-10-13-qwen4exp-fast-indexer-f16-gemm/README.md)
· [dense and hyper-connection routes](results/2026-10-13-qwen4exp-fast-dense-hc-router/README.md)
· [routed-expert kernel](results/2026-10-13-qwen4exp-fast-routed-f16/README.md).

With the Q8_0 MTP sidecar (explicit speculative provider, greedy, budget 3),
the October 19 Framework refresh measures 28.930 / 28.928 tok/s over two runs
of the full ten-prompt mtp-bench category suite, against 22.670 / 22.671 tok/s
true AR in the same process (1.276x), 64 tokens per prompt. Eight category
heldouts measure 28.164 tok/s and 1.248x AR. These request-wall rates have a
different timing window from the canonical decode-only column. Public
`LLM.generate()` checks execute plain and MTP paths with the same production
manifest; plain IDs match the canonical prefix.
[Fresh MTP measurements](results/2026-10-19-gfx1151-flashnext-topline/README.md).

The October 18 zbook MTP measurements are a separate physical-host lane:
23.8–24.5 tok/s in-process, 1.16x matched AR. Its served MTP measured 1.26x
one request, 1.33x at two concurrent requests and 0.87x at four. Those are
historical measurements, not a comparison with this Framework refresh or
with the changed serving defaults.
[Zbook measurements](results/2026-10-18-qwen4exp-mtp-economics/README.md).

Served surface, 2026-10-18: at `max_context_tokens=32768` and two slots, all
50 HTTP grid probes return 200 with 24 rows on the AR route and 24 on MTP
verified from each response payload; the 29,912-token retrieval prompt returns
its needle on both routes (33.19 s AR, 91.01 s MTP); and a three-round
mixed-traffic soak passes 35 checks with per-round `/ready` and recovery,
including the retryable `429` backpressure path.
[Grid](results/2026-10-18-zbook-qwen4exp-http-grid/summary.json)
· [soak](results/2026-10-18-zbook-qwen4exp-http-soak.json)
· [qualification record](../docs/QWEN4EXP-PRODUCTION-SERVING.md).

The opt-in `fast` profile is where a reduced-precision route lands before
its production gate. It currently selects the same routes as `production`.

## Cross-engine comparison on the same `UD-Q4_K_XL` GGUF file

The latest apples-to-apples comparison uses the same four-shard
`Qwen3.8-Flash-Next` `UD-Q4_K_XL` file, the committed 12-case exact-token
fixture, one warmup, three measured repetitions, and the same Radeon 8060S /
gfx1151 host. Prefill rates are tok/s; each column is the equal-weight mean of
per-case medians.

| Engine | p512 | p1024 | p4096 | Decode at p4096 |
| --- | ---: | ---: | ---: | ---: |
| hipEngine production (2026-10-14) | 1037.2 | 1314.2 | 1615.5 | 23.1 |
| halo-box `strix-llama.cpp` master `6aa81b88e` (2026-10-12) | 759.3 | 1109.1 | 1446.4 | 25.1 |
| gufo `v0.12.0` (2026-10-12) | 1095.0 | 1482.2 | 1736.4 | 28.1 |

hipEngine's prefill rate is ahead of halo-box at every length and at 0.95x,
0.89x and 0.93x of gufo at p512, p1024 and p4096; gufo's decode rate is
1.22x and halo-box's 1.09x hipEngine's. The external rows were measured two
days earlier on the same host and have not been rerun. The hipEngine column
is the weighted rate per prompt length.
The 18-prompt quality battery reports 13/18 strict-exact for hipEngine,
11/18 for halo-box, and 13/18 for gufo; aligned top-20 mean KL against the
hipEngine strict teacher is 0.077 for hipEngine production and 0.241 for
halo-box, both with top-1 agreement 1.00, while gufo does not expose
logprobs for that comparison. These are characterization results, not
external-engine certifications. See the [pass-1 artifact for the external
engines](results/2026-10-12-qwen4exp-cross-engine-pass1/README.md)
and [canonical protocol](../docs/CROSS-ENGINE-PROTOCOL.md).

Kyojin is recorded as an alternate engine/implementation only. It uses a
different EXL3 model pack rather than the Q4 GGUF file, so it is not part of
these apples-to-apples rows or ratios; its measurement remains blocked by the
current disk-space constraint.


On Radeon 8060S/gfx1151 host `55ea6c509d0b49eea8de7094a1023668`, a checked
32-bit/full-four-row sibling of the current dense Q8 DPP owner was measured
using Qwen3.8-Flash-Next UD-Q4_K_XL weights, synthetic F32 activations and
20 counterbalanced synchronized pairs per case. The tracked-clean screen
covers eight projection tensors at layers 0/22 and rows 512/1024/4096.

| Rows | Cases | Geometric-mean operation speedup | Range |
| ---: | ---: | ---: | ---: |
| 512 | 16 | 1.420x | 1.240–1.506x |
| 1024 | 16 | 1.424x | 1.238–1.506x |
| 4096 | 16 | 1.398x | 1.188–1.473x |

All 48 medians and both arm-order strata improve; the aggregate geometric
mean is 1.414x (29.2% less projection-operation time). Full outputs match the
parent before and after each timing loop. CPU-reference, guarded GPU parity,
byte-range admission and cached engagement checks pass. This is synthetic-input
operation evidence, not model throughput. Natural-input and model-state gates
are pending; the production selector is unchanged.
[Protocol, identities, samples and artifact](results/2026-10-07-qwen4exp-dense-dpp-checked-full-rows/README.md).

## Qwen3.8 Flash Next dense Q8 checked/full-row qualification

On the Radeon 8060S/gfx1151 host `55ea6c509d0b49eea8de7094a1023668`, the
separately registered checked-int32/full-four-row dense Q8 DPP sibling was
qualified on natural activations and complete model state. The candidate
remains diagnostic-only; the generic DPP parent remains the production path.

The natural-input owner screen covers six canonical cases and six declared Q8
geometries with **594 candidate captures**. Complete owner-operation ratios
are 1.454x at p512, 1.458x at p1024 and 1.466x–1.467x across four category
p4096 cases; geometric mean is **1.463x**. Every capture is full-output
bit-identical, input/raw-weight immutable and finite; teardown is clean.

The full-state gate covers all 12 canonical category/shape cases with
parent/candidate/candidate/candidate/parent arms, 16 decode steps and full KV
hashes: **3,564 candidate calls**, exact prefill/decode logits, exact recurrent
state/layout, exact full KV bytes and exact token IDs/repeats, with zero
teardown allocations. This is operation and correctness evidence, not request
throughput. A complete-engine A/B remains required before any default decision.
[Protocol, artifacts and identities](results/2026-10-07-qwen4exp-dense-dpp-checked-full-rows-qualification/README.md).

## Qwen3.8 Flash Next DP-2 rows-4096-only engine qualification

On the Radeon 8060S/gfx1151 host `55ea6c509d0b49eea8de7094a1023668`, the
production/BF16 route was measured in one tracked-clean process and one model
residency at commit `b59b2872068237631e869f09b25bf3976d8ec361`. The run covers
12 canonical cases plus 16 category-heldouts, one warmup, three counterbalanced
measurements per arm and 128 decode transitions. Admission is only rows=4096;
p512 and p1024 are zero-engagement controls.

| Prompt shape | Parent prefill tok/s | Candidate prefill tok/s | Candidate / parent | Parent decode tok/s | Candidate decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 294.748 | 294.777 | 1.000098x | 20.986 | 20.979 |
| 1024 | 321.450 | 321.377 | 0.999771x | 19.500 | 19.513 |
| 4096 | 324.521 | 325.134 | 1.001890x | 10.555 | 10.543 |

At p4096, all four canonical categories engage exactly 61 ordered pairs per
case. Cross-arm output IDs are exact and all 28 cases are deterministic. The
candidate's p4096 client-wall total is 0.999610x parent, but this small
complete-engine difference is not promoted as a model-wall win. The bounded
candidate remains opt-in/default-off; no unmeasured row or scope is inferred.

[Protocol, full artifact and decision](results/2026-10-06-qwen4exp-dp2-shared-cast/README.md).


On Radeon 8060S host `55ea6c509d0b49eea8de7094a1023668`, the gfx1151
UD-Q4_K_XL production profile publishes existing F32 grouped norm and identical
staging F16 bytes in one kernel, then runs the padded HC-down GEMM. Admission
is exactly 4096 rows, K10240/N320, branches4/H2560, layers13–47 HC-down roles.
The immediate-use handoff borrows the session workspace and eliminates 70
staging casts per qualified prefill without adding device allocations.
Complete norm+cast+GEMM event timing over six natural cases/420 exact calls:

| Natural input | Parent owner ms | Publisher owner ms | Speedup |
| --- | ---: | ---: | ---: |
| code p512 | 57.277 | 54.074 | 1.059x |
| code p1024 | 93.045 | 99.133 | 0.939x |
| four categories p4096 | 358.844–360.867 | 326.934–328.139 | 1.096–1.104x |

The 1024-row screen is slower and is not integrated; 512/1024 and unmeasured
rows keep parent norm+cast. All outputs/input bytes and teardown pass; kernel
scratch is zero. Fourteen native canonical/heldout state cases are exact with
full KV, 2,520 candidate publisher/consumer calls, zero decode calls and zero
teardown allocations. The tracked-clean 12-case A/B gives a 1.0015x weighted
4096 prefill ratio (322.618→323.115 tok/s, +0.154%). Aggregate request wall
is 0.9970x, with a 2.77% general-English p4096 request regression. The measured
benefit is the bounded prefill operation, not an all-request or decode speedup.
The small engine prefill effect is within run variation (maximum within-arm
prefill CV 0.608%; one paired stratum 0.9999x).
[Protocol, samples and qualification artifact](results/2026-10-06-qwen4exp-norm-f16-publish/README.md).

## Qwen3.8 Flash Next padded dense-wide tail production route

On Radeon 8060S host `55ea6c509d0b49eea8de7094a1023668`, UD-Q4_K_XL,
BF16 KV and chunk4096, the gfx1151 production profile selects a padded
F16-input Q8_0 HC-down leaf only at K10240/N320 and 512–4096 rows.
The same-host comparison uses all 12 canonical cases, one model residency
and three counterbalanced measurements per arm, with 128 decode transitions.

| Prompt tokens | Parent prefill tok/s | Padded prefill tok/s | Change |
| ---: | ---: | ---: | ---: |
| 512 | 291.581 | 294.602 | +1.036% |
| 1024 | 317.528 | 319.415 | +0.594% |
| 4096 | 324.275 | 325.719 | +0.445% |

All 12 cases improve prefill; all 72 measured trajectories are exact and
deterministic. Total client wall falls 493.851329→493.049429 s (−0.162%);
ten request cases improve, but Japanese 4K is 0.214% slower and mixed 512
is 0.279% slower. Decode by shape changes +0.018%/−0.089%/−0.200%; no
decode-speed or all-case request-wall claim follows.

The natural complete-GEMM owner improves 1.312–1.361x over six cases/420
exact input-preserving calls, with private scratch 92→0 B/lane and
spills 58→0. All 20 canonical/heldout full-logit/state/KV cases pass three
candidate repeats and clean teardown. Strict, Q4_K_M and peer backends
keep their preceding routes.
[Protocol, correctness and artifacts](results/2026-10-06-qwen4exp-dense-wide-padded-gate/README.md).

## Qwen3.8 Flash Next checked-index GDN production route

On Radeon 8060S host `55ea6c509d0b49eea8de7094a1023668`, UD-Q4_K_XL,
BF16 KV and chunk4096, the gfx1151 production profile selects checked
int32 indexing for the serial D128 GDN leaf at 2–4096 rows. A same-source,
same-residency A/B uses all 12 canonical cases and three counterbalanced
measurements per arm, with 128 decode transitions.

| Prompt tokens | Wide-index prefill tok/s | Checked-index prefill tok/s | Change |
| ---: | ---: | ---: | ---: |
| 512 | 291.658 | 292.093 | +0.149% |
| 1024 | 317.351 | 318.314 | +0.303% |
| 4096 | 324.199 | 324.936 | +0.227% |

All 72 measured trajectories are exact and deterministic. Natural-model GDN
owner timing improves 1.027–1.044x over six cases/216 exact calls; all 20
canonical/heldout full-logit/state/KV cases pass three candidate repeats and
leak-free teardown. Private scratch drops 36→28 bytes/lane; static VGPR
remains 256. Strict, Q4_K_M production and shapes outside the bound do not
select the checked-index leaf.

This is a qualified owner-time improvement with a small aggregate prefill
gain, not an all-case request-wall win. Japanese p512 prefill is 0.008% slower
(max within-arm prefill CV 0.629%); three request cases are slower, worst
mixed p512 at 0.546%. Both prefill order strata improve; decode is unchanged
and no decode-speed claim follows.
[Protocol and artifacts](results/2026-10-06-qwen4exp-gdn-narrow-index/README.md).

## Qwen3.8 Flash Next ordered router DPP production route

On Radeon 8060S host `55ea6c509d0b49eea8de7094a1023668`, UD-Q4_K_XL,
BF16 KV and chunk4096, ordered router DPP is the gfx1151 production
prefill route. The comparison uses one model residency and three
counterbalanced measurements per arm for all 12 canonical cases.

| Prompt tokens | Parent prefill tok/s | DPP prefill tok/s | Change |
| ---: | ---: | ---: | ---: |
| 512 | 287.913 | 289.505 | +0.553% |
| 1024 | 312.834 | 314.353 | +0.486% |
| 4096 | 319.356 | 321.238 | +0.589% |

All cases improve prefill; all 72 measured output trajectories are exact
and deterministic. The D128 full-logit/state/KV gate and 576 natural-router
captures match the parent; four GPU row-isolation/ownership tests pass.
Strict and Q4_K_M production keep the tile4 parent. No decode-speed claim:
the original p4096 complete-request aggregate is 0.364% worse, with one
abrupt decode slowdown. A focused same-protocol recheck does not reproduce
that slowdown; it does not replace the full-suite samples.
[Protocol and artifacts](results/2026-10-05-qwen4exp-router-dpp/README.md).

## Qwen3.8 Flash Next GR-up DPP production route

On physical host `gfx1151` (`55ea6c509d0b49eea8de7094a1023668`), Radeon
8060S, UD-Q4_K_XL, the ordered DPP reduction is enabled in the production
profile for the Q8_0 GR-up owner. The tracked-clean canonical A/B used one
process and one model residency, three measured repetitions per arm, 128 decode
transitions, and a 4096-token prefill chunk.

| Prompt tokens | Prefill before (tok/s) | Prefill after (tok/s) | Change |
| ---: | ---: | ---: | ---: |
| 512 | 271.63 | 289.49 | **+6.57%** |
| 1024 | 294.31 | 315.36 | **+7.15%** |
| 4096 | 302.20 | 323.28 | **+6.98%** |

All 12 canonical cases improve in prefill, all 72 measured output trajectories
are exact across arms, and within-mode runs are deterministic. Candidate
engagement is 0/96/0 across the before/after/before sequence. Decode is
reported as non-regressive but not as a speed improvement. The exact fused
parent remains the strict fallback. [Protocol and artifacts](results/2026-10-05-qwen4exp-gr-dpp-reduction/README.md).

## Qwen3.8 Flash Next F16-input prefill comparison

Measured on physical host `gfx1151` (`55ea6c509d0b49eea8de7094a1023668`),
AMD gfx1151, UD-Q4_K_XL, BF16 KV, warm PLE, 1024-token chunks. Each value is
the mean of four category-specific median rates from three interleaved repeats.
F16 input is the default; F32 input is its same-session rollback baseline.

| Prompt tokens | F32 input (tok/s) | F16 input (tok/s) | Change |
| --- | ---: | ---: | ---: |
| 512 | 235.5 | 243.5 | +3.39% |
| 1024 | 250.3 | 259.2 | +3.57% |
| 4096 | 241.5 | 249.7 | +3.40% |

All 12 final-logit hashes match. A separate 60-trajectory check matches logits,
tokens and recurrent/KV/index state through 64 decode steps, with zero live
allocations after close. This comparison does not qualify the underlying
production arithmetic against strict or establish decode/MTP speedups.
[Protocol and artifacts](results/2026-09-26-qwen4exp-f16-integration/README.md).

## Qwen3.8 Flash Next prefill chunk size

Measured on physical host `gfx1151` (`55ea6c509d0b49eea8de7094a1023668`), AMD
gfx1151, UD-Q4_K_XL, BF16 KV, production execution profile, cached build.
Three repetitions per arm, one process per arm.

**The serving default is now 1024.** It was 256
(`EngineLoopConfig.max_prefill_chunk_tokens`), which split the engine loop's own
1024-token round prefill budget into four chunks. The two knobs are now equal.

All 12 canonical cases (4 categories x 3 prompt sizes), both arms in one run:

| Prompt tokens | 256 -> 1024 | Peak cost |
| ---: | ---: | ---: |
| 512 | -29.0% to -29.4% | +0.77 GB |
| 1024 | -32.9% to -34.4% | +1.24 GB |
| 4096 | -32.9% to -35.7% | +1.24 GB |

12/12 cases faster, 12/12 arms separated, `token_id` identical on 12/12.
`logits_sha256` differs from 256 on 12/12, so this is an arithmetic change
against the old default rather than a pure scheduling one; chunk 1024 is the
configuration the numerical gate certifies. The workspace is
`min(chunk, max_sequence_length)`, so a 512-token request resolves to the same
768 rows either way and short requests are unaffected in time and memory.

The table above is a prefill-only measurement (`qwen4exp_profile_gap.py --mode
prefill`); the per-tick prefill bound that protects decode is the round budget,
which is unchanged.

**The decode side of the same knob is measured.** At `code-p4096`, chunk 1024
against chunk 512 is **+9.06% prefill and -1.15% decode**, measured A/B/A/B/A/B
in one session with every arm separated and within-arm spread of 0.05-0.48%. The
decode cost of the chunk default is therefore eight times smaller than its
prefill benefit. The same comparison run sequentially reads -7.1% decode,
because a non-interleaved `code-p4096` A/B on this host drifts 8%; interleaving
is required to resolve anything below that. [Chunk trade protocol and
limits](results/2026-10-04-qwen4exp-chunk-decode-trade/README.md).

[256 -> 1024 protocol and full table](results/2026-09-29-qwen4exp-rt1-256-vs-1024/README.md).

The 1024 -> 4096 step was measured separately on `code-p4096`: 15.625 -> 14.854 s
(**-4.94%**) for +4.96 GB peak, `token_id` and `logits_sha256` identical on every
arm. [Protocol, cost and
limits](results/2026-09-29-qwen4exp-rt1-chunk-remeasure/README.md).

**The production default is now a capacity guard, not a fixed chunk.** The
workspace is context-aware rather than prompt-aware — rows are
`min(chunk, max_sequence_length)` and the runner's `max_sequence_length` comes
from the model's native context — so a blanket larger default would be paid at
session construction for every session regardless of traffic. The guard takes
the largest chunk whose admission still yields the requested context and falls
back 4096 -> 2048 -> 1024 when the device cannot afford it, so a device that
cannot hold the top rung gets a smaller one instead of losing context. Measured
on `code-p4096`, production, 6 samples per arm, interleaved:

| chunk | prefill tok/s | decode tok/s |
| ---: | ---: | ---: |
| 1024 | 279.28 | 18.127 |
| 2048 | 292.59 (+4.77%) | 18.152 (+0.14%) |
| 4096 | **302.81 (+8.43%)** | **18.431 (+1.68%)** |

Chunk 4096 also improves **decode** by 1.68% rather than costing it. At native
context the scratch reservation is unchanged at 2048 (still under the 4 GiB
per-runner floor) and rises +6.80 GB at 4096; max context stays 262144 tokens at
every rung. [Guard protocol, gate and memory
table](results/2026-10-04-qwen4exp-prefill-chunk-4096/README.md).

## Qwen3.8 Flash Next full-attention context route

Measured on physical host `gfx1151` (`55ea6c509d0b49eea8de7094a1023668`), AMD
gfx1151, UD-Q4_K_XL, BF16 KV, block_size 256, 1024-token chunks, code-p4096
prefill, three repetitions, both arms in one session.

The full-attention decode-context leaf now uses an int32 per-token offset table
and a four-token QK unroll. Both changes leave every floating-point operation
alone, so the route is byte-identical to the one it replaces. A per-block base
table was implemented and rejected: it halves the shared-memory request and
doubles residency, but measured 1.41x slower.

| Metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| Category mean p4096, tokens/s | 248.487 | 255.066 | +2.65% |
| Prefill wall median, code-p4096 (s) | 16.6015 | 16.2027 | -398.8 ms (-2.40%) |
| Largest launch, 1024 rows at `max_context_len` 2048 (ms) | 54.611 | 28.563 | 1.912x |

All four mtp-bench categories at p4096 improve, from +2.23% to +3.47%. Output is
byte-identical (`maxdiff = 0.000e+00` over the full 1024 x 24 x 256 output), so
no KL or top-1 envelope applies. The 512- and 1024-token prompt lengths were not
re-measured for this change.
[Protocol, lever isolation, the rejected lever and artifacts](results/2026-09-28-fullattn-qk4-promoted/README.md).
The superseded two-lever route is in
[its own artifact](results/2026-09-27-fullattn-context-promoted/README.md).

## Root README performance summary

The root README exports this compact retained summary verbatim.

<!-- BEGIN TOPLINE:README_HIGHLIGHTS -->
Tokens/s on each named host: **prompt processing** is input, **text generation**
is output. **MTP** is speculative decoding. Dashes are unmeasured; **Max
context** is a capacity limit, not a full-length prefill.

### Radeon Pro W7900 — 48 GB (`gfx1100`)

| Model | Quant | Prompt processing | Text generation | With MTP | Max context |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | ParoQuant W4 | **2879.4** | **113.3** | 115.8 | — |
| Qwen3.6-35B-A3B | GGUF `UD-Q4_K_M` | **2924.5** | **95.0** | 122.7 | — |
| Qwen3.8-27B Dense | GGUF `Q4_K_M` (BF16 KV) | **898.6** | **30.8** | 39.7 | — |
| Qwen3.8-27B Dense | GGUF `Q4_K_M` (INT8 KV) | 868.6 | 27.9 | — | **176,128** |
| Gemma 4 26B-A4B | GGUF `UD-Q4_K_XL` | **3805.6** | **75.8** | **120.2** | — |
| Laguna S 2.1 | GGUF `UD-Q2_K_XL` | **440.9** | — | — | — |

### Strix Halo / Radeon 8060S — 120 GB (`gfx1151`)

| Model | Quant | Prompt processing | Text generation | With MTP |
| --- | --- | ---: | ---: | ---: |
| Maple-Preview | 2-bit | **754.5** | **153.2** | — |
| Qwen3.6-35B-A3B | GGUF `UD-Q4_K_M` | **1418.1** | **56.7** | 80.1 |
| Gemma 4 26B-A4B | GGUF `UD-Q4_K_XL` | **1865.6** | **43.3** | **69.4** |
| Qwen3.8 Flash-Next | GGUF `UD-Q4_K_XL` | **1036.2** | **23.7** | **28.9** |
| Laguna S 2.1 | GGUF `Q4_K_M` | **654.2** | **23.2** | — |
| Qwen3.8-27B Dense | GGUF `Q4_K_S` | **396.1** | **13.1** | **23.9** |
| Qwen3.8-27B Dense | GGUF `Q4_K_M` | **404.5** | **12.15** | **22.4** |
| Qwen3.8-27B Dense | GGUF `Q4_K_M` (INT8 KV) | — | **10.84** | **19.3** |

### NVIDIA RTX PRO 6000 Blackwell — 96 GB (`sm_120a`)

| Model | Quant | Prompt processing | Text generation | With MTP | Max context |
| --- | --- | ---: | ---: | ---: | ---: |
| Maple-Preview | 2-bit | **1917.5** | **402.4** | — | — |

### Long context on a 24 GB GPU

Using a source checkout: [Qwen3.8-27B `Q4_K_M`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/65ca473/Qwen3.8-27B-Q4_K_M.gguf)
on RX 7900 XTX, with `--max-active-requests 1`. Add the
[DMS checkpoint](https://huggingface.co/shisa-ai/Qwen3.8-27B-Q4_K_M-DMS-W8192)
for long context. Omit `--max-context-tokens` to let the server calculate the
context from available GPU memory.

| KV configuration | Auto context, MTP on | Auto context, MTP off |
| --- | ---: | ---: |
| BF16 KV | 61,440 | 83,456 |
| INT8 KV | 71,680 | 98,304 |
| DMS INT8 | 262,144 | 262,144 |

### Prefix caching

Multi-turn conversations resend the whole transcript, so hipEngine reuses the
KV pages and hybrid state of any 256-token-aligned prefix a later request
repeats. Radix is the default for both the direct engine and HTTP serving
(`--prefix-cache off` to disable). Qwen3.6-35B-A3B `UD-Q4_K_M` on Strix Halo:

| Lane | Cache off | Cache on | Wall time |
| --- | ---: | ---: | ---: |
| Coding, transcript resent | 10.09 tok/s | **16.49** tok/s | **-38.8%** |
| Coding, transcript rebuilt | 10.13 tok/s | **12.67** tok/s | **-20.0%** |
| Chat (ShareGPT) | 29.50 tok/s | 29.15 tok/s | +1.2% |
| All lanes | 16.93 tok/s | **20.81** tok/s | **-18.6%** |

### Serving several requests at once

Aggregate tokens/s across all active requests, Qwen3.8-27B `Q4_K_M` on the W7900:

| Requests | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hipEngine | **23.6** | **39.1** | **53.1** | **63.9** | **72.8** | **79.5** | **83.2** | **85.9** |
| llama.cpp HIP | 21.0 | 34.4 | 30.6 | 27.7 | 36.7 | 46.4 | 52.1 | 58.4 |
| hipEngine advantage | +12% | +14% | +74% | +130% | +99% | +71% | +60% | **+47%** |
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

### Gemma 4 26B-A4B `UD-Q4_K_XL` — RX 7900 XTX

Gemma 4 26B-A4B `UD-Q4_K_XL` on one RX 7900 XTX: single request, greedy,
BF16 KV, 8192-token context, one full-shape warmup and two to three measured
samples per row. hipEngine and llama.cpp run the identical frozen prompt
token-id sequence; decode tok/s is the 127 single-token forwards of a
128-output row divided by their measured time. hipEngine rates come from the
public `LLM.generate()` path, whose output token ids match the instrumented
timing loop at every position on every row.

The two-phase decode split previously accumulated each key slice separately and
summed the partials, which breached the production `kl_max` bar on 1 of 1023
teacher-forced rows. Its replacement partitions output dimensions instead, so
every output keeps the single-kernel path's full ascending-key accumulation
chain: the 1023-row comparison now reports zero KL and 100% top-1 agreement,
strict and default logits are bit-identical on eight category prompts with
repeats, and greedy public output is unchanged. The repair costs decode speed
against the implementation it replaces - 43.74 tok/s against 45.75 at
1024/128 - and is recorded as a correctness fix rather than a speed win. An
earlier gate did not exercise the split at all, because the chain fed one token
at a time from an empty cache and stopped one key below the split's entry
threshold; `scripts/gemma4_teacher_forced_gate.py` now takes a prefill, records
the route each row used, and refuses to compare arms whose scored key range
differs.

| Engine | Prefill 1024 | Decode at 1024/128 |
| --- | ---: | ---: |
| llama.cpp HIP `8cfc315`, same GGUF | **3910** | **68.92** |
| hipEngine | 2722.1 | 57.03 |

On the same GPU and with llama.cpp rebuilt at build `8cfc315` for native HIP, a
separately measured comparator row at `-fa on -b 4096 -ub 1024` reports 4124
prefill tok/s; the table keeps the row recorded with this section's original
flags rather than mixing protocols. On the W7900, hipEngine measures 2665.1
prefill and 50.87 decode at the same shape, against 3761 prefill for llama.cpp.

Decode's MoE `gate_up` was the largest non-attention family in a step: the
legacy raw-layout owner read Q4_K at 0.163 ms per launch across the 29 MoE
layers. The `gate_up` tensors are now repacked to the tile16 layout at load
and decode dispatches the registered t16 owner through the profile rows — a
1:1 kernel swap over the same 1856 launches in a 64-step trace window
(303.19 -> 66.59 ms), taking the family from **4.74 to 1.04 ms/token
(-78%)** and decode device busy from 19.36 to 15.77 ms/token at 1024
tokens of context. The t16 owner's outputs are bitwise-identical to the raw
owner it replaces, the public `LLM.generate()` path records the `production`
profile with parity true, and the campaign gate against the frozen baseline
passes every bar (kl_max 0.01522, top-1 1.0 on all 1023 rows).
([artifact](results/2026-09-30-gemma4-family-1024-t16.json))

Prefill now serves both MoE expert projections from grouped expert owners
instead of one CTA per output column. The owners reuse each loaded weight row
across compact rows, reuse each loaded input row across four output columns --
without which the fused gate+up kernel re-walked the expert's whole input slice
once per column, at 23x the weight traffic -- and preload each weight row's
block metadata once per CTA instead of once per column. The expert gate+up and
down projections then run int8 x int8 MMQ from the same grouped owners, which is
the route llama.cpp takes. All of these owners reproduce the strict reduction
order rather than approximating it: the 1023-row teacher-forced comparison
reports zero KL and 100% top-1 agreement. Together they take prefill from 128.5
to 1397 tok/s at 1024 tokens on the W7900 at no numerical cost.

Prefill attention was the other half of the gap, at 216 ms of a 709 ms kernel
budget. The 25 sliding-window layers now run the vendored AOTriton BF16
flash-attention kernel at `head_dim` 256 instead of the engine's exact
staged-logit kernel, which is ALU-bound on its per-key reduction. The swap is a
changed-arithmetic path, so it carries the campaign gate: `kl_max` **0.000773**
against the 0.05 bar, `kl_mean` 5.5e-06, `kl_p99` 1.4e-04, and 0 of 1023 top-1
flips -- lower than the 0.001341 the same gate measures with the exact attention
kernel, since both are rounding-level perturbations of the same reference. The
flash kernel reads no mask, so it is admitted only where the mask the exact
kernel would read is exactly causal, which for a sliding layer holds while the
attention window does not bind; a prompt longer than 1024 tokens therefore keeps
the exact kernel on its later blocks and measures 1228.9 prefill tok/s at 2048
tokens. The five global layers use `head_dim` 512, which the vendored image set
does not cover, and still run the exact kernel. Together the expert routing and
the attention swap take the W7900 lane from 1394 to 1702.8 prefill tok/s, and
the XTX lane - which could not run the model before the gate/up split removal
freed peak memory - now measures 1886.8.

The dense Q8_0 projections - `q`, `k`, `v`, `o` and the dense MLP, 159 ms of a
554 ms per-prefill kernel budget - were running at a tile chosen by a shape
cascade that is wrong on this GPU. A sweep over thirteen shapes at rows 8
through 1024 finds `(16, 32)` fastest or tied at every point, against 1.2-1.9x
losses at rows 128 and above and 1.7-3.3x below that for the tiles the cascade
picked. Flattening it to `(16, 32)` is **bit-identical** - the tile decides which
CTA computes an output, not the order its k-blocks accumulate in, and the
1023-row teacher-forced gate returns exactly its previous verdict - and takes
prefill to 1791.1 tok/s on the W7900 and 1988.7 on the XTX. Serving the same
shapes from the int8 MMQ128 chain was measured and **rejected**: it is 44%
slower. Together the tile and the attention swap take prefill from 1394 to
1791.1 tok/s on the W7900 and from 1886.8 to 1988.7 on the XTX, against
llama.cpp's 3761 and 4124 on the same GPUs.

hipEngine's own shape matrix at 128 outputs: decode 52.28 tok/s at a 128-token
prompt, 48.12 at 512, 43.84 at 1024, 40.02 at 4096; prefill 140.2 / 135.7 /
128.5 / 108.6 tok/s at the same shapes. The 128/512/4096 rows are one sample
each; the 1024/128 row is three. The matrix's prefill figures were taken before
the expert-projection routing, the flash-attention path and the tile change; the
current 1024-token prefill is the 2722.1 tok/s in the table, and the current
1024/128 decode is 57.03. First-token latency at 1024 is 0.38 s and
public request wall time is 2.61 s including prefill. Decode is close to flat
across context length - 52.28 down to 40.02 - because the sliding layers, 25 of
the model's 30, read only the keys inside their 1024-token window instead of
walking the whole cached context and masking the difference away.
For product context on the same GPU and workload, Qwen3.6-35B-A3B
`UD-Q4_K_M` in hipEngine measures 3281 prefill and 114.57 decode tok/s — a
different model and tokenizer, used as a speed reference only.

[Dimension-partitioned decode attention and its correctness repair](results/2026-09-26-gemma4-dimension-partition-correctness-repair.json);
[MoE prefill expert-projection routing](results/2026-09-26-gemma4-moe-prefill-grouped-xtx.json),
[grouped gate+up projection](results/2026-09-26-gemma4-moe-prefill-grouped-dual-xtx.json),
[amortized input reuse in the expert owners](results/2026-09-26-gemma4-moe-prefill-amortized-input-xtx.json),
[amortized down projection with shared metadata](results/2026-09-26-gemma4-moe-down-amortized-xtx.json);
[read range and slice policy](results/2026-09-26-gemma4-26b-a4b-sliding-read-range-accepted.json);
[two-phase decode split and its production-profile gate](results/2026-09-26-gemma4-26b-a4b-two-phase-split-accepted.json);
[decode attention kernel](results/2026-09-25-gemma4-26b-a4b-key-class-decode-accepted.json);
[baseline matrix, comparators and boundary rows](results/2026-09-24-gemma4-26b-a4b-g0-baseline.json).

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

**Qwen3.8 Flash Next wide dense-prefill route needs a numerical re-review
(gfx1151, UD-Q4_K_XL).** The `wmma` layer-scoped Q8_0 dense-prefill route on
layers 16-47 is the production default for the wide dense-prefill shapes. Its
recorded numerical certificate was measured at commit `035185fb2`; re-running
the identical certified configuration at `f4b12e0c2` gives mean KL 4.388e-4
(recorded 3.796e-4), max KL 2.341e-2 (recorded 1.530e-2) and top-1 0.9942
(recorded 0.9935), and one of 1548 compared rows now exceeds the 0.02 review
boundary, so the gate reports `measurement_valid: false` where it recorded
`true`. Every protocol field, route field, fixture hash and manifest hash is
identical between the two runs, and both arms' published logits changed -
including the teacher arm's exact chain - so the divergence is in the shared
baseline arithmetic, not in this route. Chunk size is not the variable: the
chunk-1024 and chunk-4096 configurations are bit-identical at `f4b12e0c2`. All
hard thresholds still pass with 1.84x-2.28x headroom on the tightest three
(mean against 1e-3, p95 against 5e-3, top-1 against 0.99), so this is a review
requirement rather than a demonstrated quality failure. A binary search over the
range attributes the baseline movement to two commits, both of which promote
decode-path arithmetic rather than changing the route under test: `c66db0264`
(decode dp4a linear to the production default) and `6c3c46b04` (a wider pack8
GEMV block at narrow projections). The gate certifies a prefill route but
measures its drift over 128 teacher-forced decode steps, so its baseline is
decode-path arithmetic and a promoted decode-path change invalidates a
prefill-route certificate without touching that route. The route live at 16-47
is `dense_wide` (`PRODUCTION_Q8_DENSE_WIDE_PREFILL_LAYERS`); the `wmma` route
was retired at that scope because the wide route holds the identical envelope
and is 5.4% faster, and the two routes' published logits are byte-identical. A
full canonical re-gate of `dense_wide` at `cac0f1da5` gives teacher hash
`294e6c08` (recorded `550bb9b8`) and candidate `10fd0ae5` (recorded `65370710`),
mean KL 4.388e-4, max KL 2.341e-2, top-1 0.9942 and one row over the review
boundary, with all hard gates still passing. The teacher hash was predicted by
identity before the run, since both gates clear every selector for the teacher
arm. One gap is recorded: the engagement detector added in `056e6fbd8` has no
recorder for the `dense_wide` route, so its `candidate_route_engaged: false` is
a coverage artifact and this gate cannot prove the route ran. No speed claim is
made.
[Evidence](results/2026-09-30-qwen4exp-rt1-chunk4096-gate/README.md).

**Qwen3.8 Flash Next decode arithmetic (gfx1151, UD-Q4_K_XL).** The routed-expert
gate+up decode GEMV uses the dp4a Q8_1-activation linear
(`gguf_q4_k_selected_dual_q8_1_dp4a_silu_logical128_t64`) by default on 43 of 48
layers. On one `code-p4096` decode step at live count 4097 this cuts the owner from
14.155 to 6.908 ms/step (2.049x) and the decode kernel sum from 54.603 to 47.359
ms/step (-13.27%), with every other kernel unchanged (non-owner sum 40.448 vs
40.451 ms). The production numerical envelope holds: mean KL 2.44e-4 (gate 1e-3),
max KL 6.14e-3 (gate 5e-2), top-1 0.9911, deterministic across three repeats, and
task quality cross-validated on the `code` category. This is a decode kernel-time
result, not a serving throughput claim. [Evidence](results/2026-09-29-qwen4exp-de1-dp4a-e2e/e2e.json).

**Qwen3.8 Flash Next decode narrow-projection block size (gfx1151, UD-Q4_K_XL).**
The decode-shaped pack8 GEMV launches one workgroup per eight output features, so a
narrow projection cannot fill the device: on gfx1100, `out=320` runs 40 workgroups at
92.4 GB/s (42% of the 221 GB/s roof) and `out=640` runs 80 at 121.5 GB/s, while every
shape at 2560 or more output features already sits at 90.8-102.9%. Raising the block
from 128 to 256 threads at the starved shapes gains 19.4-32.4% of kernel time there and
is neutral within 2% above the knee, so the change is gated at `out_features <= 640`.
The production numerical envelope holds on the full 18-prompt / 450-row suite: mean KL
0.000287 (gate 0.001), p99 0.004831 (0.02), max 0.018131 (0.05), top-1 0.99778 (0.99),
all four categories above their 0.97 per-scope floor, three repeats bit-identical, and a
clean lifecycle. The `prefill_last` scope scores mean KL 6.9e-14, so the lever is
confined to decode and prefill is untouched. This is a per-kernel projection from a
microbenchmark, not an end-to-end serving claim. Set `HIPENGINE_GGUF_PACK8_GEMV_WIDE=0`
to restore the 128-wide default. [Evidence](results/2026-09-30-qwen4exp-de3-threads-lever/kl-envelope-gate-qwen4exp.json).

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

No valid final W7900 parity row exists: the four-arm A/B was measured on the XTX, and the W7900
paired MTP run is withdrawn as invalid (see the dated retraction in [`CHANGELOG.md`](CHANGELOG.md)).
The retained paired c1 natural25/B3 measurement on a clean worktree at `92c7e3dc4` gives
`UD-Q4_K_M` 32.922 AR / 50.231 MTP B3 (**1.5258x**) against plain 38.373 / 65.347, and `UD-Q4_K_S`
32.311 / 49.356 (**1.5275x**) against plain 41.135 / 70.092. The `Q4_K_M` arms are generated-ID
exact and `speed_claim_eligible`; `Q4_K_S` carries the recorded 2/20-row `general_ja_plan` exactness
divergence and is published as a measured rate, not an eligible claim.
[Paired artifact](results/2026-09-13-ud-gfx1100-paired-clean-provenance.json). The four-arm A/B,
automatic-admission check, per-lever kernel/verifier deltas, and the withdrawn W7900/XTX census
comparison are archived in [`HISTORY.md`](HISTORY.md#xtx-rx-7900-xtx-qwen38-ud-q4km-ud-q4ks-dynamic-routing-stack).

**Qwen3.8 Flash Next WMMA-MoE27 prefill restore — withdrawn (gfx1151, UD-Q4_K_XL).**
The route that replaces production's exact grouped iu8-risk+repair prefill with
WMMA-MoE27 on layers 27-47 passes the production numerical envelope in all four
categories (mean KL 1.18e-4 to 5.29e-4 against the 1e-3 gate, p95 5.19e-4 to
2.67e-3 against 5e-3, max 2.82e-3 to 1.14e-2 against 5e-2, top-1 0.99 to 1.00)
but fails the task gate on 3 of 18 prompts, deterministically. The three
divergences are `heldout_code_rate_limiter`, `heldout_general_ja_migration` and
`heldout_general_ja_speculative`, with zero repeat mismatches across three runs
per prompt, so the cause is arithmetic rather than flakiness. The row was
previously withdrawn on speed; that reason cannot be fixed by a faster tiling,
so the rerun condition is now a mechanism that resolves those three prompts.
[Evidence](results/2026-09-30-qwen4exp-moe-restore-rejection/README.md).

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
| `qwen4exp_dense_route_ab.py` | **Layer-scoped dense prefill route A/B**: exact coltile chain vs the certified f16 WMMA route vs the opt-in wide-row kernel at layers 16-47, selectors flipped post-binder in one process with one model load, interleaved arms with rotated order, and a per-arm launch census plus logits digest so the route swap is shown rather than assumed | | | ✓ | | | | `--repetitions 3 --output <json>` |
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

Production UD-Q4_K_XL uses guarded Q8 block-scale selected-down, four-head
QSA prefill and ordered QSA decode, with conservative arithmetic elsewhere.
Guarded Q8 matches strict on 594 natural-prompt rows; the QSA-prefill
combination matches on 780 canonical 512/1K/4K rows, and ordered decode
matches on 516 overlapping 4K rows. Repeatability, state and teardown pass.
Short free outputs match strict on 18 prompts; complete-EOS factual quality
has not been compared symmetrically. The Q8 repair screen is empirical,
not a universal bit-identity guarantee.
[Correction and evidence](results/2026-09-15-q8-blockscale-restoration/README.md).
MTP and multimodal performance are not requalified by these text-AR runs.

Same-host counterbalanced comparison, four categories, three repeats,
1024-token chunks, BF16 KV, 128 decode steps, one residency with shared
unchanged decode graphs. Both arms use the same guarded Q8/QSA arithmetic;
the production host path now caches unchanged backend registrations:

| Configuration | 512 PP / TG | 1K PP / TG | 4K PP / TG |
| --- | ---: | ---: | ---: |
| Without host registration cache | 177.70 / 17.52 | 186.19 / 16.75 | 178.41 / 10.16 |
| Production with host cache | 183.55 / 17.49 | 189.15 / 16.72 | 181.76 / 10.02 |

Weighted tok/s; 72 measured samples and 24 warmups. Prefill improves
**3.29% / 1.59% / 1.88%**. Every complete request improves, by 0.22-1.54%.
Decode is lower by 0.16% / 0.17% / 1.36%; this is not a decode-speed claim
or a measured benefit at arbitrary output lengths. All measured arms match
in output IDs, final logits and state, with zero teardown allocations.
[Host-cache evidence](results/2026-09-14-journey-backend-cache/README.md).
The earlier Q8/QSA restoration and failed-composition recovery comparisons
remain in their linked records; their rates are not this comparison's denominator.

A separate same-host comparison evaluates prefill chunk sizes with the same
production arithmetic, shared decode graphs and correctly sized prefill
workspaces. Chunk1024 remains the default;2048 is a qualified explicit option:

| Prefill Chunk | 512 PP / TG | 1K PP / TG | 4K PP / TG |
| --- | ---: | ---: | ---: |
| 1024 | 185.01 / 17.50 | 190.33 / 16.73 | 182.80 / 10.08 |
| 2048 | 184.37 / 17.50 | 190.00 / 16.73 | 187.04 / 10.20 |

At4K, PP improves2.32% and TG1.22%; complete requests improve1.52-2.14%.
At512/1K, PP falls0.34%/0.18%; short complete-request changes range from
-0.40% to+0.25%. The global default does not change because those rows
include costs. Canonical/boundary numerics, six active4K tasks and native
c2 isolation pass. This does not certify native-depth or full long-form
factual quality. The rates above have their own paired denominator and
must not be compared as an update to the separate host-cache table.
[Chunk comparison and qualification](results/2026-09-14-journey-chunk-admission/README.md).

A separate1024/4096 comparison uses the same counterbalanced, shared-graph
protocol and unchanged model/BF16 configuration:

| Prefill Chunk | 512 PP / TG | 1K PP / TG | 4K PP / TG |
| --- | ---: | ---: | ---: |
| 1024 | 185.25 / 17.50 | 190.57 / 16.74 | 183.12 / 10.09 |
| 4096 | 184.99 / 17.50 | 190.64 / 16.72 | 190.33 / 10.20 |

At4K, PP improves3.94%, TG1.11%, and every complete-request case improves
2.26-3.82%. Short complete-request changes range from-0.31% to+0.32%;
chunk4096 is a qualified explicit option, not the global default.
Numerical, boundary, active-task and native c2 gates pass; all72 timing
samples match IDs,final logits and sampled state. These are separate
paired denominators, not a direct2048/4096 comparison. No native-depth,
full-EOS factual-quality or decode-kernel speed claim is added.
[4096 comparison and qualification](results/2026-09-15-chunk4096-accounting/README.md).

Exact DPP reductions improved the previously active tiled-GDN prefill suffix:
matched warm PP295.765/312.115/286.962 ->296.136/312.942/287.713 tok/s
at512/1K/4K (+0.125%/+0.265%/+0.262%). Full72-sample output/state A/B is
exact; no decode speedup or broader numerical qualification is claimed.
The DPP kernel is retained, but the tiled suffix is inactive during recovery.
[DPP evidence](results/2026-09-14-journey-gdn-dpp/README.md).

The multi-column GDN prefill route is now the production selection for the
GDN prefill scope, superseding the DPP route. It is the same kernel template
with an 8-wide per-warp column tile instead of 1, so one k-load feeds eight
columns instead of one. On the GDN prefill scope at Hk16/Hv48/Dk=Dv=128,
matched warm medians are **4.556 -> 3.118 ms at 1K tokens**, **1.864 -> 1.112
ms at 512**, **10.583 -> 7.706 ms at 2K** and **22.799 -> 16.997 ms at 4K**,
i.e. **1.34x-1.68x faster across the range**. NMSE against the DPP route is
**~1.8e-14** on the core output and **~7.2e-15** on the recurrent state -
fp32 round-off - so the route is not bit-exact and is admitted on the
production numerical envelope (mean KL 8.92e-05, max 4.45e-03, top-1 777/780),
not on exact generated IDs. Its one task finding is shared with the strict
arm and is recorded rather than dismissed. The DPP route stays registered as
the strict fallback. Measured on `gfx1151`, not the default W7900; GDN
prefill scope only, not a whole-request rate.
[Multi-column evidence](results/2026-09-30-qwen4exp-gdn-multicolumn-ab/README.md).

The route is the production selection **where its column-warp gate is open**,
which is the `gguf_q4_k_m` profile; the `gguf_ud_q4_k_xl` arithmetic-recovery
profile closes that gate and replaces tiled variants by name with its exact
owner, so its manifest declares `qwen4exp_sigmoid_wave_norm_prefill` and stays
byte-identical. A gate on the q4_k_m profile confirms the route runs there
(**1350 dispatches**, against 0 before the binding repair) and that the profile
**fails** the calibrated envelope at max KL **0.05958** with `prefill_last` mean
KL **1.333e-3**. That failure is not the route's: an attribution control that
pins the GDN prefill owner to the superseded `qwen4exp_gdn_tiled16_prefill`
fails the same five scopes with a **worse** tail (max KL **0.06645**,
`prefill_last` mean **1.475e-3**). It is the composition the September 14
recovery replaced for `gguf_ud_q4_k_xl` only, measured here at the same class
of result (that artifact recorded max KL `0.054642` with prefill-last mean/p95
also failing).
[Gate](results/2026-09-30-qwen4exp-gdn-multicolumn-q4km/gate.json) ·
[Control](results/2026-09-30-qwen4exp-gdn-tiled16-control/gate.json).

Mapping-only random advice for sparse PLE reads is now the gfx1151 production
default. In the matched file-scoped cold-cache protocol, all-category
512/1K/4K prefill improves **84.161/134.330/225.864 ->
260.484/295.260/290.598 tok/s**; decode improves
**13.388/13.053/13.544 -> 19.560/18.659/18.643 tok/s**.
Warm throughput is approximately neutral. Both72-sample matrices preserve
generated IDs, final logits/state and zero teardown; this is not a new
strict-teacher numerical certificate.
Those paired rates use the earlier arithmetic composition, not the recovered
profile's current end-to-end throughput.
[Cold/warm evidence and commands](results/2026-09-14-journey-ple/README.md).

A layer-scoped Q8_0 prefill route A/B on one host and session measures what the
wide-row kernel is worth where the certified f16 WMMA route already runs (layers
16-47). Same category, three prompt lengths, three interleaved repetitions,
selectors flipped after the production binder in one process:

| Prefill route at layers 16-47 | 512 | 1K | 4K |
| --- | ---: | ---: | ---: |
| Exact coltile chain | 2.822 s | 5.487 s | 22.648 s |
| WMMA | 2.225 s | 4.389 s | 18.208 s |
| Wide-row, now the default | **2.201 s** | **4.151 s** | **17.217 s** |

Median prefill wall; the wide-row route is **5.4%** below the WMMA route at 1K
and 4K and indistinguishable from it at 512. Against the exact chain it is
22.0%/24.3%/24.0% faster, of which the WMMA route already holds
21.1%/20.0%/19.6%. All three arms launch the same 7191 kernels over the same
roles, and the wide arm's logits digest and sampled token are identical to the
WMMA arm on all three cases in both runs, while both differ from the exact
chain. The wide route has since passed its own 12-case production envelope at
16-47, with strict and candidate logits digests byte-identical to the WMMA
route's ([gate](results/2026-09-17-q8-dense-wide-16-47-gate/README.md)), and it
**is the production default at that scope** as of 2026-09-17: the promoted
default launches `dense_wide256` on the 264 layer-16-47 roles with zero
`wmma_prefill` launches and keeps the exact coltile chain on layers 0-15. This
single-category prefill protocol is not a topline rate row.
[Route A/B evidence](results/2026-09-17-q8-dense-route-ab/README.md).

The promoted default now has its own full-matrix rate and a current attribution.
Twelve cases, four categories, three lengths, one warmup and three repetitions,
36 samples, production profile with no profile named:

| Shipped default, HEAD | 512 PP / TG | 1K PP / TG | 4K PP / TG |
| --- | ---: | ---: | ---: |
| Equal-weight mean over categories | 239.5 / 17.70 | 253.4 / 17.18 | 241.9 / 16.24 |

The 4K decode figure is the mean of the per-case medians (16.20-16.29); the
artifact's token-weighted 4K rollup is lower because five samples at the end of
the run degrade, which the source documents. The 2026-09-13 journey row for this
model (294.1/19.17 at 4K) came from a fifteen-recovery-flag composition that
failed the numerical envelope and is not this row's denominator: it is 22% faster
in prefill than the path that ships, on arithmetic that does not pass.
[Baseline](results/2026-09-17-qwen4exp-shipped-default-baseline/README.md) ·
[attribution](results/2026-09-17-qwen4exp-shipped-default-attribution/README.md).

Where that 17.0 s prefill goes, from a role-marked capture on the shipped
default (16733.4 ms attributed, 0 ms unattributed, no unmapped symbol over the
1 ms floor):

| Family | ms | Share | vs halo-box PR #63 |
| --- | ---: | ---: | ---: |
| `dense_projection` | 5234.8 | 31.3% | 3.81x |
| `expert_gate_up` | 3402.4 | 20.3% | 4.47x |
| `hyper_connection` | 2525.4 | 15.1% | 7.52x |
| `expert_down` | 2382.3 | 14.2% | 5.77x |
| `qsa_attention` | 1402.5 | 8.4% | 9.01x |
| `gdn` | 821.6 | 4.9% | 3.11x |
| `moe_reduce` | 758.4 | 4.5% | 7.30x |
| `elementwise_norm` | 178.5 | 1.1% | 0.54x |

The wide-row promotion moved one family: `dense_projection` 10562.2 -> 5234.8
ms, with every other family within 0.5% of the pre-promotion capture. The
comparator's dense path is BF16 against this engine's F16, so the ratios locate
work rather than predict recoverable time.

The `iu8` exact-repair passes inside the MoE families are bound by memory-level
parallelism, not by bytes or grid size. Unrolling their runtime-bound inner loops
takes them **1420.9 -> 1239.8 ms (-12.7%)** in the same protocol, with the main
MoE kernels unchanged to 0.1%; a bit-identical row-bucketed redesign that moves
3.3x fewer bytes is 22% slower. [Packet](results/2026-09-17-qwen4exp-iu8-repair-cost-structure/README.md)
· [engine capture](results/2026-09-17-iu8-repair-unroll/role-analysis.json).

A prefill-only hardware-counter capture shows the prefill is **not** uniformly
stalled across families: the largest kernel (`gr_up_sigmoid_mean`, 18.26% of
device time) sits at **30.82% LDS wait** while every other top-ten prefill kernel
is at **≤0.35%**, so the family ratios locate work rather than describe one shared
per-kernel deficit. The fifth-largest, the GDN prefill kernel, is register-bound at
the 256-VGPR hardware maximum (5.93% of prefill, 0.92 instructions per CU-cycle) —
a condition already recorded in the kernel source and already priced: its LDS route
is slower at 1024 rows and above, and the 85-VGPR route that escapes the bound is
gated off in the UD lane. [Packet](results/2026-10-04-qwen4exp-prefill-stall-census/README.md).

The routed-MoE main kernels are bound by expert-weight traffic at a constant
104-126 GB/s, and that traffic is set by how many 16-row WMMA tiles the experts'
rows are cut into: `sum over experts of ceil(rows/16) x 1.843 MB`. Eight times the
rows cost 5.2x the time (1280 rows 8.65 ms vs 40960 rows 45.39 ms), and an
exactly-even 20-rows-per-expert routing is the *slowest* 10240-row case (1024
tiles, 17.26 ms) because each expert's 20 rows cut into two tiles. The engine's
own `grid_y = 1120` launches agree with the model to 2-3% (2.064 GB in 14.5 ms).
Raising the M-tile to 32 rows would cut each launch to one tile per expert, a
projected 1.7-2.3 s of a 16.67 s prefill. [Packet](results/2026-09-22-moe-main-kernel-cost/README.md)
· [engine capture](results/2026-09-17-iu8-repair-unroll/role-analysis.json).

That 32-row tile is now measured and **rejected**, and the reason sharpens the
traffic model: the tile count falls exactly as projected (857 -> 512 at 20 rows
per expert) but *real* DRAM read traffic falls only **6.3%** (0.937x measured
against a modeled 0.548x), because a taller tile makes the weight read *shared*
between its two row groups rather than distinct, and the sharing misses cache.
The kernels run **1.3-2.2x slower** (gate/up 14.84 -> 20.25 ms, down 9.04 ->
12.58 ms at 10240 rows). The tile count therefore bounds traffic only for
distinct reads. Doubling the per-lane fp32 risk state also blew the register
file (VGPR 192 -> 256, 492 B/thread spilled, 8x slower); the shipped 8-warp
layout keeps 16 rows per lane, is spill-free and bit-identical, and is still a
net loss. The 16-row tile stays production. [Packet](results/2026-09-22-moe-j32-tile/README.md).

A `code-p4096` prefill runs four 1024-token chunks, and both dominant families
pay for that chunking without having to. The dense wide Q8_0 route measures
**1382.2 -> 1057.2 ms (-23.5%)** at 4096-row chunks, and the routed-MoE main
kernels drop from 4480 to 2801 weight tiles per layer, which is **57.7 -> 36.1 ms
gate/up and 34.5 -> 21.6 ms down**. Together that is **1.7-1.9 s of a 16.673 s
prefill (10-11%) with no kernel change**. [Dense packet](results/2026-09-22-dense-chunk-rows/README.md)
· [MoE packet](results/2026-09-22-moe-main-kernel-cost/README.md).

September 14 numerical refresh of the previous production profile on
UD-Q4_K_XL/BF16 KV at chunk1024 fails the full18/594-row strict-teacher
gate (max KL0.054642; prefill-last mean/p95 also fail), while deterministic
repeats and teardown pass. The throughput rows below are not a fresh quality
certificate. Numerical limits are unchanged; the recovery above replaces
that arithmetic composition.
[Evidence](results/2026-09-14-journey-production-baseline/README.md).

September 5 Framework `gfx1151` screen, UD-Q4_K_XL/BF16 KV,
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

September 13 UTC post-overlap refresh, same UD-Q4_K_XL/BF16 KV,
four categories, one warmup and three repetitions:

| Engine | p512 PP / TG | p1024 PP / TG | p4096 PP / TG |
| --- | ---: | ---: | ---: |
| hipEngine production | 297.114 / 20.396 | 316.907 / 19.711 | 294.071 / 19.166 |
| halo-box HEAD Vulkan | 423.376 / 27.477 | 475.121 / 27.104 | 506.261 / 25.721 |
| halo-box HEAD HIP, launch blocking | 427.063 / 15.441 | 571.258 / 15.347 | Unverified |

Weighted tok/s; all measured IDs repeat. hipEngine uses 1024-token chunks;
Vulkan uses ubatch4096, HIP ubatch1024, and both external lanes use the
on-disk n-gram reader. HIP's full-matrix attempt failed at 4K before overlap
clearance; its fresh subset does not certify that depth. These are sequential
screens, not paired kernel-speedup claims or a fresh full numerical/task gate.

September 13, same target plus the existing Q8_0 draft, BF16 KV, all ten
category prompts, B2, 16 output tokens, one warmup and three repetitions:

| Engine | AR request tok/s | MTP request tok/s | MTP / AR | Result |
| --- | ---: | ---: | ---: | --- |
| hipEngine production | 14.949 | 11.197 | 0.749x | 30/30 AR-exact and repeatable; every category slower |
| halo-box HEAD Vulkan | 16.031 | 16.894 | 1.054x | 30/30 AR-exact and repeatable; Japanese 0.976x AR |
| halo-box HEAD HIP | — | — | — | AR completes; first MTP warmup faults in a gather kernel |

Request throughput includes prefill. These are sequential-arm diagnostic
measurements, not decode-only rates or a promotion verdict. The HIP and
Vulkan rows use the same September 12 external source snapshot and were restarted
after reported process overlap; the interrupted HIP pair is excluded.
Earlier September 13 timing screens are provisional because the overlap
interval is unknown; hipEngine was also remeasured after the overlap.
The Vulkan Japanese result does not pass the every-category non-regression requirement.
[Comparison evidence](results/2026-09-13-framework-qwen4exp-strix-journey-baselines.json)
and [campaign plan](../docs/QWEN4EXP-STRIX-JOURNEY-CAMPAIGN.md).

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

A 2026-09-13 refresh rebuilt upstream llama.cpp at HIP `002a12ad2` and Vulkan
`37b3a9e0c` and compared both, halo-box `strix-llama.cpp` `654803517` HIP and
Vulkan, and hipEngine on that same file at 512/128, 1K/128 and 4K/128.
`strix-llama.cpp` HIP leads prompt processing at every shape, by
**+9.2%/+12.2%/+16.1%** over hipEngine's **405.0/394.4/372.1** tok/s; both
Vulkan builds lead text generation by **+4.4% to +6.9%**; upstream llama.cpp
HIP is 3.7% behind hipEngine at 512/128, level at 1K/128, and 3.0% ahead at
4K/128. The hipEngine row reproduces the then-published **404.5/12.2** to
+0.13%, and is the pre-retile measurement; the current default is faster on
the same host (see the current profile table above).
An ablation of the HIP prefill lead shows that about half of it is one compile-time
table: replacing only `ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh` in upstream
llama.cpp `002a12ad2` with the fork's version lifts 4K/128 prefill from 382.9 to
409.5 tok/s (**+6.9%**, 54% of the fork's gain) and reproduces the entire
Q4_K/Q5_K MMQ kernel-time delta with every other kernel group unchanged. The
rest is the fork's tiled Gated DeltaNet kernel with transposed concat, the Q8_1
MMQ activation quantizer, flash attention and norm/convert fusions.
[`same-GGUF five-engine comparison`](results/20260913-qwen38-27b-comparison/README.md).

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

**Host-state re-measurement (2026-10-03).** The rows above were measured while
this host's `accelerator-performance` tuned profile was inactive, a state that
drifted after 2026-09-05 and that X-8 also recorded. Re-measured on the same
physical host (`55ea6c50...`) under the restored profile, prefill is higher at
every depth (**+1.5% / +2.0% / +4.8% / +3.5%** at 512 / 4K / 32K / 64K) while
decode is higher only at short context (**+4.5% / +2.7%**) and is **1.9-3.5%
lower** at 32K / 64K. The two columns therefore have different signs and the
published values above are left as measured rather than replaced; the prefill
column is the one the host state explains. Evidence:
[`host-state re-measurement`](results/2026-10-03-gfx1151-35b-a3b-gufo-reconciliation/README.md).

**H2048-MoE prefill chunk promotion (2026-10-03).** The 512-row linear/MoE
prefill chunk this geometry carried is replaced by **1024**, which is
**bit-exact** against it (KL `0.00000000`, max absolute logit difference
`0.00000000`, identical top-1 at 4096 tokens; same-config repeats are also
bit-identical, so the run-noise floor on this path is exactly zero). Row
chunking partitions tokens rather than reductions, so the change cannot alter
any single token's arithmetic. Measured prefill on one host state at 512 / 1024
/ 4096 chunks: **1454.8 / 1486.4 / 1488.2** at 1K, **1459.3 / 1487.7 / 1503.9**
at 4K, **1358.5 / 1358.5 / 1358.3** at 16K, **969.9 / 969.5 / 969.2** at 64K,
so the lever is confined to short prompts and 4096 is faster still at 1K/4K but
is not bit-exact and costs more at 64K. Production rows after promotion are
**1401.89 / 1479.75 / 1197.59 / 969.08 tok/s** prefill at 512 / 4K / 32K / 64K
(+0.84% / +1.39% / -0.17% / -0.02% against the same host state at 512 chunks).
Evidence:
[`chunk promotion`](results/2026-10-03-gfx1151-h2048moe-prefill-chunk-1024/README.md).

**Shared-expert gate+up wave-strided loop (2026-10-03).** The Q8T16 shared
gate+up pair was the worst-bandwidth stage in the decode step that was not
already explained (87.1 MiB/token at 103 GB/s against 199 GB/s for the main
dense GEMVs). At `rows == 1` it now runs `q8_0_t16_dual_split_gemv_kernel`
(`WAVE_BLOCK_INDEX = true`) instead of `q8_0_t16_dual_gemv_kernel`. The change is
**bit-identical** — with 128 threads the concat enumeration `k = tid + 128j` maps
to `block_idx = tid/32 + 4j`, `lane = tid & 31`, exactly the split enumeration,
so the per-thread FP32 accumulation order and the reduction tree are identical
and a warp reads one Q8_0 block per iteration in both — and register pressure is
identical (VGPR 56, SGPR 128, LDS 512). Isolated: **117.9 -> 142.5 GB/s
(+20.8%)**. Real decode step, same-session A/B: `decode_ffn_moe_shared_gate_up`
**-13.84%** (30182.3 -> 26006.3 us/32tok), decode **53.1877 -> 53.5820 tok/s
(+0.741%)**. Evidence:
[`shared-expert wave loop`](results/2026-10-03-gfx1151-shexp-wave-loop/README.md).

**Selected Q4_K gate+up tile8 owner (2026-10-03).** `decode_ffn_moe_selected` was
the largest remaining bandwidth-efficiency gap in the decode step (597.8
MiB/token at 170.4 GB/s, 72% of the 237 GB/s ceiling, 3.678 ms of 17.81 ms) and
its Q4_K gate+up leaf was register/occupancy-limited rather than
bandwidth-limited: rocprofv3 at the production 2048x512x8 shape records **VGPR
200 -> 88, LDS 1024 -> 512, grid.x 32 -> 64 blocks**, all at 128 threads. A new
owner covers **8 columns per workgroup instead of 16** and is **byte-identical**
— both launch `dim3 block(128)`, so thread K ownership, the reduction tree, the
BF16 round-trip and the fused SiLU are unchanged. Real decode step, same-session
A/B: `decode_ffn_moe_selected` **-6.83%** (125.727 -> 117.140 ms/32tok), decode
**53.4247 -> 54.3652 tok/s (+1.760%)**, prefill flat. Evidence:
[`selected Q4_K tile8 owner`](results/2026-10-03-gfx1151-selected-q4-tile8-owner/README.md).

**Selected-expert prefill multi-wave pairing (2026-10-03).** The two
selected-expert prefill kernels of this model launched one wave32 per output
tile with **LDS = 0** and dequantized their weight fragment once per 16-row
tile. Both now use a `NUM_WAVES`-wave block that stages a 32-row activation tile
in 1 KiB of LDS and pairs two 16-row tiles per wave, so one dequantized weight
fragment feeds two WMMA accumulators. Both bodies are **bit-identical** to the
kernels they pair on all four T16 quants x BF16/FP16 and four shapes including
expert-straddling row splits. Isolated at the real shape: gate+up **3.809 ->
2.910 ms (1.309x)**, down Q4_K **1.983 -> 1.520 ms (1.305x)**. Same-protocol A/B
at 4096 tokens, `--public-ar-profile`, chunk 1024: prefill **1450.8 -> 1603.7
tok/s (+10.54%)** in one pair and **1454.0 -> 1639.3 tok/s (+12.75%)** in a
second pair on the committed tree; the conservative pair is carried forward. By
the design basis's FLOP accounting that puts the combined expert legs at
**11.49 TFLOP/s from 8.51**, clearing the 11.3 TFLOP/s bar. **This is now the
gfx1151 automatic route** (`GGUF_Q4_T16_SELECTED_PREFILL_AUTO_MODE = "pair32"`;
gfx1100 is unchanged), after clearing the gate that blocked `shared_x`: three
independent conservative-set 128K warmup+3 processes and three independent
production-shaped ones each passed **12/12** with finite logits and IDs
`[9707,9707,9707]`, the latter at **620.61/619.99/619.57 tok/s** against the
575.180-580.626 baseline-only figure on the same workload. **Default-path A/B**
(explicit `=baseline` rollback vs the promoted automatic route, 4096 tokens,
chunk 1024, 3 measured runs each): **1450.78 -> 1633.33 tok/s (+12.58%)**. Evidence:
[`selected-expert pair32`](results/2026-10-03-gfx1151-selected-prefill-pair32-nwave/README.md).

### Radeon 8060S: Gemma 4 26B-A4B `UD-Q4_K_XL`

#### Topline 512/128 (2026-10-15)

Framework Desktop / Radeon 8060S (`gfx1151`), production profile, BF16 KV,
512 prompt tokens and 128 greedy outputs ignoring EOS, one warmup and three
measured requests at `09b14efeb`.

| Workload | Prefill | Decode | Public wall |
| --- | ---: | ---: | ---: |
| 512/128 | **1865.6 tok/s** | **43.30 tok/s** | 3.21 s |

Repeated and public output IDs are equal and logits are finite.
[Command, samples and provenance](results/2026-10-15-gemma4-gfx1151-512p128o-topline.json).

MTP with the `Q8_0` assistant head at a draft budget of 4, same host and
commit, production profile, BF16 KV: 18 prompts from the two suites under
`benchmarks/prompts/` (code, English, Japanese, mixed), two samples each, 128
greedy tokens, whole-call rate including prefill, through the public API.

| Plain tok/s | MTP tok/s | Ratio | Draft acceptance | Rows token-identical to plain |
| ---: | ---: | ---: | ---: | ---: |
| 45.5 | **69.4** | **1.52x** | 72.3% | 28 of 36 |

All 36 rows complete their 128 tokens within the production verify envelope.
[MTP artifact](results/2026-10-15-gemma4-gfx1151-mtp-budget4.json).

#### Paired depth ladder on the promoted build (2026-10-08)

On zbook/Radeon 8060S, both engines ran identical prompt IDs, BF16 KV, greedy
128-output requests ignoring EOS, context rounded to 256 tokens above
prompt+128, one full-shape warmup per engine per depth, one engine at a time.
hipEngine runs the promoted default: `gemma4_wmma_flash`/
`gemma4_wmma_flash_full` prefill with the strict-parity staged singleton for
decode, Q8_0-K/Q4_K-K T16 tile prefill (four-wave) and the Q5_1 T16 expert
`down` tiles; llama.cpp HIP is pinned at `17252c769` (the 2026-10-07
comparator rows are reused unchanged; the assembler re-validated prompt
hashes across engines).

| Prompt tokens | hipEngine prefill | llama.cpp prefill | hip/llama | hipEngine decode | llama.cpp decode | hip/llama |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | **1415.8** | 890.9 | **1.59x** | **37.75** | **35.98** | **1.05x** |
| 1024 | **1484.6** | 1158.7 | **1.28x** | **38.81** | 33.45 | **1.16x** |
| 2048 | **1356.2** | 1136.5 | **1.19x** | **37.74** | 30.01 | **1.26x** |
| 4096 | **1313.3** | 1116.8 | **1.18x** | **36.43** | 29.13 | **1.25x** |
| 8192 | 1244.7 | **1277.3** | 0.97x | **34.74** | 27.47 | **1.26x** |
| 16384 | **1135.0** | 1105.3 | **1.03x** | **31.47** | 24.95 | **1.26x** |
| 32768 | **975.6** | 864.5 | **1.13x** | **26.26** | 21.22 | **1.24x** |
| 65536 | **767.4** | 619.3 | **1.24x** | **19.49** | 16.53 | **1.18x** |
| 131072 | **505.2** | 388.7 | **1.30x** | **12.42** | 11.45 | **1.08x** |

hipEngine leads prefill at eight of nine depths (up to 1.59x) and decode at
nine of nine (up to 1.26x); at 131072 it leads both phases (505.2/12.42 vs
388.7/11.45 tok/s). The mid-depth prefill band 8192-32768 that trailed at
0.88-0.97x on the 2026-10-07 build now stands at 0.97-1.13x. 512-token decode
was the one trailing cell on the 2026-10-08 build (0.96x): the step was
launch-bound, not kernel-bound (~505 launches/step, ~11 of 29.7 ms), and
replaying the step from a whole-step HIP graph takes it to 37.75 tok/s =
1.05x ([512 decode-graph row](results/2026-10-09-gemma4-gfx1151-512-decode-graph.json)).
The same graph now captures the 1024 and 2048 rows: the flash-decoding kernel
re-derives its key partitioning per block from a staged device live-extent
slot, so replay is bit-identical to the launched path while removing its
launch gaps — 1024-token decode moves from 33.67 to 38.81 tok/s (1.16x) and
2048 from 36.36 to 37.74 (1.26x), with greedy IDs identical to the launched
path and deeper device-bound steps keeping the launched fallback
([mid-depth decode-graph rows](results/2026-10-10-gemma4-gfx1151-decode-graph-middepth.json)). llama.cpp's own repeatability gate
rejected its 8192 and 32768 rows in its run (its greedy continuation is not
repeatable there); those comparator numbers are retained as best-available
evidence with the rejection recorded, and the 16384 row passes with five
identical samples. The remaining honest gap is the
8192 prefill row (0.97x against a rejected comparator row).
Correctness: prompt hashes match across engines at every depth; hipEngine
repeats, public-path parity and finite logits pass on every row.

[Commands, samples, identities and rejections](results/2026-10-08-gemma4-gfx1151-depth-ladder-paired.json).
[2026-10-07 ladder, prior build](results/2026-10-07-gemma4-gfx1151-depth-ladder-paired.json).

#### Full-depth 256Ki endpoint on the final tree (2026-10-10)

The endpoint re-measured on the campaign's final tree (whole-step decode
graph in its measured launch-bound band, device-resolved flash geometry,
deep capture routes validated): hipEngine **336.0 prefill / 10.28 decode
tok/s** against the pinned llama.cpp rows (227.9 / 7.21, unchanged from the
2026-10-06 endpoint, prompt-hash-identical) — **1.48x / 1.43x**, both
phases lead at the full context. The hip side ran the full prefill once
and decoded both fresh and from the reloaded cross-process KV state on the
same tree; the restored decode's 128 greedy IDs equal the fresh decode
exactly.

| Prompt tokens | hipEngine prefill | llama.cpp prefill | hip/llama | hipEngine decode | llama.cpp decode | hip/llama |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 262,015 | **336.0 tok/s** | 227.9 tok/s | **1.48x** | **10.28 tok/s** | 7.21 tok/s | **1.43x** |

[Artifact](results/2026-10-10-gemma4-gfx1151-256ki-final.json).

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

The autoregressive stage of the pinned `m-a-p/YuE2-3B` checkpoint over the frozen 18-case replay
matrix: 128/512/2048-token prefixes, CFG on and off, unequal-prefix branches, scored against the
pinned upstream oracle on the same host. Pooled over 112 recorded full-vocabulary rows and 576
decode-step rows. `Prefill total` is the wall time to prefill all 18 prefixes, so it includes every
case's 4-10 rows.

| Prefill route | Mean KL | Max KL | Top-1 | Top-8 recall | Prefill total | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `strict` (row-by-row) | 1.43e-3 | 3.27e-2 | **95.83%** | 95.88% | 522.5 s | [`strict replay`](results/yue2_ar_replay_strict_20260916.json) |
| `hipblaslt` (batched, default) | **1.16e-3** | **7.10e-3** | 95.49% | **95.88%** | **77.6 s** | [`batched replay`](results/yue2_m7_gemm_algorithm_20260917.json) |

Re-running the batched route after the paired-branch lm head landed
(`dense_gemv_bf16_f32_out_rowtile2` plus the runtime's paired `logits`) reproduces these numbers:
pooled mean KL 1.1579e-3 over the same 576 rows and 18 cases, top-8 recall 0.9588, status pass, and
a `--repeat 1` run whose aggregate is byte-identical
([`paired head`](results/yue2_ar_replay_paired_head_20260917.json),
[`repeat`](results/yue2_ar_replay_paired_head_repeat_20260917.json)). That change is bit-identical
per row to the single-row head, so the replay matrix is expected to be unchanged and is; the
session-level validation is in the greedy session gate below.

Both routes replay the first case bit-for-bit after all 18 cases have run on the same runtime, and
both stay inside the broad floor (mean KL ≤ 0.05, top-1 ≥ 90%). The batched route selects its
hipBLASLt algorithm by measured index rather than by hipBLASLt's own first entry
(`scripts/hipblaslt_algo_scan.py`); that choice took its pooled mean KL from 2.11e-3 to 1.16e-3, its
worst case's max KL from 9.10e-2 (`abc-nocfg-L128-s5678`, 5 rows) to 7.10e-3, and its prefill total
from 91.3 s to 77.6 s, at a top-1 of 95.49% against the strict route's 95.83%. The strict arm was
not re-run in the session that measured the 77.6 s, so only the batched route's own 91.3 s → 77.6 s
is a same-protocol pair; the earlier 522.5 s → 91.3 s pair came from one session and is why the
strict route's absolute wall time should be read as a band rather than a fixed figure.

### Radeon 8060S: YuE2 3B greedy session gate

The torch-free staged session (`Yue2ArSession`) generating freely at temperature zero from the
pinned `m-a-p/YuE2-3B` checkpoint, against the pinned upstream oracle's own greedy trajectories on
the same host. `Prefix` is the assembled positive-branch prefix checked against the reference's own
ABC IDs, so a trajectory divergence cannot be read as a prompt-assembly bug. `Forced` is the
per-step greedy agreement when the runtime is driven along the reference's tokens, so both sides
score the same context at every step; `Free` is the unassisted trajectory's first divergence from
the reference, and the semantic phase is budget-truncated in every case.

| Case | Prefix | ABC forced | Semantic forced | Free first divergence | Truncation (ABC / semantic) | Evidence |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `english-melody-s1234` | identical | 99.7% | 97.1% | 27 | natural / budget | [`session gate`](results/yue2_session_gate_paired_head_20260917.json) |
| `english-full-s1234` | identical | 99.1% | 98.2% | 2 | natural / budget | [`session gate`](results/yue2_session_gate_paired_head_20260917.json) |
| `mandarin-off-s1234` | identical | no ABC stage | 96.7% | 23 | no ABC stage / budget | [`session gate`](results/yue2_session_gate_paired_head_20260917.json) |

Teacher-forced agreement is at least 96.7% on every phase of every case; the symbolic ABC phases
agree on 99.7% and 99.1% of steps. Every recorded mismatch sits on a near-tie: the first semantic
mismatch is 1.2e-01 / 1.2e-01 / 6.2e-02 below my own top-1 score for melody / full / off, inside the
BF16 logit noise the replay matrix above already quantifies (mean KL 1.2e-3, top-1 95.5%). The
unassisted trajectories track the reference until their first flip and are chaotic afterwards, so
the forced columns, not the free agreement rates, carry the fidelity claim. Free comparison is
context-aligned only where the ABC stage itself matched (`english-melody-s1234`) or is absent
(`mandarin-off-s1234`); `english-full-s1234` diverged during ABC planning, so its free semantic row
compares continuations of different prefixes and is diagnostic only. Unassisted wall clock: `melody`
34.8 s, `full` 36.7 s, `off` 29.0 s.

These numbers are the post-M7 state. The `english-melody-s1234` ABC phase was 100.0% and
`mandarin-off-s1234` semantic was 96.1% before the hipBLASLt algorithm-selection and attention
changes landed; every case then moved to the values above, which is recorded as the
`session_gate_after_change` block of
[`the M7 artifact`](results/yue2_m7_gemm_algorithm_20260917.json). Re-running the gate after the
paired-branch lm head landed reproduced those post-M7 values exactly (99.7% / 99.1% ABC, 98.2% /
97.1% / 96.7% semantic), so that change is invisible at the session level.

The paired-branch head itself is validated on the session loop rather than on the gate alone:
[`pairing validation`](results/yue2_ar_pairing_validation_20260917.json) drives `generate_tokens` at
temperature 1.0 on two cases with a serial-head subclass as the control and compares the emitted
tokens, the RNG state, the CFG score rows, the sampler's masked score row and the softmax
distribution. All agree exactly (logit max abs 0.0, distribution KL 0.0, identical top-1, identical
RNG state), and a lifecycle pass that runs case A, then case B, then case A again in one process
with a `runtime.reset()` between reproduces each case's tokens exactly.

### Radeon 8060S: YuE2 3B acoustic flow-matching solver

The torch-free NAR runtime's midpoint ODE solve against the pinned upstream reference on the same
host, on a recorded 34-row / 545-token-AR chunk with a 4-step schedule. `Forced` evaluates the
native velocity at each *recorded* state, which isolates a velocity defect from a trajectory defect;
the end-to-end row runs the native solver from the recorded noise and compares its own latents.
Every row is relative L2 against the reference, whose final latent norm is 72.743.

| Quantity | rel_l2 | cosine | max abs | Evidence |
| --- | ---: | ---: | ---: | --- |
| Forced velocity, step 0 (`t=1.0`) | 0.01327 | 0.999912 | 0.0859 | [`NAR gate`](results/yue2_nar_gate_20260916.json) |
| Forced velocity, step 1 (`t=0.75`) | 0.01495 | 0.999888 | 0.0625 | [`NAR gate`](results/yue2_nar_gate_20260916.json) |
| Forced velocity, step 2 (`t=0.5`) | 0.01892 | 0.999821 | 0.1218 | [`NAR gate`](results/yue2_nar_gate_20260916.json) |
| Forced velocity, step 3 (`t=0.25`) | 0.01221 | 0.999926 | 0.0781 | [`NAR gate`](results/yue2_nar_gate_20260916.json) |
| 4-step solve, final latents | **0.01225** | **0.999926** | 0.0781 | [`NAR gate`](results/yue2_nar_gate_20260916.json) |

The solver is the reference's own scheme: `mid = state - v(state, t) * h/2`, then
`state = state - v(mid, t - h/2) * h`, with each update rounded through BF16 exactly as the
reference does. The 1.2% residual is BF16 rounding noise, not structural drift - 16.2% of the
produced latent entries are BF16-identical to the reference and the produced norm is 72.855 against
72.743. Condition 1.00 s and forced-velocity plus solve 0.80 s on this chunk.

Two further recorded cases cover the rest of the solver's contract. A song that crosses a real chunk
boundary (96 frames at a reduced chunking context, so the chunks are 26 / 26 / 26 / 18 frames) and a
restricted-visibility case (`nar_cond_end = 128`, where the NAR sees only the first 128 AR keys):

| Case | Solved latents rel_l2 | cosine | Worst forced velocity | Evidence |
| --- | ---: | ---: | ---: | --- |
| `nar_cond_end = 128` | 0.00885 | 0.999962 | 0.01443 | [`cond_end gate`](results/yue2_nar_gate_condend_20260916.json) |
| Multi-chunk, chunk 0 (26 frames) | 0.00878 | 0.999964 | 0.02141 | [`multi-chunk gate`](results/yue2_nar_gate_multichunk_20260916.json) |
| Multi-chunk, chunk 1 (26 frames) | 0.01026 | 0.999948 | 0.02141 | [`multi-chunk gate`](results/yue2_nar_gate_multichunk_20260916.json) |
| Multi-chunk, chunk 2 (26 frames) | 0.00876 | 0.999963 | 0.02141 | [`multi-chunk gate`](results/yue2_nar_gate_multichunk_20260916.json) |
| Multi-chunk, chunk 3 (18 frames) | 0.01071 | 0.999944 | 0.02141 | [`multi-chunk gate`](results/yue2_nar_gate_multichunk_20260916.json) |

These cases are why the gate exists: they exposed two defects that the single release chunk could
not. The solver's midpoint buffer was only partially re-initialized, so a chunk shorter than the one
before it evaluated its midpoint against stale boundary rows (0.16667 before the fix, 0.01071
after), and the hipBLASLt problem cache was built only for the largest chunk's tile, so a shorter
later chunk launched an unprepared row count. Both are fixed and both cases pass every chunk.

### Radeon 8060S: YuE2 3B VAE decoder

The torch-free FP32 Oobleck decoder against the pinned upstream reference on the same host. Weight
normalization is folded once at load time; the conv, transposed-conv and SnakeBeta kernels
accumulate in FP32 with the reference's own loop order. Error is measured against the reference
waveform, whose own tiled and full decodes already differ by 1.33e-06 - so these figures sit at the
reference's own FP32 noise floor rather than at a modelling difference.

| Case | Samples | max abs | rel L2 | Peak | Full decode | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 3-frame latent | 5 696 | 1.27e-06 | 1.99e-06 | 0.483541 | 0.04 s | [`VAE gate`](results/yue2_vae_gate_20260916.json) |
| 64-frame latent | 122 816 | 2.99e-06 | 1.88e-06 | 0.991600 | 0.68 s | [`VAE gate`](results/yue2_vae_gate_20260916.json) |
| 64-frame latent, tiled 16/16 | 122 816 | 2.79e-06 | - | - | 1.51 s | [`VAE gate`](results/yue2_vae_gate_20260916.json) |

Tiling is exact rather than approximate: with a 16-frame core and 16 frames of halo the tiled
waveform is **bit-identical** to the same runtime's full decode, and it matches the reference's own
tiled output to 2.79e-06. There is no crossfade, no boundary smoothing, and no zero padding of the
final audio; the last tile's crop is shorter than its own output because the natural length is
`1920 * frames - 64`. The decoder's own dependency interval gives a required halo of 12 frames for a
16-frame core, which the gate asserts against the recorded value. Tiled decoding is slower than the
full decode here (1.51 s vs 0.68 s at 64 frames) because every tile re-decodes its halo context; its
purpose is bounding device residency, not wall clock.

### Radeon 8060S: YuE2 3B end-to-end product path

The complete staged pipeline - AR plan and semantic decode, the acoustic flow-matching solve, and
the FP32 Oobleck decode - driven through the registered `yue2` model plugin with no torch in the
process. Each committed case fixture stores the reference pipeline's own assembled prefix and
semantic codes, so the gate replays the reference's exact conditioning and checks the product path
against the audio the reference produced for it: every case reproduces the recorded latent frame
count and the recorded sample count exactly.

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

Across all twelve cases that is **21,000 latent frames** and **40,319,232 samples** of audio, all
finite and all non-silent, at 4 ODE steps rather than the product's 32 (the replay takes 55 min,
from 148 s for the shortest case to 370 s for the longest; the reference's own recorded runs take
150-370 s per case). Solver arithmetic parity is the M4 gate's claim, so step count does not affect
what this one measures. Replaying `english-full-s1234` twice gives **bit-identical** latents. The
per-case replay times in the table are the figures this gate produced at the revision that
established the frame and sample counts; the same twelve cases at 2 ODE steps replay in **527.4 s**
in total with the attention and projection work that followed.

Measured against the reference's own recorded runs on the same host, per stage, across all twelve
cases at 2 ODE steps:

| Stage | hipEngine | Reference | Ratio |
| --- | ---: | ---: | ---: |
| FP32 Oobleck decode | 331.76 s | 2 049.23 s | **6.18x faster** |
| Acoustic solver, raw | 126.92 s | 537.52 s | 2.81x faster (fewer steps) |
| Acoustic solver, per ODE step | 63.46 s | 16.80 s | **3.78x slower** |

Both rows above compare against the reference's *recorded product* runs, and the decode row is not a
decode comparison: on this host the reference's first decode in a process costs 8x its steady state
(163.1 s against 19.6 s for the same 1 297 frames), and its recorded product decode carries that
first-use cost plus a per-request decoder transfer. The matched comparison is in "YuE2 3B
stage-by-stage matched timing" below. The solver row is not like for like either, because the
reference always solves at its own 32 steps; only the per-step row is matched, and that row was
**26.29x** before the attention and projection work described in the next section, **5.79x** after
the scalar attention work, and **3.78x** now. `scripts/yue2_reference_comparison.py` produces both
tables. Evidence: [`comparison`](results/yue2_reference_comparison_geometry_20260917.json),
[`previous geometry`](results/yue2_reference_comparison_wmma_20260917.json),
[`scalar attention`](results/yue2_reference_comparison_20260917.json),
[`pre-fix comparison`](results/yue2_reference_comparison_pre_attention_fix_20260917.json).

A separate live run exercises the whole session rather than recorded conditioning: a greedy request
plans and decodes 96 semantic tokens, solves them to 96 latent frames and decodes 3.84 s of audio in
16.4 s. Running it twice returns identical latent and audio identities. `torch` is not imported by
either run; the gate fails if it ever is. `scripts/yue2_e2e_gate.py` is the reproducing command.
Evidence: [`e2e replay`](results/yue2_e2e_gate_20260916.json),
[`e2e live`](results/yue2_e2e_live_gate_20260916.json).

The stage comparison above replays recorded conditioning, so it never times the AR stage.
`scripts/yue2_case_timing.py` drives one recorded case through the complete product path at the
product's 32 ODE steps - plan, semantic decode, solve, decode - and prints each stage against the
reference's own recorded timing for that request (`mandarin-off-s1234`). Per unit of work, which is
the like-for-like reading:

| Stage | hipEngine | Reference | Ratio |
| --- | ---: | ---: | ---: |
| AR semantic decode | 57.3 ms per token | 31.3 ms per token | **1.83x slower** |
| Acoustic solver, 32 steps | 22.3 ms per frame | 21.1 ms per frame | 1.06x slower |
| FP32 Oobleck decode | 16.3 ms per frame | 53.3 ms per frame | **3.27x faster** |

Those are medians of three runs at current HEAD. The harness caps the semantic phase at the
reference recording's own token count, so both sides now do the same work: 1 298 frames against the
reference's 1 297, and 51.92 s of audio against 51.88 s. The wall clocks are therefore comparable
for the first time, and they say the product path is **ahead**:

| Whole path, identical request and work | Elapsed | Audio | RTF |
| --- | ---: | ---: | ---: |
| hipEngine at current HEAD (median of 3) | 124.54 s | 51.92 s | **2.40** |
| hipEngine with the pre-change AR paths (median of 3) | 155.22 s | 51.92 s | 2.99 |
| torch reference, pinned product run | 137.80 s | 51.88 s | 2.66 |
| torch reference, fresh product runs on this host | 235.60-260.19 s | 51.88 s | 4.54-5.02 |

The two hipEngine arms differ only in the AR paths, and every run of both arms produced the same 1
298 tokens, the same 2 492 096 samples and the same PCG64 state digest. The paired elapsed ratio is
1.236 (range 1.213-1.246 over the three pairs), so the AR change is worth **19% of whole-path
time**; the AR stage itself went from 79.2 to 57.3 ms per token, a paired ratio of 1.380. The NAR
and VAE rows move together across the three repetitions in both arms (about +20% and +10% over the
session), which is machine drift, not an arm effect - the arms track each other, so the paired
ratios are the stable quantity and the absolute seconds are not.

The reference's own product path is not reproducible here. Two fresh runs today take 235.60 s and
260.19 s against its pinned 137.80 s, and the entire difference is its VAE stage: 170.09 s and
186.35 s against the pinned 69.09 s, while its AR (41.96 and 43.55 s against 40.60 s) and NAR (20.16
and 26.76 s against 27.32 s) agree within noise. Its warm tiled decode of these same latents is
20.15 s, measured in the same session as everything else above, so even its pinned product run pays
3.5x its own warm decoder and today's runs pay 8.5x. That is its per-request decoder handling rather
than the machine. All three reference rows are reported as measured, and the pinned one is what the
per-stage table compares against, because its stages are the ones its recorded timings describe.

So the product path is ahead of the reference in both states - 124.54 s against 137.80 s pinned, and
against 235.60-260.19 s today - for a reason that is worth stating precisely: this runtime's decoder
stays warm and pays 16.3 ms per frame, while the reference's re-does per-request decoder setup. Its
steady-state stages are still the faster ones, which is why the matched table above is the
implementation comparison and this table is the product one.

The AR row is the entire remaining gap, and the stage is 60% of our elapsed time. A fixed-token
comparison of the two AR decode paths pins it down, with both sides driving the same recorded
trajectory for the same 1 297 steps over the same 98-token positive and 12-token negative prefixes
and the same two CFG branches, timed alternately in one session
(`scripts/yue2_ar_matched_timing.py`, `scripts/yue2_reference_ar_timing.py`): **54.94 ms per decode
step against 30.88**, so the reference is **1.78x faster** on matched work. Sampling is excluded
there by design, because the token is recorded.

The cause is traffic, and it is now measured on both sides. A branch reads ~3.575 GB per step, of
which 2.819 GB is the 28 layers of q/k/v/o/gate/up/down and 756 MB the output head, from the
checkpoint's own tensor sizes. This path forwards the two CFG branches separately, so it reads the
layers twice, and it now reads only the semantic phase's window of the head (134 MB rather than 756
MB): **5.77 GB per token at 57.3 ms, about 101 GB/s**. The reference batches both branches into one
forward per step (`GraphAR(model, [prefix, negative], ...)` in its `yue2/sampling.py`), so it reads
the layers once: **3.58 GB per token at 30.9 ms, about 116 GB/s**. The two rates are within 15% of
each other, so the stage is not at a bandwidth ceiling - this host's LPDDR5X peak is ~256 GB/s - and
what separates them is that we move 1.6x the bytes. Removing the duplicated branch traffic would put
the same rate at ~2.95 GB per token, or about 29 ms
- the reference's own step - but that is a projection from measured traffic, not a
measurement. Two-row kernels are the route: the head already has one
(`dense_gemv_bf16_f32_out_rowtile2`, 112 GB/s to 199 GB/s at the production shape), and the layer
GEMVs are the same family, documented at 20-28% of peak on this host
(`scripts/gguf_q8_0_dense_bw_microbench.py`).

Host sampling is the smaller lever and it is now small: **7.67 ms per token over the full row
against 0.71 ms windowed** (`scripts/yue2_ar_sampling_cost.py`, CPU-only, single-threaded, 30
medians per call; the window is `semantic`'s 32 769 rows of 184 704, and the windowed arm is what
the loop runs). The full-row figure is 2.60 ms of CFG combine, 3.89 ms of repetition penalty plus
top-k/top-p, 0.22 ms of softmax and 0.94 ms of the categorical draw. So the product's 57.3 ms per
token is model work plus about a millisecond of sampling, and the step itself is the lever - not the
sampler. Evidence: [`matched AR timing`](results/yue2_ar_matched_timing_20260917.json),
[`AR sampling cost`](results/yue2_ar_sampling_cost_20260917.json),
[`product case timing`](results/yue2_product_case_timing_20260917.json).

### Radeon 8060S: YuE2 3B stage-by-stage matched timing

Every earlier stage comparison in this file put one side's warm measurement against the other side's
recorded product run, which mixes in per-request overheads and, for the decoder, an 8x first-use
cost. These three harnesses give both sides identical inputs and compare steady-state passes, timed
alternately in one session on `mandarin-off-s1234` at the product's 32 ODE steps:

| Stage | Inputs shared | hipEngine | torch reference | Reading |
| --- | --- | ---: | ---: | --- |
| AR decode, 1 297 steps | recorded token trajectory, both prefixes, 2 CFG branches | 54.94 ms/step | 30.88 ms/step | **reference 1.78x faster** |
| Acoustic solver, 32 steps | prefix, codes, seed and the reference's own noise | 27.35 s | 21.81 s | **reference 1.25x faster** |
| FP32 Oobleck decode, 1 297 frames | the case's recorded latents, tiled 1024/16 on both sides | 20.08 s | 20.15 s | reference 1.00x faster (level) |

Every row above is a same-session re-measurement of both sides at current HEAD, on
`mandarin-off-s1234` at the product's 32 ODE steps. The AR row is the fixed-token replay: both sides
drive the same recorded trajectory, so sampling is excluded by design (the token is recorded) and
the number is model work only. The replay uses the full-vocabulary head, which is what the
reference's own rows are, so the windowed head and the window-relative sampler do not appear in it;
the product loop below is where they show up.

The AR step was 58.30 ms/step before the paired-branch head (see the next section): the lm head runs
once for both branches through a two-row F32-output GEMV, which is bit-identical per row, leaves the
run's trajectory digest unchanged, and took the step to 54.88, against 54.94 on the re-measurement
above.

Each harness refuses to report when the two sides did not run the same work: the AR harness compares
prefix, negative-prefix and trajectory digests, and the solver harness compares the initial-noise
digest after being handed the reference's own whole-song draw (the two implementations' seeded draws
differ by design, and the first version of this comparison silently compared two different
trajectories). The solver's latent norm is 287.400 against 287.113 on that shared noise, which is
the arithmetic-parity cross-check the M4 gate makes on recorded fixtures. Our tiled decode is
bit-identical to our own full decode (max abs 0.0) while the reference's differ by 1.4e-06, its own
FP32 noise floor.

So on matched, steady-state work this runtime is behind on two stages and level on the third. What
the product path does differently is overhead: the reference moves its decoder off the device after
every request and back on for the next one, and its recorded product decode of these same latents is
69.09 s against its own warm 19.60 s, while this runtime pays about 8 ms per token of host sampling
that the reference does with device ops. Evidence:
[`matched AR timing`](results/yue2_ar_matched_timing_20260917.json),
[`matched solver timing`](results/yue2_nar_matched_timing_20260917.json),
[`matched decoder timing`](results/yue2_vae_matched_timing_20260917.json).

### Radeon 8060S: YuE2 3B paired-branch projections

The AR decode runs two CFG branches over one set of weights, and the profile of a step is 79%
projection GEMVs. The lm head alone streams 756 MB of BF16 weight per call, once per branch.
`dense_gemv_bf16_f32_out_rowtile2` puts two rows in one block so the weight is issued once for both
branches, keeping each row's K traversal and reduction order, which makes a row bit-identical to the
single-row kernel.

| Shape (in x out) | Serial, 2 calls | Two-row, 1 call | Speedup | Serial GB/s | Two-row GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| LM head 2048 x 184704 | 6.723 ms | **3.798 ms** | 1.77x | 112.5 | 199.2 |
| q 2048 x 2048 | 0.060 ms | 0.033 ms | 1.78x | 140.7 | 250.6 |
| o 2048 x 2048 | 0.054 ms | 0.031 ms | 1.77x | 154.3 | 272.7 |
| v 2048 x 1024 | 0.041 ms | 0.028 ms | 1.46x | 102.7 | 150.3 |
| k 2048 x 1024 | 0.028 ms | 0.022 ms | 1.30x | 149.0 | 193.1 |

Per-call latency with a synchronize per call, best of 30, after 5 warmups. In the runtime only the
head is paired so far, which is where the traffic is: the step goes from **58.30 ms to 54.88 ms**
(1.06x) on `mandarin-off-s1234` with the trajectory digest unchanged at `6a308700b3b949ac`, so the
paired head reproduces the serial head exactly. Each branch keeps its own KV spans and its own
positions (this case's positive prefix is 98 tokens against the negative branch's 12); only the
weight read is shared. Evidence: [`kernel A/B`](results/yue2_ar_rowtile_bench_20260917.json),
[`step timing`](results/yue2_ar_matched_timing_20260917.json), gate
`tests/test_unit_yue2_ar_gemv_rowtile2.py`.

### Radeon 8060S: YuE2 3B phase-windowed output head and sampler

`distribution` masks every row outside a phase's domain and the phase's end token to `-inf` before
any other arithmetic, so a phase can only ever select inside its own window: 32 769 rows for
`semantic` (the 32 768 codec tokens plus `MUSIC_END`, 17.7% of the vocabulary) and 151 849 for
`abc`. The head projects just that slice, and the sampler runs in window coordinates, so mask,
repetition penalty, top-k, top-p, softmax and the draw all work over the window instead of the full
vocabulary.

| Phase | Head, full 184 704 rows | Head, windowed | Speedup | Rows | Weight read |
| --- | ---: | ---: | ---: | ---: | ---: |
| `semantic` | 3.84 ms | **0.82 ms** | **4.68x** | 32 769 | 722 MiB → 128 MiB |
| `abc` | 4.13 ms | 3.44 ms | 1.20x | 151 849 | 722 MiB → 593 MiB |

Both head rows are medians of 15 interleaved pairs, with the two arms measured inside one repetition
and the order flipped each repetition; the paired ratio spans 4.41-5.01 for `semantic` and 1.18-1.39
for `abc`. A separate interleaved run gave 4.53x and 1.20x.

| Sampler stage, `semantic` | Full row | Window | Speedup |
| --- | ---: | ---: | ---: |
| mask, penalty, top-k, top-p, softmax, draw | 5.86 ms | **0.43 ms** | **13.5x** |
| the same for `abc` (82.2% of the vocabulary) | 3.37 ms | 2.56 ms | 1.32x |

The production session loop with two-branch CFG, temperature 1.0, top-p 0.95, top-k 100, repetition
penalty 1.2 over a 50-token window on `mandarin-off-s1234`, three arms measured inside one
repetition with the order rotated, 5 repetitions per arm:

| Arm | ms/token | Saved vs baseline |
| --- | ---: | ---: |
| full head, full-row sampler | 61.11 | — |
| windowed head, full-row sampler | 55.14 | 5.97 ms (1.11x) |
| windowed head, windowed sampler | **43.80** | **17.32 ms (1.40x)** |

A second interleaved run measured the same comparison as 60.84 → 43.75 ms/token, 17.10 ms saved,
within 1.3% of the first. The saving exceeds the two head calls' own 6.0 ms because the
device-to-host row and the host-side conversion shrink from 739 KB to 131 KB per branch as well. All
15 runs produced **identical tokens and an identical PCG64 state digest**.

The windowed row is bit-identical to the full projection inside the window and `-inf` outside it for
both phases, and the windowed sampler returns bit-identical scores and softmax probabilities to the
full-row sampler, with the same top-1 and the same drawn token, over 432 cases (both phases x 3
sampling settings x both arithmetic modes) on rows recorded from the pinned reference. Windowed
execution is covered by those checks and by the session gate below; the replay matrix keeps using
the unwindowed call, so it validates that fallback and the paired head rather than the window
itself. Evidence: [`head window validation`](results/yue2_ar_head_window_20260917.json),
[`sampler window validation`](results/yue2_sampler_window_20260917.json),
[`session gate`](results/yue2_session_gate_windowed_sampler_20260917.json).

### Radeon 8060S: YuE2 3B NAR attention packing and projection selection

The NAR attention kernel now owns eight query rows per block and reuses every loaded K/V element
across them, and the projection path selects by measured hipBLASLt index instead of the first
returned algorithm. On gfx1151 (zbook) the scalar attention kernel falls 177.22 ms to **19.33 ms**
per call (**9.17x**), the f16 WMMA tensor-core path runs the production shape at **2.13 ms** against
the scalar 19.0 ms (**9.3x**), and the product 32-step solve drops 443.4 s to **26.07 s**
(**17.01x**). The scalar kernel stays the strict fallback behind
`HIPENGINE_YUE2_NAR_ATTENTION=scalar`; the tensor-core path changes arithmetic by design and is held
to the production gates. Evidence:
[`M7 attention packing`](results/yue2_m7_attention_packed_20260917.json),
[`M7 tensor-core attention`](results/yue2_m7_attention_wmma_20260917.json). Profiling, block-geometry,
projection-ceiling, and parity detail is archived in [`HISTORY.md`](HISTORY.md#radeon-8060s-yue2-3b-nar-attention-packing-and-projection-selection).

### Radeon Pro W7900: YuE2 3B on gfx1100

YuE2 on the default claim hardware, against the same committed fixtures that were recorded on
`gfx1151`. Every gate passes and the product path generates unassisted through the registered
plugin with no torch in the process.

| Gate | Result | Artifact |
| --- | --- | --- |
| AR replay, 18 cases / 576 rows | mean KL **1.05e-3**, top-1 **95.31%**, top-8 **95.92%** | [`AR replay`](results/yue2_ar_replay_gfx1100_20261018.json) |
| NAR solver, release / multi-chunk / restricted visibility | 4-step latents rel L2 **0.01114** / **0.01144** / **0.00865**, cosine 0.99994+ | [`NAR`](results/yue2_nar_gate_gfx1100_20261018.json), [`multi-chunk`](results/yue2_nar_gate_multichunk_gfx1100_20261018.json), [`cond_end`](results/yue2_nar_gate_condend_gfx1100_20261018.json) |
| VAE decoder, 3-frame latent | max abs **1.27e-06**, rel L2 1.99e-06 | [`VAE gate`](results/yue2_vae_gate_gfx1100_20261018.json) |
| Greedy session, 3 cases | prefix identity 3/3; teacher-forced ABC 99.08% / 99.83%, semantic 98.05% / 96.88% / 96.68% | [`session gate`](results/yue2_session_gate_gfx1100_20261018.json) |
| End-to-end, 12 cases + live | **12/12** exact frame and sample counts, bit-identical repeat, 125.2 s | [`end-to-end gate`](results/yue2_e2e_gate_gfx1100_20261018.json) |

The product path at its defaults (`cot="off"`, seed 20260916, 32 ODE steps, tiled decoder)
produced 1 862 latent frames / 74.48 s of finite, non-silent 48 kHz stereo audio in **79.60 s
(RTF 1.069)** once warm and byte-identical on a repeat, from semantic decode **58.70 s**
(31.74 tok/s), solver **12.63 s** and decoder **8.27 s**. `hipengine serve` builds the pipeline in
27.5 s at a 13.23 GiB peak and serves `POST /v1/audio/songs`. There is no torch oracle on this
host, so these are absolute `gfx1100` measurements with no reference arm; the matched comparison
remains the `gfx1151` one above. Evidence:
[`product path`](results/yue2_product_gfx1100_20261018.json),
[`paired forward`](results/yue2_ar_paired_forward_gfx1100_20261018.json).

#### Paired CFG branches

Both CFG branches read the same weight matrices, so one two-row launch per projection replaces two
one-row launches. Each row keeps the single-row kernel's K traversal and reduction order, which
makes the paired step bit-identical to the serial one: 12 steps at product depth, comparing the
logits and both branch hidden rows every step, produced 0 mismatches. BF16-output twins cover the
`o`, `down` and `gate`/`up` projections; the LM head was already paired.

| Segment | Serial | Paired | Speedup |
| --- | ---: | ---: | ---: |
| Projection and attention forward | 36.07 ms | **30.20 ms** | 1.19x |
| Output head, both branches | 1.08 ms | 1.08 ms | 1.00x |
| Sampler | 0.81 ms | 0.82 ms | 1.00x |
| **AR step** | **37.96 ms** | **32.10 ms** | **1.18x** |

150 timed steps after 900 warm steps at product depth, with a synchronize after each segment so a
segment is not credited with the device's unfinished work. The end-to-end effect is 90.36 s →
**79.60 s** and RTF 1.21 → **1.069** with the audio identity unchanged.

At depth the projections are no longer the dominant term. The fixed part of a paired forward is
12.18 ms at 98 tokens of context and 30.22 ms at 998, so 18.0 ms of the step is span attention,
which is 60% of the forward there and grows linearly in context.

#### Split-K decode attention

The strict span kernel launches one block per query head, which is 16 blocks for 40 CUs on a decode
step, and reads the KV rows with a per-token stride. The production route cuts the KV range into
`splits` blocks per KV head (8 x 32 = 256 blocks at product depth), folds that KV head's query group
into the block, covers head_dim with a warp's lanes so each KV row is one coalesced run, and stages
the split's scores in shared memory. Only the softmax reduction order changes: per-split maxima and
sums combined afterwards. The strict kernel stays registered as the parent and
`attention_route="strict"` selects it.

The split count is planned per step from the context and from the device's compute-unit count, so
the same kernel fills 40 CUs on gfx1151 and 96 on gfx1100: the plan takes the larger of a 32-token
chunk target and two blocks per compute unit, bounded to 64 splits and never finer than eight tokens
a split. At product depth the chunk target already fills both parts, so both plan 32 splits. A
kernel-only sweep at 998 tokens on the 96-CU W7900 put the floor at 24-32 splits (0.026 ms per call)
against 0.032 ms at 48, which is where the two-blocks-per-unit bound comes from; at 4 098 tokens the
plan reaches the 64-split cap and costs 0.062 ms against the 0.056 ms floor at 48. Evidence:
[`split-K plan sweep`](results/yue2_splitk_plan_sweep_gfx1100_20261018.json),
`scripts/yue2_splitk_plan_sweep.py`.

| Context | Strict step | Split-K step | Speedup |
| ---: | ---: | ---: | ---: |
| 98 | 12.79 ms | **11.93 ms** | 1.07x |
| 998 | 30.81 ms | **12.54 ms** | 2.46x |
| 2 098 | 53.30 ms | **13.52 ms** | 3.94x |

Median of two repetitions per arm, the two routes alternating inside one repetition, through the
product decode loop with both CFG branches paired. The step becomes nearly context-independent
because the KV scan stops dominating. The product path goes from RTF 1.069 to **0.579**.

Against the strict parent, 16 teacher-forced steps at each of the three depths: mean row KL
**5.1e-4 - 8.4e-4**, max **2.1e-3**, top-1 agreement **47/48 rows**, largest branch-hidden relative
delta **2.2%** on bf16 rows. Against the pinned upstream oracle through the committed 18-case /
576-row replay, the promoted route measures pooled mean KL **1.02e-3** and top-1 **96.18%** against
the strict route's 1.05e-3 and 95.31% on the same host, so the reassociation does not move agreement
with the reference. The route is not bit-identical, so a sampled trajectory can diverge from the
strict route's: this product request emits 1 065 latent frames against 1 862, each repeat
bit-identical within its own route. Task quality (blinded listening, lyric adherence) is unrun on
this host for either route. Evidence:
[`split-K A/B and replay`](results/yue2_ar_attention_splitk_gfx1100_20261018.json), gate
`tests/test_unit_yue2_attention_splitk.py`.

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
