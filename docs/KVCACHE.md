# KV Cache — Architecture, Capacity, and Accuracy Report

_Status: K1 dense per-token/per-head INT8 K/V is implemented and physically fits
256K under the 24 GiB portability target, but it is **not quality-admitted** for
the current Qwen3.6 W4-PARO model. The final clean 128K/16 matched-context gate
rejects at mean/max KL `0.85128/4.97382` and `41.18%` top-1. llama.cpp Q8_0 K/V
passes the protocol-matched comparison against its own F16 reference, showing
that eight-bit K/V can be accurate but not that hipEngine's present format is.
K2 compact DMS remains planned. Last updated: 2026-07-13._

This document is the source of truth for hipEngine K/V-cache architecture,
capacity, and fidelity. It supersedes the June 24 interpretation that the old
Qwen3.5 short E2E fixture made 256K INT8 a correctness-passing product route.
That fixture remains useful kernel bring-up evidence; current Qwen3.6
long-context evidence controls the support decision.

## Executive decision

| Question | Answer |
| --- | --- |
| Does 256K INT8 physically fit below 24 GiB tracked memory? | **Yes.** Final clean tracked peak is `22.971 GiB`, leaving `1.029 GiB`; sampled HIP peak is `21.041 GiB`. |
| Is retained storage genuinely INT8 without a persistent BF16 K/V shadow? | **Yes.** Payload is `2,686,976,000` bytes plus `20,992,000` FP16-scale bytes across ten full-attention layers; audits found no BF16 shadow. |
| Does current Qwen3.6 PARO INT8 pass the repository fidelity gate? | **No.** Matched 128K/16 mean/max KL is `0.85128/4.97382`, top-1 is `41.18%`. |
| Did clipping, groupwise scales, mixed K/V precision, BF16 layers/heads, or sink/recent residuals solve it? | **No.** Some pass at 512; none pass both gates at 4K within the memory budget. |
| Does high KL change a bounded functional answer? | **Sometimes.** At 4K, INT8 flips one of two BF16-qualified restricted-choice tasks; at 32K it retains all three qualified tasks. Evidence is partial, not broad quality validation. |
| Does llama.cpp Q8_0 behave the same way? | **No.** On identical Q4_K_M weights, Q8_0-vs-F16 passes 128K/16 at mean/max KL `0.00521/0.08749`, 100% top-1. Its quantizer, weights, and engine differ from PARO. |
| Product status | BF16 K/V remains supported. INT8 is an explicit approximate/capacity diagnostic, not a default or supported 256K route. |

The retained implementation and memory reduction are still valuable. The
product boundary is simply explicit: **capacity evidence is not quality
evidence**.

## Scope and terminology

Three comparisons appear in this report and must not be conflated:

1. **Within-engine cache fidelity:** same engine and same model weights, changing
   only K/V storage. The final PARO INT8-vs-BF16 and llama.cpp Q8_0-vs-F16 rows
   are examples.
2. **Cross-engine same-weight parity:** exact same Q4_K_M file, but llama.cpp
   F16 K/V versus hipEngine GGUF BF16 K/V. This reveals baseline implementation
   drift; it does not isolate cache dtype.
3. **Cross-format context:** PARO W4 weights versus GGUF Q4_K_M weights and
   different K/V quantizers. This can calibrate scale, not prove one kernel is
   the cause of another row.

Other terms:

- **Retained K/V:** K/V still live after prefill and used by decode.
- **No shadow:** no persistent BF16 K/V cache exists beside INT8 retained K/V.
- **Transient prefill oracle:** chunk/layer-local BF16 K/V used to compute exact
  prefill attention before writing retained INT8. It is released before decode
  and is not a retained shadow, although it contributes to peak allocation.
- **Matched context / teacher forcing:** the candidate consumes the reference
  seed and generated tokens, so each scored logit row has the same token
  history.
- **Prompt-final row:** logits immediately after prefill. A `decode_steps=N`
  matched run contains this row plus N teacher-forced decode rows.
- **Independent rollout:** each route consumes its own greedy outputs. After the
  first mismatch, KL includes trajectory divergence and is not intrinsic cache
  error.

## Architectural contract and memory math

### `KVLiveSpans` remains the ABI

Every paged K/V write and attention kernel consumes `KVLiveSpans` rather than a
scalar `(block_table, context_len)` shortcut. Dense BF16 and dense INT8 fill
uniform spans; compact DMS will fill per-head variable spans. Storage precision
and eviction/compaction remain orthogonal policy axes.

### Qwen3.6/PARO dense K/V size

Only ten full-attention layers own dense K/V. The other 30 layers use recurrent
Conv/GDN state and are not part of the dense K/V byte calculation.

```text
BF16 bytes/token =
  10 layers * 2 KV heads * 256 head_dim * 2(K,V) * 2 bytes
= 20,480 bytes/token

INT8 payload bytes/token =
  10 * 2 * 256 * 2 * 1 byte
= 10,240 bytes/token

FP16 scale bytes/token =
  10 * 2 heads * 2(K,V) * 2 bytes
= 80 bytes/token

INT8 total before allocator padding ≈ 10,320 bytes/token
```

Approximate/current retained sizes:

| Context | BF16 retained K/V | INT8 retained K/V | Interpretation |
| ---: | ---: | ---: | --- |
| 128K | `2.690 GB` | `1.355 GB` | INT8 saves about 49.6%. |
| 256K | about `5.37 GB` projected | `2.708 GB` measured | 256K INT8 is approximately the retained footprint of 128K BF16. |

The final compact persistent prefill table uses `4,096 x 1,025` INT32 entries
(`16,793,600` bytes) at 256K instead of a prompt-rows-by-blocks table. The 128K
table is `8,404,992` bytes. This metadata change reclaimed `0.986 GiB` and
produced the final `1.029 GiB` margin; it does not alter retained K/V fidelity.

### Non-negotiable rules

- **No persistent BF16 shadow for an INT8 claim.** Chunk-local transient BF16 is
  allowed only when its lifetime and byte count are audited.
- **Storage dtype and eviction policy stay independent.** `paged_int8`,
  `dms_int8`, and future formats register as policies/kernels, not engine
  branches.
- **Capacity rows record tracked peak, sampled HIP memory, retained payload and
  scales, transient metadata, and no-shadow evidence.**
- **Quality gates precede support or speed claims.** A route that fits but fails
  fidelity remains a capacity diagnostic.
- **Fused INT8 attention retains an unfused/reference path** for correctness and
  bisection.

## Accuracy methodology

### Primary gate

The repository gate is:

- mean KL divergence <= `0.05`; and
- top-1 agreement >= `90%`.

Both must pass. Max KL, first mismatch, reference-token rank, and per-position
rows are retained as diagnostics. All logits must be finite.

For the current long-context gate:

1. Run BF16 reference prefill and retain the prompt-final full-logit row.
2. Greedily choose the BF16 reference token.
3. For each decode step, feed the BF16 token into both BF16 and candidate
   sessions.
4. Capture full logits after each identical input.
5. Compute KL as `KL(reference || candidate)`, top-1 agreement, first mismatch,
   and candidate rank of the reference top-1.
6. Audit K/V memory before accepting any result.

This protocol removes rollout cascade. Independent-rollout rows are retained
only to show behavioral divergence and first mismatch.

### Controls and transfer discipline

- A same-format repeatability control must be zero or numerically explained.
- Short-context passes are screens, not promotion evidence. Candidates transfer
  through 4K before an expensive 128K gate.
- Format emulation ranks ideas but does not substitute for a native runtime gate.
- Repeated token `9707` is a deterministic numerical probe, not a natural
  quality benchmark.
- Functional checks are scored only when the BF16 reference first answers the
  independently known task correctly. We do not relabel BF16 output as truth.

## Complete accuracy evidence

### 1. Historical Qwen3.5 bring-up: valid kernel evidence, superseded product evidence

The May 18 Qwen3.5/PARO E2E fixture reported `max_kl=0.015328`, 100% top-1,
and matching generated IDs. It validated the initial writer, scale handling,
INT8 split-K/GQA decode, and no-shadow layout. The old KVCACHE status promoted
that result too broadly to “256K passes correctness.”

It did **not** test the current Qwen3.6 packed model on a 128K matched history.
The current support decision therefore treats these rows as historical
bring-up/capacity evidence only:

- [`2026-05-18-hipengine-qwen35-int8-kv-128k-quality-perf-diagnostic.json`](../benchmarks/results/2026-05-18-hipengine-qwen35-int8-kv-128k-quality-perf-diagnostic.json)
- [`2026-05-18-hipengine-qwen35-int8-kv-aotriton-query-reuse-diagnostic.json`](../benchmarks/results/2026-05-18-hipengine-qwen35-int8-kv-aotriton-query-reuse-diagnostic.json)

### 2. Qwen3.6 independent-rollout rejection

The first current-model gate used independently greedy 128K/128 trajectories:

| Scales | Mean / max KL | Top-1 | First generated mismatch | Verdict |
| --- | ---: | ---: | ---: | --- |
| FP16 per-token/head | `3.76460 / 10.07964` | `3.88%` | index 2 | Reject |
| FP32 per-token/head | `3.63001 / 10.01983` | `7.75%` | index 2 | Reject |

FP32 scales did not solve the problem, which rules out scale storage precision
as the dominant explanation. Because histories diverge at index 2, later KL is
behavioral divergence rather than intrinsic error. These rows rejected support
but motivated the matched-context harness:

[`2026-07-12-w7900-v030-paro-context-capacity.json`](../benchmarks/results/2026-07-12-w7900-v030-paro-context-capacity.json).

### 3. Qwen3.6 matched-context sweep

The clean BF16-reference-token sweep shows failure at every tested length:

| Workload | Mean / max KL | Top-1 | First top-1 mismatch | Verdict |
| --- | ---: | ---: | ---: | --- |
| 512/16 | `0.53849 / 1.64209` | `52.94%` | position 2 | Reject |
| 4K/16 | `0.56875 / 1.73523` | `52.94%` | position 2 | Reject |
| 32K/16 | `1.56717 / 3.16370` | `11.76%` | position 1 | Reject |
| 128K/16 final clean | **`0.85128 / 4.97382`** | **`41.18%`** | position 3 | **Reject** |

The final 128K row reproduces on clean commit `d6504544`; all 17 positions are
finite and the no-shadow audit passes. The result is not a cascade artifact.
The non-monotonic KL/top-1 values also show why one context cannot be used to
extrapolate another.

Artifacts:

- [`2026-07-13-w7900-paro-int8-kv-fidelity-baseline.json`](../benchmarks/results/2026-07-13-w7900-paro-int8-kv-fidelity-baseline.json)
- [`2026-07-13-w7900-paro-int8-kv-accuracy-outcome.json`](../benchmarks/results/2026-07-13-w7900-paro-int8-kv-accuracy-outcome.json)

### 4. Format reconstruction screen

A clean 512/8 real-cache capture emulated candidate K/V reconstructions before
teacher-forced decode. The target was 256K storage with at most 1 GiB extra over
all-layer per-token/head INT8. These values rank ideas inside that emulation;
they are not native production gates.

| Candidate | K / V representation | Mean KL | Top-1 | Extra bytes at 256K | Result |
| --- | --- | ---: | ---: | ---: | --- |
| baseline max-abs | per-head INT8 / INT8 | `0.34450` | `66.67%` | 0 | Reject |
| calibrated clip | 0.999 clip INT8 / INT8 | `0.24194` | `66.67%` | 0 | Reject |
| clip 0.99 | clipped INT8 / INT8 | `0.65708` | `44.44%` | 0 | Reject |
| group64 | group64 INT8 / INT8 | `0.16157` | `100%` | `0.059 GiB` | Reject KL |
| group32 | group32 INT8 / INT8 | **`0.14177`** | `77.78%` | `0.137 GiB` | Best mean KL within budget; reject top-1/KL |
| group16 | group16 INT8 / INT8 | `0.89119` | `55.56%` | `0.293 GiB` | Reject |
| K group16 | group16 K / per-head V | `0.20089` | `77.78%` | `0.146 GiB` | Reject |
| V group16 | per-head K / group16 V | `0.23199` | `55.56%` | `0.146 GiB` | Reject |
| INT8 K + BF16 V | INT8 / BF16 | `0.23403` | `77.78%` | `1.240 GiB` | Reject and over budget |
| BF16 K + INT8 V | BF16 / INT8 | `0.14873` | `100%` | `1.240 GiB` | Reject KL and over budget |

Additional observations:

- K and V absolute distributions differ, but calibrated 0.999 clipping improves
  normalized reconstruction RMSE only marginally and does not approach the
  logit gate.
- Keeping K in BF16 helps more than keeping V in BF16 in this screen, but costs
  about 1.24 GiB at 256K and still fails KL.
- Finer groups are not monotonically better: group16 is worse than group32/64.
- No screened all-layer format passes both thresholds even at 512/8.

Artifact:
[`2026-07-13-w7900-paro-kv-format-ablation.json`](../benchmarks/results/2026-07-13-w7900-paro-kv-format-ablation.json).

### 5. Mixed retention and residual policy transfer

Follow-up emulation tested selective BF16 layers/heads, group32/group64,
K/V-mixed retention, and sink/recent BF16 residual windows.

| Candidate | Screen result | Transfer result | Memory consequence |
| --- | --- | --- | --- |
| group64 + BF16 layers 0-2 + sink/recent 64 | 512/8 passes: KL `0.00445`, top-1 `100%` | 4K/8 KL `0.02467`, top-1 `66.67%`: reject | `+0.787 GiB` over baseline |
| baseline INT8 + BF16 layers 0-2 + sink/recent 64 | 512/8 passes: KL `0.02281`, top-1 `100%` | 4K/8 KL `0.02173`, top-1 `77.78%`: reject | `+0.746 GiB` |
| group64 + BF16 layers 0-3 | 4K/8 KL `0.00998`, top-1 `88.89%` | 4K/16 KL `0.03254`, top-1 `76.47%`: reject | `+1.027 GiB`; nearly consumes compact-table gain |
| group32 + BF16 sensitive layers 0,2,4 | — | 4K/16 KL `0.04136`, top-1 `82.35%`: reject | `+0.840 GiB` |
| per-head/baseline prefix-four follow-up | — | 4K/16 KL `0.04922`, top-1 `64.71%`: reject | about `+0.992 GiB` |
| sink/recent 512 variants | No recovery | 4K top-1 remains below gate | Additional retained BF16 rows |

Single-layer sensitivity at 4K identified layers 2, 0, and 4 as the least-bad
individual interventions, but combinations were non-additive and still failed.
The decisive result is transfer failure: **a 512 pass is not retainable evidence
for 4K or 128K**. No rejected format/policy was implemented in production.

Artifact:
[`2026-07-13-w7900-paro-kv-policy-ablation.json`](../benchmarks/results/2026-07-13-w7900-paro-kv-policy-ablation.json).

### 6. Bounded functional checks

The first free-generation suite covered retrieval, multihop, aggregation,
long-document revision handling, and code. BF16 itself scored 0/5 strict
`FINAL:` answers, so both BF16 and INT8 were `reference_unscorable`. Every
BF16/INT8 generated token-ID row differed, which proves behavioral change but
not which output is correct. The suite therefore cannot support a “no
regression” or “breakage” conclusion.

The replacement uses independently known A-D answers. Both policies consume an
identical fixed assistant prefix, `The correct option is `, and the next token
is scored only among the declared A-D label tokens. An INT8 row counts only
when BF16 first chooses the known-correct option.

| Context | BF16-qualified | INT8 retained | Qualified regressions | Interpretation |
| ---: | ---: | ---: | --- | --- |
| 4K | 2/5 | 1/2 | multihop `D -> C` | Expected-choice margin moves `+0.3036 -> -0.2385`; full-logit KL is `0.42265`. INT8 changes a bounded functional decision. |
| 32K | 3/5 | 3/3 | none | Multihop, aggregation, and long-document remain correct at KL `0.45807`, `0.03534`, and `0.21307`; long-document margin narrows `0.6973 -> 0.1024`. |

This gives a precise answer: large KL **can** flip a task decision, but it does
not imply every answer changes. The evidence remains partial because BF16
qualifies only 2/5 at 4K and 3/5 at 32K, and restricted-choice scoring is not
free generation or a broad benchmark.

Artifact and committed suite:

- [`2026-07-13-w7900-paro-int8-kv-functional-mc.json`](../benchmarks/results/2026-07-13-w7900-paro-int8-kv-functional-mc.json)
- [`benchmarks/prompts/kv-int8-long-context-mc.jsonl`](../benchmarks/prompts/kv-int8-long-context-mc.jsonl)

### 7. GGUF Q4_K_M INT8 K/V history

GGUF has a separate accuracy history from PARO because its Q4_K_M weights,
activation distributions, and resident execution path differ. These June rows
are still useful localization evidence, but they predate the final July 13
protocol-matched llama.cpp harness and should not be mixed across protocols.

#### Pure no-mirror runtime and host-QDQ formats

The actual pure GGUF per-token/head FP32-scale runtime stored all ten
full-attention layers as INT8 with no BF16 mirrors. Its fixed repeated-token
seed-plus-one-decode rows all failed:

| Workload | Mean / max KL | Top-1 | Verdict |
| --- | ---: | ---: | --- |
| 128/1 | `0.08240 / 0.16480` | `50%` | Reject |
| 512/1 | `0.05698 / 0.11396` | `100%` | Reject KL |
| 4K/1 | `0.15398 / 0.30795` | `50%` | Reject |

FP16 scales at 512/1 also reject (`0.12144`, 100% top-1). A native block16
runtime remains primitive-correct but rejects model quality; pure 4K/1 records
mean KL `0.24063`, top-1 `50%`, and prefix-eight block16 still has top-1 `50%`
despite mean KL `0.0001505`.

Host QDQ screens replayed reconstructed rows through BF16 storage and therefore
rank formats rather than prove native kernels:

- Q8_0-like block32 at pure 4K/1 rejects with FP16-scale KL `0.57147` and
  FP32-scale KL `0.12349`, both 0% top-1.
- block16/block64 and mixed K/V layouts also reject the pure 4K screen.
- block1 FP16/FP32 passes numerically, but per-scalar scale metadata is about
  150%/250% of BF16 K/V, so it is not a compression format.

#### Hybrid layer retention and sensitivity

Layer probes found severe downstream amplification from early full-attention
quantization. In the June decomposition, value quantization dominated local
layer-0 error. A 3-BF16/7-INT8 suffix passed one 4K/1 gate at mean/max KL
`0.01403/0.02805`, 100% top-1, but failed 128K/128 at mean/max KL
`3.84901/12.29836`, `16.28%` top-1, with a prompt-final mismatch.

A prefix sweep over 32K/64K/128K found that prefixes 3-8 failed before a
layer-local prefill-oracle correction; prefix 9 was the first passing 128K/128
layout (`0.000259`, `96.90%` top-1) but tracked `25.112 GiB`. With the corrected
layer-local oracle, prefix 8 (8 BF16 + 2 INT8 layers) passes 128K/16 at
`0.005294`, `94.12%` top-1 and 128K/128 at `0.014484`, `96.12%` top-1. It still
peaks at `25.239 GiB`, above both the 24 GiB target and the BF16 reference, so it
is a safety/correctness hybrid rather than a useful capacity solution.

Other localization attempts did not improve the product boundary:

- Key-only INT8/BF16-value kernels pass primitive CPU-reference checks, but pure
  4K/1 rejects at KL `0.87312`, 0% top-1. Prefix 7 passes 128K/16 at KL
  `0.00741`, `94.12%` top-1 but peaks `25.554 GiB`; no promotion.
- Non-contiguous three-INT8-layer masks that avoid sensitive layers still fail
  the 128K/16 gates.
- The block16 hybrid is no better than the per-token/head prefix-eight guard and
  has higher short-gate peak memory.

#### Generated-corpus diagnostic versus old llama.cpp Q8_0

A separate June 24 generated-corpus comparison used different metric scopes:
hipEngine scored prompt-final plus decode rows, while `llama-perplexity` scored
corpus positions. On that corpus, pure hipEngine FP32 per-token/head INT8 was
mixed at 128/1 and 512/1, but passed 4K/1 (`0.001260`, 100% top-1) and 4K/16
(`0.007812`, `94.12%` top-1). The then-used llama.cpp build reported:

| Context | llama.cpp Q8_0 corpus mean / max KL | Same-top | Guard mapping |
| ---: | ---: | ---: | --- |
| 128 | `0.09076 / 0.09076` | `98.41%` | Reject KL |
| 512 | `0.90970 / 13.89132` | `83.92%` | Reject |
| 4K | `1.42488 / 23.20883` | `84.56%` | Reject |

Those rows do not contradict the July 13 llama.cpp Q8_0 128K/16 pass. The old
run used a different llama.cpp build, corpus-position metric, and context
protocol. The exact teacher-forced C-API harness in the next section is the
current comparison.

Current GGUF conclusion: pure/coarse INT8 remains diagnostic, while the
quality-passing prefix-eight hybrid saves too little and exceeds the target
memory envelope. BF16 remains the supported GGUF K/V path until these rows are
refreshed under the current runtime and exact multi-prompt protocol.

GGUF accuracy artifacts:

- [`2026-06-22-gguf-q4km-int8kv-hybrid-correctness.json`](../benchmarks/results/2026-06-22-gguf-q4km-int8kv-hybrid-correctness.json)
- [`2026-06-23-w7900-gguf-q4km-int8kv-hybrid-128k-quality-rejected.json`](../benchmarks/results/2026-06-23-w7900-gguf-q4km-int8kv-hybrid-128k-quality-rejected.json)
- [`2026-06-23-w7900-gguf-q4km-int8kv-prefix-sweep.json`](../benchmarks/results/2026-06-23-w7900-gguf-q4km-int8kv-prefix-sweep.json)
- [`2026-06-24-w7900-gguf-q4km-int8kv-prefill-oracle-prefix8.json`](../benchmarks/results/2026-06-24-w7900-gguf-q4km-int8kv-prefill-oracle-prefix8.json)
- [`2026-06-24-w7900-gguf-q4km-int8kv-block16-diagnostic.json`](../benchmarks/results/2026-06-24-w7900-gguf-q4km-int8kv-block16-diagnostic.json)
- [`2026-06-24-w7900-gguf-q4km-int8kv-keyonly-diagnostic.json`](../benchmarks/results/2026-06-24-w7900-gguf-q4km-int8kv-keyonly-diagnostic.json)
- [`2026-06-24-w7900-gguf-q4km-int8kv-noncontiguous-mask-diagnostic.json`](../benchmarks/results/2026-06-24-w7900-gguf-q4km-int8kv-noncontiguous-mask-diagnostic.json)
- [`2026-06-24-w7900-gguf-q4km-pure-int8kv-layout-sweep.json`](../benchmarks/results/2026-06-24-w7900-gguf-q4km-pure-int8kv-layout-sweep.json)
- [`2026-06-24-w7900-gguf-q4km-matched-int8kv-quality-sweep.json`](../benchmarks/results/2026-06-24-w7900-gguf-q4km-matched-int8kv-quality-sweep.json)

## Comparative calibration

### Protocol-matched summary

| 128K/16 comparison | Weight identity within row | Mean / max KL | Top-1 | Verdict |
| --- | --- | ---: | ---: | --- |
| llama.cpp F16 K/V vs F16 K/V repeatability | Identical Q4_K_M | `0 / 0` | `100%` | Deterministic control |
| llama.cpp Q8_0 K/V vs F16 K/V | Identical Q4_K_M | **`0.00521 / 0.08749`** | **`100%`** | Pass |
| hipEngine GGUF BF16 K/V vs llama.cpp F16 K/V | Same exact Q4_K_M file | `0.26606 / 4.51481` | `100%` | Reject all-position mean; prompt-final dominated |
| hipEngine PARO INT8 K/V vs BF16 K/V | Identical W4-PARO | **`0.85128 / 4.97382`** | **`41.18%`** | Reject |

Relative to the separately normalized PARO row, llama.cpp Q8_0 has 163.47x
lower mean KL and +58.82 top-1 percentage points. This is useful context, not a
direct implementation A/B.

### llama.cpp Q8_0 versus F16

The public-C-API harness loads one model and creates sequential F16 and Q8_0 K/V
contexts. It retains only 17 full-logit rows rather than a context-by-vocabulary
file and forces the Q8_0 run with F16 reference tokens.

- Aggregate mean/max KL: `0.00520759/0.08749123`.
- Top-1: 17/17 (`100%`); reference token rank is 1 at every position.
- Prompt-final KL: `0.08749123`.
- Sixteen decode-only rows: mean/max KL `0.00006487/0.00015978`, 100% top-1.
- F16/F16 repeatability: exactly zero KL on all 17 rows.

This establishes that the protocol is deterministic and that an eight-bit K/V
format can pass on this model/engine. It does **not** prove that simply copying
llama.cpp Q8_0 packing into PARO will pass: Q8_0 block quantization, dequant
math, Q4_K_M weights, and llama.cpp attention differ.

Artifact:
[`2026-07-13-w7900-llamacpp-q8-kv-matched-quality.json`](../benchmarks/results/2026-07-13-w7900-llamacpp-q8-kv-matched-quality.json).

### Same-weight hipEngine GGUF bridge

To separate weight-format mismatch from engine drift, llama.cpp exported its
F16 reference logits and hipEngine ran BF16 K/V on the exact same Q4_K_M file,
prompt, and teacher history.

- All 17 positions: mean/max KL `0.26605665/4.51480768`, 100% top-1: formal
  aggregate rejection.
- Prompt-final row alone: KL `4.51480768`, top-1 still matches.
- Sixteen teacher-forced decode rows: mean/max KL
  `0.00050971/0.00108810`, 100% top-1.

The bridge localizes a major cross-engine discrepancy to bulk-prefill output,
not steady teacher-forced decode. Until prompt-final parity is understood,
cross-engine raw KL should not be used as an oracle for within-PARO K/V error.
Within-engine reference normalization remains authoritative.

Artifact:
[`2026-07-13-w7900-gguf-llamacpp-matched-parity.json`](../benchmarks/results/2026-07-13-w7900-gguf-llamacpp-matched-parity.json).

## Current capacity evidence

### Final W7900 Qwen3.6 rows

| Route | Workload | Tracked peak | Sampled HIP peak | 24 GiB margin | Retained K/V | Quality status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| PARO BF16 K/V | 128K/128 | `22.124 GiB` | `21.107 GiB` | `1.876 GiB` | `2.690 GB` | Supported reference path |
| PARO INT8 per-token/head K/V | 256K/128 | **`22.971 GiB`** | **`21.041 GiB`** | **`1.029 GiB`** | **`2.708 GB`** | Layout/capacity pass; quality rejected |

The compact metadata change improves the July 12 INT8 row:

```text
tracked peak: 25,723,838,504 -> 24,665,296,404 bytes
              23.957 -> 22.971 GiB
change:       -1,058,542,100 bytes (-0.986 GiB, -4.115%)
24 GiB margin: 0.043 -> 1.029 GiB
```

One-shot diagnostic timing is effectively flat/noise-level relative to the
prior row: prefill `632.837 -> 631.457 tok/s`, decode
`40.066 -> 40.008 tok/s`. This is a memory/capacity result, not a speed claim.
The retained INT8 payload is `2,686,976,000` bytes and FP16 scales are
`20,992,000` bytes. No persistent BF16 K/V shadow is present.

The transient BF16 prefill oracle still exists and contributes to high-water;
it is reused/released before decode. Removing it is future memory work, not a
way to repair decode fidelity.

Artifact:
[`2026-07-13-w7900-paro-int8-kv-accuracy-outcome.json`](../benchmarks/results/2026-07-13-w7900-paro-int8-kv-accuracy-outcome.json).

## What the evidence says—and does not say

### Supported conclusions

1. **The capacity engineering works.** Dense 256K INT8 fits below the tracked
   24 GiB target with no persistent BF16 shadow.
2. **The current Qwen3.6 PARO representation is not fidelity-safe.** Failure
   persists under identical token histories and is not rollout cascade.
3. **Scale dtype is not the primary issue.** FP32 scales remain far outside the
   gate.
4. **Simple clipping is insufficient.** Reconstruction RMSE changes little and
   logit quality remains poor.
5. **More granular quantization is not monotonic.** group64/group32 improve some
   metrics; group16 regresses.
6. **K precision appears more sensitive than V in the bounded format screen,**
   but BF16 K alone is too expensive and still fails KL.
7. **Layer/residual fixes do not transfer reliably.** Short-context passes fail
   at 4K, and 4K/8 near-passes degrade over 16 steps.
8. **High KL is a risk signal, not a task-failure count.** One qualified 4K
   answer flips; three qualified 32K answers survive despite substantial KL.
9. **Eight-bit K/V can be accurate.** llama.cpp Q8_0 proves feasibility for a
   different quantizer/model path, not correctness of hipEngine's current one.
10. **Cross-engine prompt-final math is not yet aligned.** Decode-only parity is
    much closer than aggregate parity on the same GGUF.
11. **GGUF and PARO sensitivity differ.** GGUF can pass with only two late
    full-attention layers quantized, but that hybrid exceeds 24 GiB and saves no
    useful capacity; pure and coarser GGUF layouts remain unpromoted.

### Unsupported conclusions

- “256K INT8 is supported because it allocates.” It is not.
- “The old Qwen3.5 fixture proves current Qwen3.6 quality.” It does not.
- “Every large-KL row breaks user-visible output.” The bounded 32K tasks disprove
  that simplification.
- “llama.cpp is a direct oracle for PARO.” Weight and implementation contracts
  differ.
- “PARO accuracy transfers to GGUF, or vice versa.” The retained evidence shows
  materially different layer/format sensitivity.
- “group32/group64 should be implemented because a 512 probe improved.” Neither
  transferred through the gate within budget.
- “The five-task functional probe proves broad quality.” It is partial,
  restricted-choice evidence only.

## INT8 accuracy optimization workflow

The development loop optimizes **matched-context logit fidelity first**. Every
candidate consumes the same model weights, prompt, positions, and teacher-forced
token history as the unquantized BF16 K/V reference. Under that contract, lower
KL is direct evidence that the candidate output distribution is closer to the
unquantized cache; higher top-1/top-k agreement is supporting evidence that fewer
decision boundaries move. Downstream task benchmarks remain necessary for
promotion, but running them for every representation hypothesis would reduce
iteration velocity without improving the local optimization signal.

### Metric policy

| Metric | Development use | Direction / interpretation |
| --- | --- | --- |
| Mean `KL(BF16 || candidate)` | Primary representation-ranking metric | Lower is better; compare only identical matched contexts. |
| Maximum per-position KL | Tail-risk guard | Lower is better; prevents a good mean from hiding one catastrophic step. |
| Top-1 agreement | Greedy-decision fidelity and repository correctness gate | Higher is better; final gate remains at least `90%`. |
| Top-k overlap (`k=5`, `k=10`) | Near-boundary stability | Higher is better; distinguishes a small rank swap from loss of the reference candidate set. |
| BF16 top-1 rank under candidate logits | Decision-boundary diagnostic | Lower is better; rank `1` is exact top-1 retention. |
| K/V reconstruction error | Mechanism/debugging metric only | Useful for localization, but never overrides worse model-logit fidelity. |
| Task accuracy / executable outcome | Milestone and promotion evidence | Required before support/default claims; not part of every inner-loop screen. |

Top-1 is discrete and can move non-monotonically when logits are close, so mean
KL is the primary optimization metric while top-1/top-k are guards. A candidate
with lower KL but worse top-1 is not promoted automatically; it is retained only
as a diagnostic or transferred to the next numerical stage for resolution. No
metric from a free-running candidate rollout is used to rank cache formats,
because token-history cascade would confound the cache error being measured.

### Escalation ladder and wall-time budget

| Stage | Workload | When it runs | Intended wall time | Decision |
| --- | --- | --- | ---: | --- |
| S0: host reconstruction | Captured K/V tensors; no new model session | Every math/layout edit | Seconds | Reject non-finite, shape-invalid, or clearly dominated formats. |
| S1: fast numerical screen | Fixed small prompt mix, `512/8`, BF16-matched teacher forcing | Every credible format hypothesis | **At most 10 minutes total** for the candidate set | Rank by mean KL, then top-1/top-k and memory fit. Diagnostic only. |
| S2: transfer | Winning candidate only, `4K/16` | After a material S1 improvement | Minutes, run separately | Reject short-context overfit or decode-step accumulation. |
| S3: full numerical gate | Winning native candidate, clean `128K/16` | Before any quality/capacity promotion | Full benchmark wall time allowed | Require KL `<= 0.05`, top-1 `>= 90%`, finite logits, and intended-kernel evidence. |
| S4: functional/task gate | Bounded natural suite, then broader long-context/reasoning/code/agentic tasks | At milestones and before support/default status | Separate scheduled run | Measure user-visible impact; report BF16 and candidate absolute scores plus paired deltas. |

The S1 prompt mix is fixed before candidate measurement and contains more than a
single repeated-token prompt when practical. It is a screening fixture, not a
benchmark to tune against. Any S1 winner must transfer through S2 and S3 on
held-out shapes; prompt-specific or 512-only improvements are rejected.

### Harness and benchmark notes

| Harness / evidence | Fast-loop role | Claim boundary |
| --- | --- | --- |
| `scripts/qwen35_paro_kv_format_ablation.py` | Primary S0/S1 emulation screen; load the runner once, compare several formats, and emit KL/top-1/top-k plus 256K memory projections. | Ranks representations only; reconstructed BF16 cache and current-row semantics are not a native production result. |
| `scripts/qwen35_paro_int8_kv_quality_sweep.py` | S2/S3 native matched-context transfer and final numerical gate. | Correctness evidence for the named model/context/history; not downstream task accuracy. |
| `scripts/qwen35_paro_kv_functional_mc.py` | Small milestone smoke after a candidate survives numerical transfer. `--limit`/category selection may be used for developer sanity, while the full retained row remains canonical. | Restricted-choice evidence only; never a broad natural-task claim. |
| RULER/NoLiMa/HELMET-RAG-style length subsets | Later long-context retrieval/reasoning coverage. | Run for finalist/promoted formats, not each quantizer edit. |
| MATH/HumanEval/IFEval/BFCL-Memory-style subsets | Reasoning, executable code, instruction, and agentic/tool-call coverage. | Required to characterize real-world impact; score against ground truth rather than BF16 text equality. |

The external research pack was reviewed read-only at
`/home/lhl/amd-gpu-tuning/reference/kvcache-quantization-research@a0bb333`
(the user-named `/home/lhl/github/shisa-ai/kvcache-quantization-research` path was
not present on this host). It reinforces K/V-asymmetric treatment, chunked
KIVI-style keys, Hadamard rotation, cross-layer AQUA residual prediction, and a
cheap-before-expensive eval ladder. hipEngine screens those ideas in its own
NumPy/BF16 reconstruction harness before considering native HIP work; external
Torch/HF cache code is reference material, not a runtime dependency.

The first externally informed S1 candidate set is deliberately bounded:

| Candidate | Screened representation | Why it is included |
| --- | --- | --- |
| Current baseline | Symmetric INT8 per token/head with one scale | Native-format anchor. |
| Hadamard group32 | Deterministic orthogonal channel rotation, group32 INT8, inverse reconstruction | Tests whether spreading channel outliers improves the already-promising group32 geometry. |
| KIVI-style INT8 | Chunked per-channel K plus per-token/group V with an unquantized incomplete residual block | Tests the established K/V asymmetry under online-feasible chunk semantics. |
| KVarN-inspired INT8 | Hadamard rotation plus two-axis variance normalization and asymmetric chunk quantization | Targets the token-magnitude/tail errors associated with autoregressive accumulation. This is an emulation screen, not a claim of paper-faithful production KVarN. |

AQUA-style cross-layer residual prediction remains the next representation
screen if these local transforms do not pass transfer. It requires fitting and
auditing per-layer predictors and therefore is not mixed into the first
sub-10-minute candidate set.

## Phase K1 — Dense paged INT8 K/V

### Implementation status

The core capacity path is landed:

1. **Paged INT8 writer**
   - `paged_kv_write_int8_per_token_head`
   - Input: post-RoPE BF16/FP16 K/V.
   - Per `(row, kv_head, K/V)` max-abs quantization; separate FP16/FP32 scale
     metadata.
   - Dense/uniform `KVLiveSpans` updates.
2. **Grouped-GQA split-K INT8 decode**
   - `paged_attn_decode_int8_gqa_splitk`
   - Direct INT8 K/V loads and scale application.
   - FP32 QK/softmax accumulation and inline V dequantization.
   - No cache-sized INT8-to-BF16 decode workspace.
3. **Transient exact-prefill bridge**
   - Full-attention prefill uses a temporary BF16 oracle K/V workspace, then
     appends retained INT8 plus scales.
   - Workspace is layer-local/reused and released before decode.
4. **Policy/registry integration**
   - `FixedPagedKVPolicy(storage_dtype="int8_per_token_head")` with
     `scale_dtype="fp16"`, `scale_granularity="per_token_head"`.
   - Dispatch remains keyed by backend/layer/quant/variant.
5. **Memory and layout audits**
   - Fail on persistent BF16 K/V or missing scale metadata.
   - Compact persistent prefill metadata is reusable across chunks.

Representative public wrappers include:

- `qwen35_write_paged_kv_int8_per_token_head_spans(...)`
- `qwen35_write_paged_kv_int8_per_token_head_{prompt,batch}_spans(...)`
- `qwen35_paged_attn_decode_int8_gqa_splitk_spans(...)`
- `qwen35_paged_attn_decode_int8_gqa_splitk_gate_{bf16,fp16}_spans(...)`

### Storage format

```text
K cache: int8 [layers, pages, block_size, kv_heads, head_dim]
V cache: int8 [layers, pages, block_size, kv_heads, head_dim]
K scale: fp16/fp32 [layers, tokens/pages, kv_heads]
V scale: fp16/fp32 [layers, tokens/pages, kv_heads]
spans:   KVLiveSpans(storage_dtype=int8_per_token_head)
```

### Current promotion policy

- BF16 remains the default/supported route.
- INT8 can remain explicitly selectable for diagnostics and capacity research,
  with a warning/contract that it is approximate for current Qwen3.6 PARO.
- Do not advertise usable 256K INT8, even though allocation passes.
- Do not add native group32/group64, mixed-cache, or residual-window production
  complexity from the rejected screens.
- A future representation must pass native matched-context, memory/no-shadow,
  and broader natural-task gates before default or support status changes.

### Required gates for any replacement

1. Unit quantize/dequantize fixtures, zero-scale and page-boundary cases.
2. CPU/reference attention gate: KL <= `0.05`, top-1 >= `90%`.
3. Native 512 screen, then 4K/16 transfer, then clean 128K/16 matched context.
4. Multi-prompt natural/functional suite with a scorable BF16 reference.
5. No-shadow memory audit and a 256K row below 24 GiB tracked peak.
6. `rocprofv3` kernel trace proving the intended writer/attention kernels ran.
7. Performance measurement only after correctness; neutral speed is acceptable
   for a genuine capacity feature.

## Reproduction and evidence map

### Current canonical commands

```bash
# Final PARO matched-context 128K/16 gate
HIP_VISIBLE_DEVICES=0 HIPENGINE_HIP_ARCH=gfx1100 PYTHONPATH=$PWD \
python3 scripts/qwen35_paro_int8_kv_quality_sweep.py \
  --model /home/lhl/.cache/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-packed/snapshots/437eba06df05aad71a4dacdcaf3fff70ae1ee8a1 \
  --prompt-lengths 128K --decode-steps 16 --token-id 9707 \
  --max-layers 40 --comparison-mode matched_context \
  --compiler-version-file /tmp/hipengine-w7900-v030/capacity/hipcc-version.txt \
  --require-cached-build --kv-storage int8_per_token_head \
  --json /tmp/hipengine-final-int8-quality-128k-16.json

# Final 256K/128 capacity/layout row
HIP_VISIBLE_DEVICES=0 HIPENGINE_HIP_ARCH=gfx1100 PYTHONPATH=$PWD \
python3 scripts/qwen35_paro_bench.py \
  --model /home/lhl/.cache/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-packed/snapshots/437eba06df05aad71a4dacdcaf3fff70ae1ee8a1 \
  --backend hip_gfx1100 --shared-expert-format packed_paro_w4 \
  --token-id 9707 --prompt-length 262144 --decode-tokens 128 \
  --warmup-decode-tokens 4 --max-layers 40 \
  --compiler-version-file /tmp/hipengine-w7900-v030/capacity/hipcc-version.txt \
  --require-cached-build --attn-aotriton-min-tokens 512 \
  --kv-storage int8_per_token_head --json /tmp/hipengine-final-capacity-256k-128.json

# llama.cpp Q8_0-vs-F16 protocol match
HIP_VISIBLE_DEVICES=0 python3 scripts/llamacpp_kv_matched_context.py \
  --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --prompt-token-id 9707 --prompt-length 131072 --decode-steps 16 \
  --ctx-size 131089 --batch-size 4096 --ubatch-size 512 \
  --n-gpu-layers 99 --threads 16 --reference-cache f16 \
  --candidate-cache q8_0 --flash-attn \
  --reference-logits-bin /tmp/llamacpp-f16-reference.bin \
  --json /tmp/llamacpp-q8-vs-f16-export.json

# Same-weight hipEngine GGUF bridge
HIP_VISIBLE_DEVICES=0 HIPENGINE_HIP_ARCH=gfx1100 PYTHONPATH=$PWD \
python3 scripts/gguf_llamacpp_matched_context.py \
  --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --llama-reference-json /tmp/llamacpp-q8-vs-f16-export.json \
  --llama-reference-logits /tmp/llamacpp-f16-reference.bin \
  --prompt-token-id 9707 --prompt-length 131072 --decode-steps 16 \
  --max-sequence-length 131089 --backend hip_gfx1100 \
  --compiler-version-file /tmp/hipengine-w7900-v030/capacity/hipcc-version.txt \
  --require-cached-build --json /tmp/hipengine-bf16-vs-llamacpp-f16.json

# Bounded functional check; run with 4096 and 32768
HIP_VISIBLE_DEVICES=0 HIPENGINE_HIP_ARCH=gfx1100 PYTHONPATH=$PWD \
python3 scripts/qwen35_paro_kv_functional_mc.py \
  --suite benchmarks/prompts/kv-int8-long-context-mc.jsonl \
  --context-tokens 32768 --max-layers 40 \
  --compiler-version-file /tmp/hipengine-w7900-v030/capacity/hipcc-version.txt \
  --require-cached-build --kv-storage int8_per_token_head \
  --json /tmp/hipengine-kv-functional-mc-32k.json
```

### Artifact index

| Evidence | Artifact |
| --- | --- |
| Final capacity + 128K quality decision | [`2026-07-13-w7900-paro-int8-kv-accuracy-outcome.json`](../benchmarks/results/2026-07-13-w7900-paro-int8-kv-accuracy-outcome.json) |
| Context-length matched sweep + original task blocker | [`2026-07-13-w7900-paro-int8-kv-fidelity-baseline.json`](../benchmarks/results/2026-07-13-w7900-paro-int8-kv-fidelity-baseline.json) |
| Clipping/groupwise/K-V format screen | [`2026-07-13-w7900-paro-kv-format-ablation.json`](../benchmarks/results/2026-07-13-w7900-paro-kv-format-ablation.json) |
| Mixed layer/head/residual policy transfer | [`2026-07-13-w7900-paro-kv-policy-ablation.json`](../benchmarks/results/2026-07-13-w7900-paro-kv-policy-ablation.json) |
| Bounded 4K/32K functional choices | [`2026-07-13-w7900-paro-int8-kv-functional-mc.json`](../benchmarks/results/2026-07-13-w7900-paro-int8-kv-functional-mc.json) |
| llama.cpp Q8_0-vs-F16 + repeatability | [`2026-07-13-w7900-llamacpp-q8-kv-matched-quality.json`](../benchmarks/results/2026-07-13-w7900-llamacpp-q8-kv-matched-quality.json) |
| Same-weight hipEngine/llama.cpp GGUF bridge | [`2026-07-13-w7900-gguf-llamacpp-matched-parity.json`](../benchmarks/results/2026-07-13-w7900-gguf-llamacpp-matched-parity.json) |
| July 12 independent rollout and FP32-scale control | [`2026-07-12-w7900-v030-paro-context-capacity.json`](../benchmarks/results/2026-07-12-w7900-v030-paro-context-capacity.json) |

Format/policy development tools:

- `scripts/qwen35_paro_int8_kv_quality_sweep.py`
- `scripts/qwen35_paro_kv_format_ablation.py`
- `scripts/qwen35_paro_kv_policy_ablation.py`
- `scripts/qwen35_paro_kv_functional_mc.py`
- `scripts/qwen35_gguf_int8_kv_correctness.py`
- `scripts/qwen35_gguf_q8_format_sweep.py`
- `scripts/llamacpp_kv_matched_context.py` and companion C++ harness
- `scripts/gguf_llamacpp_matched_context.py`

## Phase K2 — FastDMS-derived compact DMS

### Goal

After dense INT8 KV lands, port compact DMS semantics from `~/FastDMS` so the
engine stores and scans fewer live tokens. DMS is the better long-context and
concurrency lever because it reduces `live_counts`, not just bytes per live row.

DMS is checkpoint-dependent. It is not a drop-in policy for arbitrary models;
the current Qwen3.6/PARO path needs a DMS-retrofitted checkpoint or a validated
borrowed-channel metadata block before DMS rows can be quality claims. DMS
bring-up must use BF16 storage first; the rejected dense INT8 format is not an
assumed quality-safe DMS storage dtype.

### FastDMS reference map

Use `~/FastDMS` as the semantic and optimization reference, but port to
hipEngine's torch-free HIP/plugin design rather than copying Triton/PyTorch
host code directly.

| FastDMS file | What to reuse |
| --- | --- |
| `fastdms/engine/dms.py` | DMS metadata loading, borrowed-query-channel eviction extraction, alpha scale/offset semantics, and zeroing the decision lane after extraction. |
| `fastdms/engine/compact_kv.py` | Compact allocator, per-layer/per-head `base_offsets`, `range_capacity`, `live_counts`, `token_positions`, `evict_mask`, streaming prefill pack, live-count/rank/scatter structure. |
| `fastdms/layers/compact_attention.py` | Fused decode preprocessing, compact append/store, inline Q RoPE option, grouped split-K compact attention, split-block tuning knobs. |
| `fastdms/engine/scheduler.py` | Admission through compact capacity instead of dense pages; releasing dense blocks after pack in non-streaming modes; streaming-pack mode with no dense blocks. |
| `fastdms/models/qwen3.py` | Qwen DMS integration points: extraction from Q, per-layer eviction recorder, fused preprocess eligibility. |
| `~/FastDMS/training/` | Retrofit recipe: neuron zeroing, DMS distillation, target compression ratio, window size, and metadata packaging. |

FastDMS performance evidence to keep in mind:

- Compact DMS was faster than vLLM BF16/FP8 on Llama-3.2-1B and Qwen3-8B in
  the validated c=1/c=8 rows while using much less allocator-visible KV memory.
- The strongest research compression stack was DMS + AQUA + HIGGS at 25.6×
  theoretical KV compression, but HIGGS speed did not hold; FastDMS promoted
  compact DMS without HIGGS/AQUA for the serving path.
- Streaming pack was important because it eliminates a persistent dense KV
  scratch. hipEngine should start with the streaming/no-shadow shape, not a
  sidecar compact cache that still reserves dense pages.

### hipEngine DMS shape

DMS should register as a `KVPolicy` and compact attention kernel family. Start
with BF16 to isolate compaction semantics; compressed storage is a second,
independently gated axis:

```python
policy = KVPolicy.dms_bf16(
    target_cr=4 or 8,
    window_size=256,
)

# Eventual shape, only after a replacement INT8 representation passes its
# own dense/native gates (current int8_per_token_head does not):
policy = KVPolicy.dms_int8(
    target_cr=4 or 8,
    window_size=256,
    storage_dtype="int8_per_token_head",
)
```

Core metadata is already aligned with `KVLiveSpans`:

```text
base_offsets    [rows, layers, kv_heads] int32
live_counts     [rows, layers, kv_heads] int32
range_capacity  [rows, layers, kv_heads] int32 (policy-owned)
token_positions [rows, layers, kv_heads, max_live] int32
evict_mask      [rows, layers, kv_heads, max_live] bool
storage_dtype   bf16 initially; quality-admitted compressed dtype later
span_role       prefill | decode | verify_chain | verify_tree
```

### Bring-up sequence

1. **DMS metadata and training checkpoint gate**
   - Add `DMSRetrofitConfig` loader for `dms_metadata.json` / training-log style
     metadata.
   - Require explicit opt-in if metadata is missing; no silent DMS on a
     non-retrofitted checkpoint.
   - For the current Qwen3.6/PARO model, train or import an eviction-head
     retrofit before any quality claim.
2. **Compact policy and admission**
   - Add `DMSKVPolicy` with allocator-visible compact capacity.
   - `admission_cap()` returns compact live-token capacity, not logical context
     length.
   - Add no-evict and forced-stride diagnostic modes only for testing the
     compact allocator/kernels; they are not quality claims.
3. **Streaming prefill pack**
   - Port FastDMS' count/rank/scatter structure to HIP.
   - Pack surviving K/V directly into compact BF16 storage first; introduce a
     compressed store only after dense/native fidelity passes.
   - Do not retain a second dense BF16 K/V arena after pack.
4. **Decode append/preprocess**
   - Port fused Q/K RoPE + DMS decision extraction + compact store.
   - Zero the borrowed query decision lane before attention, matching FastDMS.
   - Update `live_counts`, `token_positions`, and `evict_mask` transactionally.
5. **Compact grouped split-K attention**
   - Port compact decode over variable `live_counts`.
   - Reuse the grouped-GQA lesson: scan each KV stream once for all Q heads that
     share it when split geometry makes reuse worthwhile.
   - Tune block-N/split caps only after correctness fixtures pass.
6. **Scheduler and c=N integration**
   - Start c=1, then c=2/4/8 after dense batched spans are green.
   - Continuous batching must account by actual compact live rows. Prefix cache
     should be disabled initially or implemented as per-sequence eviction
     overlays; do not share evicted prefix pages blindly.
7. **Speculative decode compatibility**
   - DMS writes must obey existing KV transaction semantics. Verify rows write
     scratch/journal spans and commit only accepted rows.

### DMS acceptance gates

Correctness/quality:

- DMS-off/no-evict compact mode equals dense reference.
- DMS-on mode passes KL ≤ 0.05 and top-1 ≥ 90% against no-evict/full-KV on the
  fixture set.
- Add a longer PPL/logit-distillation smoke for the DMS-retrofitted checkpoint;
  record token-match/KLD over scored decode tokens like FastDMS did.
- Forced accept/reject speculative fixtures remain isolated from canonical KV.

Capacity:

- Report logical context length, average and max `live_counts`, target vs actual
  compression ratio, compact KV bytes, scale metadata bytes, and allocator peak.
- DMS rows must demonstrate allocator-visible savings, not only masked attention
  over a dense pool.

Performance:

- Compare against dense BF16 and dense INT8 at 128K and 256K.
- Record producer, split-reduce, store/pack, and scheduler/admission time shares.
- Do not promote if compact attention is slower without a compensating capacity
  objective clearly stated.

Soak/stability:

- Include a c=1 long-context soak and a c=8 serving-shaped soak once c=N support
  is available.
- Enable debug checks for early development: bounds, monotonic positions, live
  count ≤ capacity, no negative slot mappings, and no stale `evict_mask` entries.

## Later research: AQUA, HIGGS, TurboQuant-style int4

These are deliberately after dense INT8 and DMS:

| Technique | Current decision | Reason |
| --- | --- | --- |
| AQUA-KV | Research after DMS | FastDMS found it was not required for best FP8+DMS serving quality. It may help if we revisit 4-bit storage. |
| HIGGS 4-bit KV | Defer | Best FastDMS work reached about 50% BF16/FP8 speed on PRO 6000; RDNA3 LUT/Hadamard cost is unlikely to be better. |
| TurboQuant/int4 KV | Optional comparator | Useful if users need maximum capacity, but vLLM/FastDMS evidence showed 4-bit KV can be slower and worse quality than DMS FP8/INT8. |

## Immediate punchlist

1. [x] Land dense per-token/head INT8 storage, writer, grouped-GQA split-K
   decode, scale metadata, and `KVLiveSpans` policy plumbing.
2. [x] Add no-shadow/missing-scale memory audits.
3. [x] Remove persistent full-prompt prefill I/O buffers, release phase scratch,
   reuse AOTriton query scratch, and compact the persistent block table.
4. [x] Re-run clean 256K/128 capacity: tracked `22.971 GiB`, sampled
   `21.041 GiB`, retained K/V `2.708 GB`, no BF16 shadow.
5. [x] Add intrinsic BF16-reference-token matched-context instrumentation and
   run 512/4K/32K/128K Qwen3.6 gates; current INT8 rejects at every length.
6. [x] Screen clipping, group16/32/64, K/V-mixed, selective BF16
   layers/heads, and sink/recent residual policies; reject production work after
   all candidates fail 4K transfer within budget.
7. [x] Add exact llama.cpp F16/Q8_0 calibration and F16 repeatability control.
8. [x] Add same-Q4_K_M llama.cpp-F16-to-hipEngine-BF16 bridge; localize the
   cross-engine aggregate discrepancy to prompt-final bulk prefill.
9. [x] Replace the unscorable free-generation smoke with reference-qualified
   4K/32K restricted-choice diagnostics; retain the one observed 4K regression
   and the 3/3 qualified 32K non-regression as partial evidence.
10. [ ] Resolve hipEngine-GGUF versus llama.cpp prompt-final prefill parity on
    natural and repeated prompts before using cross-engine logits as an oracle.
11. [ ] Establish a broader, scorable BF16 natural-prompt baseline for PARO;
    current restricted-choice coverage is only 2/5 at 4K and 3/5 at 32K.
12. [ ] Investigate a materially different native representation only after
    localization. An exact Q8_0-style block contract is a research comparator;
    the current group32/group64 emulations are not promotable.
13. [ ] Stream/remove the transient BF16 INT8-prefill oracle only as additional
    capacity work; do not confuse this with a fidelity fix.
14. [ ] Port FastDMS metadata/compact allocator semantics and train/import a
    matching DMS retrofit before any DMS quality claim.
15. [ ] Port DMS streaming pack/compact decode and combine DMS with a
    quality-admitted storage dtype; do not assume current dense INT8 is that
    dtype.
