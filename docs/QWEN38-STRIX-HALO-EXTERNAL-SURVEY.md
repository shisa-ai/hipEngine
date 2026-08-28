# Qwen3.8-27B on Strix Halo: external claims and local checks

- Survey date: **2026-08-28**
- Hardware lane: **AMD Ryzen AI MAX+ 395 / Radeon 8060S / `gfx1151`**
- Local model: **Qwen3.8-27B `Q4_K_M`**, BF16 KV, SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Scope: selected public autoregressive (AR) and speculative-decoding reports,
  plus the local tests that can confirm, qualify, or reject their transfer to
  hipEngine
- Supersedes: the intake table in
  [`CONCURRENCY2-GFX1151-MTP-TUNING.md`](CONCURRENCY2-GFX1151-MTP-TUNING.md)
  §2

## Assessment

Public Qwen3.8-27B results on Strix Halo are not one leaderboard. The reported
7.2-163 tok/s range combines different model files, quantization formats,
backends, speculative methods, prompts, generation lengths, context states,
and timing boundaries. The public sources themselves document many of these
conditions: the [community route guide][S1] lists eight changing variables,
[MikeVeerman][S2] reports results by prompt and concurrency, [KyaniteLabs][S4]
labels its 148-163 tok/s count-to-30 result as a repetition artifact, and
[LaurentZuijdwijk][S5] reports 65.6 tok/s for structured output but 26.1 tok/s
for prose with the same adaptive configuration.

Our evidence supports these narrower conclusions:

1. **AR decode is near the memory-bandwidth limit.** The published decode-only
   rows imply approximately 204-228 GB/s on a 256 GB/s theoretical-memory
   system. hipEngine's direct `Q4_K_M` diagnostic is in that band and is within
   1% of julianmb's published stock-llama.cpp `Q4_K_M` result. This is a
   cross-host comparison, not a controlled engine A/B.
2. **Our closest same-host llama.cpp comparison finds C1 parity, with an
   acceptance gap.** On the same GGUF and 10-prompt suite, hipEngine and
   llama.cpp HIP are within 1% on complete-wall AR and MTP throughput.
   hipEngine's measured draft acceptance is 78.57%; llama.cpp's is 90.16%.
   The comparison is diagnostic rather than a retained performance claim
   because the hipEngine rows were reused from earlier runs, KV types differ,
   and each arm has one run.
3. **Prompt choice strongly changes speculative throughput.** In the two
   published per-prompt datasets, acceptance changes by 16.6-23.5 percentage
   points and speedup by 17-29% across four prompt classes. A derived
   cycle-efficiency estimate is much more stable within each dataset, but it
   is not portable across target sizes, draft methods, or timing boundaries.
4. **The 65.6 tok/s adaptive-DFlash result was not reproduced or refuted.** Our
   local transfer test used the same fork's adaptive policy with the target's
   built-in MTP head, not its FP4 DFlash2 sidecar. Adaptive depth recovered
   18.8% over a fixed deep draft but remained 6.7% slower than fixed depth 3
   on our `Q4_K_M` mixed suite. This rejects adaptive depth as the best policy
   for that local MTP configuration; it does not invalidate the published
   FP4+DFlash2 structured-output result.
5. **The 148-163 tok/s repetition result is real but narrow.** KyaniteLabs
   explicitly labels it as repetition-assisted. On our model and mixed suite,
   the same MTP+ngram flag stack gave only 1.07x AR on `general_en` and drafted
   3,917 tokens to accept 935. It should not be presented as general prose or
   chat throughput.

## 1. Evidence labels and comparison rules

Every result below has one of these labels:

- **Published:** reported by an external source; we checked the cited,
  commit-pinned page but did not reproduce the number.
- **Local:** measured on this host and linked to a committed JSON artifact.
- **Derived:** calculated from published or local inputs. A derived value is
  not an independent measurement.
- **Not tested:** the required model, backend, sidecar, or protocol was not run
  locally.

Raw tok/s is compared only when the model file, quant, backend, workload,
context, generation length, and timing boundary are sufficiently matched. A
same-host result is not automatically matched: the two arms must also use the
same artifact and protocol.

### 1.1 AR diagnostic: implied memory bandwidth

For single-request dense AR decode, this survey uses:

```text
implied GB/s = model file bytes x AR tok/s
```

The result is a screening diagnostic, not a hardware-counter measurement.
File size overstates streamed bytes because embeddings are row lookups and
metadata or non-AR tensors may not be streamed for each token. For our local
`Q4_K_M`, the file contains 17.096 GB of tensors, while the measured AR stream
budget is 16.091 GB: 15.048 GB across 64 AR blocks plus 1.043 GB for
`output.weight`, excluding the 0.715 GB token embedding and the MTP block
([local row-scaling artifact][L1]).

A reported AR rate that implies materially more than the 256 GB/s theoretical
LPDDR5X-8000 ceiling is not comparable to these AR rows without another
explanation, such as speculation, caching, a different timing boundary, or an
incorrect model-size assumption.

### 1.2 Speculative diagnostic: estimated cycle efficiency

For fixed draft depth `K`, this survey derives:

```text
estimated tokens/cycle = 1 + K x acceptance
estimated cycle efficiency = measured speedup / estimated tokens/cycle
```

This estimate assumes `K` draft tokens per cycle, token-level acceptance, and
one target token emitted per cycle. Adaptive lengths, tree drafts, resampling,
or different accounting violate the estimate. Even when valid, compare it only
at matched `K`, target file, draft method, and timing boundary. Target size
changes how well draft work is amortized; complete-wall timing includes prefill
and request overhead that decode-only timing excludes.

## 2. Evidence inventory

| ID | Evidence | What it establishes | Main limitation |
| --- | --- | --- | --- |
| [S1] | [Strix Halo community route guide at `029320fb`][S1] | Ollama 20.42 tok/s report; eight-variable comparison warning; status of unimported community packages | Aggregates first-party and community reports; not raw evidence for every external row |
| [S2] | [MikeVeerman benchmark at `cc527064`][S2] | Q6/Q8 Vulkan AR, MTP, prompt-class, context, and c1-c4 concurrency tables | Published on another host; no local reproduction |
| [S3] | [julianmb ROCmFP4 report at `5d097740`][S3] | Stock `Q4_K_M` control; FP4/FP8/Q3 AR; task-specific MTP results | Custom model format and fork were not run locally |
| [S4] | [KyaniteLabs report at `7fa3ca81`][S4] | Count-to-30 and prose regimes; MTP+ngram configuration and controls | Different quant and prompt regime; headline warm result is explicitly repetition-assisted |
| [S5] | [LaurentZuijdwijk fork at `c28d538df`][S5] | Adaptive DFlash2 claim; fixed/adaptive and structured/prose controls; fork implementation | Published FP4 target and sidecar were not run locally |
| [L1] | [hipEngine row-scaling diagnostic][L1] | Direct packed AR rates and exact local byte budget | Dirty-tree diagnostic; synthetic prompt; no correctness gate |
| [L2] | [hipEngine/llama.cpp same-host diagnostic][L2] | Closest same-GGUF, same-suite C1 comparison | Reused hipEngine runs, F16/BF16 KV mismatch, one run per arm |
| [L3] | [External-configuration transfer test][L3] | Mainline/fork fixed and adaptive MTP plus MTP+ngram on our GGUF and suite | One run; MTP rather than DFlash2; fork/mainline versions differ |

External pages were fetched at the pinned commits on 2026-08-28. The local
artifacts retain commands, model hashes, protocol details, and limitations.

## 3. Autoregressive decode

### 3.1 Published and local rows

| Evidence | Model / quant | File size | Backend | AR tok/s | Derived GB/s | Assessment |
| --- | --- | ---: | --- | ---: | ---: | --- |
| **Local [L1]** | Qwen3.8-27B `Q4_K_M` | 17.096 GB | hipEngine HIP, direct | **12.332** | **210.8** | Diagnostic local anchor |
| Published [S3] | stock `Q4_K_M` | 15.92 GiB, about 17.1 GB | stock llama.cpp; source's baseline | 12.27 | about 209.8 | Nearest published same-quant row; cross-host |
| Published [S3] | `ROCmFP4_FAST`, 4.26 bpw | 13.55 GiB | Vulkan fork | 14.02 | 204.0 | Custom format and engine |
| Published [S3] | `Q3_K_M`, 3.95 bpw | 12.56 GiB | Vulkan fork | 15.15 | 204.3 | Different quant |
| Published [S3] | `Q3_K_S`, 3.59 bpw | 11.40 GiB | Vulkan fork | 16.69 | 204.3 | Different quant |
| Published [S3] | `ROCmFP8`, 8.25 bpw | 26.25 GiB | Vulkan fork | 7.66 | 215.9 | Custom format and engine |
| Published [S2] | Unsloth `UD-Q6_K_XL` | 25.9 GB | llama.cpp Vulkan | 8.43 | 218.3 | Q6 mean rounds to 8.43 |
| Published [S2] | Unsloth `UD-Q8_K_XL` | 31.5 GB | llama.cpp Vulkan | 7.23 | 227.7 | Q8 mean |
| Published [S5] | FP4 target | 13.55 GiB class | fork Vulkan | 14.0-14.1 | about 204 | Structured/prose bare controls |
| **Local [L2]** | Qwen3.8-27B `Q4_K_M` | 17.096 GB | hipEngine HIP, served | **9.807** | **167.7** | Complete-wall, 24-token requests; not decode-only |
| Published [S1] | Ollama `qwen3.8:27b` `Q4_K_M` | 17.7 GB | Vulkan-RADV | 20.42 “generation” | **361.4** | Cannot be classified as dense AR from the published data |

The 204-228 GB/s cluster supports a memory-bound interpretation of the
published decode-only rows. It does **not** isolate a Vulkan-versus-HIP backend
delta because model artifacts, quants, engines, and hosts change together.
The sources report Vulkan advantages in some configurations, but this survey
contains no controlled same-file Vulkan/HIP A/B.

The Ollama row is excluded from AR comparisons. At the published 17.7 GB model
size, 20.42 tok/s implies 361.4 GB/s, or 141% of the stated 256 GB/s peak.
The route guide reports the number as generation throughput but does not show
whether NextN speculation contributed. Therefore the supported conclusion is
**“not a demonstrated AR row,”** not **“proven MTP-on.”**

### 3.2 Local row scaling

The direct packed graph diagnostic used p128/d8 and three reported samples per
configuration ([L1]):

| Rows | Aggregate tok/s | Step ms | Estimated weight sweeps/step |
| ---: | ---: | ---: | ---: |
| 1 | 12.332 | 81.09 | 1.000 |
| 2 | 23.708 | 84.36 | 1.040 |
| 4 | 43.828 | 91.26 | 1.126 |
| 8 | 46.503 | 172.03 | 2.122 |

Rows 1-4 amortize weight reads well; rows 4-8 add 88.5% step wall for twice the
rows. This result describes hipEngine's direct packed graph on the recorded
working tree. It is not an HTTP-serving result, a speculative-frontier trace,
or a retained performance claim.

## 4. Closest same-host llama.cpp comparison

Both engines used the same GGUF, the 10-prompt
`benchmarks/prompts/mtpbench-code-general-ja.jsonl` suite, 24 output tokens,
greedy sampling, no prompt cache, concurrency 1, and a fresh server per arm.
llama.cpp was HIP build 10438 at `9d57ce456`; its MTP context used the GGUF's
built-in NextN block. Full commands and inputs are in [L2].

| Metric | llama.cpp HIP | hipEngine | hipEngine delta |
| --- | ---: | ---: | ---: |
| AR decode tok/s | 12.156 | 12.332, direct [L1] | +1.4% |
| AR complete-wall tok/s | 9.750 | 9.807, served | +0.6% |
| MTP decode tok/s | 24.897 | 21.158, direct-leaf | -15.0% |
| MTP complete-wall tok/s | 15.730 | 15.609, served | -0.8% |
| MTP/AR, complete wall | 1.613x | 1.592x | -1.3% |
| Draft acceptance, depth 3 | **90.16%** | **78.57%** | **-11.59 points** |
| Derived cycle efficiency, complete wall | 43.5% | 47.4% | +3.9 points |
| Derived cycle efficiency, decode basis | 55.1% | 53.9% | -1.2 points |

This is the best local comparison in the survey, but “same-host” must not be
read as “fully controlled.” The hipEngine values were reused from earlier
retained artifacts rather than rerun in the llama.cpp measurement packet;
llama.cpp used F16 KV while hipEngine used BF16; the direct and complete-wall
rows have different timing boundaries; and there was one run per arm.

Within those limits, the complete-wall result does **not** support a large C1
engine-efficiency deficit. The measured difference is draft acceptance:
llama.cpp accepted 165 of 183 drafts and needed about 61 cycles to emit 240
tokens; hipEngine accepted 165 of 210 and needed about 70 cycles. Acceptance
repair is therefore a better-supported C1 investigation than a verifier
cycle-cost campaign based only on unmatched published rows.

## 5. Prompt dependence in published speculative results

### 5.1 MikeVeerman: Q8, maximum draft depth 3

The source used 256-token generations and reports these per-prompt values
([S2]). Cycle efficiency is our derived estimate.

| Prompt | AR tok/s | MTP tok/s | Acceptance | Speedup | Derived cycle efficiency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code | 7.24 | 16.46 | 74.3% | 2.273x | 70.4% |
| JSON extraction | 7.23 | 16.18 | 73.2% | 2.238x | 70.0% |
| Reasoning | 7.23 | 15.83 | 71.2% | 2.189x | 69.8% |
| Prose | 7.23 | 12.72 | 50.8% | 1.759x | 69.7% |
| **Spread** | | | **23.5 points** | **29%** | **0.7 points** |

### 5.2 julianmb: `ROCmFP4_FAST`, maximum draft depth 4

The source reports greedy task-specific values ([S3]). Cycle efficiency is our
derived estimate.

| Prompt | AR tok/s | MTP tok/s | Acceptance | Speedup | Derived cycle efficiency |
| --- | ---: | ---: | ---: | ---: | ---: |
| JSON extraction | 14.02 | 35.79 | 88.0% | 2.553x | 56.5% |
| Code generation | 14.02 | 34.82 | 82.6% | 2.484x | 57.7% |
| Technical explanation | 14.02 | 32.40 | 76.2% | 2.311x | 57.1% |
| Reasoning/math | 14.02 | 30.56 | 71.4% | 2.180x | 56.5% |
| **Spread** | | | **16.6 points** | **17%** | **1.2 points** |

These datasets support a bounded claim: **within each source's four prompts,
the derived cycle-efficiency estimate is much more stable than acceptance or
raw speedup.** They do not establish that acceptance belongs only to content
or that cycle efficiency belongs only to the engine. Acceptance also depends
on drafter, target quantization, sampling, depth, and state. Cycle efficiency
also depends on target size, draft method, depth, and timing boundary.

The practical rule is to publish prompt class and acceptance with every
speculative rate. A single aggregate can still be useful for a fixed suite,
but it cannot be transferred to a different prompt mix without the
per-category distribution.

## 6. Published speculative claims and local verdicts

| Claim | Published evidence | Local test | Verdict |
| --- | --- | --- | --- |
| MikeVeerman Q6/Q8 MTP gives 2.1-2.2x at C1 and falls below AR at C4 | [S2], full prompt and c1-c4 tables | No matching Q6/Q8 Vulkan run | **Published, not locally tested** |
| julianmb `ROCmFP4_FAST` reaches 30.56-36.04 tok/s depending on task/config | [S3], custom quant and fork | Custom model/engine not run | **Published, not locally testable on our GGUF** |
| Laurent adaptive FP4+DFlash2 reaches 65.6 tok/s structured, 26.1 prose, and 4.7x bare on structured output | [S5], 300-token table | Adaptive policy tested with built-in MTP on `Q4_K_M`, not DFlash2/FP4 | **Not reproduced; not refuted** |
| Adaptive depth prevents the acceptance collapse of a fixed deep draft | [S5] | Fixed depth 7: 38.58%; adaptive max 7: 61.70%, +18.8% throughput [L3] | **Mechanism reproduced in local MTP transfer test** |
| Adaptive max 7 is better than a shallow fixed draft on our MTP workload | No external claim for our workload | Adaptive 17.66 vs fixed depth 3 at 18.93 tok/s [L3] | **Refuted for our local configuration (-6.7%)** |
| KyaniteLabs warm 148-163 tok/s is general generation throughput | [S4] explicitly calls it a repetition artifact and reports real traffic at 11-24 tok/s | Same flag stack is only 1.07x AR on `general_en` [L3] | **Refuted by the source's own scope and local transfer test** |
| KyaniteLabs MTP+ngram stack helps repetition-heavy output | [S4] count-to-30 and prose controls | Best local mainline aggregate; 25.0 decode tok/s on `mixed_ja_en`, but 23.87% acceptance [L3] | **Prompt-sensitive support, not a general win** |
| Fork build 10681 reduces local MTP cycle cost versus mainline build 10438 | No clean published A/B for our GGUF | Fixed depth 3: 18.93 vs 16.02 tok/s with similar AR, acceptance, and draft counts [L3] | **Measured about +18%; cause unresolved** |

### 6.1 Local transfer-test details

The local transfer test ran every arm against our `Q4_K_M`, the same mixed
10-prompt suite, 128 output tokens, greedy sampling, no prompt cache, and a
fresh server per arm ([L3]). A 128-token output was chosen so depth 7 had time
to amortize; these results must not be compared directly with the 24-token
complete-wall rows in §4.

| Build / arm | Aggregate tok/s | vs own AR | Acceptance | Drafted | Accepted |
| --- | ---: | ---: | ---: | ---: | ---: |
| mainline AR, b10438 | 11.37 | 1.000x | — | 0 | 0 |
| mainline fixed depth 3 | 16.02 | 1.409x | 63.41% | 1,301 | 825 |
| mainline fixed depth 7 | 14.83 | 1.304x | 38.43% | 2,376 | 913 |
| mainline max 7 / min 3 | 14.79 | 1.301x | 38.29% | 2,382 | 912 |
| mainline MTP+ngram | 16.26 | 1.430x | 23.87% | 3,917 | 935 |
| fork AR, b10681 | 11.35 | 1.000x | — | 0 | 0 |
| **fork fixed depth 3** | **18.93** | **1.668x** | 63.86% | 1,295 | 827 |
| fork fixed depth 7 | 14.87 | 1.310x | 38.58% | 2,369 | 914 |
| fork adaptive max 7 | 17.66 | 1.556x | 61.70% | 1,410 | 870 |
| fork adaptive max 12 | 17.62 | 1.552x | 60.16% | 1,461 | 879 |

Mainline `--spec-draft-n-min 3` did not behave like the fork's adaptive
controller: with max 7 it was statistically indistinguishable from fixed
7 in this one-run test. The fork fixed-depth-3 arm was about 18% faster than
mainline fixed depth 3 while AR, acceptance, and draft counts were similar.
However, build 10438 and build 10681 differ by upstream changes as well as the
fork's `common/speculative.cpp` edits. The result localizes the improvement to
the speculative path; it does not attribute a responsible commit. A bisect is
required before transferring an implementation idea to hipEngine.

Per-category decode rates show why the aggregate needs its suite attached:

| Arm | code | general_en | general_ja | mixed_ja_en |
| --- | ---: | ---: | ---: | ---: |
| fork fixed depth 3 | 22.0 | 18.0 | 18.8 | 23.2 |
| fork adaptive max 7 | 20.7 | 16.8 | 17.0 | 21.8 |
| mainline fixed depth 3 | 19.2 | 13.4 | 18.4 | 20.2 |
| mainline MTP+ngram | 20.1 | 13.1 | 14.6 | **25.0** |

The MTP+ngram arm's best category was `mixed_ja_en`; its `general_en` result
was only 1.07x that category's 12.3 tok/s AR control. This is consistent with
KyaniteLabs' own distinction between repetition-assisted and prose rates.

## 7. Concurrency and non-transferable comparison shapes

MikeVeerman's published Q8 Vulkan sweep uses 256-token outputs ([S2]):

| Concurrent requests | AR aggregate tok/s | MTP aggregate tok/s | MTP/AR |
| ---: | ---: | ---: | ---: |
| 1 | 7.10 | 15.53 | 2.19x |
| 2 | 13.01 | 16.63 | 1.28x |
| 3 | 17.52 | 18.15 | 1.04x |
| 4 | 21.75 | 16.94 | 0.78x |

The source also reports acceptance remaining between 67.5% and 76.8%, so its
explanation is saturation rather than an acceptance collapse. This is a useful
shape target: speculative decoding stops helping as AR batching consumes the
available parallelism. The absolute rates do not transfer to hipEngine because
the target file, quant, backend, output length, and serving stack differ.

hipEngine's direct packed AR diagnostic scales 12.332 to 43.828 tok/s from one
to four rows (3.55x, [L1]). MikeVeerman's published AR aggregate scales 7.10
to 21.75 tok/s (3.06x, [S2]). These are not a scheduler ranking: one is a
direct synthetic row-scaling diagnostic and the other is a server concurrency
sweep.

A previously cited CIRU vLLM Ling-3.0-Flash result is excluded from the
Qwen3.8 comparison set. It used a 124B/5.5B-active mixture-of-experts model and
a different runtime. At most, it can motivate a qualitative concurrency-shape
investigation; its absolute rates and model economics do not transfer.

## 8. Implications for hipEngine

1. **Do not open an AR decode-kernel campaign from these external rates.** The
   local direct diagnostic is in the published implied-bandwidth band, and the
   nearest same-quant published row differs by less than 1%. The served
   24-token row is lower, but its complete-wall timing includes prefill and
   serving overhead and needs a separate attribution study.
2. **Prioritize C1 draft acceptance over verifier cycle cost.** The closest
   same-host comparison shows near-equal complete-wall throughput and an
   11.59-point acceptance deficit. Candidate investigations include NextN
   priming, cursor synchronization, `p_min` handling, and acceptance by draft
   position. These are hypotheses, not established causes.
3. **Keep adaptive depth rejected as the default for the tested C1 MTP
   configuration.** It improved over a fixed deep draft but lost to fixed
   depth 3. Reconsider it only for a different method or workload, such as
   C>1 frontier economics, with a full-suite gate.
4. **Treat ngram composition as prompt-sensitive.** The local stack can help
   repetition-heavy output, but it did not provide a broad suite win and its
   acceptance was low. The separate local closeout retains it default-off with
   correctness and long-horizon blockers documented in
   [`2026-08-28-gfx1151-qwen38-ngram-mtp-composition-closeout.json`](../benchmarks/results/2026-08-28-gfx1151-qwen38-ngram-mtp-composition-closeout.json).
5. **Bisect the fork/mainline fixed-depth-3 delta before porting anything.** A
   local approximately 18% MTP-path difference is worth investigating, but the current test
   confounds fork edits with upstream movement between builds 10438 and 10681.
6. **For C>1, compare crossover shape under a matched protocol.** External
   concurrency data are useful for hypothesis formation, not as an absolute
   target. A retained comparison requires the same model, quant, suite,
   generation length, timing boundary, and host.

## 9. Open measurements

The highest-value missing evidence is:

1. **Per-category and per-position hipEngine acceptance.** Current local
   artifacts publish aggregate 78.57% acceptance but not the category and
   draft-position distributions needed to distinguish content mix from a
   position-2/3 degradation.
2. **A repeated, same-lifecycle hipEngine/llama.cpp A/B.** Use the same GGUF,
   KV dtype if both engines support it, 10-prompt suite, output length, server
   timing boundary, and at least three interleaved runs per arm. Report both
   decode-only and complete-wall rates without mixing them.
3. **A true FP4+DFlash2 reproduction.** Reproducing Laurent's 65.6 tok/s claim
   requires the published FP4 target, FP4 DFlash2 sidecar, pinned fork,
   structured and prose prompts, 300-token generation, power profile, and
   context depth. The local MTP transfer test is not a substitute.
4. **A controlled Vulkan/HIP backend A/B.** The present AR table cannot
   attribute its spread to backend because model formats, quants, engines, and
   hosts differ.
5. **A bisect of llama.cpp build 10438 to the fork's build 10681.** Hold the
   local fixed-depth-3 arm constant and separate upstream changes from the
   fork's speculative edits.

## 10. Limitations

- Public reports were checked against their commit-pinned documentation, but
  their raw logs were not independently audited unless the source table itself
  exposed them. “Published” means the source states the result, not that we
  certify its harness.
- External implied-bandwidth values use model file size rather than exact
  per-token streamed bytes and therefore run high. They are screening values,
  not profiler measurements.
- The published sources use different power limits, LPDDR5X speeds, operating
  systems, Mesa/ROCm versions, model files, KV types, context sizes, and prompt
  suites. Same APU name does not make them a same-host lane.
- The local row-scaling result [L1] was collected on a dirty working tree, uses
  a fixed synthetic fixture, and has no joined correctness or profiler gate.
- The local llama.cpp comparison [L2] reuses earlier hipEngine artifacts, has
  one run per arm, and compares F16 KV with BF16 KV.
- The local external-configuration test [L3] has one run per arm. Its adaptive
  rows use MTP, not the DFlash2 sidecar behind the published 65.6 tok/s claim.
  Its mainline/fork delta also includes version drift.
- Estimated cycle efficiency is a model, not a directly reported counter. It
  is unsuitable for adaptive draft lengths and should not be compared across
  target sizes or timing boundaries.
- Quality is outside this speed survey except where a linked local artifact
  states its own correctness contract. No published speed row here implies
  equal model quality across quantization formats.

## 11. Sources

### External, commit-pinned

- **[S1]** hogeheer499-commits, *Qwen3.8 27B on AMD Strix Halo: What Works,
  What Is Fast, and What Is Actually Verified*, commit `029320fb`, accessed
  2026-08-28: [pinned document][S1].
- **[S2]** MikeVeerman, *Qwen3.8-27B on AMD Strix Halo: what MTP speculative
  decoding gives you*, commit `cc527064`, accessed 2026-08-28:
  [pinned README][S2].
- **[S3]** julianmb, *Qwen 3.8 27B ROCmFP4_FAST on AMD Strix Halo*, commit
  `5d097740`, accessed 2026-08-28: [pinned README][S3].
- **[S4]** KyaniteLabs, *Qwen3.8-27B on Strix Halo — tuned serving profile*,
  commit `7fa3ca81`, accessed 2026-08-28: [pinned README][S4].
- **[S5]** LaurentZuijdwijk, *llama.cpp — Adaptive Speculation + Fastest
  Vulkan on AMD Strix Halo*, commit `c28d538df`, accessed 2026-08-28:
  [pinned README][S5] and [adaptive implementation][S5-code].

### Local artifacts

- **[L1]** [`2026-08-28-gfx1151-qwen38-row-scaling-baseline.json`][L1]
- **[L2]** [`2026-08-28-gfx1151-qwen38-llamacpp-1to1.json`][L2]
- **[L3]** [`2026-08-28-gfx1151-qwen38-fork-claim-generalization.json`][L3]

[S1]: https://github.com/hogeheer499-commits/strix-halo-guide/blob/029320fb/QWEN38_STRIX_HALO.md
[S2]: https://github.com/MikeVeerman/qwen38-27-Strix-Halo-bench/blob/cc527064/README.md
[S3]: https://github.com/julianmb/q38rocm/blob/5d097740/README.md
[S4]: https://github.com/KyaniteLabs/qwen38-27b-strix-halo/blob/7fa3ca81/README.md
[S5]: https://github.com/LaurentZuijdwijk/llama.cpp/blob/c28d538df/README.md
[S5-code]: https://github.com/LaurentZuijdwijk/llama.cpp/blob/c28d538df/common/speculative.cpp
[L1]: ../benchmarks/results/2026-08-28-gfx1151-qwen38-row-scaling-baseline.json
[L2]: ../benchmarks/results/2026-08-28-gfx1151-qwen38-llamacpp-1to1.json
[L3]: ../benchmarks/results/2026-08-28-gfx1151-qwen38-fork-claim-generalization.json
