# Qwen3.8-27B on Strix Halo: external implementation survey

Updated: 2026-09-03

Many public Qwen3.8-27B implementations report impressive performance on
Strix Halo, but the numbers use different models, quantizations, prompts, and
timing boundaries ([overview][S1]).

We independently reproduced the major claims on one Ryzen AI MAX+ 395 system.
We checked outputs as well as speed and used a shared multilingual prompt suite
when the implementation supported the same model and protocol.

For direct engine comparisons, we use the same standard `Q4_K_M` file with
stock llama.cpp and the hipEngine build measured here. Routes that require
another model format are reported separately as source reproductions.

`C` means active requests. `K` means draft tokens per speculative cycle.

## Executive summary

### Standard `Q4_K_M` leaders

- **Autoregressive generation**
  - hipEngine leads at C1 and C3-C8.
  - Laurent Vulkan leads at C2.
- **Prefill-dominant throughput**
  - hipEngine leads at C1 and C3-C7.
  - Laurent Vulkan leads at C2 and C8.
- **MTP-enabled throughput**
  - Laurent Vulkan leads at C1-C2.
  - hipEngine's MTP-enabled route leads at C3-C7.
  - Stock llama.cpp HIP leads at C8.

The full measurements are in the [standard comparison](#standard-q4_k_m-comparison).

### hipEngine at a glance

- Active C1 reaches **19.428 tok/s**, or **1.687x** its matched
  autoregressive rate.
- The server had three request slots, with one request active.
  - K3 generated the same tokens as matched AR on all ten prompts.
- At C3, K3 reaches **32.919 tok/s**, or **1.342x** matched autoregressive
  generation.

When MTP is enabled, hipEngine uses K3 through C4. At C5-C8, it uses one
full-batch autoregressive pass instead.

## Summary table

| Usable | Implementation | Main result | Constraint or routing |
| --- | --- | --- | --- |
| Yes | hipEngine `Q4_K_M` | Leads most standard comparison cells | Automatic K3 for eligible C1; AR at C2-C8 |
| Yes | Laurent built-in K3 | Leads MTP at C1-C2 | Standard GGUF route |
| C1-only | `q38rocm` strict K4 | 38.85 decode tok/s | Custom model and one server slot |
| No | Laurent adaptive DFlash2 | 56.532 decode tok/s in a fresh process | Request state leaks when the server is reused |
| Yes | Kyanite MTP+ngram | 167.64 tok/s on warm replay | Large ngram gain requires earlier output to replay |
| Yes | PieBru recipes and Nathanw fork | Q5, Q6, and Q8 claims reproduced | llama.cpp mainline is slightly faster in DFlash decode |
| Yes | MikeVeerman Q8 MTP | 2.23x AR at C1 | 0.84x AR at C4 |

## Test method

### Standard comparison

Every compatible route used:

- the same 17,106,775,008-byte Qwen3.8-27B `Q4_K_M` file;
- ten shared prompts across four language/task categories;
- greedy sampling and no prompt cache;
- 24 generated tokens per request;
- the same physical host;
- complete request-wall timing.

The model hash and exact commands are in the [evidence artifacts](#evidence).

The engines used their deployable default key-value cache precision:

- hipEngine: BF16;
- llama.cpp routes: F16.

The comparison therefore holds the model file and workload constant. It does
not force identical internal arithmetic.

### Host and prompt suite

| Item | Value |
| --- | --- |
| System | AMD Ryzen AI MAX+ 395 |
| GPU | Radeon 8060S (`gfx1151`) |
| Unified memory | 128 GB |
| Theoretical bandwidth | 256 GB/s |
| Kernel | Linux 7.1.6-1-cachyos |
| Prompt suite | `benchmarks/prompts/mtpbench-code-general-ja.jsonl` |
| Prompt count | 10 |
| Categories | Code, English, Japanese, mixed Japanese/English |
| Heldout prompts | 4 of 10 |
| Sampling | Greedy; prompt cache disabled |

### Timing terms

- **Decode tok/s** uses the server-reported decode interval.
- **Arithmetic decode tok/s** is the mean of per-request decode rates.
- **Token-weighted decode tok/s** divides all output tokens by all decode time.
- **Complete-wall tok/s** divides output tokens by elapsed request time.
- **Prefill-dominant tok/s** divides prompt tokens by complete wall for a
  one-output-token request.

Compare rates only when model, workload, output length, and timing boundary
match.

### Correctness checks

We used the strongest correctness checks available for each route:

- exact expected structures for JSON and count-to-30 output;
- repetition checks based on character windows and word trigrams;
- fresh-server controls when request leakage was suspected;
- token equality for matched engine variants;
- category and heldout coverage on the common suite.

These checks establish generation validity. They do not prove equal model
quality across quantizations.

## Standard `Q4_K_M` comparison

The external rows were measured on 2026-08-30 ([L6]). The hipEngine rows were
measured on 2026-09-03 ([L14]-[L17]).

Values are aggregate complete-wall tok/s; higher is better. The best value in
each column is bold.

### Leadership by workload

| Workload | C1 | C2 | C3-C6 | C7 | C8 |
| --- | --- | --- | --- | --- | --- |
| Prefill | hipEngine | Laurent | hipEngine | hipEngine | Laurent |
| Autoregressive | hipEngine | Laurent | hipEngine | hipEngine | hipEngine |
| MTP-enabled | Laurent | Laurent | hipEngine | hipEngine | Stock HIP |

### Prefill-dominant throughput

| Engine | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llama.cpp mainline Vulkan | 111.2 | 133.6 | 127.0 | 127.0 | 137.6 | 146.8 | 148.1 | 162.3 |
| llama.cpp stock HIP | 146.2 | 186.9 | 190.3 | 200.5 | 226.1 | 250.1 | 250.3 | 283.5 |
| Laurent Vulkan | 149.1 | **211.9** | 192.4 | 191.9 | 222.1 | 250.4 | 252.3 | **305.8** |
| Nathan Vulkan | 153.4 | 200.8 | 186.1 | 186.3 | 208.2 | 227.5 | 228.6 | 263.9 |
| `q38rocm` Vulkan | 146.9 | 186.5 | 181.3 | 184.2 | 206.1 | 229.7 | 233.0 | 273.5 |
| hipEngine | **201.0** | 181.9 | **207.2** | **233.1** | **258.2** | **285.1** | **291.3** | 301.8 |

Key points:

- hipEngine leads six widths.
- C2 trails Laurent by 14.2%.
- C8 trails Laurent by 1.3%.

### Autoregressive complete-wall throughput

| Engine | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llama.cpp mainline Vulkan | 10.226 | 17.471 | 14.905 | 13.246 | 18.086 | 24.083 | 29.342 | 35.376 |
| llama.cpp stock HIP | 10.635 | 17.681 | 15.266 | 13.537 | 18.271 | 23.296 | 26.544 | 30.325 |
| Laurent Vulkan | 11.047 | **19.835** | 16.359 | 14.255 | 20.294 | 28.322 | 35.896 | 45.614 |
| Nathan Vulkan | 11.162 | 19.662 | 16.341 | 14.321 | 20.099 | 27.575 | 34.317 | 41.511 |
| `q38rocm` Vulkan | 10.582 | 18.645 | 15.663 | 13.866 | 19.054 | 24.970 | 29.576 | 35.021 |
| hipEngine | **11.518** | 19.249 | **24.526** | **31.478** | **37.995** | **43.093** | **46.153** | **50.605** |

Key points:

- hipEngine leads seven widths.
- C2 is the only hipEngine deficit.
- C2 trails Laurent by 3.0%.

### MTP-enabled throughput

The first hipEngine row shows routing after MTP is enabled. It uses K3 through
C4 and one full-batch AR pass at C5-C8.

The second hipEngine row forces K3 at every width. The external rows also use
fixed K3.

| Engine or route | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hipEngine after MTP opt-in | 19.428 | 30.094 | **32.919** | **37.985** | **37.995 AR** | **43.093 AR** | **46.153 AR** | 50.605 AR |
| hipEngine forced K3 | 19.428 | 30.094 | **32.919** | **37.985** | 37.280 | 41.048 | 44.492 | 50.893 |
| llama.cpp mainline Vulkan K3 | 21.022 | 30.840 | 27.283 | 26.955 | 32.713 | 32.307 | 38.282 | 45.458 |
| llama.cpp stock HIP K3 | 17.351 | 23.530 | 23.087 | 25.287 | 22.293 | 25.212 | 46.084 | **56.222** |
| Laurent Vulkan K3 | **21.126** | **32.221** | 28.067 | 26.184 | 31.737 | 37.154 | 43.888 | 50.837 |
| Nathan Vulkan K3 | 20.781 | 30.566 | 27.859 | 26.385 | 29.768 | 33.318 | 36.992 | 45.173 |
| `q38rocm` Vulkan K3 | 20.357 | 27.163 | 26.178 | 26.482 | 32.297 | 31.613 | 38.314 | 45.342 |

#### hipEngine routing

- C1 verifies the draft as a single request, even when the server reserves
  capacity for more requests.
  - Throughput: **19.428 tok/s**, or **1.687x** matched AR.
  - Acceptance: 78.89%.
  - Generated tokens matched AR on all ten prompts ([L17]).
- C2-C4 use K3 and beat matched AR.
- C5-C8 use the full-batch AR fallback after MTP opt-in.
- Forced K3 is slower than AR at C5-C7.
- Forced K3 and AR are effectively tied at C8, at 1.0057x AR.

The C3 quality test compared K3 output with hipEngine's strict reference path:

- the highest-probability token matched in all 240 standard cases;
- the highest-probability token matched in all 192 heldout cases;
- maximum Kullback-Leibler divergence was 8.69e-4 and 8.45e-4,
  respectively ([L8]).

### Public automatic admission

Automatic serving uses MTP more narrowly than the table above:

- One active request uses strict K3 for context lengths 1-67.
  - K3 reaches **18.191 tok/s**, versus **11.062 tok/s** for AR ([L7]).
- C2-C8 use K0, which means autoregressive generation.
- Unsupported context lengths, draft depths, or sampling modes also use K0.

## Source-specific reproductions

These sections use each project's own model and protocol. Compare a local
result only with the published result in the same section.

## `q38rocm` / ROCmFPX

**Verdict: usable as a specialized C1 route.**

### Tested configuration

- `q38rocm` v1.5.2 and the pinned source report ([S3]);
- Vulkan/RADV on Radeon 8060S;
- `ROCmFP4_FAST` target;
- strict MTP K4;
- one server slot.

The repository installer contained a stale checksum. We verified the prebuilt
binary against the GitHub release digest.

### Published-protocol result

| Metric | Published | Local |
| --- | ---: | ---: |
| AR decode | 14.02 | 14.31 |
| MTP decode | 30.56-36.04 | **38.85 mean** |
| MTP acceptance | — | 78.1% |
| Repetition guard | Clean | Passed |

#### Prompt breakdown

| Prompt | Decode tok/s | Acceptance |
| --- | ---: | ---: |
| Binary search tree | 41.44 | 88.6% |
| Widget factory | 38.73 | 75.7% |
| JSON extraction | 48.49 | 100.0% |
| Unified versus discrete memory | 26.75 | 48.0% |

### Common-suite result

| Mode | Arithmetic decode | Token-weighted decode | Complete wall | Acceptance |
| --- | ---: | ---: | ---: | ---: |
| AR | 14.782 | 14.782 | 12.803 | — |
| Strict MTP K4 | **35.575** | **32.969** | **24.294** | 62.35% |

#### Category breakdown

| Category | MTP decode tok/s |
| --- | ---: |
| Code | 44.79 |
| General English | 28.41 |
| General Japanese | 26.47 |
| Mixed Japanese/English | 33.42 |

Strict MTP refuses more than one server slot:

```text
Qwen strict MTP requires a single server slot/sequence
```

The result is therefore C1-only. It is not evidence for multi-user throughput.

Normal K3 with standard `Q4_K_M` supports C1-C8. It did not lead any cell in
the standard comparison.

The strict common-suite harness recorded rates and pass/fail checks, but it did
not save response text. No failure or contamination appeared in the C1 run.

## Laurent adaptive DFlash2

**Verdict: not usable for sequential serving.**

The route is fast in a fresh process. Reused-server requests can inherit stale
speculative state.

### Tested configuration

- pinned Laurent fork and implementation ([S5]);
- Vulkan/RADV;
- `ROCmFP4_FAST` target and FP4 DFlash2 draft model;
- adaptive draft depth K3-K7;
- published prose-then-JSON request sequence.

### Sequential request result

| Policy | Prose decode | JSON decode | JSON result |
| --- | ---: | ---: | --- |
| AR | 14.148 | 14.128 | On-task; truncated |
| Fixed K3 | 25.842 | 42.532 | On-task |
| Fixed K7 | 24.481 | 20.859 | Wrong-task prose |
| Adaptive K3-K7 | 25.618 | **66.838** | **Invalid stale prose** |

The 66.838 tok/s row reproduces the published speed. It is invalid because the
JSON request repeated content from the preceding prose request.

The fork's own validity check measured output length. It did not check whether
the response answered the current request.

### Fresh-server result

Restarting the server before each JSON request removed the stale prose.

| Test | Decode tok/s | Result |
| --- | ---: | --- |
| 300 tokens, trial 1 | 56.948 | Clean JSON; truncated at object 9 |
| 300 tokens, trial 2 | 56.699 | Same output hash |
| 300 tokens, trial 3 | 56.991 | Same output hash |
| 420 tokens, AR | 14.180 | Complete 12-object JSON |
| 420 tokens, adaptive | **56.532** | **Complete 12-object JSON** |

Use **56.532 tok/s** as the valid structured-output result. It is 3.99x the
matched AR route.

### Fresh-process common suite

Each prompt used a new server process.

| Metric | Result |
| --- | ---: |
| Arithmetic decode | 37.752 tok/s |
| Token-weighted decode | 34.483 tok/s |
| Acceptance | 60.43% |
| Code | 51.81 tok/s |
| General English | 28.42 tok/s |
| General Japanese | 25.44 tok/s |
| Mixed Japanese/English | 31.27 tok/s |

All ten outputs were substantive and non-repetitive.

### Required fix

The adaptive DFlash2 path must reset or repair speculative state at every
request boundary. It also needs a sequential multi-prompt correctness test.

## KyaniteLabs MTP+ngram

**Verdict: usable with a workload caveat.**

The output is correct. The 160+ tok/s peak measures warm replay rather than
novel generation.

### Tested configuration

- pinned Kyanite configuration ([S4]);
- llama.cpp HIP;
- Unsloth `UD-Q4_K_XL`;
- one slot and 98,304-token context;
- MTP K12 and a minimum ngram replay match of 24 tokens.

### Count-to-30 result

Cold is the first request. Warm repeats the same count sequence.

| Mode | Cold decode | Warm decode | Output |
| --- | ---: | ---: | --- |
| AR | 11.94 | 11.97 | Exact `1…30` |
| MTP K12 | 61.09 | 59.42-59.49 | Exact `1…30` |
| MTP K12 + ngram | 60.95 | **164.13-167.64** | Exact `1…30` |

MTP provides the cold speedup. Ngram adds no cold benefit.

### Common-suite result

| Mode | Arithmetic decode | Complete wall |
| --- | ---: | ---: |
| AR | 11.964 | 11.679 |
| MTP K12 | 24.390 | 20.450 |
| MTP K12 + ngram | 24.867 | 20.518 |

#### Category breakdown

| Category | Decode tok/s |
| --- | ---: |
| Code | 35.82 |
| General English | 16.10 |
| General Japanese | 15.45 |
| Mixed Japanese/English | 21.15 |

Ngram improved arithmetic decode by 1.96%. Complete-wall speed improved by
0.33%, which is too small to claim from one run.

All diverse-suite outputs were substantive and non-repetitive.

### hipEngine implication

hipEngine keeps its ngram replay option off by default ([L5]). A
repetition-heavy C2 test improved 2.425% over MTP alone, but reached only
0.9875x the matched autoregressive throughput.

Kyanite's replay result does not justify enabling ngram globally.

## PieBru recipes and Nathanw fork

**Verdict: usable.**

The Q5, Q6, and Q8 served-speed claims reproduced. The tested mainline build
is slightly faster in DFlash decode.

### Tested configuration

- pinned PieBru recipes ([S6]);
- Nathanw fork and matched mainline;
- Vulkan/RADV;
- Unsloth Q5, Q6, and Q8 XL targets;
- DFlash2 Q8 sidecar;
- ten prompts with up to 128 output tokens.

### End-to-end request throughput

| Quant | Published tok/s | Nathan local | Mainline local |
| --- | ---: | ---: | ---: |
| Q5 | about 23-24 | 24.706 | 24.886 |
| Q6 | 17-21 | 20.549 | 20.343 |
| Q8 | 15-18 | 18.197 | 18.092 |

All three claims are confirmed or conservative.

### Decode result

| Quant | Engine | AR decode | DFlash decode | Acceptance |
| --- | --- | ---: | ---: | ---: |
| Q5 | Nathan | 10.695 | 30.659 | 53.19% |
| Q5 | Mainline | 10.691 | **31.119** | 53.19% |
| Q6 | Nathan | 8.778 | 26.470 | 42.92% |
| Q6 | Mainline | 8.803 | **26.867** | 42.92% |
| Q8 | Nathan | 7.275 | 23.044 | 43.94% |
| Q8 | Mainline | 7.276 | **23.374** | 43.94% |

The matched outputs were token-exact and non-repetitive. Mainline was about
1.4-1.5% faster in DFlash decode.

Nathan sometimes had faster prefill. That explains its small Q6/Q8
complete-wall lead.

Category-level rates are available in the [reproduction artifact][L0].

## MikeVeerman Q8 concurrency

**Verdict: usable with concurrency-aware routing.**

MTP is valuable at low concurrency. It loses to batched autoregressive
generation at C4.

### Tested configuration

- pinned MikeVeerman benchmark ([S2]);
- stock llama.cpp;
- Vulkan/RADV;
- Unsloth `UD-Q8_K_XL`;
- four 32,768-token slots;
- greedy 256-token generations.

The pinned build disabled unsupported cache reuse in both arms.

### Result

| C | Local AR | Local MTP | MTP / AR | Acceptance |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 7.21 | **16.07** | 2.23x | 72.9% |
| 2 | 13.35 | **17.48** | 1.31x | 71.4% |
| 3 | 18.00 | 18.21 | 1.01x | 71.6% |
| 4 | **21.03** | 17.58 | 0.84x | 66.4% |

At C4, per-request throughput was 5.87 tok/s for autoregressive generation.
MTP reached 4.89 tok/s per request.

The loss is caused by saturation, not failed drafting. Acceptance alone is not
enough to decide whether MTP should run.

## Cross-route findings

### Source protocols are not one leaderboard

Specialized routes differ in:

- target format;
- draft method;
- output length;
- timing boundary;
- prompt type;
- server reuse.

Use the standard `Q4_K_M` tables for engine ranking.

On the shared FP4 target, Laurent fresh-process DFlash2 reached 34.483
weighted decode tok/s. `q38rocm` strict MTP K4 reached 32.969.

Laurent adaptive DFlash2 is not deployable because sequential requests are
incorrect.

### Plain autoregressive generation is memory-bound

Model size multiplied by decode rate gives this rough bandwidth estimate:

| Target | Approximate implied bandwidth |
| --- | ---: |
| `ROCmFP4_FAST` | 208 GB/s |
| Q5 | 223 GB/s |
| Q6 | 223 GB/s |
| Q8 | 229 GB/s |

These rates were not measured with hardware counters. The estimate omits some
tensors and metadata.

A claimed dense autoregressive rate above the 256 GB/s physical ceiling needs
another explanation.

### Prompt type changes speculative speed

Code and structured output often accept more draft tokens than explanatory
prose or Japanese heldouts.

Report speculative results with:

- prompt category;
- acceptance;
- output length;
- timing boundary.

### Replay and contamination are different

- Kyanite's 167.64 tok/s output is correct warm replay.
- Laurent's 66.838 tok/s output is incorrect stale state.

The first is narrow but valid. The second is not a valid result.

### Concurrency changes the best policy

At C1, speculative work can use compute left idle during weight reads. At C4,
batched autoregressive generation uses that compute for real requests.

A scheduler must use physical concurrency and the cost of each speculative
cycle. Acceptance alone is not enough.

## What the results mean for hipEngine

- Preserve autoregressive generation.
  - It leads the standard comparison at C1 and C3-C8.
- Improve prefill at C2 and C8.
  - The gaps to Laurent are 14.2% and 1.3%, respectively.
- Improve K3 at C1-C2.
  - Both widths trail the llama.cpp leaders.
- Keep the AR fallback at C5-C7.
  - Forced K3 is slower than matched AR at those widths.
- Keep public C2-C8 admission on AR until each width passes the full
  correctness and serving tests.

## Evidence

### hipEngine measurements

- [Active-C1 measurement][L17]
- [C1-C8 MTP measurements][L15]
- [C1-C8 prefill measurements][L16]
- [Forced-K3 C5-C8 measurements][L14]

### Comparison and reproduction

- [External source reproductions][L0]
- [Same-host six-engine comparison][L6]
- [hipEngine ngram experiment][L5]
- [Strict C1 result][L7]
- [C3 K3 quality test][L8]

The compact artifacts contain commands, model identities, hashes, rates, and
correctness results. Raw server logs are not tracked in Git.

## External sources

- [S1] Strix Halo implementation overview.
- [S2] MikeVeerman Q8 concurrency benchmark.
- [S3] `q38rocm` report.
- [S4] KyaniteLabs MTP+ngram report.
- [S5] Laurent adaptive DFlash2 implementation.
- [S6] PieBru Q5/Q6/Q8 recipes.

Pinned commits are encoded in the links below.

[L0]: ../benchmarks/results/2026-08-28-gfx1151-qwen38-external-reproduction-survey.json
[L5]: ../benchmarks/results/2026-08-28-gfx1151-qwen38-ngram-mtp-composition-closeout.json
[L6]: ../benchmarks/results/2026-08-30-gfx1151-qwen38-final-six-engine-c1c8.json
[L7]: ../benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-e0-current-baseline.json
[L8]: ../benchmarks/results/2026-08-30-gfx1151-qwen38-mtp-e5-combined-correctness.json
[L14]: ../benchmarks/results/2026-09-03-gfx1151-qwen38-b5-planar-q6-integer-mmq-retained.json
[L15]: ../benchmarks/results/2026-09-03-gfx1151-qwen38-current-head-mtp-c1c8-refresh.json
[L16]: ../benchmarks/results/2026-09-03-gfx1151-qwen38-current-head-prefill-c1c8-refresh.json
[L17]: ../benchmarks/results/2026-09-03-gfx1151-qwen38-c1-singleton-target-retained.json
[S1]: https://github.com/hogeheer499-commits/strix-halo-guide/blob/029320fb/QWEN38_STRIX_HALO.md
[S2]: https://github.com/MikeVeerman/qwen38-27-Strix-Halo-bench/blob/cc52706409b0c550636ff068b06894d27079d734/README.md
[S3]: https://github.com/julianmb/q38rocm/blob/5d0977403b0dac778598b1af499bf178b46c0b35/README.md
[S4]: https://github.com/KyaniteLabs/qwen38-27b-strix-halo/blob/7fa3ca810c82c38e7d5a8ef4018d1d1853cec576/README.md
[S5]: https://github.com/LaurentZuijdwijk/llama.cpp/blob/c28d538df5c02643e701a8004db84dbf1bb0ffb2/common/speculative.cpp
[S6]: https://github.com/PieBru/Qwen-3.8-27B_Strix-Halo_gfx1151/tree/66cfceae5edb3dfaf049279738a6fb9cfc5638f6
