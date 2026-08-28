# Qwen3.8-27B on Strix Halo: external implementation survey

Many public Qwen3.8-27B implementations report impressive performance on Strix
Halo, but the reported numbers mix different models, quantizations, draft
methods, prompts, and timing boundaries ([overview][S1]). We independently
reproduced the major claims on one Ryzen AI MAX+ 395 system, checked the
outputs—not only the speed—and ran a shared multilingual prompt suite where the
implementations allowed it. For direct engine comparisons, we use the same
standard `Q4_K_M` file and compare each compatible route with both stock
llama.cpp and the current hipEngine implementation. Routes that require another
model format remain source-claim reproductions, not direct engine rankings.
This report separates reproducible, usable performance from narrow replay
results and invalid state-contaminated output.

## Conclusions

- **No engine wins every standardized workload:** on standard `Q4_K_M`, Nathan
  led prefill and AR at C1; Laurent led AR at C2 and C8, prefill at C2 and C6-C8,
  and MTP at C1-C3; hipEngine led AR at C3-C7; stock HIP led prefill at C3-C5
  and MTP at C6-C8; mainline Vulkan led MTP at C4-C5.
- **Laurent's ordinary built-in MTP path provides broad, usable gains:** it was
  the strongest alternate llama.cpp route overall in the standardized matrix.
  This is separate from Laurent adaptive DFlash2, which remains unsafe across
  sequential requests.
- **Strongest reproduced specialized result:** `q38rocm` strict MTP K4 reached
  **38.85 decode tok/s** under its published protocol and **35.575 arithmetic /
  32.969 token-weighted decode tok/s** on our shared prompt suite. It requires
  the custom `ROCmFP4_FAST` model and exactly one server slot, so it is not part
  of the standard-`Q4_K_M` engine ranking and has no C2-C8 result.
- **Fastest valid single task:** Laurent adaptive DFlash2 reached **56.532 decode
  tok/s** for complete structured JSON in a fresh server process.
- **Laurent adaptive DFlash2 is not usable as a sequential server:** its 66.838
  tok/s JSON row repeated prose from the previous request. That result is
  invalid. Laurent's ordinary built-in MTP path did not show this defect.
- **The highest raw number is workload-specific:** Kyanite reached **167.64
  tok/s** by replaying a warm count-to-30 sequence, not by generating novel
  text.
- **MTP must be routed by concurrency:** Mike's Q8 result was 2.23x AR at C1,
  neutral at C3, and 0.84x AR at C4.
- **hipEngine is usable with the standard `Q4_K_M`:** it passed the
  standardized correctness gates and led AR at C3-C7, but lagged the llama.cpp
  routes in prefill and MTP.

### Route decisions

**Yes** means the locally tested route produced valid output and showed no
server-lifecycle correctness blocker. **No** means the tested route exposed a
correctness blocker. A Yes does not mean that different quantizations have
equal model quality.

| Route | Usable? | Decision |
| --- | :---: | --- |
| hipEngine `a9b801d59` with the standard `Q4_K_M` baseline | **Yes** | Runs C1-C8 and passed the standardized AR/MTP self-exact gates. It led AR at C3-C7. |
| `q38rocm` v1.5.2, `ROCmFP4_FAST`, strict MTP K4 | **Yes, C1 only** | Strong specialized result. Strict mode requires exactly one server slot and a custom model, so it is not ranked against standard-`Q4_K_M` engines. |
| Laurent built-in MTP K3, standard `Q4_K_M` | **Yes** | Strongest broad alternate llama.cpp route in the standardized matrix. |
| Laurent adaptive DFlash2 fork `c28d538df` | **No** | Fast in a fresh process, but unsafe for sequential requests because speculative state leaks between requests. |
| `q38rocm` normal MTP K3, standard `Q4_K_M` | **Yes** | Supports C1-C8, but did not lead any standardized cell. |
| KyaniteLabs HIP MTP+ngram | **Yes** | Correct output. The 160+ tok/s result applies only to warm repetition replay. |
| PieBru recipes on Nathanw fork `0eb528051` | **Yes** | Q5/Q6/Q8 speed claims reproduced. Latest mainline is slightly faster in decode. |
| MikeVeerman stock llama.cpp pin `152d337fa`, Q8 MTP | **Yes** | Use MTP at low concurrency. Disable it for dense parallel work. |

### Source-claim reproduction

This table answers whether we could reproduce each source's result under its
own protocol. The rows use different models, workloads, output lengths, and
timing boundaries. **Do not use this table to rank engines.** The standardized
`mtp-bench` comparison below provides the apples-to-apples ranking.

| Route | Published claim | Local measurement |
| --- | --- | --- |
| `q38rocm` strict MTP K4 | 14.02 AR; 30.56-36.04 MTP decode tok/s | 14.31 AR; **38.85** MTP on the source protocol. Common suite: **35.575 arithmetic / 32.969 token-weighted decode tok/s**. |
| Laurent adaptive DFlash2 | 65.6 structured; 26.1 prose decode tok/s | Valid complete JSON: **56.532**. Prose: 25.618. Fresh-process common suite: **37.752 arithmetic / 34.483 token-weighted decode tok/s**. |
| Kyanite MTP+ngram | 59.7 cold; 148-163 warm count-to-30; 11-24 real traffic | 60.95 cold; **164.13-167.64** warm count-to-30. Common suite: **24.867 decode / 20.518 complete-wall tok/s**. |
| PieBru Q5/Q6/Q8 | About 23-24 / 17-21 / 15-18 served tok/s | **24.706 / 20.549 / 18.197 complete-wall tok/s** on Nathan. |
| MikeVeerman Q8 concurrency | MTP is 2.19x AR at C1 and 0.78x at C4 | **2.23x** at C1, 1.01x at C3, and **0.84x** at C4. |

### Standardized `Q4_K_M` comparison

These tables provide the apples-to-apples engine comparison. Every row uses the
same standard `Q4_K_M` file, ten `mtp-bench` prompts, greedy sampling, disabled
prompt caching, 24 generated tokens per request, and one physical host. Values
are aggregate complete-wall tok/s; higher is better. The comparison uses each
engine's production/default KV precision—BF16 for hipEngine and default F16 for
the llama.cpp routes—so it compares deployable engine configurations rather
than forcing identical internal arithmetic.

The prefill pass generated one token per request and reports prompt tokens divided
by barrier-to-last-completion wall time. It therefore includes one generated
token and API overhead. We use this common end-to-end boundary because llama.cpp
exposes internal prompt timing but hipEngine does not expose an equivalent field.

#### Prefill-dominant throughput

| Engine | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hipEngine `a9b801d59` | 71.5 | 74.9 | 104.5 | 124.4 | 141.4 | 152.2 | 162.1 | 170.0 |
| Mainline Vulkan `4e97ac86` | 87.4 | 113.7 | 114.8 | 117.9 | 129.2 | 138.9 | 141.5 | 155.5 |
| Stock HIP `9d57ce456` | 136.8 | 172.0 | **180.1** | **192.5** | **217.4** | 241.1 | 242.1 | 274.1 |
| Laurent Vulkan `c28d538df` | 135.4 | **194.1** | 182.6 | 185.0 | 215.5 | **243.5** | **245.6** | **296.8** |
| Nathan Vulkan `0eb528051` | **138.9** | 184.7 | 176.8 | 179.4 | 201.6 | 221.2 | 222.8 | 257.2 |
| `q38rocm` normal Vulkan | 136.3 | 175.3 | 177.6 | 184.1 | 207.5 | 234.8 | 235.3 | 269.0 |

#### Autoregressive complete-wall throughput

| Engine | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hipEngine `a9b801d59` | 9.637 | 14.247 | **20.731** | **26.216** | **31.190** | **34.564** | **36.592** | 39.057 |
| Mainline Vulkan `4e97ac86` | 10.266 | 17.323 | 16.696 | 17.491 | 23.099 | 27.534 | 27.763 | 35.513 |
| Stock HIP `9d57ce456` | 10.754 | 17.763 | 17.079 | 17.903 | 22.923 | 25.523 | 25.466 | 30.263 |
| Laurent Vulkan `c28d538df` | 11.156 | **20.056** | 18.821 | 19.693 | 27.223 | 33.204 | 33.416 | **45.751** |
| Nathan Vulkan `0eb528051` | **11.336** | 19.889 | 18.729 | 19.560 | 26.636 | 31.716 | 31.469 | 41.227 |
| `q38rocm` normal Vulkan | 10.640 | 18.667 | 17.784 | 18.744 | 24.685 | 28.188 | 27.964 | 34.933 |

#### Built-in MTP K3 complete-wall throughput

| Engine | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hipEngine `a9b801d59` | 7.073 | 17.401 | 20.065 | 16.513 | 12.966 | 16.851 | 17.662 | 16.069 |
| Mainline Vulkan `4e97ac86` | 21.009 | 30.892 | 27.346 | **27.015** | **32.740** | 32.359 | 38.383 | 45.460 |
| Stock HIP `9d57ce456` | 17.604 | 22.989 | 20.257 | 19.704 | 16.753 | **33.381** | **41.214** | **54.834** |
| Laurent Vulkan `c28d538df` | **21.277** | **32.378** | **27.515** | 25.438 | 30.757 | 36.023 | 42.304 | 49.999 |
| Nathan Vulkan `0eb528051` | 21.046 | 30.204 | 26.906 | 25.279 | 28.632 | 31.947 | 36.868 | 45.319 |
| `q38rocm` normal Vulkan | 20.464 | 27.195 | 26.240 | 26.532 | 32.361 | 31.043 | 38.123 | 45.365 |

All llama.cpp outputs passed the character-window and word-trigram repetition
guards. hipEngine passed its AR and MTP self-exact contracts. hipEngine's MTP
route beat its own AR only at C2; its AR path led the complete matrix at C3-C7.

## 1. Test method

### Comparison framework

Every route is evaluated in two separate tracks where its model support allows
it:

1. **Claim reproduction:** run the source's model, engine, and protocol. This
   track tests whether the source's claim reproduces; it does not rank engines.
2. **Standardized `mtp-bench` comparison:** run the standard Qwen3.8-27B
   `Q4_K_M` artifact on our shared ten-prompt suite. This track ranks compatible
   engines using the same model file, prompts, output length, host, and timing
   boundary, with stock llama.cpp and hipEngine as controls.

The standardized comparison model is
`/models/gguf/Qwen3.8-27B-Q4_K_M.gguf`, SHA-256
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
Custom-format routes such as `ROCmFP4_FAST`, and quant-specific Q5/Q6/Q8
recipes, cannot participate in that 1:1 table unless the implementation also
supports this standard file. Their exact-artifact results remain useful for
verifying the source claim.

### Host and common suite

| Item | Value |
| --- | --- |
| APU | AMD Ryzen AI MAX+ 395 |
| GPU | Radeon 8060S / `gfx1151` |
| Unified memory | 128 GB |
| Theoretical memory bandwidth | 256 GB/s |
| Kernel | Linux 7.1.6-1-cachyos |
| Common suite | `benchmarks/prompts/mtpbench-code-general-ja.jsonl` |
| Suite SHA-256 | `fac920be5e691fec2cb70fd8b7eedddab8926b89d6a1627f62ec4f441d86084a` |
| Prompt coverage | 10 prompts: code, general English, general Japanese, and mixed Japanese/English; four heldouts |
| Common sampling | Greedy; prompt cache disabled |

The [compact artifact][L0] records model sizes and SHA-256 hashes, source
commits, commands, acceptance, rates, and correctness decisions. Raw logs stay
outside Git because the repository does not retain model files, binaries, or
raw server logs.

### Timing terms

- **Decode tok/s:** generated tokens divided by server-reported decode time.
- **Arithmetic decode tok/s:** the simple mean of per-request decode rates.
- **Token-weighted decode tok/s:** total generated tokens divided by total
  decode time. This prevents short, fast responses from dominating the result.
- **Complete-wall tok/s:** generated tokens divided by request wall time. It
  includes prompt evaluation and request overhead, but not model loading.
- **Prefill-dominant tok/s:** prompt tokens divided by complete wall for a
  one-output-token request. It includes one generated token and API overhead.

Compare two rates directly only when the model, backend, workload, output
length, and timing boundary match.

### Correctness checks

The campaign used the checks available for each route:

- exact task contracts, such as a complete 12-object JSON array or `1…30`;
- retained output with character-window and word-trigram repetition checks;
- fresh-server controls when request-state leakage was suspected;
- token equality between Nathan and latest mainline;
- category and heldout coverage on the common suite.

These checks establish generation validity. They do not establish equal model
quality across quantizations.

## 2. Standard comparison model

Every engine in the standardized tables used the same 17,106,775,008-byte
`Q4_K_M` file, SHA-256
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
The [compact artifact][L0] retains hashes for the additional models used only
to reproduce source-specific claims.

## 3. hipEngine

**Verdict: Yes.** hipEngine runs the standard `Q4_K_M` baseline used by every
row in the standardized comparison.

hipEngine `a9b801d59` ran C1-C8 with production-profile BF16 arithmetic. It led
AR at C3-C7, reaching 20.731-36.592 complete-wall tok/s. Its MTP K3 route beat
matched AR only at C2: 17.401 versus 14.247 tok/s. All AR and MTP self-exact
checks passed.

The separately qualified automatic routes remain narrower: normal-owner C1 is
**15.609 versus 9.807 AR tok/s** ([L1]), and production C2/K3 is **17.031 versus
14.887 AR tok/s** ([L2]).

## 4. `q38rocm` / ROCmFPX

**Verdict: Yes as a specialized C1 route.** Strict MTP reproduced the source
claim, but it requires a custom model and exactly one server slot.

### What we tested

- `q38rocm` source commit `5d097740` ([S3])
- Verified `q38rocm` v1.5.2 prebuilt runtime
- ROCmFPX source lineage `0fc9568e`
- Vulkan/RADV on Radeon 8060S
- Exact `ROCmFP4_FAST` target
- Built-in MTP, strict maximum depth 4 at C1
- Normal MTP K3 with the standard `Q4_K_M` at C1-C8

The repository installer contained a stale checksum. We verified the v1.5.2
binary against the GitHub release digest instead of disabling checksum
validation.

### Source-protocol results

| Metric | Published | Local |
| --- | ---: | ---: |
| AR decode tok/s | 14.02 | 14.31 |
| MTP decode tok/s | 30.56-36.04 | **38.85 mean** |
| MTP acceptance | — | 78.1% |
| Repetition guard | Reported clean | Passed |

| Prompt | Decode tok/s | Acceptance |
| --- | ---: | ---: |
| Binary search tree / code | 41.44 | 88.6% |
| Widget factory / reasoning | 38.73 | 75.7% |
| JSON entity extraction | 48.49 | 100.0% |
| Unified versus discrete memory | 26.75 | 48.0% |

### Common-suite results

| Mode | Arithmetic decode | Token-weighted decode | Complete wall | Acceptance |
| --- | ---: | ---: | ---: | ---: |
| AR | 14.782 | 14.782 | 12.803 | — |
| Strict MTP K4 | **35.575** | **32.969** | **24.294** | 62.35% |

| Category | MTP decode tok/s |
| --- | ---: |
| Code | 44.79 |
| General English | 28.41 |
| General Japanese | 26.47 |
| Mixed Japanese/English | 33.42 |

### Concurrency limitation

Strict Qwen MTP enforces one server slot. Starting it with `-np 8` fails during
model load with:

```text
Qwen strict MTP requires a single server slot/sequence
```

The 38.85 tok/s result is therefore C1-only. It is not evidence for multi-user
throughput. Normal MTP K3 supports C1-C8, but on standard `Q4_K_M` it measured
20.464, 27.195, 26.240, 26.532, 32.361, 31.043, 38.123, and 45.365
complete-wall tok/s. It did not lead any cell in the standardized matrix.

### What this means

- All four source-protocol repetition guards passed.
- All ten common-suite requests completed, and every category improved.
- The compact strict common-suite harness did not retain response text.
- No request failure or contamination symptom appeared at C1.
- This route uses a custom model format. Do not present its speed advantage as
  an engine-only comparison against Q4/Q5/Q6/Q8 GGUF files.

## 5. Laurent adaptive DFlash2 fork

**Verdict: No for sequential serving.** The implementation is fast in a fresh
process, but request state leaks between sequential prompts.

### What we tested

- Laurent fork commit `c28d538df`, build 10681 ([S5])
- Vulkan/RADV
- Exact `ROCmFP4_FAST` target
- Exact FP4 DFlash2 `Q4_0` sidecar
- Adaptive draft depth 3-7
- Published 300-token prose-then-JSON sequence

### Published-sequence reproduction

| Policy | Prose decode | JSON decode | JSON result |
| --- | ---: | ---: | --- |
| Bare | 14.148 | 14.128 | On-task, truncated at 300 tokens |
| Fixed K3 | 25.842 | 42.532 | On-task |
| Fixed K7 | 24.481 | 20.859 | Wrong-task prose in the JSON position |
| Adaptive K3-K7 | 25.618 | **66.838** | **Invalid: repeated prose from the previous prompt** |

The 66.838 tok/s result numerically reproduces the 65.6 claim, but it is not a
valid result. The JSON request repeated “the rhythms of the tides” from the
preceding prose request. The fork's degeneration guard still passed because it
checked output length rather than task content.

### Fresh-server controls

Restarting the server before each JSON request removed the stale prose:

| Test | Decode tok/s | Result |
| --- | ---: | --- |
| Fresh server, 300 tokens, trial 1 | 56.948 | Clean JSON, truncated at object 9 |
| Fresh server, 300 tokens, trial 2 | 56.699 | Same output hash, truncated at object 9 |
| Fresh server, 300 tokens, trial 3 | 56.991 | Same output hash, truncated at object 9 |
| Fresh server, 420 tokens, bare | 14.180 | Complete valid 12-object JSON |
| Fresh server, 420 tokens, adaptive | **56.532** | **Complete valid 12-object JSON** |

Use **56.532 tok/s** as the valid structured-output result. It is 3.99x the
matched bare route. Do not use the sequential 65.6/66.838 row.

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

### Earlier built-in-MTP transfer test

Before the FP4+DFlash2 files were available, we tested Laurent's adaptive
controller with the older local `Q4_K_M` and its built-in MTP head ([L4]).

| Arm | Decode tok/s | Versus own AR | Acceptance |
| --- | ---: | ---: | ---: |
| Mainline AR | 11.37 | 1.000x | — |
| Mainline fixed K3 | 16.02 | 1.409x | 63.41% |
| Mainline fixed K7 | 14.83 | 1.304x | 38.43% |
| Laurent AR | 11.35 | 1.000x | — |
| Laurent fixed K3 | **18.93** | **1.668x** | 63.86% |
| Laurent fixed K7 | 14.87 | 1.310x | 38.58% |
| Laurent adaptive K3-K7 | 17.66 | 1.556x | 61.70% |

Direct conclusions from this transfer test:

- adaptive sizing recovered 18.8% over fixed K7;
- adaptive sizing was 6.7% slower than fixed K3;
- mainline `n_max=7,n_min=3` behaved like fixed K7, not Laurent adaptive;
- Laurent fixed K3 was about 18% faster than b10438 mainline, but the build
  gap prevents attributing the difference without a bisect.

Adaptive depth is therefore useful for avoiding a bad deep draft, but it is
not always the fastest policy.

### Standardized built-in-MTP result

The final standard-`Q4_K_M` matrix tested Laurent's ordinary built-in MTP K3,
not adaptive DFlash2. That route passed every repetition guard and led MTP at
C1-C3. Laurent also led prefill at C2 and C6-C8 and AR at C2 and C8. This is a
broad, reusable fork result; the adaptive DFlash2 request-state failure does not
apply to this ordinary built-in-MTP path.

### What must be fixed

Laurent must reset or repair speculative state at every request boundary. The
route needs a sequential multi-prompt correctness gate before it can be used as
a reusable server. Fresh-process speed does not remove this blocker.

## 6. KyaniteLabs MTP+ngram

**Verdict: Yes, with a workload caveat.** The output is correct. The 160+ tok/s
peak measures warm replay, not novel generation.

### What we tested

- KyaniteLabs source profile `7fa3ca81` ([S4])
- llama.cpp HIP `9d57ce456`, build 10438
- Exact Unsloth `UD-Q4_K_XL`
- `HSA_ENABLE_SDMA=0`, `HSA_XNACK=1`
- 98,304 context, one slot, thinking disabled
- MTP maximum depth 12; ngram minimum 24

### Count-to-30 results

| Mode | Cold decode | Warm decode | Output |
| --- | ---: | ---: | --- |
| AR | 11.94 | 11.97 | Exact `1…30` |
| MTP K12 | 61.09 | 59.42-59.49 | Exact `1…30` |
| MTP K12 + ngram | 60.95 | **164.13-167.64** | Exact `1…30` |

MTP provides the cold speedup. Ngram provides no cold benefit. Its entire
160+ tok/s gain comes from replaying the previously generated count sequence.

### Common-suite results

| Mode | Arithmetic decode | Complete wall |
| --- | ---: | ---: |
| AR | 11.964 | 11.679 |
| MTP K12 | 24.390 | 20.450 |
| MTP K12 + ngram | **24.867** | **20.518** |

Production category rates were:

| Category | Decode tok/s |
| --- | ---: |
| Code | 35.82 |
| General English | 16.10 |
| General Japanese | 15.45 |
| Mixed Japanese/English | 21.15 |

Ngram improved arithmetic decode by 1.96% over MTP-only, but complete-wall
speed improved by only 0.33%. That difference is noise-scale for one run.

All diverse-suite outputs were substantive and non-repetitive.

### hipEngine ngram follow-up

The separate hipEngine ngram-composition route remains default-off ([L5]). In
a repetition-heavy strict C2/K3 D80 control, it improved 2.425% over MTP-only
but reached only 0.9875x true AR. D96 and D120 retained correctness or
economics blockers. Kyanite's narrow replay result does not justify enabling
ngram globally in hipEngine.

## 7. PieBru recipes and Nathanw fork

**Verdict: Yes.** The Q5/Q6/Q8 served-speed claims reproduced. Current
mainline is slightly faster in decode.

### What we tested

- PieBru recipe commit `66cfceae` ([S6])
- Nathanw fork `0eb528051a56f34567312ce63ab4e14a3fc71d89`, build 10580
- Matched mainline `4e97ac86ebe2c4cb8212d98d2641ad6768810896`
- Vulkan/RADV
- Exact Unsloth Q5/Q6/Q8 XL targets
- Exact DFlash2 `Q8_0` sidecar
- Ten prompts, up to 128 tokens, thinking disabled

### Served-speed claims

| Quant | Published band | Nathan local | Mainline local |
| --- | ---: | ---: | ---: |
| Q5 | about 23-24 | **24.706** | **24.886** |
| Q6 | 17-21 | **20.549** | **20.343** |
| Q8 | 15-18 | **18.197** | **18.092** |

All three claims are confirmed or conservative.

### Decode results

| Quant | Engine | AR decode | DFlash decode | Acceptance |
| --- | --- | ---: | ---: | ---: |
| Q5 | Nathan | 10.695 | 30.659 | 53.19% |
| Q5 | Mainline | 10.691 | **31.119** | 53.19% |
| Q6 | Nathan | 8.778 | 26.470 | 42.92% |
| Q6 | Mainline | **8.803** | **26.867** | 42.92% |
| Q8 | Nathan | 7.275 | 23.044 | 43.94% |
| Q8 | Mainline | **7.276** | **23.374** | 43.94% |

| Quant / engine | Code | General English | General Japanese | Mixed Japanese/English |
| --- | ---: | ---: | ---: | ---: |
| Q5 Nathan | 40.11 | 25.25 | 18.08 | 29.73 |
| Q5 mainline | 40.94 | 25.79 | 18.11 | 29.82 |
| Q6 Nathan | 36.48 | 19.36 | 13.54 | 26.50 |
| Q6 mainline | 37.32 | 19.44 | 13.59 | 26.67 |
| Q8 Nathan | 31.64 | 15.33 | 13.01 | 23.59 |
| Q8 mainline | 32.37 | 15.40 | 13.06 | 23.68 |

### What this means

- Nathan and mainline produced token-exact outputs in every matched arm.
- All outputs were substantive and non-repetitive.
- Mainline was about 1.4-1.5% faster in DFlash decode.
- Nathan sometimes had faster prefill, which explains its small Q6/Q8
  complete-wall lead.
- The speedup comes from the model, sidecar, and configuration—not from a
  current Nathan decode advantage.

## 8. MikeVeerman Q8 concurrency

**Verdict: Yes, with concurrency-aware routing.** MTP is valuable at low
concurrency and harmful at C4.

### What we tested

- MikeVeerman benchmark source `cc527064` ([S2])
- Exact stock llama.cpp pin `152d337fadb93c2a099653c4072d5512c92c5bfd`
- Vulkan/RADV
- Exact Unsloth `UD-Q8_K_XL`
- 131,072 total context; four 32,768-token slots
- Greedy 256-token generations at C1-C4

The pinned build reported that `--cache-reuse 256` was unsupported for this
context and disabled it in both AR and MTP arms.

### Results

| Concurrency | Published AR | Published MTP | Published ratio | Local AR | Local MTP | Local ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | 7.10 | 15.53 | 2.19x | 7.21 | **16.07** | **2.23x** |
| C2 | 13.01 | 16.63 | 1.28x | 13.35 | **17.48** | **1.31x** |
| C3 | 17.52 | 18.15 | 1.04x | 18.00 | **18.21** | **1.01x** |
| C4 | **21.75** | 16.94 | 0.78x | **21.03** | 17.58 | **0.84x** |

| Concurrency | MTP acceptance |
| ---: | ---: |
| C1 | 72.9% |
| C2 | 71.4% |
| C3 | 71.6% |
| C4 | 66.4% |

At C4, per-request throughput was 5.87 tok/s AR and 4.89 tok/s MTP.

### What this means

- MTP is a large C1 win.
- MTP is approximately neutral at C3.
- MTP loses at C4 even though acceptance remains 66.4%.
- The loss comes from saturation, not failed drafting. Batched AR uses compute
  that was otherwise available to speculative work at C1.
- Admission must account for physical concurrency. Acceptance alone is not
  enough.

## 9. Cross-route analysis

### Source-protocol results are not one leaderboard

Use the standardized `Q4_K_M` tables to rank compatible engines. The
source-protocol routes differ in:

- target format: FP4, Q4, Q5, Q6, or Q8;
- draft method: MTP, DFlash2, or ngram;
- output budget: 24, 128, 256, 300, 420, or 1,024 tokens;
- timing boundary: decode or complete wall;
- workload: code, prose, structured output, multilingual prompts, or replay.

For the one shared FP4 target, Laurent fresh-process adaptive DFlash2 was
faster than `q38rocm` strict MTP K4 on the common suite: 34.483 versus 32.969
token-weighted decode tok/s. Laurent still loses the deployment decision
because sequential requests are incorrect.

### Plain AR is memory-bound

Multiplying model-file bytes by plain-AR decode rate gives these screening
estimates:

| Target | Approximate implied bandwidth |
| --- | ---: |
| `ROCmFP4_FAST` | 208 GB/s |
| Q5 | 223 GB/s |
| Q6 | 223 GB/s |
| Q8 | 229 GB/s |

These are not hardware-counter measurements. Embeddings, metadata, and non-AR
tensors are not necessarily read for every token. The estimates show that a
claimed plain dense-AR rate above the physical 256 GB/s ceiling needs another
explanation.

### Prompt type changes speculative speed

Code and structured output generally accept more draft tokens than explanatory
prose or Japanese heldouts. Report every speculative rate with:

- prompt category;
- acceptance;
- output length;
- timing boundary.

An aggregate without those fields does not transfer to another prompt mix.

### Replay and contamination are different failures of interpretation

- **Kyanite 167.64 tok/s:** correct output on a narrow warm-replay workload.
  The number is real, but it is not novel-generation speed.
- **Laurent 66.838 tok/s:** incorrect output caused by stale request state. The
  number is not a valid result at all.

### Concurrency changes the best policy

At C1, drafting can use compute that would otherwise sit idle during weight
reads. At C4, batched AR uses that compute for real requests. This explains
Mike's 2.23x C1 gain and 0.84x C4 loss.

A scheduler must use measured physical concurrency and cycle cost. Acceptance
alone cannot decide whether to enable speculation.

### hipEngine follow-up

The external results support three concrete actions:

1. Add sequential multi-prompt contamination tests to every speculative
   provider.
2. Compare proposal quality and cycle cost with matched model state, prompt
   history, and physical concurrency.
3. Route speculative work by concurrency economics instead of a global switch.

They do not justify copying an entire external fork.

## 10. Evidence

Campaign artifact:

- [`2026-08-28-gfx1151-qwen38-external-reproduction-survey.json`][L0]

Related hipEngine evidence uses different model artifacts or protocols:

- [normal-owner C1 automatic closure][L1]
- [production C2/K3 retained result][L2]
- [older-model fork transfer test][L4]
- [hipEngine ngram/MTP closeout][L5]

## 11. Sources

- **[S1]** hogeheer499-commits, *Qwen3.8 27B on AMD Strix Halo*, commit
  `029320fb`: [pinned guide][S1].
- **[S2]** MikeVeerman, *Qwen3.8-27B on AMD Strix Halo: what MTP speculative
  decoding gives you*, commit `cc527064`: [pinned benchmark][S2].
- **[S3]** julianmb, `q38rocm`, commit `5d097740`: [pinned report][S3].
- **[S4]** KyaniteLabs, `qwen38-27b-strix-halo`, commit `7fa3ca81`:
  [pinned report][S4].
- **[S5]** LaurentZuijdwijk adaptive DFlash2 llama.cpp fork, commit
  `c28d538df`: [pinned implementation][S5].
- **[S6]** PieBru Qwen3.8 Strix Halo evidence, commit `66cfceae`:
  [pinned repository][S6].

[L0]: ../benchmarks/results/2026-08-28-gfx1151-qwen38-external-reproduction-survey.json
[L1]: ../benchmarks/results/2026-08-27-gfx1151-qwen38-dynamic-admission-d7-closure.json
[L2]: ../benchmarks/results/2026-08-28-gfx1151-qwen38-c2-production-q4-rowtile-retained.json
[L4]: ../benchmarks/results/2026-08-28-gfx1151-qwen38-fork-claim-generalization.json
[L5]: ../benchmarks/results/2026-08-28-gfx1151-qwen38-ngram-mtp-composition-closeout.json
[S1]: https://github.com/hogeheer499-commits/strix-halo-guide/blob/029320fb/QWEN38_STRIX_HALO.md
[S2]: https://github.com/MikeVeerman/qwen38-27-Strix-Halo-bench/blob/cc52706409b0c550636ff068b06894d27079d734/README.md
[S3]: https://github.com/julianmb/q38rocm/blob/5d0977403b0dac778598b1af499bf178b46c0b35/README.md
[S4]: https://github.com/KyaniteLabs/qwen38-27b-strix-halo/blob/7fa3ca810c82c38e7d5a8ef4018d1d1853cec576/README.md
[S5]: https://github.com/LaurentZuijdwijk/llama.cpp/blob/c28d538df5c02643e701a8004db84dbf1bb0ffb2/common/speculative.cpp
[S6]: https://github.com/PieBru/Qwen-3.8-27B_Strix-Halo_gfx1151/tree/66cfceae5edb3dfaf049279738a6fb9cfc5638f6
