# Qwen3.8-27B on Strix Halo: external implementation survey

- Survey date: **2026-08-28**
- Local host: **`gfx1151`**, AMD Ryzen AI MAX+ 395, Radeon 8060S, 128 GB
- Kernel: **Linux 7.1.6-1-cachyos**
- Primary evidence: [compact reproduction artifact][L0]
- Scope: locally tested Qwen3.8-27B autoregressive (AR), Multi-Token
  Prediction (MTP), DFlash2, and ngram-assisted routes on Strix Halo

## Executive summary

“Correct / usable” is deliberately binary. **Yes** means that the locally
observed outputs passed the route's applicable validity and repetition checks
and that the tested server lifecycle did not expose a correctness blocker.
**No** means either a local correctness blocker was observed or the exact route
was not locally qualified. A Yes is not a claim of equal model quality across
quantizations.

The speed columns are not a single leaderboard. Each row retains its model,
workload, output length, and timing boundary. Compare rates directly only when
the row or detailed section names a matched protocol.

**Canonical hipEngine artifact decision:** Qwen3.8 remains standardized on the
existing, qualified `Q4_K_M` artifact family. The new Unsloth `UD-Q4_K_M` is an
external-runtime comparison artifact only; its filename does not make its
mixed tensor inventory storage-compatible with hipEngine's resident Q4 path.

| Implementation / route | Published claim | Local result | Correct / usable | Notes |
| --- | --- | --- | :---: | --- |
| **hipEngine `61b83b9c3` + new Unsloth `UD-Q4_K_M`** | No external claim; intended same-artifact control | Does not load | **No** | Exact artifact contains unsupported dense `Q3_K`, `IQ4_NL`, and `IQ3_S` tensors. Existing qualified hipEngine Q4 lineages are unaffected. |
| **`q38rocm` v1.5.2 / ROCmFPX, `ROCmFP4_FAST`, strict MTP K4** | 14.02 AR; 30.56-36.04 MTP decode tok/s | 14.31 AR; **38.85** MTP on the source protocol. Common suite: 14.782 AR → **35.575** arithmetic / **32.969** token-weighted decode tok/s | **Yes** | Fastest locally tested route without a server-lifecycle correctness blocker. |
| **Laurent adaptive DFlash2 fork `c28d538df`** | 65.6 structured; 26.1 prose decode tok/s | Valid complete JSON: **56.532**; prose: 25.618. Fresh-process common suite: **37.752** arithmetic / **34.483** token-weighted decode tok/s | **No** | Fastest valid fresh-process route, but sequential requests contaminate speculative state. The local 66.838 tok/s sequential JSON row emitted repeated prose and is invalid. |
| **KyaniteLabs HIP MTP+ngram** | 59.7 cold and 148-163 warm count-to-30; 11-24 real traffic | 60.95 cold and **164.13-167.64** warm count-to-30; common suite **24.867** decode / **20.518** complete-wall tok/s | **Yes** | Count output was exact. The 160+ peak is warm repetition replay, not novel-generation throughput; ngram added no material general-suite benefit. |
| **PieBru recipe on Nathanw fork `0eb528051`** | Q5 about 23-24, Q6 17-21, Q8 15-18 served tok/s | Q5 **24.706**, Q6 **20.549**, Q8 **18.197** complete-wall tok/s | **Yes** | Claims confirmed. Latest mainline was about 1.4-1.5% faster in DFlash decode, so the fork is not the source of the decode gain. |
| **MikeVeerman stock llama.cpp pin `152d337fa`, Q8 MTP** | 2.19x AR at C1; 0.78x AR at C4 | **2.23x** at C1; 1.01x at C3; **0.84x** at C4 | **Yes** | Crossover reproduced. MTP remains saturated while AR gains from batching; acceptance does not collapse. |
| **yandaq harness + latest mainline + new `UD-Q4_K_M`** | Q4 26.47; other quants 15.20-26.26; K≥6 can produce fake repetitive speed | K4 **31.18**, K6 28.75, K8 20.99 mean decode tok/s | **Yes** | K4 is best. K6/K8 stayed non-repetitive on the current model/engine and slowed instead of producing a fake speedup. |
| **Latest mainline Vulkan `4e97ac86` + new `UD-Q4_K_M`** | Current stock control | Common suite: 13.035 AR → **33.454** MTP K3 decode; 9.758 → **17.572** complete-wall tok/s | **Yes** | Long-output yandaq controls were substantive and non-repetitive. |
| **Ollama 0.32.13 guide route** | 20.42 generation; 292.49 prompt tok/s | Not run | **No** | No means not locally qualified, not known-broken. Exact artifact/runtime were deprioritized. |
| **LlamaStash / stock b10503 Q8_0 report** | 7.3 AR → 22.4 MTP tok/s | Not reproduced exactly | **No** | Raw package and exact prompts/config were unavailable; the guide labels this unimported community evidence. |

### Bottom line

- **Fastest valid one-task result:** Laurent adaptive DFlash2 at **56.532
  decode tok/s** for a complete 12-object JSON response, but only in a fresh
  server process.
- **Fastest clean reusable route tested:** `q38rocm` strict MTP K4 at
  **35.575 arithmetic / 32.969 token-weighted decode tok/s** on the common
  suite.
- **Fastest common-suite decode:** Laurent at **37.752 arithmetic / 34.483
  token-weighted decode tok/s**, with a fresh process for every prompt. It is
  not deployable as a sequential server in the tested state.
- **Highest raw number:** Kyanite at **167.64 tok/s**, entirely from warm ngram
  replay of count-to-30. It is not general generation speed.

## 1. How the survey was run

### 1.1 Host and common suite

All local results were measured on physical host `gfx1151`:

| Item | Value |
| --- | --- |
| APU | AMD Ryzen AI MAX+ 395 |
| GPU | Radeon 8060S / `gfx1151` |
| Unified memory | 128 GB |
| Theoretical memory bandwidth | 256 GB/s |
| Kernel | Linux 7.1.6-1-cachyos |
| Common suite | `benchmarks/prompts/mtpbench-code-general-ja.jsonl` |
| Suite SHA-256 | `fac920be5e691fec2cb70fd8b7eedddab8926b89d6a1627f62ec4f441d86084a` |
| Prompt coverage | 10 prompts: code, general English, general Japanese, and mixed Japanese/English; four category-heldouts |
| Common sampling | Greedy, prompt cache disabled |

The compact artifact records every model size and full SHA-256, the runtime
commit, command, rates, acceptance, and correctness verdict ([L0]). Raw local
logs remain outside Git because compiled binaries, model files, and raw server
logs are not repository artifacts.

### 1.2 Timing boundaries

The survey keeps three timing boundaries separate:

- **Decode tok/s:** server-reported generated tokens divided by decode time.
- **Token-weighted decode tok/s:** total generated tokens divided by the sum of
  decode durations. This prevents short, fast responses from dominating an
  arithmetic mean.
- **Complete-wall tok/s:** generated tokens divided by request wall time,
  including prompt evaluation and request overhead but excluding model load.

A reported rate is not directly comparable when the model file, backend,
request shape, output length, or timing boundary changes. The executive table
therefore reports the boundary next to each result instead of bolding one
cross-route “winner.”

### 1.3 Correctness checks

The campaign used the strongest check available for each source protocol:

- exact task contracts, such as a complete 12-object JSON array or the exact
  sequence `1…30`;
- output retention plus character-window and word-trigram repetition checks;
- fresh-server controls when cross-request state was suspected;
- matched output hashes or token equality between Nathan and latest mainline;
- category and heldout coverage on the common suite.

These are generation-validity checks, not perplexity or full downstream model
quality comparisons. Different quantizations must not be treated as
quality-equivalent based on speed testing alone.

## 2. Models and exact artifacts

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| New Unsloth `UD-Q4_K_M` | 16,464,440,224 | `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482` |
| `ROCmFP4_FAST` target | 14,562,236,384 | `fb89c78d2be91cdb68eaaaa45b1270710bf34aa721dc1f0b9e3aa7b98d2e1da9` |
| FP4 DFlash2 `Q4_0` sidecar | 1,034,216,992 | `4264d8f2277ec9ae791c570ddc36940f92857f2e8a41569217e45b7563190285` |
| Unsloth `UD-Q4_K_XL` | 17,559,178,144 | `3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e` |
| Unsloth `UD-Q5_K_XL` | 20,876,938,144 | `8601193d3d5760c37fb8ce1b43afebc69df5fb24e1fbc5a547c32e2200305276` |
| Unsloth `UD-Q6_K_XL` | 25,299,061,664 | `701d8fa9ed214ab21bfc130cd2a7df19ca89bbef7713e2dfb19f3c63696aa917` |
| Unsloth `UD-Q8_K_XL` | 31,457,991,680 | `af36ecb6b5db1407953345b746c14ac93f0657dda413910b4348683a2d990377` |
| DFlash2 `Q8_0` sidecar | 2,056,414,752 | `7f1c9a31a6ed40044c69f6508b50fd63b87abd8e1fb7fe4290303df549153751` |

## 3. hipEngine

### Configuration

The intended campaign control used hipEngine commit `61b83b9c3`, the new
Unsloth `UD-Q4_K_M`, production automatic MTP, and C1/C2/C4/C8. The model was
not assumed compatible merely because its filename said `UD-Q4_K_M`.

### Result

Preparation failed before inference:

```text
ValueError: unsupported Qwen3.5 GGUF tensor type 'Q3_K' outside rank-3 expert slots: blk.0.ffn_up.weight
```

Inspection found seven dense `Q3_K`, seven dense `IQ4_NL`, and four dense
`IQ3_S` tensors. hipEngine supports `Q3_K` only for selected rank-3 expert
slots, has no native dense path for these three families, and lacks an
`IQ3_S` CPU dequantizer. Supporting the file safely is a kernel and numerics
project, not a loader allow-list change.

### Verdict

**Correct / usable: No for this exact artifact.** No tokens were generated, so
there is no hipEngine speed row for the new file. This does not invalidate the
already-qualified older Qwen3.8 `Q4_K_M` lineage. That existing lineage remains
the canonical hipEngine artifact family; the mixed-quant Unsloth file stays
external-only rather than introducing new resident storage formats solely for
its nominal `UD-Q4_K_M` label. On the canonical lineage, the retained
normal-owner C1 automatic route is 15.609 versus 9.807 tok/s AR, and production
C2/K3 is 17.031 versus 14.887 tok/s AR ([L1], [L2]). Those numbers must not be
compared as same-model results with the new Unsloth artifact.

### Prior supported-artifact context

The earlier same-host packet used a different Qwen3.8 `Q4_K_M` file (SHA-256
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`),
the same ten-prompt D24 suite, and llama.cpp HIP build 10438 ([L3]):

| Metric | llama.cpp HIP | hipEngine | hipEngine delta |
| --- | ---: | ---: | ---: |
| AR decode tok/s | 12.156 | 12.332 direct diagnostic | +1.4% |
| AR complete-wall tok/s | 9.750 | 9.807 | +0.6% |
| MTP decode tok/s | 24.897 | 21.158 direct-leaf | -15.0% |
| MTP complete-wall tok/s | 15.730 | 15.609 | -0.8% |
| MTP/AR complete wall | 1.613x | 1.592x | -1.3% |
| Draft acceptance, K3 | 90.16% | 78.57% | -11.59 points |

This packet is diagnostic, not a clean retained engine A/B: hipEngine values
were reused from earlier runs, llama.cpp used F16 KV while hipEngine used BF16,
direct and complete-wall rows mix timing boundaries, and each arm had one run.
Within those limits, complete-wall C1 throughput was near parity while proposal
acceptance differed materially. That finding remains a useful hipEngine
optimization lead, but it does not repair support for the new artifact.

The same older-artifact direct row-scaling diagnostic measured 12.332, 23.708,
43.828, and 46.503 aggregate AR tok/s at physical rows 1, 2, 4, and 8. Rows
1-4 amortized weight reads well; rows 4-8 added 88.5% step wall for twice the
rows. It was a synthetic direct-graph result, not HTTP-serving throughput.

## 4. `q38rocm` / ROCmFPX

### Configuration

- Published source: `q38rocm` commit `5d097740` ([S3])
- Installed runtime: verified `q38rocm` v1.5.2 prebuilt
- Source lineage: ROCmFPX `0fc9568e`
- Backend: Vulkan/RADV on Radeon 8060S
- Target: exact `ROCmFP4_FAST` artifact
- Draft path: built-in MTP, strict maximum depth 4

The v1.5.2 release binary was verified against the GitHub release digest. The
repository installer contained a stale checksum, so the release API digest was
used rather than weakening checksum validation.

### Claimed versus local source protocol

| Metric | Published | Local |
| --- | ---: | ---: |
| AR decode tok/s | 14.02 | 14.31 |
| MTP decode tok/s | 30.56-36.04 | **38.85** mean |
| Local MTP acceptance | — | 78.1% |
| Prompts | 4 | 4 |
| Repetition guard | Reported clean | Passed |

| Local source-protocol prompt | Decode tok/s | Acceptance |
| --- | ---: | ---: |
| Binary search tree / code | 41.44 | 88.6% |
| Widget factory / reasoning | 38.73 | 75.7% |
| JSON entity extraction | 48.49 | 100.0% |
| Unified versus discrete memory | 26.75 | 48.0% |

### Common-suite result

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

### Correctness and verdict

All four source-protocol repetition guards passed. All ten common-suite
requests completed and every category improved; the compact common-suite
harness did not retain response text. No request failure or contamination
symptom appeared. **Correct / usable: Yes.** This was the fastest locally
tested route without a server-lifecycle correctness blocker. It uses a custom
model format and must not be represented as an engine-only comparison against
Q4/Q5/Q6/Q8 GGUF routes.

## 5. Laurent adaptive DFlash2 fork

### Configuration

- Fork: `LaurentZuijdwijk/llama.cpp` commit `c28d538df`, build 10681 ([S5])
- Backend: Vulkan/RADV
- Target: exact `ROCmFP4_FAST`
- DFlash2 sidecar: exact FP4 `Q4_0` sidecar
- Adaptive policy: minimum 3, maximum 7, probability minimum 0
- Headline protocol: prose followed by structured JSON, 300 generated tokens

### Exact headline matrix

| Policy | Prose decode | JSON decode | JSON validity |
| --- | ---: | ---: | --- |
| Bare | 14.148 | 14.128 | Truncated at 300 tokens, but structurally on-task |
| Fixed K3 | 25.842 | 42.532 | On-task |
| Fixed K7 | 24.481 | 20.859 | Wrong-task prose appeared in the JSON position |
| Adaptive K3-K7 | 25.618 | **66.838** | **Invalid: repeated prose from the preceding prompt** |

The local 66.838 tok/s row numerically reproduces the published 65.6 headline,
but the emitted sequence is invalid. It repeats “the rhythms of the tides”
from the preceding prose request instead of producing JSON. The fork's
reported degeneration flag stayed false because its check covered emitted
length rather than task content.

### Fresh-server controls

Restarting the server before each JSON request removed the contamination:

| Test | Decode tok/s | Contract result |
| --- | ---: | --- |
| Fresh server, 300 tokens, trial 1 | 56.948 | Clean deterministic JSON, truncated at object 9 |
| Fresh server, 300 tokens, trial 2 | 56.699 | Same output hash, truncated at object 9 |
| Fresh server, 300 tokens, trial 3 | 56.991 | Same output hash, truncated at object 9 |
| Fresh server, 420 tokens, bare | 14.180 | Complete valid 12-object JSON |
| Fresh server, 420 tokens, adaptive | **56.532** | **Complete valid 12-object JSON** |

The 420-token result is the quality-valid structured number: **56.532 tok/s,
3.99x bare**. The 65.6/66.838 sequential number is not valid throughput.

### Fresh-process common suite

Every prompt used a new server process, preventing cross-request contamination:

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

Before the exact FP4+DFlash2 artifacts were available, the fork's adaptive
controller was tested with the older local `Q4_K_M` and its built-in MTP head
at a 128-token horizon ([L4]):

| Arm | Decode tok/s | Versus own AR | Acceptance |
| --- | ---: | ---: | ---: |
| Mainline AR | 11.37 | 1.000x | — |
| Mainline fixed K3 | 16.02 | 1.409x | 63.41% |
| Mainline fixed K7 | 14.83 | 1.304x | 38.43% |
| Laurent AR | 11.35 | 1.000x | — |
| Laurent fixed K3 | **18.93** | **1.668x** | 63.86% |
| Laurent fixed K7 | 14.87 | 1.310x | 38.58% |
| Laurent adaptive K3-K7 | 17.66 | 1.556x | 61.70% |

Adaptive sizing recovered 18.8% over a fixed deep K7 draft but remained 6.7%
slower than fixed K3 on that model and suite. Mainline `n_max=7,n_min=3`
behaved like fixed K7 rather than like Laurent's adaptive controller. The
fork's fixed-K3 arm was about 18% faster than b10438 mainline with similar AR,
acceptance, and draft counts, but the build-version gap prevents attributing
that difference to Laurent's patch without a bisect. This transfer test does
not contradict the exact FP4+DFlash2 result; it shows that adaptive depth is
not universally the best policy.

### Correctness blocker and verdict

**Correct / usable: No.** The adaptive mechanism is genuinely fast, and
fresh-process outputs are valid, but the tested server cannot safely serve a
sequence of unrelated requests. The state leak is a control/correctness defect,
not permissible numerical drift. Until the speculative state is reset or
repaired at request boundaries and a sequential multi-prompt gate passes, this
route must not be promoted as a reusable server.

## 6. KyaniteLabs MTP+ngram profile

### Configuration

- Source profile: KyaniteLabs `7fa3ca81` ([S4])
- Engine: ggml-org llama.cpp HIP commit `9d57ce456`, build 10438
- Model: exact Unsloth `UD-Q4_K_XL`
- Environment: `HSA_ENABLE_SDMA=0`, `HSA_XNACK=1`
- Context: 98,304; one slot; thinking disabled
- Production speculation: `draft-mtp,ngram-mod`, MTP maximum depth 12,
  ngram minimum 24

### Count-to-30 reproduction

| Mode | Cold decode | Warm decode | Output |
| --- | ---: | ---: | --- |
| AR | 11.94 | 11.97 | Exact `1…30` |
| MTP K12 | 61.09 | 59.42-59.49 | Exact `1…30` |
| MTP K12 + ngram | 60.95 | **164.13-167.64** | Exact `1…30` |

MTP supplies the cold gain. Ngram adds no cold benefit; the 160+ warm result
comes from replaying a sequence that the server has already seen.

### Common-suite result

| Mode | Arithmetic decode | Complete wall |
| --- | ---: | ---: |
| AR | 11.964 | 11.679 |
| MTP K12 | 24.390 | 20.450 |
| MTP K12 + ngram | **24.867** | **20.518** |

The production profile's category rates were 35.82 code, 16.10 general
English, 15.45 general Japanese, and 21.15 mixed Japanese/English tok/s.
Production improved arithmetic decode by 1.96% over MTP-only, but complete-wall
throughput improved by only 0.33%, which is noise-scale for this single run.

### Verdict

**Correct / usable: Yes.** Count output was exact and all diverse-suite outputs
were substantive and non-repetitive. The claim is accurate when described as
workload-specific: MTP generalizes, while ngram replay does not provide a
material general-traffic win.

A separate hipEngine ngram-composition closeout remains default-off ([L5]). On
a repetition-heavy strict C2/K3 D80 control it improved 2.425% over MTP-only
but still reached only 0.9875x true AR; D96 and D120 retained correctness or
economics blockers. Kyanite's result therefore does not justify enabling the
hipEngine provider globally.

## 7. PieBru recipes, Nathanw fork, and latest mainline

### Configuration

- Recipe source: PieBru `66cfceae` ([S6])
- Nathanw fork: `0eb528051a56f34567312ce63ab4e14a3fc71d89`, build 10580
- Matched mainline: `4e97ac86ebe2c4cb8212d98d2641ad6768810896`
- Backend: Vulkan/RADV
- Targets: exact Unsloth Q5/Q6/Q8 XL artifacts
- Sidecar: exact DFlash2 `Q8_0`
- Common suite: 10 prompts, 128-token maximum, thinking disabled

### Claimed versus local complete-wall throughput

| Quant | Published served band | Nathan local | Latest mainline local |
| --- | ---: | ---: | ---: |
| Q5 | about 23-24 | **24.706** | **24.886** |
| Q6 | 17-21 | **20.549** | **20.343** |
| Q8 | 15-18 | **18.197** | **18.092** |

All published bands are confirmed or conservative.

### Decode details

| Quant | Engine | AR decode | DFlash decode | Acceptance |
| --- | --- | ---: | ---: | ---: |
| Q5 | Nathan | 10.695 | 30.659 | 53.19% |
| Q5 | Mainline | 10.691 | **31.119** | 53.19% |
| Q6 | Nathan | 8.778 | 26.470 | 42.92% |
| Q6 | Mainline | **8.803** | **26.867** | 42.92% |
| Q8 | Nathan | 7.275 | 23.044 | 43.94% |
| Q8 | Mainline | **7.276** | **23.374** | 43.94% |

Category decode rates preserve the prompt dependence:

| Quant / engine | Code | General English | General Japanese | Mixed Japanese/English |
| --- | ---: | ---: | ---: | ---: |
| Q5 Nathan | 40.11 | 25.25 | 18.08 | 29.73 |
| Q5 mainline | 40.94 | 25.79 | 18.11 | 29.82 |
| Q6 Nathan | 36.48 | 19.36 | 13.54 | 26.50 |
| Q6 mainline | 37.32 | 19.44 | 13.59 | 26.67 |
| Q8 Nathan | 31.64 | 15.33 | 13.01 | 23.59 |
| Q8 mainline | 32.37 | 15.40 | 13.06 | 23.68 |

Outputs were token-exact between Nathan and mainline for every matched arm and
were substantive and non-repetitive. Mainline is about 1.4-1.5% faster in
DFlash decode. Nathan sometimes offsets that difference with faster prefill,
which explains its small Q6/Q8 complete-wall lead.

### Verdict

**Correct / usable: Yes.** PieBru's served-speed claims are supported. The
performance comes from the model, DFlash2 sidecar, and configuration rather
than a current decode advantage in the Nathan fork.

## 8. MikeVeerman Q8 concurrency

### Configuration

- Benchmark source: MikeVeerman `cc527064` ([S2])
- Exact engine pin: stock llama.cpp
  `152d337fadb93c2a099653c4072d5512c92c5bfd`
- Backend: Vulkan/RADV
- Model: exact Unsloth `UD-Q8_K_XL`
- Total context: 131,072; four slots of 32,768
- Server profile: `-ngl 999 -fa on -b 2048 -ub 512 --no-mmap
  --cache-reuse 256`
- Workload: greedy 256-token generations, C1 through C4

The pinned build reported that cache reuse was unsupported for this context and
disabled it. This condition applied to both AR and MTP arms.

### Claimed versus local crossover

| Concurrency | Published AR | Published MTP | Published ratio | Local AR | Local MTP | Local ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | 7.10 | 15.53 | 2.19x | 7.21 | **16.07** | **2.23x** |
| C2 | 13.01 | 16.63 | 1.28x | 13.35 | **17.48** | **1.31x** |
| C3 | 17.52 | 18.15 | 1.04x | 18.00 | **18.21** | **1.01x** |
| C4 | **21.75** | 16.94 | 0.78x | **21.03** | 17.58 | **0.84x** |

Local MTP acceptance was 72.9%, 71.4%, 71.6%, and 66.4% from C1 through C4.
At C4, per-request throughput was 5.87 tok/s AR versus 4.89 tok/s MTP.

### Verdict

**Correct / usable: Yes, with concurrency-aware routing.** The central claim
reproduces: MTP is a large C1 win, approximately neutral at C3, and harmful at
C4. The loss is caused by saturation rather than acceptance collapse. A server
should disable MTP for sufficiently dense parallel work instead of applying a
single speculative policy at every concurrency.

## 9. yandaq long-output depth sweep

### Configuration

- Harness: yandaq `eb68ceb0` ([S7])
- Engine: latest mainline Vulkan `4e97ac86`
- Model: new Unsloth `UD-Q4_K_M`
- Workloads: prose, code, and JSON
- Three 1,024-token runs per family after warmup, natural stopping
- Sampling: temperature 0.7, top-p 0.8, top-k 20, seed 42, thinking disabled
- Validity guard: 30-character-window uniqueness must remain at least 0.5

This is the exact yandaq harness on the current model artifact, not the same
artifact/runtime snapshot that produced every historical row in the source.

### Results

| Maximum draft depth | Prose | Code | JSON | Mean | Validity |
| ---: | ---: | ---: | ---: | ---: | --- |
| K4 | 27.90 | 34.71 | 30.92 | **31.18** | Passed |
| K6 | 24.43 | 32.62 | 29.20 | 28.75 | Passed |
| K8 | 16.45 | 24.01 | 22.53 | 20.99 | Passed; uniqueness 0.924-0.974 |

K4 acceptance was 0.58/0.78/0.67 for prose/code/JSON. At K8 it fell to
0.35/0.57/0.52.

### Verdict

**Correct / usable: Yes at K4.** The source's main operational advice—that a
shallow draft is best—holds. The historical “K≥6 can become a fake ~35 tok/s
repetition result” did not reproduce on current mainline plus the new model.
Here, deeper drafts remained valid and became progressively slower.

## 10. Latest mainline on the new Unsloth `UD-Q4_K_M`

The common-suite stock control used mainline Vulkan `4e97ac86`, 24-token greedy
outputs, and no prompt cache:

| Mode | Arithmetic decode | Complete wall | Acceptance |
| --- | ---: | ---: | ---: |
| AR | 13.035 | 9.758 | — |
| MTP K3 | **33.454** | **17.572** | 95.45% |

| Mode | Code | General English | General Japanese | Mixed Japanese/English |
| --- | ---: | ---: | ---: | ---: |
| AR | 12.92 | 13.09 | 13.12 | 13.12 |
| MTP K3 | 33.08 | 35.27 | 33.15 | 32.70 |

**Correct / usable: Yes.** The short common-suite outputs were supplemented by
yandaq's long-output K4/K6/K8 controls, all of which stayed substantive and
non-repetitive. The high 24-token acceptance must not be extrapolated to long
outputs; yandaq's K4 acceptance ranged from 58% to 78%.

## 11. Routes not locally qualified

### Ollama 0.32.13

The hogeheer499 guide reports 20.42 generation tok/s and 292.49 prompt tok/s
over nine warm API repeats ([S1]). We did not install the exact Ollama runtime
or artifact. At the stated 17.7 GB model size, interpreting 20.42 tok/s as
plain dense AR would imply 361 GB/s, above the host's 256 GB/s theoretical
memory bandwidth. The published number may use a broader generation boundary,
speculation, caching, or another condition not visible in the summary.

**Correct / usable: No—unqualified, not known-broken.** The result remains
published evidence, not a local AR row.

### LlamaStash / stock b10503 Q8_0

The guide tracks a community report of 7.3 tok/s AR and 22.4 tok/s MTP. The raw
package was never imported into the guide, the exact prompts/config could not
be recovered, and the exact Q8_0 target was not present locally.

**Correct / usable: No—unqualified, not known-broken.** MikeVeerman's exact Q8
concurrency reproduction supports the general plausibility of a large C1 MTP
uplift, but it uses `UD-Q8_K_XL` and cannot validate the LlamaStash number.

## 12. Cross-route conclusions

### 12.1 No single speed ranking is valid

The measured routes use FP4, Q4, Q5, Q6, and Q8 targets; MTP, DFlash2, and
ngram drafts; 24-, 128-, 256-, 300-, 420-, and 1,024-token budgets; and decode
or complete-wall timing. The executive table answers “what happened in each
published route,” not “which engine is universally fastest.”

For the one exact target/draft pair shared by `q38rocm` and Laurent, Laurent's
fresh-process adaptive DFlash2 common-suite result is faster than strict MTP
K4: 34.483 versus 32.969 token-weighted decode tok/s. That advantage is not
retainable as a server win because Laurent fails sequential-request
correctness.

The local plain-AR rows remain consistent with memory-bound decode. Multiplying
file bytes by decode rate gives screening estimates of approximately 208 GB/s
for `ROCmFP4_FAST`, 215 GB/s for the new `UD-Q4_K_M`, 223 GB/s for Q5, 223 GB/s
for Q6, and 229 GB/s for Q8. These are not hardware-counter measurements:
embeddings, metadata, and non-AR tensors are not necessarily streamed per
token. They do show why an apparent dense-AR row above the 256 GB/s physical
ceiling needs another explanation.

### 12.2 Prompt dependence is large

Speculative throughput tracks how predictable the continuation is. Code and
structured output generally accept more draft tokens than explanatory prose or
Japanese heldouts. Every speculative rate should therefore retain prompt
category, acceptance, output length, and timing boundary.

### 12.3 Repetition replay is not novel generation

Kyanite's 167.64 tok/s count result is correct but measures replay of a warm,
repetitive sequence. Laurent's 66.838 tok/s sequential JSON result is different:
it is not merely narrow, but invalid, because stale state changed the task
output. The survey keeps those two cases separate.

### 12.4 Concurrency changes the best policy

MikeVeerman's crossover shows that drafting can consume otherwise-idle compute
at C1 but compete with useful batched work at C4. Acceptance alone cannot drive
admission; the scheduler must account for current physical width and target
cycle cost.

### 12.5 hipEngine follow-up

The external campaign does not justify copying an entire fork. The useful
engineering leads are:

1. make exact-artifact support explicit by tensor family rather than filename;
2. preserve strict request-boundary state ownership and add sequential
   multi-prompt contamination tests for every speculative provider;
3. compare proposal quality and cycle cost at matched model, prompt history,
   and physical concurrency;
4. route speculative work by measured concurrency economics rather than a
   global on/off setting.

## 13. Reproduction evidence

The compact artifact is the durable numeric source for this campaign:

- [`2026-08-28-gfx1151-qwen38-external-reproduction-survey.json`][L0]

Related retained hipEngine evidence remains separate because it used different
model artifacts or protocols:

- [normal-owner C1 automatic closure][L1]
- [production C2/K3 retained result][L2]
- [historical same-host external comparison packet][L3]
- [older-artifact fork configuration transfer test][L4]
- [hipEngine ngram/MTP composition closeout][L5]

## 14. Sources

### External, commit-pinned

- **[S1]** hogeheer499-commits, *Qwen3.8 27B on AMD Strix Halo*, commit
  `029320fb`: [pinned guide][S1].
- **[S2]** MikeVeerman, *Qwen3.8-27B on AMD Strix Halo: what MTP speculative
  decoding gives you*, commit `cc527064`: [pinned benchmark][S2].
- **[S3]** julianmb, `q38rocm`, commit `5d097740`: [pinned report][S3].
- **[S4]** KyaniteLabs, `qwen38-27b-strix-halo`, commit `7fa3ca81`:
  [pinned report][S4].
- **[S5]** LaurentZuijdwijk, adaptive DFlash2 llama.cpp fork, commit
  `c28d538df`: [pinned implementation][S5].
- **[S6]** PieBru, Qwen3.8 Strix Halo evidence, commit `66cfceae`:
  [pinned repository][S6].
- **[S7]** yandaq, Qwen3.8 Strix Halo harness, commit `eb68ceb0`:
  [pinned repository][S7].

[L0]: ../benchmarks/results/2026-08-28-gfx1151-qwen38-external-reproduction-survey.json
[L1]: ../benchmarks/results/2026-08-27-gfx1151-qwen38-dynamic-admission-d7-closure.json
[L2]: ../benchmarks/results/2026-08-28-gfx1151-qwen38-c2-production-q4-rowtile-retained.json
[L3]: ../benchmarks/results/2026-08-28-gfx1151-qwen38-llamacpp-1to1.json
[L4]: ../benchmarks/results/2026-08-28-gfx1151-qwen38-fork-claim-generalization.json
[L5]: ../benchmarks/results/2026-08-28-gfx1151-qwen38-ngram-mtp-composition-closeout.json
[S1]: https://github.com/hogeheer499-commits/strix-halo-guide/blob/029320fb/QWEN38_STRIX_HALO.md
[S2]: https://github.com/MikeVeerman/qwen38-27-Strix-Halo-bench/blob/cc52706409b0c550636ff068b06894d27079d734/README.md
[S3]: https://github.com/julianmb/q38rocm/blob/5d0977403b0dac778598b1af499bf178b46c0b35/README.md
[S4]: https://github.com/KyaniteLabs/qwen38-27b-strix-halo/blob/7fa3ca810c82c38e7d5a8ef4018d1d1853cec576/README.md
[S5]: https://github.com/LaurentZuijdwijk/llama.cpp/blob/c28d538df5c02643e701a8004db84dbf1bb0ffb2/common/speculative.cpp
[S6]: https://github.com/PieBru/Qwen-3.8-27B_Strix-Halo_gfx1151/tree/66cfceae5edb3dfaf049279738a6fb9cfc5638f6
[S7]: https://github.com/yandaq/qwen3.8-27b-strix-halo/tree/eb68ceb0268d1c4fb4999e57e6fef0900441552e
