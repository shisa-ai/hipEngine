# Qwen3.8-Flash-Next gfx1151 Performance Campaign

Status: **active plan, reviewed 2026-08-30.** The category-balanced exact-token
screening baseline now covers p512/p1024/p4096 plus 128 autoregressive
transitions after each prefix. Named hipEngine production measures
**82.51/81.22/67.93 tok/s** prefill and **13.82/13.79/10.40 tok/s** decode.
Repeatability-valid screening comparators remain **2.90–4.34x** ahead on HIP
prefill and **2.92–3.93x** ahead on Vulkan prefill; decode gaps are
**1.24–1.42x** and **1.46–1.74x**. The first depth-specific blocker is the
2,051-token QSA path transition: p1024→p4096 adds **23.67 ms/token** in
hipEngine versus **5.61–8.47 ms/token** on the repeatability-valid upstream
lanes. This is not section-6 closure: the run used three repetitions, several
rows exceed 2% CV, and cold-PLE plus heldout modes remain open. The frozen
p508/tg32 role baseline remains the attribution anchor.
This document is the performance-specific plan and punchlist.
[`QWEN3.8-FLASH-NEXT.md`](QWEN3.8-FLASH-NEXT.md) remains the model/bring-up
authority; this file owns only the gap-closure campaign.

## 1. Objective and boundaries

Close the same-host, same-model, same-weight-quant performance gap to the best
current HIP comparator first, then to the best same-host Vulkan comparator.
Upstream llama.cpp remains the role-resolved attribution baseline; EngramHalo
HIP and Nathan Vulkan are additional competitor lanes, not inherited evidence.
A retained win must preserve the published execution-profile contract and must
come with a compact artifact, a worklog entry, and a benchmark rollup. The
campaign does not close merely because one microbenchmark, one prompt, one
backend, or one MTP budget wins.

### In scope

- Host: `zbook`, AMD Ryzen AI Max+ Pro 395, Radeon 8060S, `gfx1151`.
- Model: `Qwen/Qwen3.8-Flash-Next` through the pinned Unsloth
  `UD-Q4_K_XL` split GGUF.
- KV/cache policy: current BF16 baseline unless a row explicitly declares a
  different KV profile.
- Profiles: named `strict` and `production` manifests.
- Binding AR representation: the pinned target weights with BF16 K/V. Q8 K/V
  is a separately declared T3 product configuration after BF16 AR parity.
- Workloads: category-balanced exact matched p512, p1024, and p4096 prefill
  with 128 autoregressive transitions after each prefix, then the existing
  long-context and full-category MTP suites. Legacy p508/p1012/tg32 rows remain
  tail/continuity diagnostics, not closure targets.
- Comparator lanes: upstream llama.cpp HIP/Vulkan, EngramHalo HIP, and Nathan
  Vulkan, each pinned to an exact source and binary identity.

### Out of scope

- Changing model representation, quant recipe, or prompts to improve a score.
- Treating external EngramHalo/Nathan numbers as hipEngine results.
- Treating a different weight quant, KV type, prompt, cache state, or MTP policy
  as a same-configuration old-to-new comparison.
- Vulkan-specific code in this campaign except as design evidence for HIP work.
- New feature expansion that does not close a measured gap.
- Inferring W7900/gfx1100 performance from this host.

## 2. Current verified gap

Canonical screening evidence:
[`2026-08-30-gfx1151-qwen38-flash-next-canonical-ar-screening.json`](../benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-canonical-ar-screening.json).
It pins hipEngine commit `6fbcc721a`, current upstream `f1793c1c4`, every
external binary and patch hash, the exact 12-case fixture, all 180 measured
samples, host state, and output-repeatability verdicts. The role-resolved p508
and tg32 attribution anchor remains
[`2026-08-30-gfx1151-qwen38-flash-next-fresh-full-profile.json`](../benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-fresh-full-profile.json).

### Historical p508/tg32 attribution rows

| Workload | hipEngine production | llama.cpp HIP | llama.cpp Vulkan | HIP advantage | Vulkan advantage |
| --- | ---: | ---: | ---: | ---: | ---: |
| p508 prefill | **84.83 tok/s** | 272.83 | 331.03 | **3.22x** | **3.90x** |
| tg32 steady decode | **15.19 tok/s** | 16.64 | 24.22 | **1.10x** | **1.59x** |

### 2.1 Canonical exact-token screening

All rows use the same four category token arrays at each shape, BF16 K/V,
greedy sampling, disabled prompt reuse, one warmup per case, and three measured
requests per case. Each cell is prompt processing / 128-transition decode in
tokens per second.

| Engine | p512 | p1024 | p4096 | Repeatability |
| --- | ---: | ---: | ---: | --- |
| hipEngine production | **82.51 / 13.82** | **81.22 / 13.79** | **67.93 / 10.40** | 12/12 exact |
| Upstream Vulkan `f1793c1c4` | 240.53 / 22.97 | 259.73 / 20.11 | 266.98 / 18.07 | 12/12 exact |
| Patched-upstream HIP `f1793c1c4` | 239.23 / 17.74 | 301.68 / 16.88 | 294.47 / 14.77 | 12/12 exact; non-stock loader |
| EngramHalo HIP `1423f689` | 234.84 / 17.44 | 314.98 / 17.04 | 381.17 / 15.99 | p512/p1024 exact; `code-p4096` alternates |
| Nathan Vulkan `ad914eb` | 348.31 / 23.23 | 354.93 / 20.36 | 350.54 / 18.44 | 0/12 exact |

#### Starting-point correctness and accuracy contract

This screening measured exact greedy-continuation repeatability. It did **not**
collect logits, KL divergence, task scores, or human accuracy ratings, so it
cannot rank model accuracy. hipEngine's starting accuracy basis remains the
existing BF16-teacher
[`canonical text gate`](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-text-bringup.json),
[`heldout logits gate`](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-heldout-logits.json),
and named-production execution-profile packet (production manifest
`9e27fec0...`, strict manifest `42509601...`). Any arithmetic change must re-run
the applicable gates; a repeat-exact speed row does not replace them.

The measured correctness status is:

- hipEngine production, pristine upstream Vulkan, and patched-upstream HIP are
  repeat-exact on all 12 cases.
- EngramHalo is repeat-exact at p512 and p1024, but `code-p4096` alternates
  between two continuation hashes with the first difference at output index 1
  (the second generated token). Its p4096 rates are diagnostic only.
- Nathan fails repeatability on all 12 cases. A follow-up repeated the identical
  p1024 token array 16 times and produced 16 distinct continuations, often
  diverging at output indices 1–9. All Nathan rates are diagnostic only.
- Full 129-token hashes do not establish cross-engine parity: hipEngine matches
  pristine upstream Vulkan on 1/12 cases, patched-upstream HIP on 0/12, and
  EngramHalo on 1/12. Compounding arithmetic differences make this diagnostic,
  not an accuracy verdict; the logits/KL/task gates remain binding. No external
  lane becomes an accuracy oracle merely by being deterministic or faster.

#### Required upstream HIP loader patch on this host

Pristine upstream HIP `f1793c1c4` produced zero samples after separate
1,800-second starts with default mmap and with `--no-host`. For this exact
111-GB model on `zbook`, the only successful current-upstream HIP configuration
tested applied both documented Strix Halo loader patches:

1. `llama-cpp-25992-rocm-host-buffer.patch`, SHA-256
   `aca70db134d0e65be7a250cf1eb4237bb739d9d586f5c5153ce972372f67b4de`;
2. `llama-cpp-qwen38-per-buffer-mmap.patch`, SHA-256
   `971d428de98ecdf59941946bb391c257e82501ce98b7c71cc1f34803181fe133`.

The combined source diff is
`a37fa3bb64cb693dbe26c98177c757bda60683de2aeb5bae56222bfbfe5783b1` and the
measured HIP server is
`bb41c7555c4ad6cd14df4d2a308d991ddb5d1b44ec21d294fa0fe24f5aeafa86`.
No single-patch ablation was run, so individual necessity is unknown. The
campaign treats the pair as one host/model-scoped startup requirement until an
ablation proves otherwise; this is not a claim that every upstream HIP
installation needs the patches. The result must remain labeled **patched
upstream**; pristine upstream HIP remains `startup_blocked` and has no numeric
row.

#### Decode-depth cliff

The depth loss is larger than for the other hipEngine models in the scoreboard
and is partly specific to Qwen4Exp. Throughput and derived synchronized latency
are:

| Lane | tg128 p512 | tg128 p1024 | tg128 p4096 | Added ms/token, p1024→p4096 | p512→p4096 |
| --- | ---: | ---: | ---: | ---: | ---: |
| hipEngine production | 13.823 | 13.787 | 10.395 | **+23.67** | **-24.8%** |
| Upstream Vulkan | 22.974 | 20.113 | 18.075 | +5.61 | -21.3% |
| Patched-upstream HIP | 17.741 | 16.881 | 14.768 | +8.47 | -16.8% |
| EngramHalo HIP | 17.439 | 17.044 | 15.989 | +3.87 | -8.3% (p4096 diagnostic) |
| Nathan Vulkan | 23.225 | 20.358 | 18.442 | +5.10 | -20.6% (diagnostic) |

For context only, the same-host scoreboards show Qwen3.8-27B Dense at
13.069→13.038 tok/s (-0.2%) and Qwen3.6-35B-A3B at 54.330→54.798 tok/s
(+0.9%) from p512 to p4096. Those are different architectures and are not
comparison denominators, but they show that this is not a generic hipEngine
context penalty.

The target GGUF declares a 2,048-token QSA budget and compression ratio 4, so
`qsa_dense_equivalent_max_tokens == 2051`. p512 and p1024 decode remain on the
dense-equivalent path. p4096 crosses the boundary: each of 12 QSA layers adds
the index-query projection, normalization/RoPE, score over about context/4
pooled blocks, stable top-k expansion, and sparse attention over 2,048 selected
tokens plus the tail. The path transition is confirmed from dispatch and model
geometry. Which sub-role owns hipEngine's excess 15–20 ms/token versus the
external transitions is **not yet measured**; it must be split by a boundary
profile at QSA live counts 2,051/2,052 and at p4096 before optimization.

#### Repeatability-valid performance targets

The repeatability-valid screening targets and current hipEngine gaps are:

| Shape | HIP target and gap | Vulkan target and gap |
| --- | ---: | ---: |
| p512 / tg128 | patched upstream, **2.90x / 1.28x** | upstream, **2.92x / 1.66x** |
| p1024 / tg128 | EngramHalo, **3.88x / 1.24x** | upstream, **3.20x / 1.46x** |
| p4096 / tg128 | patched upstream, **4.34x / 1.42x** | upstream, **3.93x / 1.74x** |

These are screening ratios, not match/loss verdicts. Section 6 still requires
five same-thermal counterbalanced pairs, per-row CV at or below 2% for a match,
and paired confidence intervals. Cross-engine generated-ID equality is
recorded but remains diagnostic for named production arithmetic; each lane's
repeatability and hipEngine's execution-profile gates are separate checks.

### 2.2 Historical external-fork shape refresh

The 2026-08-30 refresh used the existing four-part `UD-Q4_K_XL`; no new
weight quant was downloaded. EngramHalo HEAD remained `1423f689...`. Its
locally built HIP binary also applied the two patches used by the fork's
published container (`#25992` host-buffer workaround and per-buffer mmap).
Nathan's toolbox HEAD was `a8631df...`, its source branch and v0.7.2 payload
were both `ad914eb...`, and a local source build reproduced the release within
1%. Compact evidence:
[`2026-08-30-gfx1151-qwen38-flash-next-external-fork-refresh.json`](../benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-external-fork-refresh.json).

| Same weights, BF16 K/V, `llama-bench` shape | p508 | p1012 | tg32 |
| --- | ---: | ---: | ---: |
| EngramHalo HIP HEAD + documented build patches | **296.12** | **362.72** | **17.62** |
| Nathan Vulkan v0.7.2 payload | **413.04** | **396.25** | **23.85** |
| Nathan Vulkan local build of the same source | 416.41 | 394.57 | 23.95 |

These are **historical shape diagnostics**, not current targets, a source-only
A/B, or a replacement for the frozen role profile. They use generated
`llama-bench` inputs, `-b 8192 -ub 2048 -t 4`, lazy mmap, and fork-recommended backend
settings. hipEngine's p508 row uses the committed text fixture, while its tg32
diagnostic starts from token `9707`; prompt-dependent MoE routing and PLE page
locality make exact-token matching material. P0 therefore adds one durable
cross-engine exact-prompt harness before closure comparisons.

Q8 K/V diagnostics were EngramHalo **301.07/361.75/17.85** and Nathan
**420.99/393.79/23.67** at p508/p1012/tg32. They do not replace the BF16-KV
campaign denominator. Nathan lazy mode partially reproduced its published
mechanism on this 128-GiB host: BF16 p508 averaged **329.23 off vs 413.04 on
(1.255x)**, but the off arm warmed from 271.87 to 396.97 tok/s inside its three
repetitions and p1012 was neutral (**399.67 off vs 396.25 on**). Decode was
also neutral. Cold/warm cache state must remain separate.

An EngramHalo MTP check on the full ten-prompt category+heldout suite reduced
complete request wall from **17.75 to 15.74 s (1.128x)** at 94.55% draft
acceptance, but AR/MTP message hashes matched only **9/10** prompts; the
`general_ja_plan` continuation differed while repeated AR was stable. That MTP
row is a correctness-failing diagnostic, not a speed target. hipEngine must
beat a true same-protocol AR denominator while retaining its own exact/full
profile gate; it must not copy a competitor's invalid speed row.

### Device-kernel windows

| Window | hipEngine kernel sum | llama HIP kernel sum | HIP advantage | hipEngine rows | llama rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| p508 | **5.959 s** | 1.625 s | **3.67x** | 3,328 | 5,119 |
| tg decode per token | **48.63 ms** | 38.90 ms | **1.25x** | 1,860 | 4,203 |

Kernel rows are not host launches. llama's p508 run expands one selected
measured graph to 5,119 kernels; hipEngine submits 2,796 direct launches and
adds 532 copy/fill rows. During decode, hipEngine submits 1,195 direct launches
plus 48 per-layer MoE graph launches per token, while llama submits nearly the
whole transition through 31 large graphs for 32 outputs.

### Prefill module gaps

| Module | hipEngine | llama HIP | HIP advantage |
| --- | ---: | ---: | ---: |
| Selected Q4 gate/up | 1.297 s | 0.477 s | 2.72x |
| Selected Q5_1 down | 1.131 s | 0.345 s | 3.28x |
| Layer-2 Q5_K gate/up | 301.5 ms | 15.4 ms | 19.61x |
| GDN prefill | 634.9 ms | 92.3 ms | 6.88x |
| QSA prefill | 110.5 ms | 13.9 ms | 7.94x |
| Total | 5.959 s | 1.625 s | 3.67x |

MoE owns **3.161 s** of the hipEngine p508 kernel sum; layers 0-26 alone own
**2.526 s**. Layer 2 owns about **397.95 ms**, or roughly **6.6%** of the whole
p508 device window.

### Decode module gaps per token

| Module | hipEngine | llama HIP | HIP advantage |
| --- | ---: | ---: | ---: |
| Dense Q8 | 25.28 ms | 21.84 ms | 1.16x |
| Selected Q4 gate/up | 7.64 ms | 4.60 ms | 1.66x |
| Selected Q5_1 down | 6.26 ms | 2.84 ms | 2.21x |
| GDN recurrence | 2.66 ms | 0.46 ms | 5.72x |
| QSA attention | 0.11 ms | 0.08 ms | 1.31x |
| Total device | 48.63 ms | 38.90 ms | 1.25x |

Decode GR projection/read/elementwise roles own **7.775 ms/token** and expose
up to **387** removable direct launches per token if the operations become
operation-complete. Decode also has a profiled span-minus-kernel gap of
**37.1 ms/token**, so decode has both device-kernel and submission headroom.

### Invalid path removed

The previous "GDN colwarps decode all layers" row is invalid: its selector sat
below the `rows == 1` branch and compared the strict owner with itself. Wiring
the actual candidate costs **6.832 + 0.117 ms/token**, versus **2.454 ms/token**
for the retained serial-column owner, and lowers full decode. Commit
`15a436766` removed the dead route. The corrected evidence is
[`2026-08-30-gfx1151-qwen38-flash-next-gdn-colwarps-decode-all.json`](../benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-gdn-colwarps-decode-all.json).

## 3. Profiling pattern

The committed tools replace the earlier `/tmp` harnesses. Use the same sequence
for every campaign claim.

### 3.1 Freeze identity

1. Record repository, host, power, model, quant, profile manifests, and
   comparator revisions before measuring.
2. Confirm HIP and device visibility:

   ```bash
   python3 -c "import ctypes; ctypes.CDLL('libamdhip64.so'); print('hip OK')"
   rocminfo | grep -E 'Name:|gfx'
   ```

3. Prebuild hipEngine kernels outside the profiler and pass
   `--compiler-version-file` plus `--require-cached-build` to the profiled
   process. Do not let `rocprofv3` spawn `hipcc` or clang children.

### 3.2 Build the comparator once

The canonical 2026-08-30 refresh used current upstream HEAD
`f1793c1c4e586022efa0b1d3aa6e30ccd67f4e2d`. The pristine Release binaries are
HIP `020d0e94...` and Vulkan `c6c9dd2b...`; HIP uses
`AMDGPU_TARGETS=gfx1151`, `GGML_HIP_GRAPHS=ON`, and
`GGML_HIP_MMQ_MFMA=ON`, while Vulkan uses RADV/Mesa 26.2.1 in a separate build
tree. Pristine HIP is startup-blocked for this 111-GB artifact. A separately
labeled HIP binary `bb41c755...` applies only the documented host-buffer and
per-buffer-mmap patches (`aca70db1...` and `971d428d...`). EngramHalo HIP is
`0514f125...` at `1423f689...` plus the same patches; Nathan Vulkan is
`d3dbb492...` at `ad914eb...`. Refresh any comparator only as a separate
baseline event; do not report old and new absolute rows as an optimization A/B.

### 3.3 Collect canonical exact-token wall rows

The committed fixture
[`qwen4exp_canonical_ar_p512_p1024_p4096.json`](../benchmarks/fixtures/qwen4exp_canonical_ar_p512_p1024_p4096.json)
has SHA-256 `42b562bd8e9644bea5b8891c61633dce7f6e75daca64cf79e9cb45c432099da1`.
It derives one code, English, Japanese, and mixed Japanese/English token array
at each canonical length from the ten-prompt MTP bench source. Every engine
consumes those exact IDs. Regenerate it only as an explicit protocol change:

```bash
MODEL_ROOT=/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL
uv run python scripts/qwen4exp_canonical_ar_bench.py fixture \
  --model-root "$MODEL_ROOT" \
  --output benchmarks/fixtures/qwen4exp_canonical_ar_p512_p1024_p4096.json
```

Run hipEngine from a clean worktree after the current compiler cache is warm:

```bash
HIPENGINE_HIP_ARCH=gfx1151 \
uv run python scripts/qwen4exp_canonical_ar_bench.py hipengine \
  --model-root "$MODEL_ROOT" \
  --compiler-version-file /tmp/hipengine-qwen4exp-hipcc-version.txt \
  --require-cached-build \
  --output /tmp/qwen4exp-canonical-hipengine.json
```

Run each llama.cpp-compatible lane with its exact binary/source identity and
backend settings. Arguments that begin with `-` use the `--server-arg=value`
form:

```bash
MODEL_PART="$MODEL_ROOT/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
uv run python scripts/qwen4exp_canonical_ar_bench.py llamacpp \
  --server-bin /path/to/llama-server \
  --source-root /path/to/source \
  --model "$MODEL_PART" \
  --engine-label ENGINE_LABEL \
  --server-arg=-ngl --server-arg=999 \
  --server-arg=-fa --server-arg=on \
  --server-arg=-ctk --server-arg=bf16 \
  --server-arg=-ctv --server-arg=bf16 \
  --server-arg=-c --server-arg=4352 \
  --server-arg=-b --server-arg=8192 \
  --server-arg=-ub --server-arg=2048 \
  --output /tmp/qwen4exp-canonical-ENGINE_LABEL.json \
  --server-log /tmp/qwen4exp-canonical-ENGINE_LABEL.server.log
```

The driver discards one warmup and records three measured requests per case.
The first sampled output belongs to prefill; it requests 129 visible outputs
from llama.cpp and reports exactly 128 post-first-output transitions, matching
hipEngine's 128 timed `runner.step()` calls. Prompt cache is disabled while OS
page-cache state is warm. Synthetic `llama-bench` p512/p1024/p4096/tg128 rows
remain a separate `shape_only` diagnostic class.

### 3.4 Collect role-resolved device traces

The frozen role ledger remains on the historical p508 fixture until P4 emits
its required fresh p512 ledger. hipEngine p508:

```bash
TRACE=/tmp/qwen4exp-role-p508
rm -rf "$TRACE" && mkdir -p "$TRACE"
HIPENGINE_HIP_ARCH=gfx1151 \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-qwen4exp-hipcc-version.txt \
HIPENGINE_REQUIRE_CACHED_BUILD=1 \
rocprofv3 --kernel-trace --hip-trace --marker-trace --output-format csv \
  -d "$TRACE" -o role-p508 -- \
  uv run python scripts/qwen4exp_profile_gap.py \
    --model-root "$MODEL_ROOT" \
    --mode prefill \
    --prompt-file benchmarks/prompts/qwen4exp-p508.txt \
    --expected-prompt-tokens 508 \
    --profile --role-markers --repetitions 1 \
    --output "$TRACE/child.json"
```

hipEngine decode uses the same driver in decode mode, normally 8 warmup steps
and 16 measured steps with `max_sequence_length=128` and `prefill_chunk_size=256`.
Do not treat the profiled wall as a speed claim; use the unprofiled wall rows for
that.

llama.cpp HIP traces use `rocprofv3 --kernel-trace --hip-trace
--hip-graph-trace --memory-copy-trace --stats` around `llama-bench -r 1`, then
select the measured graph/window. Record raw CSV hashes in the artifact.

### 3.5 Analyze without conflating rows and launches

```bash
uv run python scripts/qwen4exp_trace_analyze.py \
  --trace-dir "$TRACE" --engine hipengine \
  --marker-prefix qwen4exp_prefill_p508_ \
  --output "$TRACE/summary.json"

uv run python scripts/qwen4exp_role_analyze.py \
  --trace-dir "$TRACE" \
  --measure-prefix qwen4exp_prefill_p508_ \
  --output "$TRACE/roles.json"
```

The analyzer reports the selected marker window, kernel sum/span, row counts,
family totals, HIP API launch correlations, unmatched graph/copy rows, and
memory-copy rows. The role analyzer correlates ROCTX ranges to HIP launch
correlation IDs and then to kernel rows. `scripts/qwen4exp_perf_gap_report.py`
renders the compact artifact as markdown tables:

```bash
uv run python scripts/qwen4exp_perf_gap_report.py \
  benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-fresh-full-profile.json
```

Interpretation rules:

- End-to-end unprofiled wall is the headline.
- Kernel sum ranks device dataflow.
- Kernel span minus sum exposes gaps, but profiler inflation is not Python
  overhead without a separate unprofiled event/control.
- Direct `hipLaunchKernel` correlations count host submissions; graph-expanded
  kernels and copy/fill kernels appear as trace rows without direct launches.
- A/B decisions use same-session counterbalanced orders and identical IDs or
  the applicable production-profile gate.

## 4. External evidence review

This section records useful external hypotheses and their evidentiary status.
None of these numbers are hipEngine results.

| Source | Mechanism or claim | Status for this campaign |
| --- | --- | --- |
| [Sleeping Robots, 2026-08-29](https://sleepingrobots.com/dreams/engramhalo-qwen38-flash-next-strix-halo/) | Independently tested EngramHalo on Strix Halo with a different quant. MTP reaches 28-38 tok/s at working depths; 26K MTP regresses to 15.0; kernel-only prefill improves up to about 35% at 26K. | Useful cross-check of the direction, not a same-quant baseline. Confidence: medium-high for the external fork, low for transfer magnitude. |
| Local source/build refresh, 2026-08-30 | Existing `UD-Q4_K_XL` runs in both forks. EngramHalo BF16 reaches 296.12/362.72/17.62 and Nathan v0.7.2 reaches 413.04/396.25/23.85 at p508/p1012/tg32. Nathan lazy-on is 1.255x over the cache-cold-to-warm off average at p508 and neutral by p1012. Engram MTP is 1.128x complete-wall but only 9/10 AR-message exact. | Historical same-host shape evidence, superseded for AR targets by the exact-token screening in section 2.1. The MTP speed row fails correctness and remains diagnostic only. |
| [Aristo94/EngramHalo.cpp](https://github.com/Aristo94/EngramHalo.cpp), refreshed at `1423f689986f670417128fd545a0aa1241166103` | Wide radix top-k (`33766da`), masked-slice FA skip (`bf8412d`), QSA top-k row gather (`2606d49`), MTP sidecar (`afb80ed` + `2ba3009`), PLE lazy row prefetch (`c911e6b`), and load-page drop-behind (`5486559`). Chunked GDN prefill exists (`62160a7`) but was explicitly not active in the published numbers. The published container additionally applies the tracked #25992 host-buffer and per-buffer-mmap patches. | Code and build mechanisms verified by source inspection and a local gfx1151 HIP build. hipEngine already covers the QSA selector/gather direction; PLE advice/prefetch, loader drop-behind, full-step graphing, and MTP economics remain open. |
| [Nathanw1014/strix-halo-llamacpp v0.7.2](https://github.com/Nathanw1014/strix-halo-llamacpp/releases/tag/v0.7.2), toolbox HEAD `a8631dfbf0aeb6a4004866fce1fd7e5c10370049`, source `ad914eb6587d3da8b2bf50f0056cc20b3d3e91f5` | `TENSOR_READ_LAZY` + `MADV_RANDOM` alone loses; merged `WILLNEED` row prefetch is the paying half (`77362a8`). Qwen4Exp also adds host PLE gather and reusable decode topology (`631b9ff`), per-block QSA bias (`024b7ad`), and MTP graph/context (`3543908` + `39817c4`). Vulkan lineage includes MoE row lists (`212cca8`), route-scale epilogue, SiLU/mul fusion, transposed concat (`30d8bb0`), dense wave32 (`25c45fe`), and LDS padding (`baf6360`). | Source mechanisms verified; release and local source builds agree within 1% on this host. Vulkan shader topology is not portable to HIP, but the removed data movement, host synchronization, graph rebuild, and LDS-bank mechanisms are actionable. |
| [quimmedes/cafe-llama.cpp](https://github.com/quimmedes/cafe-llama.cpp), observed HEAD `2da84198eccb0aee59abba59e967dcc61f84ce07` | The fresh fork exposes pinned-host/CPU routed-expert placement, PLE n-gram SSD mmap or disable modes, and Qwen4Exp MTP trunk/combiner fixes. Commits `ba7bd23` and `7ee981d` add the PLE controls; `19aefd2` and `d98dc18` address MTP hidden export and mixer mapping. | Track as a source lead, not a measured comparator. SSD PLE and host-placement ownership may inform P9; `--no-ngram` changes the model and cannot close parity. No same-weight local rate or correctness packet has been verified. Confidence: high for repository/commit identity, medium for transfer applicability. |
| [omlx PR #3260](https://github.com/jundot/omlx/pull/3260), open head `3343e4414f75b9808d2d8a6de1950ad96ce8dac8` | Adds row-addressable SSD expert reads, fixed preallocated expert banks, manifest pins plus an evictable hot tier, learned route-frequency hotlists, expert-major overflow chunks, and checked/speculative miss handling with transactional KV/SSM restore. The author reports exact expert output at a 0% substitution threshold. | Track the fixed-bank, telemetry, hotlist, and transactional retry mechanisms for constrained-residency work. This is an unmerged Apple MLX/safetensors path with a dirty merge state, not direct HIP/GGUF code or local performance evidence. Confidence: high for PR state/design, low for transfer magnitude. |
| Upstream llama.cpp [#27742](https://github.com/ggml-org/llama.cpp/pull/27742) | Qwen4Exp architecture support; merged at `6c84c7d5`. | Already represented in the fresh comparator. |
| Upstream [#27794](https://github.com/ggml-org/llama.cpp/pull/27794) | `TENSOR_READ_LAZY` plumbing; merged at `fac889fb`. Nathan's branch keeps the missing batched row-prefetch half. | Useful PLE hypothesis. |
| Upstream [#27836](https://github.com/ggml-org/llama.cpp/pull/27836) | Qwen4Exp NextN/MTP draft head; open. Its key note is that the hyper-connection combiner must run per stream; mean pooling first destroys acceptance. | Matches our retained lesson; use it to audit, not re-derive, the Qwen4Exp MTP combiner. |
| Upstream [#26592](https://github.com/ggml-org/llama.cpp/pull/26592) and [#26388](https://github.com/ggml-org/llama.cpp/pull/26388) | hipCUB/CUB paths for top-k/argsort on HIP. | Superseded for our purposes by the in-tree GPU QSA selector; keep as lineage context only. |
| Upstream [#27466](https://github.com/ggml-org/llama.cpp/pull/27466) | ROCm radix top-k for long rows; open. | Confirms the long-context failure mode; hipEngine already uses a device selector. |
| Upstream [#26001](https://github.com/ggml-org/llama.cpp/pull/26001) | Chunked GDN prefill using tensor-core fragments; CUDA/NVIDIA-focused and open. | Hypothesis only. Our nearest unit is the retained colwarps owner plus the rejected decode correction. |
| Nathan's Vulkan evidence pack and branches | q8 KV dequant-once, contiguous K/V, MoE row-list prepass, scale epilogue, SiLU/mul fusion, concat transpose. | Vulkan/RADV-first; useful patterns, not HIP evidence. Map only the algorithmic dataflow, not shader specifics. |
| Upstream [#25494](https://github.com/ggml-org/llama.cpp/pull/25494) | Vulkan q8_0 KV dequant-once for prefill; merged at `dc72703f`. | Not directly applicable to HIP; reinforces "dequantize/reorganize once, then attend". |
| Upstream [#26419](https://github.com/ggml-org/llama.cpp/pull/26419) | MMA FlashAttention at head-dim 256 on RDNA; open. | Relevant to QSA prefill geometry, but measured on RDNA4, not gfx1151. |
| Upstream [#27880](https://github.com/ggml-org/llama.cpp/pull/27880) | qwen4exp graph-split reduction; merged at `6fe74980`. | Already in the remote-HEAD comparator. |
| Upstream [#27925](https://github.com/ggml-org/llama.cpp/pull/27925) and [#26686](https://github.com/ggml-org/llama.cpp/pull/26686) | Vulkan MoE padding/row-ID changes that improve the Vulkan comparator. | Already in the remote-HEAD Vulkan comparator; no HIP action implied. |

### 4.1 Mechanism transfer audit

The useful part of the external forks is the mechanism, not their headline rates.
This is how each mechanism maps to the current hipEngine implementation:

| Mechanism | Current hipEngine state | Campaign action |
| --- | --- | --- |
| Wide, graph-safe QSA radix top-k | Exact stable four-pass radix selection already lives in `qwen4_exp_qsa.hip`; no host sort is used. | No port. Keep it as the selector oracle and re-profile only its real long-context score width. |
| Persistent pooled QSA keys and per-block bias | Complete four-token blocks are pooled once into a persistent device buffer; selection is block-based rather than one bias value per KV cell. | Treat the Nathan fixes as corroboration. Audit metadata copies and append work, not the already-solved algorithm. |
| Gather only selected QSA K/V rows | Paged sparse QSA consumes explicit selected positions/counts rather than scanning a dense mask. | Audit selected-row count, sort, page locality, and head-dim-256 geometry at 16K+; do not rebuild llama's graph gather. |
| Host-side PLE row gather | `Qwen4ExpPLEMMapTable` gathers/dequantizes only requested rows, and a pinned two-buffer ring stages them. It still creates temporary arrays and performs synchronous gather/copy. | Preserve ownership, but add direct-to-ring dequantization, duplicate/page coalescing, telemetry, and prefill overlap in P9. |
| PLE random advice plus merged row prefetch | No equivalent `MADV_RANDOM` + page-aligned merged `WILLNEED` pair exists. | Implement both halves together with off/auto/on rollback. Never ship random advice alone; Nathan measured that half losing. |
| Load-page drop-behind | Hot weights are uploaded without an explicit per-tensor file-page release policy. | Measure transient free/available/swap/load wall, then add bounded unmap/fadvise only for copied tensors; never drop lazy PLE pages. |
| Fixed expert bank plus route hotlist | Current Qwen4Exp keeps the declared routed-expert representation resident; it has no SSD miss/promotion tier. | Keep omlx PR #3260 as a constrained-memory design lead. Any future port needs GGUF row indexing, exact 0%-substitution routing, request-safe bank ownership, transactional state restore, and cold/warm workload gates; it is not part of short-AR parity. |
| MoE row-list prepass | Device count/prefix/scatter and expert-sorted lanes already exist. | Do not port Vulkan topology. Remove the remaining D2H tile-count synchronization and the Q8 expert-start D2H + Python expert loop; use guarded fixed-capacity device grids. |
| Route-scale/weighted-down epilogue | c1 Q5 down can fuse ordered weighted sum; grouped prefill still writes expert outputs and runs a separate weighted lane reduction. | Add exact/T2 grouped down+route-weight+scatter/ordered-reduce candidates, with the current chain registered as fallback. |
| Shared-expert and router completion | Shared gate/up, SiLU, down, gate projection, cast, and combine remain separate; router writes all 512 logits before top-10. | P3 owns operation-complete shared-expert and exact stable router+top-10 candidates after role measurement. |
| Dense wave32 retile and LDS padding | gfx1151 HIP kernels are wave32-native, but each quant/shape has independent register/LDS behavior. Vulkan constants do not transfer. | Sweep actual rotating-weight shapes with compiler VGPR/LDS/scratch evidence. Port the bank-conflict/resource hypothesis, not constants. |
| Tiled transpose and SiLU/mul fusion | Exact bulk Conv and selected gate/up+SiLU cover analogous paths, but GDN state layout and shared-expert SiLU still expose traffic. | Use the transpose hypothesis in P7 and the activation hypothesis in P3/P6 only where a current trace names the traffic. |
| Reusable decode topology | 48 stateless per-layer MoE graphs are reused; the other 1,195 direct launches/token remain outside them. | P8 grows from one stateful layer to one transition and then a full step, with pointer/state/rollback/replay gates at every rung. |
| Device-owned decode boundary | Qwen4Exp still synchronizes, copies the full vocabulary logits to the host, and computes argmax on the CPU every token. | P5 adds device argmax/sampling and copies only the token plus requested compact telemetry; full logits remain an explicit API path. |
| Per-stream MTP combiner and graph | The combiner is correct, but draft hidden/logits and target full logits cross the host; target verification is serial. | Keep the combiner. P11 makes proposal, hidden chaining, verification, acceptance, commit, and rollback device-resident before budget tuning. |
| q8_0 K/V, `-ub 2048`, and hipBLASLt | These are different llama.cpp representation/config knobs. Current hipEngine is BF16 K/V at chunk 512. | Chunk size is a P0 same-representation sweep. Q8 K/V remains a separately gated T3 profile after BF16 AR parity. |
| Quantized-KV dequant-once/contiguization | There is no quantized-QSA-KV owner in the binding campaign. | Backend-disjoint evidence only until P10; it cannot close a BF16-KV milestone. |
| Fully masked FA slice skip | Current sparse attention no longer scans a dense selected-token mask, while short prefill has separate dense flash geometry. | Test only against a trace-proven masked slice in P4/P10; reject if it optimizes work hipEngine does not execute. |
| Chunked GDN prefill | hipEngine has strict serial/prepare+peer/column-warp owners; Engram's chunked kernel was not active in its published rows. | Treat it as a design hypothesis in P4. Require local arithmetic classification, state parity, and whole-role evidence. |

## 5. Plan

The phase order is the default priority, not permission to keep working a
low-Amdahl idea after the profile changes. After every retained unit, collect a
fresh role/launch/copy census and re-rank the remaining work. If a phase is
blocked, record the concrete blocker and continue the highest-value independent
phase; a blocker is not campaign closure.

### 5.1 Definition of done for one optimization unit

Every code or kernel unit follows the same loop:

1. Name the measured owner, arithmetic class (T0-T3), affected layers/shapes,
   Amdahl ceiling, expected mechanism, and registered strict fallback.
2. Add the RED oracle before implementation. For a port, run
   `scripts/check_lineage.py`, inspect source drift, and cite source path plus
   commit.
3. Microbenchmark actual immutable weights/shapes. Rotate more than 64 MiB of
   weights for c1 tests so MALL does not fake DRAM throughput. Use warmups plus
   counterbalanced pairs and record grid, waves, VGPR, LDS, scratch, spills,
   duration, and effective bandwidth/throughput.
4. Trace the expected kernel name with `rocprofv3`; a fast helper that is not
   selected is not evidence.
5. Run the complete strict or production numerical/control/task/lifecycle gate
   before promotion, then same-session p512/p1024/p4096 and context-conditioned
   tg128 wall measurements.
6. Retain an exact non-regression or a fully gated production win; otherwise
   reject it and restore the registered incumbent. Do not tune the same family
   again after two measured losses without a new profile or mechanism.
7. Emit the compact accepted/rejected/blocked artifact, update rollups/catalogs,
   write the immutable worklog entry, and commit the validated unit immediately.

### Phase P0 — binding comparator and measurement contract

Goal: turn the shape-only external refresh into one exact, repeatable target
matrix before implementation claims parity.

- [x] Promote the wall/profile driver, trace analyzers, role attributor, sync
      diagnostic, report generator, and historical p508 fixture.
- [x] Refresh/build EngramHalo HIP and Nathan Vulkan at exact identities; retain
      same-weight BF16/Q8 shape diagnostics and the MTP correctness failure.
- [x] Commit the category-balanced p512/p1024/p4096 fixture and cross-engine
      driver that feed identical token IDs, sampler, K/V type, prompt-cache
      state, and 128-transition output horizon to hipEngine and every comparator.
      Keep generated `llama-bench` rows in a separate `shape_only` class.
- [x] Add at least one prompt from every code/general-English/general-Japanese/
      mixed category to the AR wall packet so one routed prompt cannot define
      parity. Category-heldout expansion remains a closure gate.
- [x] Freeze the current three-repeat BF16 screening target per row for
      upstream llama.cpp, EngramHalo HIP, and Nathan Vulkan, with deterministic
      rows separated from non-binding diagnostics. Refresh a comparator only as
      a separate baseline event with old and new binaries measured on the same
      host. A five-pair section-6 closure refresh remains open.
- [x] Record hostname/machine ID, source and binary hashes, compiler/driver,
      profile manifest, model-part hashes, exact command, CPU governor/TuneD,
      `amd_iommu`, power/clock samples, free/available/swap, and active GPU
      processes in the canonical screening artifact. Repeat the capture for the
      eventual closure artifact.
- [ ] Add explicit warm-page-cache and isolated cold-PLE modes. Never average
      or compare them as one workload.
- [ ] Sweep hipEngine prompt chunk 256/512/1024 (and 2048 where the prompt
      permits) at p512/p1024/p4096 with memory and correctness controls; select
      by model evidence rather than copying an external `ubatch` value.
- [ ] Extend the gap report to carry per-layer role time, direct/graph launch
      APIs, blocking/async copies and bytes, synchronizations, compiler resource
      data, and unresolved wall-minus-device time.

### Phase P1 — layer 2 and the Q8 expert-down family

Goal: remove the largest isolated miss and replace the only host-driven grouped
expert path.

The frozen MoE map is 43 layers of Q4_K/Q4_K/Q5_1, layer 2 of
Q5_K/Q5_K/Q8_0, and layers 4/30/46/47 of Q4_K/Q4_K/Q8_0.

- [ ] Commit a generated quant/shape/owner inventory test so artifact drift
      fails before timing.
- [ ] Write actual-weight RED fixtures for layer-2 Q5_K dual gate/up and Q8_0
      down, including compact row maps, empty experts, tails, and route order.
- [ ] Route the existing selected Q5_K WMMA body for layer 2; classify its
      arithmetic before timing and preserve the strict selected chain.
- [ ] Replace the Q8 path's `group_expert_start` D2H copy and Python loop over
      512 experts with a device-driven grouped Q8 owner. Use a fixed-capacity
      grid guarded by device counts or an equivalent no-host-roundtrip design.
- [ ] Extend the proven Q8 owner to layers 4/30/46/47 when their independent
      actual-shape and composition gates pass; do not hardcode only layer 2.
- [ ] Fuse route scaling/ordered accumulation into Q8 down only if the declared
      strict/T2 contract passes.
- [ ] Run the complete 450-row/three-repeat packet, tasks, physical c2,
      lifecycle, paired p512/p1024, and the canonical p4096 gate. Bind only
      certified scopes.

Expected evidence: layer 2 falls from about 397.95 ms toward the comparator
role range; its maximum standalone p508 contribution is about 6.6%.

### Phase P2 — early routed MoE layers 0-26

Goal: attack the **2.526 s** early-MoE owner without repeating the failed broad
WMMA suffix experiment.

- [ ] Split the role into Q4 gate/up, activation, Q5_1/Q8 down, route-weight
      reduction, and host synchronization by layer and actual active-row count.
- [ ] Remove the per-layer `group_wmma_total` stream sync/D2H read. Launch a
      safe maximum tile grid with a device count guard, or prove a different
      device-only submission scheme.
- [ ] Optimize T0 exact association first: physical-lane contraction,
      multi-row weight reuse, coalesced metadata, and output grouping while
      preserving the strict reduction/publication tree.
- [ ] Add operation-complete grouped dual gate/up+SiLU and
      down+route-weight+scatter/ordered-reduce candidates. Keep primitive
      chains registered.
- [ ] Sweep wave32 ownership, workgroup size, row/output tiles, and LDS padding
      on rotating actual weights. Use Nathan's wave32/bank-conflict findings as
      hypotheses, never as transferable constants.
- [ ] Test T1/T2 WMMA only in independently calibrated layer clusters. Every
      candidate must pass category, shape, transition, repeat, task, c2, BF16-
      relative, lifecycle, and manifest gates; no final-prompt or one-layer
      screen can promote it.
- [ ] Re-profile after each retained cluster and stop widening when the next
      boundary fails. Layers that fail remain on strict owners.

### Phase P3 — shared expert, router, dense projections, and GR prefill

Goal: account for the large non-routed prefill remainder that the earlier plan
left without an implementation phase.

- [ ] Re-profile and separately name router, shared gate/up/down, attention and
      FFN GR reads/writes, `attn_qkv`, `attn_gate`, `ssm_out`, QSA projections,
      casts, and elementwise tails.
- [ ] Evaluate an exact F32 router+stable-top-10 owner so 512 router logits are
      not written and reread when the public path only needs deterministic
      routing. Keep the full-logit primitive for diagnostics.
- [ ] Fuse shared gate/up+SiLU, then shared down+sigmoid gate+combine, preserving
      F32/BF16 boundaries and the strict shared-expert chain.
- [ ] Fuse GR grouped RMSNorm + unequal down/inject where ownership permits;
      add down+scaled-SiLU and up+sigmoid+gated-mean epilogues.
- [ ] Evaluate output-projection+GR-write composites for attention and MoE
      boundaries, including the exact inject ordering.
- [ ] Extend dense Q8 MMQ/WMMA scopes earlier only through the complete
      production packet. Optimize exact coltile/rowbatch fallbacks for layers
      that reject changed arithmetic.
- [ ] Require each retained subunit to reduce its complete role and p512/p1024,
      not merely an isolated GEMM; re-run p4096 at the phase gate.

### Phase P4 — GDN and QSA prefill parity

Goal: close the remaining **634.94 vs 92.34 ms** GDN and
**110.49 vs 13.91 ms** QSA gaps after P1-P3 are stacked.

- [ ] Re-profile prepare, recurrence, norm/gate tail, projection, KV/index
      append, selection, attention, and output roles by admitted layer scope.
- [ ] For GDN, test exact early-layer column ownership, prepare+recurrence
      fusion, state residency, and bounded chunking. Engram's chunked kernel is
      a design reference only; it was not active in the fork's published rows.
- [ ] Evaluate T1/T2 GDN suffix widening only with fresh all-category boundary
      packets; keep every rejected early layer strict.
- [ ] For QSA, compare current key-parallel head-dim-256 flash geometry with the
      selected llama kernel family, including key tiles, online-softmax merge,
      grid sufficiency, and register/LDS pressure.
- [ ] Confirm selected-position attention already removes dense-mask work.
      Evaluate fully-masked-slice skipping only if a current trace proves such
      slices still execute.
- [ ] Publish a fresh p512 device-role ledger after P4; no prefill phase may be
      called closed with an unexplained multi-x owner.

### Phase P5 — device-owned AR output boundary

Goal: remove full-vocabulary host readback and synchronization before enlarging
graph scope.

- [ ] Add a registered device argmax/greedy sampler after the lm head and copy
      only the token ID plus explicitly requested compact telemetry to the host.
- [ ] Keep full logits/probabilities as an explicit API/debug path; do not
      silently change public response semantics.
- [ ] Feed the device-owned token into the next embedding path where possible;
      PLE hashing may consume one compact host token without a 248K-logit copy.
- [ ] Reduce the current 28 blocking and 12 async memcpy calls/token to a
      role-explained minimum and record bytes/directions, synchronization count,
      first-token, steady-state, and exact-ID/logit controls.
- [ ] Re-run natural multi-prompt decode, not only the repeated `9707` steady
      diagnostic.

### Phase P6 — decode GR, MoE, and dense operation completion

Goal: reduce the **48.63 vs 38.90 ms/token** device gap and the remaining
direct-launch surface before graph capture hides it.

- [ ] Stack GR down+inject, down+scaled-SiLU, up+sigmoid+gated-mean, and GR-write
      composites one at a time with exact/T1/T2 declarations.
- [ ] Re-rank dense Q8, selected Q4 gate/up, selected Q5_1/Q8 down, shared
      expert, router, QSA, and lm-head kernels after P5.
- [ ] Profile decode immediately below/above the QSA transition (live counts
      2,051/2,052) and at p4096. Split index-query projection,
      normalization/RoPE, score/top-k, selected attention, graph/submission,
      copies, and wall gaps. Explain or remove hipEngine's measured
      +23.67 ms/token p1024→p4096 cost before claiming short-AR parity.
- [ ] Tune Q4/Q5/Q8 c1 owners on rotating actual weights for coalescing,
      physical-lane contraction, occupancy, and operation-complete epilogues;
      do not force WMMA onto M=1.
- [ ] Preserve the exact fused Q5 weighted-down and Q4 dual+SiLU fallbacks;
      replace them only with same-role evidence.
- [ ] After every retained fusion, update direct launches, graph launches,
      kernel rows, API time, copy bytes, and context-conditioned tg128 wall.

### Phase P7 — normalized/transposed GDN decode

Goal: replace the measured **2.659 vs 0.465 ms/token** recurrence with a
c1-shaped layout without reviving the rejected prefill-colwarps route.

- [ ] Normalize Q/K once per head instead of once per output column.
- [ ] Keep recurrent state transposed in the decode-native layout across steps;
      define exact construction, snapshot, rollback, reset, and strict
      conversion boundaries.
- [ ] Port the relevant llama four-warp decode dataflow, not its prefill body or
      constants.
- [ ] Prove CPU-reference state/output parity on reduced and actual fixtures,
      repeated steps, restore/replay, cancellation, and c2 isolation.
- [ ] Require expected kernel trace, complete profile packet, and canonical
      tg128 win. Never re-enable the invalid `GDN_COLWARPS_DECODE_LAYERS` route.

### Phase P8 — state-safe whole-transition graph

Goal: contract 48 small MoE graphs plus 1,195 direct launches/token toward one
request-owned transition submission.

- [ ] Reproduce and localize the historical third-replay state corruption
      before changing capture scope.
- [ ] Capture in rungs: one stateful layer, one complete attention/FFN
      transition, a multi-layer segment, then the full token step.
- [ ] Keep token/PLE input buffers and every weight/state/scratch pointer stable;
      include profile-manifest hash, shape, context bucket, and fallback in the
      graph key.
- [ ] Gate GDN, QSA K/V/index append, PLE history, sampler output, snapshot,
      rollback, reset, cancellation, c2 isolation, teardown, and at least three
      consecutive replays at every rung.
- [ ] Compare direct/graph API time, graph build/reuse counts, kernel rows/span,
      compact copies, first-token latency, and context-conditioned tg128. A graph
      that merely hides a slower kernel chain is not a retained win.
- [ ] Target no per-layer graph launches and no unexplained direct launch in the
      steady transition; document any irreducible boundary.

### Phase P9 — PLE mmap, cold-cache, and load-memory lane

Goal: implement the reproduced Nathan/Engram mechanism without mixing it into
warm GPU-kernel claims.

- [ ] Instrument `Qwen4ExpPLEMMapTable` and staging with requested/unique rows,
      unique/adjacent pages, bytes, prefetch ranges, faults or resident-page
      proxy, dequant/copy/H2D wall, and cache mode.
- [ ] Add off/auto/on random-access advice only together with page-aligned,
      deduplicated and adjacent-range-merged `WILLNEED` prefetch.
- [ ] Dequantize directly into the active pinned ring where practical; remove
      temporary gather/value arrays and redundant copies.
- [ ] For prefill, overlap next-chunk prefetch/dequant with current GPU work
      using the existing two-buffer ownership plus explicit event/thread
      lifetime. Decode remains demand-driven unless a real lookahead exists.
- [ ] Add a safe isolated cold-cache protocol and a warm steady protocol. Do
      not use one process's warming repetitions as independent samples.
- [ ] Measure lazy on/off, cold/warm p512/p1024/p4096 and tg128, page reads,
      RSS/file versus anonymous memory, available/free/swap, and exact output
      hashes.
- [ ] Add optional per-tensor load drop-behind only for data already copied to
      device ownership. Never invalidate lazy PLE pages or validation readers;
      include reload-heavy and one-shot serving controls.

### Phase P10 — long-context AR and optional KV profile

Goal: close depth-dependent competitor gaps only after short BF16 AR parity and
the existing escalation thresholds permit each rung.

- [ ] Re-run natural 4K, 16K, then 64K retrieval with the current stacked
      production path and exact selected-position/CPU selector controls.
- [ ] Profile QSA score/top-k O(context/4), selected-K/V page locality, sparse
      attention, PLE I/O, graph reuse, and KV bytes separately at each depth.
- [ ] Compare exact matched same-weight/BF16-KV Engram/Nathan/upstream rows; do
      not compare against their IQ3/IQ4 or quantized-KV headlines.
- [ ] Optimize persistent compressed-key scoring, top-k, selected attention,
      and graph/context buckets only where the depth profile ranks them.
- [ ] After BF16 AR parity, open Q8 K/V as a T3 product configuration with its
      own CPU/reference, BF16-relative, task/retrieval, memory, deterministic,
      lifecycle, and same-config competitor gates. It cannot close BF16 parity.

### Phase P11 — device-resident MTP economics

Goal: exceed true AR on the full suite, then match only correctness-valid
external MTP rows.

- [ ] Reconfirm the per-stream hyper-connection combiner and sidecar tensor map;
      do not replace it with mean pooling.
- [ ] Add phase timing/census for target hidden export, draft input fusion,
      draft layer/head, sampler, target verify, acceptance, commit/rollback,
      copies, and graphs on every category.
- [ ] Keep target hidden, draft hidden chaining, logits/top-k, and candidate IDs
      on device. Remove per-draft full-logit/hidden D2H and host reconstruction;
      read one compact candidate packet per cycle at most.
- [ ] Build the rows<=8 batch-invariant target verifier before raising budget.
      Its per-row decode arithmetic, GDN/QSA/PLE state, and outputs must equal
      serial target verification under the declared contract.
- [ ] Move acceptance, first-mismatch selection, commit, rollback, and cursor
      repair to device-owned transactional kernels/graphs with exact recovery
      and cancellation tests.
- [ ] Sweep budgets 1-6 on the full category+heldout suite against a true
      no-MTP AR denominator from the same command. Record acceptance, visible
      tokens/cycle, target rows, phase wall, and speed by category and context.
- [ ] Evaluate confidence thresholds and `ngram-mod` combination only as
      explicit provider policies over the full suite. No fixed prompt, token,
      or candidate-specific policy is retainable.
- [ ] Require exact greedy outputs where that is the provider contract. The
      refreshed Engram row is 1.128x but only 9/10 exact and is therefore not a
      valid target or promotion precedent.
- [ ] Promote at >=1.0x AR with every binding category non-regressive; continue
      toward >=1.5x and the best correctness-valid same-config competitor.

### Phase P12 — final refresh, cleanup, and rollup

Goal: make the complete result reproducible, default, and reversible.

- [ ] Refresh all comparator lanes once under P0's exact protocol and fixed host
      state; do not compare old absolute rows to new binaries.
- [ ] Run the final strict/production packet, task/BF16/control/state/c2/
      lifecycle gates, exact matched p512/p1024/p4096 with tg128, and every
      unlocked long-context/MTP milestone.
- [ ] Emit compact accepted/rejected/blocked artifacts, raw-log hashes, generated
      reports, benchmark README/changelog updates, and the model checkpoint.
- [ ] Update `docs/KERNELS.md`, lineage metadata, `docs/REFACTOR.md`, and
      `docs/PLAN.md` if architecture moved.
- [ ] Remove superseded experiment flags, dead selectors, duplicate fallback
      chains, and stale graph routes only after their replacements are default.
- [ ] Commit every validated logical unit; campaign closure is its own final
      decision/worklog commit.

## 6. Acceptance and closure

All comparisons bind to the same physical host, exact target parts, weight
quant, K/V type, prompt/token fixture, cache mode, sampler, output horizon,
power policy, and declared profile. A different representation gets a separate
milestone and denominator.

### 6.1 Match and beat rules

- Screening may use three counterbalanced repetitions. Closure uses at least
  five counterbalanced pairs after warmup, reports every sample and coefficient
  of variation, and keeps competitor/hipEngine runs in the same thermal window.
- **Beat** means the paired 95% confidence interval for the hipEngine/comparator
  throughput ratio is above 1.0.
- **Match** means the median ratio is at least 0.98, both arms have coefficient
  of variation at most 2%, and the paired 95% interval includes 1.0. A noisy
  result is neither a match nor a loss; stabilize and repeat.
- p512, p1024, p4096, and context-conditioned tg128 bind independently. A large
  prefill win cannot average away a decode loss, and one category cannot
  compensate for another.
- End-to-end unprofiled wall is binding. Kernel/sub-window/launch/copy wins are
  retainable evidence but do not by themselves close parity.

### 6.2 Milestones

1. **P0 measurement closure:** exact matched cross-engine harness, frozen source
   and binary identities, host state, cold/warm separation, and category
   heldouts are committed.
2. **HIP short-AR parity:** named hipEngine production matches or beats the best
   refreshed same-configuration HIP comparator on p512, p1024, p4096, and the
   corresponding tg128 rows.
3. **Same-host short-AR parity:** the same path matches or beats the best
   refreshed same-configuration HIP or Vulkan engine on every canonical row.
   Vulkan is a required second milestone, not an optional source of ideas.
4. **Role closure:** no layer-2/early-MoE, shared/router/GR, GDN, QSA, output,
   submission, copy, or synchronization bucket remains an unexplained multi-x
   outlier. Any residual >=1.25x names a measured architectural reason and an
   exhausted or blocked action.
5. **Correctness/product closure:** production manifest and strict fallbacks,
   complete numerical/BF16/task/category packet, same-schedule deterministic
   repeats, graph/eager and restore/replay, physical c2/isolation, cancellation,
   memory, and teardown all pass.
6. **Long-context parity:** every context rung unlocked by the benchmark ladder
   passes retrieval/control gates and matches the best exact same-config
   comparator at that depth. Historical or different-KV rows do not bind.
7. **MTP parity:** the full category+heldout suite is exact under the provider
   contract and non-regressive in every binding category versus true AR. It
   reaches at least 1.5x AR or matches/beats the best correctness-valid
   same-configuration external MTP row, whichever is higher. A competitor row
   that fails output equivalence is not a target.
8. **Evidence closure:** the final artifact contains exact commands,
   source/binary/model/manifest hashes, raw trace/result hashes, role and
   launch/API/copy census, host state, all correctness verdicts, and a generated
   report from `scripts/qwen4exp_perf_gap_report.py`. Rollups, catalog, refactor
   ledger, worklog, and atomic commits are current.

A final deliberate comparator refresh freezes the closure target. A later
external commit does not retroactively invalidate the artifact; it is a new
baseline event and, if desired, a new campaign.

## 7. Copyable coder goal

```text
Execute docs/QWEN3.8-FLASH-NEXT-PERFORMANCE-CAMPAIGN.md as the active
performance authority for Qwen3.8-Flash-Next on zbook/gfx1151. Keep the pinned
Unsloth UD-Q4_K_XL target weights and BF16 K/V as the binding AR configuration.
Do not obtain or substitute a new weight quant. Treat Q8 K/V and MTP as separate
product configurations with their own gates.

Do not declare success after a microbenchmark, a single prompt, HIP-only parity,
or one accepted optimization. Continue through the highest-Amdahl unblocked
punchlist item, re-profile after every retained unit, and re-rank the remaining
work. The short-AR objective is complete only when named hipEngine production
matches or beats the final refreshed best same-host HIP and Vulkan comparators
on exact-matched p512, p1024, p4096, and their tg128 rows under section 6's
statistical rule, while the complete execution-profile, task, determinism,
state/isolation, lifecycle,
and strict-fallback gates pass. Then continue through every unlocked
long-context rung and the device-resident MTP milestone; MTP must beat true AR
on the full category+heldout suite and reach the section 6 target without
benchmark gaming.

For every implementation unit: declare the measured owner, Amdahl ceiling,
arithmetic class, affected scope, mechanism, and strict fallback; add the RED
oracle; inspect kernel lineage before a port; benchmark actual rotating weights;
prove the expected kernel ran; run the full applicable gate and same-session
whole-model A/B; retain or reject from evidence; update compact artifacts,
benchmark rollups, kernel/refactor docs, and the immutable worklog; then commit
the validated unit immediately. Never hardcode prompt/token/candidate behavior,
weaken a gate, relabel a different representation, add torch to the hot path,
or add backend/quant branches outside the registry.

Do not stop because one hypothesis loses. Record the rejection and move to the
next measured owner. If blocked, write the exact blocker and continue every
independent phase. Stop only on explicit user pause or when all campaign
acceptance criteria are satisfied and the final comparator refresh, evidence,
cleanup, and commits are complete.
```

## 8. References

- Campaign authority: [`QWEN3.8-FLASH-NEXT.md`](QWEN3.8-FLASH-NEXT.md)
- Benchmark policy: [`BENCHMARK.md`](BENCHMARK.md)
- Testing policy: [`TESTING.md`](TESTING.md)
- Profile contracts: [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md)
- Kernel catalog and port rules: [`KERNELS.md`](KERNELS.md)
- gfx1151 roofline: [`ROOFLINE-gfx1151.md`](ROOFLINE-gfx1151.md)
- Canonical exact-token fixture and driver:
  [`qwen4exp_canonical_ar_p512_p1024_p4096.json`](../benchmarks/fixtures/qwen4exp_canonical_ar_p512_p1024_p4096.json)
  and `scripts/qwen4exp_canonical_ar_bench.py`.
- Tracked implementation leads:
  [cafe-llama.cpp](https://github.com/quimmedes/cafe-llama.cpp) and
  [omlx PR #3260](https://github.com/jundot/omlx/pull/3260).
- Canonical exact-token screening:
  [`benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-canonical-ar-screening.json`](../benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-canonical-ar-screening.json)
- Fresh profile artifact:
  [`benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-fresh-full-profile.json`](../benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-fresh-full-profile.json)
- External fork refresh:
  [`benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-external-fork-refresh.json`](../benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-external-fork-refresh.json)
- Corrected invalid decode route:
  [`benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-gdn-colwarps-decode-all.json`](../benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-gdn-colwarps-decode-all.json)
- Durable profiling tools: `scripts/qwen4exp_profile_gap.py`,
  `scripts/qwen4exp_trace_analyze.py`, `scripts/qwen4exp_role_analyze.py`,
  `scripts/qwen4exp_decode_sync_ab.py`, and
  `scripts/qwen4exp_perf_gap_report.py`.
