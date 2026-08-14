# hipEngine Topline Benchmarks

Last updated: **2026-08-14**

This file is the current benchmark scoreboard. It intentionally contains only
current user-facing results, compact protocol/status notes, and links to the
authoritative evidence. It is not an optimization journal.

## Root README performance summary

The root README exports this compact retained summary verbatim.

<!-- BEGIN TOPLINE:README_HIGHLIGHTS -->
### Radeon Pro W7900 (`gfx1100`)

| Model and format | Test | Prompt processing (tok/s) | Text generation (tok/s) |
| --- | --- | ---: | ---: |
| Qwen3.6-35B-A3B ParoQuant W4 | 512 input tokens, 128 output tokens | **2917.732** | **115.599** |
| Qwen3.6-35B-A3B GGUF `Q4_K_M` | 512 input tokens, 128 output tokens | **2716.648** | **92.833** |
| Qwen3.6-27B Dense GGUF `Q4_K_M` | 512 input tokens, 128 output tokens | **670.227** | **28.444** |
| Laguna S 2.1 GGUF `UD-Q2_K_XL` | 4,096 input tokens; prompt processing only | **440.893** | — |

#### Multiple requests

Each value is the total tokens per second across all active requests:

| Model and interface | 1 request | 2 requests | 4 requests | 8 requests | 9 requests | 13 requests |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B GGUF `UD-Q4_K_M` (engine) | **98.263** | **148.944** | **209.304** | **266.479** | — | — |
| Qwen3.6-35B-A3B GGUF `UD-Q4_K_M` (server) | **72.169** | — | — | **158.542** | **137.001** | **129.507** |

#### MTP

| Model and mode | Text generation | Speed compared with AR |
| --- | ---: | ---: |
| Qwen3.6-27B Dense GGUF `Q4_K_M` — MTP-3 | **60.875 tok/s** | **2.9672x** |
| Qwen3.6-35B-A3B GGUF `UD-Q4_K_M` — MTP-2 | **122.67 tok/s** | **1.2679x** |

### Radeon RX 7900 XTX (`gfx1100`)

| Model and format | Test | Prompt processing (tok/s) | Text generation (tok/s) |
| --- | --- | ---: | ---: |
| Qwen3.6-27B Dense GGUF `Q4_K_M` | 512 input tokens, 128 output tokens | **727.961** | **33.508** |

The 27B row is an exact current single-layout snapshot, not a cross-engine win.
The campaign is explicitly blocked below llama.cpp HIP prefill and Vulkan memory
at every measured context, plus Vulkan MTP and 4K AR decode.

### Strix Halo / Radeon 8060S (`gfx1151`)

| Model and format | Test | Prompt processing (tok/s) | Text generation (tok/s) |
| --- | --- | ---: | ---: |
| Qwen3.6-35B-A3B GGUF `UD-Q4_K_M` | 512 input tokens, 128 output tokens | **1369.489** | **54.330** |
| Laguna S 2.1 GGUF `Q4_K_M` | 512 input tokens, 128 output tokens | **654.249** | **23.221** |
| Maple-Preview 2-bit | 512-token prompt test; varied prompts for generation | **754.458** | **153.201** |

#### Multiple requests

| Model and interface | 1 request | 2 requests | 4 requests | 8 requests |
| --- | ---: | ---: | ---: | ---: |
| Maple-Preview 2-bit (engine) | **123.131** | **165.697** | **202.038** | **214.788** |

#### MTP

| Model and mode | Text generation | Speed compared with AR |
| --- | ---: | ---: |
| Qwen3.6-35B-A3B GGUF `UD-Q4_K_M` — MTP-2 | **80.10 tok/s** | **1.4282x** |

### RTX PRO 6000 Blackwell (`sm_120a`)

| Model and format | Test | Prompt processing (tok/s) | Text generation (tok/s) |
| --- | --- | ---: | ---: |
| Maple-Preview 2-bit | 512-token prompt test; varied prompts for generation | **1917.492** | **402.361** |

These rows use different models and tests. Compare results only when their
protocols match. MTP-2 and MTP-3 use two and three draft tokens per cycle. The
35B-A3B MTP-2 path matches llama.cpp's MTP output on the validated prompt suite.
It remains opt-in because that output can differ from normal autoregressive
generation.
<!-- END TOPLINE:README_HIGHLIGHTS -->

## Where detailed evidence lives

| Need | Source |
| --- | --- |
| Exact commands, revisions, model fingerprints, correctness gates, samples, profiler summaries | [`benchmarks/results/`](results/) compact JSON artifacts |
| Reverse-chronological benchmark changes | [`CHANGELOG.md`](CHANGELOG.md) |
| Superseded benchmark notebook through 2026-07-10 | [`HISTORY.md`](HISTORY.md) |
| Benchmark rules and reproduction procedures | [`docs/BENCHMARK.md`](../docs/BENCHMARK.md) |
| MTP-specific protocols and terminology | [`MTP.md`](MTP.md) and [`docs/MTP-LLAMACPP-PARITY.md`](../docs/MTP-LLAMACPP-PARITY.md) |
| Kernel and implementation decisions | [`worklog/entries/`](../worklog/entries/) and [`WORKLOG-LEGACY.md`](../WORKLOG-LEGACY.md) |
| Hardware-specific RX 7900 XTX report | [`7900XTX.md`](7900XTX.md) |

The post-2026-07-10 intermediate narratives removed by this cleanup remain in
the linked JSON artifacts and changelog. The complete pre-cleanup Markdown is
also recoverable from Git without keeping a 1.1 MB notebook in the live
scoreboard:

```bash
git show 6a8d38ae70b9e2c4244df10d8621db83da6c8112:benchmarks/README.md
```

## Evidence status

| Status | Meaning | Eligible for a current numeric table? |
| --- | --- | --- |
| **Retained** | Correctness, provenance, repetition, and protocol gates passed for the named scope. | Yes. |
| **Current snapshot** | Clean current-production measurement used to describe the shipped route, but not itself a new optimization claim. | Yes, with that label. |
| **Diagnostic** | Useful attribution or comparison with a known limitation. | No; keep it in its artifact/changelog unless it explains a current blocker. |
| **Stale / superseded** | A newer route, dependency, or evidence contract replaced it. | No. |
| **Blocked / rejected** | The protocol could not complete or the candidate failed a gate. | No numeric topline. |

A row is scoped by platform, GPU, model fingerprint, quantization, KV type,
backend, workload, concurrency, speculative policy, and timing window. A newer
diagnostic never replaces a retained row.

Current blocker diagnostic: Qwen3.5-0.8B on Radeon 8060S/`gfx1151` has a
complete Q4_K_M/Q8_0 HIP/Vulkan semantic join with explicit submission
residual and `other=0`. Q4 linear-attention projections expose a 38.42%
projected stage saving and admit one bounded route audit/repair; Q8 GDN exposes
26.76%. Decode arithmetic remains blocked behind the graph/direct census.
These instrumented rows are not topline throughput; see
[`2026-08-14-gfx1151-qwen35-08b-vulkan-semantic-ledger.json`](results/2026-08-14-gfx1151-qwen35-08b-vulkan-semantic-ledger.json).

## Current single-request scoreboards

### Radeon Pro W7900: Qwen3.6-35B-A3B

These are the current six-shape hipEngine publication sweeps. PARO and GGUF use
different weight formats and should not be treated as a same-math A/B. `Peak`
is hipEngine tracked allocator high-water.

| Workload | PARO prefill | PARO decode | PARO peak | GGUF prefill | GGUF decode | GGUF peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512/128 | **2917.732** | **115.599** | 18.144 GiB | **2716.648** | **92.833** | 21.228 GiB |
| 1K/128 | **2995.876** | **103.238** | 18.367 GiB | **3052.541** | **98.148** | 21.295 GiB |
| 4K/128 | **2943.038** | **105.943** | 19.161 GiB | **2953.101** | **100.522** | 21.670 GiB |
| 32K/128 | **2108.868** | **92.438** | 19.864 GiB | **2078.038** | **88.240** | 22.234 GiB |
| 64K/128 | **1584.131** | **78.260** | 20.403 GiB | **1559.878** | **76.691** | 22.879 GiB |
| 128K/128 | **1056.252** | **60.663** | 22.124 GiB | **1037.378** | **62.669** | 24.168 GiB |

Evidence: [`PARO five-run sweep`](results/2026-07-12-w7900-v030-8116c453-hipengine-paro-packed-5run.json)
and [`GGUF final optimization sweep`](results/2026-07-16-gfx1100-gguf-final-optimization-sweep.json).
The matched llama.cpp HIP/Vulkan comparison columns and their differing memory
scopes remain in the linked artifacts and the archived rollup rather than being
repeated here.

### Radeon Pro W7900: Qwen3.6-27B Dense GGUF

| Workload | Prefill | Autoregressive decode | Tracked peak | Status |
| --- | ---: | ---: | ---: | --- |
| 512/128 | **670.227 tok/s** | **28.444 tok/s** | 15.605 GiB | Current sole-T16 package snapshot |
| 1024/128 | **714.771 tok/s** | **28.988 tok/s** | 15.720 GiB | Current sole-T16 package snapshot |
| 4096/128 | **697.749 tok/s** | **26.388 tok/s** | 16.368 GiB | Current sole-T16 package snapshot |

These rows use one discarded warmup plus three measured PM4 resets. Against the
same-commit diagnostic dual-layout rollback, the shared package default improves
prefill **152.61-184.82%**, decode **17.58-18.74%**, and whole-device peak delta
**45.50-47.03%**, with exact outputs and clean teardown. The current exact
natural suite is true AR **20.516 tok/s** and B3 **60.875 tok/s / 2.9672x**;
its lower absolute rate than the historical **61.147** row is protocol/code
drift, not a single-layout regression. Evidence:
[`same-commit W7900 non-regression`](results/2026-08-12-qwen36-27b-w7900-single-layout-non-regression.json).
The superseded dual-layout publication remains in the
[`latest-Vulkan parity exhaustion audit`](results/2026-08-07-qwen36-27b-latest-vulkan-parity-exhaustion-audit.json).

### Radeon RX 7900 XTX: Qwen3.6-27B Dense GGUF — blocked cross-engine closure

The pre-campaign dual-layout hipEngine path could not admit this model on the
23.984-GiB XTX. The package-default sole-T16 route now fits, is exact and stable,
and passes the same-commit W7900 safeguard. It does **not** meet the campaign's
cross-engine acceptance policy: llama.cpp HIP remains faster for prefill,
Vulkan remains lower-memory, and Vulkan wins selected MTP plus 4K AR decode.
The complete five-sample 512/128, 1024/128, and 4096/128 hipEngine AR matrix is
retained as a current partial result. Clean same-commit llama.cpp `c8e03ce81`
HIP and Vulkan establish the frozen speed and whole-device VRAM targets:

| Workload | HIP prefill | Vulkan prefill | HIP context AR | Vulkan context AR | Lower peak delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512/128 | **964.606** | 870.872 | **33.025** | 13.391 | **15.690 GiB** (Vulkan) |
| 1024/128 | **981.040** | 836.898 | **32.924** | 13.379 | **15.700 GiB** (Vulkan) |
| 4096/128 | **946.733** | 835.765 | **32.560** | 13.309 | **15.912 GiB** (Vulkan) |

Prefill is llama-bench `avg_ts` over five internal repetitions after one
warmup; context AR is 128 timed server transitions. On the complete ten-prompt
natural suite, HIP selects B2 at **46.863 tok/s / 1.4841x AR / 16.940 GiB**
peak delta, while Vulkan selects B4 at **81.952 tok/s / 6.1223x AR / 16.673
GiB**. The frozen hipEngine gates add a 1% speed margin and require no more than
the lower Vulkan memory row.

Current hipEngine matrix (one warmup plus five persistent-session
reset/replays per shape):

| Workload | Prefill | Decode | Tracked peak | Whole-device peak delta | Gate status |
| --- | ---: | ---: | ---: | ---: | --- |
| 512/128 | **727.961 tok/s** | **33.508 tok/s** | **15.605 GiB** | **16.095 GiB** | prefill fail / **decode pass** / memory fail |
| 1024/128 | **785.347 tok/s** | **34.537 tok/s** | **15.720 GiB** | **16.320 GiB** | prefill fail / **decode pass** / memory fail |
| 4096/128 | **779.243 tok/s** | **31.391 tok/s** | **16.368 GiB** | **17.119 GiB** | prefill fail / decode fail / memory fail |

The complete ten-prompt llama-compatible natural suite selects B3:

| Mode | Transition decode | vs true AR | Draft acceptance | Whole-device peak delta |
| --- | ---: | ---: | ---: | ---: |
| True AR | **20.782 tok/s** | 1.0000x | — | — |
| B1 | **53.222 tok/s** | 2.5609x | 91.27% | — |
| B2 | **66.350 tok/s** | 3.1926x | 82.97% | — |
| B3 | **72.887 tok/s** | **3.5071x** | 77.17% | **17.183 GiB** |

All B1-B3 outputs match true AR across all ten prompts, four categories, six
train prompts, and four heldouts. B3 is **55.53% faster** than clean llama.cpp
HIP B2 but **11.06% slower** than Vulkan B4 and exceeds Vulkan's lower memory
floor by **0.509 GiB**. One independent full suite is complete. A second suite
cannot reverse the binding speed/memory failures, so the stop rule closes the
campaign without spending another expensive variance run; no XTX MTP row is
promoted to the compact root topline.

The candidate uses one T16 payload for each of all 288 rank-2 Q4 tensors and a
sole 715,161,600-byte device-visible mapped GGUF mmap for the root Q4 token
table, with no VRAM shadow. The initial 512 screen measured PM4 at **33.424
vs 32.897 tok/s (+1.601%)** across three rearmed 128-transition runs. A stricter
same-session, counterbalanced transport matrix confirms PM4 over HIP graph at
all campaign contexts: **33.494 vs 33.027 (+1.412%)** at 512, **34.445 vs
33.854 (+1.747%)** at 1K, and **31.320 vs 30.830 tok/s (+1.588%)** at 4K.
PM4 won all **15/15** paired samples with exact recorded outputs, zero native
fallbacks or unretired submissions, and clean teardown. It remains the
narrowly-scoped default for this model at private-c1 horizons of at least 128.
The mapping cuts tracked residency **16.749 -> 16.083 GiB** and
same-workload sampled peak delta **17.347 -> 16.679 GiB**; the standard
512/128 PM4 row peaks at **16.712 GiB** and full graph/session teardown returns
tracked bytes to zero. A model-scoped dense phase-liveness arena then removes
unused MoE fields and aliases mutually exclusive linear/full-attention/FFN
scratch: physical bulk scratch falls **0.589 -> 0.111 GiB**, tracked peak falls
**16.083 -> 15.605 GiB**, and sampled peak delta falls **16.712 -> 16.214
GiB** with exact eager/PM4 final logits across three resets. A model-scoped
small-weight arena then packs 481 <=16-MiB allocations into one owner, reducing
physical weight owners **850 -> 370** and sampled peak delta another **44.000
MiB** to **16.171 GiB**; the 645,120-byte tracked alignment cost leaves tracked
peak at 15.605 GiB. Storing the shared Q4T16 K256 slab output-major then turns
scalar LDS fragment traffic into vector loads, cuts the traced 288-call Q4
family **433.351 -> 421.447 ms (-2.747%)**, and improves exact 512/128 prefill
**719.232 -> 730.589 tok/s (+1.579%)** without changing decode or residency.
The model-scoped arena cutoff then widens from 16 to the first complete 80-MiB
inventory crossover: **849** immutable allocations share one owner, only the
994.6-MiB untied head remains dedicated, physical weight owners fall **370 ->
2**, and standard process peak delta falls **16.171 -> 16.095 GiB (-77.840
MiB)** with neutral exact 512/128 behavior. The final five-sample matrix is
**727.961/33.508**, **785.347/34.537**, and **779.243/31.391 tok/s** at
512/1K/4K. Every shape fits and is deterministic; 512/1K decode pass, but all
prefill/memory rows and 4K decode fail their frozen gates. This is a retained
current snapshot, not a cross-engine win.

The final live target+NextN census proves **zero duplicate-payload and zero
alternate-layout bytes** across 870 references / 866 physical ranges; mapped
embedding and untied output head are shared exactly once and all tracked
ownership closes to zero. Stability evidence adds three AR plus three natural
MTP cold process cycles, then **100 mixed 512/1K/4K reset/rearms / 400 PM4
submissions** in one resident process with exact tokens/logits, constant
post-warm ownership, zero fallback/unretired work, and fully retired graph,
executable, and context children. Deep 512/1K/4K eager oracles match fresh
serial-prefix token, FP32 hidden, every Conv/GDN state, and all live BF16 K/V
bytes at all 12 transitions. PM4 matches those eager states plus full logits and
complete `KVLiveSpans` at the common pre-execution boundary, again with zero
fallback/unretired submissions and clean retirement. Real dense B1-B3
reject/partial/full/rollback/reseed transactions, proposal-time cancellation,
and the public AR/repeated-MTP/HTTP torch-free owner-reuse lifecycle also pass;
tracked public teardown is zero and final VRAM is within 0.8 MiB of baseline.
Public dense MTP now shares public AR's bulk prompt admission and catches NextN up from shifted device-resident target hidden rows. A fixed 601-second soak completes **204 cycles / 408 requests** across all ten prompts/four categories with every AR/MTP pair exact, stable per-prompt hashes, one reused target/draft owner, zero live-byte spread, no torch, tracked close zero, **18.209-GiB** peak VRAM delta, **62/83/94 C** max edge/junction/memory temperatures, and final VRAM **+0.79 MiB**. Public verification uses correctness-first serial-exact; native verification after bulk prefill remains a documented blocker.

Evidence: [`clean comparator floors`](results/2026-08-12-qwen36-27b-xtx-clean-llamacpp-floors.json),
[`pre-single-layout blocker`](results/2026-08-12-qwen36-27b-xtx-pre-single-layout-blocked.json),
[`sole-T16 first fit`](results/2026-08-12-qwen36-27b-xtx-sole-t16-first-fit.json),
[`mapped-host/PM4 partial pass`](results/2026-08-12-qwen36-27b-xtx-mapped-host-embedding.json),
[`dense scratch liveness`](results/2026-08-12-qwen36-27b-xtx-dense-prefill-scratch-liveness.json),
[`small-weight arena`](results/2026-08-12-qwen36-27b-xtx-small-weight-arena.json),
[`Q4T16 output-major LDS`](results/2026-08-12-qwen36-27b-xtx-q4-t16-output-major-lds.json),
[`wide weight arena`](results/2026-08-12-qwen36-27b-xtx-wide-weight-arena.json),
[`complete engine AR matrix`](results/2026-08-12-qwen36-27b-xtx-engine-ar-matrix.json),
[`llama-compatible natural MTP matrix`](results/2026-08-12-qwen36-27b-xtx-llama-compatible-mtp.json),
[`correctness/runtime residency`](results/2026-08-12-qwen36-27b-xtx-correctness-residency.json),
[`cold/warm PM4 lifecycle`](results/2026-08-12-qwen36-27b-xtx-lifecycle.json),
[`public AR/MTP bulk-admission soak`](results/2026-08-12-qwen36-27b-xtx-public-ar-mtp-soak.json),
[`PM4 versus HIP graph at all contexts`](results/2026-08-12-qwen36-27b-xtx-pm4-all-contexts.json),
and [`same-commit W7900 non-regression`](results/2026-08-12-qwen36-27b-w7900-single-layout-non-regression.json).

The shared gfx1100 default also passes its original-card safeguard. Against a
same-commit W7900 rollback that reconstructs rank-2 Q4 pack8+T16 residency, the
single-layout route improves 512/1K/4K prefill by **152.61-184.82%**, decode by
**17.58-18.74%**, natural true AR by **2.61%**, and B3 by **0.24%**, while
cutting natural-suite peak delta **31.680 -> 17.183 GiB (-45.76%)**. All outputs
and acceptance ledgers remain exact and teardown returns tracked ownership to
zero.

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
| W7900 / Qwen3.6-27B Dense `Q4_K_M` | Current exact natural25 B3 | 20.516 | **60.875** | **2.9672x** | Current sole-T16 package snapshot; exact and faster than same-commit dual-layout control, but still below selected llama.cpp Vulkan B4. [`artifact`](results/2026-08-12-qwen36-27b-w7900-single-layout-non-regression.json) |
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

## Current PM4 transport diagnostic

This is a transport diagnostic for the W7900 Qwen3.6-35B-A3B Q4_K_M one-step
replay path, not a separate model-throughput topline.

| Transport | Replay | Capture | Status |
| --- | ---: | ---: | --- |
| HIP graph | **10.726350 ms/token / 93.228 tok/s** | 34.993 ms | Portable oracle and fallback |
| Stateful PM4, global acquire | **10.052766 / 99.475** | 79.296 ms warm-metadata | Exact comparison path |
| Stateful PM4, local-cache acquire | **9.964358 / 100.358** | 132.858 ms cold | Scoped default when the measured replay window amortizes capture |

Evidence: [`local-cache capture/replay`](results/2026-08-08-gfx1100-pm4-setup-local-cache-clean.json)
and [`scoped default`](results/2026-08-08-gfx1100-pm4-scoped-default.json).

## Reading the tables

- Workload format is `prompt_tokens/decode_tokens`.
- Prefill, backend decode, full request wall, server wall, and component timing
  are different scopes. Compare only like with like.
- Aggregate concurrency throughput is total generated tokens divided by group
  wall. Per-request throughput is lower when multiple requests share the GPU.
- PARO, GGUF, and llama.cpp rows may use different quantization or KV formats;
  raw leaders are descriptive unless the artifact establishes a matched A/B.
- hipEngine tracked memory and whole-device VRAM/GTT are different scopes.
- A bold value identifies the reported row, not a universal claim across
  mismatched hardware, models, or protocols.

## Maintenance contract

Keep this file compact:

1. Replace the current row for a protocol tuple; do not append an optimization
   diary beneath it.
2. Put exact commands, samples, deltas, profiler data, correctness details, and
   candidate decisions in the compact JSON artifact.
3. Put the one-line old-to-new transition in [`CHANGELOG.md`](CHANGELOG.md) and
   substantial implementation decisions in a new immutable worklog entry.
4. Mention a blocked/rejected run here only when it removes a current numeric
   row or defines a user-visible limitation. Link one artifact and one rerun
   condition; keep candidate ladders out of the scoreboard.
5. Keep superseded tables in [`HISTORY.md`](HISTORY.md), artifacts, or Git
   history rather than copying them forward.
6. Update `Last updated`, then synchronize the public block:

```bash
python3 scripts/sync_benchmark_readme.py --write
python3 scripts/sync_benchmark_readme.py --check
git diff --check
```

The full evidence and artifact requirements remain authoritative in
[`docs/BENCHMARK.md`](../docs/BENCHMARK.md).
