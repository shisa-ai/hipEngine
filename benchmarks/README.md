# hipEngine Topline Benchmarks

Last updated: **2026-08-23**

This file is the current benchmark scoreboard. It intentionally contains only
current user-facing results, compact protocol/status notes, and links to the
authoritative evidence. It is not an optimization journal.

## Root README performance summary

The root README exports this compact retained summary verbatim.

<!-- BEGIN TOPLINE:README_HIGHLIGHTS -->
### Radeon Pro W7900 (`gfx1100`)

| Model and format | Test | Prompt processing (tok/s) | Text generation (tok/s) |
| --- | --- | ---: | ---: |
| Qwen3.6-35B-A3B GGUF `Q4_K_M` | 512 input tokens, 128 output tokens | **2716.648** | **92.833** |
| Qwen3.6-27B Dense GGUF `Q4_K_M` | 512 input tokens, 128 output tokens | **865.179** | **28.368** |
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

### RX 7900 XTX (`gfx1100`) — Qwen3.8-27B `Q4_K_M` prefill

| Workload | hipEngine | llama.cpp HIP | HE vs HIP | llama.cpp Vulkan | HE vs Vulkan |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 | **959.4** | 965.0 | -0.6% | 865.7 | +10.8% |
| 1K | **999.7** | 979.5 | +2.1% | 832.6 | +20.1% |
| 4K | **981.8** | 945.7 | +3.8% | 836.5 | +17.4% |

#### Dedicated-server context
| KV route | Server shape | Measured context | Peak / headroom |
| --- | --- | ---: | ---: |
| BF16 default | c1 operational | **32K** | 21.869 / 2.115 GiB |
| Pure INT8 explicit | c1 repeated natural soak | **112K** | 23.323 / 0.661 GiB |
| Pure INT8 explicit | c1 one-request physical ceiling | **126K** | 23.963 / 0.022 GiB |

#### Decode / MTP

| Metric | hipEngine | llama.cpp HIP | HE vs HIP | llama.cpp Vulkan | HE vs Vulkan |
| --- | ---: | ---: | ---: | ---: | ---: |
| AR decode 512 | **34.06** | 32.86 | +3.6% | 13.39 | 2.54x |
| AR decode 1K | **34.91** | 32.75 | +6.6% | 13.38 | 2.61x |
| AR decode 4K | **31.79** | 32.41 | -1.9% | 13.31 | 2.39x |
| MTP natural | **62.44 B3** | 44.33 B2 | +40.9% | 73.33 B2 | -14.8% |

### Strix Halo / Radeon 8060S (`gfx1151`)

| Model and format | Test | Prompt processing (tok/s) | Text generation (tok/s) |
| --- | --- | ---: | ---: |
| Qwen3.6-35B-A3B GGUF `UD-Q4_K_M` | 512 input tokens, 128 output tokens | **1369.489** | **54.330** |
| Qwen3.6-35B-A3B GGUF `UD-Q4_K_M` (GEMV lib hoist) | sync'd eager, per-token | — | **38.9** |
| Qwen3.8-27B Dense GGUF `Q4_K_S` | 512 input tokens, 128 output tokens | **396.091** | **13.069** |
| Laguna S 2.1 GGUF `Q4_K_M` | 512 input tokens, 128 output tokens | **654.249** | **23.221** |
| Maple-Preview 2-bit | 512-token prompt test; varied prompts for generation | **754.458** | **153.201** |

#### Multiple requests

| Model and interface | 1 request | 2 requests | 4 requests | 8 requests |
| --- | ---: | ---: | ---: | ---: |
| Maple-Preview 2-bit (engine) | **123.131** | **165.697** | **202.038** | **214.788** |

#### MTP

| Model and mode | Text generation | Speed compared with AR |
| --- | ---: | ---: |
| Qwen3.8-27B Dense GGUF `Q4_K_S` — MTP-3 | **23.853 tok/s** | **1.7845x** |
| Qwen3.6-35B-A3B GGUF `UD-Q4_K_M` — MTP-2 | **80.10 tok/s** | **1.4282x** |

### RTX PRO 6000 Blackwell (`sm_120a`)

| Model and format | Test | Prompt processing (tok/s) | Text generation (tok/s) |
| --- | --- | ---: | ---: |
| Maple-Preview 2-bit | 512-token prompt test; varied prompts for generation | **1917.492** | **402.361** |

Rows use different models and tests; compare only matching protocols. The RX 7900 XTX cross-engine rows use the same Qwen3.8 file and timing boundary.
llama.cpp Vulkan MTP is speed-only because its ledger differs from Vulkan AR; hipEngine and llama.cpp HIP match their controls. MTP-2/MTP-3 use two/three draft tokens.
The 35B-A3B MTP-2 path matches llama.cpp MTP on the validated suite and remains opt-in because it can differ from normal AR.
<!-- END TOPLINE:README_HIGHLIGHTS -->

## Current default notes

Strix Halo Qwen3.8 `Q4_K_S` defaults to FP16 recurrent state with explicit FP32 rollback after the complete packed and serving gates; see the [`retained artifact`](results/2026-08-20-gfx1151-qwen38-27b-r2-fp16-state-repaired-production.json). The broader default-off review remains in the [`promotion inventory`](results/2026-08-20-valid-faster-default-off-inventory.json).

### External DMS trained candidate (default-off)

| Exact model | Candidate | Long heldout quality | Total live-cell compression | Status |
| --- | --- | --- | ---: | --- |
| Qwen3.8-27B `Q4_K_M` | external linear CR2/window256 | max KL **0.009691**, top-1 **100%** | **1.543x** | trained candidate; integrated product gate open |

CR4/CR8 are rejected at max KL 0.08908/0.24993. The CR2 row is quality-only and
retains dense KV during replay; it is not a serving-memory or performance claim.
Dense paging remains default. See the [`candidate evidence`](results/2026-08-23-qwen38-external-dms-cr2-trained-candidate.json).

## Where detailed evidence lives

| Need | Source |
| --- | --- |
| Exact commands, revisions, model fingerprints, correctness gates, samples, profiler summaries | [`benchmarks/results/`](results/) compact JSON artifacts |
| Reverse-chronological benchmark changes | [`CHANGELOG.md`](CHANGELOG.md) |
| Superseded benchmark notebook through 2026-07-10 | [`HISTORY.md`](HISTORY.md) |
| Benchmark rules and reproduction procedures | [`docs/BENCHMARK.md`](../docs/BENCHMARK.md) |
| Execution-profile numerical calibration | [`2026-08-16 ZBook-local policy artifact`](results/2026-08-16-execution-profile-threshold-calibration.json) and [`docs/EXECUTION-PROFILES.md`](../docs/EXECUTION-PROFILES.md) |
| MTP-specific protocols and terminology | [`MTP.md`](MTP.md) and [`docs/MTP-LLAMACPP-PARITY.md`](../docs/MTP-LLAMACPP-PARITY.md) |
| Quantization-quality protocols and current tables | [`quant/README.md`](quant/README.md) |
| Kernel and implementation decisions | [`worklog/entries/`](../worklog/entries/) and [`WORKLOG-LEGACY.md`](../WORKLOG-LEGACY.md) |
| Hardware-specific RX 7900 XTX report | [`7900XTX.md`](7900XTX.md) |

## Benchmark harness catalog

There is no single "run everything" benchmark. Different questions are answered
by different harnesses, each with a specific timing scope, numerical contract,
and shape. The table below is the map: it says what each harness measures so a
result is only compared against like-for-like rows. A ✓ marks what a harness
owns and reports; a column left blank means that axis is not measured by that
harness (not that it is zero). Always run hipEngine rows through the hermetic
thecrock wrapper for the target architecture (see `docs/BENCHMARK.md`).

Legend: **AR** = true no-MTP autoregressive decode · **MTP** = speculative
multi-token-prediction decode (with a true-AR denominator where a ratio is
reported) · **Prefill** = prompt-processing tok/s · **Decode** = generation
tok/s · **Mem** = tracked/HIP/GTT memory usage · **Conc** = per-concurrency
(c=1..8) sweep.

| Harness (`scripts/`) | What it answers | AR | MTP | Prefill | Decode | Mem | Conc | Canonical entrypoint |
| --- | --- | :-: | :-: | :-: | :-: | :-: | :-: | --- |
| `qwen35_readme_sweep.py` | Single-request prefill/decode/memory per shape (llama-bench-style), one resident session, per-shape reset | ✓ | | ✓ | ✓ | ✓ | | `--engine gguf --model <model> --backend hip_gfx1151 --workloads 512/128 1K/128 ...` |
| `qwen35_gguf_bench.py` | GGUF c=1 AR prefill/decode, fresh resident session per run, HIP-graph decode | ✓ | | ✓ | ✓ | ✓ | | `--model <model> --prompt-length 512 --decode-tokens 128` |
| `gguf_true_ar_category_bench.py` | True no-MTP AR baseline over the mtp-bench category suite (the legitimate MTP speed denominator) | ✓ | | ✓ | ✓ | | | `--model <model> --prompts benchmarks/prompts/mtpbench-code-general-ja.jsonl` |
| `gguf_mtp_category_bench.py` | MTP category matrix over budgets 1..8 with guarded objective extraction; attach a true-AR baseline for ratios | | ✓ | | ✓ | | | `--budgets 1,3,5 --objective-budget b5` |
| `gguf_mtp_long_context_gate.py` | Eager-native MTP correctness vs serial-exact teacher across context/page/budget/acceptance boundaries; optional real host-proposal AR-ID gate (no speed claim) | ✓ | ✓ | | | | | `--cycle-ends 1016-1032,4K --candidate-budgets 1,2,3 --fail-on-fail` |
| `gguf_ar_mtp_suite.py` | One-command AR-vs-MTP decode ratio over the category suite under one enforced decode config | ✓ | ✓ | | ✓ | | | `--scope partial --output <json>` |
| `qwen35_batch_retained_bench.py` | **PARO-path** compact c>N batch decode; aggregate + per-request tok/s, equality vs c1, optional MTP draft depth | ✓ | ✓ | | ✓ | ✓ | ✓ | `--batch-size 8 --decode-tokens 128` |
| `qwen35_batch_gguf_diagnostic.py` | GGUF c>N generated-token **correctness** equality vs independent c1 (no throughput claim) | ✓ | | | | | ✓ | `--rows 8 --execute` |
| `server_f1_concurrency_bench.py` | Matched gfx1151 F1 HTTP concurrency (c=1..8) for hipEngine vs llama.cpp HIP/Vulkan; combined throughput + memory | ✓ | | | ✓ | ✓ | ✓ | `--engine hipengine --model <model> --concurrencies 1,2,4,8` |
| `gguf_concurrency_baseline.py` | GGUF c1 + explicit serial c2/c4 timing controls (Phase-A route baseline) | ✓ | | ✓ | ✓ | | ✓ | `--model <model> --concurrencies 1,2,4` |
| `mtp-bench.py` | llama.cpp-compatible MTP prompt-suite benchmark (server economics); can wrap hipEngine verifier economics | ✓ | ✓ | | ✓ | | | `--mode hipengine-current` |
| `exact_token_generation.py` | Direct/HTTP generated-token identity gate (correctness, not throughput) | ✓ | ✓ | | | | | `direct --model-path ...` then `http --oracle ...` |
| `benchmark_matrix.py` | Join exact-token direct/server rows into a validated matrix report | ✓ | ✓ | | | | | `build --manifest ...` |

The two rows that most closely produce the README **concurrency scoreboards**
are `qwen35_batch_retained_bench.py` (direct engine) and `server_f1_concurrency_bench.py`
(OpenAI server). The **single-request** tables come from `qwen35_readme_sweep.py`
(GGUF/PARO) and `qwen35_gguf_bench.py`. The **speculative-decode** tables come
from `gguf_ar_mtp_suite.py` / `gguf_mtp_category_bench.py` with a
`gguf_true_ar_category_bench.py` true-AR denominator.

This catalog is maintained alongside the harnesses themselves: when a harness
learns a new axis (for example MTP added to a previously AR-only bench), update
this table in the same unit.

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

## Current Generation-2 qualification

W7900 Qwen3.6-35B-A3B `UD-Q4_K_M`, BF16 KV, p128/d8, token-budget
scheduling, and same-loaded-server c1 oracles:

| Logical concurrency | 1 | 4 | 8 | 17 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Aggregate HTTP tok/s | **27.443** | **43.337** | **46.158** | **45.797** | **44.320** |
| Exact rows | 1/1 | 4/4 | 8/8 | 17/17 | 32/32 |

The canonical load packet passes all nine fixed/ragged/Poisson/cancel/overload/
recovery/soak workloads with **210/210** correctness-accounted rows, bounded
`engine_busy` overload, **271/271** admission/reclaim, zero final refs/pins, and
zero tracked-memory delta. Physical c1/c2/c4/c8 and logical c1-c32 are retained
for this package. [`Canonical artifact`](results/2026-08-18-concurrency2-c2-6-w7900-canonical-production-accepted.json).

On Radeon 8060S/gfx1151, Qwen3.8 `Q4_K_S` packed prefill improves exact c17
streaming from **9.673→10.956 tok/s (+13.27%)** and TTFT p95
**11.030→9.406 s (-14.72%)**. A subsequent exact fused packed-state transfer
reduces the marked c17 owner **420.496→410.878 ms (-2.29%)**. Direct canonical
resident state then reaches **368.413 ms**, c17 **11.271 tok/s**, and ITL p99
**0.4542 s** (3/3 fixed-SLO passes). The exact row8 Q4 two-wave owner then
reaches c17 **11.297 tok/s / 0.448 s ITL** and c32 **11.041 tok/s / 0.802 s
ITL**; c32 live admission overlaps, but c32 fixed SLO remains blocked, so
gfx1151 canonical production is not yet promoted.

The separate W7900 Qwen3.8-27B `Q4_K_M` direct graph packet qualifies physical
`(1,2,3,4,5,6,7,8)`: c1-c8 reaches **30.30/53.79/75.47/93.49/105.67/
115.30/122.36/127.32 tok/s**, all exact and repeatable. Q5 and planar-Q6 true
rowtiles own rows 5-8; dynamic compaction, state/KV, graph invalidation, cancel/
refill, memory recovery, and drain pass. Logical c>8 uses deterministic ceiling
composition; artifact-backed D2 remains explicit research only.
[`Width and lifecycle evidence`](results/2026-08-20-concurrency2-qwen38-direct-width-lifecycle.json).

The exact gfx1100 planar-Q6 row8 DPP reduction improves the marked production-
owner physical-c8 transition **58.693→57.734 ms (-1.634%)** median with identical
8×32 token IDs, route, graph transport, and drain. This is a steady-transition
kernel result, not a replacement for the direct-width throughput packet.
The detailed promotion evidence is retained in the benchmark changelog/artifacts.

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

Current Qwen3.5-0.8B gfx1151 status remains **Vulkan parity blocked**, while
the exact D08-X package is retained. Its clean three-way exact-core pp512 is
hipEngine / llama HIP / Vulkan **4896/4848/5510 Q4** and **4997/4640/5704 Q8
tok/s**. Retained operation-complete units include Q4 pack8 gate+up+SiLU,
dense-BF16 down+residual, and Q8T16 alpha/beta dual WMMA; the final natural and
category gate is **1794/1800 top-1, max KL 0.005930**, with all **72/72** graph
trajectories exact. Rejected projection, GDN-broadcast, and pack8-residual
candidates remain in artifacts and the changelog. Evidence:
[`three-way snapshot`](results/2026-08-15-gfx1151-qwen35-08b-post-x3-current-exact-three-way.json),
[`alpha/beta WMMA`](results/2026-08-15-gfx1151-qwen35-08b-q8t16-alpha-beta-dual-wmma-prefill.json),
[`cumulative gate`](results/2026-08-15-gfx1151-qwen35-08b-cumulative-semantic.json), and the
[`campaign`](../docs/QWEN35-08B-GFX1151-VULKAN-PARITY.md).

## Current single-request scoreboards

### Radeon Pro W7900: Qwen3.6-35B-A3B

The current valid six-shape publication is GGUF. `Peak` is hipEngine tracked
allocator high-water.

| Workload | GGUF prefill | GGUF decode | GGUF peak |
| --- | ---: | ---: | ---: |
| 512/128 | **2716.648** | **92.833** | 21.228 GiB |
| 1K/128 | **3052.541** | **98.148** | 21.295 GiB |
| 4K/128 | **2953.101** | **100.522** | 21.670 GiB |
| 32K/128 | **2078.038** | **88.240** | 22.234 GiB |
| 64K/128 | **1559.878** | **76.691** | 22.879 GiB |
| 128K/128 | **1037.378** | **62.669** | 24.168 GiB |

The former PARO speed row is withdrawn: its sweep predates the grouped-V-head
runtime fix and therefore followed the wrong generated trajectory. It remains a
[`pre-fix diagnostic`](results/2026-07-12-w7900-v030-8116c453-hipengine-paro-packed-5run.json),
not current performance evidence; a fresh repaired-runtime sweep is required.
GGUF evidence: [`final optimization sweep`](results/2026-07-16-gfx1100-gguf-final-optimization-sweep.json).
The matched llama.cpp HIP/Vulkan comparison columns and their differing memory
scopes remain in the linked artifacts and the archived rollup rather than being
repeated here.

### Dense-Qwen detailed comparisons

Current Qwen3.6-27B and Qwen3.8 cross-engine, context-capacity, INT8-KV, and serving-frontier detail is archived in [`benchmarks/HISTORY.md`](HISTORY.md); the compact current rows above remain authoritative.

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
| W7900 / Qwen3.6-27B Dense `Q4_K_M` | Exact/default natural25 B3 | 22.926 | **61.147** | **2.6671x** | Retained exact natural25 control; all greedy outputs and GPU/CPU acceptance agree. [`artifact`](results/2026-08-07-qwen36-27b-latest-vulkan-parity-exhaustion-audit.json) |
| RX 7900 XTX / Qwen3.8-27B Dense `Q4_K_M` | Exact/default natural25 B3 | 35.287 | **62.440** | **1.7695x** | Clean idle-card correction; exact greedy and GPU/CPU acceptance, retained fusion improves matched AR 3.764% and B3 0.439% with every category non-regressive. [`artifact`](results/2026-08-15-qwen38-27b-xtx-clean-idle-performance-correction.json) |
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
   history rather than copying them forward (`git show 6a8d38ae70b9e2c4244df10d8621db83da6c8112:benchmarks/README.md`).
6. Update `Last updated`, then synchronize the public block:

```bash
python3 scripts/sync_benchmark_readme.py --write
python3 scripts/sync_benchmark_readme.py --check
git diff --check
```

The full evidence and artifact requirements remain authoritative in
[`docs/BENCHMARK.md`](../docs/BENCHMARK.md).
