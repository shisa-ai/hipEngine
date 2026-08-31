# hipEngine Topline Benchmarks

Last updated: **2026-09-01**

This file is the current benchmark scoreboard. It intentionally contains only
current user-facing results, compact protocol/status notes, and links to the
authoritative evidence. It is not an optimization journal.

## Root README performance summary

The root README exports this compact retained summary verbatim.

<!-- BEGIN TOPLINE:README_HIGHLIGHTS -->
### Radeon Pro W7900 (`gfx1100`)

| Model and format | Test | Prompt processing (tok/s) | Text generation (tok/s) |
| --- | --- | ---: | ---: |
| Qwen3.6-35B-A3B ParoQuant W4 | 512 input tokens, 128 output tokens | **2852.100** | **115.804** |
| Qwen3.6-35B-A3B GGUF `Q4_K_M` | 512 input tokens, 128 output tokens | **2763.590** | **94.603** |
| Qwen3.6-27B Dense GGUF `Q4_K_M` | 512 input tokens, 128 output tokens | **875.364** | **28.681** |
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
| Qwen3.6-27B Dense GGUF `Q4_K_M` — MTP-3 | **60.929 tok/s** | **2.0684x** |
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
| Qwen3.8-27B Dense GGUF `Q4_K_M` — public C1 MTP-3 automatic scope | **12.940 tok/s** | **1.4337x** |
| Qwen3.6-35B-A3B GGUF `UD-Q4_K_M` — MTP-2 | **80.10 tok/s** | **1.4282x** |

### RTX PRO 6000 Blackwell (`sm_120a`)

| Model and format | Test | Prompt processing (tok/s) | Text generation (tok/s) |
| --- | --- | ---: | ---: |
| Maple-Preview 2-bit | 512-token prompt test; varied prompts for generation | **1917.492** | **402.361** |

Rows use different models and tests; compare only matching protocols. The RX 7900 XTX cross-engine rows use the same Qwen3.8 file and timing boundary.
llama.cpp Vulkan MTP is speed-only because its ledger differs from Vulkan AR; hipEngine and llama.cpp HIP match their controls. MTP-2/MTP-3 use two/three draft tokens. The 35B-A3B MTP-2 path matches llama.cpp MTP on the validated suite and remains opt-in because it can differ from normal AR.
<!-- END TOPLINE:README_HIGHLIGHTS -->

## Current default notes

Strix Halo Qwen3.8 `Q4_K_M` automatic MTP is restricted to its verified
strict/BF16/C1/B3/raw-greedy key; other scopes use K0/AR.
[`Serving closure`](results/2026-08-26-gfx1151-qwen38-q4km-mtp-serving-s5-closure.json).
Qwen3.8 `Q4_K_S` defaults to FP16 recurrent state with FP32 rollback; its exact
DMS sidecar remains default-off pending serving gates. [`DMS`](../docs/DMS.md).

Agentic quality is quality-only: Qwen3.8-27B `Q4_K_M` scores **50/68 (73.53%)**
with 64/64 valid calls; no runtime mechanism is retained.
[`Final`](results/2026-08-26-zbook-agentic-quality2-campaign-final.json).
Generation-2 automatic serving remains K0: gfx1151 P9 is exact 540/540 but c2/c4
are 0.6975x/0.5843x AR; gfx1100 exact speculative cells remain behind direct.
[`Closure`](results/2026-08-26-gfx1151-specdec2-perf-campaign-closure.json) ·
[`Recovery`](../docs/MTP-CONCURRENCY2-RECOVERY.md).

## Where detailed evidence lives

Use result artifacts for commands/samples/profilers,
[`CHANGELOG.md`](CHANGELOG.md) for rollups, [`docs/BENCHMARK.md`](../docs/BENCHMARK.md)
for protocols, and [`worklog/entries/`](../worklog/entries/) for decisions.

## Benchmark harness catalog

Compare only matching harness scopes. ✓ marks a reported axis; blanks are
unmeasured. **AR/MTP/Prefill/Decode/Mem/Conc** mean true autoregressive,
speculative, prompt, generation, memory, and concurrency respectively. Use the
hermetic target-architecture wrapper; see `docs/BENCHMARK.md`.

| Harness (`scripts/`) | What it answers | AR | MTP | Prefill | Decode | Mem | Conc | Canonical entrypoint |
| --- | --- | :-: | :-: | :-: | :-: | :-: | :-: | --- |
| `qwen35_readme_sweep.py` | Single-request prefill/decode/memory per shape (llama-bench-style), one resident session, per-shape reset | ✓ | | ✓ | ✓ | ✓ | | `--engine gguf --model <model> --backend hip_gfx1151 --workloads 512/128 1K/128 ...` |
| `qwen4exp_canonical_ar_bench.py` | Exact-token Qwen4Exp cross-engine p512/p1024/p4096 prefill plus context-conditioned tg128, output hashes, and comparison artifact | ✓ | | ✓ | ✓ | | | `hipengine --model-root <model>` or `llamacpp --server-bin <binary> --model <part1>` |
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

## Qwen3.8-Flash-Next implementation-first status

On physical host `zbook` (Ryzen AI Max+ Pro 395 / Radeon 8060S, `gfx1151`),
the pinned four-part Unsloth `UD-Q4_K_XL` artifact now runs through public
`LLM.generate()` under the strict c1/greedy text scope. Frozen same-artifact
llama.cpp PR #27742 full logits over all 10 canonical code/general-English/
general-Japanese/mixed prompts measured:

| Artifact | Context scope | Mean / p95 / p99 / max KL ↓ | Top-1 | Tracked peak / after close |
| --- | --- | ---: | ---: | ---: |
| Qwen3.8-Flash-Next `UD-Q4_K_XL` | real ≤2,051-token canonical text gate | **0.01406 / 0.04154 / 0.04776 / 0.04931** | **10/10** | 82.718 GB / **0 B** |
| Qwen3.8-Flash-Next `UD-Q4_K_XL` | predeclared eight category heldouts, matched BF16 K/V | **0.00987 / 0.02331 / 0.02766 / 0.02874** | **8/8** | same residency / **0 B** |

The pinned 111.335-GB/four-hash artifact owns one 28.800-GB sparse-mmap PLE
table and 82.523 GB hot weights. Exact batching passes 687/687 rows; the strict
prefill default chunk is now 512 (PLE staging capacity plumbed to the chunk;
previously silently capped at 256): same-session counterbalanced sweeps give
p508 **8.458→8.270 s (-2.22%)** and p1012 **17.062→16.751 s (-1.82%)** with
identical logits SHAs, and natural 16K improves to **341.177 s / 47.989 tok/s**
with the full gate passing (prior chunk-256 steady rows were p508 58.466 and
p1006 55.046 tok/s).

The exact-token screening baseline now feeds all engines the same four category
prompts at p512/p1024/p4096 and measures 128 decode transitions after each
prefix:

| Engine | p512 pp/tg128 | p1024 pp/tg128 | p4096 pp/tg128 | Repeatability |
| --- | ---: | ---: | ---: | --- |
| hipEngine production | **82.51 / 13.82** | **81.22 / 13.79** | **67.93 / 10.40** | 12/12 exact |
| Upstream Vulkan `f1793c1c4` | 240.53 / 22.97 | 259.73 / 20.11 | 266.98 / 18.07 | 12/12 exact |
| Patched-upstream HIP `f1793c1c4` | 239.23 / 17.74 | 301.68 / 16.88 | 294.47 / 14.77 | 12/12 exact; non-stock loader |
| EngramHalo HIP `1423f689` | 234.84 / 17.44 | 314.98 / 17.04 | 381.17 / 15.99 | p512/p1024 exact; p4096 fails |
| Nathan Vulkan `ad914eb` | 348.31 / 23.23 | 354.93 / 20.36 | 350.54 / 18.44 | diagnostic: 0/12 exact |
| apepojken Vulkan `843d575` | 291.73 / 23.21 | 375.23 / 22.42 | 397.43 / 22.25 | diagnostic: 8/12 exact |

Nathan produced 16 different outputs from 16 identical-prompt requests;
apepojken varies on four canonical cases; EngramHalo varies on one p4096 case.
Their affected rates remain diagnostics rather than correctness-valid targets.
Pristine upstream HIP did not finish loading in two 1,800-second attempts, so the
measured patched-upstream lane is explicitly non-stock. This is a three-repeat
screen, not section-6 closure: several rows exceed 2% CV, and five paired runs,
cold-PLE isolation, and category heldouts remain open.
[`canonical AR screening`](results/2026-08-30-gfx1151-qwen38-flash-next-canonical-ar-screening.json).

The frozen p508 role/API profile still puts hipEngine versus llama HIP device
kernels at **5.959 vs 1.625 s (3.67×)** and decode at **48.63 vs 38.90
ms/output (1.25×)**. The main p508 owners are MoE **3.161 s** (layers 0–26:
**2.526 s**), GDN **634.94 ms**, and QSA **110.49 ms**. The largest single miss
is layer-2 Q5_K gate/up at **301.47 vs 15.38 ms**. Decode submits **1,195
direct kernels plus 48 MoE graphs/token**; 625 additional rows/token are
graph-expanded nodes. A strict, layer-local stateful graph diagnostic now
captures one complete 34-kernel GDN+MoE physical layer: output and all request
state owners remain exact through four replays, while synchronized layer wall
falls **4.051→1.258 ms (3.22x)**. A chained layers-0..2 rung is likewise exact
and contracts **9.801→3.896 ms (2.52x)**. A fixed-position layers-0..3 mixed
GDN/QSA diagnostic remains device-state/output exact at **12.160→4.955 ms
(2.45x)**. Its advancing-position successor passes positions 8–11 across
position/context, K/V, QSA index, GDN state, and output at **13.882→4.974 ms
(2.79x)**. An eight-layer successor adds active PLE and a second QSA owner and
remains exact at **26.739→10.112 ms (2.64x)**. None is yet bound to production;
full-token and lifecycle gates remain open.
[`stateful layer graph`](results/2026-09-01-gfx1151-qwen38-flash-next-p8-stateful-layer-graph.json),
[`three-layer segment`](results/2026-09-01-gfx1151-qwen38-flash-next-p8-gdn-segment3-graph.json),
[`mixed fixed-position segment`](results/2026-09-01-gfx1151-qwen38-flash-next-p8-mixed-segment4-graph.json),
[`advancing mixed segment`](results/2026-09-01-gfx1151-qwen38-flash-next-p8-advancing-mixed-segment4-graph.json),
[`eight-layer segment`](results/2026-09-01-gfx1151-qwen38-flash-next-p8-advancing-segment8-graph.json).

A durable isolated-route recheck reopens the layer-2 grouped-WMMA candidate:
the p508 trace cuts layer-2 MoE **371.10→88.13 ms (4.21×)** and Q5_K gate/up
**279.86→16.66 ms**. Same-process p508 improves **90.25→95.06 tok/s
(+5.34%)**; all 20 category-balanced p512 pairs improve, with per-category
means **+4.83% to +5.20%** and every five-pair 95% CI above 1.0. Each route is
repeat-exact and keeps the same final top-1 token, but full logits differ. The
complete 450-row gate then **rejects** the T2 candidate: overall mean/p95/max KL
`5.03e-4/2.65e-3/0.01238` and 446/450 top-1 pass, as do every category,
repeat/state, and lifecycle checks, but the binding prefill-last/prefill-to-c1
mean KL is **0.001179 > 0.001**. The route remains default-off; c2 and depth
promotion gates were not run because they cannot compensate for this failure.
[`P1 layer-2 rejection`](results/2026-08-31-gfx1151-qwen38-flash-next-p1-layer2-grouped-profile-rejected.json).

The fresh P2 split keeps current production default-off for that candidate and
profiles layers 0–26 at **2.366 s**: exact Q4/Q5_K gate/up **1.200 s**, exact
Q5_1/Q8 down **1.152 s**, and activation plus routing/shared tails only
**13.25 ms**. Layers 3–26 alone retain **1.849 s**; active experts span 166–298
with median 9 rows per active expert. The next exact/T1 work therefore targets
multi-row weight reuse/output tiling in both projection halves, not the <0.6%
tail. Telemetry was collected separately and its D2H wall is excluded.
[`P2 early-MoE profile`](results/2026-08-31-gfx1151-qwen38-flash-next-p2-early-moe-profile.json).

The P3 split names another **1.670 s** of primary p508 roles outside routed MoE:
GR projection/read **709.32 ms**, Q8 `attn_qkv+attn_gate` **532.36 ms**, router
**181.91 ms**, `ssm_out` **137.84 ms**, and shared projections **121.61 ms**.
The first operation-complete target is the 36-layer qkv+gate boundary; it must
preserve current qkv-MMQ and exact-gate arithmetic or qualify a declared T1
pair, with both singleton routes retained as fallbacks. The first extension—Q8
MMQ on the omitted K2560/N6144 gate—wins **1.0352x** p508 and passes all
numerical scopes, but is rejected because candidate state repeat 1 differs from
repeats 2–3 on the first prompt. Ignoring the first same-schedule run as warmup
is not a valid production rule; exact coltile remains default. The next P3
subunit fuses GR sigmoid materialization with gated mean for rows <=256. It
removes one launch per GR read and improves clean counterbalanced
p508+128-step decode **14.162→15.111 tok/s (1.0670x, 95% CI
1.0543–1.0797)**. The complete T0 gate
is exact: **450/450 logits, 18/18 state/task prompts, three repeats, and clean
teardown**. Rows >256 remain unfused after a rows508 primitive loss. The
multirow F32 router projection also reuses each weight row across four prompt
rows while preserving dense arithmetic: clean p508 improves **89.689→91.121
tok/s (1.0160x, 95% CI 1.0143–1.0177)**, with 450/450 logits and 18/18 state/task
prompts exact. c1 remains on the dense owner. The rows>256 GR-up composite also
preserves the exact Q8 reduction while emitting sigmoid gates and branch mean:
clean p508 improves **91.158→91.600 tok/s (1.00484x)** and code-p1024
**88.754→89.239 tok/s (1.00547x)**, with 450/450 logits and 18/18 state/task
prompts exact. P4 also promotes the exact fixed256/precomputed-offset/vector2
QSA dense owner: the real primitive improves **6.846→2.485 ms (2.755x)**,
clean p508 **91.529→92.442 tok/s**, and code-p1024 **89.150→90.634 tok/s**, with
the complete exact/state/task gate passing. P5 moves normal greedy top-1 to the
device: Python-visible D2H falls from **993,280 to 8 bytes/token (124,160x)**
with 450/450 logits, 18/18 generated task sequences, compact state, physical-c2
outputs, and lifecycle exact. Resident-token chaining and normal-AR hidden-copy
elision then reduce the ledger from **28 to 26 blocking copies/token** while
preserving 12 async copies. The p508+128-step wall ratio is neutral at
**1.00343x (95% CI 0.98776–1.01909)**; this is a transfer-boundary retention,
not a wall-speed claim. The fresh canonical p512/p1024/p4096 snapshot is
**83.70/83.16/69.10 pp/s** and **14.40/14.42/10.42 tg/s**, all 36 measured
samples deterministic. Versus the same-host campaign start, pp improves
**1.44%/2.38%/1.72%** and tg improves **4.17%/4.60%/0.20%**.
[`P3 Q8-gate rejection`](results/2026-08-31-gfx1151-qwen38-flash-next-p3-q8-mmq-attn-gate-rejected.json).
[`P3 fused GR`](results/2026-08-31-gfx1151-qwen38-flash-next-p3-gr-sigmoid-mean.json).
[`P3 F32 router tile4`](results/2026-08-31-gfx1151-qwen38-flash-next-p3-router-f32-tile4.json).
[`P3 GR up+sigmoid+mean`](results/2026-08-31-gfx1151-qwen38-flash-next-p3-gr-up-sigmoid-mean.json).
[`P4 QSA dense fixed256`](results/2026-08-31-gfx1151-qwen38-flash-next-p4-qsa-dense-fixed256.json).
[`P5 device argmax`](results/2026-08-31-gfx1151-qwen38-flash-next-p5-device-argmax.json).
[`P5 current canonical AR`](results/2026-08-31-gfx1151-qwen38-flash-next-p5-current-canonical-ar.json).

P6 localizes the long-context cliff to indexed QSA activation. Identical
transition medians at live counts 2,051/2,052/4,097 are **66.61/95.88/96.02
ms**. The boundary adds **30.77 ms** of profiled kernel time; sparse attention
alone adds **27.47 ms**, while score/top-k adds **0.92 ms**. The nearly flat
2,052→4,097 result points to the fixed ~2K selected-attention budget rather than
continued context growth. [`P6 context profile`](results/2026-08-31-gfx1151-qwen38-flash-next-p6-context-transition-profile.json).

A same-weight external-fork refresh built EngramHalo HIP `1423f689` and
Nathan Vulkan `ad914eb` locally. BF16-KV p508/p1012/tg32 shape rows are
**296.12/362.72/17.62** and **413.04/396.25/23.85 tok/s**; Nathan's local
build agrees with its v0.7.2 payload within 1%. These are historical
`llama-bench` shape diagnostics, not exact-prompt or source-only A/B rows. Nathan
lazy-on/off averages **413.04/329.23 p508 (1.255x)** but converges by p1012;
an Engram MTP diagnostic is 1.128x complete-wall at 94.55% acceptance but only
**9/10** AR-message exact, so it is not a valid speed target.
[`external fork refresh`](results/2026-08-30-gfx1151-qwen38-flash-next-external-fork-refresh.json).

The cross-engine survey adds a 160-row, full-vocabulary, same-GGUF packet.
Current upstream HIP is 160/160 top-1 and effectively identical to frozen
#27742 HIP. EngramHalo is 159/160 with mean/max KL **9.85e-4/0.01431**.
Upstream Vulkan, Nathan, and apepojken are each 159/160 versus frozen HIP, but
Nathan is effectively identical to upstream Vulkan (160/160, mean KL about
**2e-10**) and apepojken remains 160/160 versus upstream Vulkan at mean/max KL
**0.00109/0.01576**. This localizes Nathan's failure to multi-step execution
rather than broad static math. Short Q8-KV apepojken MTP is **1.807x**
complete-wall at 92.8% acceptance but only **9/10** AR-message exact, matching
EngramHalo's failing prompt. Nathan MTP is provisionally **1.161x** at 95.45%
acceptance, but AR and MTP each self-repeat only 9/10 and just **8/10** prompts
match across both repeats of both modes. All affected speed rows are invalid as
targets. The survey also compares upstream/fork test coverage and
absolute-quality evidence.
[`Strix Halo survey artifact`](results/2026-08-31-gfx1151-qwen38-flash-next-strix-halo-survey.json).

The previous GDN decode-all claim is **invalid**: its selector was unreachable,
so the packet compared the strict owner to itself; the 16.2 tok/s helper also
used all-layer DP4A rather than admitted safe43. Wiring the actual candidate
costs **6.832 ms/token plus a 0.117-ms tail**, versus **2.454 ms/token** for the
retained GDN owner, and lowers full decode. Commit `15a436766` clears the dead
binder route. Prefill colwarps 27–47 remains certified. Current
production/strict manifests are `9e27fec0…` / `42509601…`; omitted routes stay
strict.
The certified
compact-WMMA MoE suffix (layers 27–47: Q4_K dual gate/up + Q5_1 down on the
f16-WMMA matrix-core kernels, tile 16×16; replaces the ds4-MMQ suffixes and
strict owners on those layers; `HIPENGINE_QWEN4_EXP_PRODUCTION_MOE_PREFILL=1`)
passes the complete 450-row/three-repeat packet at KL
mean/p95/p99/max `2.79e-4/1.53e-3/3.49e-3/5.98e-3`, **446/450 top-1** (all
scopes ≥ 98.67%), exact repeat/state, 18/18 repeat-exact free generation
(4 task-valid divergences), exact c2 with zero teardown, and improves paired
p508/p1012 **6.572→6.287 s (-4.34%, 80.82 tok/s)** /
**13.398→12.694 s (-5.26%, 79.73 tok/s)**. The layer-27 boundary is the
maximal envelope-admissible suffix (full-layer WMMA screens at mean 5.9e-3).
The certified GDN column-warp suffix (llama gated_delta_net layout, layers
27–47; 4.58× per launch, −17.1%/−15.7% paired p508/p1012, supersedes
peer-GDN) and the iu8-WMMA gate/up suffix (layers 35–47 within the WMMA-MoE27
route; exact Q4_K q values + 3 residual activation planes + min-offset
ds-trick) passes the complete packet at KL mean/p95/p99/max
`2.62e-4/2.20e-3/4.34e-3/5.52e-3`, **446/450 top-1**, zero scope failures,
exact repeat/state, 18/18 repeat-exact free generation (15/18 strict-exact),
exact c2 with zero teardown, and improves paired p508/p1012
**7.430→6.650 s (-10.5%)** / **15.260→13.469 s (-11.7%)** over the f16
production stack under matched conditions; the binder selects it via
`HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL=1` (layers 35–47). Current natural 16K improves **946.999→341.177 s (-63.96%, 47.989 tok/s; chunk-512 gate**
re-passed with retrieval/oracle/transactional/teardown exact) with every
binding control exact; 64K historical evidence is retained but not rerun because
47.989<100 tok/s. 262K is capacity-only (91.126 GB tracked), not inference. Q8 MTP is exact on 10/10
prompts but remains opt-in at **0.955x AR**. <=1K image/video/PNG chat and
request-owned c2 blocking/SSE pass with zero teardown; packed c-aware speed,
remote media, multimodal SSE, and 128K+/262K inference are not claimed.
Evidence: [`gap`](results/2026-08-28-gfx1151-qwen38-flash-next-llamacpp-matched-baseline.json) · [`MoE graph`](results/2026-08-29-gfx1151-qwen38-flash-next-exact-moe-graph-decode.json) · [`production`](results/2026-08-29-gfx1151-qwen38-flash-next-moe27-q8-32-production.json) · [`chunk512`](results/2026-08-29-gfx1151-qwen38-flash-next-prefill-chunk512.json) · [`Q8 MMQ`](results/2026-08-29-gfx1151-qwen38-flash-next-q8-mmq-prefill-production.json) · [`Q5_1 MMQ`](results/2026-08-29-gfx1151-qwen38-flash-next-q5-1-mmq-suffix32-production.json) · [`Q4_K MMQ`](results/2026-08-29-gfx1151-qwen38-flash-next-q4-k-mmq-suffix35-production.json) · [`MMQ+DP4A stack`](results/2026-08-29-gfx1151-qwen38-flash-next-production-mmq-prefill-dp4a43-stack.json) · [`profile manifest`](results/2026-08-29-gfx1151-qwen38-flash-next-production-mmq-profile-manifest.json) · [`peer GDN`](results/2026-08-29-gfx1151-qwen38-flash-next-production-gdn-peer35.json) · [`final campaign`](results/2026-08-29-gfx1151-qwen38-flash-next-prefill-mmq-campaign-final.json) · [`master re-baseline`](results/2026-08-29-gfx1151-qwen38-flash-next-llamacpp-master-rebaseline.json) · [`WMMA MoE27`](results/2026-08-29-gfx1151-qwen38-flash-next-wmma-moe27-production.json) · [`iu8 gate35`](results/2026-08-30-gfx1151-qwen38-flash-next-iu8-wmma-gate35-production.json) · [`GDN colwarps27`](results/2026-08-30-gfx1151-qwen38-flash-next-gdn-colwarps27-production.json) · [`QSA flash (key-parallel 35-47)`](results/2026-08-30-gfx1151-qwen38-flash-next-qsa-flash31-production.json) · [`fresh full profile`](results/2026-08-30-gfx1151-qwen38-flash-next-fresh-full-profile.json) · [`invalid GDN decode correction`](results/2026-08-30-gfx1151-qwen38-flash-next-gdn-colwarps-decode-all.json).

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
the ten-prompt gate is exact. [`Current-default evidence`](results/2026-08-23-w7900-qwen36-27b-current-default-publication.json). Qwen3.8 details remain in the XTX tables above.

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
| W7900 / Qwen3.6-27B Dense `Q4_K_M` | Exact/default natural25 B3 | 29.457 | **60.929** | **2.0684x** | Current clean snapshot; all ten prompts, greedy outputs, and GPU/CPU acceptance agree. The ratio replaces stale historical denominators. [`artifact`](results/2026-08-23-w7900-qwen36-27b-current-default-publication.json) |
| RX 7900 XTX / Qwen3.8-27B Dense `Q4_K_M` | Exact/default natural25 B3 | 35.287 | **62.440** | **1.7695x** | Clean idle-card correction; exact greedy and GPU/CPU acceptance, retained fusion improves matched AR 3.764% and B3 0.439% with every category non-regressive. [`artifact`](results/2026-08-15-qwen38-27b-xtx-clean-idle-performance-correction.json) |
| Radeon 8060S / Qwen3.8-27B Dense `Q4_K_M` | Exact natural25 B3 | 11.692 | **21.158** | **1.8095x** | Clean current-main direct-leaf snapshot; all ten prompts and 30 MTP comparisons are exact, GPU/CPU acceptance agrees, and cached profiling confirms the qualified scalar-C1 and native Q4 rows4/2 owners. [`artifact`](results/2026-08-26-gfx1151-qwen38-current-main-ar-mtp.json) |
| Radeon 8060S / Qwen3.8-27B Dense `Q4_K_M` | Public LLM strict/BF16 C1 natural25 B3, artifact-scoped automatic | 9.025 | **12.940** | **1.4337x** | Complete request through terminal reclaim: 30/30 exact cells, every category/heldout positive and every cell 1.2995x–1.5515x. Exact hash/profile/BF16/C1/B3/context1-67/natural25 auto-promotes after lifecycle/SSE/load qualification; every other scope is K0. [`artifact`](results/2026-08-26-gfx1151-qwen38-q4km-mtp-serving-s4-auto.json) |
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

Workloads use `prompt_tokens/decode_tokens`. Compare only matching timing,
model/quant/KV, concurrency, and memory scopes; bold identifies the reported
row, not a universal leader.

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
