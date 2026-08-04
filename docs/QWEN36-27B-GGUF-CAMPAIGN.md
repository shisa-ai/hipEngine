# Qwen3.6-27B Q4_K_M GGUF Optimization Campaign

Status: active, measured optimization (2026-08-04); clean exact hipEngine AR/MTP baseline and reconciled profiles complete.

Canonical target:
`/models/gguf/Qwen3.6-27B-Q4_K_M.gguf` on AMD Radeon Pro W7900 / GPU0 /
`gfx1100`.

Comparator: latest tracked-clean llama.cpp Vulkan source at
`/home/lhl/llama.cpp/llama.cpp-vulkan`, initially refreshed from
`67d5978bb` to `ee0445c99` on 2026-08-04 and built in Release mode with
`GGML_VULKAN=ON` and `GGML_HIP=OFF`.

The goal is an honest, same-file comparison: make hipEngine AR and native GGUF
MTP functional for this dense Qwen3.6 file, then meet or beat current
llama.cpp Vulkan for matched prefill and decode. Optimization order is always
set by measured complete-wall Amdahl impact, not by novelty, launch count, or an
isolated microbenchmark.

Related current authorities:

- [`BENCHMARK.md`](BENCHMARK.md) — evidence, timing, correctness, and anti-gaming
  contract.
- [`MTP-gguf.md`](MTP-gguf.md) and
  [`MTP-LLAMACPP-PARITY.md`](MTP-LLAMACPP-PARITY.md) — existing Qwen3.6 MoE GGUF
  MTP state/transaction work and cross-engine timing boundary.
- [`TUNING-gguf.md`](TUNING-gguf.md) — current GGUF measurement discipline and
  the closed Qwen3.6-35B-A3B tuning pass.
- [`OPTIMIZE-DENSE.md`](OPTIMIZE-DENSE.md) — older 27B PARO prefill plan; useful
  structural hypotheses, but not a same-quant baseline.
- [`LESSONS-LEARNED.md`](LESSONS-LEARNED.md) — retained and rejected RDNA3,
  Vulkan, graph, and speculative-decode lessons.
- [`HIP-vs-VULKAN.md`](HIP-vs-VULKAN.md) — current backend attribution rules.

---

## 1. Definition of done

The campaign closes only when all of the following are true on **GPU0 W7900**:

1. **Same model identity.** Both engines load the exact same GGUF fingerprint.
   The file is 17,106,773,120 bytes with SHA-256
   `a7cbd3ecc0e3f9b333edee61ae66bc87ed713c5d49587a8355814722ed329e0f`;
   retained artifacts must also add its tensor-inventory hash.
2. **AR works and is correct.** hipEngine runs resident bulk prefill and true AR
   decode for the 512/128 and 4096/128 gates, with finite logits, deterministic
   generated IDs, graph/eager state validation, and the applicable KL/top-1
   oracle.
3. **MTP works and is honest.** The dense trailing NextN block runs through the
   shared GGUF MTP proposal, target verify, accept/commit, rollback, and reseed
   contracts. The full 10-prompt category suite plus heldouts reports a true
   same-protocol AR denominator, acceptance ledgers, exact/default state
   semantics, and complete cycle wall.
4. **Matched performance.** hipEngine is at least as fast as latest llama.cpp
   Vulkan on both matched prefill gates and both matched AR/MTP decode gates.
   A win in one column does not hide a regression in another.
5. **Profiles reconcile.** Both engines have functional fine-grained profiles
   whose timed components reconcile closely enough to complete wall to rank the
   next bottleneck. Profiler-perturbed numbers are never used as topline speed.
6. **Default path.** Exact, same-suite, non-regressive wins are promoted to the
   production registry route. Temporary selectors/fallbacks have explicit
   removal conditions in `docs/REFACTOR.md`.

A compatibility/accuracy-traded MTP route may be reported separately, but it
cannot satisfy the exact/default closure gate.

---

## 2. Fixed model and hardware identity

### 2.1 GGUF inventory

The local scanner reports:

| Field | Value |
| --- | --- |
| GGUF architecture | `qwen35` |
| Quant | `MOSTLY_Q4_K_M` |
| Tensor count | `866` |
| Executable AR blocks | `64` |
| Declared blocks | `65` |
| Trailing NextN block | `blk.64` |
| Hidden / dense FFN | `5120 / 17408` |
| Layer mix | `48 linear_attention + 16 full_attention` |
| Q / KV heads, head dim | `24 / 4, 256` |
| Vocabulary | `248320` |
| Experts | `0` (dense) |
| NextN tensors | 15 total in `blk.64`, including 4 `nextn.*` tensors |

The NextN layer is itself dense: Q4_K gate/up, Q6_K down, full attention, and
`nextn.eh_proj/enorm/hnorm/shared_head_norm`. The optional NextN embedding and
head tensors are absent: embedding falls back to `token_embd.weight`, while the
head must fall back to this model's distinct Q6_K `output.weight` (the file is
untied).

### 2.2 Device policy

| Device | Role | Rule |
| --- | --- | --- |
| GPU0, Radeon Pro W7900, 48 GiB | Canonical baseline, retained profiles, final comparisons | Every llama.cpp-vs-hipEngine performance ratio is measured here. |
| GPU1, Radeon RX 7900 XTX, 24 GiB | Parallel build, lightweight correctness, micro-screening | May be used when the allocation fits. The current full resident 27B path does not fit (25.69 GiB owned / 26.90 GiB sampled on GPU0). A result is compared only with a same-GPU control and never promoted as a W7900 ratio. |

Both devices are `gfx1100`, so code correctness and many shape decisions
transfer. Clock, memory capacity, firmware residency, and absolute throughput do
not. Every artifact records physical device name, logical selector, and VRAM.
Avoid simultaneous performance measurements on the same physical GPU. GPU0 and
GPU1 work may overlap only when CPU/I/O contention is either irrelevant
(functional smoke) or explicitly excluded from retained timing.

### 2.3 Software identity

Canonical llama.cpp uses Vulkan device `Vulkan0`, which enumerates as the W7900,
with RADV/Mesa `26.1.4`. Canonical hipEngine uses `HIP_VISIBLE_DEVICES=0`,
target `gfx1100`, and the hermetic TheRock environment from
`scripts/run_w7900_readme_refresh.sh`.

The refreshed Release build completed successfully. Initial binary SHA-256s are
`0d466d22...98759` (`llama-bench`) and `e83a9d8d...5a1e3`
(`llama-server`); retained artifacts record the full hashes.

A compiler/runtime/driver update opens a new baseline row; it never silently
replaces the old denominator.

---

## 3. One-to-one benchmark contract

### 3.1 AR prefill/decode matrix

Two complementary rows are required:

1. **Standardized kernel-oriented baseline:** llama-bench `pp512`, `pp4096`, and
   `tg128`, one discarded warmup and at least five measurements.
2. **Context-matched request baseline:** exact token prompt `[9707] * N`,
   `N in {512,4096}`, 128 requested outputs, greedy/top-1, EOS ignored, prompt
   cache disabled. This captures context-dependent decode and the real server
   path that standalone `tg128` does not.

hipEngine uses the same token IDs and lengths, one resident model, reset between
runs, bulk prefill, production one-step state-bound graph decode, and separate
prefill/decode timers. Graph capture/instantiate/destruction is recorded
separately and excluded from steady-state decode only when llama.cpp's compared
row also excludes one-time setup.

Required columns:

- prompt tokens, returned/timed output transitions, and exact token IDs;
- prefill ms and tok/s;
- decode complete wall ms/token and tok/s;
- graph/capture/startup scope;
- model load and peak memory outside the throughput denominator;
- warmup and all raw measurement samples.

### 3.2 Natural-prompt MTP matrix

The canonical MTP gate is the committed 10-prompt
`benchmarks/prompts/mtpbench-code-general-ja.jsonl` suite. It covers `code`,
`general_en`, `general_ja`, and `mixed_ja_en`, with the fixed six-train/four-
heldout split from `BENCHMARK.md`.

Initial settings:

- greedy: temperature `0`, top-k `1`, top-p `1`, min-p `0`, seed `12345`;
- reasoning `off` on both engines;
- f16 K/V in llama.cpp; hipEngine's current exact/default BF16 K/V is disclosed
  as the one remaining execution-format difference;
- llama.cpp `--spec-type draft-mtp`;
- start with B2 (`--spec-draft-n-max 2`) because the historical 27B natural
  suite favored B2, then sweep B1-B4 on the new build rather than inheriting the
  old optimum;
- request 25 outputs and report 24 timed decode transitions for the cross-engine
  table, following `MTP-LLAMACPP-PARITY.md`;
- retain client/request wall separately from engine-reported generation wall.

Every MTP row includes proposed drafts, accepted drafts, accepted/output,
visible outputs, target passes, draft/target/accept-commit/replay timings, and
complete cycle wall. A no-MTP run from the same harness/suite is the only valid
MTP speedup denominator. Token-repeat MTP at 512/128 and 4096/128 remains an
artificial perfect-acceptance diagnostic, never the natural-prompt headline.

### 3.3 Comparison rules

- Same GGUF, prompt token IDs, output horizon, GPU, backend revision, driver,
  cache policy, sampling, warmup, and timing boundary or the cells are marked
  non-comparable.
- `llama.cpp predicted_n / predicted_ms` includes one untimed first token. Use
  transition-normalized fields for cross-engine decode.
- hipEngine BF16 KV vs llama.cpp F16 KV is always visible in the table. Do not
  describe it as bit-identical execution.
- A profile and a throughput run are separate runs. Timestamp-query, debug,
  validation, or trace overhead never enters a topline.
- Single-prompt or repeated-token results can bring up a path but cannot retain
  an MTP optimization.

---

## 4. Fine-grained profiling contract

### 4.1 llama.cpp Vulkan

Use three levels, in order:

1. **Request phase timing:** `llama-server --perf` response timings for prompt
   and generation, plus client wall from `scripts/llamacpp_mtp_bench.py`.
2. **Per-GGML-op GPU timestamps:** run a short, isolated leaf request with
   `GGML_VK_PERF_LOGGER=1`. The upstream Vulkan backend uses a query pool and
   prints operation/fusion name, call count, mean microseconds, total
   microseconds, and aggregate GPU time. Keep the default non-concurrent logger
   for per-op attribution. Parse it into a compact Amdahl JSON grouped by phase,
   op, shape, count, and total time.
3. **Escalation only if unresolved:** enable `GGML_VK_DEBUG_MARKERS=1` and a
   Vulkan/RADV trace or use a separate instrumented llama.cpp worktree. The
   existing `LLAMA_MTP_STAGE_TIMINGS` patch is not present in clean upstream
   `ee0445c99`; do not claim those fields from the clean binary. Apply/update an
   instrumentation patch only in a separate profile build and keep the clean
   binary as the speed denominator.

The Vulkan perf logger synchronizes for query results and can perturb graph
submission. Its output ranks work; it does not establish throughput.

### 4.2 hipEngine

- AR eager/graph leaf: `scripts/gguf_decode_rocprof.py`,
  `scripts/gguf_packed_ar_rocprof.py`, or selected regions in
  `scripts/qwen35_gguf_bench.py`.
- Dense MTP leaf: run `scripts/qwen36_dense_gguf_suite.py --limit 1
  --no-warmup --roctx-markers` directly under rocprofv3 and slice its nested
  proposal/verify/commit marker windows. The older
  `scripts/gguf_mtp_draft_rocprof.py` and
  `scripts/gguf_mtp_verifier_rocprof.py` remain MoE-oriented references. Never
  profile a parent prompt-suite process.
- Prebuild every JIT object outside rocprofv3, pass a compiler-version file, and
  require cached builds.
- Compact summaries report kernel family, exact symbol, calls/output, total and
  per-output time, wall share, grid/workgroup, VGPR/SGPR/LDS/scratch when
  available, and unmatched residual wall.

### 4.3 Reconciliation gate

For each AR and MTP profile, record:

```text
complete wall = GPU kernel/query sum + host/submission/sync residual
```

If the components differ from complete wall by more than 10%, first explain
queue overlap, asynchronous timestamps, untimed sampling, or synchronization
boundaries. Do not choose a kernel target from an unreconciled trace.

---

## 5. Prior work: what transfers and what does not

### 5.1 Transfer directly from Qwen3.5/Qwen3.6 MoE GGUF

- GGUF scanner, quant metadata, raw/replacement weight ownership, Q4_K/Q6_K/Q8
  kernels, T16/X8 decode layouts, rows>1 MMQ/WMMA prefill, and resident memory
  accounting.
- Hybrid 48-GDN/16-full-attention execution, Conv/GDN state, paged K/V,
  state-bound one-step graph replay, and exact graph/eager state oracles.
- MTP hidden-seed, proposal, target verify, transactional K/V, accept/commit,
  rollback/reseed, natural category suite, transition-normalized llama.cpp
  comparison, and B1/B2 graph ownership.
- Audit discipline: profile first, rank by time share, keep exact additive
  micro-wins, and re-profile after structural changes.

### 5.2 Dense-specific differences and remaining work

- NextN maps are now architecture-shaped: the real `blk.64` binds dense
  `ffn_gate`, `ffn_up`, and `ffn_down` rather than MoE router/expert/shared-
  expert slots.
- Dense NextN materialization and one-step execution reuse the registered dense
  FFN chain rather than emulating one expert or adding dispatch branches.
- Dense verifier rows have no router/group/scatter cost; whether gate/up/down,
  the Q6_K head, target verification, or host transaction leads complete MTP
  wall remains for the reconciled D27-M1 profile to establish.
- Dense model bandwidth and FFN shapes differ materially from sparse MoE. MoE
  expert compaction, selected-expert sidecars, router tuning, and selected-lane
  kernels do not transfer.

### 5.3 Lessons that constrain this campaign

1. **Layout before dot intrinsics.** The largest Q4 decode gains came from
   coalesced/replacement layouts; direct dp4a retries lost when activation prep
   or layout was wrong.
2. **M=1 is not WMMA by default.** Use GEMV/vector work for AR decode; use
   MMQ/WMMA for prefill or multi-row verification only after shape measurement.
3. **Launch removal is not enough.** Prior megakernels and graph segmentation
   lost by reducing occupancy or breaking state. Fusion must remove measured
   memory traffic while preserving the fast layout.
4. **One-step state-bound graph is the safe baseline.** Multi-step replay has
   produced token/state drift.
5. **Speculative economics need a complete ledger.** Acceptance alone and
   verifier-derived B0 rows are not speed evidence.
6. **No single-prompt optimization.** Full/train/heldout/category gates decide
   every MTP keep/revert.
7. **Profile after every structural keep.** The top bucket changes; historical
   MoE or PARO profiles cannot choose this dense GGUF target.
8. **Reject tiny ceilings early.** LDS, broad geometry, wave64, generic
   scheduler flags, and allocation-only output buffers need a current measured
   bottleneck before any implementation.

---

## 6. Current bring-up finding

AR block discovery correctly identifies 64 executable blocks and excludes
trailing `blk.64`. The first bring-up then exposed and fixed four generic
contract gaps rather than bypassing them:

1. dense root mapping now selects an untied `output.weight` when present;
2. Q4_K token embeddings stay raw GGUF so the registered lookup kernel can read
   them instead of receiving a linear-only pack8 layout;
3. the small-row native full-attention resolver no longer accepts a
   `cpu_reference` fallback for device pointers; and
4. GDN decode inserts the existing unfused FP32-to-BF16 cast only when the
   resident `ssm_out` layout cannot consume FP32 directly.

The hermetic W7900 eager 8/1 smoke is now green: finite logits, final token
`9707`, 0.23738 s prefill, 0.04907 s decode, 25.69 GiB tracked resident peak,
and 26.90 GiB sampled HIP use. The 33.70 prefill tok/s and 20.38 decode tok/s
are **bring-up diagnostics only** (one sample, tiny prompt, no graph), not
campaign performance claims. The same full allocation cannot fit GPU1's 24 GiB
VRAM; GPU1 remains useful for smaller/component work.

The clean external Vulkan floor is now complete. llama-bench measures
**792.308 pp512 / 754.093 pp4096 / 12.61795 tg128 tok/s**. The separate
stateful server boundary measures **79.805 / 81.792 prefill tok/s** and
**12.57431 / 12.48779 transition-normalized AR tok/s** at 512/128 and
4096/128. Natural25 selects B3 at **68.082 MTP vs 12.546 AR tok/s (5.4265x)**
engine time and **36.122 vs 9.607 tok/s (3.7600x)** client time. Query-timestamp
profiles reconcile within 3%; dense FFN leads base 512/8 and sampled B3 at
49.04% and 46.52% of query time. Full evidence is in
`benchmarks/results/2026-08-04-qwen36-27b-llamacpp-vulkan-baseline.json`.

Architecture-shaped dense NextN mapping is now green. Both GGUF MTP map
surfaces select the real 15-tensor `blk.64` contract, bind dense
`ffn_gate/ffn_up/ffn_down` instead of router/expert slots, preserve the untied
Q6_K `output.weight` fallback, validate the target's Q4_K/Q6_K/Q8_0 mix, and
emit a 17-slot call spec including embedding/head fallbacks. The existing
20-tensor MoE fixtures remain unchanged.

The independent dense CPU SwiGLU sublayer and full NextN composition are also
registered behind the dense call-spec key; the manual F32 chain and signature
binding are green. The resident GPU executor now runs the real dense `blk.64`
through the existing full-attention+dense-FFN route. At token `9707`, position
zero, zero target hidden, and empty draft KV, hipEngine and clean llama.cpp
Vulkan agree on every top-10 ID and top-1 `46424`; the full-vocabulary
llama-to-HIP KL is **0.001566**, top-1 agreement is **100%**, and maximum logit
absolute difference is **0.2541**. Direct execution and provider replay are
FP32 array-exact. The committed compact oracle is
`tests/fixtures/gguf/qwen36_27b_q4km_nextn_one_step_oracle.json`.

Dense hipEngine MTP is now end-to-end functional through the shared transaction
ABI. The `gguf_q4_k_m` accept key resolves explicitly; dense graph keys publish
zero experts rather than the inherited MoE top-8 shape. B1/B2/B3 target logits
are FP32 array-exact to scalar target execution. Reject, partial, full, and
rollback restore the selected Conv/GDN state, live full-attention K/V prefix,
hidden tap, cursor, and next correction logits byte-for-byte. A natural B1
8-output smoke matches all AR IDs, accepts three draft tokens over five cycles,
and exercises full-accept draft-tail advancement. It is a single-prompt
functional gate, not a performance result; the full category suite remains
mandatory before any MTP economics claim.

A cached GPU1 one-step trace confirms the expected dense Q4_K gate/up and Q6_K
down/head symbols; its two executions total 63 launches / 12.544 ms of kernels,
with the Q6_K full-vocabulary head accounting for 9.803 ms.

The matched dense natural-suite leaf is now available as
`scripts/qwen36_dense_gguf_suite.py`. It uses the committed 10-prompt fixture,
Qwen chat rendering with reasoning off, one true scalar-AR denominator, B1-B3
transactional MTP, the fixed train/heldout/category split, 24 timed transitions
for 25 visible outputs, complete proposal/verify/commit/residual accounting, and
optional nested ROCTX markers.

The final clean `da6865f74` W7900 baseline closes D27-F0 and D27-M1. At
512/128 and 4096/128, median hipEngine prefill is **50.515 / 50.473 tok/s**
and graph AR decode is **19.556 / 18.649 tok/s** over three measured resets.
Every final ID is `9707`; measured graph windows reuse the one 49-ms capture,
and tracked peaks are 26.123 / 28.947 GiB. Against the matched stateful Vulkan
rows, hipEngine prefill is **36.70% / 38.29% slower**, while decode is **55.52% /
49.34% faster**. This is therefore a retained exact baseline, not parity.

On the full natural25 suite, true AR is **20.361 tok/s**. Exact B1/B2/B3 MTP is
**17.128 / 16.005 / 14.858 tok/s**, or **0.8412x / 0.7861x / 0.7297x** own AR.
All 250 visible IDs at every budget and every GPU/CPU accept summary match, but
no MTP budget is speed-eligible. B1 is merely the least-slow control; Vulkan B3
remains **68.082 tok/s**, leaving hipEngine exact MTP 74.84% behind.

The profiles select the first optimization without ambiguity. The 512 prefill
trace reconciles 9,911.076 ms of kernels to 9,944.618 ms host wall (0.34%
residual): Q4_K pack8 projections consume **78.86%** and dense BF16 GEMM
**20.05%**. The 16-step AR graph trace records 752.607 ms kernels in 843.238 ms
host wall; its explained 10.75% submission/queue-gap residual spans 15,968
dispatches, while Q4_K row kernels plus dense BF16 GEMV consume **85.36%** of
kernel time. Most importantly, the natural B3 marker trace reconciles its
1,590.971-ms host wall to a 1,590.457-ms device-activity span. Target verify is
**92.58%** of host wall; Q4_K row kernels and dense BF16 GEMV are **82.73%** of
kernel time. D27-O3 target row batching/projection routing is therefore first,
followed by D27-O1 bulk Q4_K/BF16 prefill. Full evidence is retained in
`benchmarks/results/2026-08-04-qwen36-27b-hipengine-baseline.json`.

The first D27-O3 optimization is retained. A 2-4-row dense-BF16 kernel loads
each weight once while preserving the c1 pack8 FMA order and 256-thread
reduction tree for every row. On RX 7900 XTX/GPU1 it is 3.44-7.85x faster than
the prior small-row prefill GEMM across the four real dense shapes. Real
B1/B2/B3 target logits, trunk hidden rows, and all 129 Conv/GDN/KV/hidden
buffers are byte-exact to scalar execution; the complete W7900 reject/partial/
full/rollback/provider gate also passes.

Clean W7900 natural25 promotes native row-attention/block-FFN verification as
the dense suite default, with `serial-exact` as the rollback control. B1/B2/B3
improve **17.128/16.005/14.858 -> 18.751/18.752/17.983 tok/s**
(**+9.48%/+17.17%/+21.03%**) while true AR is unchanged at 20.362 tok/s. Every
train, heldout, and category row improves; all 250 IDs per budget and GPU/CPU
accept summaries remain exact. Target-verify wall falls 9.28%/15.78%/18.70%.
The accepted per-row state capture costs +0.603 GiB tracked peak (28.995 GiB,
zero after close). B1/B2 remain effectively tied at ~0.921x AR, so D27-O3
continues from a refreshed profile rather than declaring budget victory.
Artifact: `benchmarks/results/2026-08-04-qwen36-27b-native-target-rowtile-retained.json`.

The second D27-O3 projection win is also retained. An exact local32 resident-
Q4_K pack8 rowtile loads each gate/up weight once across verifier rows 2-4
while preserving the original per-row FMA/shuffle/BF16 boundary. The first clean
screen at `2fadf425c` was correctly blocked: the kernel was dormant because the
native target verifier had not entered its small-row dispatch session. The
routed `4181b85fb` transaction gate observes all `{2,3,4}` rows and remains
byte-exact for logits, Conv/GDN/KV/hidden state, commits, and provider output.

Clean W7900 natural25 improves the prior native B1/B2/B3
**18.751/18.752/17.983 -> 20.634/21.752/21.467 tok/s**
(**+10.04%/+16.00%/+19.38%**), with target verify down
10.16%/15.14%/17.91%. Every full/train/heldout/category row improves
(**+9.78% to +19.94%**); IDs and acceptance remain exact, true AR is flat at
20.372 tok/s, and peak remains 28.995 GiB. All three budgets now beat own AR;
B2 is the clear exact winner at **1.0678x**. The retained B3 re-profile still
assigns 89.57% of complete wall to target verify and promotes the serial exact
dense-BF16 GEMV (**238.766 ms / 19.63%** of wall) as the next single-kernel
audit. Artifact:
`benchmarks/results/2026-08-04-qwen36-27b-resident-q4-rowtile-retained.json`.

The third exact D27-O3 kernel is retained. A local128 block keeps the original
256 arithmetic partitions two per physical thread, performs the s=128 combine
in registers, then reproduces the s=64/32 LDS and 16..1 wave trees. The one-wave
local32 prototype was exact but only 0.47-0.93x local256 and was removed; an
isolated exact local64 sibling is 0.70-0.91x local128 on three major shapes and
is also rejected. GPU1 local128 improves qualifying production shapes by
**1.091-1.103x**. A broader screen wins through K=10,240 and crosses negative
at K=12,288, so larger K fails closed to local256.

Clean W7900 natural25 improves retained B1/B2/B3
**20.634/21.752/21.467 -> 20.846/22.102/21.840 tok/s**
(**+1.03%/+1.61%/+1.74%**) with target verify down
**1.10%/1.81%/1.78%**. Every prompt, full/train/heldout, and category row
improves; IDs, acceptance, transaction state, and 28.995-GiB peak remain exact.
True AR is noise-flat at 20.362 tok/s. B2 remains selected at **1.0854x** own
AR. Artifact:
`benchmarks/results/2026-08-04-qwen36-27b-dense-local128-retained.json`.

The clean post-local128 B3 trace now promotes submission structure above another
arithmetic microkernel. It reconciles 1,219.266 ms host to 1,218.734 ms device
activity. Target verify is **1,097.438 ms / 90.01%** of complete wall, but its
24,619 launches contain only 700.491 ms of kernels: the **396.947-ms target
host-minus-kernel gap is 32.56% of complete wall** and exceeds every remaining
kernel family. The next admitted candidate is therefore an exact reusable
native-verifier HIP graph with dynamic per-row token/position/`KVLiveSpans`
metadata, B1-B3 buckets, and exact Conv/GDN plus pre-output-norm trunk-row
capture. A conservative 50% gap recovery has a 198.474-ms / 16.28% complete-
wall saving. The existing bulk-only B1/B2 N1 graph guards cannot simply be
relaxed because its captured host positions are invalid for native row-serial
full attention; unsupported configurations must keep the eager native fallback.

The exact dense graph is now retained. Independent B1-B3 N1 executables bind
native row attention to live device positions, contexts, resident-page
`KVLiveSpans`, and a graph-owned FP32 trunk-row journal; N2 remains B1/B2. The
real W7900 transaction oracle passes B3 reject, partial, and full acceptance plus
reuse at positions 6, 7, and 9, with exact target rows, selected Conv/GDN/KV/
hidden state, correction logits, and natural B1 output.

Clean W7900 natural25 moves B1/B2/B3
**20.846/22.102/21.840 -> 23.225/24.820/25.193 tok/s**
(**+11.41%/+12.30%/+15.35%**) and target verify falls
**11.10%/12.11%/14.82%**. Every prompt, full/train/heldout, and category row
improves; IDs, acceptance, stage reconciliation, and the complete transaction
stay exact. Peak rises only 0.0007 GiB to 28.996 GiB and frees completely. B3
is now selected at **1.2362x own AR**, though still 63.00% below Vulkan B3.
Artifact: `benchmarks/results/2026-08-04-qwen36-27b-native-verifier-graph-retained.json`.

The post-graph B3 trace now moves complete profiled wall
**1,219.266 -> 1,075.551 ms (-11.79%)** and target verify
**1,097.438 -> 951.747 ms (-13.28%)**. Kernel sum is near-flat while the
queue-gap/copy-overlap bucket falls 135.320 ms; this validates submission
ownership. Target verify still owns 88.49% of wall, but its next measured target
is arithmetic: exact row-serial dense-BF16 plus Q4 singleton projections total
**379.327 ms / 35.27%** of wall. The linear-attention subset is
**339.579 ms / 31.57%**. D27-O3 therefore admits staged exact linear projections:
bulk independent norm/QKV/gate/alpha/beta, preserve serial Conv/GDN and each
state-journal boundary, then bulk exact `ssm_out`. A conservative 50% recovery
is 169.790 ms / 15.79% of wall. Full-attention staging is deferred until this
narrower transaction gate passes.

The staged dense linear scheduler is now retained. It bulks only independent
projections, then runs unchanged Conv/GDN decode kernels and each state-row copy
in token order before one exact row-bulk `ssm_out`; MoE and c1 retain scalar
fallback. The complete W7900 oracle preserves B1-B3 full logits,
reject/partial/full and positions 6/7/9 state/KV/hidden transactions, graph
submission, correction logits, and natural provider output.

Clean W7900 natural25 moves B1/B2/B3
**23.225/24.820/25.193 -> 27.734/33.544/36.652 tok/s**
(**+19.42%/+35.15%/+45.49%**) and target verify falls
**18.22%/29.49%/35.64%**. Every prompt and full/train/heldout/category rollup
improves, all IDs/acceptance/stage/transaction gates remain exact, and memory is
byte-identical at 28.996 GiB. GPU1 Q4 QKV/gate rows2-4 are independently
bit-exact and 1.17-1.67x faster. B3 is selected at **1.8047x own AR**, still
46.17% below Vulkan B3. Artifact:
`benchmarks/results/2026-08-04-qwen36-27b-staged-linear-projections-retained.json`.

The refreshed retained B3 trace confirms the gain and admits the next boundary.
Complete profiled wall falls **1,075.551 -> 742.080 ms (-31.01%)**, target
verify **951.747 -> 621.704 ms (-34.68%)**, and target dispatches
**23,318 -> 16,262 (-30.26%)**. The next untried row-serial boundary is the 16
full-attention layers: Q4 singleton projections plus per-row norm/split/key
cast/head-norm-RoPE and dense V total about **49 ms / 6.6%** of complete wall,
while attention+gate+KV itself is only 9.018 ms and must remain ordered. GPU1
component screening makes the qualified design precise: exact full-Q and output
rowtiles improve 1.35-1.67x and dense V improves 1.09-1.76x, but narrow Q4 K
regresses to 0.83-0.90x. D27-O3 therefore admits bulk norm/Q/V/split/rotary and
output around serial per-row K plus KV-append -> attention -> gate. Dense c>1
native rows only; the complete scalar helper remains fallback.

That staged full-attention schedule is now retained. CPU coverage proves
row-bulk Q/V/O, scalar K, strict per-row KV-write -> attention -> gate ordering,
and all eager/MoE/c1 fallbacks. The complete W7900 transaction oracle preserves
B1-B3 logits, graph execution, reject/partial/full and positions 6/7/9
state/KV/hidden commits, correction logits, and natural provider output.

Clean W7900 natural25 moves B1/B2/B3
**27.734/33.544/36.652 -> 28.348/34.818/38.322 tok/s**
(**+2.21%/+3.80%/+4.56%**) and target verify falls
**2.54%/4.37%/5.61%**. Every prompt and full/train/heldout/category rollup
improves, all IDs/acceptance/stage/transaction gates remain exact, and memory is
byte-identical at 28.996 GiB. B3 is selected at **1.8821x own AR**, still 43.71%
below Vulkan B3. Artifact:
`benchmarks/results/2026-08-04-qwen36-27b-staged-full-attention-retained.json`.

The refreshed exact B3 trace after that promotion measures **693.517 ms**
complete wall, **568.742 ms / 82.01%** target verify, and **500.475 ms** of
kernels across 14,613 dispatches. The staged target remains dominated by
already-bulk Q4 and dense-BF16 rowtiles; its next independent schedule is not as
clear as the proposal boundary. Proposal now owns **112.621 ms / 16.24%** of
wall. Its raw-Q6 family is **73.493 ms**, including **67.138 ms / 25 launches /
9.68% of complete wall** in full-vocabulary FP32 LM-head scoring followed by a
full-row host readback and NumPy argmax.

D27-O3 therefore admits compact **exact** proposal scoring next. Supported
raw-Q6 heads with `return_logits=False` may reuse the registered BF16 x Q6_K
pack-winner/final-top1 primitive and read back only token plus winner value.
Full logits remain the diagnostic and unsupported-weight fallback. The measured
ceiling is 67.138 ms plus full-logit drain/readback; no quantized-activation,
vocabulary-cap, or prompt-specific approximation is admitted. GPU1 may screen
the same-gfx1100 component, but retention still requires the clean W7900
natural25 B1-B3 gate against the current 38.322 tok/s baseline.

The compact exact proposal path is now retained. CPU/fake tests bind it to the
four-axis raw-Q6 registry key and preserve explicit full-logit diagnostics; real
dense and MoE one-step controls match the prior token and FP32 winner value.
The complete W7900 transaction oracle preserves B1-B3 target logits, candidate
IDs, acceptance, graph execution, reject/partial/full state, and changing
positions.

Clean W7900 natural25 moves B1/B2/B3
**28.348/34.818/38.322 -> 28.878/35.712/39.610 tok/s**
(**+1.87%/+2.57%/+3.36%**). Proposal wall falls
**16.41%/16.74%/17.87%** while target verify is effectively flat. Every prompt
and full/train/heldout/category rollup improves; all IDs/cycle semantics and
transaction gates remain exact. The required compact arrays add only 248,328
bytes and free completely. B3 is selected at **1.9422x own AR**, still 41.82%
below Vulkan B3. Artifact:
`benchmarks/results/2026-08-04-qwen36-27b-compact-proposal-scoring-retained.json`.

With the first D27-O3 pass retained, D27-O1 remains the highest-impact open
campaign gate. The unchanged clean 512-prefill profile assigns **7,815.869 ms /
78.86%** of its reconciled 9,944.618-ms wall to 216 resident-Q4_K pack8
projection launches. A registered wave32 WMMA consumer uses the same qweight
plus FP32 scale/min planes, but it casts each dequantized weight to an FP16
matrix operand and therefore cannot be presumed exact to the retained FP32-FMA
reduction tree.

A GPU1 7900 XTX component screen covered every real dense-Q4 projection *shape*
at M=512 and the 4K route's M=1024 linear chunk, but used uniform synthetic
activation/weight bytes. At M=512, legacy-to-best-WMMA medians were
**75.058 -> 4.485 ms (16.73x)** for FFN gate/up, **9.540 -> 1.458 ms
(6.54x)** for linear QKV, **5.843 -> 0.911 ms (6.41x)** for the linear gate,
**12.839 -> 1.663 ms (7.72x)** for full Q, **0.941 -> 0.198 ms (4.76x)** for
K/V, and **5.605 -> 0.949 ms (5.90x)** for attention output. M=1024 spanned
**5.01-17.28x**. All 72 synthetic comparisons happened to round to equal BF16
outputs, with stable best tiles of 32x32 for wide K=5120 outputs, 32x16 for
N=1024 K/V, and 16x32 for K=6144/N=5120. That equality was not a real-input
correctness proof.

The production-routing trial is **rejected**. Its 132 focused dispatch/kernel
tests pass, and same-process W7900 model prefill improves **52.142 -> 217.163
tok/s** at 512 and **51.827 -> 207.910 tok/s** at 4096. First and next token
remain `9707`, with KL at most `1.26e-6`, but every full logit row and both the
post-prefill and post-step Conv/GDN/KV state hashes differ at both contexts.
This fails D27-O1's exact IDs/state gate, so the `use_wmma_prefill` rewrite and
gfx1100 tile policy were reverted before natural25 or profiler promotion.
Script/result SHA-256s are `f475e433...8252` and `74482f89...985`; the earlier
synthetic screen hashes remain `229bc18b...6d25` and `728c349b...0528`.

The replacement exact primitive is now admitted. It preserves the existing
contiguous-8 K traversal, FP32 `fmaf` order, wave32 reduction tree, zero-seeded
final sum, and BF16 boundary while reusing each resident output-pack8 weight
stream across eight token rows. A balanced GPU1 screen covers all six real
projection roles at M=512 and M=1024 with nonuniform random BF16 activations,
Q4 nibbles, and scale/min planes. All **60/60** candidate comparisons are
BF16-bit exact. Tile8x8 wins **12/12** rows: singles improve **2.434-2.722x**
and replacing dual gate/up with two exact singletons improves **5.454x/4.973x**
at M=512/1024. Four uniformly slower 8x2/8x4/16x2/16x4 candidates were removed.

Cached GPU1 tracing names the tile8x8 body at local32, VGPR144, SGPR128,
LDS256 B, and scratch0. Runtime commit `68e8c10c5` now routes resident pack8
rows >=512 under `use_wmma_prefill`; smaller rows, opt-out, registry misses,
c1, and the existing exact native rows 2-4 owner fail closed. Supported wide
pairs decline the legacy dual owner so callers emit two tile8x8 singletons.

The W7900 promotion gate is retained. A same-process scalar/tile oracle is
byte-exact for first and next full logits plus post-prefill/post-step Conv/GDN/
KV state at 512 and 4096, while diagnostic prefill improves **3.0001x/2.7761x**.
Clean campaign-standard 512/128 and 4096/128 medians improve
**50.515/50.473 -> 152.910/144.308 tok/s (+202.70%/+185.91%)**. Graph decode
is non-regressive at **19.565/18.701 tok/s**, all six final IDs are `9707`, and
tracked peak remains byte-identical at **26.123/28.947 GiB**. This puts matched
prefill **91.60%/76.43% above** the stateful Vulkan floor.

Selected-region production tracing executes the exact tile8x8 symbol 288 times
and cuts the prior Q4 bucket **7,815.869 -> 1,496.173 ms (-80.86%, 5.224x)**;
complete kernel sum falls **9,911.076 -> 3,241.171 ms (-67.30%)**. Dense BF16
prefill GEMM is now the queued AR follow-up at **1,654.462 ms / 51.05%**, but
D27-O1 is complete and D27-O3 resumes before D27-L1. Artifacts:
`benchmarks/results/2026-08-04-qwen36-27b-exact-pack8-prefill-tile8x8-admitted.json`
and `benchmarks/results/2026-08-04-qwen36-27b-exact-pack8-prefill-tile8x8-retained.json`.

The post-compact B3 profile remains the current D27-O3 authority because the
new populated-prefill route starts at 512 rows while every natural25 prompt is
39-71 rows. Its **676.596-ms** complete wall is led by exact target Q4 rowtiles
at **189.267 ms / 27.97%**, followed by dense-BF16 rowtiles at **137.839 ms /
20.37%**. The first resumed candidate is a bounded two-output-pack Q4 rowtile
for verifier rows. The already-bit-exact large-row screen shows compile-time
16x4 is **6.00-9.67%** faster than 8x4 across all six real projection roles;
that implies **11.36-18.31 ms / 1.68-2.71%** complete-wall recovery before
launch savings. This is below the ordinary 5% admission floor but qualifies as
an exact, low-risk additive specialization in the already-open dominant
family. GPU1 must screen rows 2/3/4 against the retained small-row device oracle
before any runtime route changes; unsupported rows and a losing shape retain
the current 8-column owner.

That candidate is **rejected**. Its registry-only RED/GREEN passed 27 focused
checks, and all 18 GPU1 row/shape comparisons were BF16 bit-exact, but the
small-row result invalidates the large-row extrapolation. Two output packs
regress geometric mean time by **6.86% / 19.10% / 6.37%** at rows 2/3/4 and
win only three row-four singleton cases by **1.44-2.43%**. At the dominant
row-four shapes, FFN gate/up regresses **5.83%**, linear gate **21.12%**, and
attention output **19.70%**. The candidate wrapper, registry key, tests, and
export were removed completely; the dedicated eight-column verifier rowtile
remains the sole owner.

The next exact screen moves to the second-largest measured family: dense-BF16
rowtiles own **137.839 ms / 20.37%** of complete B3 wall. Of that, the three
K<=10,240 production shapes account for **70.469 ms** across 560 launches; the
remaining K=17,408 FFN-down shape keeps the current local256 owner because its
one-row control already crossed negative. The admitted candidate maps the same
256 arithmetic partitions onto local128 for every verifier row, performs the
original s=128 pair add in registers, and preserves the s=64/32 LDS plus 16..1
wave tree while retaining cross-row weight reuse. The qualifying one-row
schedule improved **1.091-1.103x**, implying **5.88-6.58 ms / 0.87-0.97%**
complete-wall recovery if it transfers. This below-floor screen is allowed only
as an exact, low-risk extension of the already-open dominant dense family.
GPU1 must compare rows 2/3/4 over all four real shapes before production routing;
all losing K ranges and every unsupported row retain the current rowtile.

The broad local128 policy is **rejected**, but one stable shape qualifies. All
12 screen outputs are BF16 bit-exact; rowwise geometric speedups are
**1.0697x / 0.9922x / 0.9508x**, so rows-four cannot use a blanket rewrite.
`ssm_out` K6,144/N5,120 alone improves **1.2095x / 1.1654x / 1.0761x** across
rows 2/3/4. FFN-down is mixed, QKV loses rows 3/4, and full V loses every row.
The exact primitive is therefore admitted without production routing; the next
RED may select only native dense-BF16 `ssm_out` rows 2-4 and must fail closed on
every other shape or registry miss. The row-four subfamily projection is
**2.397 ms / 0.354%** of current complete B3 wall. Artifact:
`benchmarks/results/2026-08-04-qwen36-27b-dense-virtual256-rowtile-admitted.json`.

The qualified runtime route is now correctness-green. Only the native-batch
K6,144/N5,120 dense-BF16 projection selects the exact-key candidate; every
other shape/session and registry miss retains local256. The complete W7900
oracle observes candidate rows exactly `{2,3,4}` at only that shape and remains
byte-exact for B1-B3 logits, reject/partial/full/rollback state, dynamic graph
reuse, correction logits, and natural provider output. Clean natural25 timing
remains required before this route is promoted as a performance win.

---

## 7. Prioritized execution plan

| Priority | ID | Work | Exit gate / impact rule | Status |
| ---: | --- | --- | --- | --- |
| 0 | D27-M0 | Freeze latest llama.cpp Vulkan revision/build, model hash, hardware/software capture, AR/MTP commands, and unprofiled W7900 baselines. | Fresh pp/tg, context-matched AR, natural25, B1-B4 sweep, and query-profile artifacts. | complete; B3 selected |
| 0 | D27-F0 | Add untied dense root/embedding/layout support, then prove dense GGUF AR load/prefill/decode on GPU0. | Strict map uses Q6_K `output.weight`; finite deterministic 8/1 smoke, then 512/128 and 4K/128 exact/state gates. | complete; clean 512/128 + 4K/128 graph gates green |
| 0 | D27-F1 | Add architecture-shaped dense NextN mapping/materialization with RED tests. | Strict real call-spec accepts 15-tensor `blk.64`; existing MoE fixtures remain unchanged. | complete; real map green |
| 0 | D27-F2 | Run dense NextN one-step and exact/default MTP cycle. | Layer CPU/llama oracle KL <= 0.05, top-1 >= 90%; full state/KV transaction exact. | complete; exact transaction green |
| 0 | D27-M1 | Establish fine-grained llama Vulkan and hipEngine AR/MTP profiles and reconcile wall. | Compact Amdahl tables with <=10% residual or an explicit queue/overlap explanation. | complete; AR + MTP walls reconciled, 10.75% AR graph gap explained |
| 1 | D27-O1 | Optimize the largest measured AR prefill bucket. | Candidate ceiling >=5% complete wall; same-suite exact win at 512 and 4K. | complete; exact tile8x8 reaches 152.910/144.308 tok/s and beats matched Vulkan |
| 1 | D27-O2 | Optimize the largest measured AR decode bucket. | Candidate ceiling >=5% or >=0.20 ms/token; same-suite exact win. | ready but lower urgency; Vulkan already beaten |
| 1 | D27-O3 | Optimize the largest measured MTP cycle bucket (draft, target, commit, or host residual). | Full and heldout MTP/true-AR ratio improves; no category or acceptance regression. | resume next; seven wins retained, compact exact B3 at 1.9422x AR |
| 2 | D27-L1 | Re-profile and close second-order gaps until Vulkan parity. | Each new target is selected from the refreshed profile, not this initial list. | blocked by O2-O3; dense BF16 prefill is queued AR target |
| 3 | D27-P0 | Final clean W7900 publication and default promotion. | Definition of done, rollups, artifacts, refactor cleanup, atomic commits. | pending |

### Impact admission rule

Before coding an optimization, write down:

```text
ceiling_ms = current complete-wall bucket ms
expected_saved_ms = ceiling_ms * credible reducible fraction
engineering/risk = low | medium | high
```

Normally admit only work with a credible **>=5% complete-wall ceiling** or
**>=0.20 ms/token** saving. Smaller work is admitted when it is an exact,
low-risk additive win in an already-open family or removes a concrete blocker.
Never spend a campaign iteration on a lower-ceiling candidate while a higher-
ceiling measured bucket has an untried credible design.

The first optimization target is deliberately **not preselected**. Dense Q4_K
FFN, Q6_K down/lm-head, GDN prefill, attention, or host submission may lead;
D27-M1 decides.

---

## 8. Correctness and promotion gates

### Dense NextN RED/GREEN order

1. Synthetic dense Qwen35 GGUF with one trailing NextN block: exact required,
   optional, unexpected, shape, qtype, and fallback contracts.
2. CPU reference for the dense NextN FFN chain; keep the existing MoE CPU
   oracle untouched or select a separate registered variant.
3. Real file inventory/call-spec test guarded by model existence.
4. One-step GPU result versus CPU/llama oracle.
5. Multi-step Conv/GDN/KV lifecycle, rollback, reseed, and reset reuse.
6. Full natural category MTP suite with true AR and exact/default semantics.

New/ported kernel gates remain KL <= 0.05 and top-1 >= 90%, plus a profile trace
showing the expected symbol. Dense support must be selected through model/layer
plugins and the four-axis registry; do not add backend/quant branches to engine
or model dispatch. Every fused path keeps an unfused fallback.

### Performance keep/revert

Keep only when:

- the intended profiled family improves;
- complete unprofiled wall improves on the same GPU and protocol;
- all primary shapes are non-regressive;
- AR IDs/state or MTP full/train/heldout/category gates pass;
- memory does not regress without an explicitly accepted tradeoff;
- no benchmark-specific token/prompt branch exists.

A micro-only win may be retained as a primitive but is not promoted to runtime
until the complete-path gate wins.

---

## 9. Canonical commands (baseline skeleton)

### Latest llama.cpp Vulkan build

```bash
cd /home/lhl/llama.cpp/llama.cpp-vulkan
git pull --ff-only
cmake -S . -B build \
  -DGGML_VULKAN=ON -DGGML_HIP=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 16
```

### llama-bench W7900 AR

```bash
/home/lhl/llama.cpp/llama.cpp-vulkan/build/bin/llama-bench \
  -m /models/gguf/Qwen3.6-27B-Q4_K_M.gguf \
  -dev Vulkan0 -ngl 99 -fa on -ctk f16 -ctv f16 \
  -p 512,4096 -n 128 -r 5 -o json
```

### llama.cpp W7900 natural AR/MTP

```bash
python3 scripts/llamacpp_mtp_bench.py \
  --server-bin /home/lhl/llama.cpp/llama.cpp-vulkan/build/bin/llama-server \
  --model /models/gguf/Qwen3.6-27B-Q4_K_M.gguf \
  --ctx-size 8192 --gpu-layers 99 --flash-attn on \
  --cache-type-k f16 --cache-type-v f16 \
  --draft-max 3 --mode both --protocol natural \
  --prompts benchmarks/prompts/mtpbench-code-general-ja.jsonl \
  --max-tokens 25 --seed 12345 --temperature 0 --top-k 1 --top-p 1 --min-p 0 \
  --server-extra-arg=-dev --server-extra-arg=Vulkan0 \
  --server-extra-arg=--reasoning --server-extra-arg=off \
  --server-extra-arg=--perf \
  --output /tmp/qwen36-27b-vulkan-natural25.json
```

### hipEngine GGUF AR development smoke on GPU1

Use the same hermetic TheRock environment as the W7900 wrapper, changing only
`HIP_VISIBLE_DEVICES=1` and labeling the physical RX 7900 XTX:

```bash
HIP_VISIBLE_DEVICES=1 HIPENGINE_HIP_ARCH=gfx1100 \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt \
PYTHONPATH=. python3 scripts/qwen35_gguf_bench.py \
  --model /models/gguf/Qwen3.6-27B-Q4_K_M.gguf \
  --quant gguf_q4_k_m --prompt-length 8 --decode-tokens 1 \
  --warmup-runs 0 --measured-runs 1 --warmup-decode-tokens 0 \
  --persistent-session --force-bulk-prefill --bulk-prefill-attention-mode bulk \
  --use-wmma-prefill --use-gemv-decode --no-graph-replay-decode \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --json /tmp/qwen36-27b-gguf-gpu1-smoke.json
```

Final GPU0 rows use the full `env -i` TheRock wrapper, cached builds, one
warmup, at least three measured resets, and production graph decode.

### hipEngine dense natural AR/MTP category suite

```bash
HIP_VISIBLE_DEVICES=0 HIPENGINE_HIP_ARCH=gfx1100 \
HIPENGINE_GGUF_DECODE_REPACK=1 \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-qwen36-27b-hipcc-version.txt \
HIPENGINE_REQUIRE_CACHED_BUILD=1 PYTHONPATH=. \
python3 scripts/qwen36_dense_gguf_suite.py \
  --model /models/gguf/Qwen3.6-27B-Q4_K_M.gguf \
  --quant gguf_q4_k_m \
  --prompts benchmarks/prompts/mtpbench-code-general-ja.jsonl \
  --max-new-tokens 25 --candidate-budgets 1,2,3 --runs 1 \
  --compiler-version-file /tmp/hipengine-qwen36-27b-hipcc-version.txt \
  --require-cached-build \
  --output /tmp/hipengine-qwen36-27b/dense-natural25-b1-b3.json
```

The suite's engine numerator is 24 timed transitions per prompt because output
zero is produced by prefill. Request/client wall and the legacy visible-output
numerator remain separate fields. Use `--limit 1 --no-warmup --roctx-markers`
only for a profiler leaf, never for a performance claim.

---

## 10. Campaign scoreboard

Do not fill cells from historical PARO, MoE, HIP, or another GPU.

| Date | Revision / route | GPU | Shape | Prefill tok/s | AR decode tok/s | MTP decode tok/s | MTP/AR | Correctness | Artifact |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 2026-08-04 | llama.cpp Vulkan `ee0445c99` | W7900 | stateful 512/128, 4K/128; natural25 B3 | 79.805 / 81.792 | 12.574 / 12.488; natural AR 12.546 | 68.082 engine / 36.122 client | 5.4265x engine / 3.7600x client | `[9707]*128` captured exact; all categories improve; 6/10 natural content hashes match AR | `benchmarks/results/2026-08-04-qwen36-27b-llamacpp-vulkan-baseline.json` |
| 2026-08-04 | hipEngine `da6865f74`, exact/default dense GGUF | W7900 | 512/128, 4K/128; natural25 B1-B3 | 50.515 / 50.473 | 19.556 / 18.649; natural AR 20.361 | B1 17.128 (best), B2 16.005, B3 14.858 | 0.8412x / 0.7861x / 0.7297x | all repeated IDs deterministic; all 750 MTP-visible IDs exact vs AR; GPU/CPU accept exact; state transaction oracle green | `benchmarks/results/2026-08-04-qwen36-27b-hipengine-baseline.json` |

Update this table only with retained or explicitly labeled blocked/diagnostic
rows. Detailed iteration history belongs in `WORKLOG.md`; benchmark toplines
also update `benchmarks/README.md`, `benchmarks/CHANGELOG.md`, and compact JSON
artifacts when measured.
