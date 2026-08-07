# Qwen3.6-27B Q4_K_M GGUF Optimization Campaign

Status: **tabled on 2026-08-07 below selected Vulkan parity after the reopened
source-faithful review exhausted current mechanisms.** Latest tracked-clean
llama.cpp `c8e03ce81` (build 10290) selects B4 at 69.798 tok/s. The post-module
hipEngine production control is B3 61.147 tok/s (**12.394% short; +14.148%
required**); the cleaner retained canonical row remains 61.394 tok/s (**12.040%
short; +13.688% required**). Populated prefill and true AR beat Vulkan at both
512/128 and 4096/128. Resume only under the explicit high-leverage conditions
in the final 2026-08-07 audit.

Canonical target:
`/models/gguf/Qwen3.6-27B-Q4_K_M.gguf` on AMD Radeon Pro W7900 / GPU0 /
`gfx1100`.

Comparator: latest tracked-clean llama.cpp Vulkan source at
`/home/lhl/llama.cpp/llama.cpp-vulkan`, refreshed from `ee0445c99` to
`c8e03ce81` on 2026-08-06 and built in Release mode with `GGML_VULKAN=ON` and
`GGML_HIP=OFF`.

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

The qualified runtime route is retained. Only the native-batch K6,144/N5,120
dense-BF16 projection selects the exact-key candidate; every other shape/session
and registry miss retains local256. The complete W7900 oracle observes candidate
rows exactly `{2,3,4}` at only that shape and remains byte-exact for B1-B3
logits, reject/partial/full/rollback state, dynamic graph reuse, correction
logits, and natural provider output.

Clean W7900 natural25 moves B1/B2/B3
**28.878/35.712/39.610 -> 28.979/35.826/39.714 tok/s**
(**+0.349%/+0.318%/+0.263%**) and improves MTP/own-AR by
**+0.542%/+0.511%/+0.456%**. Every full/train/heldout/category aggregate is
positive (**+0.117% to +0.551%**), all IDs/acceptance ledgers remain exact, and
tracked peak is byte-identical. Four of 30 individual prompt-budget timing rows
are noise-negative by at most **0.145%**; the predeclared exact component wins
all three supported row counts and complete suite wall improves at every budget.
B3 remains selected at **1.9511x** own AR, still **41.67%** below Vulkan B3.
Artifact:
`benchmarks/results/2026-08-04-qwen36-27b-dense-virtual256-rowtile-retained.json`.

The dominant Q4 family now admits a materially distinct fusion rather than
retrying rejected output-column widening. In the authoritative B3 trace, dense
FFN gate/up plus SiLU owns **110.684 ms / 16.36%** of complete wall across
**1,344 launches**. A registry-only local64 leaf assigns gate and up to separate
wave32 owners, preserves each existing rowtile's K/FMA/shuffle tree and BF16
boundary, then executes the unchanged BF16 SiLU formula from 128 B shared
memory. Production dispatch remains unchanged during admission.

The full 36-test Q4 bundle is green. A balanced K5,120/N17,408 GPU1 screen is
BF16-bit exact and improves inclusive event/wall by
**1.0888x/1.0888x**, **1.0886x/1.0889x**, and **1.1210x/1.1208x** at rows
2/3/4. Cached tracing reports local64/VGPR96/SGPR128/LDS128B/scratch0. Applying
the row-four ratio projects **11.948 ms / 1.766%** complete-wall recovery before
any separate graph-node benefit. Admit the primitive without runtime routing;
the next RED must select it only for native resident-pack8 dense FFN rows 2-4
and fail closed everywhere else. Artifact:
`benchmarks/results/2026-08-04-qwen36-27b-q4-dual-rowtile-silu-admitted.json`.

That fail-closed runtime route is now correctness-complete. The generic pair
boundary first proves both native singleton owners resolve the exact rowtile
key, then selects the fused registry leaf only at K5,120/N17,408 and rows 2-4;
c1, ordinary sessions, row/shape/layout/opt-out misses, and absent keys retain
the old two-rowtile plus separate-SiLU chain. The dense runner consumes the
fused output directly without changing the down/residual path.

Five focused policy/integration tests pass, the prior GPU1 Q4/rowtile bundle and
all existing gfx1151 c1 pair-SiLU tests remain green, and lint/compile checks
pass. The W7900 B1-B3 transaction oracle observes candidate rows exactly
`{2,3,4}` at only `(5120,17408)`, observes no legacy Q4 singleton launches, and
remains exact for logits, reject/partial/full/rollback Conv/GDN/KV/hidden state,
dynamic graph reuse, correction logits, and natural provider output. Freeze
this routing unit before the clean natural25 promotion gate; no complete-path
performance claim is made yet.

The exact fusion is now retained on mechanical cycle-wall evidence without
advancing the natural25 headline. A second balanced production-shape W7900
screen remains byte-exact and improves the unfused chain by
**1.1060x/1.1093x/1.1128x** at rows 2/3/4. Production B3 tracing observes the
fused owner 448 times, removes exactly **896 graph dispatches**, and eliminates
legacy N17,408 rowtiles. Against the pre-`ssm_out` compact profile, complete
cycle wall falls **676.596 -> 662.705 ms (-2.053%)** and queue-gap/copy-overlap
wall falls **192.822 -> 174.794 ms (-9.350%)**. The complete-window result is
compounded with the separately retained `ssm_out` route; only the dispatch
reduction is uniquely attributable here.

The clean natural25 packet measures B1/B2/B3
**29.282/35.850/39.614 tok/s** (**+1.046%/+0.066%/-0.250%**) versus the prior
clean route. Every selected-B3 aggregate scope is slightly negative
(**-0.060% to -0.368%**), so it cannot replace the canonical
**39.714 tok/s / 1.9511x own AR** row. IDs, acceptance, full transaction state,
tracked peak, and teardown remain exact. Keep the exact default because both
gfx1100 boards win the binding leaf and W7900 proves fewer launches, less
queue-gap wall, and a shorter cycle window; disclose the mixed aggregate and
continue D27-O3 from the refreshed trace. Artifact:
`benchmarks/results/2026-08-04-qwen36-27b-q4-dual-rowtile-silu-retained.json`.

The refreshed trace then exposes a larger launch-and-traffic boundary: each of
seven B3 target passes issues 48 linear layers x four serial Conv/GDN producers
and four post-row copies of both the 160-KiB Conv and 3-MiB recurrent state.
The already-landed c1-exact chain t-loops are production-shape byte-identical to
that scalar+copy chain at rows 2/4 and write every journal row directly while
leaving the initial state immutable. The new exact registry keys therefore own
only dense native c2-c4 row capture. Deferred target verification performs no
resident write; non-deferred use copies only the final row; a missing key keeps
the complete scalar chain.

The W7900 B1-B3 transaction gate observes both chain owners exactly at rows
`{2,3,4}` and only Conv/GDN shapes `(10240,4)` and `(16,48,128,128)`. Full
logits, reject/partial/full/rollback Conv/GDN/KV/hidden state, dynamic graph
reuse, correction logits, and natural provider output remain scalar-exact.
This correctness-only route projects removal of **2,688 journal copies + 2,016
serial producers = 4,704 target dispatches**, **34.17%** of the refreshed
13,767-dispatch profile. Retained performance and named trace remain the next
gate; no speed claim is made yet.

The exact chain-journal route is now retained and advances the natural25
headline. The hermetic W7900 trace confirms the projection exactly:
**13,767 -> 9,063 dispatches (-4,704 / -34.17%)**, including
**3,360 -> 672** state-sized copies and **2,688 -> 672** Conv/GDN producers.
Complete B3 cycle wall falls **662.664 -> 635.545 ms (-4.09%)**, target verify
falls **5.52%**, and queue-gap/copy-overlap wall falls **8.94%**, with the same
`[3,3,2,3,3,0,3]` acceptance ledger.

Against the immediately preceding clean route, the canonical ten-prompt W7900
B1/B2/B3 packet improves **29.282/35.850/39.614 ->
30.130/37.357/41.440 tok/s (+2.895%/+4.205%/+4.608%)**. Against the prior best
headline, B3 improves **39.714 -> 41.440 tok/s (+4.346%)** and reaches
**2.0348x own AR**. All 30 prompt/budget rows improve (**+2.53% to +5.75%**),
as do every full/train/heldout/category aggregate (**+2.71% to +4.79%**);
IDs, acceptance, state, peak memory, and teardown remain exact. The remaining
Vulkan B3 gap narrows from 41.67% to **39.13%**. Artifact:
`benchmarks/results/2026-08-04-qwen36-27b-chain-journals-retained.json`.

The next exact D27-O3 route batches short-context full attention after all
verifier K/V writes. Every N1 row extends the same resident physical page table,
while contiguous device `live_counts` keep each query causal even though later
rows are already present. A dedicated Q4 registry leaf preserves the existing
256-thread arithmetic and consumes that one shared table; the ordinary
row-table-major batch ABI is not reused. Missing keys, different cache/table
owners, or noncontiguous metadata retain the complete scalar
`write -> attention -> gate` loop.

GPU1 production-shape rows 2/3/4 screens are FP32/BF16 byte-exact across live
counts through 1,024 and improve synchronized attention+gate wall by
**1.47x/2.16x/2.84x**. A reversed-page live254-257 fixture is byte-exact to
scalar and passes the CPU oracle; cached tracing names the new shared-table
specialization at local256/VGPR40/scratch0. The W7900 transaction remains exact
and observes the N1 owner at physical rows `{2,4}`. B2/rows3 intentionally stays
with the separate N2 bulk graph rather than crossing transaction ownership.

The hermetic W7900 B3 trace confirms the schedule exactly: **9,063 -> 8,391
dispatches (-672 / -7.41%)**. Target scalar attention becomes **448 -> 0**,
replaced by 112 shared-table batches, while **448** row gates become **112**
whole-batch gates. Attention+gate kernel sum falls **7.930 -> 4.478 ms**.
Complete marker wall measures **635.545 -> 608.946 ms (-4.19%)**, although
unchanged proposal/commit phases also shifted about 4%, so that full profile
ratio is not attributed solely to the route.

Clean natural25 retains the selected B3 result at **41.440 -> 41.705 tok/s
(+0.641%)**, target verify **-0.967%**, and **2.0474x own AR**. All ten B3
prompts and every B3 full/train/heldout/category rollup improve
(**+0.097% to +0.839%** prompts; **+0.490% to +0.761%** scopes). B1 is
noise-flat at **-0.045%** and the separately owned N2 B2 row is diagnostic at
**+0.241%**; neither is claimed as a speed win. IDs, acceptance, transaction
state, 28.996-GiB peak, and teardown remain exact. Artifact:
`benchmarks/results/2026-08-04-qwen36-27b-full-attention-shared-batch-retained.json`.

The post-shared-attention trace exposes and retains one more exact launch
boundary inside that same staged full-attention owner. Target K issued **448**
scalar K5,120/N1,024 resident-pack8 calls totaling **4.919 ms**. The existing
generic local32 kernel already maps grid Y to independent rows without changing
any row's K traversal, FP32 FMA/shuffle tree, or BF16 store. The dedicated
`linear/gguf_q4_k_m/pack8_full_k_grid_y_native_exact_bf16_bf16_out` alias now
replaces each rows-long scalar loop with one rows-wide launch; a missing key
retains the complete scalar loop.

A production-shape GPU1 screen is BF16-byte exact at rows 2/3/4 and improves
synchronized event time **1.849x/2.374x/2.620x** (wall
**1.824x/2.338x/2.587x**). The W7900 B1-B3 transaction gate preserves full
logits, reject/partial/full/rollback Conv/GDN/KV/hidden state, dynamic position
reuse, correction logits, and natural provider output. Its registry census
observes N1 owner rows exactly `{2,4}`; B2/rows3 remains with the separate N2
bulk graph.

The hermetic W7900 B3 trace confirms **8,391 -> 8,055 dispatches (-336 /
-4.00%)** and exactly **448 scalar K -> 112 grid-y K** target launches. The
selected K family falls **4.919 -> 2.252 ms (-54.23%, 2.185x)**, target-verify
kernel sum falls **0.828%**, and target host wall falls **0.184%**. Complete
profile marker wall is noise-negative at **608.946 -> 611.399 ms (+0.403%)**
because queue-gap/copy-overlap grows **5.708 ms**; the profile wall is disclosed
rather than attributed as a win.

Clean natural25 retains selected B3 at **41.705 -> 41.890 tok/s (+0.442%)**,
target verify **-0.595%**, and **2.0584x own AR**. Every B1-B3
full/train/heldout/category aggregate improves (**+0.016% to +0.992%**); all B3
scopes improve **+0.379% to +0.479%**. Individual prompt timing includes small
noise-negative rows, so no 30/30-row claim is made. IDs, acceptance, complete
transaction state, the **28.996-GiB** peak, and teardown remain exact. Artifact:
`benchmarks/results/2026-08-04-qwen36-27b-full-attention-k-grid-y-retained.json`.

The refreshed profile next exposed an ownership rather than arithmetic gap.
The target already stores `output.weight` as byte-neutral Q6T16, but the
standalone NextN materializer hardcoded the raw-Q6 layout and uploaded a second
head. The target-attached provider now borrows source-compatible target
`token_embedding` and `lm_head` records for its strictly nested lifetime.
Borrowed records are excluded from `owned_weights`; the draft's distinct
`blk.64.nextn.shared_head_norm.weight` remains independently owned, and an
attempt to alias target `output_norm.weight` is rejected before allocation.

One quant-neutral four-axis contract,
`(hip_gfx1100, linear+argmax, <resident quant>,
proposal_top1_exact_bf16)`, selects either the prior exact raw scorer or an
adapter over the existing T16 full-logit/tile-winner producer and pack8 final
reducer. No device body, environment flag, prompt branch, or approximation is
added. On actual `output.weight` at K5,120/N248,320, GPU1 proves full-logit and
top-1 id/value-bit equality while reducing synchronized event median
**2,218.341 -> 1,489.425 us (-32.86%, 1.489x)**.

The W7900 B1-B3 transaction gate preserves logits, reject/partial/full/rollback
state, dynamic graph reuse, correction output, provider IDs, ownership, and
teardown. The hermetic B3 trace replaces exactly **25 raw-Q6 stage1 calls /
49.460 ms** with **25 T16 calls / 37.925 ms**. Producer plus reducer falls
**50.439 -> 38.445 ms (-23.78%, 1.312x)**, proposal stage wall falls
**15.14%**, and complete marker wall falls **611.399 -> 585.481 ms (-4.24%)**
with unchanged dispatch count and acceptance ledger.

Clean natural25 retains B1/B2/B3 at **30.154/37.663/41.890 ->
30.719/38.627/43.240 tok/s (+1.873%/+2.560%/+3.224%)**. Selected B3 proposal
wall falls **17.468%** and reaches **2.1290x own AR**. Every one of 30
prompt-budget rows and every full/train/heldout/category aggregate improves;
IDs, acceptance, and stage accounting remain exact. Avoiding duplicate roots
removes exactly **1,758,105,600 bytes**, moving tracked peak **28.996 -> 27.359
GiB** with zero allocations after close. The Vulkan B3 gap is now **36.49%**.
Artifact:
`benchmarks/results/2026-08-04-qwen36-27b-resident-t16-proposal-retained.json`.

The `b888de605` profile supersedes all earlier proposal rankings. Proposal is
no longer the first untried ceiling. The next high-impact D27-O3 boundary is
target compressed-weight ownership: rank-2 source-Q6 projections currently
materialized as dense BF16 consume about **106.6 ms**, and Q5 `ssm_out` another
**36.6 ms**, together **31.3% of kernel sum / 24.5% of complete marker wall**.
Any packed replacement must preserve the unfused dense fallback, pass the full
quality/state gate rather than one prompt, and keep >=512 prefill non-regressive.

The first replacement screen qualifies a selective sole-resident Q6T16 route,
but does not yet promote it. The 32 Q6 `ffn_down` and 24 Q6 `attn_qkv`
projections occupy **8,220,835,840 bytes** as dense BF16 versus
**3,371,827,200 bytes** in byte-neutral T16, a projected persistent saving of
**4,849,008,640 bytes / 4.516 GiB**. The eight narrow Q6 `attn_v` projections
remain dense: their actual K5,120/N1,024 c1 leaf is only **0.40x** as fast in
T16. The role plus `N>=5,120` materialization boundary also leaves smaller
Qwen3.5 tensors unchanged.

On GPU1 actual weights, the wide c1 T16 leaf is **1.573x** faster for
K17,408/N5,120 `ffn_down` and **1.341x/1.611x** faster than the current native
local128/ordinary local256 K5,120/N10,240 `attn_qkv` owners. The existing exact
T16 rowtile is **1.278-1.713x** faster at c2-c4. A new metadata-free dense WMMA
wrapper is bit-identical to the screened one-expert selected producer and
improves M64 by **2.522x/2.674x** and M512 by **3.332x/3.251x** for the two
wide shapes. Across the actual-weight c1/M64/M512 screens, maximum KL versus
materialized dense BF16 is **4.73e-5** and minimum top-1 agreement is
**98.4375%**.

The four-axis route now registers BF16 decode, c2-c6 exact rowtile, and dense
WMMA prefill variants under `linear/gguf_q6_k_t16_v1`; optional sibling misses
retain the scalar T16 chain, while `decode_repack=False` retains dense BF16.
Cached GPU1 tracing names `q6_k_t16_wmma_prefill_bf16_kernel` at **29,920 ns**,
grid `(512,2)`, local32,
VGPR72, SGPR128, LDS0, and scratch0. The binding W7900 B1-B3 transaction gate
passes with natural provider/scalar agreement and complete reject/partial/full/
rollback state, dynamic positions, correction, graph reuse, ownership, and
teardown checks.

Clean W7900 measurement promotes the route. The one-prompt B3 trace replaces
exactly **224 `ffn_down` + 168 `attn_qkv` dense-BF16 rowtiles** with the same
number of Q6T16 rowtiles while all **56 narrow `attn_v` calls remain dense**.
The wide family falls **104.936 -> 79.719 ms (-24.03%, 1.316x)**, target verify
falls **494.991 -> 469.671 ms (-5.12%)**, complete marker wall falls **585.481
-> 561.644 ms (-4.07%)**, and dispatches remain 8,055. Profile peak drops by
the projected **4,849,008,640 bytes** exactly.

On the full ten-prompt suite, true AR improves **20.310 -> 21.735 tok/s
(+7.02%)** and B1/B2/B3 improve **30.719/38.627/43.240 ->
33.456/40.067/44.886 tok/s (+8.91%/+3.73%/+3.81%)**. Every prompt-budget row
and every full/train/heldout/category aggregate improves. MTP remains exactly
greedy to candidate AR and GPU acceptance remains CPU-exact. Relative to the
prior dense-BF16 route, **249/250 tokens agree**; the sole change is fluent
Japanese `計画案` -> `計画書`. B1 keeps 115 accepted tokens while proposals fall
128 -> 126; B2/B3 accepted/proposed totals are unchanged. B2/B3 own-AR ratios
fall about 3% only because the shared AR denominator improves faster, so this
is retained as a common-path absolute win rather than an MTP-ratio claim.

Campaign-standard populated rows also improve. Median 512/4096 prefill rises
**152.910/144.308 -> 202.011/188.765 tok/s (+32.11%/+30.81%)** and graph AR
decode rises **19.565/18.701 -> 20.896/19.784 tok/s (+6.80%/+5.79%)**. All six
final IDs remain `9707`, timing variation is at most 1.29%, and tracked peaks
fall **26.123/28.947 -> 21.607/24.431 GiB**. Selected B3 is now **34.07%** below
Vulkan. Artifact:
`benchmarks/results/2026-08-04-qwen36-27b-selective-q6t16-projections-retained.json`.
Re-rank only the `c44e32ff6` profile; the remaining dense Q5 `ssm_out` family
is **37.372 ms**, while Q4 and the now-resident Q6T16 families remain larger
arithmetic buckets.

The Q5 `ssm_out` compressed-ownership branch is now closed without a runtime
change. On the actual K6,144/N5,120 weight, byte-neutral Q5T16 passes the
component quality gate and improves M64/M512 prefill **2.373x/3.382x**, but its
best c1-c4 decode rows reach only **0.516x/0.531x/0.405x/0.448x** of the retained
dense-BF16 owner. Every existing raw-Q5 control also passes quality and is
mutually BF16-bit exact, but the best c1-c4 rows reach only
**0.828x/0.361x/0.312x/0.282x**; its best M64/M512 coltile reaches
**0.603x/0.565x**. Therefore neither layout qualifies as a sole resident or
verifier sidecar. Dense BF16 remains production and the larger Q4 verifier
family becomes the next layout discriminator. Artifact:
`benchmarks/results/2026-08-04-qwen36-27b-q5-ssm-out-compressed-layouts-rejected.json`.

Compact-Q4's later row-reuse win justified one materially different Q5
follow-up, which is also closed and removed. An exact local128 Q5T16 body
reused each decoded weight across all verifier rows while preserving the direct
producer's K partitions, wave tree, ordered reduction, and BF16 output. Packed
metadata/coeff loads plus a complete 3x3 thread/column geometry screen all pass
quality; the exact paths have zero BF16 mismatches to direct Q5T16, while the
changed-association local32/local64 diagnostics reach maximum KL **1.33e-7**
and top-1 **100%** versus dense. The final 11-sample local128/col4 adjudication
still reaches only **0.712x/0.739x/0.882x** dense at rows 2/3/4. Candidate
kernel, wrapper, registry key, tests, and temporary geometry selectors are
removed. Dense BF16 remains the sole `ssm_out` owner. Artifact:
`benchmarks/results/2026-08-05-qwen36-27b-q5t16-ssm-out-rowreuse-rejected.json`.

That discriminator now admits exact compact-Q4T16 rowtiles without changing
runtime ownership. The new single local32 and two-wave fused-FFN local64 bodies
reuse each decoded T16 weight across rows 2-4 while retaining every row's
contiguous-eight K traversal, FP32 FMA sequence, wave32 tree, zero-seeded BF16
projection boundary, and unchanged SiLU expression. The complete 106-test T16
bundle passes, and production-shape tracing confirms zero scratch: row four is
VGPR144/SGPR128 with LDS0 for single and LDS512 for fused FFN.

On actual layer-0 K5,120/N17,408 gate/up weights, all candidate outputs are
BF16-bit exact to retained pack8. Single rows 2/3/4 improve
**1.039x/1.309x/1.125x** and fused FFN improves
**1.996x/1.740x/1.381x**. The current profile assigns **106.201 ms** to the
qualified fused-FFN owner, so its row-four ratio projects **29.289 ms / 5.215%**
complete-wall recovery. The separate **81.033-ms** single-Q4 bucket remains
unqualified until every actual shape is screened. Runtime remains unwired during
primitive admission. Next, route only the already-qualified FFN pair through a
compact resident or sidecar with exact pack8 fallback; independently screen all
actual single-Q4 shapes before broad ownership, then require W7900 transaction/
profile/natural25 and populated-prefill gates. Artifact:
`benchmarks/results/2026-08-04-qwen36-27b-q4t16-rowtiles-admitted.json`.

Production FFN routing is now correctness-qualified without yet advancing the
headline. With decode repacking enabled, exactly 64 dense `ffn_gate` and 64
`ffn_up` Q4_K weights retain pack8 and gain a compact-T16 `decode_tiles`
sidecar. This adds **6,595,543,040 bytes / 6.143 GiB**. Existing exact compact
c1 fusion owns AR/single-row calls, the new registered row-reuse fusion owns
native rows 2-4, and a missing sidecar/key/session or unsupported shape fails
closed to retained pack8. The focused planner/dispatch bundle passes, and the
binding W7900 transaction test passes with exclusive T16 ownership at rows
`{2,3,4}` plus exact logits, reject/partial/full/rollback state, dynamic graph
reuse, correction output, natural provider output, and teardown.

The independent actual-weight screen also closes the separate single-Q4
question before any broad route. All seven remaining production shapes are
BF16-bit exact at rows 2-4. Geometric-mean speedup over pack8 is
**1.071x/1.256x/1.180x**; all row-3/4 shapes win, while row-2 `attn_qkv` and
`attn_v` are effectively flat at **0.9977x/0.9969x**. Therefore this FFN-only
correctness unit remains scoped, and any later single-Q4 promotion must use a
measured shape/row policy rather than blanket ownership. Clean W7900 profile,
natural25, and populated-prefill gates remain required before retaining the
sidecars.

Those gates now retain the sidecars. The hermetic W7900 B3 trace replaces
exactly **448 pack8 fused-FFN launches / 106.201 ms** with **448 compact-T16
launches / 76.440 ms (-28.02%, 1.389x)**. Target verification falls **469.671
-> 452.948 ms (-3.56%)**, kernel sum falls **436.018 -> 407.085 ms (-6.64%)**,
and complete marker wall falls **561.644 -> 548.050 ms (-2.42%)** with the same
8,055 dispatches and exact acceptance ledger.

Clean natural25 advances B1/B2/B3 **33.456/40.067/44.886 ->
36.078/42.879/47.496 tok/s (+7.84%/+7.02%/+5.81%)** and selected B3 own-AR
**2.0652x -> 2.1887x (+5.98%)**. Every one of 30 prompt-budget rows and every
full/train/heldout/category scope improves; IDs and acceptance remain exact.
True AR is byte-identical and moves **-0.161%** within one-run noise. Populated
512/4096 prefill is likewise noise-flat at **201.698/188.580 tok/s
(-0.155%/-0.098%)**, while graph decode is **+0.065%/+0.101%** and every final
ID remains `9707`.

The speed win costs exactly **6.143 GiB** because pack8 remains required for
prefill/fallback: natural peak becomes **28.985 GiB**, and 512/4096 peaks become
**27.749/30.574 GiB** before clean teardown. This is retained for the 48-GiB
W7900 campaign target and is not claimed to fit the 24-GiB component board.
Selected B3 is now **30.24%** below Vulkan. Re-rank only `0439ecc3c`; its top
families are single-Q4 pack8 rowtiles **81.461 ms**, Q6T16 rowtiles **80.372
ms**, retained compact FFN **76.440 ms**, and dense Q5 `ssm_out` **36.982 ms**.
Artifact:
`benchmarks/results/2026-08-05-qwen36-27b-q4t16-ffn-sidecars-retained.json`.

The measured single-Q4 follow-up is now retained with an explicit row policy.
All **288** screened rank-2 Q4 tensors carry compact sidecars. Seven projection
families select T16 at native rows 2-4; `attn_qkv` and `attn_v` retain pack8 at
row 2 because their actual-weight component ratios were only
**0.9977x/0.9969x**, then select T16 at rows 3-4. Pack8 remains resident for
populated prefill and every fail-closed route. The binding W7900 transaction
census observes compact rows `{2,3,4}` across all six production shapes and
pack8 only at those two declared row-2 exceptions, with exact logits,
reject/partial/full/rollback state, graph reuse, provider output, ownership,
and teardown.

The hermetic W7900 B3 trace replaces exactly **1,008 pack8 single-Q4 launches /
81.461 ms** with **1,008 compact-T16 launches / 51.370 ms (-36.94%, 1.586x)**.
Target verification falls **452.948 -> 423.244 ms (-6.56%)**, kernel sum falls
**407.085 -> 378.591 ms (-7.00%)**, and complete marker wall falls **548.103 ->
515.594 ms (-5.93%)** without changing dispatch or copy counts.

Clean natural25 advances B1/B2/B3 **36.078/42.879/47.496 ->
37.544/45.634/50.344 tok/s (+4.06%/+6.43%/+6.00%)** and selected B3 own-AR
**2.1887x -> 2.3260x (+6.27%)**. All 30 prompt-budget rows improve
(**+3.73% to +7.09%**), as does every full/train/heldout/category scope
(**+3.97% to +6.62%**); IDs, acceptance, and stage accounting remain exact.
True AR's **-0.259%** movement is timing noise on an unchanged c1 route.

Populated 512/4096 prefill moves **+0.422%/+0.030%** and graph AR
**+0.144%/-0.184%**, with all six final IDs `9707`. The extension adds exactly
**4,194,959,360 bytes / 3.907 GiB**, bringing total compact-Q4T16 sidecars to
**10.049 GiB** before clean teardown. This remains a 48-GiB W7900 route, not a
24-GiB component-board claim. B3 is now **26.05%** below Vulkan. Re-rank only
`2dbf6abdd`: remaining leaders are Q6T16 rowtiles (**80.906 ms** BF16 plus
**18.229 ms** F32 output), retained compact FFN (**76.456 ms**), compact
single-Q4 (**51.370 ms**), proposal Q6T16 top-1 stage1 (**39.973 ms**), and
dense `ssm_out` (**37.160 ms**). Artifact:
`benchmarks/results/2026-08-05-qwen36-27b-q4t16-row-selective-sidecars-retained.json`.

The next refreshed leader is also retained. An exact local128 Q6T16 sibling
splits each sixteen-column slab into two eight-column blocks while preserving
all K partitions, FMA order, wave32 trees, ordered four-wave reduction, and
BF16/F32 output bits. GPU1 actual-weight screening qualifies BF16 rows 2-5 and
FP32 rows 3-4 only; row 6, FP32 rows 2/5/6, and every unsupported shape retain
col16. The materially different exact col4 control loses all nine rows-2-4
actual-shape cases and is removed.

The hermetic W7900 B3 trace observes exactly **399** col8 launches and unchanged
**8,055** total dispatches. Q6 rowtiles fall **99.135 -> 75.244 ms (-24.10%,
1.318x)**, target-verify host wall falls **423.244 -> 399.282 ms (-5.66%)**,
kernel sum falls **378.591 -> 356.972 ms (-5.71%)**, and complete wall falls
**515.594 -> 492.172 ms (-4.54%)**. B3 resources fall from 136 VGPR / 1,024 B
LDS to 80 VGPR / 512 B LDS, with zero scratch in both paths.

Clean-protocol natural25 advances B1/B2/B3 **37.544/45.634/50.344 ->
38.581/48.712/52.652 tok/s (+2.76%/+6.74%/+4.59%)** and selected own-AR
**2.3260x -> 2.4291x (+4.43%)**. All 30 prompt-budget rows and every aggregate
scope improve; all IDs, acceptance, transaction state, and teardown remain
exact, with byte-identical **32.892-GiB** peak. The selector cannot execute in
populated 512/4096 or c1 graph AR, so those controls are not repeated; unchanged
true AR and the complete transaction gate bind non-regression. B3 is now
**22.66%** below Vulkan. Re-rank only `a821d571b`: compact Q4 dual+SiLU leads
at **76.444 ms**, Q6T16 col8 follows at **75.244 ms**, compact single-Q4 is
**52.287 ms**, proposal Q6T16 top-1 is **40.466 ms**, and dense `ssm_out` is
**37.316 ms**. Artifact:
`benchmarks/results/2026-08-05-qwen36-27b-q6t16-col8-rowtiles-retained.json`.

The matched module re-rank then justified one materially different return to
Q5 `ssm_out`: hipEngine's retained dense family is **37.316 ms**, versus
**15.12 ms** for Vulkan, while the existing raw-Q5 selected Q8_1/`sudot4`
primitive had never reused a dense weight across verifier rows. A temporary
raw-Q5 rowtile reuses each decoded value over rows 2-4 and is synthetic
BF16-bit exact to the established integer-dot primitive at local128. On the
actual K6,144/N5,120 weight it passes the component gate at maximum KL
**0.001697** and top-1 **100%**.

The economics still reject it. A complete column/thread screen covers
`(1,32)`, `(2,64/128)`, `(4,64/128/256)`, and `(8,128/256)`. Final 11-sample
col4 adjudication reaches only **0.646x/0.713x/0.798x** dense including Q8_1
quantization at rows 2/3/4, with **0/11** paired wins in every final row. The
rowtile cuts the row-separate integer dot by roughly 2-3x, but not enough to
beat the 60-MiB dense-BF16 owner. Candidate kernel, wrapper, registry key,
tests, and geometry selectors are removed; no allocation or runtime route
remains. A direct Q5T16/Q8_1 control also loses at **0.605x/0.510x/0.479x**,
but that control does not adjudicate a new T16 integer-dot row-reuse body.
Artifact:
`benchmarks/results/2026-08-05-qwen36-27b-raw-q5-q8-1-dp4a-rowreuse-rejected.json`.

That final Q5 combination is now adjudicated and rejected as well. A temporary
Q5T16/Q8_1 `sudot4` rowtile is BF16-bit exact to its same-width selected
control in all **27** actual-weight geometry cases and passes maximum KL
**0.000489** / top-1 **100%** versus dense. Cross-row reuse makes the integer
dot up to **1.87x** faster than the row-separated control, but the best screen
totals still reach only **0.834x/0.838x/0.865x** dense at rows 2/3/4. Binding
11-sample local64/local128 col8 runs remain negative; the best complete row is
only **0.887x**, and even its prequantized dot is **0.934x** dense. Remove the
kernel, wrapper, key, and test; no sidecar or runtime route was added. Q5
`ssm_out` compressed ownership is closed unless a new representation or
hardware primitive changes the economics. Artifact:
`benchmarks/results/2026-08-05-qwen36-27b-q5t16-q8-1-dp4a-rowreuse-rejected.json`.

A final exact Q4T16 pressure split retains only its architecture-common narrow
shape. The four-column body lowers row-4 VGPR **144 -> 96** without changing
T16 bytes or arithmetic. GPU1 initially qualified both K5,120/N1,024 and
K5,120/N10,240, but the binding W7900 B3 trace rejects the wide crossover:
N10,240 worsens **8.914 -> 9.489 ms (+6.44%)**, kernel sum worsens **0.588%**,
target verify worsens **1.426%**, and complete wall worsens **492.172 ->
496.956 ms (+0.972%)**. N1,024 remains exact and improves **0.869 -> 0.777 ms
(1.118x)** in the complete trace; a counterbalanced 11-sample W7900 leaf
confirms **16.883 -> 12.201 us (1.384x, 11/11 wins)**. Production therefore
keeps col4 only for K5,120/N1,024 verifier rows 2-4 and restores N10,240 to the
retained tile8/pack8 path. This is a narrow physical-family win, not a new
natural25 headline; the campaign remains at **52.652 tok/s / 2.4291x own AR /
22.66% below Vulkan**. Artifact:
`benchmarks/results/2026-08-05-qwen36-27b-q4t16-col4-shape-policy.json`.

The next admitted arithmetic candidate replaces the same 56 wide Q6T16
residents with byte-neutral planar-qmicro records; it does not add another
sidecar to the already 32.892-GiB route. On GPU1 actual weights, all rows-2/3/4
FFN-down and QKV leaves are BF16-bit exact and improve **1.146-1.272x** and
**1.181-1.264x**, respectively. Weighting row 4 by the retained 224/168-call
profile projects about **12.67 ms / 2.57%** complete-wall recovery.

The complete replacement contract also clears before runtime admission. C1
improves **1.424x/1.598x**, while an aligned-record WMMA decoder improves
M64/M512 by **1.143x/1.163x** for FFN-down and **1.178x/1.172x** for QKV; all
six cells are exact and win 11/11 pairs. The first scalar-record WMMA attempt
lost and is superseded. gfx1100 now selects the distinct four-axis qmicro key
through a package capability; root lm-head, narrow V, gfx1151, and explicit
legacy plans retain legacy T16. Rows 5-6 fail closed to exact per-row planar
execution rather than an unmeasured large rowtile.

The adjacent bundle passes **145 tests with 4 environment skips**, actual
materialization/free confirms unchanged **73,113,600 + 43,008,000-byte** sample
payloads, cached tracing names c1/row4/WMMA with zero scratch, and the complete
W7900 B1-B3 transaction/state/provider/teardown gate passes.

Clean, contemporaneous W7900 tracing against `064219ec6` then confirms the
replacement. The 392 wide-Q6 calls fall **59.876 -> 46.714 ms (-21.98%)**,
total kernel sum falls **356.003 -> 343.678 ms (-3.46%)**, and complete wall
falls **504.331 -> 500.205 ms (-0.818%)** with unchanged **8,055** dispatches.
The no-warmup first target pass is slower, but steady passes 2-7 fall **47.742
-> 45.840 ms (-3.98%)**, separating one-time graph capture from retained
arithmetic.

Natural25 true AR advances **21.676 -> 23.031 tok/s (+6.252%)** and B1/B2/B3
advance **38.581/48.712/52.652 -> 39.830/49.848/54.547 tok/s
(+3.239/+2.331/+3.599%)**. All 30 prompt-budget rows and every aggregate scope
improve with exact IDs and acceptance. AR improves faster, so B3/own-AR declines
**2.4291x -> 2.3685x (-2.497%)**; only the absolute throughput win is claimed.
Populated 512/4096 prefill advances **202.550/188.637 -> 207.022/192.528 tok/s
(+2.208%/+2.062%)**, graph AR advances **20.940/19.768 -> 22.107/20.926
(+5.572%/+5.859%)**, peaks are byte-identical, and teardown is clean. Canonical
B3 is now **54.547 tok/s / 19.88% below Vulkan**. Artifact:
`benchmarks/results/2026-08-05-qwen36-27b-q6t16-qmicro-planar-retained.json`.

The same byte-neutral layout is now retained for the distinct K5,120/N248,320
root head. The owner is shape-qualified: the K2,048 MoE head, gfx1151,
explicit legacy plans, X8 requests, and registry misses retain legacy T16.
Actual-weight GPU1 leaves are FP32/top-1 bit-exact and improve proposal top-1
**1.245x** plus target rows 2/3/4 **1.199x/1.184x/1.285x**, every cell with
**11/11** paired wins. Materialization and actual-route tracing prove one
unchanged **1,042,944,000-byte** allocation, intended qmicro c1/row4/top-1
symbols, zero scratch, and clean free; the complete W7900 B1-B3 transaction is
exact.

The tracked-clean W7900 profile cuts combined root work **55.638 -> 48.851 ms
(-12.20%)**: proposal stage1 **40.606 -> 37.127 ms (-8.57%)** and target FP32
rows **15.032 -> 11.724 ms (-22.00%)**. Complete no-warmup profile wall falls
**500.205 -> 478.361 ms (-4.37%)**, but passes 2-7 improve only **45.840 ->
45.553 ms (-0.63%)** because first-use graph capture also moves; physical
kernels and the unprofiled suite bind attribution.

Natural25 true AR advances **23.031 -> 23.069 tok/s (+0.166%)** and B1/B2/B3
advance **39.830/49.848/54.547 -> 40.179/50.813/55.899 tok/s
(+0.875/+1.935/+2.478%)**. Every aggregate scope and every B2/B3 prompt
improves; one B1 prompt is noise-negative **-0.653%**, so no 30/30-row claim is
made. Populated 512/4096 prefill advances **+0.407%/+0.246%** to
**207.864/193.001 tok/s**, graph AR advances **+0.456%/+0.438%** to
**22.208/21.017**, peaks are byte-identical, and teardown is clean. B3 reaches
**2.4231x own AR** and is now **17.90% below Vulkan**. Artifact:
`benchmarks/results/2026-08-05-qwen36-27b-q6t16-qmicro-root-head-retained.json`.

The proposal call-count audit then finds four non-final full-accept catch-ups
that must advance the draft's full-attention K/V state but discard their final
shared RMSNorm, 248,320-row head score, token, and output hidden. The retained
state-only executor preserves the complete embedding/fusion and NextN block,
including its draft K/V write, and omits only that unused final scoring. Partial
accepts, scored proposals, target work, acceptance, and transaction semantics
are unchanged; compatible external executors retain the old full-step fallback.

The tracked-clean one-prompt B3 trace removes exactly four stage-1 heads, four
reducers, and **16 dispatches**. Proposal-update kernel time falls **9.731 ->
3.615 ms (-62.85%)**, update host wall **12.747 -> 7.190 ms (-43.60%)**, and
complete marker wall **478.319 -> 464.238 ms (-2.94%)**. The ordinary 21-head
proposal body is unchanged at **48.671 -> 48.658 ms** of kernels; only the
named discarded-tail subwindow is attributed.

Natural25 B1/B2/B3 advances **40.179/50.813/55.899 ->
41.512/51.974/56.802 tok/s (+3.317/+2.286/+1.616%)** and proposal wall falls
**24.99%/13.56%/7.35%**. All 30 prompt-budget rows and every
full/train/heldout/category scope improve, while IDs, acceptance, complete
state transactions, the **32.892-GiB** peak, and teardown remain exact. True AR
moves **+0.642%** in one-run noise; the NextN tail path cannot execute there.
B3 reaches **2.4466x own AR** and is now **16.57% below Vulkan**. Artifact:
`benchmarks/results/2026-08-05-qwen36-27b-nextn-state-only-tail-retained.json`.

The cache-representative Q5T16 reopening is now retained. Exactly 48 gfx1100
K6,144/N5,120 `ssm_out` tensors become sole-resident source-faithful Q5T16;
other shapes/roles, decode-repack-off, registry misses, and gfx1151 retain dense
BF16. Direct c1, exact local128/col4 rows 2-4, direct larger-row fallback, and
dense WMMA prefill are registered siblings. The complete B1-B3 transaction
proves intended ownership plus logits, reject/partial/full/rollback state,
dynamic graph reuse, provider output, lifecycle, and teardown. Aggregate
component quality is **7.38e-5 max KL / 99.79% top-1**, above the project gate;
the route is therefore quality-gated rather than bit-identical to dense BF16.

Tracked-clean W7900 tracing confirms the production mechanism: 336 dense
`ssm_out` calls fall **37.004 -> 22.911 ms (-38.09%)**, target-verify host wall
falls **380.843 -> 361.440 ms (-5.095%)**, and complete marker wall falls
**464.238 -> 444.023 ms (-4.354%)** at unchanged **8,039** dispatches. Natural25
true AR advances **23.217 -> 24.049 tok/s (+3.585%)** and B1/B2/B3 advance
**41.512/51.974/56.802 -> 43.170/54.621/59.551 tok/s
(+3.994/+5.094/+4.839%)**. Every aggregate scope and every B2/B3 prompt improves;
one B1 prompt is **-2.116%**. Candidate MTP matches its own AR on every row.
Nine of ten prior-route outputs remain byte-identical; only
`general_ja_explain` diverges after token 18 into a fluent memory-bandwidth
explanation rather than the prior fluent FLOPS explanation. B3 acceptance
improves **168/222 -> 169/219** accepted/proposed.

The populated gate is strongly positive as well. At 512/4096, prefill advances
**207.864/193.001 -> 234.014/215.771 tok/s (+12.58%/+11.80%)**, graph AR advances
**22.208/21.017 -> 23.241/21.841 (+4.65%/+3.92%)**, all six final IDs remain
`9707`, and each peak falls by exactly **1,958,215,680 bytes / 1.824 GiB** before
clean teardown. Canonical B3 reaches **2.4762x own AR** and is now **12.53% below
Vulkan**. Artifact:
`benchmarks/results/2026-08-05-qwen36-27b-q5t16-ssm-out-retained.json`.

The refreshed profiler-only matched-module ledger is now:

| Module | hipEngine | llama.cpp Vulkan | Directional gap / status |
| --- | ---: | ---: | --- |
| Q5T16 `ssm_out` | 22.911 ms | 15.119 ms | +7.792 ms; reduced from +22.274 ms, now second-order |
| Q4 FFN gate/up + SiLU | 76.311 ms | 89.534 ms | hipEngine ahead 13.223 ms |
| Wide Q6 FFN-down + QKV | 47.570 ms | 41.486 ms | +6.084 ms; largest remaining arithmetic residual |
| Proposal root stage1 | 31.241 ms / 21 calls | 31.873 ms / 22 calls | hipEngine ahead 0.632 ms; call counts disclosed |
| Target root FP32 rows | 11.798 ms | 9.170 ms | +2.628 ms; second-order residual |

These are synchronized-profiler ranking values, not toplines. No matched
arithmetic family now has a credible 5%-of-complete-wall gap. Exact launch and
transport removal is the next leverage class: the explicit GDN F32-to-BF16
handoff alone is **336 launches / 0.606 ms** of kernels before queue cost.
Consumer-side F32 rounding is rejected before model admission: although exact,
it raises the c2/c3/c4 Q5T16 body from **99.0/105.4/112.6** to
**101.1/112.5/121.7 us**, exceeding the removed **2.9-4.8 us** cast.

The replacement performs the identical round-to-nearest-even BF16 store in the
GDN producer while retaining its FP32 output. Registry keys are owned only by
the sole-resident Q5T16 weight plugin; every miss keeps ordinary GDN, explicit
cast, and BF16 Q5T16. GPU1 c1-c4 production shapes preserve every FP32
output/state bit and every explicit-cast BF16 bit, while complete
GDN-plus-handoff event medians improve **14.270/22.570/29.770/37.066 ->
11.287/19.582/26.762/34.064 us**. The W7900 B1-B3 transaction is exact and
observes both scalar and chain dual-output owners.

The route is retained after the complete gate. Tracing removes **336** casts
and dispatches (**8,039 -> 7,703**), changes GDN+cast **15.522 + 0.606 ->
15.465 ms**, and cuts target kernels/host **0.809%/0.332%**. Complete marker wall
is **+0.209%** queue variance while an independent exact run is **-1.875%**;
retention follows the exact physical and target sub-window wins. Natural25
B1/B2/B3 improves **0.433%/0.104%/0.402%** to **59.790 tok/s / 2.5175x own
AR**, with every token/acceptance decision exact and target verify positive at
every budget. Populated graph AR improves **0.184%/0.281%** at 512/4K; prefill
is noise-flat and peaks are unchanged. B3 is now **12.18% below Vulkan**.
Artifact:
`benchmarks/results/2026-08-05-qwen36-27b-gdn-bf16-handoff-retained.json`.

The subsequent attribution audit corrects that ledger before another kernel is
admitted: those **672 dispatches / 4.528 ms** are seven pre-verify rollback
snapshots of 48 Conv plus 48 recurrent resident states, not selected-row commits.
Native row journals are deferred, so the screened non-deferred direct-final
producer would never execute in production and is removed before W7900 timing.

The retained replacement gives `_StateJournal` immutable live/snapshot pointer
tables and reuses the registered 64-KiB-chunked pair-copy body. Each 96-copy
snapshot becomes one launch; registry, shape, layer-pair, and backend misses
retain the original chain. GPU1 production-shape wall improves **656.504 ->
436.108 us (1.505x)** with every forward/rollback byte exact. The W7900 trace
changes **672 copies / 4.528 ms -> 7 launches / 3.458 ms**, removes **665
dispatches**, and cuts target host and complete marker wall **0.683%/1.555%**.
An independent exact no-marker profile also improves **53.932 -> 54.315 tok/s**.

Natural25 B1/B2/B3 improves **43.357/54.678/59.790 ->
43.792/55.254/60.262 tok/s (+1.004/+1.053/+0.789%)**. All 30 prompt-budget
rows and every full/train/heldout/category aggregate improve; IDs, acceptance,
state, and teardown remain exact. The MTP-only route cannot affect AR, whose
**+1.533%** timing movement lowers the same-session B3 ratio to **2.4991x**.
Canonical B3 is now **11.49% below Vulkan**. Artifact:
`benchmarks/results/2026-08-05-qwen36-27b-journal-snapshot-copy-retained.json`.

The rollback-safe version of that boundary is retained in `82a7f8691`.
Registered gfx1100 chain producers now store the immutable resident Conv/GDN
input bits into one rollback row while writing their ordinary output-row
journals. The journal publishes that snapshot only after the complete native
block retires, so prepare failures cannot restore a partial image while a later
failure after selected-state commit still restores all 96 buffers. Serial mode
and every paired-owner, shape, registry, or backend miss retain the five-row
journal and pointer-copy fallback; gfx1151 deliberately misses.

GPU1 rows 2/3/4 are bit exact and show only **2.478/2.922/3.714 us per layer**
of added store cost. The binding W7900 transaction includes an explicit
post-commit forced failure and restores every original state byte. Tracing
removes all **7 snapshot launches / 3.458 ms**; producers rise only **17.079 ->
18.943 ms**, directly saving **1.593 ms**, while target/complete kernel sums
fall **3.025/2.977 ms** and peak allocation falls exactly **635,437,056 bytes**.
Natural25 target verify improves **0.140%/0.296%/0.506%** at B1/B2/B3. Unrelated
proposal/commit/host variance makes complete B1/B2/B3
**-0.524/-0.318/-0.303%**, so the route is retained as an exact physical and
memory win without advancing the canonical **60.262 tok/s / 11.49%-below-
Vulkan** headline. Artifact:
`benchmarks/results/2026-08-05-qwen36-27b-producer-folded-rollback-snapshot-retained.json`.

Re-rank only the clean `82a7f8691` trace next. Do not treat profiler queue gaps
as production savings. The largest matched arithmetic residuals remain Q5T16
`ssm_out` (**+7.792 ms**), wide Q6 FFN-down/QKV (**+6.084 ms**), and target
root FP32 rows (**+2.628 ms**) versus Vulkan; their previous representation and
row-reuse ladders are closed, so reopening requires a materially new exact
representation, hardware primitive, or cross-family fusion with credible
complete-wall impact.

The first such hardware-primitive screen is negative. On GPU1, the qualified
planar-Q6 integer-WMMA body plus its required D4 activation pack reaches only
**0.412-0.477x** retained direct BF16 T16 for QKV and **0.273-0.279x** for
FFN-down across rows 2-4, with **0/31** candidate wins in every production
cell. Maximum KL is **0.0008801** and top-1 is **100%**, so speed—not quality—
closes this route before source/runtime work or a W7900 gate. The next admitted
cross-family screen was target-root Q6 rowtile plus per-row top-1 publication.
It is FP32 winner-bit and ID exact at rows 2-4 but loses **0.961x/0.949x/0.947x**
with **0/31** wins because its per-eight-logit winner surface and final reducer
cost more than the retained full-FP32-row-plus-argmax chain. No source or W7900
work is admitted. Root compressed publication is now closed alongside the
arithmetic ladders; re-rank exact launch/transport fusion against the clean
trace's **110.084-ms target host-minus-kernel gap across 6,294 dispatches**.

The first launch/transport candidate is retained selectively. Every target pass
contains 64 FFN-down projections followed by 64 standalone BF16 residual adds.
Two gfx1100 `linear+residual` composites preserve both existing BF16 rounding
boundaries and keep projection-plus-add as the registered fallback. GPU1 actual
Q6/Q4 FFN-down weights are bit exact at rows 2-4 and save
**2.372-3.684 / 2.136-3.924 us per layer**. The initial all-row W7900 profile
removed **448 dispatches** and cut marked wall **2.573%**, but row-4 planar Q6
raised its 224-call family **32.831 -> 38.892 ms (+18.46%)** and made every B3
prompt/scope negative (**60.079 -> 59.274 tok/s, -1.341%**). The backend
therefore caps planar Q6 at rows 2-3 while compact Q4 retains rows 2-4.

The tracked-clean selective B3 profile removes **224 dispatches**, improves
complete marker **451.442 -> 448.386 ms (-0.677%)**, and improves target host
**365.954 -> 364.985 ms (-0.265%)**; kernel sums rise **0.453%/0.555%**, so the
retained result is launch/queue contraction rather than arithmetic. Natural25
B1/B2 improve **43.563 -> 43.922 (+0.825%)** and **55.079 -> 55.196 tok/s
(+0.213%)**, with every aggregate scope positive. B3 is mixed at **60.079 ->
59.951 tok/s (-0.214%)** and target verify **+0.240%**. Tokens, acceptance,
state, ownership, memory, and teardown remain exact. The route stays default
under the physical-sub-window policy but does not replace the canonical
**60.262 tok/s / 11.49%-below-Vulkan** headline. Artifact:
`benchmarks/results/2026-08-05-qwen36-27b-ffn-down-residual-fusion-retained.json`.

The rounded successor is now the retained exact physical default. The explicit
gfx1100 `add+rmsnorm/gguf_f32_weight/rounded_bf16_out` owner leaves FFN-down
unchanged, rounds the BF16 residual sum exactly as the standalone add, and feeds
that rounded value through the unchanged RMSNorm tree. It owns all 63
inter-layer boundaries only in native N1 B1/B3 rows2/4; B2's N2 graph, c1,
prefill, F32 residuals, unsupported policies/backends, and every key miss retain
projection -> add -> RMSNorm. GPU1 actual Q6/Q4 boundaries preserve both output
surfaces and save **2.543-4.320 us/layer**, with **22-30/31** paired wins.

The tracked-clean W7900 B3 profile executes **441** rounded leaves and removes
**217 graph dispatches**. Target host/kernel move **364.985 -> 349.141 ms
(-4.341%)** and **257.289 -> 256.721 ms (-0.221%)**; complete marker/kernel move
**448.386 -> 431.236 ms (-3.825%)** and **313.141 -> 312.507 ms (-0.202%)**.
The exact **17/21** ledger, peak bytes, and teardown are unchanged. Natural25 is
mixed against the immediate selective route: B1/B2/B3 move
**43.922/55.196/59.951 -> 43.741/55.332/60.012 tok/s
(-0.412%/+0.247%/+0.103%)**. B2 cannot execute the route; B1 aggregate scopes
are negative, and B3 prompt/scope timing straddles zero. Retain the exact
physical/launch contraction without replacing canonical **60.262 tok/s /
11.49%-below-Vulkan**. Artifact:
`benchmarks/results/2026-08-05-qwen36-27b-rounded-next-rmsnorm-retained.json`.

The next shared-cache KV append candidate is exact but rejected as a runtime
default. Its gfx1100 rows2/4 primitive preserves scalar K/V cache bits across a
reversed physical-page boundary and improves GPU1 graph medians
**1.762x/3.172x**. Routed commit `e3adc89fa` passes the complete W7900 B1-B3
transaction and delivers the intended B3 physical contraction: target writers
fall **448 / 1.083574 ms -> 112 / 0.378642 ms (-65.05%)**, target/complete
dispatches each fall **336**, and target/complete kernel sums improve
**0.832%/0.675%**. The predeclared complete-path gate fails, however: target
host rises **1.405%** and complete marked wall rises **2.786%**. Do not move the
gate or spend natural25 on an ineligible route. The runner is restored to scalar
writes; retain only the exact measured primitive. The canonical physical
default remains `fda35418e` and the headline remains **60.262 tok/s / 11.49%
below Vulkan**. Artifact:
`benchmarks/results/2026-08-05-qwen36-27b-shared-kv-write-runtime-rejected.json`.

The next dense-F32 alpha/beta pair is also retained only as an exact primitive.
Its gfx1100 key combines the two same-input K5,120/N48 projections; rows1-3 use
one flat pair grid and row4 uses two rows/block. All 48 real weight pairs are
scalar-BF16-bit exact at rows1-4 and improve GPU1 48-layer medians
**1.771x/1.786x/1.837x/1.560x**, with **21/21** wins each. The routed W7900
B1-B3 transaction remains exact and observes only rows `{1,2,3,4}`.

Tracked-clean B3 profiling cuts the physical pair family **672 / 3.572950 ms ->
336 / 2.802259 ms (-21.57%)**, removes **336** target/complete dispatches, and
improves target/complete kernel sums **0.140%/0.086%**. The frozen complete-path
gate fails because target host and complete marked wall regress **0.201%/0.189%**.
Do not run natural25 or populated AR. Remove generic pair dispatch ownership and
restore the two scalar leaves; retain only the measured primitive. Artifact:
`benchmarks/results/2026-08-05-qwen36-27b-dense-f32-alpha-beta-pair-runtime-rejected.json`.

A materially different alpha/beta plus snapshot-Conv composite is likewise
correct but runtime-rejected. Its gfx1100 one-grid primitive preserves both
scalar dense-F32 reduction trees and the registered rollback-safe Conv body.
All 48 real layers are bit-exact at rows1-4; GPU1 event medians improve
**2.028x/1.949x/1.892x/1.731x**, synchronized wall agrees, and every cell wins
**21/21**. Routed commit `95fff80d3` passes the complete W7900 B1-B3 state and
output transaction.

Tracked-clean B3 profiling cuts the three-leaf family **1,008 / 5.469564 ms ->
336 / 3.221938 ms (-41.09%)**, removes **672** target/complete dispatches, and
improves target/complete kernel sums **1.016%/0.884%**. The frozen complete-path
gate still fails: target host rises **0.546%** and complete marked wall rises
**1.629%**. Do not run natural25 or populated AR. Remove generic dispatch and
runner ownership; production again uses two scalar projections plus snapshot
Conv. Retain only the measured primitive. Artifact:
`benchmarks/results/2026-08-05-qwen36-27b-dense-f32-pair-chain-conv-runtime-rejected.json`.

The dependent alpha/beta-to-snapshot-GDN owner is also retained only as an
exact primitive. Correctness commit `4e6c8e21d` and producer-locality repair
`648cd4bf0` preserve both scalar dense-F32 projection trees, every recurrent
journal/snapshot bit, FP32 output, and the Q5T16 BF16 handoff. GPU1 rows1-4 and
the complete routed W7900 B1-B3 transaction pass; the repaired row-4 W7900
graph screen improves the three-launch control **3.2212 -> 3.0046 ms/48
layers (1.0721x, 21/21)**.

The final tracked-clean current-clock profile nevertheless fails the frozen
compound gate. The dependent family moves **1,008 / 20.613994 ms -> 336 /
21.480201 ms (+4.202%)** despite removing **672** target/complete dispatches.
Target host and complete marked wall improve **357.514821 -> 346.148599 ms
(-3.179%)** and **439.371729 -> 428.649403 ms (-2.440%)**, but target and
complete kernel sums regress **0.351%/0.268%**. Do not move the gate or spend
natural25/populated-AR time on a profiler-only mixed result. Remove the generic
resolver and runner ownership; production again uses two scalar projections,
snapshot Conv, and snapshot+cast GDN. Post-unroute coverage passes **129/129**
and both runtime files are byte-identical to scalar parent `7eaa6fa05`. Retain
only the registered gfx1100 primitive; gfx1151 remains excluded. Artifact:
`benchmarks/results/2026-08-05-qwen36-27b-dense-f32-pair-gdn-runtime-rejected.json`.

The final graph-runtime audit closes the apparent profiler residual. Seven
remaining pointer-copy launches are required selected-row commits, not
reintroduced snapshots. Explicit graph upload changes only first launch and is
steady-flat. More decisively, the real W7900 B3 target capture is one **816-node
/ 815-edge** chain; four exact **204-node** clones preserve target IDs, hidden
and trunk rows, and every captured linear-state row byte. Across 21 alternating
same-state pairs, unprofiled graph-submit medians move only **40.563484 ->
40.500983 ms (1.00154x)** and complete graph-call medians **40.983989 ->
40.918268 ms (1.00161x)**. The paired median saving is **0.089741 ms**, not the
**~17.2 ms** implied by three rocprof gaps exactly 256 dispatches apart. Those
gaps are instrumentation artifacts; segmentation is rejected without source or
natural25 work. Artifact:
`benchmarks/results/2026-08-06-qwen36-27b-native-target-graph-split-rejected.json`.

A post-closure definition-of-done audit then found one omitted algorithmic
family: the shared MTP ABI and trailing NextN provider support budgets
`(1,2,3,5)`, but the dense GGUF verifier/graph/suite independently stopped at
B3. At this B5 checkpoint B4 was still unsupported and became the remaining
audit gap. The initial exact ten-prompt screen
shows B5 accepting more drafts (**169 -> 184**) and requiring fewer cycles
(**76 -> 61**), but direct six-row fallbacks regress B3 **59.942 -> 23.523
tok/s**. Temporary RED-first row-6 instantiations of the existing exact
Q4/Q5/Q6 templates remove every fallback and improve one favorable code prompt
from **30.103 -> 63.511 tok/s**, versus same-process B3 **59.597 tok/s**.

That prompt does not override the frozen admission gate. Optimized B5 still
costs **61.335 ms/target pass**, above the predeclared **<=50.3 ms** full-suite
break-even and roughly **42.5 ms** Vulkan break-even. A steady trace shows B5
at **923 nodes / 47.774 ms kernels / 53.279 ms span**, versus B3 **819 / 35.163
/ 39.718 ms**; all direct fallbacks are gone and the residual is primarily the
real six-row Q4 (**+7.599 ms/pass**) and Q6 (**+2.275 ms**) work. Do not game a
second full-suite run after the preliminary gate fails. Remove all row-6 and
B5 dense-GGUF production changes, retain B1-B3, and reopen only for a different
speculative algorithm or hardware primitive. Artifact:
`benchmarks/results/2026-08-06-qwen36-27b-dense-b5-budget-rejected.json`.

A final definition-of-done audit then closed the missing B4 interpolation.
Temporary RED-first admission extended the dense verifier, reusable graph,
provider, and suite to budget 4 / five target rows without widening N2 device
accept/commit beyond B1/B2. The first B4 graph divergence localized to the
full-attention V projection: the generic five-row dense-BF16 prefill reduction
differed by one BF16 bit from five retained c1 launches. An exact `ROW_TILE=5`
sibling restored the complete graph, including all 96 Conv/GDN state buffers.
Row-5 sibling instantiations of the existing Q4 pack8 single/dual and Q5T16
templates were then gated on GPU1 and named in scratch-free traces.

The complete W7900 B1-B4 transaction passes eager/graph logits, GPU/CPU
acceptance, reject/partial/full commit, forced rollback, correction, dynamic
positions/reuse, provider output, ownership, memory lifecycle, and teardown.
Before the final Q6 repair, the favorable code prompt measured B4 **51.419
versus same-process B3 57.309 tok/s**, at **79.116 ms/pass**. Its marked trace
found one remaining implementation omission: row-5 planar Q6 still used the
direct body for **127.897 ms across five passes**. Extending the already-gated
col8 template to row 5 raises B4 to **63.199 tok/s**, **11.89% above** the
same-process B3 sample, and cuts target cost to **61.957 ms/pass**.

That is still **23.18% above** the frozen **50.3-ms** full-suite ceiling. The
single fixed code prompt is also only directionally **7.17% below** Vulkan's
aggregate **68.082 tok/s** row; it is not a cross-suite topline. Do not move the
gate or spend the ten-prompt suite after observing it. All B4 runtime, row-5
kernel, test, and harness changes are removed; restored B1-B3 cap tests,
compilation, and diff checks pass. Canonical production remains B3 **60.262
tok/s / 11.49% below Vulkan**. Reopen B4 only for a materially different
speculative schedule, proposer-quality shift, or hardware primitive that can
credibly bring five-row target cost below 50.3 ms/pass. Artifact:
`benchmarks/results/2026-08-06-qwen36-27b-dense-b4-budget-rejected.json`.

The exact/default pass is therefore tabled below parity, not declared a Vulkan
win. Populated prefill and AR already beat the matched stateful Vulkan rows;
natural B3 improved **14.858 -> 60.262 tok/s (4.056x)** and narrowed its gap
from **78.18% to 11.49%** versus Vulkan B3 **68.082 tok/s**. The remaining
matched arithmetic residuals—Q5T16 `ssm_out` **+7.792 ms**, wide Q6
**+6.084 ms**, and root FP32 rows **+2.628 ms** in profiler ranking—are each
below the campaign impact threshold, and their representation, row-reuse,
integer-WMMA, compressed-root, launch-fusion, graph-upload, and graph-split
ladders have been measured and closed. Resume only for a materially new
algorithm/hardware primitive, a changed model/runtime baseline, or an
unprofiled production reproducer with a new >=5%-wall ceiling.

### Residual tuning coverage audit (2026-08-06)

This closing audit maps every profiler-ranked residual to the experiments already
run, then screens only the one materially distinct candidate that was still
uncovered. It does **not** rerun closed grids or turn profiler timing into a
topline. Initial/current hipEngine natural rows use the same ten-prompt
natural25 protocol; Vulkan B3 is the clean `ee0445c99` selected row.

| W7900 metric | Initial hipEngine | Current retained | llama.cpp Vulkan | Initial -> current | Current vs Vulkan | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Populated prefill 512 / 4096 | 50.515 / 50.473 tok/s | 234.580 / 215.127 tok/s | 79.805 / 81.792 tok/s | +364.38% / +326.22% | +193.94% / +163.02% | hipEngine ahead |
| Populated AR 512 / 4096 | 19.556 / 18.649 tok/s | 23.284 / 21.903 tok/s | 12.574 / 12.488 tok/s | +19.06% / +17.45% | +85.18% / +75.39% | hipEngine ahead |
| Natural AR | 20.361 tok/s | 24.114 tok/s | 12.546 tok/s | +18.43% | +92.20% | hipEngine ahead |
| Natural B1 / B2 | 17.128 / 16.005 tok/s | 43.792 / 55.254 tok/s | not canonical; Vulkan selected B3 | +155.68% / +245.23% | n/a | absolute hipEngine gains retained |
| Natural B3 | 14.858 tok/s | **60.262 tok/s** | **68.082 tok/s** | **+305.59% / 4.056x** | **-11.49%** | parity blocked by 7.821 tok/s |
| Natural B3 / own AR | 0.7297x | 2.4991x | 5.4265x | +242.48% | -53.95% | disclosed; absolute B3 is the closure metric |

The refreshed synchronized-profiler module ledger is directional attribution,
not a throughput denominator:

| Module | Last retained step | Current hipEngine | llama.cpp Vulkan | Current directional gap | Coverage verdict |
| --- | --- | ---: | ---: | ---: | --- |
| Q5T16 `ssm_out`, 336 calls | 37.004 -> 22.911 ms (-38.09%) | 22.911 ms | 15.119 ms | +7.792 ms; 1.75% of complete wall | Raw/T16, row reuse, workgroups, DP4A, WMMA, rotating residency, and producer/consumer handoffs covered; exhausted |
| Q4 FFN gate/up + SiLU | retained compact/fused route | 76.311 ms | 89.534 ms | hipEngine ahead 13.223 ms | no residual debt |
| Wide Q6 FFN-down + QKV, 392 calls | 59.876 -> 46.714 ms (-21.98%); refreshed clocks shown at right | 47.570 ms | 41.486 ms | +6.084 ms; 1.37% of complete wall | Legacy/planar T16, col4/8/16, row2-6, WMMA, qmicro, residual fusion, integer-WMMA, and scalar-DP4A covered; exhausted |
| Proposal root stage1 | 40.606 -> 37.127 ms in the retained root pass | 31.241 ms / 21 calls | 31.873 ms / 22 calls | hipEngine ahead 0.632 ms; call mismatch disclosed | no positive residual |
| Target root FP32, 7 calls | 15.032 -> 11.724 ms (-22.00%); refreshed clocks shown at right | 11.798 ms | 9.170 ms | +2.628 ms; 0.59% of complete wall | Root residency, rowtiles, compact top-1, WMMA/quantized, fusion, and scalar-DP4A covered; exhausted |

The family-specific conclusions are binding:

- **Q5T16 `ssm_out`:** dense BF16, raw-Q5 wave/SWAR/pack8/column/rowbatch,
  direct and exact T16 rows2-4 at local32/64/128 and col4/8/16, raw-Q5 and
  Q5T16 Q8_1 `sudot4` row reuse, dense WMMA, all-48-plane rotating residency,
  and F32/BF16 handoffs have all been measured. Reopen only for a genuinely new
  representation or hardware primitive; a fused GDN-to-Q8_1 idea is sub-ms and
  does not reopen the rejected DP4A layouts.
- **Wide Q6:** the new planar-Q6/Q8_1 scalar-dot leaf wins its tracked-clean
  qualified family **46.900 -> 40.992 ms (-12.60%)** and full-suite B1/B2/B3
  **+1.876%/+1.106%/+2.316%**, but planar c1 regresses true AR **23.271 ->
  21.821 tok/s (-6.235%)**. A transaction-consistent X8-c1 rescue then wins the
  all-56-weight W7900 component loop **6.269 -> 5.803 ms (1.080x)** with
  **54/56 top-1** and max KL **9.71e-6**, but costs **3.140 GiB** and fails the
  binding one-prompt gate: AR **24.194 -> 22.533 (-6.867%)**, B3 **65.896 ->
  67.592 tok/s (+2.573%)**. Broader sidecar work is intentionally skipped; all
  runtime ownership and sidecar code are removed, while the registered planar
  primitive remains diagnostic-only.
- **Target root FP32:** the piggyback K5,120/N248,320 scalar-DP4A screen does
  beat retained planar qmicro inclusively at rows2/3/4: best local policies are
  **1.405x/1.420x/1.357x**, all **31/31** paired wins. It nevertheless matches
  top-1 in only **8/9 rows (88.89%)**, below the project-wide 90% gate, despite
  max KL **3.79e-7**. It is rejected before runtime/W7900 work. No credible
  untried exact root design remains, and perfect root closure is only 0.59% of
  complete wall.

The three positive module gaps sum to only **16.504 ms / 3.72%** of the
444.023-ms reference marker. Even perfect arithmetic elimination cannot supply
the **12.98%** throughput increase needed to move 60.262 to 68.082 tok/s.
There is no optimization or benchmark running after this publication. Resume
only for a materially new speculative algorithm/schedule, hardware primitive,
changed model/runtime/compiler/driver/Vulkan baseline, or a newly profiled
production reproducer with a credible >=5%-wall ceiling. Do not repeat the
closed Q5/Q6/root grids, planar-Q8 runtime, duplicate-X8 sidecar, graph
upload/split, or exact B4/B5 work without such a new mechanism.

Artifacts:

- `benchmarks/results/2026-08-06-qwen36-27b-residual-tuning-coverage-audit.json`
- `benchmarks/results/2026-08-06-qwen36-27b-planar-q6-q8-1-runtime-rejected.json`
- `benchmarks/results/2026-08-06-qwen36-27b-dense-b4-budget-rejected.json`

### Historical first parity-or-exhaustion checklist (2026-08-06)

This checklist closed the first `ee0445c99` campaign and is superseded for
current comparison by the later `c8e03ce81` refresh and 2026-08-07 audit below.
At that publication, the prompt-to-artifact audit accounted for exact
model/platform identity, true-AR and transactional-MTP correctness, full
train/heldout/category promotion gates, retained and rejected families,
per-module Vulkan attribution, B4/B5 economics, source/default cleanup,
commands, commits, rollups, and active process/task state are all accounted for.
Matched MTP parity is **not** achieved; completion is by measured exhaustion at
canonical B3 **60.262 versus Vulkan 68.082 tok/s (-11.49%)**.

Dense production and the campaign suite are restored to B1-B3; no B4/B5 row
template, planar-Q8 runtime selector, or X8 sidecar remains. Campaign-scoped
post-removal tests and W7900 transactions are green. The repository-wide
2026-08-03 baseline's 20 pre-existing failures in unchanged parent paths remain
disclosed and are not falsely relabeled green. No optimization, benchmark, or
campaign task remained at that publication. Compact historical checklist:
`benchmarks/results/2026-08-06-qwen36-27b-parity-or-exhaustion-completion-audit.json`.

### Latest Vulkan refresh and campaign reopen (2026-08-06)

The human refreshed `/home/lhl/llama.cpp/llama.cpp-vulkan` to tracked-clean
`c8e03ce81` (build 10290). A new Release Vulkan build on the same W7900, Mesa
26.1.4, model bytes/hash, full offload, FA, and F16 K/V gives:

| Boundary | Prior `ee0445c99` | Latest `c8e03ce81` | Change |
| --- | ---: | ---: | ---: |
| llama-bench pp512 | 792.308 | 792.621 tok/s | +0.040% |
| llama-bench pp4096 | 754.093 | 753.758 tok/s | -0.045% |
| llama-bench tg128 | 12.61795 | 12.61632 tok/s | -0.013% |
| stateful 512/128 prefill / AR | 79.805 / 12.57431 | 79.351 / 12.53468 tok/s | -0.570% / -0.315% |
| stateful 4096/128 prefill / AR | 81.792 / 12.48779 | 80.622 / 12.45871 tok/s | -1.430% / -0.233% |
| natural matched B3 | 68.082 | 67.682 tok/s | -0.588% |
| natural selected budget | 68.082 B3 | **69.798 B4** | **+2.520%** |

The selected-budget shift is a steady-state protocol correction, not evidence
that the 40-commit source delta accelerated Qwen3.6 arithmetic. The prior B4
sweep had one unique-row first prompt at 48.60 tok/s while its other nine rows
and 171/242 ledger match the refresh. Candidate-local warmup moves that prompt
to the mid-70s; the explicit-device B1-B4 sweep and rich direct packet both
select B4. Latest B4 accepts **171/241 drafts**, reaches **69.798 transition
tok/s**, and is **3.126% above** same-harness B3. After the retained submission
graphs and direct retirement below, canonical hipEngine B3 is **61.394 tok/s**,
**9.290% below matched Vulkan B3** and **12.040% below selected Vulkan B4**;
end-to-end closure requires **+13.688%**.

The old query profile is superseded. Fresh latest-source B3/B4 Vulkan query
profiles and a clean current hipEngine B3 kernel/marker/copy trace now reconcile
to their own walls: Vulkan B3/B4 query totals are **625.042/606.104 ms**, within
**4.97%/5.83%** of client wall, while the hipEngine **455.694-ms** marker is
within **0.01%** of its suite decode wall. Explicit row/call normalization
removes Vulkan's `n=36` prefill and first-shape graph-build executions and
replaces hipEngine's first target/commit capture with passes 2-7.

The exhaustive result changes the optimization diagnosis. For seven matched
rows=4 target passes, hipEngine kernels total **255.485 ms** versus Vulkan
**277.203 ms** (hipEngine ahead **21.718 ms**). Proposal/update is also ahead at
**52.139 vs 65.224 ms**, and complete call-normalized GPU work is **311.225 vs
342.427 ms** (ahead **31.202 ms**). The complete wall still loses because the
capture-normalized hipEngine non-kernel residual is **65.134 ms** versus
Vulkan's inferred **23.789 ms**: a **41.346-ms** graph/queue/host deficit, almost
exactly the natural matched-B3 wall gap of **43.664 ms**.

| Refreshed matched bucket | hipEngine | llama.cpp Vulkan | HIP - Vulkan | Status |
| --- | ---: | ---: | ---: | --- |
| Steady graph/queue/host residual | 65.134 ms | 23.789 ms | **+41.346 ms** | largest deficit |
| Q5 `ssm_out` | 22.577 ms | 15.144 ms | **+7.433 ms** | slower |
| Q6 FFN-down | 32.602 ms | 28.857 ms | **+3.745 ms** | slower |
| Full-attention K/V | 4.958 ms | 2.040 ms | **+2.918 ms** | slower |
| Linear-attention Q6 QKV | 14.035 ms | 12.268 ms | **+1.767 ms** | slower |
| Target root projection | 11.762 ms | 10.693 ms | **+1.069 ms** | slower |
| All FFN | 128.995 ms | 138.501 ms | -9.507 ms | hipEngine ahead |
| GDN fused core | 16.765 ms | 19.697 ms | -2.932 ms | hipEngine ahead |
| Complete target kernels | 255.485 ms | 277.203 ms | -21.718 ms | hipEngine ahead |

This means the user's module rule remains necessary but must include submission
as the first and largest module. Arithmetic rows alone cannot close the gap
because aggregate hipEngine arithmetic is already faster. A steady target
replay spends **5.2-5.6 ms** in internal device gaps across 835 dispatches;
first-pass capture adds another **~80.3 ms** but is normalized out. Graph upload
and four-way graph splitting are already rejected, so the reopened source audit
must find a materially different command-buffer/persistent-composite mechanism
before repeating those screens. After submission, close Q5, Q6-down,
full-attention K/V, Q6-QKV, and root sequentially.

Selected Vulkan B4 is a separate topology endpoint: it trades one fewer target
cycle (**275.59 -> 246.21 ms** normalized target GPU work) for more draft work
(**65.22 -> 71.69 ms**). Thus matching B3 shaders and queue behavior is still
not sufficient; the final gate remains complete natural25 against B4. Compact
artifacts:
`benchmarks/results/2026-08-06-qwen36-27b-llamacpp-vulkan-c8e03ce81-refresh.json`
and
`benchmarks/results/2026-08-06-qwen36-27b-latest-vulkan-profile-ledger.json`.

### D27-R2 submission progress: proposal/target graphs and direct retirement

The first materially different submission mechanism is retained. Clean
`a3e4912ee` captures a complete fixed-address B1/B2/B3 NextN chain with device
token handoff and one result drain. Against `01291b066`, natural25 B1/B2/B3
improves **43.792/55.254/60.262 -> 44.035/56.014/61.020 tok/s
(+0.555%/+1.376%/+1.259%)**. Every one of the 30 prompt-budget rows and every
train/heldout/category scope improves with exact IDs and acceptance. A
same-loaded-model B3 transaction cuts proposal median **67.593 -> 58.429 ms
(-13.56%)** and complete decode median **370.917 -> 365.821 ms (1.01393x)**.

Clean `09abd51968` then extends the existing N2 device
`VERIFY|ACCEPT|selected COMMIT|UPDATE_CURSORS` contract through B3 and wires it
into the exact transactional verifier. Full-room session-stream cycles return
acceptance and every target top-1 row in the same bounded payload; selected
Conv/GDN/BF16-hidden state and target cursors are already committed when the
graph retires. Diagnostic logits, caller streams, output-cap tails, and misses
remain on N1/eager paths. The reject/partial/full, forced-rollback, dynamic-
reuse, correction, K/V, hidden/state, ownership, and teardown gate is exact.

The immediate N2 natural packet is mixed at **44.224/56.037/60.903 tok/s
(+0.428%/+0.040%/-0.193%)** versus the proposal graph. Nine of ten prompts
improve at every budget, but one transient row per budget makes some aggregate
scopes negative. At that stage the exact physical default did not replace the
proposal packet's **61.020 tok/s** row; direct retirement below supersedes it.
The physical evidence remains decisive: all 17 paired target+
policy samples improve **42.441009 -> 41.489807 ms (1.022926x)**, and suite-wide
`target_commit_finish` falls **96.675/68.441/57.919 ->
2.032/2.646/6.354 ms** at B1/B2/B3.

Clean `92e823a2d` removes the remaining proposal-to-target host boundary for
cached full-room cycles without combining arithmetic. The proposal records a
private completion event; N2 waits, injects each device proposal ID into both
i64-embedding and i32-acceptance metadata columns, and returns proposal
`(ID, value)` rows in its existing bounded payload. Scheduler planning uses
shape-only rows and is rebound only after the same CPU acceptance oracle checks
the real IDs. Captures, diagnostics, caller streams, long contexts, and
output-cap tails retain independent synchronization.

Natural25 B1/B2/B3 becomes **44.496/56.350/61.394 tok/s
(+0.616%/+0.559%/+0.808%)** versus immediate N2. B2/B3 improve all **10/10
prompts and 7/7 scopes**; B1 improves **9/10 prompts and 7/7 scopes** with one
**-0.183%** timing row. Versus the prior canonical proposal packet, all
headlines improve **+1.046%/+0.599%/+0.613%**. IDs, acceptance, state, rollback,
and teardown remain exact. A 17-pair B3 screen improves **361.138 -> 359.828 ms
(1.003640x, 15/17 wins)** with median paired change **-1.821 ms**.

Capture-normalized profiling first moves complete wall **376.569 -> 366.417 ->
364.004 ms** across control, proposal graph, and N2 target policy. The warm
post-keep direct profile reports **366.034 ms** because distinct-run kernel sum
rises **315.018 -> 321.779 ms**, but non-kernel residual falls unambiguously
**48.985 -> 44.255 ms (-9.656%)**. The complete submission stack cuts that
residual **65.134 -> 44.255 ms (-32.056%)** and leaves **20.467 ms** versus
Vulkan B3's inferred **23.789 ms**. Canonical hipEngine B3 is now **9.290%
below** matched Vulkan B3 and **12.040% below** selected Vulkan B4, requiring
**13.688%**.

The final materially different submission screen is exact but rejected. A
single parent executable clones the cached proposal and N2 graphs as dependent
child nodes and places the six B3 metadata-ID copies plus bounded proposal
payload copy between them. HIP child composition already loses in a tiny GPU1
screen (**39.061 -> 42.090 us**, 0.9280x). On W7900, the production-shaped
13-pair B3 screen moves decode **358.160 -> 360.861 ms (0.99251x)** with a
**+2.881-ms** paired median and **0/13** wins, despite identical IDs and
acceptance. Besides child-node overhead, one-submit assembly defers the proposal
until the target transaction is ready and therefore gives up the retained early
asynchronous proposal overlap. No code or flag is kept. The direct event/device
handoff is the current submission optimum; D27-R2 advances to Q5 `ssm_out`, then
Q6 FFN-down, full-attention K/V, Q6 QKV, and root. Artifacts:
`benchmarks/results/2026-08-06-qwen36-27b-native-submission-graphs-retained.json`,
`benchmarks/results/2026-08-06-qwen36-27b-direct-proposal-target-handoff-retained.json`,
and
`benchmarks/results/2026-08-06-qwen36-27b-parent-child-submission-rejected.json`.

The refreshed Q5 `ssm_out` row is source-audited and closed without repeating
rejected kernels. Latest matched attribution is **22.577 ms hipEngine versus
15.144 ms Vulkan** across the same 336 rows=4 calls. However, the three Q5
shader/dataflow blobs (`mul_mat_vec_q5_k.comp`, `mul_mat_vecq.comp`, and
`mul_mat_vecq_funcs.glsl`) are byte-identical between `ee0445c99` and
`c8e03ce81`; the comparator update only adds the separate GLA op. The current
sole-resident Q5T16 owner remains the retained **37.004 -> 22.911 ms** result,
and no Q5-specific implementation or selector hunk changed after its promotion.
The prior ladder already covers dense BF16, raw Q5 wave/SWAR/pack8, direct and
rowtiled T16 local/column geometries, raw/T16 Q8_1 `sudot4`, WMMA, rotating
48-plane residency, and producer handoffs. Binding integer-dot candidates topped
out at **0.798x** raw and **0.887x** T16 versus dense. Reopen only for a new
representation, RDNA3 primitive, credible cross-family fusion, or changed Q5
source. D27-R2 advances to Q6 FFN-down. Artifact:
`benchmarks/results/2026-08-06-qwen36-27b-latest-q5-source-audit-exhausted.json`.

The refreshed wide-Q6 family is likewise source-audited and closed. Current
rows=4 attribution is **32.602 versus 28.857 ms** for 224 FFN-down calls and
**14.035 versus 12.268 ms** for 168 linear-attention QKV calls, hipEngine versus
Vulkan. All six shader dependencies of Vulkan's floating four-column Q6 matvec,
plus `ggml_vk_get_dequantize_mul_mat_vec()` and the AMD Q6 MMVQ rejection in
`ggml_vk_should_use_mmvq()`, are byte-identical between `ee0445c99` and
`c8e03ce81`; the refresh adds only the unrelated GLA op. hipEngine's retained
byte-neutral sole-resident planar-qmicro owner already reduced the 392-call
family **59.876 -> 46.714 ms (-21.98%)**. The prior coverage matrix closes dense
BF16, legacy T16, col16/8/4, interleaved/planar qmicro, aligned/scalar WMMA,
rows 1-6, residual/RMSNorm fusion, D4 integer WMMA, Q8_1 `sudot4`, and the
transaction-consistent X8 c1 rescue. Reopen only for a new byte-neutral or
sole-resident representation, RDNA3 primitive, credible cross-family fusion,
or changed Q6 source. D27-R2 advances to full-attention K/V. Artifact:
`benchmarks/results/2026-08-06-qwen36-27b-latest-q6-source-audit-exhausted.json`.

The full-attention K audit finds one real uncovered route rather than another
representation ladder. All 16 K tensors already carry the retained exact
compact-Q4T16 sidecar, and the actual `blk.3.attn_k` W7900 leaf measures
**20.152 -> 12.201 us (1.652x)**, but the staged full-attention helper predates
that promotion and hard-bypasses it into the old pack8 grid-Y batch. The helper
now prefers the registered compact-col4 owner and retains pack8 grid-Y, then
scalar linear, as exact missing-sidecar/key fallbacks. The CPU/fake native-cycle
file passes **15/15**; the binding W7900 B1-B3 transaction is exact across
logits, reject/partial/full/rollback state, dynamic reuse, K/V, hidden/provider
output, ownership, and teardown, and observes no old pack8 K owner.

A marked B3 trace replaces exactly **112 pack8 / 2.409537 ms** with **112
compact-col4 / 1.501054 ms (-37.70%, 1.605x)**. The separate ten-prompt packet
is exact but timing-negative by **0.158%-0.444%** at B1-B3, so it does not
replace canonical **61.394 tok/s**. The binding same-loaded-model,
separately-cached, counterbalanced B3 screen instead improves median decode
**361.601 -> 361.232 ms (1.001021x)** with a **-0.908-ms paired median** and
**13/17** wins. Retain the exact target-window win without changing the
headline. The inferred matched full-attention K/V bucket falls **4.958 ->
4.050 ms**, still **2.010 ms** behind Vulkan's **2.040 ms**, so D27-R2 advances
within the module to V before root. Artifact:
`benchmarks/results/2026-08-06-qwen36-27b-full-attention-k-sidecar-retained.json`.

The next V ownership unit is retained, and review recovers the complete
promotion packet that was measured but omitted from the original commit's
publication. Eight Q6_K `layers.*.attn_v` tensors now use one
`gguf_q6_k_t16_qmicro_planar_v1` owner across c1, native rows, and populated
prefill. Relative to committed production this replaces **80.0 MiB** of dense
BF16 with **32.8125 MiB** of qmicro tiles, saving **47.1875 MiB**; the earlier
“removes duplicate 32.8 MiB” wording was imprecise and referred to a rejected
transient dual-resident experiment.

Actual-weight GPU1 rows 2/3/4 improve **1.237x/1.251x/1.341x**, are BF16-bit
exact to independent qmicro c1 launches, and pass dense-relative max KL
**1.28e-7** / top-1 **100%**. The complete W7900 binding B1-B3 transaction
passes logits, accept, reject/partial/full/rollback state, dynamic graph reuse,
K/V, physical ownership, provider output, and teardown. A marked W7900 trace
replaces exactly **56 dense V / 1.760127 ms** with **56 qmicro V / 1.472815 ms
(-16.323%, 1.195x)**.

The full ten-prompt packet versus the K-sidecar route remains exact and moves
true AR/B1/B2/B3 **24.249/44.319/56.261/61.122 ->
24.247/44.635/56.290/61.235 tok/s (-0.006%/+0.714%/+0.052%/+0.185%)**.
Generated IDs are unchanged and every GPU accept matches CPU. B1 aggregate
acceptance improves **115/127 -> 115/126**; B2/B3 aggregates remain **151/182**
and **169/219**, with prompt-local sequence changes disclosed in the artifact.
Candidate B3 is still **0.260%** below canonical **61.394**, so the topline does
not change. The inferred combined K/V bucket falls **4.050 -> 3.762 ms**, still
**1.723 ms / 1.845x** above Vulkan's **2.040 ms**. D27-R2 therefore remains in
K/V for exact residual source/dispatch reconciliation before root. Artifact:
`benchmarks/results/2026-08-06-qwen36-27b-full-attention-v-planar-qmicro-retained.json`.

The residual source audit next reproduces Vulkan's AMD Q6 floating geometry
rather than assuming the earlier local128 col4 screen covered it. Vulkan
explicitly rejects Q6_K Q8_1/MMVQ on AMD and assigns one local32 wave two output
columns plus every activation row. On actual GPU1 K5,120/N1,024 V bytes, the
source-layout local32/col2 leaf is exact and improves rows 1/2/3 from
**13.495/13.871/15.066 us** to **8.695/11.780/13.029 us**, but the binding B3
rows=4 specialization reaches **144 VGPR** and regresses **15.403 -> 28.247 us
(0.545x)**. Replacing the sole qmicro owner would slow B3; preserving raw Q6 as
an additional resident would add **32.8125 MiB** across eight tensors without
helping B3. The diagnostic source, wrappers, registry leaves, and tests are
removed. Q6 V is source-audited closed unless a materially different rows=4
register schedule appears; reconciliation advances to Vulkan's genuinely
uncovered Q4_K Q8_1/integer-dot route. Artifact:
`benchmarks/results/2026-08-06-qwen36-27b-q6-vulkan-source-geometry-rejected.json`.

The Q4_K route is now source-reproduced and rejected rather than inferred from
older selected-MoE screens. Clean Vulkan `c8e03ce81` uses one local32 wave per
output, `K_PER_ITER=16`, four packed integer dots per lane iteration, and one
cached Q8_1 quantization when Q/K/V share an activation. Disposable raw-layout
and compact-T16 leaves reproduce that schedule on actual GPU1 Q/K/Q4-V tensors.
At binding rows=4, raw prequantized Q/K/V move current exact
**54.708/9.200/9.238 -> 54.850/8.237/8.293 us**: narrow K/V improve, but wide Q
is neutral-negative. Even using the most favorable **3.412-us** quantization
median once across all projections, a Q4-V layer regresses **73.145 -> 74.792 us
(+2.25%)** and a Q6-V layer regresses Q+K **63.908 -> 66.499 us (+4.06%)**.
True-AR rows=1 is worse still: a Q4-V Q+K+V layer moves **48.095 -> 57.167 us
(+18.86%)**. Every actual-weight top-1 agrees and max KL is at most
**7.57e-7**, so performance—not quality—rejects the route. The compact col2
upper bound is also slower inclusive; all diagnostic code is removed.

Full-attention K/V therefore remains **3.762 ms** versus Vulkan **2.040 ms**, but
the current Vulkan Q4/Q6 source-mechanism ladder is explicitly exhausted under
hipEngine's exact compact owners. Reopen only for a producer-fused Q8_1 path
below the approximately **1.29-us** mixed-layer break-even with full-category
quality, a materially new primitive, or changed source/compiler/model. D27-R2
advances to root projection. Artifact:
`benchmarks/results/2026-08-07-qwen36-27b-q4-vulkan-q8-1-source-rejected.json`.

The root review finds one final source mechanism but rejects the complete
package. Clean Vulkan `c8e03ce81` assigns one local32 wave two adjacent Q6_K
outputs, two half-wave QK-block partitions, four nested scaled segment sums,
and every activation row. A temporary byte-neutral planar-qmicro reproduction
passes the CPU-reference gate and actual-root quality at maximum KL
**1.873e-15 / 100% top-1**. On W7900, local32/col2 moves rows1/2/3/4
**1493.577/2327.534/2470.719/2507.544 ->
1533.753/1944.428/2237.430/2391.231 us**: verifier rows win, but scalar c1
regresses **2.69%**.

That c1 loss cannot be omitted. Routing only the verifier changes logits versus
scalar AR by up to **2.86e-6**; routing source c1 and rows together restores the
exact B1-B3 transaction and makes every batched source output FP32-bit equal to
repeated source c1. The call-normalized profile then gives the binding result:
41 MTP scalar roots regress **60.695327 -> 62.733971 ms**, while seven row-4
roots improve only **11.915263 -> 11.365432 ms**. MTP root total therefore
regresses **72.610590 -> 74.099403 ms (+2.050%)**; including the 24 AR roots,
all traced root work regresses **108.110536 -> 110.818849 ms (+2.505%)**.

The complete ten-prompt packet happens to move AR/B1/B2/B3
**22.926/44.730/56.194/61.147 -> 23.616/44.823/56.452/61.429 tok/s**, with
identical IDs and acceptance, but that one-run result conflicts with both the
counterbalanced leaf and exact family trace and is treated as clock/queue
variance. Completion screens do not rescue the package: an old-association
local64/col4 sibling loses **5-11%**; independent source-wave packing is mixed
below 1%; source local32/col4 accelerates rows2-4 but worsens c1 on both W7900
(**0.961x**) and RX 7900 XTX (**0.933x**); col8 loses decisively. Every temporary
HIP body, wrapper, key, selector, flag, and test is removed.

Root remains **11.762 ms** versus the matched Vulkan **10.693 ms** in the frozen
module ledger, but its current source-faithful floating, exact-association,
integer-dot, compressed-publication, and geometry ladders are exhausted. Reopen
only for a materially new association-preserving c1 primitive or changed
source/compiler/model. D27-R2's ranked module pass is complete without a new
retained route; canonical B3 remains **61.394 tok/s**. Artifact:
`benchmarks/results/2026-08-07-qwen36-27b-vulkan-source-root-geometry-rejected.json`.

### Final latest-Vulkan parity-or-exhaustion gate (2026-08-07)

Clean `d1f26fb5c` populated controls close D27-R3 without repeating the already
completed equivalent natural25 production control. With one discarded warmup
reset and three measured persistent-session resets, exact/default hipEngine
measures:

| W7900 boundary | Initial hipEngine | Final hipEngine | Latest Vulkan | Final vs initial | Final vs Vulkan |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512/128 populated prefill | 50.515 | **235.434 tok/s** | 79.351 | **+366.07%** | **+196.70%** |
| 4096/128 populated prefill | 50.473 | **216.784 tok/s** | 80.622 | **+329.50%** | **+168.89%** |
| 512/128 graph AR | 19.556 | **23.296 tok/s** | 12.535 | **+19.12%** | **+85.85%** |
| 4096/128 graph AR | 18.649 | **21.897 tok/s** | 12.459 | **+17.42%** | **+75.76%** |
| Natural true AR | 20.361 | **22.926 tok/s** | 12.528 | **+12.60%** | **+83.00%** |
| Natural exact B3 | 14.858 | **61.147 tok/s** current / **61.394** canonical | **69.798 B4** | **+311.54%** current | **-12.394%** current / **-12.040%** canonical |

All six populated final IDs are `9707`, logits are finite, measured graph
captures are reused, timing variation is at most 1.20%, tracked peaks are
29.786/32.611 GiB, and every tracked byte frees. The post-module ten-prompt
control remains exact at AR/B1/B2/B3 **22.926/44.730/56.194/61.147 tok/s**,
with all greedy IDs, GPU/CPU acceptance, stage accounting, and teardown green.
It was measured with the rejected root selector forced off; production source
was subsequently byte-restored, so policy forbids another equivalent >5-minute
suite merely to change dirty provenance.

The review is complete but parity is not. Submission reduced its inferred
matched residual **41.346 -> 20.467 ms**; Q5 and wide Q6 were source-unchanged
and already exhausted; exact compact K plus sole-resident planar-Q6 V reduced
full-attention K/V **4.958 -> 3.762 ms**; source-faithful Q6 V, Q4 MMVQ, and Q6
root packages all lose under binding rows/quantization/c1 economics and were
removed. The review also recovered the omitted V publication, corrected the Q4
source reproduction to `K_PER_ITER=16`, and repaired the root CPU-oracle,
fail-closed, cleanup, and scalar-association gaps before rejecting that route.
No unresolved correctness or cleanup finding remains.

“Exhausted” is scoped, not absolute: it covers the current exact/default
algorithm, latest comparator source, compiler/driver, and known gfx1100
primitives. Reopen for a materially new speculative schedule or proposer-quality
shift, an association-preserving/producer-fused primitive, changed source or
stack, or a newly profiled production mechanism with a credible >=5%-wall
ceiling. Compact authority:
`benchmarks/results/2026-08-07-qwen36-27b-latest-vulkan-parity-exhaustion-audit.json`.

### Post-closure MTP-only cycle audit (2026-08-07)

A follow-up deep profile resolves why the module-improved path still trails.
The unprofiled ten-prompt walls decompose as follows; cycles are target passes
needed for the 240 timed transitions:

| Comparison | hipEngine cycles / ms per cycle | Vulkan cycles / ms per cycle | Complete gap | Cycle-count contribution | Per-cycle contribution |
| --- | ---: | ---: | ---: | ---: | ---: |
| Matched B3 | 76 / **51.644** | 77 / **46.052** | **+379.005 ms** | **-46.298 ms** (topology favors hipEngine) | **+425.303 ms** |
| hipEngine B3 vs selected Vulkan B4 | 76 / **51.644** | 69 / **49.833** | **+486.478 ms** | **+376.685 ms / 77.4%** | **+109.793 ms / 22.6%** |

Thus matched B3 is not behind because of aggregate acceptance or extra target
passes: hipEngine executes one fewer pass and still loses because every complete
proposal/verify transaction costs about **5.59 ms more**. The selected B4 gap is
different and mostly algorithmic. B4 saves seven passes aggregate; the largest
single effect is `mixed_ja_en_review`, where it needs five passes versus
hipEngine B3's nine and contributes **249.950 ms** of the total gap. B4 is not
uniformly better—it needs more passes on two Japanese/mixed prompts—so this is
suite-wide schedule evidence, not permission for prompt-conditioned selection.

The unprofiled hipEngine B3 stage ledger across all 76 cycles assigns
**3665.650 ms / 48.232 ms per cycle (93.39%)** to the device-chained proposal
plus target graph submit. Metadata/call-boundary preparation is **1.104
ms/cycle**, proposal launch/full-accept update **1.608 ms**, result readback
**0.239 ms**, scheduler replay **0.380 ms**, and commit/finish only **0.082
ms**. Commit is therefore not the residual.

A clean `69b3b691b` one-prompt runtime trace confirms the boundary. Its seven
`target_verify` markers consume **349.592 / 364.677 ms**; seven
`hipStreamSynchronize` calls waiting for proposal+target retirement consume
**336.511 ms**, while target `hipGraphLaunch` API time is **3.782 ms**, target
copy-API time **2.474 ms**, and target host work outside traced APIs **6.785
ms**. The
matching unprofiled post-module hipEngine control is **356.127 ms / seven
cycles**. Fresh unprofiled Vulkan controls are **320.969 ms B3 / seven cycles**
and **300.016 ms B4 / six cycles**, with the same content hash.

Per-graph Vulkan timestamp queries close the comparison without assigning HTTP
or server residual to GPU work. B3's 21 draft graphs, seven target graphs, and
seven post-target/update graphs total **50.782 + 261.589 + 6.918 = 319.289
ms**, or **99.48%** of its independent 320.969-ms wall. B4 totals
**57.995 + 229.985 + 6.338 = 294.318 ms**, or **98.10%** of its independent
300.016-ms wall. On the exact B3 trajectory, about **2.91 ms/cycle** of
hipEngine's **5.02-ms** deficit lies inside the combined proposal+target submit
relative to Vulkan's queried graphs, and about **2.11 ms/cycle** lies in
metadata/proposal-update/readback/scheduler control outside that submit.

The causal verdict is therefore precise: **matched B3 is limited by the
proposal-to-target verification transaction, not commit, CPU sampling, or
HTTP; selected B4 additionally wins mostly by avoiding target passes.** The
prior semantic kernel ledger remains an arithmetic ranking tool, but summed HIP
kernel durations and Vulkan adjacent-query intervals are not complete-wall
interchangeable. In particular, rocprof's three approximately 5.7-ms gaps
exactly 256 dispatches apart remain proven instrumentation artifacts.

No current mechanism is newly admitted. The largest route is a materially new
five-row/B4 verifier below the frozen **50.3-ms/pass** gate; the already exact
row-5 implementation costs **61.957 ms/pass** and was removed. The matched-B3
residual would require a genuinely persistent/hostless multi-cycle owner or a
new target-verification composite; graph upload, splitting, and parent/child
composition are already rejected. Any proposer-quality or adaptive-budget work
must pass the complete category and heldout suite without prompt-conditioned
reranking. Artifact:
`benchmarks/results/2026-08-07-qwen36-27b-mtp-cycle-deep-profile.json`.

---

## 7. Prioritized execution plan

### Reopened latest-Vulkan pass

| Priority | ID | Work | Exit gate / impact rule | Status |
| ---: | --- | --- | --- | --- |
| 0 | D27-R0 | Rebuild and freeze latest llama.cpp Vulkan, rerun low-level, stateful AR, natural B3/B4, and budget selection. | Same model/device/protocol; candidate-local warmup; compact raw hashes and rollup. | complete at `c8e03ce81`; B4 selected at 69.798 tok/s |
| 0 | D27-R1 | Reprofile latest Vulkan B3/B4 and current hipEngine B3; reconcile every kernel, queue/host, copy/state, proposal, target, commit, and sampling bucket to wall. | Matched one-prompt trajectories and <=10% residual or explicit overlap/measurement explanation. | complete; aggregate HIP kernels are 31.20 ms ahead, but steady graph/queue/host is 41.35 ms behind |
| 1 | D27-R2 | Close profiler-ranked module deficits sequentially using the exact Vulkan shader/dispatch/generated behavior as source evidence. | Do not advance to the next slower hipEngine module until the current module is >= Vulkan under a matched call/shape normalization and all correctness/state gates pass, or the source-faithful mechanism ladder is explicitly exhausted. | complete; submission residual **41.346 -> 20.467 ms** and parent/child is rejected; latest Q5/wide-Q6 ladders remain exhausted; exact compact K plus sole-resident planar-Q6 V reduce K/V **4.958 -> 3.762 ms**, then source Q6/Q4 K/V routes lose at binding economics and are removed; final Vulkan-source root rows win in isolation but required matching c1 makes MTP root **72.611 -> 74.099 ms (+2.05%)**, so all temporary code is removed and every ranked module is either retained-improved or source-audited exhausted |
| 2 | D27-R3 | Close non-arithmetic/algorithmic residuals, including budget/schedule topology. | Complete natural25 selected hipEngine path >= selected Vulkan B4, without fixed-prompt tuning, or close by measured exhaustion. | complete by exhaustion; clean populated controls beat Vulkan, post-module B3 is **61.147** and retained canonical is **61.394** versus selected Vulkan B4 **69.798 tok/s**; no current mechanism closes the 12.04-12.39% gap |
| 3 | D27-R4 | Publish final controls, artifacts, rollups, refactor cleanup, and defaults. | 512/4096 prefill+AR controls, full category/heldout natural gate, exact state, atomic commits. | complete as a below-parity publication; rejected source routes are removed and no Qwen3.6-specific refactor debt remains |

### Historical first pass

The table below records the now-superseded `ee0445c99` B3 campaign. Its
measurements and rejection decisions remain evidence, but its B3 selection and
"exhausted" status do not govern the reopened latest-source pass.

| Priority | ID | Work | Exit gate / impact rule | Status |
| ---: | --- | --- | --- | --- |
| 0 | D27-M0 | Freeze latest llama.cpp Vulkan revision/build, model hash, hardware/software capture, AR/MTP commands, and unprofiled W7900 baselines. | Fresh pp/tg, context-matched AR, natural25, canonical B1-B3 plus B4/B5 admission audits, and query-profile artifacts. | complete; B3 selected, B4/B5 exact but runtime-rejected and unsupported in production |
| 0 | D27-F0 | Add untied dense root/embedding/layout support, then prove dense GGUF AR load/prefill/decode on GPU0. | Strict map uses Q6_K `output.weight`; finite deterministic 8/1 smoke, then 512/128 and 4K/128 exact/state gates. | complete; clean 512/128 + 4K/128 graph gates green |
| 0 | D27-F1 | Add architecture-shaped dense NextN mapping/materialization with RED tests. | Strict real call-spec accepts 15-tensor `blk.64`; existing MoE fixtures remain unchanged. | complete; real map green |
| 0 | D27-F2 | Run dense NextN one-step and exact/default MTP cycle. | Layer CPU/llama oracle KL <= 0.05, top-1 >= 90%; full state/KV transaction exact. | complete; exact transaction green |
| 0 | D27-M1 | Establish fine-grained llama Vulkan and hipEngine AR/MTP profiles and reconcile wall. | Compact Amdahl tables with <=10% residual or an explicit queue/overlap explanation. | complete; AR + MTP walls reconciled, 10.75% AR graph gap explained |
| 1 | D27-O1 | Optimize the largest measured AR prefill bucket. | Candidate ceiling >=5% complete wall; same-suite exact win at 512 and 4K. | complete; populated route reaches 234.014/215.771 tok/s, 193.23%/163.80% above stateful Vulkan |
| 1 | D27-O2 | Optimize the largest measured AR decode bucket. | Candidate ceiling >=5% or >=0.20 ms/token; same-suite exact win. | complete for this pass; populated graph AR is 23.284/21.903 tok/s and beats stateful Vulkan by 85.18%/75.40% |
| 1 | D27-O3 | Optimize the largest measured MTP cycle bucket (draft, target, commit, or host residual). | Full and heldout absolute MTP improves; own-AR ratio improves or a faster-AR denominator decline is disclosed; no category or acceptance regression. | tabled at exact/default local optimum; canonical B1/B2/B3 is 43.792/55.254/60.262 tok/s, optimized B4/B5 both fail the 50.3-ms/pass admission screen, rounded next-RMSNorm is the physical default, and rejected launch contractions are excluded |
| 2 | D27-L1 | Re-profile and close second-order gaps until Vulkan parity. | Each new target is selected from the refreshed profile, not this initial list. | exhausted for current algorithms; the residual coverage audit closes Q5, wide-Q6 scalar-DP4A/X8-sidecar, root scalar-DP4A, graph upload/splitting, and exact five-/six-row B4/B5, leaving B3 11.49% below Vulkan |
| 3 | D27-P0 | Final clean W7900 publication and default promotion. | Definition of done, rollups, artifacts, refactor cleanup, atomic commits. | complete as a blocked publication; retained defaults and evidence are published, but matched MTP parity is explicitly not claimed |

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
| 2026-08-05 | hipEngine `3728531ba`, byte-neutral planar Q6T16 | W7900 | 512/128, 4K/128; natural25 B1-B3 | 207.022 / 192.528 | 22.107 / 20.926; natural AR 23.031 | B1 39.830, B2 49.848, B3 54.547 | 1.7294x / 2.1644x / 2.3685x | all 750 MTP IDs and acceptance exact; all 30 rows/scopes improve; populated IDs deterministic; byte-identical peaks and clean teardown | `benchmarks/results/2026-08-05-qwen36-27b-q6t16-qmicro-planar-retained.json` |
| 2026-08-05 | hipEngine `094941175`, planar Q6T16 root head | W7900 | 512/128, 4K/128; natural25 B1-B3 | 207.864 / 193.001 | 22.208 / 21.017; natural AR 23.069 | B1 40.179, B2 50.813, B3 55.899 | 1.7417x / 2.2026x / 2.4231x | all IDs/acceptance exact; all aggregate scopes and all B2/B3 prompt rows improve; one B1 prompt -0.653% disclosed; byte-identical peaks and clean teardown | `benchmarks/results/2026-08-05-qwen36-27b-q6t16-qmicro-root-head-retained.json` |
| 2026-08-05 | hipEngine `24fef47da`, sole-resident Q5T16 `ssm_out` | W7900 | 512/128, 4K/128; natural25 B1-B3 | 234.014 / 215.771 | 23.241 / 21.841; natural AR 24.049 | B1 43.170, B2 54.621, B3 59.551 | 1.7951x / 2.2712x / 2.4762x | aggregate Q5 quality 7.38e-5 KL / 99.79% top-1; candidate MTP exact vs own AR; all scopes and all B2/B3 prompts improve; one B1 timing and one fluent prior-route trajectory change disclosed; peaks -1.824 GiB | `benchmarks/results/2026-08-05-qwen36-27b-q5t16-ssm-out-retained.json` |
| 2026-08-05 | hipEngine `a5f25c9ad`, exact GDN BF16 handoff | W7900 | 512/128, 4K/128; natural25 B1-B3 | 234.580 / 215.127 | 23.284 / 21.903; natural AR 23.750 | B1 43.357, B2 54.678, B3 59.790 | 1.8256x / 2.3023x / 2.5175x | all IDs/acceptance/state exact; 336 casts removed; target physical windows improve; marker-wall variance disclosed | `benchmarks/results/2026-08-05-qwen36-27b-gdn-bf16-handoff-retained.json` |
| 2026-08-05 | hipEngine `01291b066`, one-launch rollback snapshots | W7900 | inherited unchanged 512/128, 4K/128; natural25 B1-B3 | 234.580 / 215.127 | 23.284 / 21.903; natural AR 24.114 | B1 43.792, B2 55.254, B3 60.262 | 1.8161x / 2.2914x / 2.4991x | all 30 prompt-budget rows and scopes improve; IDs/acceptance/state exact; 665 launches removed; +1,540 bytes; clean teardown | `benchmarks/results/2026-08-05-qwen36-27b-journal-snapshot-copy-retained.json` |
| 2026-08-06 | llama.cpp Vulkan `c8e03ce81`, latest selected budget | W7900 | stateful 512/128, 4K/128; natural25 matched B3 / selected B4 | 79.351 / 80.622 | 12.535 / 12.459; natural AR 12.528 | B3 67.682, **B4 69.798** | selected B4 5.5714x | tracked-clean build 10290; candidate-local warmup; B4 accepts 171/241; all categories improve | `benchmarks/results/2026-08-06-qwen36-27b-llamacpp-vulkan-c8e03ce81-refresh.json` |
| 2026-08-06 | hipEngine `a3e4912ee` + `09abd51968`, native submission graphs | W7900 | inherited unchanged 512/128, 4K/128; natural25 B1-B3 | 234.580 / 215.127 | 23.284 / 21.903; canonical-packet natural AR 24.085 | B1 44.035, B2 56.014, B3 61.020 | 1.8283x / 2.3257x / 2.5336x | proposal graph: all 30 rows/scopes exact and improved; N2 target policy retained physically with mixed immediate 44.224/56.037/60.903 packet; clean teardown | `benchmarks/results/2026-08-06-qwen36-27b-native-submission-graphs-retained.json` |
| 2026-08-06 | hipEngine `92e823a2d`, direct proposal-to-target retirement | W7900 | inherited unchanged 512/128, 4K/128; natural25 B1-B3 | 234.580 / 215.127 | 23.284 / 21.903; natural AR 23.524 | **B1 44.496, B2 56.350, B3 61.394** | 1.8915x / 2.3954x / 2.6098x | exact event/device handoff; B2/B3 10/10 prompts and 7/7 scopes improve vs N2; B1 9/10 and 7/7; 48 bytes; clean teardown | `benchmarks/results/2026-08-06-qwen36-27b-direct-proposal-target-handoff-retained.json` |
| 2026-08-07 | hipEngine `d1f26fb5c`, latest-source exhaustion gate | W7900 | 512/128, 4K/128; natural25 B1-B3 | **235.434 / 216.784** | **23.296 / 21.897**; natural AR **22.926** | B1 **44.730**, B2 **56.194**, B3 **61.147 current / 61.394 canonical** | 1.9511x / 2.4511x / 2.6671x current | populated controls clean/deterministic/finite; natural IDs/acceptance/stages exact; every ranked module retained-improved or source-audited exhausted; parity not claimed | `benchmarks/results/2026-08-07-qwen36-27b-latest-vulkan-parity-exhaustion-audit.json` |

Update this table only with retained or explicitly labeled blocked/diagnostic
rows. Detailed iteration history belongs in `WORKLOG.md`; benchmark toplines
also update `benchmarks/README.md`, `benchmarks/CHANGELOG.md`, and compact JSON
artifacts when measured.
