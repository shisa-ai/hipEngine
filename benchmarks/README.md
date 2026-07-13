# hipEngine Topline Benchmarks

Last reviewed: **2026-07-13**

Latest measured hipEngine revision in this scoreboard:
`2670ed0434f6a396b901fbe7e5fd04b93dd14afe`

This file is the source of truth for repository-level performance tables. It
records which snapshots are eligible for use, the exact protocol behind each
table, the measured source revision and build environment, and the command used
to refresh it. [`README.md`](../README.md) contains copies of the marked export
blocks below; update them with:

```bash
python3 scripts/sync_benchmark_readme.py --write
python3 scripts/sync_benchmark_readme.py --check
```

Machine-readable evidence is under [`benchmarks/results/`](results/). Promotion
requirements are defined in [`docs/BENCHMARK.md`](../docs/BENCHMARK.md).
Reverse-chronological changes are in [`benchmarks/CHANGELOG.md`](CHANGELOG.md).
The previous experiment notebook is preserved in
[`benchmarks/HISTORY.md`](HISTORY.md).

## Status Rules

| Status | Meaning | May appear as a repository topline? |
| --- | --- | --- |
| **Retained** | The artifact passes the protocol's correctness, provenance, and performance gates. | Yes, for the named protocol only. |
| **Diagnostic** | The run is useful but has a known comparability, correctness, repetition, or provenance limitation. | No. Link it from the separate diagnostic section; do not place its numbers in a current table. |
| **Stale** | A measured path, dependency, or required evidence contract changed after the run. | No. It may remain as the last dated snapshot while a refresh is pending. |
| **Blocked** | No row satisfies the protocol. | No numeric topline. Record the blocker and the next command. |

`Latest` means the newest artifact for one exact protocol tuple. A newer
diagnostic does not replace a retained row. A row is identified by:

```text
platform + GPU + model fingerprint + quant + KV type + backend +
workload + concurrency + sampling/speculative policy + timing scope
```

Documentation-only commits do not make a row stale. Changes to a measured
runtime path, model, quant, KV policy, compiler/runtime, benchmark timing scope,
correctness gate, or comparison engine do.

New server, retained PARO, GGUF, and micro artifacts must embed a valid
`hipengine_artifact_provenance` v1 block. The canonical schema is
[`schemas/artifact-provenance.schema.json`](schemas/artifact-provenance.schema.json).
For retained model-performance rows, the resolved backend must be concrete,
the selected target/device must be recorded, the model fingerprint must refer
to existing content, and staged/unstaged/untracked dirtiness must all be false.
Legacy provenance fields remain useful diagnostics but do not satisfy this
contract for a new row.

New non-streaming hipEngine server rows also require a complete
`hipengine.generation_shape` v1 rollup. Route caps retain their
`queue_requests` scope; queue request/prompt counts, actual backend calls and
widths, and verifier rows remain separate and are deduplicated by queue-group
ID. Client concurrency is never substituted for backend or verifier width.

Direct/server comparisons additionally require the
`hipengine_exact_token_oracle` v1 gate from
[`scripts/exact_token_generation.py`](../scripts/exact_token_generation.py).
The committed 512-ID fixture feeds both PARO/GGUF direct generation and
`/v1/completions` without detokenization. HTTP input hashes/counts, exact usage,
and every generated ID must match the direct oracle. The formal contract is
[`schemas/exact-token-oracle.schema.json`](schemas/exact-token-oracle.schema.json).
The 2026-07-11 gfx1151 PARO 512/128 correctness gate passed; it is not a
throughput row and changes no topline.

Unified direct/server reports use `hipengine_benchmark_matrix` v1 from
[`scripts/benchmark_matrix.py`](../scripts/benchmark_matrix.py). The matrix
recomputes exact-ID denominators, enforces timing ownership, preserves backend
and verifier shapes, and attaches memory/profiler summaries. Its schemas are
[`benchmark-matrix.schema.json`](schemas/benchmark-matrix.schema.json) and
[`benchmark-matrix-manifest.schema.json`](schemas/benchmark-matrix-manifest.schema.json).
The committed SOL-E5 PARO manifest is diagnostic: direct-call wall includes
model/session setup while HTTP is client-E2E, so the report intentionally emits
no direct/server speed ratio. A retained matrix requires the normal clean,
repeated, scoped-timing, memory, profiler, correctness, and shape gates.

The accepted gfx1151 GGUF eager correctness gate is
[`2026-07-11-sol-g1-gfx1151-gguf-eager-p512-d4.json`](results/2026-07-11-sol-g1-gfx1151-gguf-eager-p512-d4.json).
For the exact Q4_K_M file and `[9707] * 512` prompt, llama.cpp and hipEngine's
bulk-prefill/eager route both emit five `9707` IDs. Four teacher-forced eager
transitions are byte-exact against fresh serial-prefix recomputation for all 40
layer outputs, 30 Conv/GDN state pairs, and 10 live K/V layer pairs. This
classifies the repeated stream as valid model behavior on gfx1151; it is a
correctness artifact with `performance_claim=false`, not a throughput row.

The accepted SOL-G2 fused/chain prefill gate is
[`2026-07-11-sol-g2-gfx1151-gdn-prefill-exact-matrix.json`](results/2026-07-11-sol-g2-gfx1151-gdn-prefill-exact-matrix.json).
At committed revision `332f01f8`, the GGUF-only raw-Q/K-plus-scale split chain
matches fused production prefill in all 6/6 clean gfx1151 cases: the exact
17-token greeting, repeated-token 512, the 1024/1025 segment threshold, and the
4095/4096 four-chunk boundary. Sampled tokens, FP32 hidden seeds, and all 30
resident Conv/GDN state pairs are byte-exact; greeting and 512 also match every
captured layer output. The earlier
[`104fad87` prefix artifact](results/2026-07-11-sol-g2-gfx1151-gdn-prefill-greeting-prefix.json)
preserves the normalized-Q/K layer-0 recurrent RED. Both artifacts set
`performance_claim=false`; the repeated, interleaved G3 result below selects
the default.

That G3 protocol is now complete in
[`2026-07-11-sol-g3-gfx1151-gdn-prefill-interleaved-ab.json`](results/2026-07-11-sol-g3-gfx1151-gdn-prefill-interleaved-ab.json).
From a clean detached `ad773eba` worktree with one warmup and four balanced
same-session repetitions per mode/context, the exact chain is slower than fused:
`1248.436` versus `1186.842 ms` at 512 (**+5.19% wall**) and `10870.022` versus
`10187.300 ms` at 4096 (**+6.70% wall**). Every timed pair returns exact token
`9707`, and the artifact links the accepted state matrix by SHA-256. This is a
valid retained negative result (`performance_claim=true`): fused remains the
default, and the exact split remains a diagnostic/unfused fallback.

The follow-on GPF-2B candidate performance gate is retained in
[`2026-07-13-gfx1151-gguf-prefill-gpf2-balanced-ab.json`](results/2026-07-13-gfx1151-gguf-prefill-gpf2-balanced-ab.json).
At clean detached `31d4204d` on TheRock HIP 7.15 and TuneD
`accelerator-performance`, one warmup plus four balanced same-session
repetitions move 512 prefill **1212.462 -> 535.136 ms** (**422.281 -> 956.765
tok/s, 2.266x**) and 4096 prefill **9977.239 -> 4848.216 ms** (**410.534 ->
844.847 tok/s, 2.058x**). All 16 timed final IDs are `9707`; the linked
six-case project gate has KL at most `5.39e-5` and 100% top-1. Because the
candidate changes recurrent-state bits, this is a retained candidate
performance result rather than a default/topline replacement. The public GGUF
column remains the fused route until multi-prompt generated-trajectory/decode
and explicit numerical-contract gates pass.

That natural-prompt gate subsequently rejects default promotion in
[`2026-07-13-gfx1151-gguf-prefill-gpf2-trajectory-rejection.json`](results/2026-07-13-gfx1151-gguf-prefill-gpf2-trajectory-rejection.json).
At clean `2670ed04`, all ten prompts and four categories run a fused/candidate
prefill sample plus 24 logit-checked transitions and two balanced 128-step
graph windows per mode. Only **7/10** prompts keep the first 25 samples and
only **3/10** keep the complete 129-token trajectory; first divergence ranges
from transition 4 to 126. The diagnostic execution wall is flat (**53.316 vs
53.324 tok/s**), but seven timing legs execute different outputs, so that
number is not a retained decode comparison. The numerical-contract decision
keeps the predeclared exact natural trajectory requirement; `auto` remains
fused and the tree is an explicit rejected diagnostic.

SOL-G4 is accepted on gfx1151 in
[`2026-07-11-sol-g4-gfx1151-gguf-eager-decode-audit.json`](results/2026-07-11-sol-g4-gfx1151-gguf-eager-decode-audit.json).
At clean detached `5f4c6561`, the exact repacked/GEMV eager route measures
**49.285 tok/s** (`20.290 ms/token`) for `[9707] * 512` plus 128 timed decode
steps, using one discarded and four measured full runs; every recorded token is
9707 and the artifact links the G1 state oracle by SHA-256. The same synchronized
p8/d32 protocol localizes the first eager performance change to direct-parent
commit `4499fb13`: **17.799 -> 54.963 tok/s** (**+208.79%, 3.088x**) from loaded
HIP-library memoization. Current p8 remains **55.208 tok/s** (+0.45%). A
24-step marker-only profile records **18.402 ms GPU kernels/token** versus
**20.766 ms profiled host wall/token** (88.62%); raw trace CSVs remain under
`/tmp`, while their hashes and the full family Amdahl table are retained.

The current TheRock HIP 7.15 / TuneD refresh promotes explicit wave/block
indexing for the BF16 Q8T16 dual-split leaf at clean detached `e20cdc13`.
Against clean scalar parent `8184355c`, a control/candidate/control p512/d128
eager A/B moves **20.5342 -> 20.4709 ms/token** (**-0.308%**, **48.699 ->
48.850 tok/s**) with non-overlapping ranges and every token exact. Matching
24-step profiles move the named leaf **4245.4 -> 4188.2 us/token** (**-1.349%**)
and total marked GPU time **19256.1 -> 19199.2 us/token** (**-0.296%**). The
state-bound graph path also improves **20.5736 -> 20.5324 ms/token** across
commits, but current G5 runs on both commits find graph slightly slower than
same-run eager; this refresh makes no new graph-over-eager speed claim.

The historical HIP 7.13 SOL-G5 result is accepted on gfx1151 in
[`2026-07-11-sol-g5-gfx1151-gguf-decode-graph-production-audit.json`](results/2026-07-11-sol-g5-gfx1151-gguf-decode-graph-production-audit.json).
At clean detached `7f611fe3`, the production state-bound graph matches eager
byte-for-byte for all 128 launches across generated tokens, the FP32 hidden
seed, 30 Conv/GDN state pairs, and 10 live BF16 K/V pairs. One warmup and four
rotating same-session repetitions measure capture-inclusive graph wall at
**20.311 ms/token** (**49.233 tok/s**) versus same-run eager at
**20.334 ms/token** (**49.178 tok/s**), a **+0.112%** throughput improvement.
The one capture/instantiate and final destroy are charged to every 128-token
window. Per-token recapture is rejected at **35.429 ms/token**. That result
introduced the graph default only for non-streaming c1 greedy gfx1151 windows
with at least 128 remaining transitions. The current HIP 7.15 refresh above
supersedes its speed-policy conclusion: graph replay remains exact but is
slightly slower than same-run eager on both clean commits. The rollback remains
available and a scoped default-policy follow-up is required.

SOL-G6 is accepted on gfx1151 in
[`2026-07-11-sol-g6-gfx1151-gguf-residency-audit.json`](results/2026-07-11-sol-g6-gfx1151-gguf-residency-audit.json).
At clean detached `d70c9464`, the Q4_K_M p512/d128 BF16-KV production graph
session owns **21.478 GiB**, leaving **2.522 GiB** to the explicit 24 GiB gate.
Its 733 planned source tensors have zero raw+replacement duplicates and zero
enabled optional replacement sidecars: **20.461 GiB** is replacement layout,
**0.503 GiB** is the required raw token embedding, and **0.097 GiB** is dense
metadata. Decode scratch is **0.080 GiB** (including **15 MiB** KV and
**63.75 MiB** linear state), while session/prefill buffers are **0.337 GiB**.
Production `record_steps=0` graph capture adds no tracked buffer and a measured
**308 KiB** HIP graph/exec delta. G5 remains the cryptographically linked exact
and performance non-regression gate; this G6 artifact makes no new speed claim.

SOL-P2 is accepted on gfx1151 in
[`2026-07-11-sol-p2-gfx1151-paro-ragged-lifecycle.json`](results/2026-07-11-sol-p2-gfx1151-paro-ragged-lifecycle.json).
At clean detached `6f1910c9`, exact prompt lengths 449 through 512 shrink from
c8 to c1 without compaction. One row retires through EOS; explicit
cancellations create middle, tail, then front holes while every post-event
width remains exact. All eight generated sequences, all 30 linear recurrent
state families, and all 10 live full-attention K/V families match independent
c1. Ragged prefill uses the explicitly labelled `per_segment_ragged_exact`
fallback; this is a correctness artifact with `performance_claim=false`.

## Platform Index

| Platform | Benchmark family | Run date | Measured revision / build | Evidence status | Root README | Refresh condition |
| --- | --- | --- | --- | --- | --- | --- |
| Radeon Pro W7900, gfx1100 | PARO BF16/INT8 KV context capacity | 2026-07-12 | clean measured hipEngine `8116c453`, rebased-equivalent reachable `8708304f` (runtime/benchmark code identical); TheRock HIP `7.15.0-0000000`; current Qwen3.6 packed model fingerprint retained | **Current capacity / correctness rejection**: 128K BF16 is **22.124 GiB** tracked; 256K INT8 is **23.957 GiB** with no BF16 shadow. The required Qwen3.6 128K/128 rollout rejects both FP16- and FP32-scale INT8, so 256K is allocation evidence only, not a usable route. | Current diagnostic table | Rerun after INT8 KV write/decode math, scale policy, attention accumulation, or model changes; require the Qwen3.6 long-rollout gate before support. |
| Radeon Pro W7900, gfx1100 | Qwen3.6 35B model sweep | 2026-07-12 | clean measured hipEngine `8116c453`, rebased-equivalent reachable `8708304f` (runtime/benchmark code identical); TheRock HIP `7.15.0-0000000`; llama.cpp HIP `1ebf790cd` build 9648; Vulkan `263cc04a5` build 9600 | **Accepted four-column topline**: all six shapes pass W7900-local state/token correctness, clean provenance, finite/stable IDs, exact Q4_K_M identity, five-sample variance, and corrected whole-device VRAM scope. | Yes | Rerun after PARO/GGUF measured paths, graph policy, model, compiler/runtime, llama.cpp builds, or W7900 clock policy changes. |
| Radeon Pro W7900, gfx1100 | PARO gfx1151 optimization transfer gate | 2026-07-12 | clean detached hipEngine `255e5aca`; TheRock HIP `7.15.0-0000000`; exact PARO model fingerprint retained | **Retained scoped-default validation / negative chunk decision**: the balanced global-isolation screen is exact at 512/1K/4K. Its 4K/4096-query leg directly validates the merged scoped default with total wall **-0.562%**; 512/1K used 256-query isolation that the final policy intentionally excludes. The gfx1151 linear/MoE-256 profile is rejected at **-7.72%/-8.78%/-6.40% prefill**. | Linked, not a new topline | Rerun after AOTriton/ROCr stream scheduling, PARO chunks, compiler/runtime, or gfx1100 clock policy changes. |
| Radeon Pro W7900, gfx1100 | GGUF graph AR, exact/default MTP, `llama-compat`, and llama.cpp HIP | 2026-07-12 | clean graph gate `833921ce`, admitted route `ac0adb3f`, clean suites `202bd2f0`; ROCm 7.2.4; exact Q4_K_M/prompt fingerprints; llama.cpp HIP `1ebf790cd` build 9648 | **Current retained AR / corrected MTP economics**: natural24 graph AR is **93.30 tok/s**, exact B3 is **68.50 vs 98.75 AR (0.6936x)**, and accuracy-traded `llama-compat` is **79.70 vs 93.30 AR (0.8542x)**. All 24 repeated-state transitions and all ten natural generated previews/tails are exact. At matched timing boundaries hipEngine AR is **93.30** versus llama.cpp **78.29 tok/s (+19.19%)**. | Yes, qualified | Rerun after graph policy/state, GGUF/MTP route, model/prompt suite, compiler/runtime, or output-horizon changes; keep exact fixed-cycle and natural24 contracts separate. |
| Radeon Pro W7900, gfx1100 | PARO/llama.cpp/vLLM concurrency | 2026-07-07 | hipEngine `b4edca09`; same TheRock stack; vLLM `0.22.1rc1.dev499+g470229c37.d20260613` | **Stale diagnostic**: cross-quant and mixed timing scopes; source artifacts set `performance_claim=false`; measured PARO code predates the July concurrency changes | Diagnostic link only | Rerun one timing scope with exact generated-token accounting across all engines |
| Radeon Pro W7900, gfx1100 | Dense 27B DFlash | 2026-06-11 | hipEngine `9faa731c`; ROCm 7.2; artifact records a dirty tree | **Retained under the recorded DFlash gate**, with legacy dirty-source provenance | Yes, qualified | Refresh on a clean tree before changing the public claim |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Qwen3.6 35B matched four-engine reference | 2026-07-11 | clean hipEngine `d1231ee0`; TheRock HIP `7.13.60980-c76140fa27`; llama.cpp HIP `1ebf790cd` build 9648; Vulkan `6e9007ae6` build 9641 | **Retained reference**: all six shapes pass clean provenance, finite/stable-ID correctness, five-sample variance, matched Q4_K_M identity, and the four-column promotion gate. The current table replaces only its PARO column with the separately retained recovery below. | Yes | Rerun GGUF/llama columns after a measured path/build/stack change; rerun all four together when a fully matched refresh is required. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | PARO exact c1 prefill recovery | 2026-07-12 | clean control `240c5daf` and candidate `9944e481`; TheRock HIP `7.15.0-0000000`; TuneD accelerator-performance; exact PARO model fingerprint retained | **Retained**: exact linear/MoE 256-row architecture profile improves all six prefill shapes by **14.35%-51.11%**, leaves decode within **-0.25%..+0.26%**, and matches final hidden plus all Conv/GDN/KV state at 512/4K/128K. | Yes, PARO column | Rerun after PARO prefill chunk/staging/math, compiler, model, prompt, or tuned/clock policy changes; validate separately on gfx1100 before transfer. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | PARO 4K-128K AOTriton queue isolation | 2026-07-12 | clean same-commit control/candidate `01e2cec5`; TheRock HIP `7.15.0-0000000`; TuneD accelerator-performance; exact PARO model fingerprint retained | **Retained at 4K/32K/64K/128K**: event-linked isolated AOTriton queue improves matched prefill by **13.32%-23.03%**, leaves decode within **-0.16%..+0.12%**, holds tracked peak unchanged, and matches final hidden plus all 30 Conv/GDN and 10 K/V families at every retained shape. The 1K 256-query negative control does not enter isolation and is unchanged. | Yes, PARO column | Validate separately on gfx1100 before transfer; 512/1K remain on the proven-safe caller-stream route. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF eager token/state oracle | 2026-07-12 | clean detached hipEngine `3ce60e56`; TheRock HIP `7.15.0-0000000`; exact Q4_K_M fingerprint and llama binary hashes retained | **Accepted correctness-only gate**: the repeated external and production token stream matches; four hidden/layer/30-Conv-GDN/10-KV transitions are finite and byte-exact. `performance_claim=false`. | Diagnostic link only | Rerun after eager math/state/KV, model, compiler/runtime, or device changes. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF fused/chain GDN prefill correctness and default selection | 2026-07-11 | correctness at clean tracked `332f01f8`; clean performance worktree `ad773eba`; TheRock HIP `7.13.60980-c76140fa27`; exact Q4_K_M fingerprint retained | **Accepted correctness / retained negative performance decision**: exact chain passes 6/6 state cases but is +5.19%/+6.70% slower in balanced 512/4K walls. Fused remains default. | Diagnostic link only | Rerun after GDN math/scheduler/chunk changes; do not retry unchanged split scheduling. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF GPF-2B register-resident GDN candidate | 2026-07-13 | clean performance `31d4204d`, clean trajectory gate `2670ed04`; TheRock HIP `7.15.0-0000000`; TuneD accelerator-performance; exact Q4_K_M fingerprint retained | **Retained performance signal / correctness-rejected default**: balanced 512/4K prefill improves **422.281 -> 956.765 tok/s (2.266x)** and **410.534 -> 844.847 tok/s (2.058x)**, but only **3/10** natural prompts preserve the complete fused 128-step trajectory. `auto` remains fused. | Diagnostic link only | Implement the register-resident exact ordered-wave GPF-2C candidate; require exact natural trajectories before any default/topline change. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF correct eager baseline, revision bisect, and decode-only Amdahl | 2026-07-11 | clean detached hipEngine `5f4c6561`; TheRock HIP `7.13.60980-c76140fa27`; exact Q4_K_M fingerprint retained | **Retained**: p512/d128 exact eager is 49.285 tok/s; `4499fb13` is the direct-parent 3.088x speed boundary; 24 exact marker windows isolate the current family profile. | Yes, named repeated-token protocol | Rerun after eager decode math, route, dispatch/build caching, or a material family-kernel change; run separately on W7900. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF Q8T16 dual-split wave/block indexing | 2026-07-12 | clean scalar `8184355c` and promoted `e20cdc13`; TheRock HIP `7.15.0-0000000`; TuneD accelerator-performance; exact Q4_K_M fingerprint retained | **Retained**: clean p512/d128 eager **20.5342 -> 20.4709 ms/token** (-0.308%); marked dual-split leaf **4245.4 -> 4188.2 us/token** (-1.349%); graph route **20.5736 -> 20.5324 ms/token** (-0.200%); every token/state gate exact. | Yes, named repeated-token protocol | Rerun after Q8T16 indexing/layout, compiler, graph policy, or gfx1151 launch geometry changes; validate separately on gfx1100 before transfer. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF state-bound production decode graph | 2026-07-11 historical; 2026-07-12 refresh | clean detached hipEngine `7f611fe3` on HIP 7.13; clean `8184355c`/`e20cdc13` on HIP 7.15; exact Q4_K_M fingerprint retained | **Historical retained / current speed-policy stale**: all 128 graph launches remain byte-exact. HIP 7.13 measured +0.112% over eager; both current HIP 7.15 reruns reject at -0.246%/-0.293%. | Current table reports exact diagnostic wall, not a graph-over-eager win | Run a scoped balanced current-stack A/B; restore eager default if graph does not reproduce a win. Validate separately on gfx1100 before any admission. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF replacement-layout residency and 24 GiB-class gate | 2026-07-11 | clean detached hipEngine `d70c9464`; TheRock HIP `7.13.60980-c76140fa27`; exact Q4_K_M fingerprint retained | **Retained memory/correctness gate**: 733 unique sources, no raw+replacement duplicates or optional sidecars, 21.478 GiB owned/tracked p512/d128 graph session, 2.522 GiB budget margin. `performance_claim=false`; G5 supplies linked speed non-regression. | Diagnostic link only | Rerun after weight materialization/layout, KV/state, prefill scratch, graph allocation, or max-sequence policy changes; context-specific capacity remains separate. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | PARO exact c1-c8 shape/routing catalog | 2026-07-11 | clean detached hipEngine `a18ff7bc`; TheRock HIP `7.13.60980-c76140fa27`; exact model and prompt fingerprints retained | **Retained c1 performance and routing correctness**: exact-fixture c1 graph is 66.910 tok/s median; clean c2-c8 native rows all fail independent-c1 equality at index 2 and are explicitly serial. Native rates are diagnostic only. | Yes for c1 exact-fixture graph only | Rerun after c1 graph/prefill changes or any general native c>N algorithm change. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | PARO ragged c8-to-c1 lifecycle correctness | 2026-07-11 | clean detached hipEngine `6f1910c9`; TheRock HIP `7.13.60980-c76140fa27`; same exact model/fixture fingerprints as P1 | **Accepted correctness-only gate**: eight token sequences, 30 linear-state families, and 10 full-KV families match c1 through EOS and front/middle/tail sparse cancellation. `performance_claim=false`; ragged prefill uses an exact per-segment fallback. | Diagnostic link only | Rerun after ragged prefill, scheduler retirement, slot/state/KV addressing, or true-c1 decode changes; run independently on W7900. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | PARO/llama.cpp concurrency | 2026-06-15 | measured hipEngine revision not recorded in summary; gfx1151 forced through `HIPENGINE_HIP_ARCH` | **Stale diagnostic**: `performance_claim=false`, mixed quant, and incomplete backend provenance | Diagnostic link only | Rerun c=1..8 plus shrinking batches at one clean revision with detected arch and all-choice token counts |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF MTP exact/default, `llama-compat`, and llama.cpp HIP refresh | 2026-07-12 | clean detached hipEngine `3ce60e56`; TheRock HIP `7.15.0-0000000`; TuneD `accelerator-performance`; exact Q4_K_M/prompt fingerprints; llama.cpp HIP `1ebf790cd` build 9648 plus retained patchset | **Current**: exact B5 is a negative **51.81 vs 54.14 AR tok/s (0.9571x)**; explicit accuracy-traded `llama-compat` is **69.50 vs 54.40 AR tok/s (1.2776x)** and wins every category/heldout. At matched decode boundaries it is **69.38 tok/s** versus transition-normalized llama.cpp **66.66 tok/s**. The current clean state oracle passes. | Yes, qualified | Rerun after route/verifier lifecycle, model/prompt suite, compiler/runtime, clock policy, output horizon, or timing-boundary changes. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Historical GGUF MTP pre-correctness-pass rows | 2026-07-02–03 | hipEngine exact `44c4d3d4`, `llama-compat` `ca571bf6`; environment provenance incomplete | **Superseded history**: exact 61.98 and compatibility 71.52 tok/s remain useful deltas, but no longer define the current table. | Historical links only | Do not promote without the current state lifecycle, clean provenance, and transition-matched timing contract. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF OpenAI server automatic-route gate | 2026-07-11 | tracked-clean hipEngine `d2b1e742`; TheRock HIP `7.13.60980-c76140fa27`; exact GGUF and prompt-suite fingerprints retained; unrelated untracked files disclosed | **Diagnostic correctness rejection**: compatibility MTP is faster at c1/c2 but changes true-AR IDs on heldouts, so it cannot select automatic routing. One c8 AR repetition also exposes the separate exact-concurrency blocker. | Diagnostic link only | Implement an exact/default server MTP hook, then rerun full plus category-heldout realized-group economics before admitting it to `auto`. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | HIP versus Vulkan timing-contract v2 micro matrix | 2026-07-12 | clean detached hipEngine `50bea8f3`, TheRock ROCm `7.15.0a20260711`, kernel `7.1.3-2-cachyos`, RADV/Mesa `26.1.4`, corrected gfx1151 device wheels | **Matched and strict matrices retained**: each passes 22/22 comparisons and 232 burst GPU rows after portable q8_1 RNE/ties-away rounding eliminates the systematic scale mismatch | Linked, not copied here | Run the portable shader's strict gate on gfx1100 when W7900 access returns; otherwise rerun after a timed kernel/harness, ROCm, Mesa, or clock-policy change |
| Radeon Pro W7900, gfx1100 | HIP versus Vulkan timing-contract v2 micro matrix | 2026-07-11 | clean hipEngine `c57f21b5`; TheRock ROCm `7.15.0a20260711`; RADV/Mesa `26.1.4` | **Retained**, 22/22 comparisons and 232 burst GPU rows pass provenance, correctness, exact-matrix, device, and clock gates | Linked, not copied here | Rerun after a timed kernel/harness, ROCm, Mesa, or device clock-policy change |

## Current Eligible Toplines

Only rows with an eligible evidence status appear in this section. The sync
script copies this marked block into the root README byte-for-byte.

### gfx1100 model throughput

This is the clean 2026-07-12 W7900 refresh measured at hipEngine `8116c453`
(rebased-equivalent reachable `8708304f`; only `WORKLOG.md` and
`docs/PROCESS-IMPROVEMENT.md` differ) on TheRock HIP 7.15. Each hipEngine shape uses its own right-sized resident
session, two discarded warmups, and five measured repetitions; the tables
report medians. PARO and GGUF both use their admitted state-bound graph decode
routes, with capture excluded from steady decode timing. llama.cpp uses one
internal warmup plus five samples per split prefill/decode phase. The W7900
four-step GGUF oracle passes external tokens plus byte-exact hidden, all 30
Conv/GDN state families, and all 10 live K/V families; every measured sweep
row has finite logits, stable final IDs, clean provenance, and <=5% sample
variance.

Bold marks the best raw value in each row. It is descriptive only: PARO is W4
PARO/BF16 KV, while the other columns use the same Q4_K_M GGUF with hipEngine
BF16 KV and llama.cpp F16 KV. Memory scopes also differ.

<!-- BEGIN TOPLINE:W7900_SWEEP -->
#### Prefill tok/s

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **2917.732** | 644.719 | 2412.320 | 2627.990 |
| 1K/128 | **2995.876** | 676.177 | 2389.670 | 2631.750 |
| 4K/128 | **2943.038** | 677.618 | 2255.080 | 2521.770 |
| 32K/128 | **2108.868** | 628.364 | 1667.640 | 1943.920 |
| 64K/128 | **1584.131** | 572.612 | 1291.820 | 1414.470 |
| 128K/128 | 1056.252 | 484.212 | 891.949 | **1079.280** |

#### Decode tok/s

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **115.599** | 89.873 | 80.756 | 107.786 |
| 1K/128 | 103.238 | 94.751 | 80.805 | **107.555** |
| 4K/128 | **105.943** | 96.551 | 79.768 | 103.066 |
| 32K/128 | **92.438** | 83.673 | 74.304 | 91.835 |
| 64K/128 | 78.260 | 71.644 | 69.010 | **83.746** |
| 128K/128 | 60.663 | 56.745 | 60.933 | **70.833** |

#### Peak memory GiB

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **18.144** | 21.478 | 21.606 | 21.260 |
| 1K/128 | **18.367** | 21.710 | 21.618 | 21.220 |
| 4K/128 | **19.161** | 22.995 | 21.674 | 21.278 |
| 32K/128 | **19.864** | 23.559 | 22.216 | 21.855 |
| 64K/128 | **20.403** | 24.203 | 22.895 | 22.512 |
| 128K/128 | **22.124** | 25.493 | 24.089 | 23.824 |
<!-- END TOPLINE:W7900_SWEEP -->

hipEngine memory is its tracked allocator high-water; llama.cpp is absolute
whole-device W7900 VRAM used, sampled from DRM sysfs `card1` every 10 ms. The
host's `rocm-smi` card labels use a different numbering scheme; the retained
artifact validates the 48 GiB W7900 device rather than the idle 24 GiB XTX.
Use memory values for within-column context growth, not small cross-column
allocator-efficiency claims.

Artifacts: [accepted summary](results/2026-07-12-w7900-v030-8116c453-summary.json),
[hipEngine PARO](results/2026-07-12-w7900-v030-8116c453-hipengine-paro-packed-5run.json),
[hipEngine GGUF](results/2026-07-12-w7900-v030-8116c453-hipengine-gguf-q4km-5run.json),
[llama.cpp HIP](results/2026-07-12-w7900-v030-8116c453-llamacpp-hip-q4km-f16kv.json),
[llama.cpp Vulkan](results/2026-07-12-w7900-v030-8116c453-llamacpp-vulkan-q4km-f16kv.json),
and [W7900 GGUF oracle](results/2026-07-12-w7900-v030-gguf-eager-p512-d4.json).

### gfx1151 model throughput

The GGUF and llama.cpp columns are the clean 2026-07-11 matched refresh at
hipEngine `d1231ee0`. PARO 512/1K are the clean six-shape exact recovery at
`9944e481`; 4K and 32K-128K are the clean scoped AOTriton queue-isolation
refresh at `01e2cec5`, all on TheRock HIP 7.15 and TuneD
`accelerator-performance`. Each hipEngine shape uses its own right-sized
resident session, two discarded warmups, and five measured repetitions; the
tables report medians. The 1K follow-up shared a max-32K session and was a
structural negative control: both settings use the same 256-query route, so it
does not replace the existing right-sized 1K row.
llama.cpp uses one internal warmup plus five samples per split prefill/decode
phase. The linked artifacts check sample variance, model/build/device identity,
clean provenance, and path-specific correctness before setting
`performance_claim=true`. Because the PARO quant/path differs from the other
three columns, this table is a throughput rollup, not a same-math four-engine
A/B.

Bold marks the best raw value in each row (highest throughput or lowest
reported peak memory). It is descriptive only: PARO uses W4 PARO rather than
Q4_K_M, and the memory columns do not share one allocator scope.

<!-- BEGIN TOPLINE:GFX1151_SWEEP -->
#### Prefill tok/s

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **1140.101** | 430.767 | 1061.260 | 1067.770 |
| 1K/128 | **1208.343** | 437.467 | 1043.230 | 1069.870 |
| 4K/128 | **1089.031** | 403.946 | 1009.240 | 1016.580 |
| 32K/128 | **906.145** | 369.942 | 743.547 | 814.923 |
| 64K/128 | **716.775** | 334.395 | 573.611 | 660.974 |
| 128K/128 | 474.641 | 270.601 | 390.441 | **476.788** |

#### Decode tok/s

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **66.767** | 49.536 | 50.939 | 62.396 |
| 1K/128 | 61.746 | 52.192 | 50.818 | **62.136** |
| 4K/128 | **62.715** | 52.999 | 50.126 | 60.097 |
| 32K/128 | 50.342 | 43.947 | 44.240 | **51.319** |
| 64K/128 | 42.094 | 37.477 | 39.326 | **44.422** |
| 128K/128 | 30.386 | 27.862 | 32.114 | **34.948** |

#### Peak memory GiB

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **18.039** | 21.478 | 21.375 | 21.551 |
| 1K/128 | **18.051** | 21.710 | 21.387 | 21.501 |
| 4K/128 | **19.026** | 22.995 | 21.444 | 21.507 |
| 32K/128 | **19.729** | 23.559 | 21.987 | 22.191 |
| 64K/128 | **20.403** | 24.203 | 22.666 | 22.627 |
| 128K/128 | **22.124** | 25.493 | 23.862 | 24.254 |
<!-- END TOPLINE:GFX1151_SWEEP -->

The PARO column is W4 PARO/BF16 KV. The other three columns use the same
Q4_K_M GGUF; hipEngine uses BF16 KV and llama.cpp uses f16 KV. Peak-memory
scopes differ: hipEngine reports its tracked allocator high-water, while
llama.cpp reports absolute whole-device amdgpu GTT used, sampled every 10 ms.
Use the memory table for within-column context growth; small cross-column
deltas are not allocator-efficiency claims. hipEngine load and graph capture
are excluded from phase throughput, while the separate SOL-G5 row below is the
capture-inclusive GGUF graph proof.

Artifacts: [PARO exact prefill recovery](results/2026-07-12-gfx1151-paro-prefill-recovery.json),
[PARO 4K-128K AOTriton queue isolation](results/2026-07-12-gfx1151-paro-aotriton-stream-isolation.json),
[accepted July 11 matched summary](results/2026-07-11-gfx1151-readme-refresh-20260711-d1231ee0-summary.json),
[hipEngine PARO](results/2026-07-11-gfx1151-readme-refresh-20260711-d1231ee0-hipengine-paro-packed-5run.json),
[hipEngine GGUF](results/2026-07-11-gfx1151-readme-refresh-20260711-d1231ee0-hipengine-gguf-q4km-5run.json),
[llama.cpp HIP](results/2026-07-11-gfx1151-readme-refresh-20260711-d1231ee0-llamacpp-hip-q4km-f16kv.json),
and [llama.cpp Vulkan](results/2026-07-11-gfx1151-readme-refresh-20260711-d1231ee0-llamacpp-vulkan-q4km-f16kv.json).

### Speculative decode

The public table includes only contracts with a true same-protocol AR control.
The exact/default and `llama-compat` columns are intentionally separate:
exact/default is the semantic control, while `llama-compat` is the closer
structural comparison with llama.cpp's B2 natural-output-horizon route.

#### Cross-engine GGUF decode timing contract

Use this contract whenever a hipEngine GGUF decode rate is placed beside a
llama.cpp rate:

- hipEngine true AR excludes model load, prefill, its prompt-produced first
  token, and warmup. Timing begins before the first of `N` measured
  `session.step()` calls and ends after the `N`th call returns.
- hipEngine MTP excludes prefill and draft warmup. Cross-engine throughput uses
  complete `cycle_wall_ms`, measured from proposal-cycle entry through draft,
  target verification, recurrent/KV state commit, acceptance, and output
  accounting. The canonical same-harness MTP/AR objective may retain the
  slightly narrower summed stage wall, but that value is not ranked against
  llama.cpp.
- llama.cpp sets `server_slot::t_start_generation` **after** sampling the first
  output token, while `predicted_n` includes that token. Native
  `predicted_n / predicted_ms` therefore counts one untimed token per request.
  To compare `N` timed transitions, request `N+1` outputs and report
  `sum(predicted_n - 1) * 1000 / sum(predicted_ms)`.
- Client/request wall includes prompt processing, HTTP, and response handling;
  it is a separate end-to-end diagnostic and is never compared with direct
  decode-only wall. Record KV dtype differences beside every cross-engine row.

The committed runner emits native, client, and transition-normalized fields.
The exact local llama.cpp source/binary lineage and instrumentation are retained
under [`benchmarks/llama.cpp/`](llama.cpp/).

<!-- BEGIN TOPLINE:SPECULATIVE -->
#### GGUF MTP comparison, Radeon Pro W7900/gfx1100

| Metric | hipEngine GGUF true AR | hipEngine GGUF exact/default | hipEngine GGUF `llama-compat` | llama.cpp HIP base AR |
| --- | ---: | ---: | ---: | ---: |
| Route | State-bound graph, no MTP | B3, fixed 10 cycles | B2, natural24/cyclecap24 | Natural25 request / 24 timed transitions |
| Decode | **98.75 tok/s fixed / 93.30 tok/s natural24** | 68.50 tok/s | 79.70 tok/s | 78.29 tok/s transition-normalized |
| Own true AR | same route | 98.75 tok/s | 93.30 tok/s | same route |
| MTP / own AR | 1.0000x | **0.6936x** | **0.8542x** | n/a |
| Draft acceptance | n/a | 73.53% | 82.95% | n/a |
| Accepted draft/output | n/a | 50.00% | 60.83% | n/a |
| Complete wall per output/transition | 10.718 ms natural24 | 14.696 ms | 12.578 ms | 12.774 ms |
| State/commit contract | serial autoregressive | serial-prefix preserving | direct partial commit/dp4a; accuracy-traded | native llama.cpp autoregressive |

The old `34.28-34.49 tok/s` true-AR denominator was an eager-only benchmark
path, not the fastest production no-MTP route. gfx1100 had no backend graph
capability even though the state-bound implementation was already shared with
gfx1151. A clean W7900 p512/d24 gate now passes all 24 hidden/GDN/KV/token
transitions and moves capture-inclusive wall from **30.536 to 12.514 ms/token
(2.4402x)**. The full natural24 suite matches every prior eager generated-token
preview/tail and moves **34.28 -> 93.30 tok/s** in the same MTP wrapper.

At the matched cross-engine boundary, hipEngine counts 240 complete post-prefill
transitions including graph capture/instantiate/close; llama.cpp build 9648
requests 25 outputs and counts the 240 timed transitions inside `predicted_ms`.
hipEngine is **93.30 versus 78.29 tok/s (+19.19%)**. BF16 versus F16 KV remains
disclosed. llama.cpp stays an external diagnostic with
`performance_claim=false` because its local instrumentation patchset is dirty
but preserved.

Neither MTP route beats the corrected production AR control. Exact/default
remains the semantic control; `llama-compat` remains explicit-only because
direct partial commit is not serial-prefix-equivalent. The fixed-cycle exact
and natural24 compatibility rows are different protocols and are not ranked
against each other.

##### W7900 `llama-compat` full-suite gate against graph AR

| Scope | Prompts | True AR tok/s | `llama-compat` tok/s | MTP / AR | Draft acceptance | Accepted/output | Cycle wall/output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 10 | **93.30** | 79.70 | **0.8542x** | 82.95% | 60.83% | 12.578 ms |
| Train | 6 | **93.73** | 82.01 | **0.8749x** | **88.12%** | 61.81% | 12.224 ms |
| Heldout | 4 | **92.67** | 76.47 | **0.8252x** | **76.00%** | 59.38% | 13.110 ms |
| `code` | 4 | **93.63** | 86.99 | **0.9291x** | 95.38% | 64.58% | 11.523 ms |
| `general_en` | 2 | **90.99** | 75.87 | **0.8338x** | 75.68% | 58.33% | 13.212 ms |
| `general_ja` | 2 | **94.38** | 72.17 | **0.7647x** | 69.23% | 56.25% | 13.889 ms |
| `mixed_ja_en` | 2 | **93.98** | 78.71 | **0.8375x** | 82.86% | 60.42% | 12.744 ms |

All four categories and heldout lose to graph AR despite unchanged strong draft
acceptance. This corrects the earlier false MTP-win conclusion without changing
the compatibility semantics. Artifact:
[`2026-07-12-w7900-gfx1100-gguf-graph-ar-refresh.json`](results/2026-07-12-w7900-gfx1100-gguf-graph-ar-refresh.json).

#### GGUF MTP comparison, Radeon 8060S/gfx1151

| Metric | hipEngine GGUF exact/default | hipEngine GGUF `llama-compat` | llama.cpp HIP |
| --- | ---: | ---: | ---: |
| Route | B5, fixed 10 cycles | B2, natural24/cyclecap24 | B2, natural25 request / 24 timed transitions |
| Canonical/native MTP decode | 51.81 tok/s (0.9571x own AR) | **69.50 tok/s (1.2776x own AR)** | 69.44 tok/s native (1.3752x own AR; not cross-engine comparable) |
| Cross-engine MTP decode-transition rate | n/a: fixed-cycle horizon | **69.38 tok/s** | 66.66 tok/s |
| Cross-engine own AR transition rate | n/a: fixed-cycle horizon | **54.40 tok/s** | 48.47 tok/s |
| Cross-engine MTP / own AR | n/a | 1.2755x | 1.3752x |
| Draft acceptance | 72.33% | 77.72% | 79.56% |
| Accepted draft/output | 53.49% | 59.58% | 57.60% |
| Full-cycle/predicted wall per counted output or timed transition | 19.360 ms/output | 14.413 ms/output | 15.001 ms/transition |
| State/commit contract | exact/default, serial-prefix preserving | direct partial commit/dp4a; accuracy-traded | native llama.cpp compatibility target |

The current exact/default B5 route no longer beats true AR after the
correctness/state-lifecycle pass: **51.81 vs 54.14 tok/s (0.9571x)**. Its old
61.98 tok/s row is retained only as history. `llama-compat` remains a separate,
explicit-only semantic contract and is not serial-prefix-equivalent.

The cross-engine rows use the canonical transition-matched timing contract:
hipEngine uses complete cycle wall; llama.cpp requests 25 outputs and counts
the 24 transitions inside `predicted_ms`. This removes llama.cpp's native
one-untimed-token numerator advantage. hipEngine uses BF16 KV while llama.cpp
uses F16 KV, which remains a model-execution difference even with matched timer
boundaries. The captured llama.cpp source is dirty but fully preserved in the
repository patchset; the binary hash is authoritative and
`performance_claim=false`.

##### gfx1151 `llama-compat` full-suite gate

| Scope | Prompts | True AR tok/s | `llama-compat` tok/s | MTP / AR | Draft acceptance | Accepted/output | Cycle wall/output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 10 | 54.40 | **69.50** | **1.2776x** | 77.72% | 59.58% | 14.413 ms |
| Train | 6 | 54.44 | **70.96** | **1.3034x** | **82.08%** | 60.42% | 14.116 ms |
| Heldout | 4 | 54.33 | **67.42** | **1.2408x** | **71.79%** | 58.33% | 14.858 ms |
| `code` | 4 | 54.42 | **74.81** | **1.3747x** | 91.04% | 63.54% | 13.387 ms |
| `general_en` | 2 | 54.50 | **67.62** | **1.2407x** | 71.79% | 58.33% | 14.811 ms |
| `general_ja` | 2 | 54.40 | **66.60** | **1.2242x** | 69.23% | 56.25% | 15.042 ms |
| `mixed_ja_en` | 2 | 54.25 | **64.90** | **1.1964x** | 69.23% | 56.25% | 15.438 ms |

All four categories and the heldout split beat their true same-protocol AR
controls. Train/heldout draft acceptance is **82.08% / 71.79%**; the gap is
kept visible rather than averaged away.

#### Dense PARO DFlash

| Path | Platform and protocol | Result | Evidence status |
| --- | --- | ---: | --- |
| DFlash B=4 online-gated | W7900/gfx1100; Qwen3.6-27B PARO target plus Qwen3.6-27B DFlash drafter; 9 prompts; 64 decode tokens | 40.10 vs 32.57 AR tok/s, **1.231x** | Retained under the recorded DFlash gate; source tree was dirty and must be refreshed before changing the claim |
<!-- END TOPLINE:SPECULATIVE -->

Artifacts: [W7900 GGUF MTP transfer](results/2026-07-12-w7900-gfx1100-gguf-mtp-transfer.json),
[DFlash](results/2026-06-11-hipengine-dflash-27b-dense-hardening-rerun.json),
[current gfx1151 GGUF MTP refresh](results/2026-07-12-gfx1151-gguf-mtp-refresh.json),
and [llama.cpp instrumentation manifest](llama.cpp/manifest.json). Historical
gfx1151 sources remain [exact B5](results/2026-07-02-ar-mtp-default-parallelattn-full.json),
[`llama-compat` B2](results/2026-07-03-ar-mtp-llama-compat-directcommit-nocopy-natural24-cyclecap24-f32head-full.json),
and [llama.cpp B2](results/2026-07-02-llamacpp-mtp-stage-timing-b2-natural24-rerun.json).

The current clean gfx1151 PARO DFlash profile remains outside this eligible
table: it is exact but measures only `9.676` versus `65.266 tok/s` AR
(`0.14825x`), so DFlash stays default-off. Branch-copy is faster but diverges
at generated token 1, and fused target LM-head is 5.16% slower. The diagnostic
artifact is [SOL-S4](results/2026-07-11-sol-s4-gfx1151-paro-dflash-profile.json).

### GGUF decode

These are exact repeated-token SOL-G4/G5 rows, not natural-prompt quality or
speculative-economics results. The graph delta uses its same-run eager control;
the Q8T16 row is the current eager timing while SOL-G4 remains the historical
revision-bisect/Amdahl baseline.

<!-- BEGIN TOPLINE:GFX1151_GGUF_EAGER -->
| Path | Platform and protocol | Result | Evidence status |
| --- | --- | ---: | --- |
| GGUF eager c1 | Radeon 8060S/gfx1151; Qwen3.6-35B-A3B UD-Q4_K_M; BF16 KV; `[9707] * 512`; TheRock HIP 7.15; TuneD accelerator-performance; clean scalar/candidate/scalar, 1 discarded + 4 measured runs per leg; 128 eager steps; graph off | **48.850 tok/s** (`20.471 ms/token`), **+0.309%** vs clean scalar control | Retained for this exact repeated-token protocol; control/candidate ranges do not overlap, every output ID is 9707, and the G1 hidden/state/KV oracle is linked |
| GGUF state-bound graph c1 | Radeon 8060S/gfx1151; same current model/KV/prompt/stack; 1 warmup + 4 measured rotating same-session runs; 128 steps; capture and destroy charged | **48.704 tok/s** (`20.532 ms/token`), **-0.293%** vs same-run eager; **+0.201%** vs scalar graph | Exact 128/128 state/KV/token replay, but current G5 rejects a graph-over-eager speed claim; graph default policy is tracked separately |
<!-- END TOPLINE:GFX1151_GGUF_EAGER -->

Artifacts: [`Q8T16 wave/block production A/B`](results/2026-07-12-gfx1151-q8-t16-waveblock-production.json),
[`SOL-G4 eager audit`](results/2026-07-11-sol-g4-gfx1151-gguf-eager-decode-audit.json),
and [`SOL-G5 production graph audit`](results/2026-07-11-sol-g5-gfx1151-gguf-decode-graph-production-audit.json).

### PARO concurrency and production routing

The current gfx1151 concurrency table publishes only the exact production
classification. c1 has a retained exact-fixture timing; c2-c8 use independent
width-1 sessions because every native candidate fails the independent-c1
sequence at generated index 2. This is not a cross-engine concurrency-speed
claim, and the rejected native rates remain in the detailed record below.

<!-- BEGIN TOPLINE:GFX1151_PARO_CURRENT -->
| Client c | Production backend groups | Exact classification | Retained aggregate decode |
| ---: | --- | --- | ---: |
| 1 | `1` | c1 oracle / accepted | **66.910 tok/s** (`14.946 ms/token`) |
| 2 | `1+1` | explicitly serial | no separate c>N claim |
| 3 | `1+1+1` | explicitly serial | no separate c>N claim |
| 4 | `1+1+1+1` | explicitly serial | no separate c>N claim |
| 5 | five width-1 groups | explicitly serial | no separate c>N claim |
| 6 | six width-1 groups | explicitly serial | no separate c>N claim |
| 7 | seven width-1 groups | explicitly serial | no separate c>N claim |
| 8 | eight width-1 groups | explicitly serial | no separate c>N claim |
<!-- END TOPLINE:GFX1151_PARO_CURRENT -->

Artifacts: [P1 exact catalog](results/2026-07-11-sol-p1-gfx1151-paro-c1-c8-exact-catalog.json)
and [P2 ragged lifecycle](results/2026-07-11-sol-p2-gfx1151-paro-ragged-lifecycle.json).

The retained gfx1100 and gfx1151 HIP/Vulkan timing-contract v2 micro matrices
are linked from the platform index and
[`docs/HIP-vs-VULKAN.md`](../docs/HIP-vs-VULKAN.md); they are not
model-throughput toplines.

## Platform Records And Diagnostics

The dated records below preserve protocols, blockers, commands, and artifact
links without publishing their numeric rows as current results. Their removed
tables remain recoverable from the linked compact artifacts, changelog, and
[`benchmarks/HISTORY.md`](HISTORY.md).

### W7900 PARO context capacity, 2026-07-12

**Status: current capacity measurement; INT8 quality rejected.** Clean detached
hipEngine `8116c453` (rebased-equivalent reachable `8708304f`; runtime and
benchmark code identical) on W7900/gfx1100 and TheRock HIP 7.15 used the current
Qwen3.6 packed PARO snapshot, repeated token `9707`, 128 generated tokens, four
warmup tokens, current chunk autotuning, and graph decode. The physical layout
and 24 GiB portability gates pass, but the required Qwen3.6 long-rollout gate
does not. Therefore 256K INT8 is an allocation-capacity result, not a usable or
supported route.

<!-- BEGIN TOPLINE:W7900_MEMORY_CAPACITY -->
| Route | Context/decode | Tracked peak | 24 GiB margin | Retained KV | Layout audit | Quality status |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| PARO BF16 KV | 128K/128 | **22.124 GiB** | 1.876 GiB | 2.690 GB | Passed | Reference path |
| PARO INT8 per-token/head KV, FP16 scales | 256K/128 | **23.957 GiB** | 0.043 GiB | 2.708 GB | Passed; no BF16 shadow | **Rejected** by Qwen3.6 128K/128 rollout |
<!-- END TOPLINE:W7900_MEMORY_CAPACITY -->

The capacity rows are one run each because memory needs one high-water
measurement, not five performance repetitions. Timings are diagnostic only.
The INT8 row retains 2,686,976,000 payload bytes plus 20,992,000 FP16 scale
bytes across ten full-attention layers, with no persistent BF16 K/V shadow.

The matched Qwen3.6 BF16-vs-INT8 128K/128 quality run is finite and passes the
layout audit, but diverges at generated index 2. FP16 scales measure mean/max KL
**3.7646/10.0796** and **3.88%** top-1 agreement; an FP32-scale follow-up also
rejects at **3.6300/10.0198 KL** and **7.75%** top-1. This supersedes the old
Qwen3.5-fixture implication. Do not describe 256K INT8 as supported until a
current Qwen3.6 long-rollout gate passes.

Artifact:
[`2026-07-12-w7900-v030-paro-context-capacity.json`](results/2026-07-12-w7900-v030-paro-context-capacity.json).
Quality diagnostics:
[FP16 scales](results/2026-07-12-w7900-v030-paro-int8-kv-128k-quality.json)
and [FP32 scales](results/2026-07-12-w7900-v030-paro-int8-kv-fp32scale-128k-quality.json).

### W7900 PARO gfx1151 transfer gate, 2026-07-12

**Status: retained scoped-default validation and retained negative transfer
decision.** Clean detached `255e5aca` on W7900/GPU0 used the exact packed W4
PARO/BF16-KV model, repeated token `9707`, graph decode, TheRock HIP 7.15, two
discarded plus five measured runs per leg, and cached JIT. Because the first
off/on/off AOTriton sequence drifted with run order, the reverse on/off/on
sequence completes a balanced 15-sample comparison per mode.

| Workload | Same-stream prefill | Isolated prefill | Prefill delta | Total measured wall reduction |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | `2843.083` | `2889.650` | **+1.638%** | **1.653%** |
| 1K/128 | `2951.433` | `2966.051` | **+0.495%** | **0.127%** |
| 4K/128 | `2924.276` | `2929.897` | **+0.192%** | **0.562%** |

At 512, 1K, and 4K the isolated and same-stream legs match sampled seed, final
hidden, all 30 Conv/GDN state families, and all 10 live K/V families
byte-for-byte. This matrix was measured before queue isolation was narrowed by
query shape, so its isolated leg includes the 256-query 512/1K route as well as
the 4096-query 4K route. The merged runtime keeps 256-query AOTriton on the
caller stream and isolates query rows >=512. The 4K result therefore directly
validates the merged gfx1100 default; 512/1K remain supporting exact transfer
diagnostics rather than claims for the final route. No additional runtime
change is needed.

The architecture-specific chunk profile does not transfer. With AOTriton queue
mode held equally same-stream, linear/MoE-256 changes prefill by
`-7.723%/-8.782%/-6.398%` at 512/1K/4K. Its `0.58%-1.72%` tracked-memory
reduction does not offset disjoint, uniformly slower throughput ranges, so
`gfx1100` keeps the generic chunk policy.

Artifact:
[`2026-07-12-w7900-gfx1100-paro-gfx1151-transfer.json`](results/2026-07-12-w7900-gfx1100-paro-gfx1151-transfer.json).

### Superseded W7900 model sweep, 2026-07-07

**Status: superseded diagnostic.** This used one max-128K hipEngine session,
eager GGUF decode, one llama.cpp sample per phase, and no W7900-local state
oracle. The accepted clean 2026-07-12 table above replaces it. Historical
artifacts remain available from
[`2026-07-07...summary.json`](results/2026-07-07-w7900-gpu0-readme-refresh-20260707-104756-summary.json).

### Superseded gfx1151 model sweep, 2026-06-15

**Status: superseded diagnostic.** This was one measured run per shape with no
measured warmup, incomplete summary provenance, and unusable 512 MiB aperture
memory readings for llama.cpp. The accepted 2026-07-11 sweep above replaces
every public row with five-sample, clean-provenance evidence and proper GTT
sampling. Keep the old record only for history.

Artifacts: [old summary](results/2026-06-15-gfx1151-readme-udq4km-20260615-040438-summary.json),
[hipEngine PARO](results/2026-06-15-gfx1151-readme-udq4km-20260615-040438-hipengine-paro-packed-1run.json),
[hipEngine GGUF](results/2026-06-15-gfx1151-readme-udq4km-20260615-040438-hipengine-gguf-ud-q4km-1run.json),
[llama.cpp HIP](results/2026-06-15-gfx1151-readme-udq4km-20260615-040438-llamacpp-hip-ud-q4km-f16kv.json),
and [llama.cpp Vulkan](results/2026-06-15-gfx1151-readme-udq4km-20260615-040438-llamacpp-vulkan-ud-q4km-f16kv.json).

### W7900 concurrency, 2026-07-07

**Status: stale diagnostic.** hipEngine uses PARO W4/BF16 KV, llama.cpp uses Vulkan
Q4_K_M/f16 KV, and vLLM uses GPTQ Int4. hipEngine and llama.cpp report backend
decode timing; vLLM reports OpenAI client wall throughput. The artifact exposes
scaling behavior within each column, not an apples-to-apples engine ranking.

<!-- BEGIN TOPLINE:W7900_CONCURRENCY -->
No eligible concurrency row; the mixed-quant, mixed-timing sweep remains linked below pending rerun.
<!-- END TOPLINE:W7900_CONCURRENCY -->

Protocol: prompt 512, decode 128, 8 warmup decode tokens, median of 3. hipEngine
`c=1` uses the single-sequence graph-replay benchmark and `c>1` uses the native
batch benchmark. llama.cpp restarts `llama-server` for each concurrency and
repetition with `-np c -c 1024*c`. vLLM uses the OpenAI completions endpoint.

Artifacts: [hipEngine](results/2026-07-07-w7900-gpu0-readme-refresh-20260707-104756-hipengine-concurrency-w7900/summary.json),
[llama.cpp Vulkan](results/2026-07-07-w7900-gpu0-readme-refresh-20260707-104756-llamacpp-vulkan-concurrency-w7900/summary.json),
[vLLM](results/2026-07-07-w7900-gpu0-readme-refresh-20260707-104756-vllm-localbuild-gptq-int4-concurrency-c1-c8-w7900.json), and
[combined summary](results/2026-07-07-w7900-gpu0-readme-refresh-20260707-104756-summary.json).

### gfx1151 PARO exact shape/routing catalog, 2026-07-11

**Status: retained for c1 performance, c1-c8 routing correctness, and production
lifecycle safety.** P1 ran from clean detached hipEngine `a18ff7bc`; P2 ran from
clean detached `6f1910c9` on the same Radeon 8060S and exact model/fixture.
P1 uses the same 512-token row at every width and compares 137 generated IDs
against true single-request sessions. P2 uses ragged lengths 449 through 512
and checks every persistent state/KV family through c8-to-c1 retirement.

No eligible native-batch timing row exists. The c2-c8 native measurements below
are correctness-rejected diagnostics; production uses the exact serial groups
shown in the current concurrency table above.

Protocol: Qwen3.6-35B-A3B PARO snapshot
`437eba06df05aad71a4dacdcaf3fff70ae1ee8a1`, W4 PARO, BF16 KV, 40 layers,
8 warmup decode steps, 128 measured decode steps, and greedy sampling. Exact
prompt-ID SHA-256 is `b162b2d0...2388`; model fingerprint is
`995a8c67...d917`. c1 is the median of three fresh processes
(`66.948/66.754/66.910 tok/s`). c2-c8 have one diagnostic native timing each;
correctness rejection makes more repetitions immaterial.

Rejected native diagnostics:

| c | Candidate shape | Equal prefix per row | Aggregate tok/s | Decision |
| ---: | --- | --- | ---: | --- |
| 2 | full native, selected-c1 MoE, batch-GEMV output | `2,2` | 78.525 | reject at index 2 (`17` vs `220`) |
| 3 | rowchunk2 full attention, selected-c1 MoE | `2,2,2` | 87.472 | reject at index 2 |
| 4 | rowchunk2 full attention, selected-c1 MoE | `2 x4` | 99.641 | reject at index 2 |
| 5 | rowchunk2 full attention, selected-c1 MoE | `2 x5` | 102.178 | reject at index 2 |
| 6 | selected-layer rowchunk2, selected-c1 MoE | `2 x6` | 109.806 | reject at index 2 |
| 7 | rowchunk2 full attention, selected-c1 MoE | `2 x7` | 109.580 | reject at index 2 |
| 8 | rowchunk2 full attention, selected-c1 MoE | `2 x8` | 115.508 | reject at index 2 |

The c8 teacher-forced bisect keeps packed-prefill hidden, recurrent state, and
full-attention KV bit-exact. On decode step 0, the selected-c1 route first
changes the input/state of linear layer 4; the visible token flips on the next
step. Grouped-compact produces the correct token at index 2 but fails the full
shrinking sequence at index 4, so it is not a replacement default.

The retained P2 lifecycle gate keeps physical slots sparse and un-compacted.
Slot 3 exits by EOS at c8; later explicit cancellation creates middle, tail,
and front holes while slot 4 survives to c1. Every generated sequence, all 30
linear Conv/GDN state pairs, and all 10 live K/V layer pairs are SHA-256 exact
against independent c1 at each row's retirement boundary. Ragged packed prefill
selects `per_segment_ragged_exact`; equal-length packed prefill is unchanged.

Run record:

| Field | Value |
| --- | --- |
| GPU/backend | AMD Ryzen AI MAX+ 395 / Radeon 8060S, detected gfx1151, target gfx1151 |
| Source/build | clean hipEngine `a18ff7bc428833a5f3d87ed422d04633abbf0b10`; Python 3.12.13; TheRock HIP `7.13.60980-c76140fa27`; detected/target gfx1151 |
| Timing scope | Direct resident backend decode wall; c1 median of 3; rejected native rows one run each |
| Correctness | c1 endpoints repeat across 3/3 runs and match the independent sequence final ID. c2-c8 native rows fail every row at index 2. The production serial route passes ragged c8-to-c1 for 8/8 token/state/KV rows. |
| Production route | `true_c1_graph` for c1; `scheduler_true_c1_fallback` for c2-c8. No gfx1100 artifact may select this gfx1151 catalog. |
| Lifecycle route | `per_segment_ragged_exact` prefill plus true-c1 decode; EOS and front/middle/tail sparse cancellation are exact. No throughput claim is attached to the fallback. |
| Current artifacts | [`P1 exact catalog`](results/2026-07-11-sol-p1-gfx1151-paro-c1-c8-exact-catalog.json), [`P2 ragged lifecycle`](results/2026-07-11-sol-p2-gfx1151-paro-ragged-lifecycle.json) |
| Historical correction | [`2026-07-10...current-diagnostic-summary.json`](results/2026-07-10-gfx1151-paro-cn-current-diagnostic-summary.json), [`true-c1 shrink gate`](results/2026-07-10-gfx1151-paro-true-c1-shrinking-gates.json) |

Reproduce c2-c8 with `scripts/qwen35_batch_equality_matrix.py --batch-sizes
2,3,4,5,6,7,8`; use `scripts/qwen35_paro_bench.py --prompt-fixture ...
--prompt-row 0` for the exact c1 control. Commands and raw SHA-256 values are
embedded in the compact artifact. Reproduce P2 with
`scripts/qwen35_batch_shrinking_correctness.py --batch-size 8
--prompt-lengths 449,458,467,476,485,494,503,512 --steps-per-width 1
--survivor-slot 4 --eos-slot 3` and the same model/fixture.

### gfx1151 PARO DFlash S4 profile, 2026-07-11

**Status: retained diagnostic profile; no performance claim.** Clean detached
hipEngine `8eb27215` ran the curated 35B W4 PARO/BF16-KV target and 35B BF16
DFlash drafter on the first `code_promotion` fixture, B4 and 32 output tokens.
The exact/default replay route matches all AR IDs and finite-logit gates, but it
is decisively slower:

| Route | AR tok/s | DFlash tok/s | DFlash/AR | Exact | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| Canonical replay, graph auto | 65.266 | 9.676 | 0.148x | yes | S4 profile accepted; speed rejected |
| Branch-copy commit | 65.269 | 14.450 | 0.221x | no, first mismatch 1 | S5 correctness rejection |
| Canonical replay + fused target LM-head | 65.223 | 9.177 | 0.141x | yes | S7 performance rejection (-5.16%) |

The exact row accepts 1/114 proposed draft tokens and spends 5.6875 target rows
per output. Coarse attribution is 74.62% target verify and 25.21% draft; the
profiling-only synchronized companion identifies target linear layers (37.41%
of total wall), drafter decoder+LM-head (25.55%), and canonical replay plus
scratch canonicalization (20.80%) as the largest buckets. Commit scatter is
0.25%, drafter top-k/readback 0.41%, and accept readback 0.04%.

Exact replay records 30 validated verifier-graph misses and zero hits across
two shapes. Branch-copy records 27 hits after two captures, but inherits the
known non-canonical c>N state and fails output equality. S6 is therefore parked:
wider verification would amplify rejected work, and this c1 row shows no
multi-request draft group-cap bottleneck. Compact evidence:
[`2026-07-11-sol-s4-gfx1151-paro-dflash-profile.json`](results/2026-07-11-sol-s4-gfx1151-paro-dflash-profile.json).

### gfx1151 GGUF server automatic-route gate, 2026-07-11

**Status: diagnostic correctness rejection; no performance claim.** The first
post-E1/E2/E3 server matrix runs the committed ten-prompt category JSONL and its
documented four-prompt heldout directly. It records exact choice IDs, canonical
model/suite provenance, owned batch timing, and realized queue/backend groups.
The source tree had no staged or unstaged changes; 255 unrelated untracked
benchmark files are disclosed, so this diagnostic is not a clean retained
performance row.

| Client c | Realized groups (full suite) | AR median tok/s | Compatibility MTP median tok/s | MTP/AR | Exact vs c1 AR |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | ten c1 groups | 35.92 | 39.35 | 1.095x | fail `general_ja_explain` |
| 2 | five c2 groups | 56.84 | 58.50 | 1.029x | fail 3/10 prompts |
| 3 | c3+c3+c3+c1 | 60.83 | 61.47 | 1.011x | fail 2/10 prompts |
| 4 | c4+c4+c2 | 69.70 | 66.13 | 0.949x | fail 3/10 prompts |
| 8 | c4+c4+c2 (route cap 4) | 69.84 | 65.87 | 0.943x | fail 3/10 prompts |

Full-suite values are medians of three after one discarded route/shape warmup;
heldout values use five repetitions. The mixed client-c3 aggregate hides the
reason realized groups matter: isolated full-suite c3 groups are +1.71% for
MTP, but isolated heldout c3 groups are **-3.92%**, so c3 does not activate.
More importantly, the current server hook is the documented
`llama-compat` direct-commit/dp4a route and is not serial-prefix-equivalent.
Its apparent c1/c2 speed benefit cannot enter automatic/default routing.

True AR c1-c4 is exact across every repetition. One of three client-c8 AR runs
changes `general_ja_explain` even though its actual backend groups are c4+c4+c2;
that remains a separate SOL-G8 exact-concurrency blocker, not evidence for a
width-8 backend. SOL-S1 now makes automatic MTP fall back to the default AR
route until an exact/default hook exists; explicit opt-in keeps the
compatibility contract. The compact artifact is
[`2026-07-11-sol-s1-gfx1151-server-auto-route-gate.json`](results/2026-07-11-sol-s1-gfx1151-server-auto-route-gate.json).

### gfx1151 historical cross-engine concurrency, 2026-06-15

**Status: stale diagnostic.** hipEngine uses PARO W4/BF16 KV; llama.cpp uses
Vulkan Q4_K_S/f16 KV. vLLM did not produce a healthy server. The summary lacks
the measured hipEngine commit, and the then-used per-run device properties could
report gfx1100 even though the run forced `HIPENGINE_HIP_ARCH=gfx1151`.

<!-- BEGIN TOPLINE:GFX1151_CONCURRENCY -->
No eligible concurrency row; the `performance_claim=false` snapshot remains linked below pending rerun.
<!-- END TOPLINE:GFX1151_CONCURRENCY -->

Protocol: prompt 512, decode 128, 8 warmup decode tokens, median of 3. Primitive
c>1 attention/KV checks passed. The generated-token field used the older
batch-shaped reference and is not independent-c1 evidence. Profiler, scaling,
and provenance gates also did not pass.

Artifacts: [combined summary](results/2026-06-15-gfx1151-readme-concurrency-20260615-213804-summary.json),
[hipEngine](results/2026-06-15-gfx1151-readme-concurrency-20260615-122207-hipengine-paro/summary.json),
[llama.cpp Vulkan](results/2026-06-15-gfx1151-readme-concurrency-20260615-213804-llamacpp-vulkan/summary.json), and
[vLLM blocker](results/2026-06-15-gfx1151-readme-concurrency-20260615-122207-vllm-gptq-int4-blocked.json).

## README Sweep Test Procedure

### W7900 model and concurrency refresh

Use a clean detached worktree. The wrapper fixes the GPU mapping, TheRock
environment, model paths, llama.cpp binaries, JIT cache policy, and output
layout.

```bash
RUN_TAG=$(date -u +%Y%m%d-%H%M%S)
WORKTREE="/tmp/hipengine-readme-w7900-${RUN_TAG}"
git worktree add --detach "$WORKTREE" HEAD

OUTDIR="$PWD/benchmarks/results" \
RUN_TAG="$RUN_TAG" \
REPO_ROOT="$WORKTREE" \
  "$WORKTREE/scripts/run_w7900_readme_refresh.sh" all
```

Subset commands:

```bash
scripts/run_w7900_readme_refresh.sh hipengine
scripts/run_w7900_readme_refresh.sh llamacpp
scripts/run_w7900_readme_refresh.sh concurrency
scripts/run_w7900_readme_refresh.sh vllm
```

Required W7900 settings:

| Surface | Settings |
| --- | --- |
| Device mapping | `HIP_VISIBLE_DEVICES=0`; W7900 is amdgpu `card1`; llama.cpp uses `ROCm0` and `Vulkan0` after masking |
| hipEngine environment | `/home/lhl/mambaforge/envs/therock/bin/python3.12`; hermetic TheRock root from `python -m rocm_sdk path --root`; `HSA_OVERRIDE_GFX_VERSION=11.0.0` |
| Model sweep | `512/128 1K/128 4K/128 32K/128 64K/128 128K/128`; 2 warmups; 5 measured; resident max-context session |
| PARO | snapshot `437eba06df05aad71a4dacdcaf3fff70ae1ee8a1`; `hip_gfx1100`; `packed_paro_w4`; BF16 KV; AOTriton threshold 512; graph replay decode |
| hipEngine GGUF | MTP-bearing Qwen3.6-35B-A3B `UD-Q4_K_M`; decode repack; WMMA bulk prefill; GEMV eager decode; BF16 KV |
| llama.cpp | Same GGUF; `-ngl 99 -fa 1 -ctk f16 -ctv f16`; split prefill/decode; one repetition per phase |
| Concurrency | prompt 512; decode 128; warmup 8; c=1,2,4,8; 3 repetitions; fixed token-id fixture |

Never add a combined summary with `performance_claim=false` to the current
topline table. Keep its artifact linked in the diagnostic section instead. A
retained refresh also needs the correctness and repetition gates from
[`docs/BENCHMARK.md`](../docs/BENCHMARK.md).

### gfx1151 model and concurrency refresh

The committed
[`run_gfx1151_readme_refresh.sh`](../scripts/run_gfx1151_readme_refresh.sh)
replaces the unreproducible 2026-06-15 `/tmp/run_gfx1151_readme_udq4km.sh`.
Run it from a clean detached worktree so component provenance observes no
tracked or untracked source changes:

```bash
RUN_TAG=$(date -u +%Y%m%d-%H%M%S)
WORKTREE="/tmp/hipengine-readme-gfx1151-${RUN_TAG}"
git worktree add --detach "$WORKTREE" HEAD

OUTDIR="$PWD/benchmarks/results" \
RUN_TAG="$RUN_TAG" \
REPO_ROOT="$WORKTREE" \
  "$WORKTREE/scripts/run_gfx1151_readme_refresh.sh" all
```

Subset commands are `... hipengine`, `... llamacpp`, and `... summary`. The runner fixes the
model identities, six standard shapes, native gfx1151 compiler target,
torch-free hermetic TheRock environment, two discarded plus five measured
hipEngine runs, and five internal llama-bench repetitions. It records a
canonical provenance object in every component artifact. Each hipEngine shape
runs in its own process with a right-sized resident session, then the committed
merge gate verifies and preserves all samples in one compact rollup. This keeps
512/1K memory honest and avoids imposing a 128K allocation on every row.
Discarded runs warm the same kernels through eager submission; each measured
run captures and destroys a fresh state-bound graph after reset/prefill/warmup,
so no captured graph crosses a session reset. The summary phase verifies all
four component artifacts together and generates the Markdown tables only when
their provenance, model/build identity, correctness, return-code, variance,
and memory-scope gates pass.

gfx1151 is a UMA APU: sysfs reports only a 512 MiB visible-VRAM aperture while
the amdgpu GTT domain is 120 GiB and holds model allocations. The runner
therefore samples `mem_info_gtt_used` for llama.cpp HIP/Vulkan. The public
memory table must label that whole-device GTT scope and separately identify
hipEngine tracked or HIP phase-sampled peaks; it must not relabel the 512 MiB
aperture as total model memory.

Before updating the gfx1151 tables:

1. Detect and record `gfx1151` from the runtime/build output; do not fill the
   artifact from a CLI label alone.
2. Run 2 discarded warmups and 5 measured repetitions for the six model-sweep
   shapes.
3. Run PARO concurrency for c=1 through c=8, including odd widths and dynamic
   c=8 to c=1 shrinking, with exact all-choice generated-token counts.
4. Keep comparison engines in separate columns when quant or timing scope
   differs. Bold may mark the raw row leader, but the nearby text must state
   that a cross-quant or cross-memory-scope maximum is descriptive rather than
   a controlled backend win.

The clean P1/P2 artifacts now satisfy the current c1-c8 independent-c1 and
ragged shrinking lifecycle gates. They retain c1 timing and classify c2-c8 as
exact width-1 production groups because every native candidate is
correctness-red. A future cross-engine concurrency-speed table still requires
one matched quant/timing protocol; do not republish the superseded 2026-06-15
native numbers as production throughput.

The lower-level hipEngine sweep command is:

```bash
PYTHONPATH=. \
HIPENGINE_HIP_ARCH=gfx1151 \
python3 scripts/qwen35_readme_sweep.py \
  --engine paro \
  --model /home/lhl/.cache/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-packed/snapshots/437eba06df05aad71a4dacdcaf3fff70ae1ee8a1 \
  --backend hip_gfx1151 \
  --shared-expert-format packed_paro_w4 \
  --token-id 9707 \
  --workloads 512/128 1K/128 4K/128 32K/128 64K/128 128K/128 \
  --warmup-runs 2 --measured-runs 5 --warmup-decode-tokens 4 \
  --attn-aotriton-min-tokens 512 --graph-replay-decode \
  --compiler-version-file /tmp/hipengine-hipcc-version-gfx1151.txt \
  --require-cached-build \
  --json benchmarks/results/<date>-gfx1151-hipengine-paro-readme-sweep.json
```

This lower-level command is not a complete refresh: use the committed wrapper
for GGUF, llama.cpp, environment capture, and artifact assembly. Concurrency
remains a separate gate because production c2-c8 is currently exact width-1
fallback, not the rejected native timing path.

### Speculative decode refresh

Exact/default GGUF MTP, fixed 10-cycle suite:

```bash
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 \
python3 scripts/gguf_ar_mtp_suite.py \
  --scope full \
  --mtp-route resident-b1-probe-block-direct-cap32k-minrows2-pmin05 \
  --record-cycle-stage-timings \
  --require-cached-build \
  --output benchmarks/results/<date>-ar-mtp-exact-full.json
```

`llama-compat` natural24 direct contract:

```bash
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 \
python3 scripts/gguf_ar_mtp_suite.py \
  --scope full \
  --mtp-route llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm-routerrow-draftdenseq8-draftonly-directcommit \
  --budgets 2 --cycles 24 --max-output-tokens 24 \
  --record-cycle-stage-timings \
  --require-cached-build \
  --output benchmarks/results/<date>-ar-mtp-llama-compat-natural24.json
```

llama.cpp B2 with 24 transition-matched decode steps. Request 25 outputs
because the first is sampled before llama.cpp starts `predicted_ms`:

```bash
python3 scripts/llamacpp_mtp_bench.py \
  --server-bin /path/to/llama-server \
  --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --ctx-size 8192 --concurrency 1 --gpu-layers 99 \
  --flash-attn on --cache-type-k f16 --cache-type-v f16 \
  --draft-max 2 --mode both --protocol natural \
  --prompts benchmarks/prompts/mtpbench-code-general-ja.jsonl \
  --max-tokens 25 --seed 12345 --temperature 0 \
  --top-k 1 --top-p 1 --min-p 0 \
  --server-extra-arg=--reasoning --server-extra-arg=off \
  --output benchmarks/results/<date>-llamacpp-natural25.json
```

Use `aggregate_decode_transition_per_second` for the cross-engine column;
retain `aggregate_decode_predicted_per_second` only as llama.cpp's native
self-report. See the [timing contract](#cross-engine-gguf-decode-timing-contract).

Dense DFlash B=4:

```bash
python3 scripts/dflash_chain_e2e_bench.py \
  --target-model /home/lhl/.cache/huggingface/hub/models--z-lab--Qwen3.6-27B-PARO/snapshots/84f86409151d4f2ec86dc0b6a096d5f6daa7f207 \
  --drafter-model /home/lhl/.cache/huggingface/hub/models--z-lab--Qwen3.6-27B-DFlash/snapshots/0919688658996800f86b895034249700e9481106 \
  --backend hip_gfx1100 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --max-prompts 9 --decode-tokens 64 --draft-budgets 4 \
  --draft-top-k 2 --whole-cycle-gate 0.90 \
  --verifier-mode native_bulk_bplus1 --verifier-graph auto \
  --full-attn-chain-mode batched --canonical-commit-mode branch_copy \
  --adaptive-budget off --hardware-gpu "AMD Radeon Pro W7900" \
  --json benchmarks/results/<date>-dflash-27b-b4.json
```

Use equivalent immutable local snapshots only when the recorded fingerprints
match these target and drafter revisions.

### HIP versus Vulkan microbenchmarks

Microbenchmark claims do not belong in the model-throughput tables. The v2
timing contract and exact bounded rerun commands are in
[`docs/HIP-vs-VULKAN.md`](../docs/HIP-vs-VULKAN.md) and
[`benchmarks/micro/README.md`](micro/README.md). Retained evidence is
[`gfx1100/W7900`](micro/results/gfx1100/w7900/2026-07-11-hip-vulkan-timing-v2-bounded.json)
and
[`gfx1151/Strix Halo`](results/2026-07-12-gfx1151-hip-vulkan-portable-q8.json).
The original stricter Q4/Q6 correctness misses are isolated in
[`2026-07-12-gfx1151-vulkan-q8-isolation-diagnostic.json`](results/2026-07-12-gfx1151-vulkan-q8-isolation-diagnostic.json): both Vulkan dot kernels pass
when given CPU q8_1 blocks, while stock packed-FP16 activation scales are
systematically one code below the CPU/HIP oracle. The retained portable shader
eliminates those scale mismatches; both the gfx1100-matched and current strict
gfx1151 matrices now pass 22/22 comparisons and all 232 burst rows.

## Update Checklist

1. Choose one protocol tuple and record the old artifact before running.
2. Create a clean detached worktree at the revision being measured.
3. Capture the canonical provenance block: GPU identity, configured/resolved
   backend, target arch, VBIOS, power/clock state, kernel, Python, ROCm/HIP
   compiler, Vulkan driver, comparison-engine commit, existing model
   fingerprint, exact argv/environment, and separate staged, unstaged, and
   untracked source state.
4. Run the named warmup, repetition, correctness, and memory protocol. Store raw
   logs outside git and a compact artifact under `benchmarks/results/`.
5. Reject artifacts with missing provenance or failed correctness. A diagnostic
   may be recorded, but it cannot replace a retained row.
6. Update the platform index, table, run record, artifact links, run date, and
   measured revision in this file.
7. Add the required entry to [`benchmarks/CHANGELOG.md`](CHANGELOG.md) and append
   the commands and decision to `WORKLOG.md`.
8. Run the root README sync and validation commands:

```bash
python3 scripts/sync_benchmark_readme.py --write
python3 scripts/sync_benchmark_readme.py --check
python3 -m json.tool benchmarks/results/<new-artifact>.json >/dev/null
git diff --check
```

Run `json.tool` once for each new or changed compact artifact. Do not scan
untracked experiment files as part of the rollup gate.

<a id="natural24-mtp-vs-ar-concurrency-diagnostic"></a>
<a id="blocked--diagnostic-benchmark-attempts"></a>

## Blocked and Diagnostic Benchmark Attempts

- **W7900 GGUF Q4_K_M:** the [2026-07-07 summary](results/2026-07-07-w7900-gpu0-readme-refresh-20260707-104756-summary.json) is the last measured path and
  has `performance_claim=false`. Repetition of token `9707` is confirmed as
  valid for the exact model by llama.cpp and the gfx1151 G1 oracle; the W7900
  measurement still needs its own current state/KV gate and repeated clean
  performance rerun before it can become a baseline.
- **OpenAI MTP server c=1/2/3/4/8:** the corrected
  [2026-07-11 route gate](results/2026-07-11-sol-s1-gfx1151-server-auto-route-gate.json)
  supersedes the pre-contract 2026-07-06/07 timing rows. It counts exact IDs,
  owns batch timing once, records canonical provenance, and separates client,
  queue, backend, and verifier widths. Compatibility MTP is diagnostically
  faster at c1/c2 but changes true-AR IDs, so no automatic-route performance
  claim is eligible; explicit opt-in remains separately labelled.
- **gfx1151 PARO native batching:** P1's clean direct matrix rejects every
  native c2-c8 width at generated index 2; P2's clean ragged lifecycle gate
  accepts the production true-c1 bridge through EOS and front/middle/tail
  sparse slots. Production correctness is closed, but no native width is
  routing-eligible until a general c>N algorithm passes the same independent-c1
  token/state/KV gates. The 2026-07-10 native timing artifact remains diagnostic.
- **gfx1151 model sweep:** the [committed summary](results/2026-06-15-gfx1151-readme-udq4km-20260615-040438-summary.json) omits source/build provenance
  and contains one measured repetition. Its values remain a dated diagnostic.
- **llama.cpp 24 GiB Q8_0 memory:** the former root README tables had no compact
  artifact, model fingerprint, llama.cpp revision, or run date. The numbers were
  removed; rerun before publishing another capacity table.

Rejected and superseded rows remain in JSON artifacts, `WORKLOG.md`,
[`benchmarks/CHANGELOG.md`](CHANGELOG.md), and
[`benchmarks/HISTORY.md`](HISTORY.md). Source-lineage targets and external
baselines in the archive are reference values, not hipEngine toplines.

## Table Conventions

- Workload format is `prompt_tokens/decode_tokens`.
- `tok/s` is reported separately for prefill, backend decode, and full request
  wall. Never compare those scopes without labeling them.
- Aggregate concurrency throughput is total generated tokens divided by the
  concurrent group wall. Per-sequence throughput is aggregate divided by live
  rows only when every row generates the same number of tokens.
- `Peak GiB` names the allocator or whole-card scope in the run record.
- Bold ratios in retained speculative rows identify speedup against the true
  same-protocol AR control. Plain maxima in diagnostic cross-engine tables are
  not promoted as wins.
