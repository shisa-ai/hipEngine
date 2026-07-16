# hipEngine Topline Benchmarks

Last reviewed: **2026-07-17**

Latest retained hipEngine revisions in this scoreboard:
`7ab8eb3b60de772f61b1b2d55785e7872586abcd` for real GGUF graph replay
accounting and `b49bc0ef8dd74678e7477541f0e455cf73d11b67` for the Prometheus
surface in the correctness-retained, live-observable gfx1100 GGUF OpenAI
continuous-membership closure (D4 lifecycle source `f03957cc`),
`666a72dbac0af1d27661860e7f09facb77dd1299` for the focused post-sweep gfx1100
GGUF router convergence gates, `d59d7cf0c3532f4fd7a5601a26805c85698f1db8`
for the retained gfx1100 GGUF direct native-c4 graph-scaling closure (graph
runtime `6f7851f3`, clean profiler `a05c560b`, and category provenance
`799d29b9`), `52b0db25a20607f51e08abc89c43d200d2fe0ea5` for the retained
native-c8 profiler/scaling packet (correctness runtime `bbe6deb0`), and
`61a27d7279549843bb3fb0464cb8b120689b9ff1` for the current gfx1151 GGUF
refresh. The gfx1151 production refresh is retained through 64K; repeated 128K
is explicitly blocked by the residual gfx11 scheduler lifecycle failure rather
than carrying a stale number.

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

The exact follow-on is also rejected in
[`2026-07-13-gfx1151-gguf-prefill-gpf2c-ordered-resident-rejected.json`](results/2026-07-13-gfx1151-gguf-prefill-gpf2c-ordered-resident-rejected.json).
GPF-2C keeps four state rows per wave lane in registers while preserving every
ordered shuffle and FMA site. Plain/segment output and FP32 state stay byte-
exact and 46 focused tests pass, but 512/1K/4K prefill is only
**368.702/383.292/354.672 tok/s**, **12.98%/14.58%/13.50% below** the clean
fused control. Decode is within -0.31%..-0.24%. A cache-clean trace attributes
**928.006 ms / 30** to recurrence, 16.86% slower than fused. Register residency
therefore fixes global state traffic but not the ordered cross-lane cost;
`auto` remains fused.

The next exact schedule passes its focused candidate gate in
[`2026-07-13-gfx1151-gguf-prefill-gpf2d-lds32-focus-candidate.json`](results/2026-07-13-gfx1151-gguf-prefill-gpf2d-lds32-focus-candidate.json).
GPF-2D assigns one scalar-exact value column to each thread and retains its
128-row FP32 state in a 16 KiB LDS tile across the token loop. Plain/segment
tile32/tile64 fixtures are byte-exact. After rejecting a forced-unroll build
that spilled 1,880 bytes/thread, the rolled LDS32 kernel uses 64 VGPR and zero
scratch; its cache-clean 512 recurrence is **221.873 ms / 30**, 72.06% below
fused. Focused 512/1K/4K prefill improves **423.708/448.694/410.023 ->
753.489/799.844/686.840 tok/s** (**+77.83%/+78.26%/+67.51%**) with decode
−0.10%/+0.03%/+0.03%. This dirty-tree focus artifact is not a retained topline
or default change. The subsequent clean six-case
[`exact matrix`](results/2026-07-13-gfx1151-gguf-prefill-gpf2d-exact-matrix.json)
is byte-identical for sampled tokens, hidden seed, all resident Conv/GDN state,
and the required layer outputs. A clean balanced
[`A/B`](results/2026-07-13-gfx1151-gguf-prefill-gpf2d-balanced-ab.json) moves
512 **420.959 -> 753.891 tok/s (1.791x)** and 4K **408.359 -> 687.831 tok/s
(1.684x)** with exact timed IDs. The clean ten-prompt
[`trajectory/decode gate`](results/2026-07-13-gfx1151-gguf-prefill-gpf2d-trajectory-decode-gate.json)
passes all **250/250** checked logits with `KL=0`, preserves every timed token,
and moves weighted decode **53.4295 -> 53.4416 tok/s (+0.023%)**. GPF-2D is
now the gfx1151-scoped automatic route; gfx1100 remains fused. Its clean
[`six-shape max-context stress gate`](results/2026-07-13-gfx1151-gguf-prefill-gpf2d-default-six-shape.json)
records **751.993/804.420/688.545/589.866/504.730/372.892 tok/s** prefill with
stable five-run IDs and completes in **66.66 minutes**. That one 128K-sized
session is default/long-context validation, not the canonical right-sized
short-shape memory rollup.

The next selected-MoE schedule is promoted from
[`2026-07-13-gfx1151-gguf-prefill-gpf3a-q4t16-shared-x-replay.json`](results/2026-07-13-gfx1151-gguf-prefill-gpf3a-q4t16-shared-x-replay.json).
GPF-3A shares one activation fragment across the existing two independent
Q4T16 WMMA output halves while preserving each accumulator's K/WMMA order.
BF16/FP16 fixture bytes are exact; the tiny trace is **44.725 -> 33.343 us
(-25.45%)**, and identical real 40-layer routing moves Q4 gate/up
**114.633 -> 97.082 ms (-15.31%)**. Its clean balanced
[`full-model gate`](results/2026-07-13-gfx1151-gguf-prefill-gpf3a-full-model-ab.json)
moves 512/1K/4K prefill **747.764/804.150/687.676 ->
771.027/823.624/701.042 tok/s** (**+3.11%/+2.42%/+1.94%**). All three full
logit vectors are byte-exact, every 128-step measured decode trajectory
matches, and aggregate decode wall is **7527.985 -> 7527.750 ms (-0.0031%)**.
The gfx1151 backend capability now selects shared-X automatically; gfx1100
remains on baseline pending its own transfer gate. A clean selector-unset
[`focus confirmation`](results/2026-07-13-gfx1151-gguf-prefill-gpf3a-default-focus.json)
at promoted `431fe1e4` reproduces **774.653/823.149/701.389 tok/s** prefill and
stable IDs. It uses four measurements in one max-4K session, so it confirms
routing/performance; the later right-sized 1+3 publication rollup supersedes
it for the public throughput and memory rows.

The next exact GDN refinement is promoted on gfx1151. GPF-2E removes
prompt-sized raw Q/K/V materialization and computes one Q/K norm per shared K
head, then reads canonical `conv_out` from the scalar-exact LDS32 recurrence.
The clean [`six-case matrix`](results/2026-07-13-gfx1151-gguf-prefill-gpf2e-exact-matrix.json)
matches fused sampled tokens, FP32 hidden seeds, all resident Conv/GDN state,
and required layer outputs. Its
[`balanced A/B`](results/2026-07-13-gfx1151-gguf-prefill-gpf2e-balanced-ab.json)
moves current-default 512/1K/4K prefill
**776.428/825.319/700.824 -> 823.093/889.209/744.577 tok/s**
(**+6.01%/+7.74%/+6.24%**). The
[`natural/decode gate`](results/2026-07-13-gfx1151-gguf-prefill-gpf2e-trajectory-decode-gate.json)
passes 250/250 exact logits and every timed trajectory; weighted decode is
**53.3282 -> 53.3684 tok/s (+0.075%)**. gfx1151 `auto` now uses direct-conv;
gfx1100 remains fused pending transfer evidence. A clean selector-unset
[`focus confirmation`](results/2026-07-13-gfx1151-gguf-prefill-gpf2e-default-focus.json)
reproduces **821.755/897.160/750.896 tok/s** with stable IDs. The right-sized
publication sweep remains.

The original explicit screen remains in
[`2026-07-13-gfx1151-gguf-prefill-gpf2e-direct-conv-screen.json`](results/2026-07-13-gfx1151-gguf-prefill-gpf2e-direct-conv-screen.json).

The dense Q8T16 follow-on is now retained on gfx1151 through 64K. GPF-5A
pairs two production-order 32-column waves and shares one activation tile in
1 KiB LDS. Tail fixtures and 512/4K full-model state are byte-exact; the clean
focus gate improves 512 by **8.35%** and stable 4K by **2.54%**. The automatic
right-sized 1+3 sweep refreshes 512/1K/4K/32K/64K to
**889.904/919.598/762.940/648.948/546.296 tok/s**, all with three token `9707`
IDs and unchanged memory. A stable same-commit 128K gate rejects two-wave
there (**382.041 vs 392.219 tok/s, -2.59%**), so final package policy restores
the production wrapper above 65,536 prompt tokens. The unchanged accepted
**387.334 tok/s** 128K row carries forward: a final scoped retry completed one
**385.474 tok/s** measurement before reproducing the separately documented
later-pass lifecycle stall, which is not enough to replace the accepted 1+3
row. Evidence:
[`2026-07-14-gfx1151-gguf-prefill-gpf5a-right-sized-3run.json`](results/2026-07-14-gfx1151-gguf-prefill-gpf5a-right-sized-3run.json).

LCP-2A further promotes the exact GDN route on gfx1151. It instantiates the
same rolled scalar recurrence with compiler-cacheable LDS state accesses while
keeping the volatile GPF-2E symbol as rollback. At clean detached `53928aaf`,
the six-case state matrix and all **250/250** natural transitions are byte-
exact. One warmup plus four balanced repetitions moves 512/1K/4K prefill
**900.814/940.736/941.462 -> 1213.912/1285.266/1285.888 tok/s**
(**+34.76%/+36.63%/+36.58%**); every pair and timed ID matches. Weighted
decode is **53.348 -> 53.359 tok/s (+0.021%)**. The named kernel uses 32 VGPR,
16 KiB LDS, and zero scratch versus 64 VGPR for GPF-2E. gfx1151 `auto` uses
LCP-2A; gfx1100 remains fused. It is included in the current clean 512-64K
production refresh. Evidence:
[`2026-07-14-gfx1151-gguf-gdn-lcp2a-clean-promotion.json`](results/2026-07-14-gfx1151-gguf-gdn-lcp2a-clean-promotion.json).

The follow-up LCP-M2 device-metadata path is promoted only through 4K. Clean
512/1K/4K automatic-vs-explicit state is **83/83** exact and balanced prefill
improves **+1.56%/+0.90%/+0.53%**; longer prompts retain synchronous metadata.
That scoped fallback is not the remaining 128K trigger: the final current
production run and an explicit metadata-off/router-rollback control both
complete one warmup then enter the same low-power measured-pass-1 stall. The
current 512-64K refresh is retained, while 128K is withheld from the topline:
[`2026-07-15-gfx1151-gguf-production-refresh-512-64k-128k-blocked.json`](results/2026-07-15-gfx1151-gguf-production-refresh-512-64k-128k-blocked.json).
A matched user-space-stack follow-up does not clear the blocker. HIP 7.13
completes two full warmup+3 gates at **509.659/499.895 tok/s** with all six IDs
`9707`, but a post-HIP-7.15 third gate stalls after one measured pass. HIP 7.15
stalls in both controls. All persistent stalls show 100%/2.9 GHz at only
**42-48 W** with no kernel-journal fault. Therefore HIP 7.13 is not a safe
workaround and no cross-stack 128K number is published:
[`2026-07-15-gfx1151-128k-hip713-vs-715-lifecycle.json`](results/2026-07-15-gfx1151-128k-hip713-vs-715-lifecycle.json).

A persistent same-stream flight recorder now narrows one clean HIP 7.15,
one-queue measured-pass-1 failure. The warmup completes at **503.876 prefill /
27.970 decode tok/s**; the next prefill then remains at **100% / 2.9 GHz** and
median **49 W** for 1,436 seconds through the process bound. A retired chunk
marker proves all work through token 28,672 completed, while the host reaches
the layer-11 full-attention checkpoint in chunk `[28672,32768)` and advances no
further. Because that source records layer entry before synchronous chunk
metadata and the layer call, the safe unresolved window is layer-10
linear-attention retirement, layer-11 metadata, or layer-11 full-attention/MoE
work—not proof that one named kernel launched or failed. Kernel logs remain
clean and `amdgpu_fence_info` exposes no mismatch but still cannot see KFD user
queues. The capture predates merged request/chunk metadata reuse, so the merged
scheduler is a distinct lifecycle experiment rather than an inferred fix:
[`2026-07-16-gfx1151-128k-prefill-flight-recorder-stall.json`](results/2026-07-16-gfx1151-128k-prefill-flight-recorder-stall.json).

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
| Radeon Pro W7900 + Radeon RX 7900 XTX, gfx1100 | PARO BF16/INT8 KV context capacity and fidelity | 2026-07-13 | clean profile-aware BF16 frontier `5a49b16d`; clean INT8 capacity `d6504544`; clean functional check `2743798f`; clean external-format screen `d0b56364`; current Qwen3.6 packed model fingerprint retained | **Current capacity / correctness outcome**: on the physical 24 GB XTX, the automatic all-768 low-memory prefill profile makes **208 Ki BF16 the recommended safe cap** at **23.623 GiB whole-device peak / 0.361 GiB free**. **220 Ki physically completes** at 23.908 GiB but leaves only **0.076 GiB (~78 MiB)** and is edge-only; a 232 Ki low-profile screen exceeds capacity. Compact 256K INT8 fits at 22.971 GiB tracked but remains unsupported. External-format S1 lowers mean KL to **0.13342**, but the winning Hadamard group32 row rejects 4K/16 at **0.15512 KL** despite **94.12% top-1**. | Current diagnostic table | Rerun after chunk policy, model/runtime, or allocator changes; do not promote 220 Ki without more margin, and require matched-context plus broader task quality before INT8 support. |
| Radeon Pro W7900, gfx1100 | llama.cpp Q8_0 KV protocol/arithmetic isolation | 2026-07-13 | clean harness `a344d32a`; llama.cpp HIP build 9648 / `1ebf790cd`; exact library/model hashes retained; external instrumentation tree disclosed dirty | **Repeated-token pass superseded as representative quality evidence**: native Q8_0/F16 at 4K/16 is **0.000006 KL / 100% top-1** on repeated token 9707 but **0.075654/1.26009 mean/max KL / 94.12% top-1** on exact mixed `mixed_v1`, failing the KL gate; an exact rerun reproduces every row. Mixed K-only and V-only Q8 reach **0.09668** and **0.24322** mean KL, while full Q8 improves through non-additive K/V interaction. The old 128K repeated row remains a saturation control, not broad fidelity evidence. No performance claim. | Current diagnostic table | Require multiple mixed/natural prompt families after cache arithmetic, format, model/build, or protocol changes; do not promote from repeated-token rows. |
| Radeon Pro W7900 + Radeon RX 7900 XTX, gfx1100 | Native GGUF/PARO tail-four Hadamard-group32 mixed KV | 2026-07-15 | clean GGUF closure `c971262f`; therock HIP 7.15; exact Q4_K_M and prompt-suite identities; prior PARO/XTX outcome retained separately | **Quality-safe GGUF explicit diagnostic; no default promotion**: clean GGUF passes all 11 prompts at 512/8 and 4K/16 (**0.0001369/0.009926 mean/max KL, 99.47% aggregate and 94.12% minimum-prompt top-1** at 4K) plus bounded `mixed_v1` 128K/16. Persistent 128K K/V drops **2,689,597,440 -> 2,185,297,920 bytes (-18.75%)** with no persistent BF16 shadow, but production 4K prefill/decode regress **0.67%/0.75%**, 128K decode regresses **3.82%**, and a **1.002 GiB** prefill transient raises allocator high water **24.168 -> 24.700 GiB** despite lowering live owned memory by **0.470 GiB**. Prior PARO quality and 256 Ki capacity blockers remain. Explicit-only; unsupported/default status unchanged. [`clean GGUF gate`](results/2026-07-15-gfx1100-gguf-tail4-hadamard-clean-gate.json) · [`prior split outcome`](results/2026-07-14-gfx1100-native-tail4-hadamard-kv-outcome.json). | Diagnostic link only | Remove the inferred four-layer BF16 prefill transient and optimize long-context group32 attention, then repeat the clean GGUF gate; PARO requires its own quality-safe layout. |
| Radeon Pro W7900, gfx1100 | Qwen3.6 35B model sweep | 2026-07-16 | clean GGUF `28b37356` on therock HIP 7.15; retained PARO `8116c453`; llama.cpp HIP `1ebf790cd` build 9648; Vulkan `263cc04a5` build 9600 | **Accepted current four-column topline**: the GGUF column is the final right-sized 1+3 defaults-only refresh; PARO and llama.cpp columns retain their clean July 12 protocols. All six GGUF shapes have clean provenance, finite/stable IDs, exact Q4_K_M identity, and <=0.658%/0.223% prefill/decode stdev over median. | Yes | Rerun after PARO/GGUF measured paths, graph policy, model, compiler/runtime, llama.cpp builds, or W7900 clock policy changes. |
| Radeon Pro W7900, gfx1100 | GGUF Q4_K_M direct native-c1/c2/c4/c8 graph decode + observable OpenAI continuous-membership closure | 2026-07-17 | clean native-c4 graph/equality/profiler/scaling `6f7851f3`/`a05c560b`/`d59d7cf0`, category `799d29b9`, server lifecycle/metrics/accounting `f03957cc`/`b49bc0ef`/`7ab8eb3b`, native-c8 correctness `bbe6deb0`, native-c8 profiler/scaling `52b0db25`; TheRock HIP 7.15; exact Q4_K_M/prompt fingerprints; BF16 KV; cached builds | **Retained direct native-c8 model-step scaling and correctness-only OpenAI continuous membership**: eager/graph p512/c8/d128 each pass **41,280/41,280** hidden comparisons; ragged, masked c8→c1, and non-edge live cancellation keep tokens/state/KV exact. The c8 trace is **748 packed-native / 0 row-local / 0 copies**, including exact `6+2` Q6 LM-head lowering. Clean same-session c1/c2/c4/native-c8/chunked-c8/serial-c4 is **85.469/127.427/184.575/246.872/183.020/84.738 aggregate tok/s**. One physical c8 is **2.888x c1**, **1.349x c4+c4 (+34.89%)**, and **2.913x the serial-c4 rate**, with **30.859 per-request tok/s**; the lower per-request rate and higher **32.414/32.749 ms** model-step ITL are explicit. D4/D5 separately prove exact live membership, bounded SSE lifecycle, complete metrics, and graph ownership but add no server throughput/TTFT/ITL claim. [`B4`](results/2026-07-16-gfx1100-gguf-concurrency-b4-category-lifecycle.json) · [`C4`](results/2026-07-16-gfx1100-gguf-concurrency-c4-native-graph-scaling-closure.json) · [`D4`](results/2026-07-16-gfx1100-gguf-concurrency-d4-openai-streaming-closure.json) · [`D5`](results/2026-07-17-gfx1100-gguf-concurrency-d5-live-observability-closure.json) · [`E2 correctness`](results/2026-07-17-gfx1100-gguf-concurrency-e2-native-c8-correctness.json) · [`E2 scaling`](results/2026-07-17-gfx1100-gguf-concurrency-e2-native-c8-scaling-closure.json). | Yes for direct model-step throughput; D4/D5 are correctness-only | Rerun after packed graph/model math, server lifecycle/metrics, prompt/model, compiler/runtime, or device policy changes; a server performance row still requires the F1 burst/live-admission timing protocol. |
| Radeon Pro W7900, gfx1100 | GGUF final architecture-local prefill/decode/memory optimization | 2026-07-16 | clean right-sized rollup `28b37356`; therock HIP 7.15; exact Q4_K_M fingerprint; selector-unset BF16-KV package defaults | **Accepted final gfx1100 GGUF route**: six-shape prefill is **2716.648/3052.541/2953.101/2078.038/1559.878/1037.378 tok/s**, beating llama.cpp HIP by **12.62-30.95%** everywhere and Vulkan from 512-64K; graph decode is **92.833/98.148/100.522/88.240/76.691/62.669 tok/s**, ahead of llama.cpp HIP everywhere and closest to Vulkan at 4K (**-2.47%**). Tracked memory is within **-0.378 to +0.079 GiB** of llama.cpp HIP whole-device readings. All 18 IDs are exact. [`artifact`](results/2026-07-16-gfx1100-gguf-final-optimization-sweep.json). | Yes | Rerun after model/runtime/default-policy, compiler/runtime, or reference-engine changes; decode-to-Vulkan and 128K Vulkan prefill are the concrete residuals. |
| Radeon Pro W7900, gfx1100 | GGUF pp512 request-scoped metadata reuse | 2026-07-15 | clean retained scheduler `e03e5a34`; matched HIP API/kernel traces around the identical source change; system HIP 7.2.53211; exact Q4_K_M fingerprint retained | **Retained scheduler / diagnostic GPF-9C row**: exactly **240 synchronous copies** are removed, matched queue idle falls **27.956 -> 15.163 ms (-45.76%)**, and clean `chain_peer_wave32` pp512 improves **2210.729 -> 2292.186 tok/s (+3.68%)** with stable IDs, unchanged **22.995 GiB** peak, and decode +0.51%. 4K is within -0.44%, but 512 remains **4.98% below** the frozen llama.cpp HIP floor, so exact direct-LDS32 remains production. [`artifact`](results/2026-07-15-gfx1100-gguf-prefill-chunk-metadata-reuse.json). | Diagnostic link only; scheduler code retained | Superseded for the next queue boundary by the compact-WMMA no-read row below; retain as the isolated 240-copy attribution. |
| Radeon Pro W7900, gfx1100 | GGUF pp512 compact-WMMA tight no-read | 2026-07-15 | clean retained gfx1100 default `31c9cdc5`; matched HIP API/kernel traces against `e03e5a34`; system HIP 7.2.53211; exact Q4_K_M fingerprint retained | **Retained gfx1100 scheduler default / diagnostic GPF-9C row**: the tight routing-independent tile bound removes the remaining **40 synchronous D2H copies**, cuts matched queue idle **15.163 -> 11.634 ms (-23.27%)**, and improves clean `chain_peer_wave32` pp512 **2292.186 -> 2334.451 tok/s (+1.84%)** with stable IDs, unchanged **22.995 GiB** peak, and decode within -0.053%. 4K improves +0.70%; pp512 remains **3.23% below** the frozen llama.cpp HIP floor, so exact direct-LDS32 remains production. gfx1151 stays scalar pending its independent gate. [`artifact`](results/2026-07-15-gfx1100-gguf-compact-wmma-tight-no-read.json). | Diagnostic link only; scheduler code retained | Superseded as the final pp512 residual by the no-scratch Conv row below; retain as the isolated 40-copy attribution. |
| Radeon Pro W7900, gfx1100 | GGUF normal-prefill Conv no-scratch exact math | 2026-07-15 | clean retained default `683ddab6`; clean post-profile/floor `c85c2880`; system HIP 7.2.53211; exact Q4_K_M fingerprint retained | **Retained exact default; residual closed**: explicit sequential `v_mul_f32_e32`/`v_add_f32_e32` removes **20 private bytes/thread**, cuts cached pp512 Conv body **8.496 -> 1.894 ms / 30 (-77.71%)**, and improves clean production 512/4K prefill **+1.44%/+1.86%** with exact state/IDs. The clean follow-up puts production exact / peer / llama.cpp kernels at **369.285/203.808/203.301 ms** and GDN at **199.030/20.840/16.522 ms**: the shipped gap is exact GDN. Peer improves clean 512/4K **2334.451/2519.871 -> 2385.677/2585.343 tok/s (+2.19%/+2.60%)**, but 512 remains **1.104% below** its floor, so production is unchanged. [`win`](results/2026-07-15-gfx1100-gguf-conv-no-scratch.json) · [`residual`](results/2026-07-15-gfx1100-gguf-post-conv-residual-attribution.json). | Retained exact-kernel win; peer conclusion superseded by LCP-5A below | Preserve as the post-Conv attribution baseline; the next row closes the peer promotion boundary. |
| Radeon Pro W7900, gfx1100 | GGUF LCP-5A spill-free T16 selected prefill and peer-GDN promotion | 2026-07-15 | clean retained default `487e658c`; system HIP 7.2.53211; exact Q4_K_M/prompt-suite identities retained | **Prefill parity gap closed; gfx1100 default promoted**: rolling only the outer T16 selected-Q4/Q5/Q6 loops removes Q5's 176 private bytes/75 spills, cuts pp512 Q5 **51.009 -> 29.544 ms (-42.08%)**, and moves complete peer kernels/span to **184.513/194.886 ms**, faster than llama.cpp HIP's **203.301/212.236 ms**. The clean 18-prompt gate passes at **0.041737 KL / 445/450 top-1 / -0.103% decode wall**. Selector-unset 512/4K prefill is **2588.231/2757.752 tok/s**, **7.29%/22.29% above** the frozen floors, with stable IDs and **21.670 GiB** tracked peak. [`artifact`](results/2026-07-15-gfx1100-gguf-prefill-lcp5a-spill-free-peer-promotion.json). | Retained default; included in the final `28b37356` six-shape publication above | Rerun after selected-prefill compiler/schedule, peer GDN math, liveness allocation, model, or ROCm changes; keep explicit exact direct-LDS32 rollback for one release. |
| Radeon Pro W7900, gfx1100 | GGUF exact F32-weight cooperative c1 router | 2026-07-14 | clean hipEngine `4c743994` plus persistent-counter default `0ec2a813`; TheRock HIP 7.15; exact Q4_K_M fingerprint retained | **Retained scoped default**: the cooperative fold first improves its complete leaf **17.845 -> 14.666 us (-17.81%)** and clean 4K graph decode **97.234 -> 98.273 tok/s (+1.07%)**. The self-resetting counter then removes 40 reset nodes/token, improves the fused leaf **14.667 -> 10.444 us (-28.79%)**, and cleanly improves 4K graph decode **98.812 -> 100.446 tok/s (+1.65%)**. Every router output bit and all measured IDs/final values are exact; the counters add only eight tracked bytes. | Retained defaults; included in the final `28b37356` six-shape table | Remove temporary cooperative/persistent rollback flags after one release window; rerun after router math, model, compiler/runtime, or graph-policy changes. |
| Radeon Pro W7900, gfx1100 | PARO gfx1151 optimization transfer gate | 2026-07-12 | clean detached hipEngine `255e5aca`; TheRock HIP `7.15.0-0000000`; exact PARO model fingerprint retained | **Retained scoped-default validation / negative chunk decision**: the balanced global-isolation screen is exact at 512/1K/4K. Its 4K/4096-query leg directly validates the merged scoped default with total wall **-0.562%**; 512/1K used 256-query isolation that the final policy intentionally excludes. The gfx1151 linear/MoE-256 profile is rejected at **-7.72%/-8.78%/-6.40% prefill**. | Linked, not a new topline | Rerun after AOTriton/ROCr stream scheduling, PARO chunks, compiler/runtime, or gfx1100 clock policy changes. |
| Radeon Pro W7900, gfx1100 | GGUF graph AR, exact/default MTP, `llama-compat`, and llama.cpp HIP | 2026-07-12 | clean graph gate `833921ce`, admitted route `ac0adb3f`, clean suites `202bd2f0`; ROCm 7.2.4; exact Q4_K_M/prompt fingerprints; llama.cpp HIP `1ebf790cd` build 9648 | **Current retained AR / corrected MTP economics**: natural24 graph AR is **93.30 tok/s**, exact B3 is **68.50 vs 98.75 AR (0.6936x)**, and accuracy-traded `llama-compat` is **79.70 vs 93.30 AR (0.8542x)**. All 24 repeated-state transitions and all ten natural generated previews/tails are exact. At matched timing boundaries hipEngine AR is **93.30** versus llama.cpp **78.29 tok/s (+19.19%)**. | Yes, qualified | Rerun after graph policy/state, GGUF/MTP route, model/prompt suite, compiler/runtime, or output-horizon changes; keep exact fixed-cycle and natural24 contracts separate. |
| Radeon Pro W7900, gfx1100 | PARO/llama.cpp/vLLM concurrency | 2026-07-07 | hipEngine `b4edca09`; same TheRock stack; vLLM `0.22.1rc1.dev499+g470229c37.d20260613` | **Stale diagnostic**: cross-quant and mixed timing scopes; source artifacts set `performance_claim=false`; measured PARO code predates the July concurrency changes | Diagnostic link only | Rerun one timing scope with exact generated-token accounting across all engines |
| Radeon Pro W7900, gfx1100 | Dense 27B DFlash | 2026-06-11 | hipEngine `9faa731c`; ROCm 7.2; artifact records a dirty tree | **Retained under the recorded DFlash gate**, with legacy dirty-source provenance | Yes, qualified | Refresh on a clean tree before changing the public claim |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Qwen3.6 35B matched four-engine reference | 2026-07-11 | clean hipEngine `d1231ee0`; TheRock HIP `7.13.60980-c76140fa27`; llama.cpp HIP `1ebf790cd` build 9648; Vulkan `6e9007ae6` build 9641 | **Retained comparison reference**: all six shapes passed the then-current four-column gate. The current composite table replaces PARO with the July 12 recovery and hipEngine GGUF with the July 15 production refresh through 64K plus an explicit blocked 128K cell; llama.cpp remains this matched reference. | Yes, reference columns | Rerun llama columns after a build/stack/path change; rerun all four together when a fully matched refresh is required. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | PARO exact c1 prefill recovery | 2026-07-12 | clean control `240c5daf` and candidate `9944e481`; TheRock HIP `7.15.0-0000000`; TuneD accelerator-performance; exact PARO model fingerprint retained | **Retained**: exact linear/MoE 256-row architecture profile improves all six prefill shapes by **14.35%-51.11%**, leaves decode within **-0.25%..+0.26%**, and matches final hidden plus all Conv/GDN/KV state at 512/4K/128K. | Yes, PARO column | Rerun after PARO prefill chunk/staging/math, compiler, model, prompt, or tuned/clock policy changes; validate separately on gfx1100 before transfer. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | PARO 4K-128K AOTriton queue isolation | 2026-07-12 | clean same-commit control/candidate `01e2cec5`; TheRock HIP `7.15.0-0000000`; TuneD accelerator-performance; exact PARO model fingerprint retained | **Retained at 4K/32K/64K/128K**: event-linked isolated AOTriton queue improves matched prefill by **13.32%-23.03%**, leaves decode within **-0.16%..+0.12%**, holds tracked peak unchanged, and matches final hidden plus all 30 Conv/GDN and 10 K/V families at every retained shape. The 1K 256-query negative control does not enter isolation and is unchanged. | Yes, PARO column | Validate separately on gfx1100 before transfer; 512/1K remain on the proven-safe caller-stream route. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF eager token/state oracle | 2026-07-12 | clean detached hipEngine `3ce60e56`; TheRock HIP `7.15.0-0000000`; exact Q4_K_M fingerprint and llama binary hashes retained | **Accepted correctness-only gate**: the repeated external and production token stream matches; four hidden/layer/30-Conv-GDN/10-KV transitions are finite and byte-exact. `performance_claim=false`. | Diagnostic link only | Rerun after eager math/state/KV, model, compiler/runtime, or device changes. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF fused/chain GDN prefill correctness and default selection | 2026-07-11 | correctness at clean tracked `332f01f8`; clean performance worktree `ad773eba`; TheRock HIP `7.13.60980-c76140fa27`; exact Q4_K_M fingerprint retained | **Accepted correctness / retained negative performance decision**: exact chain passes 6/6 state cases but is +5.19%/+6.70% slower in balanced 512/4K walls. Fused remains default. | Diagnostic link only | Rerun after GDN math/scheduler/chunk changes; do not retry unchanged split scheduling. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF GPF-2 register-resident GDN diagnostics | 2026-07-13 | clean tree performance `31d4204d`, clean tree trajectory gate `2670ed04`, exact ordered candidate based at `cf3e8250`; TheRock HIP `7.15.0-0000000`; TuneD accelerator-performance; exact Q4_K_M fingerprint retained | **Both default candidates rejected**: relaxed tree balanced 512/4K prefill improves **422.281 -> 956.765 tok/s (2.266x)** and **410.534 -> 844.847 tok/s (2.058x)**, but only **3/10** natural prompts preserve the complete fused 128-step trajectory. Exact ordered residency preserves byte identity but regresses 512/1K/4K by **12.98%/14.58%/13.50%**. `auto` remains fused. | Diagnostic link only | Test scalar-exact value columns with recurrent state resident in a 32/64-column LDS tile; require exact natural trajectories before any default/topline change. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF GPF-2D scalar-exact LDS-resident GDN | 2026-07-13 | clean candidate `a6f389d2`, promoted default `5f082783`; TheRock HIP `7.15.0-0000000`; TuneD accelerator-performance; exact Q4_K_M fingerprint retained | **Promoted gfx1151-scoped default**: six-case matrix and 250/250 natural logits are exact; balanced 512/4K prefill improves **420.959 -> 753.891 tok/s (1.791x)** and **408.359 -> 687.831 tok/s (1.684x)**; decode is +0.023%. The clean automatic max-context stress gate records **751.993/804.420/688.545/589.866/504.730/372.892 tok/s** across six shapes with stable IDs. | Superseded within current GGUF rollup | Keep gfx1100 fused until an independent transfer gate. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF GPF-3A Q4T16 shared-activation prefill | 2026-07-13 | clean A/B `95d484df`, clean automatic confirmation `431fe1e4`; TheRock HIP `7.15.0-0000000`; TuneD accelerator-performance; exact Q4_K_M fingerprint retained | **Promoted gfx1151-scoped default**: BF16/FP16 fixtures and 248,320 full-model logits/shape are byte-exact; balanced 512/1K/4K prefill improves **747.764/804.150/687.676 -> 771.027/823.624/701.042 tok/s (+3.11%/+2.42%/+1.94%)**; every measured 128-step trajectory matches and aggregate decode is -0.0031%. Selector-unset focus medians reproduce **774.653/823.149/701.389 tok/s** with stable IDs. | Included in current GGUF rollup | Keep gfx1100 baseline until an independent transfer gate. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF GPF-2E compact-scale/direct-conv GDN and right-sized rollup | 2026-07-13 | clean exact matrix `c3a065ee`, balanced A/B `ffbcc4d9`, trajectory/decode `5501aeb9`, automatic focus `b8949477`, clean measured sweep `28b45d38`; TheRock HIP `7.15.0-0000000`; TuneD accelerator-performance; exact Q4_K_M fingerprint retained | **Promoted and published on gfx1151**: clean 512/1K/4K A/B is +6.01%/+7.74%/+6.24%; the final 1+3 right-sized prefill row is **819.641/893.266/752.308/640.096/540.850/387.334 tok/s** with <=0.132% stdev/median. Six-case state and 250/250 natural logits are exact. The log-recovered 128K row discloses that the interrupted process did not serialize IDs and links those stronger independent gates. | Superseded by LCP-1/LCP-D1 | Keep gfx1100 fused; investigate later-pass 128K lifecycle no-progress separately from the calibrated performance protocol. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF GPF-5A two-wave dense Q8T16 prefill | 2026-07-14 | clean candidate `4a1fff53`, clean promoted sweep `e9baf563`, final scoped policy `6418b278`; TheRock HIP `7.15.0-0000000`; TuneD accelerator-performance; exact Q4_K_M fingerprint retained | **Promoted through 64K and published**: the clean kernel is byte-exact, uses 80 VGPR/1 KiB LDS/zero scratch, and improves the prior right-sized 512/1K/4K/32K/64K prefill row to **889.904/919.598/762.940/648.948/546.296 tok/s (+1.01% to +8.57%)** with unchanged memory. Stable same-commit 128K rejects two-wave at **-2.59%**, so package policy restores production above 65,536 tokens and carries forward the accepted **387.334 tok/s** row. | Superseded by LCP-1/LCP-D1 | Keep the env rollback and 64K ceiling for one release; validate independently on gfx1100; treat later-pass 128K no-progress as lifecycle diagnosis, not extra timing repetitions. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF LCP-1 tiled convolution and LCP-D1 long-split decode reduction | 2026-07-14 | clean focus/promotion `3ff8e2d7`/`631498dd`, clean reducer `71e61524`, final six-shape sweep `71e61524`; TheRock HIP `7.15.0-0000000`; TuneD accelerator-performance; exact Q4_K_M fingerprint retained | **Promoted and published on gfx1151**: LCP-1's clean 4K body falls **954.134 -> 49.790 ms** and its 512/4K focus is **+1.73%/+22.91%** with 82/82 exact state parts. LCP-D1 cuts the clean 128K reducer **234.714 -> 196.466 us/call (-16.30%)**. Final right-sized prefill is **906.979/929.724/946.366/778.371/636.330/433.811 tok/s** and graph decode is **49.061/51.569/52.432/43.543/37.562/28.047 tok/s**; all 18 IDs are exact, memory unchanged, and variance <=0.140%. | Superseded by LCP-2A prefill promotion | Keep LCP-1's production fallback for one release and validate gfx1100 independently; continue decode only from the measured grouped-GQA context or dense-Q8 residual. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF LCP-2A compiler-cacheable exact GDN | 2026-07-14 | clean detached candidate `53928aaf`; TheRock HIP `7.15.0-0000000`; TuneD accelerator-performance; exact Q4_K_M fingerprint retained | **Promoted on gfx1151 and included in the current 512-64K production refresh**: six-case state and 250/250 natural transitions are exact. Balanced 512/1K/4K prefill improves **900.814/940.736/941.462 -> 1213.912/1285.266/1285.888 tok/s (+34.76%/+36.63%/+36.58%)**; decode is +0.021%. The kernel uses 32 VGPR/16 KiB LDS/zero scratch. | Superseded within targeted prefill gate by LCP-3 | Keep volatile GPF-2E rollback for one release; validate gfx1100 independently. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF LCP-3 four-wave dense Q8T16 prefill | 2026-07-15 | clean detached candidate `d34476da`; TheRock HIP `7.15.0-0000000`; TuneD accelerator-performance; exact Q4_K_M fingerprint retained | **Promoted on gfx1151 through 64K and included in the current production refresh**: clean 512/4K full-model capture is **83/83 exact**. Five balanced pairs improve automatic GPF-5A **1214.510 -> 1220.993 tok/s (+0.53%)** and **1269.030 -> 1288.986 tok/s (+1.57%)**; all 20 timed IDs are `9707`. The named kernel uses 128 threads, 80 VGPR, 1 KiB LDS, and zero scratch. | Superseded within targeted prefill gate by LCP-4A | Keep two-wave then production as rollback paths; validate gfx1100 independently. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF LCP-4A exact F32 router launch geometry | 2026-07-15 | clean detached candidate `3ef55ad4`; TheRock HIP `7.15.0-0000000`; TuneD accelerator-performance; exact Q4_K_M fingerprint retained | **Promoted on gfx1151 and included in the current 512-64K production refresh**: clean 512/4K full-model state is **83/83 exact**. Five balanced pairs improve **1218.536 -> 1252.147 tok/s (+2.76%)** and **1290.923 -> 1333.229 tok/s (+3.28%)**. Clean 512/128 graph decode is exact and **48.987 -> 49.021 tok/s (+0.071%)**. Trace confirms 256 threads, 32 VGPR, and zero scratch. | Superseded within targeted prefill gate by LCP-4B | The 4K refresh is complete; it rejects risky logits+top-k fusion in favor of exact select geometry. Validate gfx1100 independently. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | HIP one-hardware-queue lifecycle stabilization | 2026-07-15 | clean current production `4d0aa281`; TheRock HIP `7.15.0-0000000`; MES `0x88`, MES KIQ `0x6f`; TuneD accelerator-performance; exact Q4_K_M fingerprint retained | **Retained as a risk-reducing gfx1151 process default; not lifecycle-safe for repeated 128K**: default ROCm queue policy enters a reproducible first-warmup 128K stall at 100%/2.9 GHz but only 41-43 W, with four identical host stacks in synchronous metadata H2D and no kernel fault. Changing only `GPU_MAX_HW_QUEUES=1` completes 128K warmup+3 at **499.755 warmup** and **500.210/500.873/500.687 tok/s measured**, all IDs `9707`. Clean 512/4K A/B is non-regressive at **+0.35%/+0.46% prefill** and **+0.066%/+0.072% decode**. | Yes, stability/process-policy gate | Preserve explicit `GPU_MAX_HW_QUEUES` overrides; `=4` restores ROCm's documented default. Current production later reproduces the stall under one queue, so 128K remains blocked. Remove after a fixed gfx11 firmware/runtime passes the same 128K gate. Upstream evidence: [initial ROCm#5107 comment](https://github.com/ROCm/ROCm/issues/5107#issuecomment-4976739824) and [follow-up](https://github.com/ROCm/ROCm/issues/5107#issuecomment-4979442043). |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF LCP-M2 stream-ordered contiguous prefill metadata | 2026-07-15 | clean explicit A/B `6131e891`, clean scoped policy `37b39269`; TheRock HIP `7.15.0-0000000`; one HIP hardware queue; TuneD accelerator-performance; exact Q4_K_M fingerprint retained | **Promoted through 4K and included in the current 512-64K production refresh**: full-model 512/1K/4K automatic-vs-explicit state is **83/83 exact**. Five balanced pairs improve prefill **1261.643 -> 1281.323 tok/s (+1.56%)**, **1333.877 -> 1345.928 (+0.90%)**, and **1356.934 -> 1364.103 (+0.53%)**. The explicit 128K one-queue gate completes warmup at only 483.439 tok/s, then re-enters the low-power GPU-active no-progress state on measured pass 1, so automatic policy retains synchronous metadata above 4K. | Yes, scoped prefill promotion gate | Keep `HIPENGINE_GGUF_PREFILL_DEVICE_METADATA=0|1` for rollback/diagnosis; never extend the 4K ceiling without a completed long-context lifecycle gate. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF LCP-4B exact prefill router-select geometry | 2026-07-15 | clean profiles `37b39269`/`89443a1f`, balanced candidate `c10c794c`, clean promoted policy `89443a1f`; TheRock HIP `7.15.0-0000000`; one HIP hardware queue; TuneD accelerator-performance; exact Q4_K_M fingerprint retained | **Promoted on gfx1151 and included in the current 512-64K production refresh**: the fresh 4K profile leaves router select at 12.539 ms / 0.41%, so a launch-only screen replaces risky logits+top-k fusion. 128 threads is **83/83 exact** and improves five-pair 512/4K prefill **1274.062 -> 1278.414 tok/s (+0.34%)** and **1361.337 -> 1366.173 (+0.36%)**. Clean trace cuts the named select family **12.539 -> 3.741 ms (-70.17%)** with 24 VGPR, 512 B LDS, zero scratch. Faster 64 threads is rejected because 4K full-model state is not exact. | Yes, final targeted prefill gate | Keep `HIPENGINE_GGUF_PREFILL_ROUTER_SELECT_THREADS=512` for rollback; gfx1100 stays 512 and decode stays 256. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF current decode closure profile and graph admission | 2026-07-15 | clean current code `89443a1f`; TheRock HIP `7.15.0-0000000`; one HIP hardware queue; TuneD accelerator-performance; exact Q4_K_M fingerprint retained | **Retained diagnostic / no new kernel promotion**: exact 16-step marker profiles confirm 512/4K dense Q8 at **8.560/8.541 ms/token** and 128K attention/dense-Q8 at **17.504/8.555 ms/token**. Grouped-GQA chunk 128 is +2.89% in isolation but changes one BF16 output; chunk 512 is inexact and slower. Dense-Q8 64 threads is 15.8% slower than 128. Current graph replay remains admitted over eager at **+1.00%/+0.86%** on 512/4K 1+3 and **+0.36%** in the bounded 128K confirmation, all IDs exact. | Diagnostic link; current graph rows retained through 64K, 128K topline blocked by prefill lifecycle | Retain chunk-256 LCP-D1 attention, 128-thread Q8, and graph replay. A future decode attempt needs a new exact algorithm/layout, not another launch-only sweep. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF final production refresh and residual 128K lifecycle gate | 2026-07-15 | clean detached `61a27d72`; TheRock HIP `7.15.0-0000000`; automatic one hardware queue; TuneD accelerator-performance; exact Q4_K_M fingerprint retained | **Retained at 512-64K; 128K blocked**: clean right-sized 1+3 prefill is **1294.885/1358.342/1365.720/1034.845/796.083 tok/s** and graph decode is **49.041/51.623/52.422/43.572/37.622 tok/s**. All 15 IDs are `9707`, memory is unchanged, and variance is <=0.187%. Current automatic one-queue, router-512/metadata-off, and SDMA-disabled full 128K gates all complete warmup then enter the low-power measured-pass-1 stall. | Yes at 512-64K only; no current 128K number | Keep one queue as risk reduction, not a lifecycle guarantee. Require fixed gfx11 firmware/kernel or a stronger production-quality workaround before restoring 128K. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF 128K HIP 7.13 versus 7.15 lifecycle diagnostic | 2026-07-15 | clean detached `61a27d72`; unchanged kernel `7.1.3-2-cachyos`, MES `0x88`, MES KIQ `0x6f`; one hardware queue; HIP 7.13/AOTriton 0.11.2 versus HIP 7.15/AOTriton 0.11.1 | **Retained diagnostic; no performance claim**: HIP 7.13 completes two independent warmup+3 gates at **509.659/499.895 tok/s** with all six IDs `9707`, then a third gate reproduces the stall after one measured pass. HIP 7.15 fails both matched controls, one in warmup and one after measured pass 1. Persistent states remain 100%/2.9 GHz at only **42-48 W** with no amdgpu/KFD journal fault. | No; full stacks differ and incomplete legs are not topline eligible | Do not recommend a HIP 7.13 downgrade as a lifecycle fix. The common firmware/kernel scheduler path remains the leading suspect; quantify stack-specific incidence only with a larger fixed-stack campaign. |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF 128K persistent prefill flight-recorder capture | 2026-07-16 | clean detached `d697b971`; TheRock HIP 7.15; kernel `7.1.3-2-cachyos`; MES `0x88` / KIQ `0x6f`; one hardware queue; chunk markers | **Retained diagnostic; no performance claim**: a **503.876/27.970 tok/s** recorder-enabled warmup completes, then measured prefill 1 times out. The last retired marker certifies chunk `[24576,28672)`; the host reaches layer 11 in `[28672,32768)` and stops. Ordering narrows but does not identify the failing dispatch: layer 10 was enqueued, while layer-11 metadata or full-attention/MoE may not have returned. The persistent state is 100%/2.9 GHz at median 49 W for 1,436 seconds with no amdgpu/KFD/MES journal fault. | No; instrumentation-enabled incomplete run | Gate merged request/chunk metadata reuse separately, then use post-metadata/layer markers or KFD tracing to identify the last submitted and retired dispatch without over-reading `amdgpu_fence_info`. |
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

The GGUF column is the clean 2026-07-16 defaults-only right-sized sweep at
`28b37356` on the complete therock HIP 7.15 stack. Each shape uses one discarded
eager warmup plus three measured runs in an independent resident process; every
measurement captures and closes a fresh state-bound decode graph. Package
automatic policy selects peer-wave GDN, spill-free selected prefill, the
persistent cooperative router, and the long-context parallel reducer while KV
remains default BF16. Focused post-sweep transfers now also select the exact
256-thread F32-weight router-logits wrapper and 128-thread bulk router selector
on gfx1100.

Prefill is now
**2716.648/3052.541/2953.101/2078.038/1559.878/1037.378 tok/s**, graph decode is
**92.833/98.148/100.522/88.240/76.691/62.669 tok/s**, and tracked right-sized
memory is **21.228/21.295/21.670/22.234/22.879/24.168 GiB** from 512 through
128K. All 18 final IDs are `9707`; the largest prefill/decode stdev over median
is **0.658%/0.223%**. The six-shape values remain the last clean publication
sweep. A same-session balanced W7900 gate for the newly retained router default
moves focused 512/4K prefill **2689.171 -> 2795.242 (+3.94%)** and **2955.867 ->
3070.905 tok/s (+3.89%)**; graph decode is **-0.022%/+0.159%**, tracked memory
is unchanged, the 4K primitive is bit-exact, and all timed final IDs match. An
incremental 128-thread selector gate on top improves aggregate 512/4K medians
another **+0.32%/+0.81%** (paired medians **+0.30%/+0.12%**), with graph decode
**-0.068%/+0.216%**, unchanged memory, bit-exact selected IDs/routing weights,
and matching final IDs. A direct legacy-512/512 versus final-package stack gate
confirms paired prefill gains of **+3.87%/+4.16%** at 512/4K, graph decode
**+0.11%/+0.07%**, and unchanged memory/IDs. The subsequently retained
stream-ordered metadata path adds aggregate **+0.41%/+2.43%** at 512/4K
(paired **+0.26%/+2.26%**), with non-regressive decode, unchanged memory/IDs,
and an exact metadata primitive. Production peer-wave GDN remains unchanged;
the strict-exact rollback now resolves to nonvolatile direct-LDS32, which moves
volatile-direct 512/4K prefill **+73.01%/+82.46%**, halves VGPR **64 -> 32**,
and preserves byte-exact state, decode, and compact-scratch memory. A final clean
selector-unset confirmation moves the pre-screen 512/4K package baseline
**2699.283/2972.935 -> 2808.249/3173.723 tok/s (+4.04%/+6.75%)**; graph decode
is **-0.26%/+0.24%**, tracked memory unchanged, and all IDs exact.

Relative to the July 14 GGUF table, prefill improves **+35.27% to +118.78%**
and decode improves **+2.24% to +3.46%**. Prefill now beats llama.cpp HIP at
all six shapes by **12.62-30.95%** and beats llama.cpp Vulkan from 512 through
64K by **3.37-17.10%**; only 128K Vulkan prefill remains ahead, by **3.88%**.
Decode beats llama.cpp HIP everywhere by **2.85-26.02%**, while Vulkan remains
ahead by **2.47-13.87%**. The tracked-memory count is within **-0.378 to
+0.079 GiB** of llama.cpp HIP's broader whole-device readings, so memory is at
practical parity but small cross-scope differences are not allocator-efficiency
claims.

Evidence: [`focused convergence confirmation`](results/2026-07-16-gfx1100-gguf-convergence-final-confirmation.json),
[`final optimization sweep`](results/2026-07-16-gfx1100-gguf-final-optimization-sweep.json),
[`256-thread router transfer`](results/2026-07-16-gfx1100-gguf-router-threads256-promotion.json),
[`128-thread router-select transfer`](results/2026-07-16-gfx1100-gguf-router-select-threads128-promotion.json),
[`retained router stack`](results/2026-07-16-gfx1100-gguf-router-stack-promotion.json),
[`device-metadata transfer`](results/2026-07-16-gfx1100-gguf-prefill-device-metadata-promotion.json),
[`nonvolatile exact rollback`](results/2026-07-16-gfx1100-gguf-gdn-nonvolatile-exact-rollback.json),
[`peer-GDN promotion`](results/2026-07-15-gfx1100-gguf-prefill-lcp5a-spill-free-peer-promotion.json),
[`decode attribution`](results/2026-07-15-gfx1100-gguf-decode-lcpd3-attribution.json),
[`LCP-D2 gate`](results/2026-07-14-gfx1100-gguf-decode-lcp-d2-parallel-reduce.json),
[`LCP-M1 memory gate`](results/2026-07-14-gfx1100-gguf-lcp-m1-prefill-scratch-liveness.json),
[`persistent router counter`](results/2026-07-14-gfx1100-gguf-persistent-router-counter.json),
and [`mixed-KV closure`](results/2026-07-15-gfx1100-gguf-tail4-hadamard-clean-gate.json).

PARO remains the clean 2026-07-12 `8116c453` two-warmup/five-measurement row.
llama.cpp HIP/Vulkan remain the matched July 12 Q4_K_M/F16-KV references with
one internal warmup plus five samples per split phase. Every engine uses the
stated graph/eager route and excludes graph capture from steady decode timing.

Bold marks the best raw value in each row. It is descriptive only: PARO is W4
PARO/BF16 KV, while the other columns use the same Q4_K_M GGUF with hipEngine
BF16 KV and llama.cpp F16 KV. Memory scopes also differ.

<!-- BEGIN TOPLINE:W7900_SWEEP -->
#### Prefill tok/s

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **2917.732** | 2716.648 | 2412.320 | 2627.990 |
| 1K/128 | 2995.876 | **3052.541** | 2389.670 | 2631.750 |
| 4K/128 | 2943.038 | **2953.101** | 2255.080 | 2521.770 |
| 32K/128 | **2108.868** | 2078.038 | 1667.640 | 1943.920 |
| 64K/128 | **1584.131** | 1559.878 | 1291.820 | 1414.470 |
| 128K/128 | 1056.252 | 1037.378 | 891.949 | **1079.280** |

#### Decode tok/s

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **115.599** | 92.833 | 80.756 | 107.786 |
| 1K/128 | 103.238 | 98.148 | 80.805 | **107.555** |
| 4K/128 | **105.943** | 100.522 | 79.768 | 103.066 |
| 32K/128 | **92.438** | 88.240 | 74.304 | 91.835 |
| 64K/128 | 78.260 | 76.691 | 69.010 | **83.746** |
| 128K/128 | 60.663 | 62.669 | 60.933 | **70.833** |

#### Peak memory GiB

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **18.144** | 21.228 | 21.606 | 21.260 |
| 1K/128 | **18.367** | 21.295 | 21.618 | 21.220 |
| 4K/128 | **19.161** | 21.670 | 21.674 | 21.278 |
| 32K/128 | **19.864** | 22.234 | 22.216 | 21.855 |
| 64K/128 | **20.403** | 22.879 | 22.895 | 22.512 |
| 128K/128 | **22.124** | 24.168 | 24.089 | 23.824 |
<!-- END TOPLINE:W7900_SWEEP -->

hipEngine memory is its tracked allocator high-water; llama.cpp is absolute
whole-device W7900 VRAM used, sampled from DRM sysfs `card1` every 10 ms. The
host's `rocm-smi` card labels use a different numbering scheme; the retained
artifact validates the 48 GiB W7900 device rather than the idle 24 GiB XTX.
Use memory values for within-column context growth, not small cross-column
allocator-efficiency claims.

Artifacts: [current hipEngine GGUF throughput and memory](results/2026-07-16-gfx1100-gguf-final-optimization-sweep.json),
[superseded July 14 hipEngine GGUF sweep](results/2026-07-14-gfx1100-gguf-optimization-right-sized-3run.json),
[LCP-M1 memory gate](results/2026-07-14-gfx1100-gguf-lcp-m1-prefill-scratch-liveness.json),
[July 12 accepted summary](results/2026-07-12-w7900-v030-8116c453-summary.json),
[hipEngine PARO](results/2026-07-12-w7900-v030-8116c453-hipengine-paro-packed-5run.json),
[superseded hipEngine GGUF](results/2026-07-12-w7900-v030-8116c453-hipengine-gguf-q4km-5run.json),
[llama.cpp HIP](results/2026-07-12-w7900-v030-8116c453-llamacpp-hip-q4km-f16kv.json),
[llama.cpp Vulkan](results/2026-07-12-w7900-v030-8116c453-llamacpp-vulkan-q4km-f16kv.json),
and [W7900 GGUF oracle](results/2026-07-12-w7900-v030-gguf-eager-p512-d4.json).

### gfx1151 model throughput

The current GGUF column is the clean 2026-07-15 selector-unset production
refresh at hipEngine `61a27d72`, TheRock HIP 7.15, TuneD
`accelerator-performance`, and the automatic one-hardware-queue gfx1151 policy.
Each accepted shape uses one independent right-sized resident process, one
discarded warmup, and three measured runs with fresh production graph decode.
512 through 64K pass clean provenance, finite logits, exact final IDs, and the
5% variance gate; the largest prefill/decode stdev over median is only
**0.187%/0.049%**. All 15 measured IDs are `9707`, and tracked memory is
unchanged. Those five prefill rows improve **25.11%-46.10%** over the previous
public GGUF column and are the highest raw prefill values in the composite.

Repeated 128K production is **blocked**, not carried forward. The current route
completes warmup at 509.708 tok/s, then enters the same 100%/2.9 GHz, 44-46 W
no-progress state on measured pass 1 despite `GPU_MAX_HW_QUEUES=1`. Explicit
metadata-off/router-512 and `HSA_ENABLE_SDMA=0` full-gate controls reproduce the
failure. Therefore no current hipEngine GGUF 128K throughput or memory number
appears in the topline; first-pass rates remain diagnostics only. llama.cpp
stays the clean July 11 matched reference. PARO retains the July 12 exact
recovery and scoped AOTriton queue-isolation rows. Since PARO uses W4 PARO
rather than Q4_K_M and memory scopes differ, bold values are descriptive raw
leaders rather than same-math allocator claims.

<!-- BEGIN TOPLINE:GFX1151_SWEEP -->
#### Prefill tok/s

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | 1140.101 | **1294.885** | 1061.260 | 1067.770 |
| 1K/128 | 1208.343 | **1358.342** | 1043.230 | 1069.870 |
| 4K/128 | 1089.031 | **1365.720** | 1009.240 | 1016.580 |
| 32K/128 | 906.145 | **1034.845** | 743.547 | 814.923 |
| 64K/128 | 716.775 | **796.083** | 573.611 | 660.974 |
| 128K/128 | 474.641 | — (blocked) | 390.441 | **476.788** |

#### Decode tok/s

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **66.767** | 49.041 | 50.939 | 62.396 |
| 1K/128 | 61.746 | 51.623 | 50.818 | **62.136** |
| 4K/128 | **62.715** | 52.422 | 50.126 | 60.097 |
| 32K/128 | 50.342 | 43.572 | 44.240 | **51.319** |
| 64K/128 | 42.094 | 37.622 | 39.326 | **44.422** |
| 128K/128 | 30.386 | — (blocked) | 32.114 | **34.948** |

#### Peak memory GiB

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **18.039** | 21.478 | 21.375 | 21.551 |
| 1K/128 | **18.051** | 21.710 | 21.387 | 21.501 |
| 4K/128 | **19.026** | 22.995 | 21.444 | 21.507 |
| 32K/128 | **19.729** | 23.559 | 21.987 | 22.191 |
| 64K/128 | **20.403** | 24.203 | 22.666 | 22.627 |
| 128K/128 | **22.124** | — (blocked) | 23.862 | 24.254 |
<!-- END TOPLINE:GFX1151_SWEEP -->

The PARO column is W4 PARO/BF16 KV. The other three columns use the same
Q4_K_M GGUF; hipEngine uses BF16 KV and llama.cpp uses f16 KV. Peak-memory
scopes differ: hipEngine reports its tracked allocator high-water, while
llama.cpp reports absolute whole-device amdgpu GTT used, sampled every 10 ms.
Use memory values for within-column context growth; small cross-column deltas
are not allocator-efficiency claims. hipEngine load and graph capture are
excluded from phase throughput.

Artifacts: [current hipEngine GGUF 512-64K refresh and 128K blocker](results/2026-07-15-gfx1151-gguf-production-refresh-512-64k-128k-blocked.json),
[HIP 7.13 versus 7.15 128K lifecycle diagnostic](results/2026-07-15-gfx1151-128k-hip713-vs-715-lifecycle.json),
[PARO exact prefill recovery](results/2026-07-12-gfx1151-paro-prefill-recovery.json),
[PARO 4K-128K AOTriton queue isolation](results/2026-07-12-gfx1151-paro-aotriton-stream-isolation.json),
[previous GGUF LCP-1/LCP-D1 rollup](results/2026-07-14-gfx1151-gguf-lcp1-lcpd1-right-sized-3run.json),
[queue-stall follow-up](https://github.com/ROCm/ROCm/issues/5107#issuecomment-4979442043),
[accepted July 11 matched summary](results/2026-07-11-gfx1151-readme-refresh-20260711-d1231ee0-summary.json),
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

### gfx1100 PARO context capacity and mixed-KV fidelity, 2026-07-14

**Status: 208 Ki BF16 is the recommended safe cap on a physical 24 GB card;
220 Ki is a validated edge, and neither all-layer INT8 nor native tail-four
mixed KV makes 256K a supported route.** Clean hipEngine `5a49b16d` ran
profile-aware BF16 sweeps on the W7900 and directly on this host's
25,753,026,560-byte (23.984 GiB) RX 7900 XTX. Every retained BF16 row uses the
current Qwen3.6 packed PARO snapshot, repeated token `9707`, 128 decode tokens,
full-run approximately 1 Hz whole-device monitoring, finite-output checks, and
a passing layout audit. Clean `d6504544` supplies the separate compact 256K
all-layer INT8 row; the native mixed-KV diagnostic records exact source hashes
and physical request-scratch probes in the July 14 artifact.

<!-- BEGIN TOPLINE:W7900_MEMORY_CAPACITY -->
| Route / profile | Hardware | Context/decode | Tracked peak | Observed device peak | Device/card margin | Capacity / quality status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| PARO BF16 KV reference | W7900, default chunks | 128K/128 | **22.124 GiB** | 21.107 GiB phase sample | n/a | Reference path |
| PARO BF16 KV, automatic 24 GB low-memory profile | RX 7900 XTX 24 GB | **208 Ki (212,992)/128** | **23.082 GiB** | **23.623 GiB** | **+0.361 GiB** | **Recommended practical safe cap** |
| PARO BF16 KV, automatic 24 GB low-memory profile | RX 7900 XTX 24 GB | 220 Ki (225,280)/128 | 23.369 GiB | **23.908 GiB** | **+0.076 GiB (~78 MiB)** | Physical pass, but **edge only—not safe cap** |
| PARO BF16 KV, default 48 GB-card profile | W7900 | 220 Ki (225,280)/128 | 24.090 GiB | at least 24.832 GiB | at most -0.848 GiB vs 24 GB card | Rejected for this larger-chunk profile |
| PARO INT8 per-token/head KV, FP16 scales | W7900 | 256K/128 | **22.971 GiB** | 21.041 GiB phase sample | +1.029 GiB tracked | **Rejected** by Qwen3.6 matched-context and task gates |
| PARO tail-four Hadamard-group32 mixed KV, BF16-oracle prefill | RX 7900 XTX 24 GB | 256 Ki (262,144)/128 request scratch | **23.469 GiB before failed allocation** | 22.566 GiB after clean OOM | 1.418 GiB free before request scratch; insufficient | **Rejected:** `HIP error 2` OOM and PARO fidelity failure; no segfault |
| PARO tail-four Hadamard-group32 mixed KV, direct-streaming control | RX 7900 XTX 24 GB | 256 Ki (262,144)/128 request scratch | **23.290 GiB** | **23.590 GiB** live sample | **+0.394 GiB** live | Allocation passes, but direct packed prefill is **correctness-rejected** |

The native explicit `tail4_hadamard_group32` layout keeps K/V for
full-attention layers `3,7,11,15,19,23` in BF16 and stores only layers
`27,31,35,39` as Hadamard-group32 INT8 with FP16 scales. At 262,400 retained
rows it uses `4,366,336,000` K/V bytes—**18.75% below BF16**—with no persistent
BF16 shadow. PARO's quality-preserving prefill uses a temporary BF16 oracle;
GGUF's post-quality layout audit reports zero persistent oracle/mirror buffers.
Native PARO still fails 1/11 prompts at 512/8 and 2/11 at 4K/16 (58.82%
worst-prompt top-1), and its 256 Ki quality-preserving request scratch OOMs.

The clean `c971262f` therock-7.15 GGUF-only closure passes all 11 prompts at
512/8 (max KL `0.007455`, top-1 100%) and 4K/16 (mean/max KL
`0.0001369/0.009926`, aggregate/minimum-prompt top-1 `99.47%/94.12%`) plus
bounded `mixed_v1` at 128K/16 (max KL `5.19e-5`, top-1 100%). At 128K,
persistent K/V is `2,185,297,920` bytes versus BF16 `2,689,597,440` bytes and
live owned memory falls `24.168 -> 23.698 GiB`. It still rejects promotion:
production 4K prefill/decode regress `0.67%/0.75%`, one-shot 128K decode
regresses `3.82%`, and production prefill allocates then frees
`1,075,838,976` bytes—byte-exact to four BF16 layer caches—raising allocator
high water `24.168 -> 24.700 GiB`. The transient attribution is inferred from
the exact bytes; it is not a persistent shadow. The policy remains explicit
and non-default. Evidence:
`benchmarks/results/2026-07-15-gfx1100-gguf-tail4-hadamard-clean-gate.json` and
`benchmarks/results/2026-07-14-gfx1100-native-tail4-hadamard-kv-outcome.json`.
<!-- END TOPLINE:W7900_MEMORY_CAPACITY -->

The physical-card result differs from the earlier W7900 220 Ki rejection
because the runtime is card-aware. W7900's 48 GB total selects
`1024/1024/4096/1024/1024` prefill chunks; a 24 GB card automatically selects
all-`768` via `low_memory_full_context_24gb`. The earlier W7900 result remains a
valid rejection of its default larger-chunk profile, but it cannot by itself
predict behavior under the actual low-memory profile.

The clean profile-aware sweep is:

| Hardware / prefill profile | Context | Tracked peak | Full-run device peak | Margin vs physical 24 GB bytes | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| W7900 default `1024/4096` | 176 Ki | 23.033 GiB | 23.779 GiB | +0.205 GiB | Fits byte envelope, limited margin |
| W7900 default `1024/4096` | 184 Ki | 23.226 GiB | 23.971 GiB | +0.013 GiB | Not safe |
| W7900 default `1024/4096` | 200 Ki | 23.610 GiB | 24.356 GiB | -0.371 GiB | Does not model a fitting 24 GB route |
| RX 7900 XTX automatic all-`768` | 176 Ki | 22.315 GiB | 22.857 GiB | +1.127 GiB | Direct physical-card pass |
| RX 7900 XTX automatic all-`768` | **208 Ki** | **23.082 GiB** | **23.623 GiB** | **+0.361 GiB** | **Recommended safe cap** |
| RX 7900 XTX automatic all-`768` | 220 Ki | 23.369 GiB | 23.908 GiB | +0.076 GiB | Direct pass, edge only |
| W7900 manual all-`768` screen | 232 Ki | 23.657 GiB | 24.163 GiB | -0.178 GiB | Rejected without risking physical-card OOM |

For this report, “safe” requires a directly tested point with at least 0.25 GiB
of observed whole-device headroom. Thus **208 Ki (212,992 tokens)** is the
practical cap; ordinary 200K or 200 Ki requests are below it. **220 Ki is the
largest physically validated point**, but only about 78 MiB remains, so it
should not be the configured maximum. The exact mathematical frontier is not
claimed: 209-219 Ki were not tested. No row segfaulted; the 232 Ki physical-card
run was intentionally skipped after its same-profile W7900 screen exceeded the
target byte capacity. Throughput is single-run/concurrent diagnostic data only,
not a performance claim.

The clean 256K/128 INT8 row retains 2,686,976,000 payload bytes plus 20,992,000
FP16 scale bytes across ten full-attention layers. The compact table is
16,793,600 bytes (`4,096 x 1,025` INT32 entries). Tracked peak falls
**25,723,838,504 -> 24,665,296,404 bytes** (**23.957 -> 22.971 GiB**, -0.986
GiB / -4.12%), increasing the 24 GiB margin from 0.043 to 1.029 GiB. One-shot
diagnostic throughput is effectively flat within run variance: prefill
632.837 -> 631.457 tok/s and decode 40.066 -> 40.008 tok/s.

The final BF16-reference-token matched 128K/16 gate is finite and passes the
no-shadow audit, but rejects at mean/max KL **0.85128/4.97382** and **41.18%**
top-1 agreement. This is the intrinsic comparison; the older 128K/128
independent-rollout KL/top-1 headline includes cascade after histories diverge.
Clipping, group16/32/64, K/V mixed formats, selective BF16 layers/heads, and
sink/recent residual windows all failed to clear both gates at 4K within the
reclaimed budget.

The clean `d0b56364` external-format screen adds matched-context top-k evidence
without changing support status. Its fixed 512/8 mixed-prompt S1 run completed
in **28.78 s** including setup (600 s budget):

| Emulated INT8 representation | Mean / max KL | Top-1 | Top-5 / top-10 overlap | 256K bytes / extra vs baseline | S1 decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Current per-token/head baseline | `0.36841 / 1.20200` | `66.67%` | `71.11% / 77.78%` | `2.520 GiB / 0` | Reject anchor |
| **Hadamard group32** | **`0.13342 / 0.45135`** | `77.78%` | `84.44% / 84.44%` | `2.656 / +0.137 GiB` | Lowest mean KL; transfer |
| KIVI-style K-per-channel/V-per-token-group | `0.16667 / 0.60739` | **`88.89%`** | **`86.67% / 88.89%`** | `2.698 / +0.178 GiB` | Better decision fidelity; higher primary KL |
| KVarN-informed eight-pass INT8 + BF16 sink/tail | `0.27125 / 1.25017` | **`88.89%`** | `77.78% / 80.00%` | `2.593 / +0.073 GiB` | Improve baseline only |

Only Hadamard group32 advanced to 4K/16. It passes top-1 at **94.12%** but
rejects mean/max KL at **`0.15512/1.14267`**; top-5/top-10 overlap is
`88.24%/84.71%`. The run stops before native kernels, 128K, or task benchmarks.
This is a representation diagnostic with `performance_claim=false`, not a new
supported cache format.

The same representation set was then isolated on hipEngine GGUF with identical
Q4_K_M weights, BF16 reference, fixed `mixed_v1` prompts, and teacher history.
Unlike PARO, every GGUF row passes the 512/8 screen by a wide margin; plain
symmetric group32 has the lowest mean KL:

| hipEngine GGUF representation | 512/8 mean / max KL | 512/8 top-1 | 4K/16 mean / max KL | 4K/16 top-1 | Transfer |
| --- | ---: | ---: | ---: | ---: | --- |
| Per-token/head max-abs | `0.0001646 / 0.0005551` | `100%` | **`0.12779 / 2.03039`** | `88.24%` | Reject |
| **Plain group32 (Q8_0 storage geometry)** | **`0.0000812 / 0.0003984`** | `100%` | `0.28106 / 4.39924` | `88.24%` | Reject |
| Hadamard group32 | `0.0000974 / 0.0003191` | `100%` | `0.25180 / 4.09533` | **`94.12%`** | Reject KL |
| KIVI-style INT8 | `0.0001753 / 0.0012793` | `100%` | `0.33306 / 5.43878` | `88.24%` | Reject |

The 4K failures are dominated by `decode_3`; Hadamard preserves 16/17 top-1
rows, while the other formats also miss `decode_4`. A reverse candidate-order
run reproduces every per-position KL and candidate top-1 exactly, ruling out
session-reset or ordering contamination. Compact evidence:
[`2026-07-13-w7900-gguf-int8-kv-external-format-screen.json`](results/2026-07-13-w7900-gguf-int8-kv-external-format-screen.json).

The exact native follow-up separates prompt content, Q8 K, Q8 V, and host
reconstruction on the same mixed token IDs:

| Engine / cache arithmetic | Repeated 4K/16 mean KL | Mixed 4K/16 mean / max KL | Mixed top-1 | Verdict |
| --- | ---: | ---: | ---: | --- |
| llama.cpp F16/F16 control | — | `0 / 0` | `100%` | Deterministic control |
| **llama.cpp native Q8_0 K/V** | **`0.00000619`** | **`0.075654 / 1.26009`** | **`94.12%`** | Reject KL |
| llama.cpp native Q8 K / F16 V | — | `0.096682 / 1.56852` | `94.12%` | Reject KL |
| llama.cpp native F16 K / Q8 V | — | `0.243219 / 3.99543` | `94.12%` | Reject KL; largest isolated error |
| hipEngine native per-head INT8 | `0.00000235` | `0.19038 / 2.99555` | `88.24%` | Reject both |

Mixed input raises llama.cpp full-Q8 mean KL 12,227x. F16/F16 is exactly zero,
and exact reruns reproduce both llama.cpp full-Q8 and hipEngine native
per-head results. Q8_0 has no F16 shadow: it writes FP32 K/V to INT8+FP16
block32 scales, quantizes Q to Q8_1 for integer K dots, and uses FP16 V/softmax
accumulation on RDNA3. V-only Q8 is worse than K-only, while full Q8 is better
than either, showing partial K/V error cancellation.

Direct arithmetic is not a universal repair. Native llama.cpp full-Q8 is 73.08%
lower mean KL than host group32 on mixed input, but still rejects. Native
hipEngine per-head is 48.98% worse than its host-reconstruction row
(`0.19038` versus `0.12779`). Therefore no group32/Hadamard native kernel or
128K gate follows. The prior llama.cpp repeated 128K/16 result
(`0.00521/0.08749`, 100% top-1) remains mechanically valid only as a saturation
control; it no longer establishes representative eight-bit fidelity.

The same-weight cross-engine BF16 bridge remains separate: hipEngine GGUF BF16
versus llama.cpp F16 at repeated 128K/16 rejects aggregate mean/max KL at
`0.26606/4.51481` because of prompt-final drift, while decode-only mean KL is
`0.000510` with 100% top-1. It does not isolate cache dtype.

The original free-generation task smoke remains `reference_unscorable` because
BF16 scored 0/5. A replacement restricted-choice probe provides partial bounded
functional evidence: at 4K, BF16 qualifies 2/5 and INT8 flips the qualified
multihop answer `D -> C` while retaining aggregation; at 32K, BF16 qualifies
3/5 and INT8 retains all three (multihop, aggregation, long-document). Thus high
KL can change a real decision but does not imply every answer changes. This is
not a full/free-generation quality claim and does not make 256K INT8 supported.

Capacity and outcome artifacts:
[profile-aware BF16 frontier](results/2026-07-13-gfx1100-paro-bf16-context-frontier.json),
[W7900 default-profile 220 Ki diagnostic](results/2026-07-13-w7900-paro-bf16-220ki-capacity.json), and
[INT8 accuracy outcome](results/2026-07-13-w7900-paro-int8-kv-accuracy-outcome.json).
Detailed diagnostics:
[llama.cpp Q8_0 repeated-token matched quality](results/2026-07-13-w7900-llamacpp-q8-kv-matched-quality.json),
[repeated/mixed prompt and native K/V arithmetic isolation](results/2026-07-13-w7900-gguf-q8-kv-protocol-arithmetic-isolation.json),
[same-weight GGUF bridge](results/2026-07-13-w7900-gguf-llamacpp-matched-parity.json),
[bounded functional check](results/2026-07-13-w7900-paro-int8-kv-functional-mc.json),
[matched baseline](results/2026-07-13-w7900-paro-int8-kv-fidelity-baseline.json),
[format screen](results/2026-07-13-w7900-paro-kv-format-ablation.json),
[PARO external-format KL/top-k screen](results/2026-07-13-w7900-paro-int8-kv-external-format-screen.json),
[same-weight GGUF external-format screen](results/2026-07-13-w7900-gguf-int8-kv-external-format-screen.json), and
[policy screen](results/2026-07-13-w7900-paro-kv-policy-ablation.json).

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

### W7900 direct GGUF concurrency, 2026-07-17

**Status: retained direct native-c4/c8 decode-model-step throughput with exact
c1/c2 controls, plus correctness-retained, live-observable OpenAI continuous
membership; no server throughput claim.** All rows use the same Qwen3.6-35B-A3B
`UD-Q4_K_M`, BF16 KV, greedy-top1, W7900/gfx1100, and synchronized graph-step
timing. Aggregate counts
every row advanced by each logical transition; per-request is aggregate divided
by live rows. ITL is model-step completion latency and excludes streaming-token
D2H. Packed prefill, graph capture, and memory are reported rather than hidden.

<!-- BEGIN TOPLINE:W7900_CONCURRENCY -->
| Route | Logical C | Native groups | Aggregate decode tok/s | Per-request tok/s | Aggregate / c1 | Aggregate / serial-c4 | TTFT p50 / p95 | Model-step ITL p50 / p95 | Tracked peak |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct c1 | 1 | 1x c1 | 85.469 | 85.469 | 1.000x | 1.009x | 0.209 / 0.209 s | 11.693 / 11.955 ms | 21.783 GiB |
| direct c2 | 2 | 1x c2 | 127.427 | 63.714 | 1.491x | 1.504x | 0.951 / 0.954 s | 15.765 / 16.023 ms | 22.394 GiB |
| direct c4 | 4 | 1x c4 | 184.575 | 46.144 | 2.160x | 2.178x | 2.020 / 2.023 s | 21.715 / 22.021 ms | 23.396 GiB |
| **direct c8** | **8** | **1x c8** | **246.872** | **30.859** | **2.888x** | **2.913x** | **3.475 / 3.479 s** | **32.414 / 32.749 ms** | **25.401 GiB** |
| chunked c8 control | 8 | 2x c4, serialized | 183.020 | 22.878 | 2.141x | 2.160x | 3.055 / 4.084 s | 43.767 / 44.281 ms | 26.069 GiB* |
| serial-c4 rate control | 4 | 4x c1, serialized | 84.738 | 21.185 | 0.991x | 1.000x | 0.548 / 0.877 s | 47.225 / 48.142 ms | 26.985 GiB* |
<!-- END TOPLINE:W7900_CONCURRENCY -->

Protocol: prompt 512 per row, 128 decode transitions, one discarded full-route
warmup and median of three, one shared model load. Resident sessions grow
c1→c2→c4→c8, so each direct row's memory is scoped to its current session count.
The starred controls execute after native c8 and retain later graph/workspace
allocations; they are throughput controls, not direct-route memory rows. Native
c8 improves aggregate decode **188.84%** over c1, **34.89%** over two serialized
c4 groups, and **191.33%** over the serial-c4 rate. Its per-request rate is
**63.89% lower** than c1; TTFT, model-step ITL, and memory tradeoffs remain
explicit. The clean marker window contains **748 packed-native, 0 row-local,
and 0 copy dispatches** for one real physical c8 graph replay. D4/D5 admit the
model step to a correctness-retained bounded OpenAI live-membership route and
expose lock-consistent occupancy, latency, KV, graph, and fallback telemetry,
but no server-wall, TTFT, or ITL performance result is inferred from those
correctness/instrumentation gates. Optional compaction, arbitrary C, and gfx1151
symmetry remain separate gates.

Artifacts: [retained C4 throughput closure](results/2026-07-16-gfx1100-gguf-concurrency-c4-native-graph-scaling-closure.json),
[D4 OpenAI lifecycle closure](results/2026-07-16-gfx1100-gguf-concurrency-d4-openai-streaming-closure.json),
[D5 live-observability closure](results/2026-07-17-gfx1100-gguf-concurrency-d5-live-observability-closure.json),
[E2 native-c8 correctness](results/2026-07-17-gfx1100-gguf-concurrency-e2-native-c8-correctness.json),
and [retained E2 native-c8 scaling](results/2026-07-17-gfx1100-gguf-concurrency-e2-native-c8-scaling-closure.json).
Historical mixed-quant/mixed-scope references remain linked in
[`HISTORY.md`](HISTORY.md) and the 2026-07-07 result directories.

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
torch-free hermetic TheRock environment, PARO's two discarded plus five
measured runs, GGUF's calibrated one discarded plus three measured runs, and
five internal llama-bench repetitions. It records a
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
2. Run PARO with 2 discarded warmups and 5 measured repetitions. Run GGUF with
   1 discarded warmup and 3 measured repetitions. Escalate GGUF to 5 only when
   a named variance, stability, or borderline-decision trigger fires; test
   lifecycle soak separately.
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
