# gfx1151 PARO/GGUF Optimization Ledger

Last updated: 2026-07-13.

Status: v0.3.0 is published, so the unreproduced `SOL-R0` report is no longer a
release blocker. Reactivate it only with a named matched known-good/current A/B.
The active pickup is now exact production concurrency: finish `SOL-R3` controls,
then make `SOL-R4` exact path by path, starting with PARO c2 and GGUF lifecycle
coverage. PARO native c2 still diverges on the first decode transition. GGUF's
production packed route now passes a three-repeat, full-natural-suite generated-
token diagnostic, but deeper state/KV/lifecycle and retained throughput gates
remain open.

The local gfx1151 evidence sprint is otherwise closed, and the post-topline
recovery phase is ready for pickup. Every evidence-sprint item is accepted,
rejected, or parked with a named reactivation gate; `SOL-R0` through `SOL-R9`
below turn the remaining performance gaps into ordered experiments. Cross-GPU
V10 remains blocked on W7900 hardware, V11 remains blocked until queued R7
supplies matched Q6 math/layout, and DFlash R8 remains blocked on a materially
better target-matched drafter. The P0 foundation is accepted on top of the
`7ea21e98b097` release-default baseline; gfx1151 HIP/Vulkan v2 is clean at
`ca241dae`, PARO P1/P2 are clean at `a18ff7bc` / `6f1910c9`, and the real
DFlash profile is clean at `8eb27215`. The final six-shape c1/topline refresh is
accepted at measured `d1231ee0`, with the four-column promotion gate assembled
cleanly at `7e9aad21`.

This is the active coordinator for making the PARO and GGUF paths correct,
fast, memory-efficient, and scalable on gfx1151 without regressing gfx1100. It
consolidates the next work from:

- `MTP-LLAMACPP-PARITY.md`, the current GGUF MTP parity dashboard;
- `PARO-GGUF-MTP-TRANSFER.md`, the current PARO transfer queue;
- `HIP-vs-VULKAN.md`, the current compiler/runtime decision dashboard;
- `TUNING-gfx1151.md`, `TUNING-gguf.md`, and `CONCURRENCY.md`;
- the 2026-07-10 current-HEAD code and evidence audit.

`PLAN.md` remains the architecture source of truth. `BENCHMARK.md`
and `TESTING.md` remain the promotion contracts. Each dashboard links a
byte-for-byte `*-HISTORY.md` snapshot containing its dated lab notebook and
implementation record. This file owns the cross-cutting ordering,
prerequisites, and completion state.

## Scope And Completion

In scope:

- PARO and GGUF AR prefill/decode on gfx1151 and gfx1100;
- HTTP concurrency, true backend row width, continuous shrinking, and sparse
  resident slots;
- GGUF MTP and PARO MTP/DFlash routing, verifier economics, and commit paths;
- architecture-specific tuning through registry/config profiles;
- a bounded, corrected HIP/Vulkan comparison that can guide production work;
- memory residency, launch/synchronization, and host/device transfer costs.

Out of scope until a retained profile activates them:

- a broad Vulkan backend;
- generic hand-written ISA;
- speculative kernel rewrites without an exposed end-to-end bucket;
- new approximation modes or accuracy trades;
- prompt-specific acceptance tuning of any kind.

"Tried everything" has a bounded meaning: every unconditional item in this
ledger is accepted, rejected, or blocked with evidence, and every conditional
item is either run because its activation trigger fired or parked with that
trigger recorded. It does not mean enumerating arbitrary kernel variants.

Status values:

| Status | Meaning |
| --- | --- |
| `open` | Ready once its dependencies are satisfied. |
| `blocked` | A named prerequisite prevents useful work. |
| `conditional` | Run only when its activation trigger is present in a corrected profile. |
| `in_progress` | The current logical unit; name it in `WORKLOG.md`. |
| `accepted` | Correctness, end-to-end, artifact, rollup, and commit gates passed. |
| `rejected` | The premise was tested and did not pass; preserve the artifact and reason. |
| `parked` | Do not retry until the stated premise changes. |

Landing instrumentation is not `accepted` for a performance item.
"Done" means the exit gate in this document passed.

## Current Evidence Snapshot

The table names the source revision for each result.

| Area | Current defensible result | Qualification / immediate consequence |
| --- | --- | --- |
| GGUF MTP on gfx1151 | `llama-compat` B2 reports `71.52 tok/s` versus llama.cpp HIP `71.91 tok/s`, with hipEngine stage wall `14.005` versus `14.269 ms/output`. | This is an opt-in direct-commit/dp4a compatibility contract, not exact/default semantics. Keep it as a replication lane, not the production default. |
| Exact/default GGUF MTP | Fixed 10-cycle B5 reports `61.98` versus AR `54.79 tok/s`. | Natural `max_tokens=24` loses at B1/B2/B5: `52.13/52.04/50.65` versus AR `54.80`. Fixed-cycle rows do not close production MTP economics. |
| MTP server routing | The corrected clean-tracked `d2b1e742` natural24 matrix shows the current `llama-compat` hook is faster at c1/c2 but changes exact AR IDs on heldout `general_ja_explain` even at c1; wider groups add more mismatches. | Compatibility MTP cannot select `auto`. Automatic requests must remain on exact/default AR until an exact MTP hook exists; explicit opt-in keeps the documented compatibility contract. Actual groups, not client c, remain the routing key. |
| Exact server measurement | `SOL-E1` carries exact IDs through every choice; `SOL-E2` gives timing payloads explicit scope/row/owner metadata; `SOL-S2` separately records the request-scoped route cap, queue request/prompt grouping, actual backend calls/widths, and target verifier rows. | `mtp-bench.py` fails closed on incomplete shape groups and counts each timing owner and queue group once. Retokenized visible text remains non-authoritative; historical server rows predate these contracts. |
| Canonical artifact provenance | `SOL-E3` gives server, retained PARO, GGUF category/true-AR, and HIP/Vulkan micro artifacts one torch-free schema with dynamic backend/arch/device identity, separate staged/unstaged/untracked state, and content-derived model fingerprints. | New retained rows must contain a valid `hipengine_artifact_provenance` v1 block and an existing model fingerprint where a model ran. Legacy provenance remains diagnostic until rerun. |
| gfx1151 c1 model toplines | The clean `d1231ee0` six-shape refresh passes stable finite-output, five-sample variance, matched Q4_K_M identity, and GTT-memory gates for hipEngine GGUF plus llama.cpp HIP/Vulkan. Clean `9944e481` supplies exact PARO 512/1K rows; clean `01e2cec5` supplies exact queue-isolated 4K-128K rows. PARO 512/128 is `1140.101 prefill / 66.767 decode tok/s`; 128K/128 is `474.641 / 30.386`. | The repository and benchmark README tables use the newer retained PARO column and preserve the July 11 matched GGUF/llama columns. The PARO-vs-Q4_K_M cells are throughput targets, not a same-math A/B. |
| PARO c>N | `SOL-P1` closes the clean gfx1151 c1-c8 p512/d128 catalog at `a18ff7bc`: c1 graph replay is retained at `66.910 tok/s`; every c2-c8 native row fails the independent-c1 sequence at index 2 (`17` vs `220`) and is explicitly serial. `SOL-P2` closes ragged c8-to-c1 lifecycle safety at clean `6f1910c9`: all generated IDs, 30 linear-state families, and 10 full-KV families match independent c1 through EOS plus front/middle/tail sparse cancellation. | Production greedy and sampled batches use exact width-1 sessions; ragged packed prefill uses the explicit `per_segment_ragged_exact` fallback. gfx1100 is stale/non-selecting pending W7900 hardware. P3/P4/P7-P9 are parked behind a general exact native c>N algorithm; P6's invalid splitter is removed. |
| GGUF c>N | The production packed-prefill/packed-decode route now has an executable independent-c1 gate. On gfx1151, all 10 `mtpbench-code-general-ja` rows match for three c10 repeats at eight output tokens, using packed c4+c4+c2 chunks with no serial fallback. | This is generated-token evidence only (`performance_claim=false`). It does not clear hidden/Conv/GDN/KV identity, shrinking/sparse slots, long context, server cancellation/admission, or retained profiler/scaling gates. Artifact: [`2026-07-13...token-equality.json`](../benchmarks/results/2026-07-13-gfx1151-gguf-natural10-cn-token-equality.json). |
| PARO DFlash | Clean `8eb27215` S4 runs the curated 35B pair with same-session AR, exact output, coarse/fine phase buckets, and verifier graph shapes. Exact replay is `9.676` versus `65.266 tok/s` AR (`0.14825x`). | S5 branch-copy is correctness-red, S6 has no wider-group premise at 1/114 accepted proposals, and S7 fused LM-head is 5.16% slower. DFlash stays default-off. |
| GGUF eager correctness / gfx1100 refresh | On gfx1151, SOL-G1 proves the exact Q4_K_M `[9707] * 512` continuation matches llama.cpp and is byte-exact for four eager hidden/Conv/GDN/KV transitions. The last W7900 diagnostic remains about `654 prefill / 35.8 decode tok/s`. | Repetition of `9707` is valid model behavior, but the W7900 row is still `performance_claim=false` and predates the hardware-local oracle/provenance contract. Rerun both gates on W7900 before using the rate. |
| HIP versus Vulkan | The gfx1151 timing-contract v2 matrix at `ca241dae` records `hipengine_dirty=false`, retains 22/22 comparisons, and separates `serial_latency` from `independent_throughput`. Serialized production slices are mostly HIP-favored; synthetic packed dot and dispatch retain Vulkan leads. | Keep HIP as the production backend. gfx1100 still needs the same bounded v2 matrix; Q6 lm-head remains incomparable because the implementations use different math/layouts. |

The evidence sprint has therefore closed measurement/routing correctness, GGUF
correctness recovery, PARO shape safety, and the bounded speculative transfer
audit. The accepted topline comparison now activates the bounded recovery queue
below. Parked P/G/S/V work still starts only through its recorded reactivation
gate; the R queue does not authorize retrying a rejected implementation.

## Post-Topline Regression Analysis And Recovery Plan

This is the current pickup section. It separates measured regressions from
invalid historical speed, identifies the smallest exact experiment that can
change each premise, and delays the expensive public rerun until a default has
actually changed.

### Release Blocker: R0 PARO Decode Recovery

The recent correctness/verification pass made the PARO route defensible, but
the working release comparison reports a broad **30%+ decode loss** versus the
intended pre-pass point-release baseline. Recovering that lost throughput is
more valuable and more urgent than the deferred GGUF/compiler candidates below.
It blocks the next point release.

The June 15-to-July 11 c1 table below does not close this question. It is not a
matched last-known-good/current A/B for the correctness-pass boundary and does
not cover every shipping decode route. R0 must first name the exact good commit,
bad commit, runtime policy, model fingerprint, shape matrix, and commands. It
must then separate unavoidable correct math from accidental costs such as
diagnostic readback, validation retained on the hot path, forced serialization,
fallback dispatch, graph-policy changes, or lost kernel selection. Those are
hypotheses to bisect, not presumed causes.

The recovery rule is strict: preserve exact hidden/state/KV/token behavior and
retain every measured, non-regressive reduction as soon as it is proven. Do not
trade correctness back for the old rate, and do not require a minimum aggregate
percentage before accepting a clear component or end-to-end win.

### Published Checkpoint And Comparison Limits

The current public checkpoint is complete. The root [`README.md`](../README.md)
and canonical [`benchmarks/README.md`](../benchmarks/README.md) contain separate
six-shape prefill, decode, and peak-memory tables for hipEngine PARO, hipEngine
GGUF, llama.cpp HIP, and llama.cpp Vulkan; the speculative section presents
GGUF exact/default, GGUF `llama-compat`, and llama.cpp HIP side by side; and the
concurrency section publishes the exact PARO production route. The accepted
source is the
[`d1231ee0` four-engine summary](../benchmarks/results/2026-07-11-gfx1151-readme-refresh-20260711-d1231ee0-summary.json).

The comparison below uses the superseded
[`2026-06-15` one-run summary](../benchmarks/results/2026-06-15-gfx1151-readme-udq4km-20260615-040438-summary.json)
only as a diagnostic lead. It is not a same-commit regression A/B: it used one
measured run per shape, no measured warmup, incomplete source/build provenance,
and unusable llama.cpp aperture-memory readings. The current sweep uses clean
provenance, right-sized sessions, two discarded plus five measured hipEngine
runs, and proper llama.cpp GTT sampling. Therefore old-to-current percentages
are hypotheses to reproduce on current code, never promotion baselines.

The control columns make broad hardware drift unlikely. Across the six shapes,
llama.cpp HIP/Vulkan decode changes by at most 1.9% from the old diagnostic;
prefill is within -2.5% to +2.4% except the llama.cpp HIP 512 row at +4.4%.
The much larger hipEngine prefill movement is path/policy-sensitive.

### Diagnostic Old-To-Current Movement

Each cell is `2026-06-15 -> 2026-07-11 tok/s (delta)`. Decode length is 128.

| Prompt | PARO prefill | PARO decode | GGUF prefill | GGUF decode |
| --- | ---: | ---: | ---: | ---: |
| 512 | `956.666 -> 994.866` (+3.99%) | `66.967 -> 66.753` (-0.32%) | `833.366 -> 430.767` (-48.31%) | `56.581 -> 49.536` (-12.45%) |
| 1K | `1067.175 -> 810.029` (-24.10%) | `61.768 -> 61.628` (-0.23%) | `854.308 -> 437.467` (-48.79%) | `52.832 -> 52.192` (-1.21%) |
| 4K | `1062.248 -> 671.985` (-36.74%) | `62.910 -> 62.715` (-0.31%) | `729.117 -> 403.946` (-44.60%) | `53.638 -> 52.999` (-1.19%) |
| 32K | `822.255 -> 599.063` (-27.14%) | `50.368 -> 50.362` (-0.01%) | `619.570 -> 369.942` (-40.29%) | `44.383 -> 43.947` (-0.98%) |
| 64K | `622.752 -> 511.248` (-17.91%) | `41.966 -> 42.032` (+0.16%) | `522.872 -> 334.395` (-36.05%) | `37.741 -> 37.477` (-0.70%) |
| 128K | `425.727 -> 375.635` (-11.77%) | `30.286 -> 30.316` (+0.10%) | `384.011 -> 270.601` (-29.53%) | `28.043 -> 27.862` (-0.65%) |

Interpretation:

- PARO c1 decode is stable to 0.32% *within this particular June 15-to-July 11
  diagnostic comparison*. That does not supersede R0: these are not the named
  last-known-good/current correctness-pass endpoints, and the table does not
  cover every release decode route. Establishing and recovering the reported
  release regression now precedes c1 prefill work.
- The old PARO path forced every prefill family to chunk 256 at every shape.
  Current code is unchunked at 512/1K and uses 1024 for linear/MoE/post/RoPE
  plus a 4096 attention-query chunk at 4K and above. Reproducing the old rates
  on current exact code would be worth +31.75% at 1K, +58.08% at 4K, +37.26%
  at 32K, +21.81% at 64K, and +13.34% at 128K. These are diagnostic ceilings,
  not promised wins. Current unchunked 512 is already 3.99% faster, so a global
  force-256 rollback is specifically disallowed.
- The old GGUF row is not a valid speed target. Its 512 run ended at token
  `2814`; G1 proves the current exact model/llama.cpp continuation is repeated
  `9707`. The old split-GDN route failed recurrent-state equality, and the old
  graph corrupted state on third-and-later replays. G2 repaired an exact split
  chain, but G3 measured it 5.19%/6.70% slower than fused at 512/4K; G5's exact
  graph recovers only 0.112%. Do not restore either rejected path to chase the
  old number.
- Current GGUF decode is within +5.7%/-4.7% of llama.cpp HIP through 64K, then
  is 13.2% behind at 128K. Current GGUF prefill is 30.7%-60.0% behind llama.cpp
  HIP, so prefill needs a new exact algorithm rather than historical flag
  restoration. PARO/GGUF 128K decode also needs a context-local profile before
  transferring the short-context G4 Amdahl ordering.
- Use the accepted peak-memory table rather than old-to-current memory deltas:
  session sizing and llama.cpp measurement scope changed. Current PARO remains
  below 24 GiB through 128K (`22.124 GiB` tracked), while GGUF crosses the
  24 GiB-class line at 64K/128K (`24.203/25.493 GiB`). G6 proves no duplicate
  512-shape replacement weights; R6 must attribute context-local KV/state/
  scratch growth before selecting a lower-memory long-context policy.

### Ordered Recovery Queue

The R IDs coordinate existing P/G/S/V gates; they do not supersede their
correctness contracts.

| Priority | ID | Work package | Current state | Exit / handoff gate |
| ---: | --- | --- | --- | --- |
| 0 | `SOL-R0` | Reproduce, bisect, and recover the reported PARO decode regression while preserving the correctness/verification fixes. | `parked` after v0.3.0 publication; reactivate only with a named reproducible matched boundary | A clean matched known-good/current A/B names the first performance-changing revision or policy; every affected default route recovers its exact non-regressive wins, and the release matrix passes hidden/state/KV/token plus wall-time gates. |
| 1 | `SOL-R1` | Current-code PARO c1 prefill chunk A/B. | `accepted`: exact gfx1151 linear/MoE recovery plus scoped AOTriton queue isolation retained across all six shapes; 1K negative control unchanged; the 4096-query isolation policy transfers positively to gfx1100 while linear/MoE-256 does not | Per-shape current policy versus forced-256 is state/token exact and five-run faster; promote only winning buckets. |
| 2 | `SOL-R2` | Exact/default GGUF MTP long-horizon economics, then exact commit recovery if still needed. | `open` | Full-suite plus heldout natural 64/128 rows use true AR; an exact route beats AR with margin before server/`auto` work. |
| 3 | `SOL-R3` | Measure exact serial c1-c8 server controls: shipping PARO width-1 groups and forced-serial GGUF. | `in_progress`: host cancellation isolation is fixed and both paths have executable equality diagnostics; the complete latency/occupancy/memory controls remain open | Both paths have exact IDs, aggregate/per-request throughput, latency, occupancy, and memory under final accounting. |
| 4 | `SOL-R4` | Build a general shape/lifecycle-safe native c2 algorithm with path-specific PARO/GGUF math, then expand through c8/shrink/sparse. | `in_progress`: PARO c2 RED reproduces on the first decode transition; GGUF natural10 generated tokens are green, with deeper lifecycle gates open | Each path preserves its independent-c1 hidden/state/KV/order; only then reopen P3/P4/P7-P9 or promote G8. |
| 5 | `SOL-R5` | Profile current fused GGUF prefill, then prototype a materially different parallel-exact GDN/layout schedule. | `open for profile`; candidate work is conditional on a named dominant bucket | Exact short/512/4K/segment/chunk oracles plus same-run wall beat fused; unchanged G2 split scheduling is not retried. |
| 6 | `SOL-R6` | Profile 512 versus 128K PARO/GGUF decode and memory growth, then recover the context-local dominant family. | `open` | Exact ROCTX/Amdahl and allocation tables name the 128K speed/capacity gaps; retain only a family-local exact wall or residency win. |
| 7 | `SOL-R7` | Implement matched HIP/Vulkan Q6 LM-head math/layout and close V11. | `open for implementation`; V11 comparison remains blocked | Rows 1/8, 2048->152064 use the same inputs/layout/output contract and pass correctness before any ratio. |
| 8 | `SOL-R8` | Obtain/train and validate a materially higher-acceptance DFlash drafter for this exact target, then build exact transactional commit. | `blocked on drafter quality` | Full-suite acceptance changes the economics; exact state/KV commit enables graph hits and the complete route exceeds same-protocol AR. |
| 9 | `SOL-R9` | Final validation and publication refresh after retained defaults settle. | `blocked until a default changes or recovery closes` | Full test gate passes; affected toplines, speculative/concurrency tables, artifacts, changelog, and both READMEs are regenerated together. |

### Recovery Playbooks

#### R0: PARO Decode Release Regression

Start with measurement and revision/policy bisection, not a speculative kernel
rewrite. Freeze the point-release workload identity: exact PARO model snapshot
and fingerprint, `w4_paro`, KV dtype, prompt fixtures, context/decode lengths,
concurrency and sampling mode, resolved gfx1151 backend, HIP/compiler stack,
TuneD/clock state, and graph/eager policy. Include the six c1 decode shapes and
every c>N/server decode route whose shipping default changed during the
correctness pass.

Run the last-known-good release candidate and current exact route under the same
cached-build protocol. If the 30%+ working report reproduces, bisect commits and
runtime policy independently until the first performance-changing boundary is
named. Join wall time to a marker/kernel-family profile and launch/sync counts;
audit whether correctness instrumentation, readbacks, serial fallbacks, route
selection, graph admission, or cache/build behavior leaked into steady decode.

Recover the smallest exact unit first. Each retained change must pass the
existing hidden, all 30 Conv/GDN state-family, all 10 live-KV-family, generated
token/order, lifecycle, and sampler gates that motivated the correctness pass.
Update the compact artifact and benchmark rollup for every retained win. R0
closes only when the affected release matrix is recovered or each remaining
delta is bounded to a named correctness-essential operation with profiler
evidence and an explicit release decision.

#### R1: PARO c1 Prefill

Run the six standard shapes on current clean code with identical resident model,
quant, KV dtype, prompt IDs, right-sized session, warmups, and repetition count.
Compare the current bucket policy against forced 256 for linear, MoE, full
attention query/post/RoPE; 512 is a required negative control. Before timing is
eligible, require generated IDs plus final hidden, all 30 Conv/GDN state
families, and all 10 live K/V families to match. Select chunks independently by
shape/family rather than restoring one global policy. If a bucket wins, route it
through the architecture profile/registry and rerun the affected six-shape PARO
component before the final four-engine publication.

##### R1 Retained Exact Recovery, 2026-07-12

The current matched selection run uses Radeon 8060S/gfx1151, Qwen3.6-35B-A3B
PARO snapshot `437eba06df05aad71a4dacdcaf3fff70ae1ee8a1`, W4 PARO, BF16 KV,
repeated token `9707`, right-sized sessions, graph-replay decode, two discarded
plus five measured repetitions, TheRock HIP 7.15 / clang `aa451e1f`, and TuneD
`accelerator-performance`. The clean control is detached `240c5daf`; the clean
six-shape exact candidate is detached `9944e481`. The 128K five-run
control/candidate sweep is complete. The subsequent queue-isolation
control/candidates are clean detached `01e2cec5`; they supersede the 4K,
32K, 64K, and 128K rows. The 512/1K rows remain the right-sized `9944e481`
results because those shapes use the proven-safe 256-query caller-stream route.

Keep four checkpoints distinct. The June 15 old diagnostic used one measured
run and the former all-256 route; it is history, not a promotion control. The
July 11 published pre-recovery row is clean and five-run but used HIP 7.13. The
ROCm 7.15 matched control/candidate pair is the only selection A/B below.
The old artifact also reports final token `9707` at every shape, unlike the
control-identical nontrivial continuation from the correctness-hardened route,
so its speed is an opportunity ceiling rather than proof that every delta is
recoverable with current semantics. llama.cpp HIP uses Q4_K_M rather than PARO
W4; it is an external throughput target, not a same-math control.

###### Prefill tok/s

| Workload | Old diagnostic | Published pre-recovery | Matched control | Recovered exact | Matched delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512/128 | `956.666` | `994.866` | `997.025` | **`1140.101`** | **+14.35%** |
| 1K/128 | `1067.175` | `810.029` | `799.651` | **`1208.343`** | **+51.11%** |
| 4K/128 | `1062.248` | `671.985` | `669.658` | **`854.346`** | **+27.58%** |
| 32K/128 | `822.255` | `599.063` | `607.134` | **`761.011`** | **+25.34%** |
| 64K/128 | `622.752` | `511.248` | `513.689` | **`619.374`** | **+20.57%** |
| 128K/128 | `425.727` | `375.635` | `379.873` | **`436.582`** | **+14.93%** |

###### Decode tok/s

| Workload | Old diagnostic | Published pre-recovery | Matched control | Recovered exact | Matched delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512/128 | `66.967` | `66.753` | `66.933` | `66.767` | -0.25% |
| 1K/128 | `61.768` | `61.628` | `61.657` | `61.746` | +0.14% |
| 4K/128 | `62.910` | `62.715` | `62.685` | `62.765` | +0.13% |
| 32K/128 | `50.368` | `50.362` | `50.307` | `50.351` | +0.09% |
| 64K/128 | `41.966` | `42.032` | `42.038` | `42.149` | +0.26% |
| 128K/128 | `30.286` | `30.316` | `30.320` | `30.371` | +0.17% |

###### Final queue-isolation retention

| Workload | Prior retained (`9944e481`) | Same-commit control (`01e2cec5`, isolation off) | Current default (`01e2cec5`) | Default vs control | Default vs prior |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4K/128 prefill tok/s | `854.346` | `885.141` | **`1089.031`** | **+23.03%** | **+27.47%** |
| 32K/128 prefill tok/s | `761.011` | `765.316` | **`906.145`** | **+18.40%** | **+19.07%** |
| 64K/128 prefill tok/s | `619.374` | `621.691` | **`716.775`** | **+15.29%** | **+15.73%** |
| 128K/128 prefill tok/s | `436.582` | `418.838` | **`474.641`** | **+13.32%** | **+8.72%** |

Every retained row uses a right-sized maximum-shape session, two discarded
warmups, and five measured repetitions. Candidate/control ranges are fully
separated: `1070.490..1091.709` versus `826.061..898.713` at 4K,
`903.515..912.905` versus `764.416..765.912` at 32K,
`701.297..728.763` versus `618.457..626.949` at 64K, and
`470.602..478.628` versus `415.735..423.616 tok/s` at 128K. Decode deltas are
`+0.017%/-0.157%/+0.110%/+0.116%`, respectively, and tracked peak memory is
identical in every matched pair.

The 1K follow-up is a required negative control, not a win. It shares the
max-32K session and moves `1180.652 -> 1191.136 tok/s` (+0.89%) with overlapping
ranges, but both settings execute the same 256-query caller-stream path; the
isolation branch is never entered. Keep the right-sized `1208.343 tok/s` 1K
topline from `9944e481`. The independent clean `c3a03ed1` 4K A/B
(`883.600 -> 1114.634 tok/s`, +26.15%) and isolated-kernel replay continue to
corroborate the 4K selection.

The retained candidate is intentionally narrower than the old all-256 route.
The gfx1151 architecture overlay changes only linear-attention and MoE layer
chunks from the current `1024`/unchunked policy to `256`; it leaves the existing
full-attention query/post/RoPE and low-memory policies intact. Explicit manual
chunks and `--no-prefill-chunk-autotune` still take precedence. Short prompts
therefore resolve to `256/256/0/0/0`; ordinary prompts above 1K resolve to
`256/256/4096/1024/1024`.

Correctness selected that narrower policy:

- Pairwise 512, 4K, and 128K gates match the control seed, final-hidden
  SHA-256, all 30 Conv/GDN state families, and all 10 live K/V families. The
  128K control/candidate aggregate state SHA-256 is identically
  `886a68bc2294a370e71d2ad6b43fa7dd34abdbd5d0232b3693b31ad253e63365`.
  All six five-run shapes also have a control-identical generated preview and
  stable measured IDs.
- At 4K, linear-only 256, MoE-only 256, and combined linear+MoE 256 are
  byte-exact. Setting only the full-attention query/outer-layer chunk to 256
  changes final hidden plus 72 persistent state components. The combined
  all-256 candidate likewise diverges. Its tempting `1186.096 tok/s` five-run
  4K result is therefore rejected, not a retained recovery number.
- The mismatch topology localizes the first change to the first full-attention
  layer after its K/V append: layers 0-2 and layer-3 K/V still match. The 72
  changed persistent components are exactly 27 downstream linear layers times
  Conv+GDN (`54`) plus nine later full-attention layers times K+V (`18`). A
  layer-3 RED stage oracle should therefore compare prepared Q/K/V/gate,
  AOTriton output, O projection, post-attention normalization, and MoE output
  in order instead of treating all 40 layers as one opaque failure.
- The older serial/native fixture is not a useful selector for this model
  snapshot: unchanged control and candidate fail it identically (`max KL
  7.1119`, top-1 `0%`, identical native tokens/logits). The pairwise current-
  policy oracle above proves the candidate delta while that pre-existing
  fixture/snapshot mismatch remains separate work.

Relative to the old diagnostic, the current retained rows are **+19.17% at
512**, **+13.23% at 1K**, **+2.52% at 4K**, **+10.20% at 32K**, **+15.10% at
64K**, and **+11.49% at 128K**. Relative to the published llama.cpp HIP prefill
row, they are **+7.43%**, **+15.83%**, **+7.91%**, **+21.87%**, **+24.96%**,
and **+21.57%**, respectively. PARO is now the raw-prefill leader at 512, 1K,
4K, 32K, and 64K; at 128K it is only **0.45%** behind llama.cpp Vulkan.
Decode is unchanged within the matched noise envelope; this is a prefill-only
change.

The current implementation couples `full_attn_query_chunk_size` to the outer
chunk for the entire full-attention layer. Reducing it to 256 therefore chunks
QKV/rotation/KV append, AOTriton attention, output projection, post-attention,
and MoE together. The speed result proves that row shape has enough headroom to
close the 4K gap, but changing AOTriton's query/reduction shape breaks the
byte-state contract. Do not simply promote that knob.

The helper semantics explain the context pattern. A linear layer takes the
minimum positive linear/MoE chunk, so either 256 setting chunks the entire
linear-attention+MoE layer. A full-attention layer uses the query chunk
exclusively when it is nonzero; only when query is zero does it take the
minimum of post/RoPE/MoE. Consequently, at 512/1K the base query is zero and
`moe=256` also puts the ten full-attention layers on the exact 256-row outer
shape. At 4K+ the base query is 4096, so the exact profile chunks only the 30
linear layers while all ten full-attention layers remain 4096-row. This is why
the short rows fully recover and why the remaining 4K opportunity is
concentrated behind the query/outer-layer coupling rather than another global
linear/MoE threshold.

The clean selected-region 4K rocprof comparison is now complete. These are
single profiled rows used for attribution, not replacements for the retained
five-run throughput medians above:

| Profile leg | Host prefill | Traced GPU | Calls | GDN | Conv | Rotation | W4 prefill | Routed MoE | AOTriton |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Matched control | `5900.851 ms` (`694.137 tok/s`) | `5785.735 ms` | `4,710` | `1430.379 ms` | `965.975 ms` | `1078.755 ms` | `738.769 ms` | `715.952 ms` | `150.710 ms` |
| Recovered exact | `4831.139 ms` (`847.833 tok/s`) | `4666.760 ms` | `17,670` | `1061.292 ms` | `913.881 ms` | `571.212 ms` | `771.465 ms` | `794.968 ms` | `150.656 ms` |
| Rejected full-query 256 | `3763.046 ms` (`1088.480 tok/s`) | `3619.753 ms` | `23,370` | `1021.179 ms` | `88.150 ms` | `326.138 ms` | `803.722 ms` | `789.408 ms` | `213.623 ms` |

The exact profile saves `1118.975 ms` of traced GPU time versus control. The
remaining invalid full-query shape saves another `1047.007 ms`, but AOTriton
itself gets `62.967 ms` *slower*. The largest delta is convolution:
`913.881 -> 88.150 ms`, or `825.731 ms`/78.9% of the total GPU-time gap. The
exact and rejected legs both launch 480 copies of the same main convolution
kernel with identical 256-row shape and metadata (workgroup `256`, grid
`2,097,152`, VGPR `32`, scratch `20`). In the exact leg, layers 0-2 run at
about `150 us/chunk`; immediately after the first full-attention layer, every
later linear layer runs at about `2.15-2.23 ms/chunk`. The rejected leg stays
at about `170-186 us/chunk` for every linear layer. This is a downstream
execution cliff triggered at the same layer boundary as the correctness split,
not evidence that the attention core is the remaining hot family.

The follow-up isolation rejects the activation-domain hypothesis. Exact
layer-2/layer-4 FP32 accumulators span `-13.584869..4.723076` and
`-7.517517..4.157738`; all `2,097,152` values per slice are finite and none lie
outside `+/-16`. Crossing activation and weight captures does not move the
cliff. Releasing the whole prefill workspace also does not recover it.

The transition is queue-local and occurs inside AOTriton dispatch. Replaying
the identical captured convolution measures `120.936 us` before layer 2,
`118.018 us` after layer-3 K/V append, and `1834.637 us` immediately after
layer-3 AOTriton on the same stream. A fresh replay stream measures
`118.385 us`; running AOTriton on a dedicated nonblocking stream and returning
to the original stream measures `119.093 us`. The exact 4096-query AOTriton
kernel reports `2560` bytes of scratch, while the numerically rejected
full-layer 256-row experiment reports only `992/1008` bytes and does not poison
later convolution execution. ROCr's documented persistent per-queue scratch
assignment is a plausible mechanism, but the upstream ROCr/AOTriton root cause
remains an inference rather than a concluded compiler defect.

The retained default therefore keeps all pre/post work on the caller stream
and sends only AOTriton query rows `>=512` through one lazy session-owned
nonblocking stream linked by two reusable HIP events. The proven-safe 256-row
bucket stays on the caller stream, so the established 512/1K route is
unchanged. `HIPENGINE_QWEN35_AOTRITON_ISOLATED_PREFILL_STREAM=0` remains the
rollback control. Disabling AOTriton entirely is rejected: the 4K screen falls
from `849.557` to `270.656 tok/s` with the same final token.

The final clean differential gates pass at every isolated shape. Control and
candidate share the same sampled seed, final-hidden SHA-256, aggregate
persistent-state SHA-256, all 30 Conv/GDN families, and all 10 live K/V
families with zero mismatch paths:

| Workload | Seed | Final hidden SHA-256 | Aggregate state SHA-256 |
| --- | ---: | --- | --- |
| 4K | `13743` | `f2fd15eef230...3063a434c5d640b7af3496c45d2678` | `c2328617d0bd...48d9e55523e335d2c5f` |
| 32K | `13743` | `747b503a2dec...3eed80b28e180bdf7b52c7` | `496af033eb2e...fabdcee83f6ecc4a885` |
| 64K | `4256` | `13378dcdb630...9175ca4886bf2f509570690` | `acc07c0d1dfd...d63d13413f4b991b9` |
| 128K | `49556` | `87c78fbae561...498233c8c8952929acd00` | `886a68bc2294...3693b31ad253e63365` |

Retained evidence is
[`2026-07-12-gfx1151-paro-aotriton-stream-isolation.json`](../benchmarks/results/2026-07-12-gfx1151-paro-aotriton-stream-isolation.json).
The gfx1151 six-shape refresh and hardware-local gfx1100 transfer validation
are complete.

##### R1 gfx1100 Transfer Check, 2026-07-12

The clean W7900/gfx1100 matrix at `255e5aca` tests the two transferable R1
levers rather than assuming gfx1151 ratios apply:

- AOTriton queue isolation **does transfer**, but at a much smaller magnitude.
  The initial off/on/off ordering drifted enough to give a false negative, so a
  reverse on/off/on sequence balanced 15 measured samples per mode. Isolated
  prefill changes by **+1.638%/+0.495%/+0.192%** at 512/1K/4K, while total
  measured wall falls by **1.653%/0.127%/0.562%**. Differential gates at all
  three shapes match the sampled seed, final hidden, all 30 Conv/GDN state
  families, and all 10 live K/V families byte-for-byte. This matrix measured
  the earlier global-isolation policy: 512/1K use the 256-query route that the
  merged threshold now leaves same-stream, while 4K uses 4096-query AOTriton
  and directly validates the scoped default. Keep that scoped bridge and its
  explicit rollback flag.
- The gfx1151 linear/MoE-256 chunk profile **does not transfer**. With queue
  mode held equally same-stream, gfx1100 prefill changes by
  **-7.723%/-8.782%/-6.398%** at 512/1K/4K. Candidate ranges are wholly below
  control at every shape. Tracked peak falls only `0.58%/1.72%/0.70%`; stable
  IDs do not justify paying the throughput loss. Keep the generic gfx1100
  chunks and do not broaden the gfx1151 architecture overlay.

The compact transfer evidence is
[`2026-07-12-w7900-gfx1100-paro-gfx1151-transfer.json`](../benchmarks/results/2026-07-12-w7900-gfx1100-paro-gfx1151-transfer.json).
R1 is complete; do not retry activation-domain shortcuts, allocator churn, or
global chunk transfer.

#### R2: Exact GGUF MTP

The
[`natural24 exact/default baseline`](../benchmarks/results/2026-07-03-ar-mtp-default-natural24-budget-sweep-c1.json)
is already close enough to justify one bounded recovery pass: B1 is `52.1308`
versus true AR `54.8037 tok/s`
(`19.2172` versus `18.2469 ms/output`), a `0.9703 ms/output` or 5.05% gap; B2 is
`52.0410 tok/s`, a 5.20% wall gap. Run full-suite plus category-heldout natural
64/128 first because horizon amortization may change the premise without code.
Fixed 10-cycle B5 `1.1312x` remains diagnostic and cannot substitute.

If longer horizons still lose, target B1's exact
`target_block_replay_or_commit` bucket (`1.2762 ms/output`). A transactional
prefix-state/KV commit that removes at least `0.9703 ms/output` in end-to-end
wall has enough measured headroom in principle, but it must match serial-prefix
hidden/Conv/GDN/KV and generated IDs through rejection and tail cases. After a
direct exact win, add the exact/default server route, rerun realized groups and
heldouts against true AR, and only then allow S1 `auto`. S3 hysteresis and wider
verification stay parked until that static route is exact and profitable.
`llama-compat` remains an explicit accuracy-traded comparison lane.

#### R3-R4: Exact Production c2-c8

First measure what production actually ships: c1-c8 HTTP requests advancing
through reused width-1 sessions. The artifact must expose each backend call,
aggregate and per-request tok/s, TTFT/inter-token/p50/p95 latency, occupancy,
memory, and exact IDs. This is the baseline missing from P1; multiplying c1 by
client width is not a measurement.

Then start PARO with exact native c2, not c8. Freeze the
[`P1 independent-c1 oracle`](../benchmarks/results/2026-07-11-sol-p1-gfx1151-paro-c1-c8-exact-catalog.json)
and preserve each row's c1 reduction/update/sampler order while sharing only work
that cannot change its result. The current RED localizes the first numerical
drift to layer-4 linear input/Conv state and the first K/V drift to layer 7;
packed prefill itself is exact. Require byte-exact hidden, 30 Conv/GDN families,
10 live K/V families, and the full 137-token sequence. The rejected native
c2-c8 rates (`78.52/87.47/99.64/102.18/109.81/109.58/115.51 tok/s`) are only a
1.17x-1.73x opportunity envelope over c1 `66.91`, never eligible results.

After c2 is exact, extend the same algorithm through every integer width,
ragged prompts, c8->c1 EOS/cancel shrink, and front/middle/tail sparse slots.
Only then reopen sparse native admission (P3), selected/grouped policy (P4),
graph buckets (P7), architecture tuning (P8), and weight-reusing
MMQ/GEMM/WMMA/grouped MoE (P9), in that order.

GGUF still needs its own forced-serial R3 control; PARO exactness cannot certify
Q*_K/T16/X8 math. The new executable gate did not reproduce G8's intermittent
token failure: the complete 10-prompt suite passed three c10 repeats through
packed c4+c4+c2 chunks. Treat that as a narrowed blocker, not closure. Add
first-hidden/Conv/GDN/KV capture to the same multi-prompt gate, then repeat
c1-c8, ragged/shrink, sparse-slot, cancellation, and long-context transitions.
Only that shape/lifecycle evidence can promote G8.

#### R5-R6: GGUF Prefill And Long-Context Decode

The detailed R5 evidence hierarchy, implementation comparison, exclusions,
ranked experiments, and exact value-column-tiling design live in
[`GGUF-PREFILL-OPTIMIZATION.md`](GGUF-PREFILL-OPTIMIZATION.md).

Do not infer the current prefill bottleneck from the invalid old route. Add a
clean fused-prefill family profile at 512, 4K, and 128K, including GDN,
dense/selected MoE, attention, projection, launches, and host wall. If GDN is
material, the next candidate must be a new parallel-exact schedule that keeps
raw-Q/K scaling and recurrent update order while avoiding G2's split-chain
materialization/launch cost. It must pass G2's short, 512, 1024/1025, and
4095/4096 state oracles, then beat fused at both 512 and 4K before any broader
shape sweep. The starting evidence is the
[`G2 exact matrix`](../benchmarks/results/2026-07-11-sol-g2-gfx1151-gdn-prefill-exact-matrix.json)
and [`G3 interleaved A/B`](../benchmarks/results/2026-07-11-sol-g3-gfx1151-gdn-prefill-interleaved-ab.json).

Separately profile exact decode and allocation growth at 512 and 128K for both
PARO and GGUF. At 128K PARO is `30.316` versus llama.cpp HIP `32.114 tok/s`
(-5.6%); GGUF is `27.862` (-13.2%). Attribute attention/KV growth separately
from dense Q8, selected-MoE, Q6 head, recurrent state, and reusable scratch.
G4's 512 ordering (dense Q8 44.25%, selected-MoE 21.72%, attention 10.68%, Q6
10.06%) is a hypothesis, not a 128K profile. Choose one exact family-local
change from the new table and stop if its end-to-end ceiling is immaterial. If
24 GiB-class GGUF support is required beyond 32K, prove the chosen KV/scratch
policy exact and report its speed/memory trade rather than comparing tracked
hipEngine allocation with whole-device llama.cpp GTT.

#### R7: Matched Q6 HIP/Vulkan

V11 requires one shared Q6_K algorithm/layout/output contract at rows 1/8 and
2048->152064. Today HIP T16 and Vulkan q8_1/X8 perform different work, so their
ratio is prohibited. Implement the missing peer path, validate the actual timed
N-repetition command and stable value/index output, then run both serialized
latency and independent-throughput modes. Reopen V12 only if the matched result
favors Vulkan and the corresponding production profile says Q6 is material.

#### R8: PARO DFlash To Greater Than 1x AR

The current
[`S4 exact row`](../benchmarks/results/2026-07-11-sol-s4-gfx1151-paro-dflash-profile.json)
needs a qualitative premise change, not tuning around the edges. It is
`3.3073 s` versus AR `0.4903 s`, so it needs a 6.745x speedup or
85.18% wall reduction. Draft alone is `0.8337 s` (1.70x the complete AR wall),
target verify is `2.4679 s` (5.03x), only 1/114 proposals is accepted, and the
target evaluates 5.6875 rows/output. Even the incorrect graph-reusing
branch-copy route reaches only `0.2214x` AR and still needs 4.52x.

The required order is:

1. Obtain or train a materially higher-acceptance drafter for the exact
   target/tokenizer and validate acceptance/quality on every mtp-bench category
   plus heldouts. Do not tune a fixed prompt or widen verification at the
   current rejection rate.
2. With improved acceptance, implement transactional scratch/prefix state and
   KV commit/rollback that is byte-exact to canonical c1 replay. This must
   enable repeat-shape graph hits without S5's token-1 divergence.
3. Reprofile. Attack target linear layers (37.41%), drafter decoder+LM-head
   (25.55%), and canonical replay/canonicalization (20.80%) in that measured
   order. Commit scatter and readbacks are immaterial today.
4. Consider wider draft/verifier batching only after a multi-request profile
   shows serialization or group caps dominate and the full route is already
   near break-even.

#### R9: Final Rerun And README Publication

Do not spend another full four-engine sweep merely to confirm an unchanged
default. R0 is the exception that activates an immediate point-release handoff:
once its PARO decode defaults settle, run the narrow R0 gates and the
repository-wide `uv run pytest -v` milestone gate, refresh the affected release
tables, and make the release decision without waiting for R1-R8 or the deferred
candidate queue. The last runtime milestone passed all 5,997 collected tests at
`8d0e0f24`; later benchmark/docs work passed its focused suites, so any new
runtime/kernel change requires a new full pass.

Then rerun the complete gfx1151 four-engine six-shape protocol and regenerate
the separate prefill, decode, and peak-memory tables. Rerun the exact/default,
`llama-compat`, and llama.cpp HIP speculative lanes if MTP changed, and the exact
production c1-c8 concurrency matrix if scheduling changed. Update compact
artifacts, `benchmarks/README.md`, `benchmarks/CHANGELOG.md`, the synchronized
root README, and this ledger in the same promotion unit.

## Non-Negotiable Gates

Every optimization unit must satisfy all applicable gates:

1. **Exact workload identity.** Record model fingerprint/revision, quant, KV
   dtype, prompt token IDs or prompt-suite hash, context/generation lengths,
   concurrency, choices, sampling mode, and speculative mode.
2. **Exact runtime identity.** Record configured and resolved backend, target
   arch, GPU, ROCm/HIP/compiler versions, build profile, hipEngine commit, and
   full dirty state including staged and untracked files.
3. **Correct denominators.** Count generated token IDs across every choice.
   Keep visible-text re-tokenization only as a separately named compatibility
   diagnostic. Never use it as backend throughput.
4. **Owned timing.** Every timing payload declares `timing_scope`,
   stable `batch_id` where applicable, and `group_rows`.
   Batch timing is counted once, never once per choice.
5. **Correctness before speed.** Math/kernel changes need RED coverage where
   practical, the relevant CPU/reference or generated-token oracle, finite
   outputs, and the documented KL/top-1 gate. State/KV/graph changes also need
   multi-step state equality, not only final top-1.
6. **End-to-end before promotion.** A microbenchmark or profiler sub-window may
   justify keeping an exact low-level win, but it does not support a server or
   engine throughput claim without the matching end-to-end gate.
7. **Architecture isolation.** A gfx1100 result cannot select a gfx1151 default,
   or vice versa. Unverified architecture rows are explicit.
8. **No hidden fallback.** Artifacts record requested and effective attention,
   linear-attention, MoE, projection, sampler, graph, and speculative modes.
9. **No benchmark gaming.** MTP/acceptance work uses the complete
   `mtpbench-code-general-ja.jsonl` category suite plus held-outs and
   a true same-protocol AR baseline.
10. **Atomic retention.** Accepted performance work updates the compact artifact,
    `benchmarks/README.md`, `benchmarks/CHANGELOG.md`,
    `WORKLOG.md`, and default route in the same logical unit unless a
    concrete blocker is logged.

## Canonical Accounting Contract

The server and benchmark harnesses must distinguish four shapes:

- HTTP concurrency: number of simultaneous client requests;
- choices `n`: outputs requested by one HTTP request;
- backend group width `C`: live requests advanced together;
- verifier rows `V`: flattened speculative rows processed by the target.

Non-streaming hipEngine responses expose these as
`hipengine.generation_shape` schema v1. The route cap is an object with
`scope="queue_requests"`; it is never interpreted as a backend-row or verifier
limit. `queue_group` records the coalesced HTTP-request count, total prompt
rows, and this response item's row slice. `backend_groups[]` records each
actual generator call and any internal width split. `verifier_rows` is the sum
of target rows across those backend calls. The harness deduplicates repeated
response copies by `queue_group.id` and requires every group item exactly once.

Required generated-work fields:

| Field | Definition |
| --- | --- |
| `choice_generated_token_ids` | Exact token IDs emitted by each choice. |
| `choice_generated_tokens` | Length of that choice's exact ID list. |
| `total_generated_tokens` | Sum across all choices and requests. |
| `draft_tokens` / `accepted_draft_tokens` | Speculative work only; never substituted for visible output. |
| `target_rows` | Target model rows actually evaluated. |
| `retokenized_visible_tokens` | Optional decoded-text diagnostic, clearly non-authoritative. |

Exact-token direct/server comparisons also retain
`hipengine.prompt_token_accounting`: input type, per-row token-ID SHA-256,
per-row lengths, and total prompt tokens. The raw rows enter the common
`GenerationRequest` and bypass PARO/GGUF tokenizers; the hash echo and generated
ID oracle must match before timing is comparable.

Required timing scopes:

| Scope | Examples | Aggregation |
| --- | --- | --- |
| `choice` | Per-choice stop/sample/output handling | Sum only when measuring total per-choice work. |
| `batch` | Packed prefill, native decode step, draft/verify/commit phase | Deduplicate by `batch_id`. |
| `request` | Queue delay, TTFT, request wall | Report distribution; do not sum into GPU work. |
| `client` | Whole benchmark wall/makespan | Denominator for aggregate server generated tok/s. |

Primary server metrics:

```text
aggregate_generated_tok_s = total_generated_tokens / client_makespan_seconds
per_request_generated_tok_s = request_generated_tokens / request_wall_seconds
backend_batch_decode_tok_s = dedup_batch_generated_tokens / dedup_batch_decode_seconds
```

Also report TTFT, inter-token latency, completion latency, makespan, and
p50/p95. A batch-wide timing copied to six choices must still contribute once.

Required provenance:

```text
configured_backend, resolved_backend, target_arch, device_name
model_path, model_revision, model_fingerprint, quant, kv_dtype
hipengine_commit, staged_dirty, unstaged_dirty, untracked_dirty
rocm_version, hipcc_version, build_profile, exact command and env
timing_protocol, warmups, repetitions, profiler identity/status
```

The existing stronger dirty-tree handling in
`scripts/gguf_mtp_category_bench.py` should become shared
infrastructure rather than being reimplemented inconsistently.

## Architecture And Shape Identity

gfx1151 may reuse gfx1100 source bodies, but it must not reuse gfx1100 semantic
identity. The resolved backend and target arch must flow through generator,
runner/session, build, registry resolve, tuning selection, telemetry, and
artifact creation.

Do not mechanically relocate every physical import from
`hipengine.kernels.hip_gfx1100`. Those modules are the shared source
lineage used by the gfx1151 alias layer. Remove semantic hard-codes instead:

- generator registry keys fixed to `hip_gfx1100`;
- model defaults fixed to `hip_gfx1100`;
- registry `resolve()` calls fixed to `hip_gfx1100`;
- wrapper/build defaults that ignore the resolved target;
- capability/provenance surfaces that report configured `auto` instead
  of the resolved backend.

Use one immutable architecture tuning profile selected at model/session build
time. It may contain chunk sizes, workgroups, rowtile limits, route caps,
attention splits, and graph policies. It must be keyed through registry/config
composition, not hot-path `if backend == ...` branches.

Any c>N algorithm decision uses at least:

```text
resolved backend + target arch + model fingerprint + quant + KV dtype
+ rows + context bucket + mode + active-mask shape
+ attention + linear-attention + MoE + projection + sampler + graph variants
```

An unknown key falls back to the serial/exact route and reports why.

## PARO/GGUF Parity Audit

This table prevents a win in one path from being forgotten in the other while
also preventing incompatible quant kernels from being copied blindly.

| Surface | PARO today | GGUF today | Required comparison / transfer |
| --- | --- | --- | --- |
| Backend identity | PARO has gfx1100/gfx1151 factories and carries backend/target arch. | `SOL-B1` tags resident GGUF models and every weight with the resolved backend; embedding, linear/fused-linear, router, GDN, and compact/sidecar MoE resolves rebind shared gfx1100 source templates to that identity. A live gfx1151 public smoke retained `hip_gfx1151` through generator/runner/model/weights and generated token ID `11`. | Backend identity is complete; B2 is parked until an exact A/B selects a concrete architecture value. |
| Prefill chunking | gfx1151 all-256 chunking was a large diagnostic win. | SOL-G2 certifies the raw-Q/K exact GDN chain across segment/chunk boundaries, but clean G3 walls reject it at 512/4K (+5.19%/+6.70%). | Keep fused; G7 parks undirected chunk/threshold sweeps unless a materially different exact candidate and named bucket appear. |
| Decode graph | PARO has graph/bucket infrastructure, with path-specific evidence. | SOL-G5 rebuilds the GGUF graph by full shape/state identity and passes 128 third-and-later state/token checkpoints; capture-inclusive wall retains a 0.112% win for long c1 greedy windows. | Keep path-specific keys; do not transfer a graph policy without exact replay and same-path wall evidence. |
| c>N decode | P1 rejects native gfx1151 c2-c8 and classifies production as exact width-1 sessions; P2 proves ragged sparse shrinking on that route. | S2 exposes route cap, queue/backend groups, and verifier rows. G8 is parked because client c8 realizes c4+c4+c2 and one of three runs has an intermittent c4-group mismatch; no true multi-row AR algorithm exists. | Reactivate each path only with its own independent-c1 hidden/state/KV oracle and a general algorithm. |
| Sparse slots | P2 proves sorted sparse physical slots through EOS and front/middle/tail cancellations without compaction. | GGUF actual groups are observable, but true multi-row AR sparse-state semantics remain part of parked G8. | Preserve PARO's exact serial lifecycle; require the full G8 shrink/sparse gate before GGUF native promotion. |
| Full attention | PARO c2-c8 candidates are correctness-red and production is explicitly serial; ragged prefill uses `per_segment_ragged_exact`. | GGUF eager/full-attention state is exact for G1/G5 c1; packed multi-row behavior remains behind G8. | Bucket context, row width, reducer, KV ABI, and fallback separately; never select from a rejected width. |
| GDN/linear state | PARO has segmented multi-row state work plus shape-specific fallbacks. | GGUF eager state is exact; the raw-Q/K split passes G2 but loses balanced G3 full-prefill wall at both primary contexts. | Keep fused as the GGUF default and preserve the exact chain as the required unfused fallback/bisection path. |
| MoE | Selected-c1 and grouped-compact remain named diagnostics, but P1 rejects both as general native c>N production algorithms. | GGUF uses Q*_K/T16/X8 and dp4a-specific selected paths. | Transfer row/group policy and measurement, not quant kernel bodies; no rejected PARO width may select GGUF or PARO defaults. |
| Projection | P1 catalogs every gfx1151 width under the full model/quant/KV/context identity and classifies production serial. | GGUF G6 proves replacement-only residency; selected/dense paths remain quant-specific. | Compare true weight reuse versus row-GEMV only after an exact multi-row profile exists. |
| LM-head/sampler | PARO has exact serial fallback; current native widths are rejected. S4 synchronized DFlash attribution measures the target and drafter heads explicitly. | G4 profiles Q6 head/top1 at 10.06%; G10's dominance trigger is false. | S7 rejects fused DFlash target head and readback work; reactivate only on a changed measured bucket. |
| Speculative lifecycle | S4 supplies a clean real GPU row with coarse/fine buckets and graph shapes; S5 branch-copy is faster but fails at token 1, so canonical replay remains exact. | S2 supplies stable queue/backend/verifier identity; S1 keeps non-exact compatibility MTP explicit-only. | Current transfer audit is closed. Reopen commit/group/fusion work only after exact/profitable route evidence changes. |
| Startup/cache | PARO/GGUF both contain warmup and resident-cache paths. | G5 records GGUF capture/hit/replay identity; S4 records DFlash verifier shape misses and the non-exact branch-copy hit control. | Keep cache evidence path-specific; future claims must continue to separate cold, warmed, hit, and miss behavior. |
| Memory residency | PARO packed rows are near the consumer-card target. | SOL-G6 cleanly audits gfx1151 Q4_K_M p512/d128 at 21.478 GiB owned/tracked: 733 unique sources, no raw+replacement duplicates or enabled optional sidecars, and 2.522 GiB margin to 24 GiB. | Keep the replacement-only default; context-specific long-KV capacity remains a separate policy gate. |

GGUF Q*_K, T16/X8, q8_1/dp4a, and Q6 LM-head kernels are not PARO
`w4_paro` kernels. Transfer scheduler, lifecycle, shape, graph,
warmup, and accounting lessons. Transfer device math only after matching the
layout and profiled bottleneck.

## P0 Foundation Punchlist

| ID | Work | Status | Dependencies | Exit gate |
| --- | --- | --- | --- | --- |
| `SOL-E1` | Carry exact generated IDs/counts through `GenerationOutput` and OpenAI responses; aggregate every choice in `mtp-bench.py`. | `accepted`: PARO/GGUF outputs carry exact IDs; completion/chat `n=6` and retokenization-mismatch regressions pass; `mtp-bench.py` validates and aggregates all rows; API/benchmark semantics are documented. | none | Retokenization-mismatch and `n=6` regressions prove exact all-choice totals; usage semantics are documented. |
| `SOL-E2` | Add `batch_id`, `group_rows`, `timing_scope`, and timing owner; deduplicate batch metrics in harnesses. | `accepted`: the telemetry contract defaults unscoped timing to explicit choice ownership and requires complete batch ownership; PARO/GGUF live c2 groups expose one owner; `mtp-bench.py` rejects malformed ownership and deduplicates copied batch walls by ID. | none | Synthetic duplicate payload and live PARO/GGUF group tests count batch wall once. |
| `SOL-E3` | Create shared artifact/provenance helpers; detect backend/arch dynamically; include full dirty state and model fingerprint. | `accepted`: the stdlib/torch-free collector emits `hipengine_artifact_provenance` v1 for server, retained PARO, GGUF category/true-AR, and micro artifacts; dynamic gfx1151 identity resolves from `auto`; staged, unstaged, untracked, snapshot-revision, file, directory, and missing-model cases have regressions. | none | Server, PARO retained, GGUF, and micro artifacts satisfy one schema; staged/untracked tests pass. |
| `SOL-E4` | Repair dashboards: remove `performance_claim=false` rows from "Current fastest," correct server token headlines where raw IDs suffice, and mark timing rows awaiting rerun. | `accepted`: the canonical and root current-topline tables contain only retained speculative rows; stale/false-claim model, capacity, and concurrency numbers are replaced by linked rerun notices; historical server rows explicitly await an exact-ID/scoped-timing rerun. | E1-E3 | Current tables contain only eligible rows; diagnostics remain linked in a separate section. |
| `SOL-E5` | Add an exact-token server benchmark route shared by PARO/GGUF direct and server runs. | `accepted`: raw token rows are a common generation input; OpenAI `int[]`/`int[][]` prompts preserve IDs through batching and `n`; prompt hashes/counts and exact generated IDs are exposed; the shared tool/schema fail closed on parity; live gfx1151 PARO 512/128 direct/HTTP matched all IDs. | E1, E3 | 512/128 token-ID prompts produce the same prompt IDs and generated-ID oracle through direct and HTTP paths. |
| `SOL-B1` | Register GGUF for `hip_gfx1151` and thread resolved backend/target through generator, runner/session, registry resolves, builds, capabilities, and telemetry. | `accepted`: backend packages have a refreshable registration hook; the GGUF generator/runner defaults resolve `auto`; resident models/weights carry the concrete backend; embedding, single/fused linear, router, GDN, and compact/sidecar MoE registry paths use it. AST regression rejects literal gfx1100 resolver arguments, lazy gfx1151 reconstruction passes, and a live public gfx1151 smoke generated ID `11` with all layers tagged gfx1151. | E3 | gfx1151 factory/dispatch/build tests pass; no semantic gfx1100 resolver key remains on the selected path. |
| `SOL-B2` | Add registry/config-owned architecture tuning profiles without changing defaults. | `parked`: B1 and the corrected gfx1151 profiles produce no winning cross-cutting architecture value to carry. Adding an empty/equal profile would be dead indirection; retained changes already use registry variants or narrowly evidenced route policy. Reactivate when an exact same-device A/B selects a concrete gfx1151 value that cannot live in an existing registry key. | first retained architecture-specific value | Empty/equal profiles are behavior-identical; future gfx1151 values require same-device evidence. |
| `SOL-M1` | Add one matrix driver/report that joins exact tokens, scoped timings, path variants, latency, memory, and profiler summaries. | `accepted`: manifest/schema v1 normalizes PARO/GGUF direct/server rows, recomputes rates from exact IDs, deduplicates timing owners, preserves backend/verifier shapes, attaches memory/profiler artifacts, and rejects forged denominators or cross-scope ratios; the four-surface contract and real SOL-E5 PARO diagnostic smoke pass. | E1-E3, E5 | One artifact can compare direct/server and PARO/GGUF without manual denominator repair. |
| `SOL-D1` | Split the three source docs into a short current dashboard and dated lab notebook/history; reconcile stale concurrency and "Done" wording. | `accepted`: MTP parity, PARO transfer, and HIP/Vulkan are 94/87/87-line current dashboards with retained/diagnostic/open/blocked language; their 6,812/597/2,602-line notebooks are linked `*-HISTORY.md` files whose blob hashes exactly match the originals. | E4 | Each current dashboard contains only eligible results and open blockers; historical diagnostics remain linked and unchanged. |

The P0 foundation is accepted. The initial local baseline matrix is complete;
retain the matrix below as the rerun contract before future architecture
tuning. Do not combine backend plumbing with kernel tuning.

## Baseline Matrix

The first gfx1151 pass is represented by P1/P2, G1-G6, S1-S7, and the accepted
topline. Use these rows as the minimum matrix after a relevant route changes.
gfx1100/W7900 remains a separate rerun when that hardware is available. Never
merge the architectures into one dispatch decision.

### AR Correctness And Throughput

| Matrix | Required rows | Purpose |
| --- | --- | --- |
| Short concurrency | c1-c8, prompt 512 / decode 128, every integer width | Find odd-width/c6 holes and record the actual backend group. |
| Mid-context concurrency | c1/c2/c4/c8, prompt 4K / decode 128 | Exercise attention/context buckets without exploding matrix cost. |
| Long context | c1 first at 32K/64K/128K; c2/c4 only after c1 is green and memory-safe | Validate chunk/KV policy and architecture-specific memory limits. |
| Dynamic shrink | Start c8 and force completion/cancel transitions through c7...c1 | Prove state/KV/slot correctness under live shape changes. |
| Sparse slots | Holes at front/middle/tail with sorted physical slots | Prove native decode is not accidentally compact-from-zero only. |
| Ragged context | Mixed prompt lengths within one group | Exercise per-row positions, spans, attention, and graph keys. |
| Sampling | Greedy first; then supported per-row normal sampling | Keep sampler correctness separate from core AR bring-up. |

For every c>N row report aggregate/c1, per-request/c1, native/serial, actual
group histogram, active occupancy, TTFT, inter-token latency, completion
p50/p95, makespan, memory, and exact generated-ID equality versus independent
c1.

### Speculative Economics

Use all categories in
`benchmarks/prompts/mtpbench-code-general-ja.jsonl` plus held-outs.

| Matrix | Required rows | Purpose |
| --- | --- | --- |
| Natural short horizon | c1/c2/c3/c4/c8, `max_tokens=24`, true AR and MTP | Establish immediate auto-routing policy with actual backend group widths. |
| Longer horizon | At least `max_tokens=64/128` for routes that survive natural24 | Measure setup amortization and avoid fixed-cycle conclusions. |
| Context buckets | Short and 4K first; longer only if route remains positive | Decide whether routing depends on attention/context cost. |
| Budget | Exact/default budgets first; compat is a separately labeled lane | Never merge accuracy-traded and exact economics. |
| PARO DFlash | Same prompt categories, same target AR protocol | Compare verifier/drafter lifecycle rather than headline from another path. |

Required speculative metrics include visible outputs/cycle, accepted/output,
target rows/output, draft/verify/commit wall, group width, route decision reason,
and a true no-spec AR baseline.

## GGUF Recovery And Optimization

| ID | Work | Status | Dependencies | Exit gate |
| --- | --- | --- | --- | --- |
| `SOL-G1` | Build a teacher-forced token, hidden, recurrent-state, and KV oracle for eager GGUF decode across at least four steps. | `accepted` on gfx1151 at `c941c158`: the exact Q4_K_M `[9707] * 512` prompt and five-token continuation match llama.cpp; production bulk/eager tokens match; positions 513-516 are finite and byte-exact against fresh serial prefixes across every layer output, 30 Conv/GDN pairs, and 10 live K/V pairs. The old W7900 performance row still requires a hardware-local rerun. | E3, B1 | Repeated-token/current eager behavior is classified as correct or localized to the first divergent layer/state. |
| `SOL-G2` | Add explicit GDN prefill `fused|chain|auto` diagnostic selection. Reproduce the 17-token mismatch and bisect first hidden/recurrent divergence. | `accepted` at `332f01f8`: the RED localized normalized-Q/K materialization as the first layer-0 recurrent divergence. The GGUF-only raw-Q/K-plus-scale exact split passes 6/6 clean gfx1151 cases: greeting, 512, 1024/1025 segment threshold, and 4095/4096 four-chunk boundary. Sampled tokens, FP32 hidden seeds, and resident Conv/GDN state are byte-exact; greeting/512 all-layer rows are exact. Focused tests pass 48/48 and the expected zero-scratch kernels appear in a cached-only trace. | G1 | Chain matches target tokens/state at short, 512, 4K, segment, and chunk boundaries. |
| `SOL-G3` | Promote the split prepare + segmented-k2 + RMSNorm chain only if same-run wall wins. | `rejected` at clean `ad773eba`: exact timed tokens and the linked G2 state/trace gates pass, but four balanced repetitions show chain `1248.436` vs fused `1186.842 ms` at 512 (+5.19%) and `10870.022` vs `10187.300 ms` at 4K (+6.70%). Fused remains default. | G2 | Exact state/tokens plus prefill wall win on both primary contexts; expected kernel trace present. |
| `SOL-G4` | Bisect correct eager decode against the last fast revision and profile the correct route by layer family. | `accepted` on gfx1151 at clean `5f4c6561`: p512/d128 exact eager is 49.285 tok/s; direct-parent `4499fb13` is the 17.799 -> 54.963 tok/s (+208.79%, 3.088x) library-cache boundary; current p8 is 55.208 tok/s. Twenty-four exact ROCTX windows yield the decode-only Amdahl table below. W7900 remains blocked on hardware. | G1 | Correct eager baseline, first performance-changing revision, and Amdahl table are recorded. |
| `SOL-G5` | Rebuild correct graph replay by full shape/state key; test third-and-later replay explicitly. | `accepted` on gfx1151 at clean `7f611fe3`: the production graph is exact for all 128 hidden/Conv/GDN/KV/token checkpoints and capture-inclusive wall improves same-run eager `20.3343 -> 20.3115 ms/token` (+0.112% throughput). Admission is gfx1151-only, non-streaming c1 greedy, and at least 128 remaining transitions. W7900 remains blocked on hardware. | G4 | Eager/graph hidden, recurrent state, KV, and tokens match over long replay; wall beats eager. |
| `SOL-G6` | Audit replacement layout residency and eliminate raw+packed duplicates where the replacement path is complete. | `accepted` on gfx1151 at clean `d70c9464`: 733 unique source tensors map to one resident layout each, with zero raw+replacement duplicates and zero enabled optional sidecars. The p512/d128 BF16-KV production graph session is 21.478 GiB owned/tracked (2.522 GiB under 24 GiB); graph/exec adds 0 tracked bytes and 308 KiB sampled HIP residency. G5 is linked by SHA-256 for exact speed non-regression. | E3 | Allocation census names raw/packed/KV/scratch/graph bytes; 24 GiB-class goals are checked without speed regression. |
| `SOL-G7` | Tune gfx1151 chunk, workgroup, rowtile, attention split, and route thresholds. | `parked`: the corrected G4 profile supplies no safe broad-knob premise. G3 rejects the alternate GDN prefill chain, G9/G10 triggers are false, and the retained G5 graph threshold is already scoped by exact wall evidence. Reactivate only for a named dominant family and concrete candidate; do not run an undirected threshold sweep. | changed G4-family profile plus exact candidate | Same-device exact A/B selects profile values; gfx1100 remains unchanged. |
| `SOL-G8` | Close the GGUF resident multi-row AR path across c1-c8 and sparse slots. | `in_progress`: packed prefill/decode is registered and the new direct natural10 gate matches independent c1 generated tokens for 3/3 c10 repeats through c4+c4+c2 chunks. The prior intermittent server mismatch did not reproduce. Hidden/Conv/GDN/KV, ragged/shrink/sparse, cancellation, long-context, and profiler/scaling evidence are still absent. | executable token gate landed; deeper path-local lifecycle oracle next | Exact c1-c8, shrink, sparse-slot, and profiler gates pass with aggregate scaling. |
| `SOL-G9` | Narrow HIP Q4 selected-dual recovery using source/layout/reduction/waitcnt changes. | `parked`: corrected V6 serialized Q4 is parity/HIP-favored (`0.922x-0.973x` Vulkan/HIP), so the activation premise is false; G4 also identifies dense Q8 and selected-MoE GEMV as the dominant production families. | corrected V6 result and real profile | Activate only if serialized matched Q4 still favors Vulkan and Q4 is material in production wall. |
| `SOL-G10` | Four-wave Q6 verifier LM-head rowtile: each wave owns four output columns to reduce accumulators. | `parked`: G4 attributes 10.06% of eager GPU time to Q6 LM-head, behind dense Q8 and selected-MoE; the exact small-B rowtile is already retained, while the later rowtile+top1 server experiment was flat/rejected. The “Q6 remains dominant” trigger is false. | E2, corrected profile | Activate only if Q6 head remains dominant; rowtile candidates R6/R8/R12 show no spills, exact output, lower GPU event time, and server wall win. |

Do not restore the old GGUF graph as a shortcut. SOL-G1 proves the repeated
`9707` stream is valid on gfx1151; it still does not make a stale timing row
eligible or replace the full prompt-suite quality/performance gates.

### SOL-G4 gfx1151 eager Amdahl

The retained audit uses clean `5f4c6561`, exact Q4_K_M, BF16 KV,
`[9707] * 512`, four decode warmup steps, and 24 synchronized eager steps.
Only kernels fully contained in each ROCTX step range enter the table. Kernel
sum is **18.402 ms/token** versus **20.766 ms/token** profiled host wall
(**88.62%**); the unprofiled four-run p512/d128 median is **20.290 ms/token**.

| Family | GPU us/token | GPU share | Overall if family 2x | 4x | Infinite |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense Q8_0 GEMV | 8142.998 | 44.25% | 1.284x | 1.497x | 1.794x |
| Selected-MoE GEMV | 3996.701 | 21.72% | 1.122x | 1.195x | 1.277x |
| Full-attention core/KV | 1965.390 | 10.68% | 1.056x | 1.087x | 1.120x |
| Q6 LM-head/argmax | 1851.998 | 10.06% | 1.053x | 1.082x | 1.112x |
| MoE router | 807.582 | 4.39% | 1.022x | 1.034x | 1.046x |
| GDN/linear attention | 705.191 | 3.83% | 1.020x | 1.030x | 1.040x |
| RMSNorm/RoPE | 532.355 | 2.89% | 1.015x | 1.022x | 1.030x |
| Dense BF16 GEMV | 192.909 | 1.05% | 1.005x | 1.008x | 1.011x |
| MoE combine/SiLU | 165.271 | 0.90% | 1.005x | 1.007x | 1.009x |
| Other + embedding + copies/fills | 41.641 | 0.23% | 1.001x | 1.002x | 1.002x |

The old 2026-06-29 whole-process profile included prefill and warmup and is not
the G4 Amdahl source. The full current table, top kernels, VGPR/scratch counts,
commands, trace hashes, and linked G1 SHA are in
[`2026-07-11-sol-g4-gfx1151-gguf-eager-decode-audit.json`](../benchmarks/results/2026-07-11-sol-g4-gfx1151-gguf-eager-decode-audit.json).

### SOL-G5 gfx1151 production graph

The retained clean `7f611fe3` audit exercises the production
`capture_decode_graph()` API rather than a benchmark-local reconstruction. Its
key covers backend/target, model and quant identity, layer/KV layout, route and
sampler, resident weight and buffer pointers, recording options, state
generation, and a bounded context/replay window. Eager and stable graph replay
match byte-for-byte through all 128 third-and-later launches; conservative
state-generation recapture also passes 128/128 but is too slow to retain.

| Route | Median ms/token | Throughput | Decision |
| --- | ---: | ---: | --- |
| Same-run eager | 20.3343 | 49.178 tok/s | Control |
| State-bound graph, capture inclusive | 20.3115 | 49.233 tok/s | **Retained, +0.112%** |
| Per-token state-generation recapture | 35.4290 | 28.225 tok/s | Rejected |

The graph row charges one capture/instantiate plus final destroy to each
128-token window. The strict 128-transition gfx1151 threshold and eager
fallbacks remain part of the claim. Full commands, samples, key, provenance,
and checkpoint hashes are in
[`2026-07-11-sol-g5-gfx1151-gguf-decode-graph-production-audit.json`](../benchmarks/results/2026-07-11-sol-g5-gfx1151-gguf-decode-graph-production-audit.json).

### SOL-G6 gfx1151 replacement residency

The clean `d70c9464` census runs the production p512/d128 graph session and
audits both the materialization plan and live owned buffers. All **733** source
tensors have one planned resident layout; there are **zero** same-source
raw+replacement duplicates and **zero** enabled optional raw/X8 sidecars.

| Resident family | GiB | Notes |
| --- | ---: | --- |
| Replacement weights | 20.461 | Q4/Q5/Q6/Q8 T16 replacements; no optional X8 sidecar |
| Required raw GGUF | 0.503 | Device token embedding, not a duplicate |
| Dense weights/metadata | 0.097 | F32/BF16 residents |
| Decode scratch | 0.080 | 15 MiB BF16 KV, 63.75 MiB linear state, metadata/other |
| Session/prefill buffers | 0.337 | 0.330 GiB bulk-prefill scratch dominates |
| **Owned/tracked total** | **21.478** | **2.522 GiB below the 24 GiB gate** |

The production `record_steps=0` graph owns no tracked `DeviceBuffer`; the
synchronized live-minus-closed `hipMemGetInfo` delta is **315,392 bytes**
(308 KiB) for HIP graph/exec internals. Tracked allocations return exactly to
their pre-load baseline after session close. The artifact links the accepted
G5 SHA-256 for 128-launch exactness and capture-inclusive speed non-regression;
G6 itself makes no new throughput claim. Full allocation maps, close deltas,
commands, and clean provenance are in
[`2026-07-11-sol-g6-gfx1151-gguf-residency-audit.json`](../benchmarks/results/2026-07-11-sol-g6-gfx1151-gguf-residency-audit.json).

## PARO Concurrency And Optimization

| ID | Work | Status | Dependencies | Exit gate |
| --- | --- | --- | --- | --- |
| `SOL-P1` | Run the exact c1-c8 512/128 matrix and publish a full shape/algorithm catalog per architecture/model. | `accepted`, with a 2026-07-13 gfx1151 c2 follow-up: the original clean catalog still rejects every fully native c2-c8 candidate. The new production-safe greedy-BF16 c2 hybrid explicitly combines batch-GEMV linear output, selected-c1 batch MoE, and row-local full-attention pre-O work. It matches independent eager c1 for both p512/d128 fixture rows and all 10 canonical category prompts at d8, without serial layer decode. It remains `throughput_claim_eligible=false`; c3-c8 and gfx1100 still select true c1. | none | Every width is green or explicitly serial; no gfx1100 evidence silently selects gfx1151. |
| `SOL-P2` | Run c8->c1 EOS/cancel shrink, ragged contexts, and sparse-slot transitions. | `accepted` on gfx1151 at clean `6f1910c9`: exact prompt lengths `449,458,467,476,485,494,503,512` shrink c8-to-c1 without compaction, with slot 3 retiring by EOS, explicit cancellations creating middle/tail/front holes, and slot 4 surviving. All eight generated sequences, all 30 Conv/GDN state pairs, and all 10 live K/V layer pairs match independent c1 SHA-256 at each retirement boundary; every post-event width is exact and no group-wide cancellation occurs. Ragged packed linear/full-attention prefill automatically uses `per_segment_ragged_exact`; equal-length packed routes are unchanged. `performance_claim=false`. gfx1100 remains unverified. | P1 | Per-row state/KV/output identity matches independent c1; no group-wide cancellation. |
| `SOL-P3` | Remove the generator's compact-from-zero gate and use sorted sparse physical slots accepted by `step_batch_native()`. | `parked`: P2 proves sorted sparse physical-slot addressing and exact lifecycle behavior on the production true-c1 scheduler. P1 rejects every native c2-c8 route, so removing the native admission gate would only expose uncertified shapes; width-1 sessions do not compact live slots. | reactivate after a general exact native c>N algorithm | Holey live groups stay native and exact; no serial fallback solely because of slot holes. |
| `SOL-P4` | Make selected-c1 MoE a named multi-row algorithm and compare it with grouped-compact at every supported width. | `partially accepted for gfx1151 c2`: selected-c1 batch MoE is now one required component of the exact c2 hybrid and is selected explicitly rather than through an experiment env. Broader widths and a retained wall/profile comparison remain parked because c3-c8 are still correctness-ineligible. | retained c2 profile or another correctness-eligible native width | Full-layer/server wall, routed-lane counts, and correctness select the mode; c6's current advantage is rechecked. |
| `SOL-P5` | Close odd-width and c6 attention/linear/MoE/projection/sampler shapes with full identity keys. | `accepted by explicit serial classification`: the clean P1 catalog keys gfx1151/model/quant/KV/context/width and rejects c3/c5/c6/c7 native rows at the same index-2 gate. Production maps each to width-1 graph groups; rowchunk, selected-c1, grouped, projection, and sampler diagnostics cannot auto-select. | P1 | c3/c5/c6/c7 are retained-safe or serial; unproven rowchunk/grouped routes cannot auto-select. |
| `SOL-P6` | Benchmark c6 direct versus sequential c4+c2 splitter with all-choice counts and latency distributions. | `rejected`: P1 rejects direct c6 and both c4/c2 native components against independent c1, so no latency distribution from either branch can select production policy. The default-off `HIPENGINE_QWEN35_AVOID_C6_GROUPS` flag and split/remap path are removed; production keeps one server group and the exact width-1 scheduler owns execution. | reopen only with a correctness-eligible c6/c4/c2 route and named latency objective | Keep a splitter only for an explicitly chosen latency objective; aggregate throughput, makespan, and p95 are non-regressive for that policy. |
| `SOL-P7` | Capture/replay decode buckets keyed by active rows, context, mask, variants, and replay length. | `parked`: the retained harness records shape-bucket metadata, but P1 leaves no exact native c>N execution to capture. The accepted c1 state-bound graph remains the only replay route; manufacturing c>N cache hits would replay rejected math. | reactivate after an exact retained c>N shape | Cache hit/miss/fallback telemetry is complete; exact replay improves server wall for retained shapes. |
| `SOL-P8` | Retune gfx1151 prefill chunks, AOTriton threshold, projection, and sampler modes after the route is shape-safe. | `parked`: P1 supplies no correctness-eligible c>N route, and B2/baseline tuning remains blocked. Architecture-specific threshold search cannot select from rejected shapes. | B2, corrected baseline, and an exact native c>N route | Same-device profile and end-to-end matrix select values without gfx1100 regression. |
| `SOL-P9` | Replace row-parallel GEMV with weight-reusing MMQ/GEMM/WMMA/grouped algorithms where c>N profiles justify it. | `parked`: c2 now has an exact hybrid, but it is not throughput-retained or profiled and deliberately uses the correctness-backed batch-GEMV linear output. c3-c8 remain rejected, so there is still no eligible weight-reload/occupancy trigger for a replacement algorithm. | reactivate when a corrected exact c>N profile shows material weight reload/occupancy | Activate per family when weight reload/occupancy is material; prove c1 non-regression and c>N aggregate wall win. |

The P3-P9 activation audit leaves no eligible native c>N optimization premise.
The schema-1 c2-c8 profile has `performance_claim=false` and an invalid
batch-shaped oracle, while the clean P1 catalog independently rejects every
current native width and records production as `scheduler_true_c1_fallback`.
The c6 splitter was removed because direct c6 and sequential c4+c2 both depend
on rejected native widths; setting its former env variable has no effect. A
future schema-2 profile must pass independent-c1 packed-prefill, sparse-slot,
ragged, and shrinking gates before the planner can select native groups.

For c9-c16, `scripts/qwen35_batch_retained_bench.py
--batch-decode-execution=profile_partitioned` remains a diagnostic driver. Use
`--batch-decode-execution=serial` for the matched fallback control and
`direct_native` only for correctness localization. Production c9+ requests use
reused width-1 sessions until an accepted schema-2 profile exists.

## MTP, DFlash, And Routing

| ID | Work | Status | Dependencies | Exit gate |
| --- | --- | --- | --- | --- |
| `SOL-S1` | Move `auto` MTP choice from per-request eligibility to the realized backend group. | `accepted`: the clean-tracked gfx1151 full-suite/heldout matrix at `d2b1e742` records exact IDs and realized groups. The compatibility hook is diagnostic-fast at c1/c2 but is not true-AR exact; isolated heldout c3 groups are also 3.92% slower. Automatic requests now carry a `speculative_mtp_auto` intent into the realized queue group, then select exact/default AR with reason, group-width, horizon, and evidence telemetry. A live gfx1151 smoke reports default/verifier-rows 0 for auto and preserves explicit compatibility-MTP/verifier-rows 3. | E1-E2 and corrected natural matrix complete; exact/default server MTP unavailable | Auto never selects a non-exact compatibility route; explicit opt-in always requests MTP. Policy records reason/group/horizon and can admit widths only after exact/default evidence exists. |
| `SOL-S2` | Record route cap, actual backend group, queue grouping, and verifier rows separately. | `accepted`: non-streaming server responses emit `generation_shape` v1 with a request-scoped cap, queue-group ID/request/prompt counts and item slice, actual backend call widths, and deduplicated verifier rows; `mtp-bench.py` validates complete groups. The c8 regression produces two c4 queue/backend groups, never a width-8 verifier row. | E2 | A c8 client row cannot be mistaken for a width-8 verifier row. |
| `SOL-S3` | Add context/output-length buckets and EWMA hysteresis only after static policy is stable. | `parked`: S1 admits no exact automatic MTP route, so there is no static MTP policy for context/output buckets or hysteresis to improve. Reactivate only after an exact/default route passes the full-suite heldout gate and establishes a stable static policy. | exact/default MTP route plus retained S1 static policy | Online policy beats/equals static on held-out full-suite traffic without prompt-conditioned branches. |
| `SOL-S4` | Run a real PARO DFlash row using the landed coarse phase and graph-shape telemetry. | `accepted as a profile; speed rejected` at clean `8eb27215`: curated 35B PARO/DFlash B4 produces the exact 32-token AR sequence with finite logits, but `9.676` versus `65.266 tok/s` is only `0.14825x` AR. Target verify is 74.62% of wall; synchronized attribution identifies target linear layers (37.41%), drafter decoder+LM-head (25.55%), and canonical replay/canonicalization (20.80%) as the parent buckets. Exact replay records 30 validated graph misses and zero hits across two shapes. | E1-E3 | Same-session AR, exact output, phase coverage, and shape hit/miss data identify the dominant parent bucket. |
| `SOL-S5` | Compare GGUF deferred accepted-row scatter/tail discard with PARO verifier commit/canonicalization. | `rejected on correctness`: branch-copy removes exact replay work, enables 27 graph hits after two captures, and moves `9.676 -> 14.450 tok/s` (+49.34%), but diverges from AR at generated token 1. Direct bulk state therefore cannot replace canonical c1 replay; tiny commit-scatter/final-sync buckets do not justify a separate scatter port. | S4 profile | Activate only if commit/scatter/sync is material; exact state/KV and cycle/server wall must improve. |
| `SOL-S6` | Add true draft-side batching and/or wider verifier groups. | `parked`: the real row accepts only 1/114 proposed draft tokens and spends 5.6875 target rows/output. Wider verifier groups would amplify rejected work, while a single-request row establishes no multi-request draft serialization or group-cap bottleneck. Reactivate only after a full-suite drafter-quality gate plus multi-request phase evidence. | exact profitable static DFlash route plus multi-request profile | Activate only if current phase serialization/group caps dominate; retain on full suite and server wall. |
| `SOL-S7` | Re-evaluate LM-head/top1 fusion, readback, and sampler boundaries. | `rejected and parked`: current-shape fused target LM-head remains exact but moves `9.676 -> 9.177 tok/s` (-5.16%). Synchronized drafter top-k/readback and target accept-readback are only 0.407% and 0.042% of wall; the existing generic fusion stays default-off and no sampler/readback rewrite activates. | S4 synchronized attribution | Existing generic fusion/readback probes stay rejected unless a changed shape exposes the bucket again. |

Do not retry generic LM-head fusion, deferred readback, rowtile+top1, broad
route-cap increases, or confidence policies merely because attribution moved.
Require a changed premise and name it in the new artifact.

## HIP/Vulkan Measurement Repair

### Current Claim Classification

| Evidence | Current use |
| --- | --- |
| Dispatch/grid floor | Retained v2: Vulkan/HIP `1.162x-16.789x` serialized and `1.116x-150.459x` independent. This is runtime/submission evidence, not compiler evidence. |
| Geometry/reduction/sampler/two-stage | Retained v2: HIP wins or is mixed under required ordering; Vulkan wins independent overlap. Timing mode is part of the workload. |
| Synthetic packed dot | Retained v2 Vulkan lead: `3.052x-3.243x` serialized and `3.840x-4.272x` independent. |
| Production Q4/Q6/Q8 slices | Retained v2: serialized Q4 is parity/HIP-favored, Q6 is about `1.82x` HIP-favored combined, and dense Q8 is HIP-favored on every serialized row. |
| Q6 lm-head HIP T16 versus Vulkan q8_1/X8 | Blocked: different math/layouts, so no cross-backend ratio is permitted. |
| ISA dot4/VOPD/waitcnt/spill counts | Structural evidence remains valid. |

### Harness Contract V2

Implemented at `ca241dae` with `hipengine_dirty=false`; the executable contract is in
`benchmarks/micro/timing_contract.py`, the v2 schema, and the HIP/Vulkan runner
headers. The retained bounded artifact is
`benchmarks/micro/results/gfx1151/strix-halo/2026-07-10-hip-vulkan-timing-v2-bounded.json`.

1. Use `serial_latency` and `independent_throughput` modes.
2. In serial mode, add compute-to-compute execution and memory dependencies
   between every Vulkan repetition, including WAW and read-to-next-write hazards.
3. In independent mode, use disjoint outputs and compare with a HIP
   multi-stream/independent-graph path.
4. Record Vulkan GPU timestamps and host submit-to-fence wall separately.
   Record HIP event time and equivalent host wall.
5. Include `reps=1` plus a burst, and equalize warmup by dispatch count.
6. Validate the actual timed N-repetition command, not a separate one-dispatch
   command. Run Vulkan synchronization validation outside timing.
7. Match input bytes, selected IDs, algorithm/layout, output dtype, workgroup,
   cache state, and hot versus rotating working sets.
8. Extend the micro result schema with timing mode, dependency contract,
   timestamp metadata, memory flags, commit/dirty state, and claim eligibility.

### Bounded Rerun

| ID | Family | Corrected gfx1151 anchors | Status |
| --- | --- | --- | --- |
| `SOL-V1` | Harness/schema | Implement the contract above and a dependency litmus test. | `accepted` at `ca241dae` |
| `SOL-V2` | Dispatch | counts 1/50/941, grids 1/8192, reps 1 and burst, both modes/timings. | `accepted` on gfx1151 |
| `SOL-V3` | Geometry | K 512/8192, rows 1/8, wg 64/128/256. | `accepted` on gfx1151 |
| `SOL-V4` | Sampler | top-1/top-k8, rows 1/8, vocab 32768, wg 64/128/256. | `accepted` on gfx1151 |
| `SOL-V5` | Dot/memory | q8/q4 N=32768; coalesced-4 plus gather control; wg 64/128/256. | `accepted` on gfx1151 |
| `SOL-V6` | Q4 selected-dual | Active production layout, 4x32, 2048->512, HIP/Vulkan wg 64/128/256. | `accepted`: parity/HIP-favored |
| `SOL-V7` | Q6 selected-down | rows 8, 512->2048, wg 64/128/256. | `accepted`: HIP-favored |
| `SOL-V8` | Two-stage control | K 32768, rows 1/8, wg 128/256, split 4, serialized. | `accepted` on gfx1151 |
| `SOL-V9` | HIP independent control | Multi-stream/disjoint-output throughput against Vulkan independent mode. | `accepted` on gfx1151 |
| `SOL-V10` | gfx1100 portability | Repeat corrected anchors and every gfx1151 delta above 5%. | `blocked` on W7900 hardware |
| `SOL-V11` | Q6 LM-head | Same math/layout at rows 1/8, 2048->152064. | `blocked`: no matched math/layout implementation |
| `SOL-V12` | Production Vulkan probe | Persistent registry Q4/sampler object and real engine wall. | `parked`: corrected production slices do not justify backend work |

The gfx1151 matrix is complete. Run V10 on W7900 before transferring any ratio.
The gfx1151 result does not justify an LLVM issue, inline ISA program, or Vulkan
registry path.

## Profiling And Optimization Loop

For every performance item:

1. Select the highest actionable end-to-end or verified sub-window bucket.
2. State the hypothesis, affected shape keys, baseline artifact, expected
   movement, and stop condition in `WORKLOG.md`.
3. Add the narrow RED oracle before math/state changes, or log why RED-first is
   impractical.
4. Make one logical change. Keep a registered unfused/exact fallback.
5. Run the narrow correctness gate, expected-kernel trace, and same-suite A/B.
6. Use at least three timing samples for retention and apply variance rules.
7. If exact and non-regressive, promote the default and commit the artifact and
   rollups. If rejected, record the measured reason and remove or ledger the
   temporary path in `REFACTOR.md`.
8. Refresh this ledger status/result link before taking the next item.

Prioritize by wall reduction, but retain exact cycle-wall, verified sub-window,
launch-count, and H2D/D2H improvements even when aggregate variance hides a
small compounding win, as required by the project evidence policy.

## Execution Order

This is the active pickup order. Closed P0/P/G/S/V evidence remains available
in the item tables above; do not replay that historical order.

| Order | Work package | Items | Why now |
| ---: | --- | --- | --- |
| 1 | Establish production controls and close exact native concurrency | R3, R4 | The host cancellation boundary is fixed and both path-local RED/GREEN harnesses now execute. Finish serial controls, then solve PARO c2 exactness and GGUF state/KV/lifecycle coverage before tuning throughput. |
| 2 | Close exact GGUF MTP horizon economics | R2 | A no-code 64/128 rerun may change the natural24 premise; exact commit is bounded if it does not. |
| 3 | Profile and replace the GGUF prefill bottleneck | R5 | The old fast route is invalid; a current family profile must select a new exact candidate. |
| 4 | Localize long-context decode gaps | R6 | 128K cannot inherit the 512-token Amdahl table by assumption. |
| 5 | Implement and close matched Q6 | R7, V11 | Lower-priority backend work starts with the missing peer implementation; unmatched HIP/Vulkan ratios remain prohibited. |
| 6 | Reopen DFlash only when drafter quality changes | R8, S4-S7 | Current 0.14825x economics cannot be repaired by wider groups or readback tuning. |
| 7 | Revalidate deferred kernel/compiler candidates | Post-R0 queue | Run only after concurrency correctness exposes an eligible production bucket. |
| conditional | Reopen the reported PARO decode regression | R0 | Require a named, reproducible matched boundary; v0.3.0 publication removed the release-blocker status. |

V10's W7900 matrix is independent hardware work and may run whenever the GPU is
available. It does not reorder local gfx1151 recovery.

No remaining R item is implicitly a point-release blocker. R9 is a publication
refresh only after a retained default or claim changes.

Cross-GPU work is not allowed to block useful local gfx1151 progress, but no
gfx1100/gfx1151 shared default is promoted without both architectures or an
explicit architecture-specific profile.

## Definition Of Evidence-Sprint Closure

The evidence sprint was called complete only when:

- all P0 accounting, provenance, backend identity, and dashboard items pass;
- PARO and GGUF have exact current-HEAD baselines with trustworthy denominators;
- every PARO c1-c8 width and c8->c1 transition is retained-safe or explicitly
  serial with a named blocker;
- local gfx1151 GGUF repeated-token/state behavior is resolved and the correct
  eager path is profiled; GDN chain and graph are accepted or rejected with
  evidence; gfx1100 carries a named W7900-hardware rerun block;
- MTP `auto` uses actual group/horizon economics and explicit opt-in
  remains available;
- a real PARO DFlash profile either activates or parks each transfer candidate;
- every currently relevant conditional kernel is accepted, rejected, or parked
  by its trigger;
- HIP/Vulkan retained timing language is rebuilt from synchronized measurements,
  with throughput and latency kept separate;
- current dashboards contain only eligible claims, while rejected/diagnostic
  history remains discoverable.

The P3-P9 activation audit is closed. P2 supplies the exact production true-c1
lifecycle; P3/P4/P7-P9 are parked until a general algorithm changes P1's native
c2-c8 result, and P6 is rejected with its dead splitter removed. S1 now keeps
automatic compatibility MTP on exact/default AR and S3 is parked until an
exact/default MTP route establishes a stable static policy. The S4-S7 PARO
coordinator unit is now closed: the real DFlash row is exact but only
0.14825x AR, branch-copy is correctness-red, wider groups lack an activation
premise, and fused LM-head is slower. Fixture-specific rounding repair is not
an optimization target.

B2/G7 are parked because the corrected profile selects no concrete
architecture-specific value or broad-threshold candidate. G8 is parked behind
the intermittent realized-c4 exactness blocker and a missing general multi-row
AR algorithm. G9/G10 remain parked because their triggers did not fire. G2/G3
establish fused as the exact, measured GGUF prefill default.

## Definition Of Recovery-Phase Closure

The post-topline phase is pickup-complete only when each ready R item is
accepted, rejected, or parked on new evidence rather than restating the old
premise:

- R0 identifies the matched last-known-good/current PARO decode boundary,
  preserves all correctness/verification guarantees, and either recovers every
  affected release route or bounds the residual to named correctness-essential
  GPU/host work with an explicit point-release decision;
- R1 has a current-code exact forced-256 A/B at all six shapes and a per-bucket
  keep/reject decision;
- R2 has natural 64/128 full-suite and heldout true-AR evidence, plus an exact
  commit decision if the longer route still loses;
- R3 publishes exact serial c1-c8 server controls for PARO and GGUF, and R4
  either lands the general shape/lifecycle contract with path-specific exact
  c2-to-c8 math or records each first unresolved hidden/state/KV blocker with
  its RED fixture;
- R5 records a current fused-prefill family profile and either accepts/rejects
  one materially different exact candidate or parks candidate work because the
  trigger is false; R6 records context-local 512/128K decode/allocation tables
  and the resulting family/capacity decision;
- R7 either lands the matched Q6 peer or records a concrete engineering
  blocker; R8 either passes its full gate or remains blocked on materially
  better target-matched drafter quality. Their dependent comparisons/tuning
  stay closed while blocked;
- R9 reruns the full test and publication gates after the last retained default
  change. If no default changes, the existing `d1231ee0`/`7e9aad21` checkpoint
  remains current and the no-rerun decision is recorded instead of generating
  redundant toplines.

## Deferred Post-R0 Performance Candidates

This queue records the next likely gfx1151 opportunities so they are not lost,
but it is subordinate to the PARO decode release blocker. Do not begin an
unrelated implementation from this table while R0 is open unless the release
owner explicitly changes the order. Percentage ranges below are Amdahl steering
estimates, not performance claims or acceptance cutoffs; every exact,
non-regressive measured win remains retainable regardless of size.

The source profile is the clean SOL-G4 p512/d128 GGUF eager trace, refreshed for
the retained Q8T16 wave/block result by the 2026-07-12 production A/B. Its main
buckets are dense Q8T16 `44.25%`, selected-MoE GEMV `21.72%`, attention core
`10.68%`, and Q6 lm-head `10.06%` of traced GPU time.

| Deferred order | Candidate | Why it remains plausible | First bounded experiment after R0 | Amdahl steering estimate |
| ---: | --- | --- | --- | ---: |
| 0 | Revalidate the current gfx1151 GGUF graph default. | On HIP 7.15, clean scalar and wave/block G5 controls remain exact but state-bound replay is `0.246%`/`0.293%` slower than same-run eager. | One more balanced current-stack eager/graph A/B; restore eager as the production selector if graph does not reproduce a win. | Measured recovery is about `0.25%-0.30%` for this protocol. |
| 1 | Extend explicit wave/block indexing to sibling Q8T16 bodies. | The retained dual-split change improved its production leaf `1.349%`. The still-scalar single, triple-split, and concatenated-dual bodies use the same `k -> block/lane` structure and together represented `21.59%` of SOL-G4 GPU time. | Test triple-split first for structural transfer confidence (`5.93%` share), then single for aggregate leverage (`12.21%`), then concatenated dual (`3.44%`); require paired micro and exact model-family A/B for each. | Roughly `0.25%-0.60%` full-model if those bodies inherit a `1.3%-3.1%` leaf reduction. |
| 2 | Reduce exact selected-MoE GEMV live ranges/instruction cost. | Selected GEMVs represented `21.72%` of GPU time; the Q4 dual+SiLU kernel reports `200` VGPR while selected-down reports `104/128`, a kernel-specific resource signal rather than a blanket spill theory. | Join counters and ISA to the exact hot shapes, then try a bounded lifetime/column-work change. Do not repeat rejected launch-bound sweeps or assume existing accuracy-traded dp4a diagnostics transfer. | About `1%-2%` full-model for a real `5%-10%` family reduction. |
| 3 | Build a GQA-aware c1 paged-attention decode shape. | Attention represented `10.68%` of GPU time. The current per-Q-head blocks serve 16 Q heads from 2 KV heads, so a grouped design may reuse K/V work across each eight-head group; upside grows with context. | First run a context/counter sweep proving repeated KV traffic or value-scan limits, then test grouped-Q-head reuse while preserving the `KVLiveSpans` ABI and exact reduction/state gates. The already-rejected generic 128-thread override is not this experiment. | Potentially `1%-3%` at p512 if the attention bucket falls materially; larger long-context upside is conditional. |
| 4 | Remove surgical dispatch and intermediate-memory boundaries. | The latest marked wave/block profile records `20.873 ms/token` host wall versus `19.199 ms/token` summed GPU kernels, a `1.674 ms/token` residual that is not all proven dispatch. Router and RMSNorm are high-count candidates, but grid parallelism must survive. | Capture a dedicated steady decode gap/launch trace; fuse only a single-use producer/consumer boundary that removes both a launch and real tensor traffic. Avoid recomputing a shared normalized vector per output tile. | Several tenths per safe boundary; approximately `1%` cumulative is plausible but unmeasured. |
| 5 | Revisit Q6T16 lm-head layout/work distribution only with new evidence. | The full-vocabulary Q6 lm-head is one `~1.84 ms/token` call and `~10%` of GPU time, so a genuine body improvement transfers directly. | Use counters/ISA to justify a replacement layout or output-work mapping. Do not retry the rejected block-256, launch-bound, chunked/fused-argmax, or unmatched HIP/Vulkan variants. | Roughly `0.9%-1.8%` full-model for a `10%-20%` leaf reduction; confidence is lower than rows 1-3. |

Packed-dot compiler attribution remains diagnostic rather than a product target
until a current production body exposes the same bottleneck. Broad Wave64,
generic launch-bound sweeps, blanket `-ffast-math`, manual LDS spilling, and
undocumented scheduler flags remain outside this queue without new counter and
production-wall evidence.
