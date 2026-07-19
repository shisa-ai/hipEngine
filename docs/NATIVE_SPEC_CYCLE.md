# NativeSpecCycle Milestones and Status

> Canonical guide to hipEngine's `N0`–`N5` speculative-cycle milestones.
> Status snapshot: **2026-07-19**. Performance source of truth remains
> [`benchmarks/README.md`](../benchmarks/README.md) and the linked compact
> artifacts; this document explains what each milestone owns and consolidates
> the current qualified results.

## Executive Summary

`N0`, `N1`, `N2`, `N3`, `N3P`, and `N4` are project-local
**NativeSpecCycle delivery milestones**. They are not candidate budgets, model
versions, release numbers, or a promise that a larger number is faster.

The milestones move speculative decoding across three separate boundaries:

1. **API ownership:** one provider call owns a complete transaction.
2. **Native submission ownership:** Python no longer submits every child kernel
   or copy in that transaction.
3. **Device-state ownership:** acceptance, selected state, KV transactions, and
   cursors stay resident instead of round-tripping through the host.

Those boundaries do not advance in lockstep. In particular:

- `N1R` is the current W7900 performance winner because it removes the dominant
  target-verifier submission overhead while leaving cheap policy work alone.
- On W7900, `N2` and `N3` own more of the transaction but are not faster than
  `N1R` yet. On gfx1151, N3 retains essentially all of N1 and improves the clean
  current-main direct-commit control by 14.39%.
- `N3P` graph-submits the proposal too, but still uses one proposal graph and
  one target graph per cycle rather than one combined native submission.
- the first `N4` slice is a cross-provider adapter for PARO MTP and DFlash, not
  a faster GGUF successor to `N3P`.

In status reports and the benchmark rollup, retained reusable `N1R` is often
shortened to **N1**. The original one-shot `N1` experiment is rejected and is
kept only as historical evidence.

## One Speculative Cycle

A speculative cycle has five conceptual stages:

| Stage | Meaning | Principal state |
| --- | --- | --- |
| `PROPOSE` | The draft provider generates one or more candidate token IDs. | draft hidden/KV, candidate IDs, parent/depth metadata |
| `VERIFY` | The target model evaluates the root plus candidate rows. | target logits/top-1, verifier hidden rows, provisional target state/KV |
| `ACCEPT` | The target result determines the accepted prefix and correction/bonus token. | accepted count, selected row, visible IDs |
| `COMMIT` | The selected recurrent, hidden, and KV state becomes live; rejected provisional state is discarded or repaired. | Conv/GDN state, full-attention KV, hidden seed, MTP KV transaction |
| `UPDATE_CURSORS` | Target and proposal positions/contexts advance consistently. | token positions, context counts, target/MTP cursors |

The stages describe **semantic ownership**, not necessarily individual kernels.
A graph can contain many model kernels, metadata kernels, and copies. Conversely,
one public Python method can logically own all five stages while still
submitting proposal leaves from Python.

## Three Kinds of Ownership

Use these terms precisely when describing a milestone:

### API ownership

The scheduler/provider sees one transaction call and one bounded result. This
improves lifecycle clarity and fallback safety, but does not by itself reduce HIP
submissions.

### Native submission ownership

A C/C++ launcher or reusable HIP graph submits the work. Python may still call
one launcher per graph or stage. `N3P`, for example, has two native graph
submissions per public cycle.

### Device-state ownership

Intermediate acceptance, selected rows, recurrent/KV state, and cursors remain
on device. Only the scheduler-visible bounded result is read back. This removes
small synchronizations and makes a future multi-cycle launcher possible.

A claim such as "complete cycle" must say which of these boundaries is
complete. `N3` is complete at the scheduler-facing API boundary, not at the
single-native-submission boundary.

## Milestone Map

| Milestone | Meaning | Newly owned boundary | Still outside that boundary | Current status |
| --- | --- | --- | --- | --- |
| `N0` | Versioned ABI and oracle | Host/device control/result layouts, lifecycle, validation, CPU/fake launcher | Real model submission | Landed; no performance claim |
| `N1` | Initial fixed-B2 native target graph | One native `VERIFY` submission | Reusable positions; proposal, accept, commit, cursors | Exact but rejected because recapture regressed wall |
| `N1R` | Reusable B1/B2 target graphs | Stable native target `VERIFY` submission with live device metadata | Proposal and policy/commit remain on prior path | **Retained W7900 performance topline**; usually called N1 |
| `N2` | Device acceptance and selected-state commit | `VERIFY + ACCEPT + selected COMMIT + target cursors` | Proposal invocation and remaining MTP-KV repair/reseed/accounting | Exact ownership diagnostic |
| `N3` | Complete GGUF cycle adapter | One scheduler-facing call owns `PROPOSE` through cursor/result accounting | Proposal child kernels still Python-submitted | Exact API-ownership diagnostic |
| `N3P` | Reusable proposal graph | One proposal graph plus the existing target graph per cycle | Combined proposal+target submission; provider-general path | Exact submission-ownership diagnostic |
| `N4` | Shared PARO MTP / DFlash adapters | Current slice wraps shared target `VERIFY + ACCEPT` through the common ABI | Provider proposal, selected state/KV/hidden commit, full-cycle ownership | gfx1100 strict B1/B2/B3 exact; default-off because current control/marshal path adds 0.216-0.447 ms/cycle; DFlash/gfx1151 ungated |
| `N5` | Multi-cycle native option | Native loop may continue to EOS, cancellation/deadline, output limit, or scheduler yield | Future work | Planned only after provider/backend gates |

## Milestone Details

### N0 — ABI and oracle

`N0` establishes the version-1 control and result contract in
`hipengine.speculative.native_cycle` and
`hipengine/speculative/native_cycle_abi.h`:

- raw device pointers and bounded scalar capacities;
- explicit dtypes and `KVLiveSpans` rather than a block-table shortcut;
- stage masks, lifecycle/error enums, and borrowed-pointer ownership;
- host-only validation and CPU/fake launchers;
- pre-launch fallback as the correctness oracle.

`N0` deliberately does not claim model execution or speed. It lets providers
attach without adding backend/quant branches to engine/model dispatch.

### N1 and N1R — target verification submission

The initial `N1` proved that a single-request fixed-B2 target graph could match
target top-1, hidden, recurrent state, KV, and cursors. It embedded
position-bound addresses and therefore required per-cycle capture. The graph
body was exact, but recapture made the path slower, so it was rejected.

`N1R` corrected the ownership model:

- independent reusable B1/two-row and B2/three-row executables;
- graph-owned fixed-address scratch;
- live device token IDs, positions, context counts, and cursor metadata;
- eager fallback for unsupported tails before graph mutation;
- no silent replay after a post-launch failure.

Conceptually:

```text
Python/device-chain proposal
    -> one reusable native target VERIFY graph
    -> existing acceptance, commit, repair, and cursor path
```

The benchmark rollup calls retained `N1R` simply **N1** because the rejected
one-shot experiment is no longer a candidate route.

### N2 — device acceptance and selected-state commit

`N2` extends the reusable target transaction to own:

- strict-chain acceptance;
- selected FP32 hidden row;
- selected Conv/GDN recurrent state;
- target position/context cursor updates;
- a bounded visible-ID result payload.

Conceptually:

```text
Python/device-chain proposal
    -> native VERIFY + ACCEPT + selected target COMMIT + target cursors
    -> existing MTP-KV repair, reseed, and remaining accounting
```

This removes real small host windows, but the aggregate result remains below
`N1R`; it is retained as infrastructure for complete-cycle ownership.

### N3 — complete scheduler-facing GGUF adapter

`N3` adds `Qwen35GGUFResidentSession.run_native_spec_mtp_cycle()`. One provider
call owns:

- strict B1/B2 NextN proposal invocation;
- the `N2` target/accept/selected-state transaction;
- verifier-row MTP reseed;
- speculative MTP-KV rollback and accepted-row repair;
- target/MTP cursor and bounded result accounting.

Conceptually:

```text
scheduler
    -> one N3 provider call
       -> Python-submitted proposal leaves
       -> one N2 target graph
       -> provider repair/reseed/accounting
    <- one bounded cycle result
```

`N3` is therefore a complete **API transaction**, not a single native
submission. Unsupported backends/shapes fail before target mutation and retain
the prior exact loop.

### N3P — proposal submission collapse

`N3P` stages changing proposal inputs into fixed runner buffers and captures the
strict B1/B2 NextN device chain:

- FP32 hidden seed and root embedding;
- RoPE, position, and context rows;
- indexed FP32 draft-KV destinations;
- independent B1/B2 proposal executables over runner-owned stable KV storage.

Conceptually:

```text
scheduler
    -> one N3 provider call
       -> one native proposal graph   [N3P]
       -> one native target graph     [N1R/N2]
       -> provider repair/accounting
```

This is two graph submissions, not one combined proposal+target native call and
not child-kernel fusion.

### N4 — provider-neutral expansion

`N4` asks PARO MTP and DFlash to reuse the NativeSpecCycle ABI rather than
forking separate native loops. Their proposal owners differ, but both converge
on `Qwen35ParoResidentSession.verify_chain_bulk_and_commit()`.

The first landed slice registers:

```text
(hip_gfx1100, speculative_cycle, w4_paro, native_v1_target_graph)
```

For an eligible single-request B1/B2/B3/B4/B5/B8 replay it binds fixed resident
metadata/accept buffers, real verify-chain `KVLiveSpans`, and either resident
FP16 verifier rows or BF16 DFlash hidden taps. It accurately declares only:

```text
VERIFY | ACCEPT
```

Provider proposal, linear/KV/hidden commit, cursors, and scheduler results remain
on the existing path. Graph capture/miss, graph-off, tree/inactive layouts,
unsupported shape/backend, registry miss, and control-build failure retain
pre-launch direct fallback. gfx1151 remains unregistered.

`N4` is therefore a cross-provider compatibility milestone, not a numerical
successor that should be compared directly with GGUF `N3P` throughput.

### N5 — multi-cycle native loop

`N5` is planned only after complete per-provider cycles are exact and
non-regressive. A native launcher may continue across cycles until a bounded
condition such as EOS, cancellation/deadline, output capacity, or scheduler
yield. It must not hide unbounded work from the scheduler.

## Why a Higher Milestone Can Be Slower

Milestone numbers track architecture and ownership, not a monotonic performance
score.

`N1R` attacks the dominant W7900 bottleneck: thousands of target-verifier host
submissions. `N2`–`N3P` add state selection, staging, richer lifecycle, and
additional graph boundaries. Their small sub-windows improve, but those savings
have not yet offset all aggregate overhead and run variance.

The retained decision is consequently:

- use `N1R` as the explicit gfx1100 `llama-compat` performance route;
- retain `N2`, `N3`, and `N3P` as exact ownership/submission infrastructure;
- do not promote a larger milestone merely because it owns more stages;
- require the same full category+heldout correctness and wall gate before any
  default or topline change.

## Semantic and Policy Boundaries

- `llama-compat` is explicit and accuracy-traded relative to hipEngine's
  serial-prefix-preserving exact/default MTP policy. Native milestones match the
  existing `llama-compat` trajectory; they do not make it the automatic exact
  route.
- GGUF, PARO MTP, and DFlash use the shared ABI but keep provider-specific model
  math, proposal policy, graph buckets, and state semantics.
- A gfx1100 result does not admit gfx1151. Every backend/provider combination
  needs its own full correctness, performance, profiler, and fallback gate.
- `KVLiveSpans` remains the attention ABI at every milestone.
- Fused/native paths retain exact unsupported-shape fallbacks and the existing
  unfused primitive chains.

## Current Performance Scorecard

All cross-engine and cross-process qualifications below are part of the claim;
raw `tok/s` values must not be copied without them.

### Measurement scope and qualifications

The W7900 NativeSpecCycle scorecard uses Qwen3.6-35B-A3B UD-Q4_K_M, one
request, greedy/reasoning-off generation, candidate budget B2, and the full
10-prompt `mtpbench-code-general-ja` category+heldout suite. hipEngine emits 24
outputs per prompt with BF16 KV. The external llama.cpp protocol requests 25
outputs and times the 24 post-first-token transitions with F16 KV. The
transition-normalized external row is the closest validated floor, not identical
arithmetic.

Each clean hipEngine milestone ran in a fresh process and recorded its own true
AR denominator. AR varied from about 92.2 to 96.7 tok/s across those processes,
so compare both complete wall and absolute MTP rate; do not infer milestone
quality from the ratio alone. `N1R` has two repetitions. `N2`, `N3`, and `N3P`
are correctness/ownership diagnostics, not replacement toplines.

### W7900 headline

The requested external floor is closed:

- retained reusable `N1R`: **122.667 tok/s / 8.186 ms-output**;
- refreshed llama.cpp MTP: **115.444 tok/s / 8.662 ms-transition**;
- hipEngine lead: **+6.26% rate / -5.50% wall**;
- all full/train/heldout/category rows beat their corresponding AR controls;
- all 240 output IDs and 96 cycle semantics match the prior explicit
  `llama-compat` route, with **80.45% draft acceptance / 60.00% accepted-output**.

[`N1R retained artifact`](../benchmarks/results/2026-07-19-w7900-llama-compat-reusable-native-cycle.json)
· [`llama.cpp floor`](../benchmarks/results/2026-07-19-w7900-llamacpp-mtp-natural25-refresh.json)

### W7900 route comparison

| Route | Own true AR tok/s | MTP tok/s | MTP / own AR | Complete wall | Relative to retained N1R | Evidence status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| llama.cpp base/MTP, F16 KV | 78.053 | 115.444 | 1.4791x | 8.662 ms/transition | N1R is +6.26% | external diagnostic floor |
| pre-native hipEngine compatibility route | 92.262 | 54.880 | 0.5948x | 18.259 ms/output | -55.26% rate | superseded baseline |
| **N1R reusable target graph** | **96.746** | **122.667** | **1.2679x** | **8.186 ms/output** | — | **retained performance topline** |
| N2 device accept/selected commit | 92.395 | 117.557 | 1.2723x | 8.529 ms/output | -4.17% | exact ownership diagnostic |
| N3 complete public adapter | 92.233 | 118.592 | 1.2858x | 8.497 ms/output | -3.32% | exact API-ownership diagnostic |
| N3P proposal graph | 92.187 | 118.183 | 1.2820x | 8.610 ms/output | -3.66% | exact submission diagnostic |
| N4 shared PARO/DFlash target adapter | no retained row | no retained row | no retained row | no retained row | not comparable | strict gfx1100 PARO B1/B2 correctness admitted; performance unclaimed |

`N1R` repeated at **123.332 and 122.667 tok/s** (0.54% max/min spread).
The old route to `N1R` change is **+123.52% rate / -55.17% complete wall** with
unchanged acceptance.

Artifacts:
[`baseline`](../benchmarks/results/2026-07-19-w7900-hipengine-llama-compat-current-baseline.json)
· [`N2`](../benchmarks/results/2026-07-19-w7900-llama-compat-native-cycle-n2.json)
· [`N3`](../benchmarks/results/2026-07-19-w7900-llama-compat-native-cycle-n3.json)
· [`N3P`](../benchmarks/results/2026-07-19-w7900-llama-compat-native-cycle-n3p.json)

### W7900 full-suite/category comparison

This table uses conservative `N1R` repetition 2 and transition-normalized
llama.cpp MTP:

| Split | hipEngine AR | hipEngine N1R | N1R / AR | llama.cpp MTP | N1R lead | N1R draft acceptance | N1R accepted/output | N1R wall/output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 96.746 | **122.667** | 1.2679x | 115.444 | **+6.26%** | 80.45% | 60.00% | 8.186 ms |
| Train | 96.129 | **124.704** | 1.2973x | 116.867 | **+6.71%** | 87.25% | 61.81% | 8.052 ms |
| Heldout | 97.685 | **119.734** | 1.2257x | 113.374 | **+5.61%** | 71.43% | 57.29% | 8.388 ms |
| Code | 97.064 | **127.814** | 1.3168x | 126.349 | **+1.16%** | 93.94% | 64.58% | 7.854 ms |
| General English | 94.247 | **123.374** | 1.3091x | 114.046 | **+8.18%** | 75.68% | 58.33% | 8.138 ms |
| General Japanese | 98.039 | **118.419** | 1.2079x | 104.746 | **+13.05%** | 69.23% | 56.25% | 8.480 ms |
| Mixed JA/EN | 97.403 | **116.783** | 1.1990x | 109.092 | **+7.05%** | 72.97% | 56.25% | 8.604 ms |

llama.cpp records **81.56% draft acceptance / 58.40% accepted-output**. Its
acceptance is slightly higher, but hipEngine emits more accepted draft tokens
per counted output and wins complete transition-normalized wall.

### Why N1R wins: target-verifier profiler

Matched cached B2 verifier profiling shows that graph reuse removes host
submission residual rather than changing model math materially:

| Metric per B2 step | Pre-native target | N1R reusable graph | Change |
| --- | ---: | ---: | ---: |
| Host wall | 52.419 ms | **18.671 ms** | **-64.38%** |
| GPU kernel duration | 14.013 ms | **13.670 ms** | -2.45% |
| Host/profiler residual | 38.405 ms | **5.001 ms** | **-86.98%** |
| Kernel calls | 977 | **940** | -3.79% |
| Kernel share of host wall | 26.7% | **73.2%** | submission bottleneck removed |

The pre-native kernel body was led by selected-MoE GEMV **4.722 ms/step**,
dense-Q8 GEMV **3.934 ms**, GDN linear attention **1.132 ms**, MoE routing
**1.101 ms**, and LM head **0.943 ms**. Those bodies remain optimization
opportunities, but they did not explain the old 38.4 ms residual.

A same-environment source-control pair also measures current code at
**43.219 ms** versus historical `202bd2f0` at **48.204 ms** (-10.34%). The
older full-suite slowdown therefore was not attributable to an isolated current
verifier source regression.

### N1R cycle attribution

Conservative complete wall is **8.186 ms/output**. Stage timers are nested and
must not be summed blindly:

| Timer | ms/output | Relationship |
| --- | ---: | --- |
| target block verify total | 6.722 | contains native submit/capture/readback and some tail handling |
| native target graph submit | 6.179 | inside target block verify |
| first B1/B2 capture amortization | 0.323 | inside target block verify; zero captures in measured profiler windows |
| native result readback | 0.031 | inside target block verify |
| serial verify tail | 0.290 | unsupported/no-draft tail path |
| draft initial | 0.948 | proposal path |
| draft device-chain drain | 0.705 | proposal path |
| MTP device-KV commit | 0.192 | provider commit |
| target replay/commit bookkeeping | 0.054 | provider bookkeeping |
| accept policy and seed | 0.002 | host-visible policy boundary |

The target remains the largest complete-cycle component, but native reuse
changes it from a host-bound path to a mostly kernel-bound path.

### N2 sub-window result

N2 produces meaningful device-residency wins even though its aggregate rate is
not the topline:

| Sub-window | Same-tree N1 | N2 | Approximate change |
| --- | ---: | ---: | ---: |
| MTP device-KV commit | 0.1349 ms/output | **0.1023 ms** | -24% |
| target replay/selected commit | 0.0588 ms | **0.00677 ms** | -88% |
| draft seed upload | 0.01994 ms | **0.00235 ms** | -88% |
| MTP context replay append | 0.01750 ms | **0.000125 ms** | -99% |

The same-tree aggregate screen was **117.773 -> 117.235 tok/s (-0.46%)**, so
these mechanical savings are retained without replacing `N1R`.

### N3 and N3P result

Clean `N3` is **118.592 tok/s / 8.497 ms-output**, +0.88% versus clean N2 but
-3.32% versus retained `N1R`. It proves complete scheduler-facing ownership and
matches N2 across all **240 IDs / 96 cycles**.

Matched eight-cycle N3 -> N3P tracing preserves the same 22 visible IDs and
changes host API calls as follows:

| HIP API | N3 | N3P | Change |
| --- | ---: | ---: | ---: |
| `hipLaunchKernel` | 3273 | **2731** | -542 |
| synchronous `hipMemcpy` | 1204 | **1124** | -80 |
| `hipGraphLaunch` | 8 | **16** | +8, exactly one proposal graph/cycle |

Same-source aggregate timing is neutral: N3 **116.793 tok/s / 8.634 ms-output**
versus N3P **117.589 / 8.653**. Excluding first capture, proposal wall improves
**0.964 -> 0.953 ms/output (-1.19%)**. Clean detached N3P is **118.183 tok/s /
8.610 ms-output**. This is useful submission infrastructure, not a new topline
or evidence for one combined proposal+target native call.

### N4 verifier correction and strict admission

The original `7bf3439e` B3 flag-off/on packet correctly proved adapter equality:
it matched **265/265** non-timing/non-route leaves and all eight IDs while the
N4 arm submitted four `VERIFY|ACCEPT` replays. Its conclusion was wrong,
however. Both arms emitted verifier token `59` instead of target-AR token `19`
after rejecting the draft. Rejection must use the target correction, so this
was verifier semantics, not target/sidecar incompatibility. The historical
**39.898 -> 39.085 tok/s** observation remains invalid as speed evidence.

The model audit keeps the current artifact:

- the target is the later full8192-old+fresh, 149-hour packed PARO checkpoint;
  versus full4096-e5 its held-out PPL/KL/top-1/max-KL improve from
  **6.6216/0.034684/92.000%/11.0422** to
  **6.6090/0.027939/92.856%/6.3961**;
- the live target blob is `a5c9100b…cc60`;
- the BF16 sidecar is `556c607c…26f`, passes all **19/19** config/dtype/shape
  checks, and has the same 19 tensor payloads as the older assembly.

The strict GDN and linear-output controls first reduced B3 to exact AR, then a
B2 RED localized the remaining mismatch: serial and verifier row 0 were exact
through linear attention, routing, and selected experts, but **253/2048** BF16
post-MoE values differed at the linear layer's shared expert. The chain/tree
t-loop called `run_moe_c1_fp16()` without forwarding the existing
`HIPENGINE_QWEN35_MOE_C1_FORCE_SMALL_BATCH_SHARED_EXPERT` control. Forwarding it
restores the serial-c1 shared-expert path without changing default fast behavior
or any model byte.

Clean `5ef02aff` W7900 evidence under the complete explicit strict stack now
passes:

- B2 native-off/on: identical eight target-AR IDs, target top-1 paths,
  acceptance, and GPU/CPU decisions; five steady native `VERIFY|ACCEPT` replays;
- B2 three-cycle state: exact **60/60** resident and selected Conv/GDN buffers,
  **20/20** live/selected K/V cells, and **60/60** scratch-state commits each
  cycle;
- canonical B1 category+heldout suite: **10/10 prompts and 240/240 IDs exact**,
  **16/214** draft accepts overall, **13/125** train, and **3/89** heldout; all
  **150/150** retained trace records use native `VERIFY|ACCEPT` and GPU/CPU
  acceptance agrees.

This closes the model-artifact blocker and admits the existing target+sidecar
for strict gfx1100 PARO N4 correctness. It does **not** retain a speed row: B1
acceptance is low, the strict fallbacks are costly, provider proposal/commit is
still outside N4, DFlash is ungated, and gfx1151 remains unregistered. N4 stays
explicit/default-off.

[`N4 corrected artifact`](../benchmarks/results/2026-07-19-w7900-paro-mtp-native-target-graph-n4-correctness.json)
· [`superseded diagnosis`](../benchmarks/results/2026-07-19-w7900-paro-mtp-native-target-graph-n4-blocked.json)

### Uncontended N4 economics and host attribution

The 2026-07-20 exclusive-GPU0 production-GPU-accept rerun closes the timing
ambiguity without promoting a speed row. The direct canonical B1/B2/B3 matrix
passes all **30 prompt-budget rows / 720 IDs** exactly, while pooled MTP/true-AR
falls from **0.5767x** at B1 to **0.4242x / 0.3568x** at B2/B3. Pooled draft
acceptance is only **7.48% / 4.85% / 3.49%**; wider strict verification buys
only **1.1215 -> 1.1429 -> 1.1483 visible tokens/cycle** while complete wall
grows **16.562 -> 23.055 -> 28.019 ms/cycle**. B1 is the only sensible
optimization budget, but it remains well below AR on every split/category.

A matched B1 N4-on/off/on bracket holds **240 IDs, 214 cycles, and 16 accepts**
constant. The direct graph control is **65.584 tok/s / 16.115 ms-cycle**; N4 is
**63.736/64.511 tok/s / 16.562/16.330 ms-cycle**, a reproducible
**1.64-2.82% rate loss / 0.216-0.447 ms-cycle regression**. Cached final-child
HIP API+kernel tracing gives the mechanism rather than guessing:

- cycle kernels are unchanged at **1248.5 calls and 11.165/11.169 ms/pass**;
- proposal update is unchanged at **1.475/1.474 ms/pass**;
- verify host wall moves **14.923 -> 15.235 ms/pass** while verify kernels move
  only **9.912 -> 9.915 ms/pass**;
- N4 adds exactly one `hipStreamSynchronize` per verifier (**2 -> 3**) and the
  remaining pre-`hipGraphLaunch` gap is repeated Python control construction,
  validation, binding-signature generation, and ctypes/result marshalling.

N4+ therefore starts by caching/reusing the state-bound ABI/ctypes control and
removing duplicate synchronization under exact pointer/shape/active-mask
fallback. Expanding commit/proposal ownership before removing this adapter tax
would hide the first causal result. N4 remains explicit/default-off; this is a
retained diagnostic, not a speculative speed claim.

[`N4 uncontended baseline`](../benchmarks/results/2026-07-20-w7900-paro-mtp-n4-uncontended-baseline.json)

## gfx1151 Status

The current transfer ran clean at `1163e1bb` under TheRock HIP 7.15,
`amd_iommu=off`, TuneD `accelerator-performance`, and the automatic one-HIP-
hardware-queue policy. The target-only N1 and public complete-cycle N3 paths are
now independently admitted; N3P proposal graphs and N4 remain unregistered.
That boot disables XDNA.

### gfx1151 MTP comparison

| Route | Own AR tok/s | MTP tok/s | MTP / AR | Wall | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| exact/default B5 (`2edbb2ee`) | 56.983 | **56.386** | **0.9895x** | **17.808 ms/output** | semantic control; still below AR |
| clean current-main direct-commit control | 56.236 | 70.020 | 1.2451x | 14.314 ms/output | optimization control |
| **N1 reusable target graph** | 55.490 | **80.132** | **1.4441x** | **12.512 ms/output** | retained target-only performance boundary |
| **N3 public complete cycle** | 56.085 | **80.099** | **1.4282x** | **12.551 ms/output** | retained production adapter; -0.042% vs N1 |
| historical direct commit (`2edbb2ee`) | 56.783 | 81.900 | 1.4423x | 12.233 ms/output | prior retained absolute; different revision/run |
| llama.cpp base/MTP, F16 KV | 50.371 | 68.153 | 1.3530x | 14.673 ms/transition | external diagnostic |

N3 improves the clean same-commit control by **14.39% rate / -12.32% complete
wall** and matches N1 plus the control across all **240 output IDs / 97 cycle
semantics**. Draft acceptance and accepted-output remain **77.72% / 59.58%**.
Every category and the heldout split improves versus direct commit and beats its
own true AR denominator:

| Split | AR tok/s | N3 tok/s | N3 / AR | N3 vs control |
| --- | ---: | ---: | ---: | ---: |
| Full | 56.085 | **80.099** | 1.4282x | **+14.39%** |
| Train | 55.968 | **80.912** | 1.4457x | **+11.37%** |
| Heldout | 56.263 | **78.908** | 1.4025x | **+18.83%** |
| Code | 56.123 | **86.083** | 1.5338x | **+9.91%** |
| General English | 57.257 | **78.984** | 1.3795x | **+19.45%** |
| General Japanese | 55.611 | **75.122** | 1.3509x | **+12.65%** |
| Mixed JA/EN | 55.351 | **75.658** | 1.3669x | **+19.20%** |

The real-model graph oracle covers B1/B2, changing-position replay, reject/full-
accept N2, target IDs, FP32 hidden, 60 Conv/GDN buffers, 20 full-KV buffers,
selected commit, and cursors. The cached six-step trace is **24.891 ms host /
21.674 ms kernels / 3.218 ms residual**, with 940 calls/step, zero measured
recaptures, and the expected `copy_i32_to_i64_kernel` at **1.002-1.323 us**, 8
VGPR, zero scratch/LDS.

[`gfx1151 NativeSpecCycle artifact`](../benchmarks/results/2026-07-19-gfx1151-llama-compat-native-cycle-transfer.json)
· [`prior gfx1151 MTP artifact`](../benchmarks/results/2026-07-17-gfx1151-amd-iommu-off-mtp-refresh.json)

### Adjacent concurrency toplines

These are retained concurrency results, not NativeSpecCycle milestone
measurements. They show the current backend floor that future speculative-cycle
work must not regress.

#### Direct model-step aggregate throughput

| Backend/provider | c1 | native c2 | native c4 | native c8 | Qualification |
| --- | ---: | ---: | ---: | ---: | --- |
| W7900 PARO | 116.022 | **121.923** | not retained | not retained | explicit direct selected-batch c2 |
| gfx1151 PARO | 70.810 | 79.237 | **100.209** | 99.943 | explicit direct selected-batch c2/c4/c8 |
| W7900 GGUF | 85.469 | — | 183.020 as c4+c4 | **246.872** | one physical native-c8 is +34.89% vs c4+c4 |
| gfx1151 GGUF | 50.277 | — | 102.606 as c4+c4 | **127.902** | one physical native-c8 is +24.65% vs c4+c4 |

#### Real OpenAI/SSE aggregate throughput

| Platform/provider | c1 | c8 | c9 | C13 grouped | Serial comparison |
| --- | ---: | ---: | ---: | ---: | ---: |
| W7900 GGUF | 25.583 | 136.122 | 88.592 | **111.380** | 31.708 serial C13 |
| gfx1151 GGUF | 15.701 | 86.338 | 57.127 | **72.522** | 42.764 serial C13 |
| gfx1151 PARO SSE | 36.327 | **41.487** | — | — | 35.633 serial c8 |

gfx1151 PARO blocking HTTP c1/c2/c4/c8 is
**47.124 / 51.962 / 60.323 / 61.253 aggregate tok/s**.

Evidence:
[`gfx1100 PARO c2`](../benchmarks/results/2026-07-18-gfx1100-paro-g2-selected-batch-c2-retained.json)
· [`gfx1151 PARO c2/c4/c8`](../benchmarks/results/2026-07-18-gfx1151-paro-g3-native-c248-direct-retained.json)
· [`gfx1151 PARO server`](../benchmarks/results/2026-07-18-gfx1151-paro-g5-f1-server-scaling.json)
· [`gfx1100 GGUF c8`](../benchmarks/results/2026-07-17-gfx1100-gguf-concurrency-e2-native-c8-scaling-closure.json)
· [`gfx1100 GGUF server`](../benchmarks/results/2026-07-17-gfx1100-gguf-concurrency-f1-server-scaling-closure.json)
· [`gfx1151 GGUF c8`](../benchmarks/results/2026-07-17-gfx1151-gguf-concurrency-e1-native-c8-scaling-closure.json)
· [`gfx1151 GGUF server`](../benchmarks/results/2026-07-18-gfx1151-gguf-concurrency-f1-server-scaling-closure.json)

## vLLM RDNA3 Status

The current comparison environment is not a validated vLLM performance row:

- vLLM v0.25.1 does not contain merged fixes
  [#46384](https://github.com/vllm-project/vllm/pull/46384) or
  [#47782](https://github.com/vllm-project/vllm/pull/47782);
- current upstream main contains both, and regression-test PR
  [#48970](https://github.com/vllm-project/vllm/pull/48970) has the corruption
  arms green;
- primary issue [#43559](https://github.com/vllm-project/vllm/issues/43559)
  remains open;
- [#45614](https://github.com/vllm-project/vllm/pull/45614) and
  [#47861](https://github.com/vllm-project/vllm/pull/47861) remain unresolved.

The older PARO/llama.cpp/vLLM repository row is stale and cross-quant. Use
[`VLLM_RDNA3.md`](VLLM_RDNA3.md) for build/runtime details, but do not treat it
as a current MTP competitor until current main passes the exact same model,
category+heldout, correctness, and timing protocol.

## Current Decisions and Next Gates

1. **Keep N1R as the explicit gfx1100 `llama-compat` performance route.** It is
   the only milestone with a repeated retained speed claim and clears the
   external floor.
2. **Keep gfx1100 N2/N3/N3P as exact infrastructure.** Promote only after the
   full suite is non-regressive versus 122.667 tok/s and 8.186 ms-output; a
   larger ownership label is not sufficient. On gfx1151, N3 is retained because
   it is +14.39% versus its clean control and only 0.042% below N1.
3. **Do not claim a combined native cycle yet.** N3P still submits separate
   proposal and target graphs. Combining them is worthwhile only if the measured
   complete wall improves without changing semantics.
4. **Keep the current PARO target+sidecar and optimize the strict N4 route.**
   The artifact blocker is closed; next profile strict verifier wall, then extend
   provider proposal/commit ownership. Run DFlash category+heldout independently.
5. **Keep gfx1151 N1/N3 admitted independently.** They pass the real-model and
   full category+heldout gates at 80.132/80.099 tok/s. N3P and N4 remain
   unregistered until they show an independent correctness and complete-wall
   reason to transfer; no gfx1100 result aliases them automatically.
6. **Use vLLM main, not v0.25.1, for the next comparison**, and require the
   unresolved corruption/correctness gates before timing.

## Evidence Index

- [`benchmarks/README.md`](../benchmarks/README.md) — canonical toplines and
  protocol status.
- [`2026-07-19 W7900 pre-native baseline`](../benchmarks/results/2026-07-19-w7900-hipengine-llama-compat-current-baseline.json)
- [`2026-07-19 W7900 llama.cpp floor`](../benchmarks/results/2026-07-19-w7900-llamacpp-mtp-natural25-refresh.json)
- [`2026-07-19 W7900 N1R retained route`](../benchmarks/results/2026-07-19-w7900-llama-compat-reusable-native-cycle.json)
- [`2026-07-19 W7900 N2`](../benchmarks/results/2026-07-19-w7900-llama-compat-native-cycle-n2.json)
- [`2026-07-19 W7900 N3`](../benchmarks/results/2026-07-19-w7900-llama-compat-native-cycle-n3.json)
- [`2026-07-19 W7900 N3P`](../benchmarks/results/2026-07-19-w7900-llama-compat-native-cycle-n3p.json)
- [`2026-07-19 W7900 N4 strict correction`](../benchmarks/results/2026-07-19-w7900-paro-mtp-native-target-graph-n4-correctness.json)
- [`2026-07-19 W7900 N4 superseded diagnosis`](../benchmarks/results/2026-07-19-w7900-paro-mtp-native-target-graph-n4-blocked.json)
- [`2026-07-19 gfx1151 N1/N3 transfer`](../benchmarks/results/2026-07-19-gfx1151-llama-compat-native-cycle-transfer.json)
- [`2026-07-17 gfx1151 MTP refresh`](../benchmarks/results/2026-07-17-gfx1151-amd-iommu-off-mtp-refresh.json)
