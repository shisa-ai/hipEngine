# Nathanw1014 Strix Halo llama.cpp review for hipEngine gfx1151 GGUF

**Reviewed:** 2026-08-04

**Scope:** `Nathanw1014/strix-halo-llamacpp` releases and evidence pack,
`Nathanw1014/llama.cpp` optimization branches through `strix-halo-vulkan`
`b7b85da9c4a9fdeb3cab51030a40d1552270f272`, and the current hipEngine
Qwen3.6/Laguna GGUF gfx1151 paths.

**Decision type:** source/evidence review followed by prioritized local
execution and a later user-requested exact-model fork diagnostic. Nathan's
published speedups remain upstream evidence; hipEngine's only new accepted
performance claim is the separately measured head-major scratch artifact linked
below. The local fork row is descriptive, not a strict cross-engine claim.

## Executive decision

The original source-transfer review identified **one immediate, bounded
experiment** on hipEngine, which has now completed:

> Measure one reusable, cross-layer pair of **head-contiguous BF16 K/V prefill
> scratch buffers** in front of AOTriton at 32K and 64K, including the copy in
> wall time. hipEngine currently gives AOTriton a token-major, head-interleaved
> paged cache through
> explicit strides. Nathan's strongest and most transferable finding is that
> converting that exact layout to per-head-contiguous storage before cooperative-
> matrix Flash Attention removed a severe long-context prefill collapse.

Do **not** change the persistent `KVLiveSpans` layout first. A temporary scratch
A/B is smaller, preserves the paged-KV ABI and all decode kernels, and directly
answers whether AOTriton on gfx1151 pays the same strided-load tax as RADV
coopmat1.

**Execution update (2026-08-04): completed and promoted.** One bounded tracked
head-major pair is now the gfx1151 default through the validated 65,792-token
rounded capacity. Copy-inclusive full prefill changes 512/4K/32K/64K by
**-0.028%/+0.616%/+3.383%/+7.001%** with byte-exact complete model state;
allocation denial, explicit disable, unsupported backends, and larger sessions
retain strided AOTriton. Evidence:
[`2026-08-04-gfx1151-q4km-aotriton-head-major-prefill.json`](../benchmarks/results/2026-08-04-gfx1151-q4km-aotriton-head-major-prefill.json).

The later exact-model fork run changes the next objective without invalidating
that source-review closure: **protect hipEngine's prefill lead, close the exact
BF16 c1 decode gap, and reduce prefill-scratch high-water before pursuing an
approximate KV headline.** The new campaign is defined below. It is not a plan
to port every fork patch: Nathan's F16 decode is only 1.27%-1.79% above the
previous vanilla llama.cpp Vulkan lane, while Q8_0 becomes a large additional
lever only at 32K/64K.

Most of Nathan's other high-value ideas are already represented in hipEngine:

- INT8 decode uses a grouped-GQA producer whose grid is `(kv_head, split)` and
  scans each K/V stream once while sharing loads across eight query heads.
- Normal quantized-KV prefill avoids the slow direct-INT8 attention path by using
  a temporary BF16/AOTriton bridge; GGUF retained INT8 also has layer-local BF16
  prefill-oracle storage.
- MoE prefill already builds expert counts, prefix offsets, stable compact row
  lists, active-expert lists, and tile maps before the expert kernels. Laguna
  selects the parallel one-workgroup-per-expert implementation on gfx1151.
- gfx1151 already has measured architecture-local chunk and tile policies,
  low-precision activation paths, wave32 HIP kernels, and extensive exact fused
  composites.
- `amd_iommu=off` is already the active benchmark boot and is documented as a
  directional, security-relevant system tradeoff rather than a causal engine
  optimization.

Nathan's DeepSeek V4 work is valuable, but it is **future-model work**, not a
Qwen3.6 or Laguna optimization: lightning indexer, indexed sparse attention,
gather-to-compact decode, fused hyper-connections, indexer-cache precision, and
small-batch O-projection contiguization belong in a future DeepSeek V4 plugin.

The Vulkan command-buffer byte cap is not portable to HIP. Its robustness
principle is relevant to the open repeated-128K stall, but hipEngine already has
a qualified opt-in layer drain. A follow-up kernel audit rejects the former MES
`lr_compute_wa` A/B: upstream removed that incomplete workaround because it
caused instability and identified the gfx1151 VGPR-size correction as the real
fix. The captured kernel already has that correction active, so its stall is not
a missing-`lr_compute_wa` configuration mismatch.

## Source and evidence quality

The user-facing toolbox snapshot is
[`b166a56e`](https://github.com/Nathanw1014/strix-halo-llamacpp/tree/b166a56e58ab0f27fd03f60fff060eebdf5f64b5).
Its own summary separates the dominant algorithmic changes from marginal or
negative knobs ([README lines 35-55](https://github.com/Nathanw1014/strix-halo-llamacpp/blob/b166a56e58ab0f27fd03f60fff060eebdf5f64b5/README.md#L35-L55))
and maintains clean per-concern upstream branches
([README lines 133-167](https://github.com/Nathanw1014/strix-halo-llamacpp/blob/b166a56e58ab0f27fd03f60fff060eebdf5f64b5/README.md#L133-L167)).

Release status matters:

- [`v0.1`](https://github.com/Nathanw1014/strix-halo-llamacpp/releases/tag/v0.1)
  silently fell back to CPU because its bundled ICD manifest was invalid;
  `v0.1` performance is therefore invalid as GPU evidence.
- [`v0.2`](https://github.com/Nathanw1014/strix-halo-llamacpp/releases/tag/v0.2)
  repaired that fallback and added an explicit backend check.
- [`v0.3`](https://github.com/Nathanw1014/strix-halo-llamacpp/releases/tag/v0.3)
  adds DeepSeek V4 lightning-indexer/indexed sparse attention and
  gather-to-compact decode.
- [`v0.4`](https://github.com/Nathanw1014/strix-halo-llamacpp/releases/tag/v0.4)
  adds fused DeepSeek V4 hyper-connections and reports the community gfx1151
  measurements.
- [`dev-20260803-b7b85da`](https://github.com/Nathanw1014/strix-halo-llamacpp/releases/tag/dev-20260803-b7b85da)
  is explicitly compile/container-smoke tested only by its publisher; it is not
  an upstream benchmark or correctness-validation release. We subsequently ran
  that exact payload locally, creating an independent diagnostic rather than
  changing its upstream release status.

The evidence pack also names claims for which raw logs are not vendored. This
review therefore treats commit-level mechanisms as source facts and toolbox
measurements as upstream evidence. The local exact-model run below is the only
fork execution measured by this project and remains explicitly qualified.

## Local exact-model fork diagnostic

The exact `dev-20260803-b7b85da` portable payload was run on the same Radeon
8060S with the exact hipEngine
`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`. Its manifest pins source
`b7b85da9c4a9fdeb3cab51030a40d1552270f272`, Mesa `d18d598e2`, libdrm
2.4.133, and shaderc v2026.3-dev `49a8724d`; the binary identifies build
`b7b85da9` number 10283, Vulkan, and RADV STRIX_HALO. All sixteen matched
phase runs and the separate depth run exit zero with those identities.

### Full-prompt prefill (tok/s)

| Workload | hipEngine BF16 | Fork F16 | hipEngine lead | Fork Q8_0 | hipEngine lead |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512/128 | **1394.772** | 1390.978 | **+0.273%** | 1387.763 | **+0.505%** |
| 4K/128 | **1472.330** | 1399.336 | **+5.216%** | 1389.530 | **+5.959%** |
| 32K/128 | **1171.610** | 1102.982 | **+6.222%** | 1104.183 | **+6.107%** |
| 64K/128 | **952.348** | 893.919 | **+6.536%** | 879.963 | **+8.226%** |

### Decode (tok/s)

| Workload | hipEngine BF16 | Fork F16 | Fork lead | Fork Q8_0 | Fork lead |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512/128 | 52.710 | **64.658** | **+22.666%** | **64.246** | **+21.885%** |
| 4K/128 | 55.183 | **62.648** | **+13.528%** | **63.107** | **+14.361%** |
| 32K/128 | 45.943 | **53.220** | **+15.841%** | **57.553** | **+25.273%** |
| 64K/128 | 39.362 | **45.913** | **+16.642%** | **52.538** | **+33.474%** |

### Peak memory (GiB; descriptive scopes)

| Workload | hipEngine BF16 tracked | Fork F16 whole-GTT | hipEngine minus fork | Fork Q8_0 whole-GTT | hipEngine minus fork |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512/128 | 21.480 | 20.831 | +0.649 | 20.832 | +0.648 |
| 4K/128 | 23.007 | 21.161 | +1.847 | 21.253 | +1.754 |
| 32K/128 | 23.654 | 21.999 | +1.655 | 21.630 | +2.024 |
| 64K/128 | 24.392 | 22.871 | +1.521 | 22.239 | +2.153 |

hipEngine leads every full-prompt prefill row. The fork leads every tg128
decode row. Within the fork, Q8_0 versus F16 decode changes
**-0.637%/+0.734%/+8.142%/+14.430%** at 512/4K/32K/64K and saves
**0.632 GiB** at 64K. That isolates a long-context KV-bandwidth lever; it does
not explain the short-context gap.

This is a retained diagnostic, not a strict cross-engine performance or memory
claim. The exact weight file, hardware, and split phase shapes match, but KV
dtype, timing owner, memory scope, and output-oracle protocol do not. In
particular, llama-bench does not prove shared token/logit equality; hipEngine
memory is tracked/owned allocation while the fork is 10-ms whole-device UMA GTT.

The fork's published Q8-KV `pp512/tg32 at depth` command was also rerun on the
exact local GGUF with `-b 512 -ub 512 -r 3`. It measures pp512/tg32
**1391.809/64.406**, **1281.593/62.965**, **1067.613/60.513**,
**860.787/57.544**, and **623.634/52.462 tok/s** at d0/4K/16K/32K/64K. At
the three published overlap depths, those are **+1.707%/+6.952%**,
**+10.900%/+7.539%**, and **+16.044%/+6.521%** above the fork's different-
model Q4_K_XL row. This validates the local payload/backend; the model difference
precludes calling it a reproduction speedup. Complete commands, standard
deviations, qualifications, and raw hashes are in
[`2026-08-04-gfx1151-nathan-fork-q4km-matched-comparison.json`](../benchmarks/results/2026-08-04-gfx1151-nathan-fork-q4km-matched-comparison.json).

## Campaign 2: close the exact GGUF decode and memory gap

This is a new locally measured campaign, not a reopening of the completed six-
item source-transfer list. It prioritizes the gaps the local run actually
exposes.

### Measured signals and ownership

| Signal | Evidence | Campaign implication |
| --- | --- | --- |
| Prefill is already ahead | hipEngine leads fork F16 by **0.273%-6.536%** and Q8_0 by **0.505%-8.226%**. | Treat current prefill as a guard, not the next optimization target. Do not port MMID row lists, F16B, or generic ubatch values into a path already winning. |
| Nathan-specific F16 decode uplift is small | Versus the previous same-model vanilla Vulkan lane, fork F16 decode is only **+1.353%/+1.268%/+1.787%/+1.666%** at 512/4K/32K/64K. | Most of the BF16/F16 decode gap is base Vulkan-versus-hipEngine execution, not a missing Nathan toggle. Start with matched family attribution. |
| Q8_0 is a long-context lever | Fork Q8_0 is effectively neutral through 4K, then **+8.142%/+14.430%** versus its F16 lane at 32K/64K. | Separate short/mid weight-kernel work from long-context KV work. Do not credit Q8 for the 512/4K gap. |
| hipEngine short/mid decode is weight-kernel dominated | The retained eager profile attributes **14.628/14.588 ms/token** at 512/4K to dense Q8T16, selected-MoE T16, and Q6T16 lm-head work: **76.1%/80.6%** of measured GPU wall. | The first implementation lane needs a new exact algorithm/layout for row-1 weight kernels, not another graph or launch-only sweep. |
| The memory gap is not only KV | hipEngine tracked peak jumps **21.480 -> 23.007 GiB** from 512 to 4K while BF16 full-attention KV growth is only about **0.068 GiB**. The current full-attention query scratch uses 4,096 rows at 4K/32K/64K. | Audit and reduce/alias row-dependent prefill scratch before treating all of the **1.5-2.2-GiB** descriptive gap as a KV-format problem. |
| Q8_0 explains a bounded portion of 64K memory | Fork F16 -> Q8_0 saves **0.632 GiB** at 64K, close to halving the ten full-attention layers' K/V payload. | A strict compact-KV path is valuable but cannot close the entire memory gap; it also must pass hipEngine's stricter quality gate. |

Sources are the [local comparator](../benchmarks/results/2026-08-04-gfx1151-nathan-fork-q4km-matched-comparison.json),
the [prior vanilla Vulkan lane](../benchmarks/results/2026-07-17-gfx1151-amd-iommu-off-topline-refresh.json),
the [retained decode family profile](../benchmarks/results/2026-07-15-gfx1151-gguf-decode-closure-profile.json),
and the [current prefill/memory overlay](../benchmarks/results/2026-08-04-gfx1151-q4km-aotriton-head-major-prefill.json).
The family profile predates the final head-major prefill-only change, but decode
was measured neutral in that promotion.

### SH-C0 execution update (2026-08-05)

SH-C0 is complete. Fresh right-sized eager processes use one discarded full run,
three measured 128-token runs, exact repeated-token trajectories, one hardware
queue, cached builds, and simultaneous 10-ms whole-GTT sampling. Independent
8-token profiler children join nested ROCTX roles to asynchronous kernels by HIP
`Correlation_Id`; the fork runs its own Vulkan perf logger and is never mixed
into hipEngine timing. Profiled rates are diagnostic, not topline replacements.

| Context | hipEngine non-profiled decode tok/s | GPU kernel ms/token | Profiled host minus kernel ms/token | attributed dispatches | whole-GTT GiB | tracked GiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | **52.857** | 17.596 | 4.658 | 4,664/5,024 | 21.916 | 21.480 |
| 4K | **55.389** | 16.950 | 3.985 | 4,664/5,024 | 23.552 | 23.007 |
| 32K | **46.004** | 20.613 | 2.161 | 4,664/5,024 | 24.204 | 23.654 |
| 64K | **39.419** | 24.106 | 2.042 | 4,664/5,024 | 24.943 | 24.392 |

The same-scope memory result is now stronger than the former tracked-versus-GTT
description:

| Context | hipEngine whole-GTT | Fork F16 whole-GTT | hipEngine minus fork | Fork Q8_0 whole-GTT | hipEngine minus fork |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 21.916 | 20.831 | **+1.085** | 20.832 | **+1.084** |
| 4K | 23.552 | 21.161 | **+2.391** | 21.253 | **+2.299** |
| 32K | 24.204 | 21.999 | **+2.205** | 21.630 | **+2.575** |
| 64K | 24.943 | 22.871 | **+2.071** | 22.239 | **+2.704** |

Role attribution separates the work that the old symbol-only bucket combined:

| Context | dense Q8 projections | selected gate/up + down | full-attention core | lm-head | router/combine | total GPU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 8.384 | 3.920 | 1.579 | 1.831 | 0.980 | 17.596 |
| 4K | 8.405 | 3.912 | 0.912 | 1.837 | 0.979 | 16.950 |
| 32K | 8.420 | 3.925 | 4.533 | 1.836 | 0.986 | 20.613 |
| 64K | 8.429 | 3.914 | 8.041 | 1.833 | 0.982 | 24.106 |

`dense Q8 projections` above is the sum of GDN input/decay/output, full-attention
QKV/output, and shared-expert gate/up/down. Its largest constituent is now known:
GDN input projections cost **4.113/4.143/4.157/4.166 ms/token**. Selected
expert gate/up and down cost **2.317+1.603**, **2.309+1.603**,
**2.319+1.606**, and **2.311+1.603 ms/token**.

The fork logger independently reports these operation families:

| Context | Fork F16 total | dense Q8 | selected Q4+Q5 | F16 FA | Fork Q8 total | Q8 FA |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 16.440 | 7.562 | 2.975 | 0.186 | 16.533 | 0.150 |
| 4K | 16.881 | 7.507 | 2.978 | 0.660 | 16.765 | 0.415 |
| 32K | 19.622 | 7.524 | 2.965 | 3.401 | 18.261 | 1.930 |
| 64K | 22.608 | 7.515 | 2.966 | 6.390 | 19.871 | 3.595 |

The profilers perturb and overlap work differently, so those totals are not a
cross-engine speed ratio. They do select the next owners: hipEngine's dense
projection ledger is **0.822-0.913 ms/token** above the fork logger, selected
experts are **0.934-0.959 ms/token** above, and BF16 full-attention is
**0.252-1.652 ms/token** above except for the short 512 split-policy outlier.
SH-D1 therefore starts with the 4.11-4.17-ms GDN input role, then the selected
pair. SH-A1 remains gated but is now admitted as material at 32K/64K.

SH-M1's first screen is now decided. Reducing only full-attention query rows
from 4,096 to 1,024 does remove the predicted **1.335-1.338 GiB tracked** and
**1.351-1.355 GiB whole-GTT** at 4K+, but it changes exact state/logits from 4K
onward and loses **1.603%/7.997%/11.835%** prefill at 4K/32K/64K. Keep 4,096
rows. The 2,048/768 row-only sweep is closed: 2,048 cannot clear the 1-GiB gate,
and 1,024 already proves that deeper slicing is the wrong exact/performance
tradeoff.

Complete SH-C0 commands, role resources, GTT samples, owned components, fork
logs, and raw hashes are in
[`2026-08-05-gfx1151-gguf-sh-c0-attribution.json`](../benchmarks/results/2026-08-05-gfx1151-gguf-sh-c0-attribution.json).

### SH-M1 query-row execution update (2026-08-05)

Each policy ran in an independent right-sized process with one discarded run,
three measured prefill/decode runs, cached builds, one hardware queue, and
10-ms whole-GTT sampling. All five chunk controls are explicit; only
`full_attn_query` changes. Tracked allocations return exactly to process
baseline after close.

| Context | q4096 prefill | q1024 prefill | q1024 delta | q4096 decode | q1024 decode | q1024 delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 1359.472 | **1360.721** | **+0.092%** | 52.891 | **52.898** | **+0.013%** |
| 4K | **1438.860** | 1415.795 | **-1.603%** | 55.462 | **55.525** | **+0.113%** |
| 32K | **1154.435** | 1062.115 | **-7.997%** | **46.121** | 46.054 | **-0.144%** |
| 64K | **931.596** | 821.344 | **-11.835%** | 38.732 | **39.458** | **+1.876%** |

| Context | q4096 tracked | q1024 tracked | saving | q4096 whole-GTT | q1024 whole-GTT | saving |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 21.480 | 21.480 | 0.000 GiB | 21.916 | 21.916 | 0.000 GiB |
| 4K | 23.007 | 21.672 | **1.335 GiB** | 23.552 | 22.201 | **1.351 GiB** |
| 32K | 23.654 | 22.317 | **1.337 GiB** | 24.204 | 22.849 | **1.355 GiB** |
| 64K | 24.392 | 23.054 | **1.338 GiB** | 24.943 | 23.587 | **1.355 GiB** |

The 512 state is byte-exact because both right-sized sessions resolve only 768
scratch rows. At 4K/32K/64K, q1024 changes prefill logits, final hidden state,
37 layer outputs, 54 Conv/GDN state parts, 18 later-layer K/V parts, the first
fixed-input decode logit row, and final decode state. The sampled ID remains
9707, which is not sufficient for the exact contract. The first changed layer
output is layer 3, where the new full-attention chunk boundary changes F32
association; later state then inherits the difference.

Even if it were admissible, q1024 whole-GTT would remain **0.716-1.040 GiB**
above fork F16 and **0.948-1.348 GiB** above fork Q8_0 at 4K+. Beating the
memory row therefore requires a stack, not a smaller query chunk: first alias
mutually-exclusive dedicated 4,096-row linear/full-attention/MoE fields without
changing execution shape, then combine that exact saving with a strict compact-
KV result. The current dedicated scratch's largest owners are
`moe_down_out_f32` (**256 MiB**) and `conv_out`, `linear_qkv_f32`, and
`moe_down_out` (**128 MiB each**).

The complete diagnostic, commands, raw hashes, exact-state summary, and
provenance qualification are in
[`2026-08-05-gfx1151-gguf-sh-m1-q1024-rejected.json`](../benchmarks/results/2026-08-05-gfx1151-gguf-sh-m1-q1024-rejected.json).
The aggregate collector saw unrelated untracked files, so this is not a
retainable performance claim; staged and unstaged tracked source were clean at
`e6eb49628`, and the candidate independently fails both correctness and prefill
stop rules.

### Cumulative decode targets

These are reporting/stop targets, never license to specialize to the repeated-
token benchmark. Every candidate must also win on natural prompts and category
heldouts with no token-, prompt-, or candidate-ID-conditioned branch.

| Stage | 512/128 | 4K/128 | 32K/128 | 64K/128 |
| --- | ---: | ---: | ---: | ---: |
| Current SH-C0 hipEngine BF16 | 52.857 | 55.389 | 46.004 | 39.419 |
| C1: close at least half the F16 **time** gap | **58.165** | **58.795** | **49.350** | **42.419** |
| C2: match the local fork F16 lane | **64.658** | **62.648** | **53.220** | **45.913** |

The exact/default lane targets F16 parity, not the fork's approximate Q8_0
headline. Preserve current prefill within **1%** at every context while keeping
all existing exact state, allocator-lifecycle, and fallback gates. For memory,
the first internal target is at least **1.0 GiB** less hipEngine tracked peak at
4K+ without more than **1%** prefill loss. SH-C0 now supplies same-scope
whole-GTT peaks for both engines; component attribution remains an own-engine
tracked/owned comparison.

### Ordered work packages

| ID | Priority | Hypothesis and experiment | Promotion / stop rule |
| --- | --- | --- | --- |
| **SH-C0 — matched attribution freeze** | **Completed 2026-08-05** | Role-resolved HIP correlation traces, separate fork F16/Q8_0 logger runs, 10-ms same-scope GTT, and owned allocation/lifetime census now cover 512/4K/32K/64K. | No production change. GDN input, selected experts, long-context attention, and 4,096-row scratch all clear admission; see the execution update and artifact. |
| **SH-D1 — exact row-1 weight-kernel redesign** | **P0; audit complete 2026-08-05** | Split dense Q8T16 by role (full Q/K/V/O, GDN projections, shared expert), selected Q4/Q5/Q6 T16, and lm-head. The first implementation candidate is now the exact permlanex16/DPP-add reduction sibling for the row-1 GDN `attn_qkv+attn_gate` pair; replacement-layout and non-temporal alternatives remain behind its measured gate. Compare PARO/Vulkan as algorithm/shape references, not drop-in formats. | Primitive CPU oracle, named trace, no scratch spill, and full model quality/state gate. A leaf must be at least **1.15x** or save **0.5 ms/token** before a full-model run. First cumulative gate is the half-time-gap row above; continue until F16 parity or measured Amdahl headroom is exhausted. Reject any 512/4K win that loses >1% prefill or regresses another context. |
| **SH-M1 — query-row Pareto screen** | **Completed/rejected 2026-08-05** | The explicit 4,096 -> 1,024 query-row A/B saves **1.335-1.338 GiB tracked** and **1.351-1.355 GiB whole-GTT** at 4K+, with exact cleanup and neutral decode. | Reject q1024 and close 2,048/768 row-only retries: q1024 changes exact state/logits at 4K+ and loses **1.603%-11.835%** prefill. Keep q4096. |
| **SH-M2 — exact scratch-liveness aliases** | P0, parallel with D1 | Keep every 4,096-row execution shape fixed. Build a route/stage liveness map for the currently `dedicated` scratch, beginning with the 256-MiB `moe_down_out_f32` and three 128-MiB owners; alias only fields proven non-overlapping across the isolated AOTriton stream and linear/full-attention/MoE phases. | Require byte-exact logits/hidden/Conv/GDN/KV state at 512/4K/32K/64K, zero tracked close delta, <=**1%** prefill/decode loss, and >=**1.0 GiB** tracked plus same-direction whole-GTT saving. Do not claim fork memory parity until the exact alias and strict-KV stack beats both fork rows. |
| **SH-K1 — strict compact-KV frontier** | P1, after C0 | Do not repeat failed all-layer per-token/head, q8_0-block32, block16, or K-only screens. Start from the passing guarded 2/10-layer hybrid and sweep fixed layer policies and K-only versus K+V only if they are genuinely new. Use the complete multi-prompt quality suite plus category-heldouts, then profile the existing grouped-GQA direct consumer at 32K/64K. | Exact/default promotion still requires KL <= **0.05**, aggregate and minimum-prompt top-1 >= **90%**, no BF16 mirror in the claimed compact layers, and repeated speed/memory wins. An all-layer relaxed mode may be reported separately with its quality loss, but it never replaces or inflates the strict default claim. |
| **SH-A1 — page-internal head-major decode screen** | P2, only if C0 leaves BF16 attention material | Microbenchmark the current `[block, token, kv_head, D]` layout against a page-internal `[block, kv_head, token, D]` variant with complete `KVLiveSpans`, charging append/copy and reducer wall. Do not first rewrite every writer/compactor/graph. | Proceed to runtime plumbing only if copy/append-inclusive attention improves >=**1.10x**, projected whole decode saves >=**1%**, memory does not grow, and dense/permuted/evicted page oracles are exact. The fork's F16 lane being only 1.27%-1.79% above vanilla makes this a bounded screen, not P0. |
| **SH-G — retained recertification** | Required after each retained unit | Re-run hipEngine one-warmup/three-measurement 512/4K/32K/64K rows, exact natural/heldout prompts, allocator/GTT sampling, and the applicable cached kernel trace. Re-run the five-repetition fork comparator only at a campaign milestone, not after every micro-change. | Publish an own-engine A/B only when exact and non-regressive. Keep the external row diagnostic until timing ownership, dtype, and output oracle align. Update artifact, benchmark rollup, changelog, and worklog for every retained performance unit. |

#### SH-D1 GDN-input audit update

The first SH-D1 leaf is frozen in
[`2026-08-05-gfx1151-gguf-sh-d1-gdn-input-audit.json`](../benchmarks/results/2026-08-05-gfx1151-gguf-sh-d1-gdn-input-audit.json).
The active row-1 `attn_qkv[2048->8192] + attn_gate[2048->4096]` owner runs once
in each of 30 linear-attention layers and costs **4.113-4.166 ms/token**. A
three-copy, 80.2-MB cycling leaf reproduces the role at **135.16 us/call** and
**197.8 GB/s** effective matrix reads.

The current gfx1151 code object is already vectorized, spill-free, and
scratch-free at 50 logical VGPR, but its exact 16-column wave reduction compiles
to **80 `ds_bpermute_b32`** instructions plus **67 `s_waitcnt`** instructions.
SH-D1 therefore starts with a separately registered exact sibling that changes
only the 16/8/4/2/1 peer tree to `permlanex16` plus direct DPP adds. It must
preserve the 128-thread/four-wave K partition, every FMA, lane-0 publication,
one barrier, serial wave0..3 sum, BF16 bytes, ABI, and production fallback. The
cycling leaf must reach **<=118.493 us** (projected >=0.5 ms/token saving), with
**<=117.530 us** satisfying the independent 1.15x rule, before any full-model
run. Non-temporal loads and a byte-neutral paired Q8T32 slab are deferred until
this same-layout communication seam is measured.

### Campaign guardrails and closed retries

- Keep the persistent `KVLiveSpans` ABI, four-axis registry, torch-free runtime,
  and unfused fallback for every new fused owner.
- Do not repeat the closed 64/256-thread dense-Q8 sweep, 128/512 split-chunk
  alternatives, all-eight-query-head register tile, Q8T16 d-scale cache,
  row-GEMV launch-width sweep, or launch-only graph work. Current graph replay
  is about a 1% lever; host/C++ work requires the project-wide >3% dispatch-wall
  trigger.
- Do not call fork Q8_0 an exact target. Existing GGUF pure INT8 and component-
  only formats have prompt-dependent quality failures; a new strict lane needs a
  new fixed policy/format and complete quality evidence.
- Do not optimize the repeated token or the four publication lengths with
  prompt-conditioned behavior. Development screens must include all natural
  prompt categories and heldouts before retention.
- The full persistent head-major cache rewrite remains closed for prefill. SH-A1
  authorizes only a decode-layout micro-screen; it does not authorize changing
  every paged-KV consumer without a measured whole-decode gate.

## Current hipEngine baseline relevant to the comparison

### Persistent KV and prefill layout

The Qwen GGUF session defaults to BF16 fixed-page KV, accepts guarded
`int8_per_token_head`, and records storage/layout policy explicitly in
[`hipengine/runtime/qwen35_gguf_runner.py`](../hipengine/runtime/qwen35_gguf_runner.py).
The retained physical cache layout used by attention is logically:

```text
[num_blocks, block_size=256, num_kv_heads, head_dim]
```

That is token-major/head-interleaved inside a page. The AOTriton call constructs
K/V tensor views as `[1, Hkv, context, D]` with strides that step by
`Hkv * D` between tokens; see
[`_run_full_attention_prefill_layer_aotriton`](../hipengine/runtime/qwen35_gguf_runner.py).
This is the same layout class Nathan found expensive in RADV coopmat1, although
the consumer is different and must be measured rather than assumed equivalent.

Qwen3.6 uses only ten full-attention layers, so any win is diluted by the other
30 GDN/linear-attention layers. The current gfx1151 publication retains GGUF
through 64K and blocks repeated 128K on lifecycle safety; see
[`2026-07-17-gfx1151-amd-iommu-off-topline-refresh.json`](../benchmarks/results/2026-07-17-gfx1151-amd-iommu-off-topline-refresh.json)
and [`DEBUG-GFX1151-STALL.md`](DEBUG-GFX1151-STALL.md).

### Quantized KV

hipEngine GGUF `int8_per_token_head` is not llama.cpp `q8_0`; the format and
quality distinctions are documented in [`GGUF.md`](GGUF.md#gguf-q8--int8-kv-cache-status).
The normal performance path already follows the useful part of Nathan's
"dequantize once and reuse" lesson:

- direct streaming INT8 prefill exists but is a capacity diagnostic because it
  is much slower than the temporary BF16/AOTriton bridge;
- retained GGUF INT8 prefill can write a temporary layer-local BF16 cache for
  attention while separately retaining INT8 K/V;
- short guarded GGUF INT8 sessions may retain bounded BF16 mirrors for strict
  correctness;
- pure/no-mirror GGUF INT8 is not the default because it failed the project
  quality gate on relevant prompts.

The 2026-08-04 execution audit preserves this split. Current PARO `auto` policy
keeps the BF16-oracle/AOTriton bridge below 224 Ki tokens and on larger-memory
systems even above that threshold; direct streaming requires both very-long
context and memory pressure, or an explicit diagnostic override. GGUF retained
INT8 prefill continues to write layer-local BF16 attention-oracle K/V separately
from retained INT8 K/V. A new structural guard proves that replacing the layer's
primary K/V with this oracle preserves the admitted gfx1151 head-major scratch,
so the bounded 64K AOTriton consumer from priority 1 applies without an INT8-
specific route. Fresh host/policy coverage and the gfx1151 direct-INT8 NumPy
primitive gate pass. The retained 128K evidence still shows direct streaming
prefill regressing **1020.723 -> 23.425 tok/s (-97.7%)**, so no new performance
run or promotion is warranted.

The decode kernel is already grouped by KV head. In
[`qwen35_paged_full_attn_decode_split_k_ctx_tensor_gqa_int8_kernel`](../hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.hip),
one workgroup owns one `(kv_head, split)`, loads/dequantizes K/V once, and loops
over all `q_per_kv=8` query heads. This is the same redundancy removal that
Nathan added to llama.cpp HIP tile attention.

### MoE row compaction

hipEngine already has the full row-list pipeline in
[`group_scatter.hip`](../hipengine/kernels/hip_gfx1100/moe/group_scatter.hip):

1. count routed lanes per expert;
2. exclusive-prefix those counts into exact expert starts;
3. build a stable expert-major compact row list and active-expert list;
4. map compact starts to WMMA/MMQ tiles;
5. launch only active expert/tile work.

The gfx1151 package selects parallel count/prefix/scatter for Laguna through
`LAGUNA_MOE_GROUP_COMPACT_MODE = "parallel"` in
[`hip_gfx1151/__init__.py`](../hipengine/kernels/hip_gfx1151/__init__.py).
This is structurally the same fix as Nathan's `MUL_MAT_ID` row-list prepass, not
a missing port.

### Architecture-specific shape tuning

The gfx1151 package is already intentionally different from gfx1100. Among
other retained settings it uses:

- 256-row Qwen/PARO linear and MoE prefill chunks in
  [`runtime/prefill.py`](../hipengine/runtime/prefill.py);
- model/shape-qualified Laguna selected gate/up and down MMQ schedules;
- BF16/FP16 or quantized activation representations instead of the Vulkan
  MMID F32-B baseline;
- exact grouped-GQA, tile, staged-value, prefetch, and dense-prefix attention
  variants;
- fair 256-token server prefill chunks for Q4_K_M.

Nathan's `-ub 1024/2048` guidance is therefore useful evidence that batch shape
matters, but it is not a value to copy into hipEngine. The local 256-row profile
was selected by same-device exact A/B and supersedes generic llama.cpp ubatch
advice for these paths.

## Applicability matrix

Status meanings:

- **Present** — the mechanism or a stronger equivalent is on the current path.
- **Measure** — transferable hypothesis, but no local A/B yet.
- **Future model** — valid for a model architecture hipEngine does not support.
- **Backend-specific** — tied to Vulkan/RADV or llama.cpp graph internals.
- **No action** — negative, reverted, or superseded by local evidence.

### Flash Attention and KV

| Nathan change | Source evidence | hipEngine status | Decision |
| --- | --- | --- | --- |
| Quantized-KV dequantize+transpose once for Vulkan prefill | The Vulkan route explicitly creates per-head-contiguous FP16 scratch and reuses it ([`484ad9b`, lines 10263-10470](https://github.com/Nathanw1014/llama.cpp/blob/484ad9ba068ad946a835b6097558c5b15603aae3/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L10263-L10470)). | **Present and revalidated.** Normal hipEngine quantized-KV prefill uses a temporary BF16/AOTriton bridge instead of repeatedly consuming retained INT8, and GGUF's layer-local BF16 oracle preserves the admitted head-major consumer. | Do not port the Vulkan shader. Retain the bridge and its dedicated policy/head-major regression guard. |
| All-quant q4/q5 extension | Toolbox inventory identifies the extension and its correctness routing ([BRANCHES lines 32-36](https://github.com/Nathanw1014/strix-halo-llamacpp/blob/b166a56e58ab0f27fd03f60fff060eebdf5f64b5/BRANCHES.md#L32-L36)). | **Not format-compatible.** hipEngine's KV INT8 is sideband-scale, not GGML q4/q5/q8 blocks. | No literal port. Any future KV format gets its own CPU oracle and registry quant axis. |
| Contiguize strided BF16 K/V before prefill FA | The copy shader converts the interleaved source to contiguous output ([`ab5910a`, lines 9-31](https://github.com/Nathanw1014/llama.cpp/blob/ab5910a15e85b919b228193ed297a35beaf135c6/ggml/src/ggml-vulkan/vulkan-shaders/dequant_f16_transpose.comp#L9-L31)); toolbox reports the long-context effect ([README lines 103-109](https://github.com/Nathanw1014/strix-halo-llamacpp/blob/b166a56e58ab0f27fd03f60fff060eebdf5f64b5/README.md#L103-L109)). | **Completed and promoted.** A bounded tracked head-major pair is the gfx1151 default through rounded capacity 65,792. | Retain the copy-inclusive default: 32K/64K improve **3.383%/7.001%** with exact state and bounded strided fallback. |
| Persistent head-major K/V cache | The experimental layout changes K from token-major to `[head_dim, kv_size, n_head_kv]` ([`0f74840`, lines 231-254](https://github.com/Nathanw1014/llama.cpp/blob/0f748408e2af0f4fe05b2ccdf7a7765bf6cc29fe/src/llama-kv-cache.cpp#L231-L254)). Later commits restrict formats/consumers after correctness failures. | **Rejected after the prefill scratch A/B.** All paged writers, decode kernels, copies, compaction, graph captures, and `KVLiveSpans` consumers assume the current physical row layout, while the 64K copy is only **0.032%** of full prefill across ten layers. | Keep persistent paged KV unchanged for prefill. SH-A1 may test only a page-internal decode layout before any runtime rewrite. |
| HIP tile dequant-on-load, shared across GQA heads | The tile loader dequantizes into SRAM once and reuses it across `ncols2` query heads ([`b781a8d`, lines 485-547](https://github.com/Nathanw1014/llama.cpp/blob/b781a8d5dc73331b4f8413dcf820d017e1938c67/ggml/src/ggml-cuda/fattn-tile.cuh#L485-L547)). | **Present and guarded.** hipEngine INT8 split-K decode is KV-head grouped and shares K/V across all eight query heads. | No port. The new source/launch regression guard freezes `(kv_head, split)`, not `(q_head, split)`. |
| P-fragment load hoist | The P fragments move outside the `hsv_tile` loop ([`e11cafa`, lines 428-437](https://github.com/Nathanw1014/llama.cpp/blob/e11cafa02f96b009c3088f9f601edc13e75524ab/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm1.comp#L428-L437)). | **Backend-specific.** hipEngine's production Qwen prefill core is a precompiled AOTriton image, not this GLSL shader. | Feed upstream to AOTriton/native-FA work only if profiling makes FA a top wall component. |
| `Psh` query-major relayout | The relayout changes cooperative-matrix load orientation ([`40f85eb`, lines 47-51 and 382-437](https://github.com/Nathanw1014/llama.cpp/blob/40f85eb859959d9416f601deef287275d354680f/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm1.comp#L382-L437)). Nathan reports no standalone speed win. | **Backend-specific / no action.** | Do not reproduce a perf-neutral GLSL layout change in HIP. |
| Head-size-gated Vulkan wave32 | `dfb619c` controls Vulkan subgroup selection; toolbox says it is not yet upstream-ready without more hardware coverage ([README lines 153-158](https://github.com/Nathanw1014/strix-halo-llamacpp/blob/b166a56e58ab0f27fd03f60fff060eebdf5f64b5/README.md#L153-L158)). | **Already native/consumer-owned.** gfx1151 HIP has wave32 and AOTriton selects gfx11xx images. | No host knob copy. Inspect selected AOTriton image metadata only if the contiguity A/B leaves an FA residual. |
| Non-native KV-type routing hardening | Nathan unified admission/dispatch after an iq4_nl correctness hole ([README lines 115-119](https://github.com/Nathanw1014/strix-halo-llamacpp/blob/b166a56e58ab0f27fd03f60fff060eebdf5f64b5/README.md#L115-L119)). | **Present architectural rule.** hipEngine quant/layout routes are exact four-axis registry keys and wrappers validate `KVLiveSpans` dtype/scale metadata. | Retain exact-key tests; no new route. |

### MoE prefill and matrix kernels

| Nathan change | Source evidence | hipEngine status | Decision |
| --- | --- | --- | --- |
| MMID row-list prepass | Prefix counts and scatter packed rows once, then kernels read direct lists ([`ffe5cb4` shader lines 5-87](https://github.com/Nathanw1014/llama.cpp/blob/ffe5cb4a9e144a16a94a28f88d02c52f6133261f/ggml/src/ggml-vulkan/vulkan-shaders/mmid_row_lists.comp#L5-L87)). | **Present and coverage-audited.** hipEngine count/prefix/stable-scatter/active-expert/tile-map pipeline is the same algorithmic class, with natural-shape and trace guards. | No port. Profile the local metadata only if it becomes material after weight-kernel wins. |
| Select tile from expected per-expert rows (`SMALLN`) | [`954ae8e`](https://github.com/Nathanw1014/llama.cpp/commit/954ae8edd16ad2f788130aef8b9f64738c8aecb2) makes tile choice depend on per-expert occupancy. | **Present and more specific.** hipEngine has exact model/quant/row-qualified package schedules and tile maps. | Continue local measured selectors; do not import env heuristics. |
| Taller M tiles (`BM64`, `M128`) | [`fbec25f`](https://github.com/Nathanw1014/llama.cpp/commit/fbec25f2e79bcf9fc03cebee69f4ee1fba3aa34c) and [`7c3ba9f`](https://github.com/Nathanw1014/llama.cpp/commit/7c3ba9f6df00d2338508c2153ce628ca26af02b0) reduce repeated operand reads at particular ubatches. | **Present as a tuning dimension.** Laguna/Qwen kernels already carry 32/64/128-row and model-qualified schedules. | Use Nathan's result as a reminder to sweep tile M with real per-expert occupancy, not as a direct tile selection. |
| FP16 B activations (`F16B`) | [`b47a5b1`](https://github.com/Nathanw1014/llama.cpp/commit/b47a5b1cf7df7bad76b37616e0b90a5314c49580) converts Vulkan MMID F32 activations to F16. | **Present.** hipEngine's main MoE paths already use BF16/FP16 or explicitly quantized activation layouts. | No action. |
| MMID wave32 | [`4a5cf2d`](https://github.com/Nathanw1014/llama.cpp/commit/4a5cf2d8247718ecf25137b70cabc4a04d0a4e30) repairs Vulkan workgroup geometry when forcing subgroup 32. | **Backend-specific.** hipEngine gfx11 kernels are authored for wave32 directly. | No action beyond normal resource/profiler checks. |
| Q4/Q5 scale cache | Initially positive, then disabled after later tile changes made it regress; toolbox records -4% to -20% on the current stack ([BRANCHES lines 81-85](https://github.com/Nathanw1014/strix-halo-llamacpp/blob/b166a56e58ab0f27fd03f60fff060eebdf5f64b5/BRANCHES.md#L81-L85)). | **No missing win.** hipEngine's retained paths use different raw/repacked layouts; local raw-dequant and precompute candidates already require end-to-end A/B. | Do not port. This is evidence against carrying unmeasured caches after tile/layout changes. |
| `TILE16` | Nathan measured more expert weight re-streaming and a regression ([BRANCHES lines 87-94](https://github.com/Nathanw1014/strix-halo-llamacpp/blob/b166a56e58ab0f27fd03f60fff060eebdf5f64b5/BRANCHES.md#L87-L94)). | **No action.** | Keep as a negative design lesson: do not make N smaller than typical per-expert occupancy without accounting for repeated weight traffic. |
| Scalar packed-int MMID | [`e8ba41b`](https://github.com/Nathanw1014/llama.cpp/commit/e8ba41b90c743ac73dbdf7912f646c82da050c8e) lost to cooperative F16 despite lower activation bytes. | **Not directly transferable, but cautionary.** hipEngine has measured dp4a/WMMA/T16 alternatives and retains them per exact shape. | Do not infer all integer kernels lose; do require full-model evidence rather than byte-count reasoning. |
| Larger `-ub` | Toolbox recommends model-specific 1024/2048 and explicitly reports that 2048 regresses Qwen3.6 shallow prefill. | **Already model/architecture tuned and revalidated.** | Keep hipEngine's measured gfx1151 256-row linear/MoE policy. Re-sweep only when the active kernel/layout changes. |

### DeepSeek V4

| Nathan change | Source evidence | hipEngine status | Decision |
| --- | --- | --- | --- |
| Lightning indexer + indexed sparse prefill FA | The fork adds scalar/coopmat indexer kernels and a top-k FA API ([`163bfd9`](https://github.com/Nathanw1014/llama.cpp/commit/163bfd91584df060695583c8b7a62e4a7d2cdcfb)). | **Future model.** hipEngine has no DeepSeek V4 model plugin. | Preserve as primary source material for a future model+layer+kernel plugin. Do not add DSv4 branches to Qwen/Laguna dispatch. |
| Gather-to-compact sparse decode | The gather copies dense-prefix plus selected rows into a compact KV/mask buffer before ordinary FA ([`2f651ad`, lines 37-64](https://github.com/Nathanw1014/llama.cpp/blob/2f651ad5df0663b937c55ca12af4e42e84b66adc/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_gather.comp#L37-L64)). | **Future model; concept aligns with `KVLiveSpans`.** | Implement as a sparse policy/attention variant when DSv4 exists. Keep dense fallback and validate invalid/padded indices. |
| Fused hyper-connection pre/comb/post | The comb shader performs per-token softmax and Sinkhorn normalization ([`3bc783c`, lines 3-56](https://github.com/Nathanw1014/llama.cpp/blob/3bc783ca7d55e00291b8e92a556e729ba6130685/ggml/src/ggml-vulkan/vulkan-shaders/dsv4_hc_comb.comp#L3-L56)). | **Future model.** Current Qwen/Laguna layers do not have this operation. | Port from the model's reference under new layer/variant registry keys with an unfused fallback and CPU oracle. |
| Keep indexer key cache F16 under quantized main KV | The fused indexer explicitly requires F16 keys ([`487b923`, lines 1100-1127](https://github.com/Nathanw1014/llama.cpp/blob/487b923a33165bb6d8e3405951bb26416aa00575/src/llama-kv-cache-dsv4.cpp#L1100-L1127)). | **Future model.** | Treat indexer KV as a distinct policy/quant role; never assume the main KV quant applies to every auxiliary cache. |
| Contiguize small-B grouped O-projection input | [`637e4de`](https://github.com/Nathanw1014/llama.cpp/commit/637e4dec5942fcb078bfa51da456c1aec78c8cde) targets DSv4 2-8-token verify batches. | **Future model; general verifier lesson.** hipEngine already owns explicit packed verifier buffers for current models. | Re-evaluate on DSv4 B2/B4 traces; do not add an unconditional copy. |

### Runtime, robustness, host, and packaging

| Nathan change | Source evidence | hipEngine status | Decision |
| --- | --- | --- | --- |
| Bound Vulkan command buffers by estimated bytes | The fork defaults to an 8-GiB traffic cap ([`e709b94`, lines 6628-6631](https://github.com/Nathanw1014/llama.cpp/blob/e709b949e7ef43db08a7b1f42d0d6a5a18946153/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L6628-L6631)) and submits when accumulated bytes cross it ([line 17967](https://github.com/Nathanw1014/llama.cpp/blob/e709b949e7ef43db08a7b1f42d0d6a5a18946153/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L17967)). | **Backend-specific, analogous issue only.** HIP/AQL submission has no llama.cpp Vulkan command-buffer batching layer. hipEngine has an exact, qualified, default-off layer `hipStreamSynchronize` containment path. | Do not emulate byte estimates in Python and do not restore the rejected MES workaround. Keep layer drain explicit while ROCm/ROCm#6437 remains open. |
| Bound FA scratch and fall back when it cannot remain resident | [`e21d01e`, lines 10545-10562](https://github.com/Nathanw1014/llama.cpp/blob/e21d01ed4ddb4eb0193c148daa2569972bcfd115/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L10545-L10562) turns an oversized Vulkan storage-buffer abort into fallback. [`8a2c6b2`, lines 10778-10834](https://github.com/Nathanw1014/llama.cpp/blob/8a2c6b29c45bf0346ad9dde6a0ae1b38ac005b13/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L10778-L10834) adds a discrete-VRAM residency gate; the commit explicitly exempts UMA. | **Guard transferred; Vulkan policy rejected.** gfx1151 uses bounded tracked admission for one reusable pair, while UMA has no imported discrete-heap heuristic. | Capacity/byte/registry/allocation failure selects exact strided AOTriton; high-water is recorded and `GGML_VK_FA_DEQUANT_RESERVE_MB` is not ported. |
| `amd_iommu=off` | Toolbox reports a modest prefill effect and a DMA-isolation tradeoff ([README line 48](https://github.com/Nathanw1014/strix-halo-llamacpp/blob/b166a56e58ab0f27fd03f60fff060eebdf5f64b5/README.md#L48)). | **Already exercised.** Current gfx1151 publication uses IOMMU-off but correctly says cross-revision deltas are not causal; XDNA is unavailable in this boot. | No engine change. A causal claim still needs a same-commit reboot A/B. |
| Verify the actual GPU backend | `v0.2` fixed silent CPU fallback; the README requires checking the backend column ([README lines 86-90](https://github.com/Nathanw1014/strix-halo-llamacpp/blob/b166a56e58ab0f27fd03f60fff060eebdf5f64b5/README.md#L86-L90)). | **Present process rule.** hipEngine artifacts record backend/arch and kernel traces verify expected symbols. | Keep explicit backend/arch and trace evidence in every retained benchmark. |
| Bundle Mesa/libdrm and repair ICD metadata | Vulkan distribution concern. | **Not applicable.** hipEngine is a native HIP runtime and does not ship a RADV ICD. | No action. Pin HIP/compiler provenance instead. |
| Perf-logger graph-split flush | llama.cpp Vulkan instrumentation fix in the v0.4 stack. | **Not applicable to hot path.** | Only borrow the principle that timing boundaries must flush/record the queue they claim to measure. |

## Executed experiment: BF16 head-contiguous AOTriton prefill

### Original hypothesis and result

For long-context Qwen3.6 full-attention layers on gfx1151, AOTriton pays a
material penalty when K/V tokens for one head are separated by
`num_kv_heads * head_dim` BF16 elements. Copying the visible paged prefix once
into reusable `[Hkv, context, D]` contiguous BF16 scratch will save more
AOTriton time than the copy costs.

**Result:** the hypothesis is confirmed for long context and neutral at short
context. Copy-inclusive full prefill changes 512/4K/32K/64K by
**-0.028%/+0.616%/+3.383%/+7.001%**; the complete state is byte-exact. The
bounded implementation and evidence are retained in priority 1 above.

### Minimal implementation shape

1. Add a registry-resolved BF16 paged-KV-to-head-major copy kernel under the
   attention layer, not a backend branch in the runner.
2. Consume complete `KVLiveSpans` metadata so non-identity page tables remain
   correct. A dense-prefix specialization may exist only with an explicit
   predicate and generic fallback.
3. Allocate one K and one V scratch sized to the active context and reuse them
   across full-attention layers. Do not allocate one duplicate per layer. Admit
   the scratch through tracked runtime capacity; allocation/admission failure
   must select an existing attention path rather than abort or overcommit.
4. Run the copy after append and before AOTriton; pass contiguous K/V strides to
   AOTriton.
5. Keep current strided AOTriton and native paged attention registered as
   fallbacks.
6. Apply the same scratch consumer to default BF16 KV and to the layer-local BF16
   oracle used by retained INT8 KV; do not add a GGML q8/q4 format.

### RED/GREEN and measurement gates

**Primitive correctness**

- Synthetic page permutations, lengths `1/255/256/257`, two KV heads, and
  untouched-sentinel regions.
- Byte-exact copy against a NumPy gather/transpose oracle.
- Dense-prefix and generic-page variants must produce identical contiguous
  tensors.
- A forced scratch-capacity denial must select the existing strided/native
  fallback, produce the same output, and leave no partial allocation.

**Attention correctness**

- Same Q/K/V and causal positions through current strided AOTriton versus copied
  contiguous AOTriton.
- Run the normal full-model bulk-prefill hidden/state/KV gate.
- For any arithmetic path change, require project correctness thresholds
  `KL <= 0.05` and top-1 `>= 90%`; because this should be a layout-only change,
  investigate any material drift rather than accepting the threshold by
  default.

**Performance**

- Measure copy-inclusive wall and AOTriton kernel time at 512, 4K, 32K, and 64K.
- Use the current exact Qwen3.6 35B-A3B UD-Q4_K_M file, BF16 KV, gfx1151,
  cached builds, one hardware queue, and the same benchmark process protocol as
  the retained GGUF row.
- Require 512/4K non-regression and repeated 32K/64K improvement. Report the
  attention sub-window separately from end-to-end prefill.
- Record scratch bytes and tracked/sampled high-water.
- Use `rocprofv3 --kernel-trace` only after prebuilding; confirm both the copy
  symbol and expected AOTriton image launch.

**Promotion rule**

Promote only if copy-inclusive full prefill is exact/non-regressive at short
context and repeatedly faster at long context. A faster AOTriton sub-window that
loses end-to-end wall is a rejected experiment.

**Gate result:** passed. Short context is neutral/positive, both long contexts
improve repeatedly, full state is exact, allocation denial falls back exactly,
and the named copy/AOTriton kernels appear in the cached trace.

Do not include repeated 128K in the first screen. The existing lifecycle gate is
more than five minutes and remains subject to explicit approval and the
`DEBUG-GFX1151-STALL.md` protocol.

## 128K robustness interpretation

Nathan's byte-capped Vulkan submission and hipEngine's layer drain express the
same broad lesson: do not let unbounded long-context work accumulate behind a
single opaque retirement boundary. They act at different layers:

- Nathan controls when Vulkan command buffers are submitted.
- hipEngine already submits HIP launches to AQL and can only add host-side stream
  drains without replacing the runtime.

The open hipEngine capture has an active non-empty compute queue with unread AQL
packets and no reported HQD error. Follow-up upstream and live-kernel evidence
changes the former system recommendation:

1. [`1fb710793ce2`](https://github.com/torvalds/linux/commit/1fb710793ce2619223adffaf981b1ff13cd48f17)
   introduced `enable_lr_compute_wa`, but upstream later said it did not fully
   fix gfx1151 hangs.
2. [`b42f3bf9536c`](https://github.com/torvalds/linux/commit/b42f3bf9536c9b710fd1d4deb7d1b0dc819dc72d)
   corrected gfx1151's KFD VGPR-size accounting from the generic 256 KiB to
   384 KiB per CU.
3. [`6b0d81297137`](https://github.com/torvalds/linux/commit/6b0d812971370c64b837a2db4275410f478272fe)
   removed `lr_compute_wa`, explicitly citing incomplete efficacy and
   instability on other products.
4. The exact captured CachyOS source includes gfx1151 in the 384-KiB branch
   ([`kfd_queue.c` lines 412-427](https://github.com/CachyOS/linux/blob/0e558f948dfe28b50d2eb9ddda58900d7de01aac/drivers/gpu/drm/amd/amdkfd/kfd_queue.c#L412-L427)),
   and the running KFD topology reports `cwsr_size=19185664`, exactly the value
   computed with that correction rather than the old `13942784` value.

Therefore do **not** patch or test `lr_compute_wa`. The actual upstream fix was
already active when this workload stalled, so ROCm/ROCm#6437 remains a distinct
or incompletely fixed queue-retirement problem. Keep the qualified
`--prefill-queue-drain layer` path explicit/default-off. Only test a newer
kernel when it contains a relevant additional fix or as an approved broad
system screen; only consider finer application batching with measured cost and
without a firmware/driver root-cause claim.

A single successful 128K pass is not closure; any future default-path stack gate
still requires at least three independent warmup+3 processes with exact IDs,
finite logits, normal telemetry, and clean logs.

## Future DeepSeek V4 plugin checklist

**Execution decision (2026-08-04): deferred at the model-admission gate.** The
live model registry contains Laguna, Moonshine, Qwen3.5/3.6, and toy plugins,
but no DeepSeek architecture. Local model storage and the Hugging Face cache
contain no DeepSeek V4 checkpoint/config, the repository has no DeepSeek V4 CPU
reference or tokenizer fixture, and the phase model table in `PLAN.md` does not
approve a DeepSeek V4 target. Generic old DeepSeek vocab/conversion files are not
a model oracle. Likewise, existing `q8_1_ds4*` symbols mean llama.cpp's DS4
activation-record layout, not DeepSeek V4. Implementing a kernel now would
therefore invent semantics and bypass the model-plugin boundary.

When a checkpoint and model target are explicitly approved, review Nathan's
clean commits and execute in this order:

1. pin checkpoint/config/tokenizer provenance and the authoritative model CPU
   oracle, then register a metadata-only model plugin with exact architecture
   names, layer sequence, weight map, cache roles, and chat/special-token rules;
2. add CPU-reference semantics for lightning indexer, deterministic top-k,
   indexed attention, gather/padding behavior, and hyper-connections;
3. model main compressed KV and mandatory F16 indexer keys as separate cache
   roles/policies; both attention paths still consume complete `KVLiveSpans`;
4. register indexed sparse prefill attention under new layer/quant/variant keys
   with a numerically equivalent dense fallback;
5. add gather-to-compact c1 decode, then a union-gather design for verifier B>1,
   preserving invalid-index, duplicate-index, padding, and dense-prefix rules;
6. register fused HC pre/comb/post only beside a mandatory unfused primitive
   chain and compare both against the model CPU oracle;
7. add explicit small-B contiguous projection buffers only where a copy-inclusive
   B2/B4 trace proves a stride tax;
8. run full invalid-index, padding, dense-prefix, long-context, KL/top-1, and
   named-kernel trace gates before any performance claim.

These must be new model/layer/quant/variant registrations. They must not appear
as `if model == deepseek4` or backend conditionals in generic dispatch.

## No-action execution audit

The 2026-08-04 post-experiment audit finds no rejected item whose premise has
changed:

- **Vulkan P/Psh remains consumer-specific.** hipEngine invokes a precompiled
  AOTriton `attn_fwd`; there is no GLSL P-fragment loop or `Psh` storage to edit.
  The head-major copy already removed the measured stride bottleneck without
  reproducing a perf-neutral Vulkan layout change.
- **Persistent head-major KV is not the next prefill step.** At 64K the
  standalone copy is **2.233 ms**, only **0.708%** of the **315.417-ms** copy-
  inclusive attention sub-window. Even charging one copy to each of ten full-
  attention layers is **22.329 ms**, about **0.032%** of the measured
  **68.815-s** full prefill. That removable cost cannot justify changing every
  writer, decode reader, graph, compactor, and `KVLiveSpans` consumer. Campaign
  SH-A1 permits only a page-internal decode micro-screen after C0; the full
  persistent rewrite stays closed unless that screen passes its whole-decode
  gate.
- **MMID negative experiments remain negative evidence.** Nathan's current scale
  cache was disabled after 4%-20% regressions, `TILE16` increased expert-weight
  re-streaming, and scalar packed-int MMID lost to cooperative F16. hipEngine's
  independently qualified integer/T16 variants stay on their exact registry
  keys; none makes these Vulkan implementations portable or promotable.
- **Generic ubatch values remain superseded.** Current gfx1151 policy keeps the
  same-device-qualified 256-row linear/MoE profile while full-attention chunks
  remain shape/memory selected. Copying llama.cpp's generic 1024/2048 advice
  would overwrite local evidence rather than extend it.
- **RADV packaging remains out of scope.** hipEngine ships HIP, not a Vulkan
  ICD/Mesa stack. The later user-requested exact-payload run supplies a qualified
  external diagnostic, but it creates no reason to bundle RADV or infer that
  Vulkan-specific code is portable to HIP.

No hipEngine implementation is warranted for these items. The external
comparison above measures the fork rather than changing any no-action code
decision.

## Original source-review priority list (completed)

1. **P0 — completed 2026-08-04:** head-contiguous BF16 AOTriton prefill scratch
   is the bounded gfx1151 default after exact copy-inclusive 32K/64K gains of
   **3.383%/7.001%**; see the execution update and artifact above.
2. **P0 — completed/rejected 2026-08-04:** do not enable MES
   `lr_compute_wa`. Upstream removed the incomplete, destabilizing workaround;
   the captured kernel already has the replacement gfx1151 VGPR-size fix active
   and nevertheless reproduced. Keep the qualified layer drain opt-in while the
   upstream issue remains open.
3. **P1 — completed 2026-08-04:** grouped-GQA INT8 decode now has an
   explicit source/launch guard proving `(kv_head, split)` producer ownership
   plus a fresh exact gfx1151 smoke and named trace. Existing H7U/H7U-source
   gates already cover stable expert starts, active lists, lane/source-row
   order, MMQ tile maps, packed hidden, edge cases, and profiler topology; their
   full GPU bundle remains green after refreshing only an orthogonal gfx1151
   package hash.
4. **P1 — completed 2026-08-04:** retain the fast BF16 prefill bridge; current
   policy keeps direct streaming INT8 limited to explicit diagnostics or
   very-long memory-pressure fallback. Fresh route/GPU gates pass, and the GGUF
   layer-local BF16 oracle now has an explicit guard proving it retains the
   bounded gfx1151 head-major AOTriton scratch. Do not promote the measured
   **-97.7%** 128K direct-streaming path merely to remove scratch.
5. **P2 — completed/deferred 2026-08-04:** no DeepSeek V4 plugin, checkpoint,
   CPU oracle, tokenizer fixture, or approved phase target exists. Preserve the
   indexer/sparse/gather/HC commits as references and start with the model/CPU-
   oracle admission unit above only after explicit approval; do not add Qwen or
   Laguna branches now.
6. **Completed/revalidated 2026-08-04 — no action:** Vulkan P/Psh source edits,
   a full persistent head-major KV rewrite, MMID scale cache/TILE16/int-dot
   negatives, generic ubatch values, and RADV packaging remain unsupported. The
   successful scratch A/B strengthens the prefill decision: its 64K copy is only
   **0.708%** of the candidate attention sub-window and at most **0.032%** of
   full prefill across ten full-attention layers. Campaign SH-A1 is only a
   bounded decode-layout screen and makes none of those backend-specific changes
   portable by itself.

## Original campaign completion audit

Every prioritized item has an explicit retained, rejected, or deferred outcome:

| Priority | Incorporated result | Validation and evidence | Commit |
| --- | --- | --- | --- |
| 1 — head-major BF16 AOTriton scratch | Added registry-resolved dense/generic `KVLiveSpans` copies, one bounded tracked cross-layer K/V pair, gfx1151 capability admission, and exact strided/allocation-denial fallbacks. Promoted through rounded capacity 65,792. | Primitive permutations/boundaries and complete-model state are byte-exact; pre-promotion focused gate **39 passed** and post-promotion bundle **61 passed, 15 fixture skips**. The accepted artifact records the cached named-kernel trace and copy-inclusive 512/4K/32K/64K deltas **-0.028%/+0.616%/+3.383%/+7.001%**. Benchmark README, changelog, root README, kernel catalog, and refactor ledger are updated. | `d5e95d1c9` |
| 2 — repeated-128K MES hypothesis | Rejected `lr_compute_wa`; retained only the already-qualified default-off layer drain. Corrected the stall guide, roofline interpretation, refactor condition, and diagnostic metadata. | Upstream chronology and exact running-kernel/KFD accounting prove the replacement 384-KiB/CU VGPR fix was active when the stall reproduced. The focused benchmark provenance/README guard passes **14/14**. No unsafe kernel patch, reboot, or invalid causal benchmark was run. | `b8909a22b` |
| 3 — grouped GQA and MoE structure | Added a direct source/launch guard for `(kv_head, split)` INT8 producer ownership; existing MoE/H7U coverage already freezes stable counts, prefixes, row/lane ordering, active experts, tile maps, and trace topology. | Fresh gfx1151 INT8 NumPy-oracle smoke has max error **<=1.53e-05** and the trace names the expected producer/grid. The combined gate established **25 passing nodes** before one orthogonal stale package hash; focused repair then passed H7U **9/9** and its source/owner guard **3/3**. | `849b680b9` |
| 4 — quantized-KV prefill bridge | Retained BF16-oracle/AOTriton as the normal policy and added a structural guard proving GGUF's layer-local BF16 oracle preserves the bounded head-major consumer. Direct streaming remains diagnostic or memory-pressure fallback only. | Host policy/artifact gate **38 passed**; gfx1151 direct-INT8 plus head-major primitives **14 passed**; runtime dispatch/layout **178 passed**; GGUF full-attention/INT8/head-major bundle **34 passed, 2 fixture skips**. Existing [128K diagnostic](../benchmarks/results/2026-06-15-gpu1-int8-prefill-streaming-throughput-diagnostic.json) rejects direct streaming at **-97.7%**; no redundant 93-minute rerun was warranted without a kernel change. | `a8d64d3b9` |
| 5 — DeepSeek V4 | Deferred at the model-admission boundary and documented a CPU-first plugin checklist; no Qwen/Laguna/backend branch was added. | Registry, repository, plan, local-model, and Hugging Face cache audit finds no approved plugin, checkpoint/config/tokenizer, or CPU oracle. Existing `ds4` quant symbols were verified to mean an activation layout, not the model. | `09a262bd1` |
| 6 — no-action set | Revalidated Vulkan P/Psh, persistent-KV, negative MMID, generic-ubatch, RADV, and dev-release decisions; no rejected backend code was ported. | Live dispatch/profile audit confirms the consumer and local policies differ. Accepted-artifact arithmetic bounds removable persistent-layout copy cost to **0.708%** of the 64K attention sub-window and **0.032%** of full prefill. | `788b87f77` |

The original source review and permalink audit are commits `aef75209c` and
`db87f6d1e`. The final machine-readable closure check reparsed the accepted
artifact, matched all four published deltas and benchmark rollups, found every
new regression guard, confirmed all six execution commits are ancestors of the
current tree, and passed `git diff --check`. Expensive GPU gates were not rerun
after later docs/test-only units because their completed evidence remains valid
under the focused-repair rule. There is no unassigned action left in the
original six-item campaign; future 128K driver work requires a named stack fix,
and DeepSeek V4 requires explicit model admission. The later exact-model fork
comparison does not reopen any of those six source-transfer items; it creates
the separate SH-C0 through SH-G decode/memory campaign above.
