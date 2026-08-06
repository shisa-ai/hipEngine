# Nathanw1014 Strix Halo llama.cpp review for hipEngine gfx1151 GGUF

**Reviewed:** 2026-08-04; **Campaign 2 recertified and closed:** 2026-08-06; **SH7-A1 retained and SH8-A1 activated:** 2026-08-06

**Scope:** `Nathanw1014/strix-halo-llamacpp` releases and evidence pack,
`Nathanw1014/llama.cpp` optimization branches through `strix-halo-vulkan`
`b7b85da9c4a9fdeb3cab51030a40d1552270f272`, and the current hipEngine
Qwen3.6/Laguna GGUF gfx1151 paths.

**Decision type:** source/evidence review followed by a completed prioritized
local campaign and user-requested exact-model fork diagnostics. Nathan's
published speedups remain upstream evidence. hipEngine's retained own-engine
claims are the separately measured head-major prefill scratch, exact selected-Q5
tile8, and exact scratch owner-slot changes linked below. Every local fork row is
descriptive, not a strict cross-engine claim.

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

**Final campaign update (2026-08-06): completed.** hipEngine retains the exact
selected-Q5 tile8 decode owner and 21-slot scratch-liveness allocation in
addition to the earlier head-major prefill scratch. The strict compact-KV and
page-internal head-major decode candidates are rejected after complete quality,
wall, memory, and trace screens. Final current-production decode is
diagnostically **0.840%-1.314%** above every SH-C0 row, and 4K+ tracked peak
remains **1.4086 GiB** lower, but no row reaches the C1 half-time-gap target,
Nathan-fork F16 decode parity, or whole-GTT parity. Fresh prefill beats both
fork KV lanes at 4K/32K/64K and loses at 512. Therefore hipEngine does **not** beat this fork overall on the
matched exact-model diagnostic; Campaign 2 is closed because all declared
owners are decided, not because cross-engine parity was reached.

**Beat-fork continuation update (2026-08-06): active.** SH7-A1 independently
admits the already-registered prepare-plus-coalesced split reducer on gfx1151
from 32K. A fresh package-default one-queue same-source pair moves 32K/64K
decode **46.066 -> 46.785 tok/s (+1.560%)** and **39.441 -> 40.386 tok/s
(+2.394%, -0.593 ms/token)** at byte-identical tracked peak and unchanged
whole-GTT. The primitive is exact versus NumPy; all **1,296/1,296**
category+heldout semantic logits are byte-exact; named prepare/output traces are
scratch-free. Serial reduction remains the default below 32K and under explicit
opt-out. C1, C2, fork-F16 decode, and fork-F16 whole-GTT nevertheless remain
**0/4**, so the objective continues to SH8-A1's structurally new grouped-GQA
producer occupancy screen rather than stopping at this retained unit.

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

### SH-M2 owner-slot execution update (2026-08-06)

SH-M2 keeps the full 4,096-row execution shape and changes only physical scratch
ownership. A route/stage interval map graph-colors mutually exclusive logical
fields into **21 separate allocator-owned slots**. Fields with overlapping
lifetimes never share a slot; diagnostics, non-exact GDN routes, rows below
4,096, and peer backends retain dedicated owners. The separate owners are the
important performance property: a 0.304-GiB contiguous arena and a 0.556-GiB
attention/common split both missed the frozen 4K prefill guard, while owner slots
remove the large-range address/color coupling and still save 1.409 GiB.

| Context | dedicated prefill | owner-slot prefill | delta | dedicated decode | owner-slot decode | delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 1363.985 | **1364.456** | **+0.035%** | **53.500** | 53.487 | **-0.025%** |
| 4K | 1438.478 | **1445.185** | **+0.466%** | **56.094** | 56.071 | **-0.042%** |
| 32K | **1150.315** | 1149.883 | **-0.038%** | **46.490** | 46.435 | **-0.119%** |
| 64K | 932.902 | **939.234** | **+0.679%** | 38.999 | **39.750** | **+1.925%** |

| Context | dedicated tracked | owner-slot tracked | tracked saving | dedicated whole-GTT | owner-slot whole-GTT | GTT saving |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 21.480 | 21.480 | 0.000 GiB | 21.916 | 21.916 | 0.000 GiB |
| 4K | 23.007 | **21.599** | **1.409 GiB** | 23.552 | **22.148** | **1.404 GiB** |
| 32K | 23.654 | **22.245** | **1.409 GiB** | 24.204 | **22.800** | **1.404 GiB** |
| 64K | 24.392 | **22.984** | **1.409 GiB** | 24.943 | **23.538** | **1.404 GiB** |

Full payloads are byte-identical at all four contexts: prefill logits,
hidden/layer/Conv/GDN/live-BF16-KV state, four fixed-input decode transitions,
and final state. Every process returns tracked bytes exactly to baseline. A
separate D->L->L->D 4K confirmation measures **1414.892 -> 1446.750 tok/s
(+2.252%)** prefill, **+0.043%** decode, and the same **1.4086-GiB** tracked
saving, so the retained decision does not depend on one favorable order.

The same-scope whole-GTT gap to fork F16 is now **0.987/0.801/0.667 GiB** at
4K/32K/64K, down from **2.391/2.205/2.071 GiB**, but SH-M2 alone does not beat
the fork memory rows. Strict compact KV therefore proceeds under SH-K1 rather
than relabeling the remaining gap as parity. Complete evidence is in
[`2026-08-06-gfx1151-gguf-sh-m2-owner-slots-retained.json`](../benchmarks/results/2026-08-06-gfx1151-gguf-sh-m2-owner-slots-retained.json).
The committed `7b675670a` checkpoint reproduces **+0.578%** 4K prefill,
**+0.271%** decode, exact state, and both memory deltas; the temporary
disable-only comparison seam is removed afterward.

### SH-K1 strict compact-KV execution update (2026-08-06)

SH-K1 begins from the only unclosed strict-quality map rather than repeating the
failed all-layer, three-layer, block16, key-only, or tail-four format screens.
The candidate keeps full-attention indices `0-7` in BF16 and stores K+V for
indices `8-9` as per-token/per-KV-head INT8 with effective FP32 sideband scales.
The quality child reserves **65,792 tokens**, so a short-session BF16 mirror
cannot hide the actual layout. Its audit finds zero persistent BF16 mirror
bytes, the exact 8/2 partition, and zero surviving layer-local prefill-oracle
buffers.

The complete 10-prompt category/train/heldout corpus plus `mixed_v1` passes:
**11/11 prompts, 187/187 positions, mean KL 3.344e-5, max KL 7.875e-4,
100% aggregate top-1, and 100% minimum-prompt top-1**. Thus the map is
strict-quality-safe on gfx1151. Quality alone is not the promotion rule,
however; long repeated wall and peak memory must also win.

| Context | BF16 prefill | 8/2 prefill | delta | BF16 decode | 8/2 decode | delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32K | 1149.883 | **1150.247** | **+0.032%** | **46.435** | 46.232 | **-0.436%** |
| 64K | **939.234** | 938.695 | **-0.057%** | **39.750** | 39.484 | **-0.670%** |

| Context | BF16 tracked | 8/2 owned live | live saving | 8/2 tracked peak | peak change | BF16 / 8/2 whole-GTT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32K | 22.245 | 22.183 | **0.0620 GiB** | 22.309 | **+0.0640 GiB** | 22.800 / 22.870 GiB |
| 64K | 22.984 | 22.860 | **0.1235 GiB** | 23.111 | **+0.1274 GiB** | 23.538 / 23.671 GiB |

The persistent compact payload does reduce live ownership, but the required
layer-local BF16 prefill-oracle pair is **0.1260/0.2510 GiB** at 32K/64K and
sets a higher allocator high-water mark. Ten-millisecond whole-GTT confirms the
same negative direction at **+0.0703/+0.1328 GiB**. Every process returns
tracked bytes to zero, so this is a real lifetime result rather than a leak.

Cached full-process traces prove the existing direct consumer is active at both
required shapes. The decode-only grouped-GQA INT8 producer runs **10 times**
(two compact layers times one warmup plus four measured transitions), uses
local256, **80 VGPR, 0 LDS, and 0 scratch**, and averages **454.6 us at 32K**
and **802.5 us at 64K**. Its split grid scales from **129 to 257**. The intended
consumer therefore works; its arithmetic plus writer cost does not repay the
BF16 path, and storage savings cannot outrun the prefill-oracle high water.

Close SH-K1 without a production change. Keep BF16 KV plus SH-M2 owner slots as
the gfx1151 default and preserve the existing guarded/explicit hybrid only as a
quality-qualified option, not a speed or peak-memory claim. The production
whole-GTT gap to fork F16 remains **0.801/0.667 GiB** at 32K/64K; selecting the
candidate would worsen it to **0.871/0.800 GiB**. Complete quality, wall,
allocator, trace, primitive, and validation evidence is in
[`2026-08-06-gfx1151-gguf-sh-k1-compact-kv-closed.json`](../benchmarks/results/2026-08-06-gfx1151-gguf-sh-k1-compact-kv-closed.json).
The campaign advances immediately to SH-A1 rather than inventing another
already-closed compact format.

### SH-A1 page-internal head-major decode update (2026-08-06)

SH-A1 implements only the predeclared bounded leaf: current BF16
`[physical_block, token, kv_head, D]` versus page-internal
`[physical_block, kv_head, token, D]`. A transient converter consumes complete
`KVLiveSpans` page, position, and eviction metadata; the candidate grouped-GQA
producer changes only K/V addressing and retains the production BF16 gated
reducer. No writer, compactor, graph, or runtime dispatch is rewritten before
the continuation gate.

Dense, permuted, and stale/evicted-unreferenced physical-page fixtures are
exact: copy mismatches are **0**, candidate-versus-current gated BF16 output
mismatches are **0**, and maximum absolute error versus NumPy is
**1.49e-8**. All four timed contexts are also byte-exact. Correctness therefore
does not explain the performance result.

| Context | Current attention+reducer | Page-head attention+reducer | attention speedup | complete page copy | current append-inclusive | candidate copy+append-inclusive | inclusive speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | **0.0650 ms** | 0.1131 ms | **0.575x** | 0.0149 ms | **0.0669 ms** | 0.1094 ms | **0.612x** |
| 4K | **0.0693 ms** | 0.1196 ms | **0.579x** | 0.0550 ms | **0.0718 ms** | 0.1722 ms | **0.417x** |
| 32K | **0.4563 ms** | 0.6033 ms | **0.756x** | 1.1651 ms | **0.4593 ms** | 1.6606 ms | **0.277x** |
| 64K | **0.8406 ms** | 1.0554 ms | **0.797x** | 2.2347 ms | **0.8421 ms** | 3.1988 ms | **0.263x** |

Thus the candidate misses the required **1.10x** continuation threshold even
before conversion. Charging current append plus complete live-cache conversion
projects a ten-full-attention-layer decode change of **-58.3% at 32K** and
**-97.8% at 64K**, rather than the required >=1% saving. The cached trace names
both producers at identical local256/LDS0/scratch0 geometry; current uses
**72 VGPR**, while page-head rises to **80 VGPR**. The converter is
local256/VGPR16/LDS0/scratch0.

Reject runtime plumbing and remove the converter, producer, wrappers, registry
keys, screen, and test after hashing the transient evidence. Production remains
on token-major BF16 KV plus the current grouped-GQA producer/reducer. The full
persistent head-major rewrite stays closed because its bounded prerequisite
fails by a wide margin. Complete measurements and hashes are in
[`2026-08-06-gfx1151-gguf-sh-a1-page-head-decode-rejected.json`](../benchmarks/results/2026-08-06-gfx1151-gguf-sh-a1-page-head-decode-rejected.json).
SH-G subsequently recertifies that restored production tree and closes the
campaign below.

### SH-G final retained-campaign recertification (2026-08-06)

SH-G ran all final-production stages sequentially on the same Radeon 8060S:
independent right-sized one-warmup/three-measurement hipEngine processes at all
four depths, 10-ms whole-GTT and allocator sampling, cached eight-token
kernel/HIP-API/ROCTX traces, the exact 10+8 natural/category-heldout oracle, and
fresh five-repetition F16/Q8_0 runs of pinned fork build `b7b85da9`.

| Context | Final prefill tok/s | Final eager decode tok/s | Decode vs SH-C0 (diagnostic) | Tracked peak GiB | Whole-GTT GiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 1358.015 | **53.446** | **+1.114%** | 21.480 | 21.916 |
| 4K | 1451.836 | **56.116** | **+1.314%** | 21.599 | 22.148 |
| 32K | 1150.162 | **46.489** | **+1.055%** | 22.245 | 22.800 |
| 64K | 940.779 | **39.750** | **+0.840%** | 22.984 | 23.538 |

The final-versus-SH-C0 decode delta is a cross-day diagnostic, not a new A/B;
the same-revision SH-D1 and SH-M2 artifacts own the retained performance claims.
Measured final decode is **0.208-0.234 ms/token** faster than the SH-C0 rows at
every depth, and the 4K/32K/64K allocator and whole-GTT deltas remain
**-1.4086/-1.4043 GiB**. All repeated IDs are exact, every process closes to
zero tracked bytes, and the 18-prompt, 54-execution oracle passes **1,350 token
comparisons, 54,000 hidden comparisons, and all initial/final Conv/GDN/KV state
comparisons** with zero
mismatches. Cached traces retain **628 dispatches/token**, 4,664 role-attributed
dispatches, and **37 named Q5 tile8 calls/token** at every depth. Against the
August 4 accepted hipEngine prefill rows, these fresh rates are
**-2.635%/-1.392%/-1.831%/-1.215%**; that cross-day drift is diagnostic and
does not replace the same-session non-regression gates used for each retained
promotion.

| Context | Prefill vs fork F16 / Q8_0 | Decode vs fork F16 / Q8_0 | hipEngine minus fork whole-GTT F16 / Q8_0 |
| ---: | ---: | ---: | ---: |
| 512 | **-2.833% / -2.097%** | **-17.281% / -16.690%** | **+1.085 / +1.084 GiB** |
| 4K | **+3.357% / +4.515%** | **-10.297% / -10.948%** | **+0.987 / +0.895 GiB** |
| 32K | **+4.318% / +4.366%** | **-12.670% / -18.903%** | **+0.804 / +1.170 GiB** |
| 64K | **+6.631% / +7.522%** | **-12.995% / -24.035%** | **+0.696 / +1.376 GiB** |

Thus final hipEngine wins three of four fresh prefill rows against each fork KV
lane, but zero of four decode and zero of four whole-GTT rows. No C1 or C2 row
passes. These remain external diagnostics: exact weights, hardware, split
shapes, and GTT sampling align, while KV dtype, timing owner, and a shared
cross-engine token/logit oracle do not. Complete commands, samples, role
resources, raw hashes, and qualifications are in
[`2026-08-06-gfx1151-gguf-sh-g-final-recertification.json`](../benchmarks/results/2026-08-06-gfx1151-gguf-sh-g-final-recertification.json).

## Campaign 3: post-SH-G beat-fork continuation

SH-G remains closed: every package declared in Campaign 2 was decided and its
final production matrix is immutable evidence. The higher-level thread objective
is different, however: **beat the pinned fork**, not merely finish those six
packages. Because SH-G measured zero decode and zero whole-GTT wins, that
objective remains active. This continuation does not reopen SH graph, compact-KV,
page-head, or individual byte-neutral schedule ladders; it selects only
structurally new ownership with enough measured Amdahl.

### Residual gap and admitted decode owner

| Context | hipEngine / fork F16 ms/token | F16 gap | gap to C1 | whole-GTT gap |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 18.711 / 15.477 | **3.233 ms** | **1.518 ms** | **1.085 GiB** |
| 4K | 17.820 / 15.985 | **1.835 ms** | **0.812 ms** | **0.987 GiB** |
| 32K | 21.510 / 18.785 | **2.725 ms** | **1.247 ms** | **0.804 GiB** |
| 64K | 25.157 / 21.888 | **3.269 ms** | **1.583 ms** | **0.696 GiB** |

The only unclosed exact decode family with sufficient scope is a new **mixed-T16
selected-MoE producer/consumer composite**. Final selected Q4 gate/up, selected
down, and router/combine own about **4.689 ms/token** at 512. Closing C1 from
that owner alone requires approximately **1.479x/1.209x/1.362x/1.510x** at
512/4K/32K/64K. This is P10.D1: retain Q4T16 gate/up and Q5T16/Q6T16 down bytes,
publish the exactly rounded SiLU intermediate, then let an output-tiled stage
replay the current down and route-weight reduction orders. It is not the existing
raw-Q4 megakernel: that body accepts neither the deployed mixed quant/layout nor
enough c1 blocks.

A leaf must preserve the unfused registered fallback, match its complete BF16
outputs, run under the expected cached symbol without spills, and reach at least
**1.15x** or project **0.5 ms/token** before full-model routing. A retained exact
sub-window win is still kept under project policy even if it does not alone reach
C2; cumulative attribution then selects another structurally new owner rather
than ending the objective.

### Residual memory arithmetic

The final allocation census identifies `root.token_embedding` as the complete
**540,344,320-byte / 0.503 GiB raw-Q8** family. Ideal removal would still leave
hipEngine **0.582/0.484/0.301/0.193 GiB** above fork F16 whole-GTT at
512/4K/32K/64K, so embedding offload alone cannot claim parity. The existing
host-copy diagnostic is exact but disables device-token-fed graphs; its old
11.1-tok/s gfx1100 128K result is a blocker, not current gfx1151 evidence. The
current machine also has an **8 MiB soft and hard memlock limit**, so registering
the full 0.503-GiB table as mapped host memory is not admitted and this campaign
will not change system limits silently.

The legitimate same-file stack is therefore:

1. re-screen the existing exact host-copy path on current gfx1151 against both
   the admitted graph and eager c1 owners, with exact state and same-scope GTT;
2. attribute and eliminate overlap among active, prefill-only, decode-only, and
   never-used loaded code objects, but continue only if at least **0.49 GiB** is
   measurably phase-exclusive at 512/4K; and
3. separately extend SH-M2's interval coloring to the 768-row short owner. The
   4,096-row alias map projects **157,138,758 bytes / 0.146 GiB** at 768 rows,
   but that projection is not a result and must pass its own exact/performance
   gate.

No host embedding route is promoted if prefill or decode loses more than 1%, if
production graph-class behavior regresses, or if c>N/unsupported sessions lose
the device-resident fallback. No library unloading/lazy-load change proceeds
without cached-build, module-lifecycle, complete prefill/decode, and teardown
evidence.

### Ordered continuation packages

| ID | Experiment | Continuation / stop rule |
| --- | --- | --- |
| **SH2-C0** | Post-SH-G role, allocation, graph, and closed-frontier audit. | **Complete.** The table above and the compact audit artifact freeze the residual owners; no production change. |
| **SH2-M1** | Current-gfx1151 exact c1 host-embedding matrix: device graph, device eager, existing host-copy eager, tracked and whole-GTT. | **Complete: retain the existing exact c1 opt-in; global default blocked.** Complete state is byte exact, live ownership falls 0.503 GiB, whole-GTT falls 0.443/0.504 GiB at 512/4K, and all prefill/decode deltas are inside 1%. The current load path and shared c>N/packed/MTP device-pointer contracts block default promotion. Full-table mapped work remains blocked by the 8-MiB hard limit. |
| **SH2-M2** | Phase-resolved code-object/library GTT census, then bounded lazy/deferred ownership if sufficient. | **Complete: precondition failed; no implementation.** At 512, the complete untracked process-residency upper bound is only 0.418588 GiB on the device route and 0.478414 GiB on the host stack. Any active/prefill/decode/never-used code-object subset is smaller than the complete set, so neither can reach the frozen 0.49-GiB two-context gate. No lazy loader or `dlclose` path is added. |
| **SH2-M3** | 768-row owner-slot map with dedicated fallback. | **Complete: retained/default.** Reuse the existing 21 independent owner slots from 768 rows: physical scratch falls 355,182,664 -> 69,790,760 bytes, tracked/whole-GTT fall 0.265792/0.267578 GiB, prefill/decode improve 0.621%/0.160%, complete state is byte exact, and committed teardown is clean. Diagnostics/capability denial retain dedicated owners; no arena coupling. |
| **SH2-D1** | P10.D1 mixed Q4T16 -> Q5/Q6T16 output-tiled selected-MoE composite. | **Complete: exact, below admission; no production change.** Down+tail reaches only **1.053x / 0.080 ms/token** aggregate. Full cooperative and lifecycle-correct standard-queue composites regress to **0.520x/0.739x**; the cooperative trace also crashes in ROCr. Named standard traces are scratch-free. All transient surfaces are removed; no model routing. |
| **SH2-C1** | Cumulative role/memory re-attribution after each retained unit. | **Checkpoint complete: continue.** Fresh clean 512 is **53.332 tok/s, 21.214 GiB tracked, 21.648 GiB whole-GTT**; unchanged 4K+ rows carry from SH-G. C1/C2/fork-F16 decode/fork-F16 GTT each remain **0/4**. Select the new 0.457031-GiB compact-T16 metadata owner. |
| **SH2-M4** | Compact selected-expert Q4/Q5 T16 scale/min metadata. | **Complete: retain/default Q5 subset; reject Q4 production route.** Compact Q5 removes exactly **155,189,248 bytes / 0.144531 GiB** with 512 prefill/decode changes **-0.426%/-0.459%**, whole-GTT **-0.144363 GiB**, scratch-free named kernels, and byte-exact four-depth state. Full Q4+Q5 projects **+1.598%** decode, so all 80 Q4 tensors remain current T16. |
| **SH2-C2** | Post-M4 cumulative re-attribution. | **Checkpoint complete: continue to SH2-G.** Fresh 512/4K/32K/64K decode is **53.374/55.851/46.315/39.673 tok/s** and whole-GTT is **21.504/22.003/22.656/23.394 GiB**. Compact Q5 saves 0.144 GiB at every depth, but C1/C2/fork-F16 decode/fork-F16 whole-GTT remain **0/4**. Freeze the complete shared-expert composite and runner-safe host embedding as post-milestone owners. |
| **SH2-G** | Fresh four-depth hipEngine plus pinned-fork recertification. | **Complete: milestone passes, objective continues to SH3.** All four prefill guards and exact correctness/lifecycle/trace gates pass, but C1/C2/fork-F16 decode/fork-F16 whole-GTT remain **0/4**. Fresh F16 decode is **64.411/62.590/53.042/45.818 tok/s** versus hipEngine **53.319/55.895/46.353/39.644**, and F16 whole-GTT remains **0.673/0.843/0.660/0.558 GiB** lower. |
| **SH3-D1** | Complete Q8T16 shared-expert gate/up -> exact BF16 SiLU -> down -> residual producer/consumer chain. | **Complete: exact, below admission; no production change.** RED/GREEN and CPU-oracle gates pass with byte-identical gate/up, intermediate, shared-down, and final BF16 outputs. The best actual-weight 128-block candidate is **0.899x** wall and **0.929x** kernel-only; 40/64/80-block siblings regress further. The named trace is **72 VGPR, 512 B LDS, 0 scratch**. All transient surfaces are removed. |
| **SH3-M1** | Runner-safe exact 540,344,320-byte host embedding policy. | **Complete: retained/default for private gfx1151 c1.** Loader-time deferral removes the allocate-then-free high water and saves exactly **0.503235 GiB tracked** plus **0.503933 GiB whole-GTT** at 512/4K. Prefill/decode changes are **+0.488%/-0.035%** and **-0.229%/-0.403%**; complete state is byte exact. Shared/c>N stay resident; graph, packed, MTP, native-row, and device-pointer paths restore the table transactionally once, with allocation-denial rollback. |
| **SH3-C1** | Post-SH3 cumulative four-depth re-attribution and beat-fork policy gate. | **Complete: policy fails; objective continues.** Fresh canonical decode is **53.177/55.664/46.241/39.602 tok/s** and whole-GTT is **21.000/21.499/22.152/22.890 GiB**. All four prefill, exact-oracle, lifecycle, and trace gates pass, but C1/C2/fork-F16 decode/fork-F16 whole-GTT remain **0/4**. |
| **SH4-D1** | Exact gfx1151 private-c1 routed/shared MoE branch overlap. | **Complete: exact, rejected; no production change.** A real second hardware queue overlaps **0.456 ms/token / 33.4%** of the **1.363-ms/token** auxiliary branch, but five same-resident pairs regress **18.811 -> 19.207 ms/token (-2.058%)**. All transient surfaces are removed. |
| **SH5-D1** | Byte-neutral row-1 dense-Q8 replacement-layout/vector algorithm. | **Complete: decode-positive, blocked from production.** The fork-attributed raw local64 leaf reaches **1.1557x** and the byte-neutral model route improves 512 decode **+2.934% / -0.539 ms/token**, but raw prefill loses **13.457%** and changed reduction is not byte-exact. The exact-tree repair is **0.779x**; production stays Q8T16. |
| **SH6-P1** | Phase-exclusive raw-to-T16 prefill bridge plus complete quality gate. | **Complete: exact bridge, rejected; all model routing removed.** The converter is host-packer-byte-exact, scratch-free, and lifecycle-correct with one 25.5-MiB owner; complete 512 prefill state is exact. Charged prefill regresses **1369.120 -> 1318.196 tok/s (-3.720%)**, failing the first 1% guard, so 4K-64K/quality continuation stops and production remains Q8T16. |
| **SH6-C1** | Post-SH6 cumulative four-depth re-attribution and fork policy gate. | **Complete: policy fails; objective continues.** Canonical decode is **53.153/55.832/46.196/39.579 tok/s** and whole-GTT is **21.000/21.499/22.152/22.890 GiB**. All four prefill/exact-oracle/lifecycle/trace gates pass, but C1/C2/fork-F16 decode/fork-F16 whole-GTT remain **0/4**. |
| **SH7-A1** | Independently transfer the registered prepare-plus-coalesced parallel split-K reducer to gfx1151 long-context decode. | **Complete: retained/default from 32K.** One-queue wall improves **+1.560%/+2.394% (-0.333/-0.593 ms/token)** at 32K/64K; the reducer falls **424.162 -> 109.346** and **744.973 -> 207.485 us/token**. Primitive and 1,296-logit semantic gates are exact, memory/lifecycle are unchanged, traces are named and scratch-free, and serial fallback remains below 32K/under opt-out. |
| **SH8-A1** | Split the 72-VGPR grouped-GQA producer's eight query heads into exact four-head ownership groups to test register occupancy against duplicated K/V traffic. | **Active.** The post-SH7 producer owns **4.037/7.212 ms/token** at 32K/64K. First require exact per-head reduction order and `KVLiveSpans` coverage, then an actual-shape cached leaf A/B. Continue to model wall only at >=1.10x producer speedup or >=0.5-ms/token projected saving; otherwise remove the transient sibling and close this occupancy tradeoff. |

SH2-M1 then overturns the old gfx1100 throughput extrapolation without weakening
its scope warning. On current gfx1151, device graph/device eager/host-copy eager
decode is **53.301/53.521/53.378 tok/s** at 512 and
**55.813/56.054/55.972 tok/s** at 4K. Host-copy prefill changes
**+0.085%/+0.176%** versus device eager and decode changes
**-0.267%/-0.147%**; it is also non-regressive versus the measured graph owner.
The complete 512/4K prefill and four-transition hidden/layer/Conv/GDN/live-KV
fingerprints are byte exact. Live ownership drops exactly **0.503235 GiB** and
whole-GTT drops **0.443409/0.503933 GiB**. The shorter row exposes the remaining
load-order debt: the device table is materialized before it is freed, so tracked
high water drops only **0.418797 GiB** there. Keep the exact c1 env route, but do
not make it global default until a shared-runner-safe policy preserves
multi-row, packed-AR, and MTP device-pointer fallback. Evidence:
[`2026-08-06-gfx1151-gguf-sh2-m1-host-embedding-screen.json`](../benchmarks/results/2026-08-06-gfx1151-gguf-sh2-m1-host-embedding-screen.json).

SH2-M2 closes before unsafe module-lifecycle work. Subtracting each 10-ms GTT
baseline and live owned bytes from the fresh SH2-M1 process peak bounds **all**
untracked code-object, runtime-library, driver, page-table, and other process
residency at **0.418588/0.531532 GiB** for device eager and
**0.478414/0.530792 GiB** for host-copy at 512/4K. Because a phase-exclusive
code/library subset cannot exceed the complete residual, both 512 routes miss
the required **0.49 GiB** even under the impossible assumption that every
untracked byte is removable. No `dlclose`, lazy loader, or extra benchmark can
repair that precondition; proceed directly to SH2-M3. Evidence:
[`2026-08-06-gfx1151-gguf-sh2-m2-code-residency-closed.json`](../benchmarks/results/2026-08-06-gfx1151-gguf-sh2-m2-code-residency-closed.json).

SH2-M3 finds substantially more short-row headroom than the launch projection:
the unchanged 21 independent owner slots reduce 768-row physical scratch
**355,182,664 -> 69,790,760 bytes**. A same-source D->L->L->D screen moves
512/128 prefill/decode **1361.744/53.322 -> 1370.204/53.408 tok/s**
(**+0.621%/+0.160%**), tracked peak **21.479979 -> 21.214187 GiB**, and
whole-GTT **21.916004 -> 21.648426 GiB**. Complete logits, hidden/layer,
Conv/GDN, live-KV, and four-transition state are byte exact; commit `edb151447`
reproduces the 69,790,760-byte owner, memory peaks, IDs, and clean teardown.
F32/capture diagnostics and capability denial retain dedicated allocation. This
is not a single/split arena and adds no runtime comparison flag. Evidence:
[`2026-08-06-gfx1151-gguf-sh2-m3-short-owner-slots-retained.json`](../benchmarks/results/2026-08-06-gfx1151-gguf-sh2-m3-short-owner-slots-retained.json).

SH2-D1 closes the newly admitted mixed-T16 ownership without weakening its
frozen gate. The exact output-tiled down+tail leaf moves the deployed 37-Q5/
3-Q6 aggregate **1580.242 -> 1500.048 us/token (1.053x, 0.080 ms saved)**:
Q5 is only **1.040x**, while the three Q6 layers reach **1.222x** but contribute
too little total time. The complete one-launch cooperative producer/consumer
then regresses to **0.520x** and cannot produce the required cached trace because
`rocprofv3` crashes in ROCr's signal path. Replacing the cooperative barrier
with lifecycle-correct standard task queues is exact and profileable at
**104/112 VGPR, 0 scratch**, but still regresses the deployed aggregate
**3927.785 -> 5317.336 us/token (0.739x, -1.390 ms)**. No leaf clears `1.15x`
or `0.5 ms/token`; nothing reaches model routing. Remove every transient kernel,
wrapper, registry key, test, and harness, then proceed to SH2-C1 rather than
ending the beat-fork objective. Evidence:
[`2026-08-06-gfx1151-gguf-sh2-d1-mixed-t16-composite-closed.json`](../benchmarks/results/2026-08-06-gfx1151-gguf-sh2-d1-mixed-t16-composite-closed.json).

SH2-C1 reproduces the current short row on clean commit `b8989cbb4`:
**53.332 tok/s**, **21.214187 GiB tracked**, **21.648426 GiB whole-GTT**, exact
IDs, 628 profiled dispatches/token, and the retained **69,790,760-byte** scratch
owner. Because SH2-M3 changes only the 768-row class and SH2-D1 changes no
production code, SH-G's 4K/32K/64K rows remain mechanically current. The four
rows still pass **0/4 C1, 0/4 C2, 0/4 fork-F16 decode, and 0/4 fork-F16
whole-GTT**; F16 whole-GTT gaps are **0.818/0.987/0.804/0.696 GiB**.

The next owner is not another closed graph/KV/page-head/tile-width retry. All 80
Q4 selected gate/up T16 tensors expand packed scale/min metadata by exactly
**335,544,320 bytes**, and the 37 Q5 selected-down tensors add **155,189,248**;
Q6 adds none. A compact bit-lossless T16 metadata variant therefore targets
**490,733,568 bytes / 0.457031 GiB**. Stacking that *unimplemented projection*
with the independently measured host-embedding deltas would project F16 GTT
gaps of **-0.083/+0.026 GiB** at 512/4K; 32K/64K remain ideal-byte projections,
not evidence. SH2-M4 must earn pack/unpack, CPU/GPU byte, prefill/decode,
trace, four-depth state/lifecycle, and whole-GTT gates before any default change.
Evidence:
[`2026-08-06-gfx1151-gguf-sh2-c1-cumulative-reattribution.json`](../benchmarks/results/2026-08-06-gfx1151-gguf-sh2-c1-cumulative-reattribution.json).

SH2-M4 closes the 0.457031-GiB owner by retaining only its separable Q5 subset.
Both compact layouts preserve FP16 `d/dmin` and quant planes while encoding each
four-column scale/min quartet in 24 bits. Host roundtrip/dequant and direct/WMMA
GPU oracles are exact. On the actual all-256-expert leaf, full Q4+Q5 compaction
projects **+0.299674 ms/token / +1.598%**, above the frozen 1% gate; two bounded
unpack schedules worsen that to **+2.494%/+3.239%**. Q4 therefore remains
`gguf_q4_k_t16_v1` in production.

The 37 Q5 tensors independently remove exactly **155,189,248 bytes /
0.14453125 GiB**. Same-protocol 512 prefill/decode moves
**1372.347/53.383 -> 1366.497/53.138 tok/s (-0.426%/-0.459%)**. Fresh tracked
and whole-GTT fall **21.214187/21.648426 -> 21.069656/21.504063 GiB**, all
measured IDs are exact, and teardown returns tracked ownership to zero. Clean
parent-vs-candidate state children at 512/4K/32K/64K match FP32 logits,
hidden/layer/Conv/GDN/live-BF16-KV state, four-transition trajectories, and
final state byte-for-byte. Production tracing names 333 compact Q5 decode
launches at **56 VGPR, 512 B LDS, 0 scratch** and 37 compact WMMA prefill
launches at **72 VGPR, 0 LDS/scratch**. Promote Q5 compact metadata by default,
retain current Q5/Q4 T16 registry fallbacks, and proceed immediately to SH2-C2;
this memory win does not end the beat-fork objective. Evidence:
[`2026-08-06-gfx1151-gguf-sh2-m4-compact-q5-t16-retained.json`](../benchmarks/results/2026-08-06-gfx1151-gguf-sh2-m4-compact-q5-t16-retained.json).

SH2-C2 freshly measures committed `e39aba0e1` at all four depths rather than
carrying long rows. Decode is **53.374/55.851/46.315/39.673 tok/s**, tracked
peak is **21.070/21.454/22.100/22.839 GiB**, and whole-GTT is
**21.504/22.003/22.656/23.394 GiB** at 512/4K/32K/64K. Every repeated token is
exact, tracked ownership returns to zero, each trace has 628 dispatches/token,
and compact Q5 is named at 37 calls/token with **0 scratch**. The exact
0.144531-GiB owner saving therefore reproduces at every depth.

The frozen comparator still yields **0/4 C1, 0/4 C2, 0/4 fork-F16 decode, and
0/4 fork-F16 whole-GTT**. F16 decode time gaps are
**3.259/1.919/2.806/3.318 ms/token** and whole-GTT gaps are
**0.673/0.843/0.660/0.552 GiB**. Independently stacking SH2-M1's measured host
embedding delta would still be projection, not parity. SH2-G must now freshly
rerun the pinned fork. If that milestone remains below policy, the next
structurally new owners are a complete Q8T16 shared-expert gate/up + exact-BF16
SiLU + down + residual composite (**1.061-1.075 ms/token** scope) and a
runner-safe exact **540,344,320-byte** host-embedding policy that preserves
c>N, packed AR, MTP, and device-pointer fallbacks. Evidence:
[`2026-08-06-gfx1151-gguf-sh2-c2-cumulative-reattribution.json`](../benchmarks/results/2026-08-06-gfx1151-gguf-sh2-c2-cumulative-reattribution.json).

SH2-G then recertifies the exact current stack against fresh pinned fork build
`b7b85da9` rows. Independent hipEngine prefill/decode is
**1368.737/53.319**, **1445.181/55.895**, **1151.255/46.353**, and
**938.924/39.644 tok/s** at 512/4K/32K/64K. Pinned-fork F16 decode is
**64.411/62.590/53.042/45.818 tok/s** and Q8_0 is
**64.179/63.087/57.379/52.167**. hipEngine protects all four prior-production
prefill guards and beats fresh fork prefill at 4K/32K/64K, but reaches **0/4
C1, 0/4 C2, 0/4 F16 decode, and 0/4 F16 whole-GTT parity**. F16 decode deficits
are **10.767%-17.135%**, and hipEngine whole-GTT remains
**0.673/0.843/0.660/0.558 GiB** higher.

The exact 18-prompt category/heldout oracle passes 1,350 token and 54,000 hidden
comparisons plus all initial/final Conv/GDN/KV state checks, repeated rows are
exact, tracked ownership closes to zero, and compact-Q5 traces remain valid.
Thus SH2-G is complete, but the declared beat-fork policy and thread objective
are not. Proceed immediately to SH3-D1, then SH3-M1 and SH3-C1. Evidence:
[`2026-08-06-gfx1151-gguf-sh2-g-fork-parity-recertification.json`](../benchmarks/results/2026-08-06-gfx1151-gguf-sh2-g-fork-parity-recertification.json).

SH3-D1 implements the complete decode-only Q8T16 shared-expert chain under its
frozen first gate, without routing it into the model. A RED CPU/unfused oracle
becomes GREEN at **2/2**, and the cooperative K-block scale-LDS candidate is
byte-identical at gate/up, exact-BF16 SiLU intermediate, shared-down, and final
routed/shared-gate/residual output. On actual layer-0 `2048 -> 512 -> 2048`
weights with a 24-copy **80,216,064-byte** cycling pool, five counterbalanced
2,000-iteration repeats measure **34.719 us** for the four-kernel fallback and
**38.611 us** for the best 128-block composite (**0.899x**, projected
**-0.156 ms/token** over 40 layers). The bounded 40/64/80-block siblings reach
only **0.444x/0.754x/0.749x**. Cached tracing names the candidate at median
**25.287 us, 72 VGPR, 512 B LDS, 0 scratch**, versus **23.482 us** summed
baseline kernels (**0.929x**); rocprof writes the complete CSV before a recorded
ROCr signal-path teardown fault. The separate exactness and wall screens are
clean. Neither `1.15x` nor `0.5 ms/token` passes, so all transient kernel,
wrapper, registry, test, and harness surfaces are removed. Production and the
registered unfused chain remain unchanged; continue immediately to SH3-M1.
Evidence:
[`2026-08-06-gfx1151-gguf-sh3-d1-shared-expert-composite-rejected.json`](../benchmarks/results/2026-08-06-gfx1151-gguf-sh3-d1-shared-expert-composite-rejected.json).

SH3-M1 promotes the exact host embedding only through a backend capability and
only for a private `max_batch_size=1` session. The loader keeps
`root.token_embedding` as a validated allocation-free Q8_0 spec and maps the
GGUF bytes directly, so the admitted path never creates the 540,344,320-byte
device table. Indexed/cached Q8_0 row dequantization preserves the existing BF16
hidden boundary. Shared and c>N sessions remain device resident; graph capture,
packed AR, native rows, MTP, non-default streams, and device token pointers call
a lock-protected one-shot materializer. The runner publishes the new weight map
only after upload succeeds, so allocation denial leaves the host route usable.

On the same frozen source, 512/128 device -> host-auto prefill/decode moves
**1368.003/53.263 -> 1374.684/53.245 tok/s (+0.488%/-0.035%)** and 4K/128 moves
**1435.036/55.985 -> 1431.754/55.759 (-0.229%/-0.403%)**. Tracked peak drops
**21.069656 -> 20.566421 GiB** and **21.454266 -> 20.951031 GiB**, exactly
**540,344,320 bytes / 0.503235 GiB** at each depth. Independent 10-ms
whole-GTT drops **21.504063 -> 21.000130 GiB** and
**22.003361 -> 21.499428 GiB**, both **0.503933 GiB**. All repeated IDs,
prefill FP32 logits, 40 layer outputs, 30 Conv/GDN state pairs, 10 live BF16-KV
pairs, four-transition trajectories/final state, and teardown checks are exact.
The focused policy/materializer/graph/packed/MTP/backend bundle passes **62/62**.
Retain/default the private-c1 policy and advance immediately to SH3-C1; this
memory win is not campaign completion. Evidence:
[`2026-08-06-gfx1151-gguf-sh3-m1-runner-safe-host-embedding-retained.json`](../benchmarks/results/2026-08-06-gfx1151-gguf-sh3-m1-runner-safe-host-embedding-retained.json).

SH3-C1 freshly reruns every current-production dimension at committed
`16b961a6b`. Canonical 512/4K/32K/64K prefill is
**1368.743/1436.083/1148.130/939.441 tok/s**, all within 1% of SH2-G, while
decode is **53.177/55.664/46.241/39.602 tok/s**. Independent right-sized
attribution children measure tracked peak **20.566/20.951/21.597/22.336 GiB**
and 10-ms whole-GTT **21.000/21.499/22.152/22.890 GiB**, with exact IDs and
zero tracked bytes after every close. Against pinned fork F16, whole-GTT gaps
shrink to **0.169/0.339/0.156/0.054 GiB**, but decode time gaps remain
**3.280/1.988/2.773/3.426 ms/token**. Thus C1, C2, fork-F16 decode, and
fork-F16 whole-GTT all remain **0/4**.

The fresh category/heldout gate passes 18/18 prompts, 1,350 token comparisons,
54,000 hidden comparisons, initial/final Conv/GDN/KV state, and deterministic
repeats with zero mismatch. Every trace has 628 dispatches/token and the compact
Q5 owner remains named 37 times/token. SH3-C1 is complete, but policy and the
thread objective fail. Select SH4-D1 rather than reopening a closed kernel,
graph, KV, page-head, or metadata ladder: after router completion, overlap the
current exact **1.066-ms/token shared branch** with the independent
**3.754-ms/token selected branch** only for private gfx1151 c1 eager decode,
then join before the existing combine. Require exact state/fallbacks, a cached
cross-queue trace, and at least 1% wall or 0.5 ms/token saving. Evidence:
[`2026-08-06-gfx1151-gguf-sh3-c1-cumulative-reattribution.json`](../benchmarks/results/2026-08-06-gfx1151-gguf-sh3-c1-cumulative-reattribution.json).

SH4-D1 implements the frozen fork/join only as a default-off, private-gfx1151-c1
ordinary eager screen. Router completion records an event on the caller stream;
the complete shared gate/up, exact-BF16 SiLU, and shared-down chain uses a
nonblocking auxiliary stream plus a dedicated 2-KiB concat row; the caller keeps
selected experts and waits before the unchanged combine. Shared/c>N, graph,
prefill, packed AR, MTP, diagnostics, nondefault streams, unsupported backends,
and resource denial remain serial. Fake-HIP policy/lifecycle/order contracts pass
**28/28**, and four real-HIP teacher-forced transitions plus the first 128-step
same-resident pair match final hidden, every Conv/GDN state, every live BF16-KV
row, and token trajectory byte-for-byte.

The cached eight-token trace records **4,064** caller dispatches on queue 1 /
stream 0 and **960** shared dispatches on queue 2 / stream 1, preserving the
production **628 dispatches/token**. Timestamp intersection proves **189**
auxiliary kernels and **0.455550 ms/token** overlap selected gate/up, or
**33.420%** of the auxiliary branch's **1.363092 ms/token**. Nevertheless five
counterbalanced same-resident 512/128 pairs move median serial/candidate wall
**18.811444 -> 19.206655 ms/token**, or **53.159 -> 52.065 tok/s
(-2.058%)**. Thus real overlap is not a performance win: queue/event costs and
bandwidth contention exceed the hidden work. Reject and remove every transient
surface.

Do not retry this stream topology or reinterpret the trace as a retained
sub-window win. Activate SH5-D1 instead: the current trace still assigns about
**8.2 ms/token** to dense Q8T16 projections. Unlike the closed thread/cache/
adjacent-T16 schedules and slow existing q8_1-dp4a diagnostic, SH5-D1 must first
establish a genuinely different byte-neutral row-1 vector/layout algorithm on
the actual `attn_qkv[2048->8192] + attn_gate[2048->4096]` owner. It receives no
model route without CPU-oracle correctness, cached no-spill evidence, no
resident-weight duplication, and >=1.15x or >=0.5-ms/token projected saving.
Evidence:
[`2026-08-06-gfx1151-gguf-sh4-d1-moe-branch-overlap-rejected.json`](../benchmarks/results/2026-08-06-gfx1151-gguf-sh4-d1-moe-branch-overlap-rejected.json).

SH5-D1 then mechanically follows the pinned fork's measured F16 decode dispatch:
`dst->ne[1] == 1` enters `ggml_vk_mul_mat_vec_q_f16`, and Q8_0 selects
`mul_mat_vec_q8_0_f16_f32`. The shader reads raw 34-byte Q8_0 rows, maps one
workgroup to one output, consumes eight adjacent K values per lane, and reduces
one scalar. hipEngine already had a raw local256 scalar-row kernel and a raw
local128 eight-output/eight-K pack kernel, so the admitted local64 one-output/
eight-K pair leaf is not a duplicate T16 schedule.

On actual layer-0 `attn_qkv[2048->8192]` plus `attn_gate[2048->4096]` bytes, a
three-copy **80,216,064-byte** cycling pool and 15 counterbalanced HIP-event
samples measure production T16 **0.134737 ms** and raw local64 **0.116588 ms**,
or **1.15566x**, with 15/15 wins and at most one BF16 code of difference. Cached
tracing names the local64 kernel at **24 VGPR, 512 B LDS, 0 scratch**. The
materializer replaces exactly 60 T16 residents / **802,160,640 bytes** with one
raw allocation each; it creates no sidecars and changes no other Q8 owner.

The binding 512/128 route improves decode **52.876 -> 54.427 tok/s (+2.934%,
18.912 -> 18.373 ms/token)** at unchanged **20.566421-GiB** tracked peak, but
raw WMMA prefill falls **1369.120 -> 1184.884 tok/s (-13.457%)**. Prefill logits
and complete state are byte-exact. Four fixed decode transitions retain top-1
9707 but hidden, logits, and recurrent/KV state diverge because local64 changes
the FP32 reduction tree. The bounded exact-tree local64 repair restores T16
bytes but regresses to **0.7786x**; the fast local128 point reaches only
**1.1458x** and still differs by two BF16 codes. Production remains Q8T16.

SH6-P1 then implements the bounded repair exactly. The registered byte-only
transform reproduces host-packed Q8T16 for both production tensors at
local64/128/256; local64 measures **0.360914 ms/pair** and traces at **40 VGPR,
128 SGPR, 0 scratch**. Runtime owns one **26,738,688-byte** buffer, creates no
persistent duplicate, falls back transactionally on allocation denial, and
returns tracked ownership to zero. Prefill logits and complete 512 state are
byte exact.

The binding one-warmup/three-run 512/128 gate nevertheless moves baseline to
candidate prefill **1369.120 -> 1318.196 tok/s (-3.720%)** while decode moves
**52.876 -> 54.488 tok/s (+3.047%)**. The frozen conjunction requires every
prefill depth within 1%, so 512 failure makes the 4K-64K and complete category
continuations non-admissible. Remove the SH5 materializer, dispatcher, env, and
model route plus SH6 runtime bridge and scratch owner. Retain only the standalone
rowvec8 and transform leaves as diagnostic/source evidence; production remains
Q8T16. This activated clean-production SH6-C1 rather than campaign closure.
Evidence:
[`SH5-D1`](../benchmarks/results/2026-08-06-gfx1151-gguf-sh5-d1-raw-rowvec8-blocked.json),
[`SH6-P1`](../benchmarks/results/2026-08-06-gfx1151-gguf-sh6-p1-raw-to-t16-prefill-bridge-rejected.json).

SH6-C1 then measures restored production source at clean commit `b074e8054`.
The process inherited a two-queue cap from SH4, but every profiled decode
kernel used only queue 1 / stream 0; this remains a qualified diagnostic rather
than a retained performance claim. Canonical 512/4K/32K/64K prefill/decode is
**1373.558/53.153**,
**1446.862/55.832**, **1149.718/46.196**, and **938.363/39.579 tok/s**;
independent attribution whole-GTT is **21.000/21.499/22.152/22.890 GiB**.
Every prefill row remains within 1% of SH3-C1, all attribution children close
to zero tracked bytes, cached traces retain 628 dispatches/token, and the fresh
18-prompt oracle passes 1,350 token plus 54,000 hidden comparisons and all
initial/final Conv/GDN/KV state checks exactly. C1, C2, fork-F16 decode, and
fork-F16 whole-GTT nevertheless remain **0/4**.

SH7-A1 then admits that existing reducer transfer. On a fresh package-default
one-queue serial/candidate pair, 32K moves **21.7080 -> 21.3746 ms/token
(+1.560%, -0.3334 ms)** and 64K moves **25.3541 -> 24.7614 ms/token
(+2.394%, -0.5928 ms)**. Tracked peak is byte-identical at
**21.5973/22.3358 GiB**, 10-ms whole-GTT is unchanged at
**22.1518/22.8901 GiB**, and every process returns tracked ownership to zero.

The cached trace replaces serial reduction **424.162 -> 109.346 us/token** at
32K and **744.973 -> 207.485 us/token** at 64K. Its two named kernels use
prepare/output **24/16 VGPR**, **1 KiB LDS**, and **zero scratch** on queue 1 /
stream 0. At 8,448 tokens / 33 splits the candidate has zero BF16 mismatches
versus NumPy; the 18-prompt, three-repeat semantic gate is stronger still at
**1,296/1,296 byte-exact logits, KL 0, top-1 100%**, deterministic repeats, and
clean lifecycle. Threshold tests keep serial at 32K-1 and explicit opt-out.
Retain the gfx1151 capability from 32K. Evidence:
[`SH7-A1`](../benchmarks/results/2026-08-06-gfx1151-gguf-sh7-a1-parallel-split-reducer-retained.json).

Parity remains open: new 32K/64K decode is **46.785/40.386 tok/s**, still
**1.111/1.187 ms/token** short of C1 and **2.522/2.936 ms/token** short of the
pinned-fork F16 lane; whole-GTT does not move. Select **SH8-A1**, not another
reducer, split-count, page-layout, compact-KV, raw-Q8, or overlap retry. The
remaining grouped-GQA producer owns **4.037/7.212 ms/token** and runs one
72-VGPR local256 block per `(kv_head, split)`. Screen a structurally new exact
four-query ownership sibling that trades duplicated K/V reads for lower register
pressure. Require exact per-head reduction order and a cached actual-shape leaf
win of >=1.10x or >=0.5-ms/token projection before any full-model route.

The launch audit is frozen in
[`2026-08-06-gfx1151-gguf-post-sh-g-parity-gap-audit.json`](../benchmarks/results/2026-08-06-gfx1151-gguf-post-sh-g-parity-gap-audit.json).
`python3 scripts/check_lineage.py --kind kernel --diff stat` is currently blocked
because the read-only Atlas reference path is absent. That tooling failure is
recorded for SH3-D1 and is not a green lineage verdict.

### Cumulative decode targets

These are reporting/stop targets, never license to specialize to the repeated-
token benchmark. Every candidate must also win on natural prompts and category
heldouts with no token-, prompt-, or candidate-ID-conditioned branch.

| Stage | 512/128 | 4K/128 | 32K/128 | 64K/128 |
| --- | ---: | ---: | ---: | ---: |
| SH-C0 hipEngine BF16 | 52.857 | 55.389 | 46.004 | 39.419 |
| Final SH-G hipEngine BF16 | **53.446** | **56.116** | **46.489** | **39.750** |
| SH7-A1 one-queue candidate (32K+ scoped) | — | — | **46.785** | **40.386** |
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
| **SH-D1 — exact row-1 weight-kernel redesign** | **Completed 2026-08-06; Q5 tile8 retained, all other measured ladders closed** | Split dense Q8T16 by role (full Q/K/V/O, GDN projections, shared expert), selected Q4/Q5/Q6 T16, and lm-head. Exact selected-Q5 tile8 is retained at **1.1715x** and committed 512/128 **+0.998%**. GDN tops out at **1.0648x**, selected Q4 at **1.1466x**, Q6 down at **1.0803x**, and Q6 lm-head tile8 regresses **0.218%**; all transient siblings are removed. Cumulative decode is **53.535 tok/s**, **1.487 ms/token** short of C1. | Primitive CPU oracle, named trace, no scratch spill, and full model quality/state gate. A leaf must be at least **1.15x** or save **0.5 ms/token** before a full-model run. First cumulative gate is the half-time-gap row above; continue until F16 parity or measured Amdahl headroom is exhausted. Reject any 512/4K win that loses >1% prefill or regresses another context. |
| **SH-M1 — query-row Pareto screen** | **Completed/rejected 2026-08-05** | The explicit 4,096 -> 1,024 query-row A/B saves **1.335-1.338 GiB tracked** and **1.351-1.355 GiB whole-GTT** at 4K+, with exact cleanup and neutral decode. | Reject q1024 and close 2,048/768 row-only retries: q1024 changes exact state/logits at 4K+ and loses **1.603%-11.835%** prefill. Keep q4096. |
| **SH-M2 — exact scratch-liveness aliases** | **Completed/retained 2026-08-06** | Keep every 4,096-row execution shape fixed and graph-color route/stage-disjoint fields into 21 independent allocator-owned slots. Rows below 4,096, diagnostics, unvalidated routes, and peer backends keep dedicated owners. | Exact state and lifecycle pass at 512/4K/32K/64K. The default saves **1.4086 GiB tracked** and **1.4043 GiB whole-GTT** at every 4K+ row; prefill/decode remain within 1%. Contiguous and split arenas are rejected. SH-K1 later fails to stack another peak-memory win. |
| **SH-K1 — strict compact-KV frontier** | **Completed/rejected for default 2026-08-06** | The fixed `0-7` BF16 / `8-9` per-token/head INT8 K+V map passes the actual no-mirror 65,792-capacity category+heldout gate at **3.344e-5 mean KL, 7.875e-4 max KL, and 100% aggregate/min-prompt top-1**. Broader and key-only/block formats remain closed by prior evidence. | Reject default promotion: 32K/64K decode changes **-0.436%/-0.670%**, live ownership saves only **0.0620/0.1235 GiB**, and layer-local BF16 prefill oracles instead raise tracked peak **0.0640/0.1274 GiB** plus whole-GTT **0.0703/0.1328 GiB**. Named direct-consumer traces pass; BF16 remains default and SH-A1 is next. |
| **SH-A1 — page-internal head-major decode screen** | **Completed/rejected 2026-08-06** | The bounded current-vs-page-head grouped-GQA leaf is exact on dense/permuted/evicted page fixtures, but candidate attention+reducer is only **0.756x/0.797x** at 32K/64K before conversion. | Reject runtime plumbing: append+copy-inclusive speedup is only **0.277x/0.263x**, projecting **-58.3%/-97.8%** decode across ten full-attention layers. Remove every transient surface and keep token-major BF16 KV. |
| **SH-G — retained recertification** | **Completed 2026-08-06; campaign closed** | Final production passes the four one-warmup/three-measurement rows, exact 18-prompt oracle, allocator/GTT lifecycle, cached role/Q5 trace, and fresh five-repetition fork milestone rerun. Decode is diagnostically **0.840%-1.314%** above SH-C0 and 4K+ tracked peak stays **1.4086 GiB** lower. | Final correctness/lifecycle/trace gates pass, but **0/4 C1**, **0/4 C2**, **0/4 fork decode**, and **0/4 fork whole-GTT** rows pass. Publish the prior same-revision own-engine gains and qualified final diagnostic; close the declared campaign without claiming fork parity. |

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
The first implementation result is frozen in
[`2026-08-06-gfx1151-gguf-sh-d1-gdn-dpp-rejected.json`](../benchmarks/results/2026-08-06-gfx1151-gguf-sh-d1-gdn-dpp-rejected.json).
The transient exact sibling met the intended mechanical contract: BF16 bytes
match production and the CPU Q8_0 oracle at boundary, ordinary, and full Qwen
shapes; its 2,976-byte body emits **16 `v_permlanex16_b32`**, **64 direct
`v_add_f32_dpp`**, zero bpermutes, one barrier, 50 VGPR, and zero spills/private
bytes; cached `rocprofv3` names the expected kernel.

It nevertheless fails the frozen leaf gate. Three order-balanced 80-warmup /
400-launch cycling pairs measure production at **135.02/135.92/135.20 us** and
the candidate at **132.61/132.84/132.77 us**. Median speedup is only
**1.0183x**; the **2.43-us** saving projects to **0.0729 ms/token** over 30
calls. That is **12.05% slower** than the **118.493-us** continuation target and
also misses the independent **117.530-us** 1.15x target. Per the predeclared
stop rule, no full-model matrix was run and the kernel, wrapper, registry key,
test, and microbench mode were removed. Production remains unchanged.

The follow-up same-layout ladder is frozen in
[`2026-08-06-gfx1151-gguf-sh-d1-gdn-samelayout-rejected.json`](../benchmarks/results/2026-08-06-gfx1151-gguf-sh-d1-gdn-samelayout-rejected.json).
The non-temporal sibling preserves three 16-byte weight loads and emits gfx1151
`slc dlc` policy on all three; its stacked sibling adds the exact 16
permlanex16/64-DPP tree. Both are byte-identical and CPU-oracle exact, spill-
free at 51 VGPR, and named by cached traces. Across three order-balanced pairs,
production is **135.44/135.26/135.23 us**, non-temporal is
**130.82/131.56/131.64 us**, and stacked is **130.11/130.83/129.85 us**.

The best **130.11-us / 1.0396x** median saves **5.15 us/call**, projecting only
**0.1545 ms/token**. It remains **9.80% slower** than the 118.493-us gate and
misses 1.15x, so both siblings are removed before full-model work. Same-layout
communication/cache-policy headroom is now exhausted.

The bounded replacement-layout result is frozen in
[`2026-08-06-gfx1151-gguf-sh-d1-gdn-q8t32-rejected.json`](../benchmarks/results/2026-08-06-gfx1151-gguf-sh-d1-gdn-q8t32-rejected.json).
The diagnostic packer pairs adjacent Q8T16 scales and per-K-lane payloads into a
byte-neutral 1,088-byte Q8T32 record. A local256/eight-wave first object keeps
two independent production-equivalent four-wave teams; its best sibling stacks
cache-bypassing loads and exact DPP at 53 VGPR with zero spills. Three cycling
triples measure production **136.25/136.19/135.79 us**, cached Q8T32
**130.79/130.89/131.23 us**, and stacked Q8T32
**127.77/128.96/127.90 us**. A local128 register-wide sibling reaches 91 VGPR
and regresses production **4.96%**, so it is physically closed.

The best paired result is **127.90 us / 1.0648x**, saving **8.29 us/call** and
projecting **0.2487 ms/token**. It remains **7.94% slower** than the 118.493-us
continuation target and misses 1.15x. All candidates pass byte identity, CPU
Q8_0 oracles, zero-scratch ISA, and named traces, but all kernels, diagnostic
packer, wrappers, registry keys, tests, and modes are removed before runtime
materialization or full-model timing. Production stays Q8T16. GDN same-layout
and byte-neutral paired-layout headroom are now exhausted under the frozen gate;
SH-D1 advances to the measured **3.912-3.925-ms/token selected gate/up+down**
role instead of stopping the campaign.

#### SH-D1 selected-Q4 ownership update

The exact Qwen c1/top8 selected-Q4 ownership ladder is frozen in
[`2026-08-06-gfx1151-gguf-sh-d1-selected-q4-tile-ladder-rejected.json`](../benchmarks/results/2026-08-06-gfx1151-gguf-sh-d1-selected-q4-tile-ladder-rejected.json).
The production fused-SiLU owner runs at **57.824 us/call** in the immutable
67.9-MB cycling screen and costs **2.309-2.319 ms/token** across 40 MoE layers.
It is local128 with 16 gate plus 16 up accumulators, 200 VGPR, 1,024 B LDS, and
zero scratch.

The exact tile8 candidate keeps the T16 bytes and every K/FMA/reduction/BF16
boundary while splitting each 16-column tile into two workgroups and loading
adjacent packed-Q/coefficient pairs. It reduces resources to **72 VGPR / 512 B
LDS / zero scratch** and measures **50.429 us / 1.14664x**, saving **7.395
us/call** and projecting **0.2958 ms/token**. That is only **0.292%** below the
frozen 1.15x threshold, but the predeclared rule is fail-closed; it also misses
the independent 0.5-ms alternative. The final exact tile4 point falls to
**52.915 us / 1.09145x / projected 0.1936 ms/token** despite only 40 VGPR.

Both candidates pass the CPU Q4_K oracle, four-axis resolution, exact BF16
hashes, and named cached traces. Neither qualifies for a full-model run, and all
candidate kernels, wrappers, keys, tests, and microbench code are removed.
Production stays on the 16-column owner. SH-D1 continues immediately with the
**1.603-1.606-ms/token Q5/Q6 selected-down** role; the overall campaign remains
open.

#### SH-D1 selected-down Q5 retention / Q6 closure

The selected-down decision is frozen in
[`2026-08-06-gfx1151-gguf-sh-d1-selected-down-q5-tile8-retained.json`](../benchmarks/results/2026-08-06-gfx1151-gguf-sh-d1-selected-down-q5-tile8-retained.json).
The production c1/top8 owner reads resident Q5/Q6 T16 at K512/N2048 with one
local128 workgroup per 16 columns. The exact candidate changes only independent
column ownership: two eight-column workgroups retain every column's K/FMA,
wave32 tree, serial wave-0..3 reduction, and BF16 store.

Q5 passes the predeclared leaf gate. The final 70.8-MB cycling screen measures
production **40.815 us** and tile8 **34.865 us**, or **1.17067x**, saving
**5.950 us/call** and projecting **0.2202 ms/token** across 37 Q5 layers. A
cached-only trace names the expected tile8 kernel at local128, grid256x8,
**56 VGPR / 512 B LDS / zero scratch**, versus production grid128x8 and 128
VGPR. gfx1151 therefore defaults the exact shape through
`GGUF_Q5_T16_SELECTED_QWEN_TILE8`; gfx1100 and all shape/quant misses retain the
16-column owner. The short-lived environment comparison seam was removed after
committed publication and cumulative attribution; the direct owner remains the
required peer/shape fallback and primitive oracle.

The post-commit same-revision checkpoint at `3e836edea` independently confirms
the leaf at **40.855 -> 34.876 us (1.1715x / projected 0.2212 ms/token)** and
512/128 eager at **53.027 -> 53.557 tok/s (+0.998%)** or **18.8582 ->
18.6718 ms/token**. All three candidate samples beat all controls, every
128-token run is exact, and tracked close bytes remain zero. The shared source
recorder reports dirty only for unrelated `docs/ROCM-AI.md` plus pre-existing
untracked artifacts; runtime, kernel, benchmark, and test sources match the
implementation commit. Complete state payloads at **512/4K/32K/64K** are
byte-identical across prefill logits,
hidden/layer/Conv/GDN/live BF16-KV state, four fixed-input transitions, and
final state. The natural ten-prompt plus eight-heldout oracle passes all 54
prompt repeats, **1,350 token** and **54,000 layer-hidden** comparisons with
zero mismatch.

Q6 fails the same gate and is physically closed. Tile8 reaches only **41.534 ->
38.448 us / 1.0803x / projected 0.0093 ms/token**; the final tile4 point
regresses to **44.817 us / 0.9325x**. Both are exact, but their wrappers, C ABI
symbols, registry keys, exports, and tests are removed. Production Q6 remains
unchanged. SH-D1 now advances to cumulative attribution and remaining measured
parity headroom rather than stopping at this retained leaf.

#### SH-D1 cumulative 512 checkpoint

The post-retention role refresh is frozen in
[`2026-08-06-gfx1151-gguf-sh-d1-cumulative-512-checkpoint.json`](../benchmarks/results/2026-08-06-gfx1151-gguf-sh-d1-cumulative-512-checkpoint.json).
Non-profiled 512/128 is **53.535 tok/s / 18.6792 ms/token**, exact across all
three runs, while cached role attribution is **17.3591 ms/token** over the same
628 dispatches/token. Selected-down falls **1.6028 -> 1.3933 ms/token**, a
**0.2095-ms** role saving that accounts for most of the **0.2367-ms** profiled
total reduction versus SH-C0.

This still misses C1 by **1.4868 ms/token** and fork-F16 parity by **3.2132
ms/token**. The largest roles remain GDN input (**4.1132 ms**) and selected Q4
gate/up (**2.3154 ms**), but their exact ladders are closed under admission.
The next untested ownership point is the **1.8310-ms/token** Q6T16 lm-head:
screen a c1 K2048/N248320 tile8 owner once, preserving each FP32 logit's exact
K/FMA/reduction/store order and charging the complete 417-MiB-class matrix.
The older producer-owned tile-max fusion saves only **6.853 us** and is below
SH-D1 admission by itself. Do not run a full-model route unless tile8 reaches
**1.15x** or projects **0.5 ms/token**.

The final Q6T16 lm-head screen is frozen in
[`2026-08-06-gfx1151-gguf-sh-d1-q6-lm-head-tile8-rejected.json`](../benchmarks/results/2026-08-06-gfx1151-gguf-sh-d1-q6-lm-head-tile8-rejected.json).
Tile8 lowers allocated VGPR **72 -> 48** with zero scratch and preserves every
one of the 248,320 FP32 logits plus top-1, but the complete 417,177,600-byte
matrix screen moves **1.83174 -> 1.83575 ms (0.99782x, -0.218%)**. Doubling
workgroups cannot repay lower accumulator pressure. The candidate body,
wrapper, key, test, and microbench are removed before runtime routing.

SH-D1 is therefore complete, not successful at parity: its one retained exact
leaf compounds a measured win, while all currently supported byte-neutral
row-1 weight ownership families are retained or below their frozen admission
gates. The residual **1.4868 ms/token** to C1 and **3.2132 ms/token** to fork
F16 remain explicit campaign gaps. Attention/layout work proceeds under SH-A1;
the campaign now advances to SH-M2 and then SH-K1 rather than repeating closed
weight schedules.

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
