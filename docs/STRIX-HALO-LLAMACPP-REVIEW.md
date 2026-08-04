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

The initial review identified **one immediate, bounded experiment** on
hipEngine, which has now completed:

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

| Workload | hipEngine BF16 prefill | Fork F16 / Q8 prefill | hipEngine BF16 decode | Fork F16 / Q8 decode |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **1394.772** | 1390.978 / 1387.763 | 52.710 | **64.658** / 64.246 |
| 4K/128 | **1472.330** | 1399.336 / 1389.530 | 55.183 | 62.648 / **63.107** |
| 32K/128 | **1171.610** | 1102.982 / 1104.183 | 45.943 | 53.220 / **57.553** |
| 64K/128 | **952.348** | 893.919 / 879.963 | 39.362 | 45.913 / **52.538** |

hipEngine leads every full-prompt prefill row: **0.273%-6.536%** versus fork
F16 KV and **0.505%-8.226%** versus fork Q8_0 KV. The fork leads every tg128
decode row: **13.528%-22.666%** with F16 and **14.361%-33.474%** with Q8_0.
Fork whole-device GTT peaks are **20.831-22.871 GiB** with F16 and
**20.832-22.239 GiB** with Q8_0, descriptively **0.648-2.153 GiB** below
hipEngine's **21.480-24.392-GiB** tracked peaks.

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
| Persistent head-major K/V cache | The experimental layout changes K from token-major to `[head_dim, kv_size, n_head_kv]` ([`0f74840`, lines 231-254](https://github.com/Nathanw1014/llama.cpp/blob/0f748408e2af0f4fe05b2ccdf7a7765bf6cc29fe/src/llama-kv-cache.cpp#L231-L254)). Later commits restrict formats/consumers after correctness failures. | **Rejected after the scratch A/B.** All paged writers, decode kernels, copies, compaction, graph captures, and `KVLiveSpans` consumers assume the current physical row layout, while the 64K copy is only **0.032%** of full prefill across ten layers. | Keep persistent paged KV unchanged; revisit only if a future consumer makes copy wall material. |
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
- **Persistent head-major KV is not the next step.** At 64K the standalone copy
  is **2.233 ms**, only **0.708%** of the **315.417-ms** copy-inclusive attention
  sub-window. Even charging one copy to each of ten full-attention layers is
  **22.329 ms**, about **0.032%** of the measured **68.815-s** full prefill. That
  removable cost cannot justify changing every writer, decode reader, graph,
  compactor, and `KVLiveSpans` consumer. Revisit only if a future consumer makes
  the copy material.
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

## Final priority list

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
   persistent head-major KV, MMID scale cache/TILE16/int-dot negatives, generic
   ubatch values, and RADV packaging remain unsupported. The successful scratch
   A/B strengthens this decision: its 64K copy is only **0.708%** of the
   candidate attention sub-window and at most **0.032%** of full prefill across
   ten full-attention layers. The later locally measured fork diagnostic
   establishes a real prefill/decode split without making any of those
   backend-specific changes portable.

## Completion audit

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
under the focused-repair rule. There is no unassigned action left in this review;
future 128K driver work requires a named stack fix, and DeepSeek V4 requires
explicit model admission. The later exact-model fork comparison is separately
retained as diagnostic evidence and does not reopen any completion item.
