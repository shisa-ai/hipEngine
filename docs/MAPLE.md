# MAPLE — Ternary MoE inference on hipEngine

Last updated: **2026-08-08** (branch `maple`)

## Summary

Maple-Preview is DeepGrove's approximately **20B-total / 1B-active**
Mixture-of-Experts language model. It uses 24 all-MoE transformer layers,
selects 8 of 256 small experts per token, alternates three sliding-window
attention layers with one global layer, and was trained for ternary projection
weights. From the pinned geometry, the model contains about **20.21B logical
weight coefficients**; about **1.18B coefficients participate in one token
forward** when the selected experts and exact full-vocabulary head are counted.

Two related checkpoints matter:

| Checkpoint | Role | Size / format | hipEngine use |
| --- | --- | --- | --- |
| [`deepgrove/maple-preview`](https://huggingface.co/deepgrove/maple-preview) | BF16 source model and independent Transformers oracle | ~40.4 GB, custom `MapleForCausalLM` | Correctness oracle only; not the deployed runtime payload |
| [`deepgrove/maple-preview-2bit-mlx`](https://huggingface.co/deepgrove/maple-preview-2bit-mlx) | Official quantized deployment checkpoint | **5,308,186,624 bytes** of exact weights (5.31 GB / 4.944 GiB), 463 required tensors | The checkpoint hipEngine loads directly |

The deployment checkpoint stores the model's trained ternary projections in a
2-bit packed representation and keeps embeddings plus the exact LM head in
4-bit affine form. hipEngine consumes those layouts directly—there is no Torch
hot path, no full model dequantization, and no host-side expert repack. The
optional approximate FlashHead sidecar is present in the repository but is **not
used** by the exact production path.

### Current status at a glance

- Public model-ID loading and greedy text generation work on
  `hip_gfx1151`/Radeon 8060S and the peer `hip_gfx1100` registry key.
- Native prefill now continues safely beyond the 512-token sliding window up
  to the configured context. The retained 520/770-token gates are byte-exact to
  serial through physical K/V rings, complete `KVLiveSpans`, final state, and
  continuation; the fixed performance protocol remains 128/320/512.
- Public `SubmitPollTextGenerator` admission is live at fixed capacities through
  8. One resident checkpoint owner feeds request-local native prompt prefill,
  exact c1 decode for a lone active row, and D1 batched decode for c=2/4/8.
- Public c1/c2/c4/c8 throughput is **123.131/165.697/202.038/214.788 aggregate
  tok/s** at 64 generated tokens/request, or **1.346x/1.641x/1.744x** versus the
  same public c1 protocol. Sparse and staggered slot reclaim are exact.
- Sampling remains greedy-only (`temperature=0`), and all tracked allocations
  return to zero on close.

## Model architecture

| Field | Pinned value |
| --- | --- |
| Layers | 24, all MoE (`first_k_dense_replace=0`) |
| Hidden / head dimension | 2048 / 128 |
| Attention | GQA: 16 query heads / 4 KV heads; per-head QK-RMSNorm |
| Layer pattern | 18 sliding-attention + 6 full-attention layers, repeated 3:1 |
| Sliding window | 512 tokens |
| Position encoding | partial RoPE on sliding layers only; rotary dimension 64, theta 10,000; global layers are NoPE |
| Context | Marketing claim: 131,072 tokens; pinned deployment config: `max_position_embeddings=128000`; hipEngine public default: 4K; retained fixed-shape performance qualification: <=512; exact long-prompt state gate: 770 |
| MoE | 256 experts, stable top-8, FP32 router logits, all-expert softmax, selected-weight renormalization |
| Expert MLP | 2048 → 512 gate/up, trained clamp-7 SwiGLU, 512 → 2048 down; no shared expert |
| Norms | RMSNorm, epsilon 1e-6, FP32 internal arithmetic and BF16 boundary |
| Vocabulary | 151,936 Qwen2 BPE tokens; untied exact LM head |
| EOS | 151645 (`<|im_end|>`) |

## Deployment checkpoint format

### Ternary projections

Self-attention Q/K/V/O and all expert gate/up/down projections use U32 words
with 16 LSB-first 2-bit codes per word. Each output row has a BF16 `row_alpha`:

```text
weight = row_alpha * (code - 1),  code in {0, 1, 2}
```

The logical values are therefore `{-alpha, 0, +alpha}`. Expert tensors remain
stacked as `[256, out, in/16]`, and selected-expert kernels read only the routed
experts.

### Embedding and LM head

The embedding and untied LM head use LSB-first affine 4-bit/group-64 storage:

```text
weight = q4_code * bf16_scale + bf16_bias
```

The exact LM head covers all 151,936 vocabulary rows. The separate
`model-flashhead.safetensors` file is optional and intentionally excluded from
the exact 463-tensor / 5,308,186,624-byte manifest.

## hipEngine execution paths

### Public resident generation

`LLM("deepgrove/maple-preview-2bit-mlx", backend="auto", quant="auto")`
resolves to `MapleGenerator` without importing Torch. Direct generation keeps a
resident `MapleRunner`; the public submit/poll engine asks the generator for a
fixed-capacity `MapleResidentModelRunner` (maximum 8 physical rows).

- **Prompt prefill:** `MapleRunner.prefill_native()` processes rows in chunks of
  at most 256 and returns only the final row's exact sampled result. Beyond the
  512-token SWA boundary, global layers remain chunk-batched while each sliding
  layer restores its pre-chunk ring mapping, batches the safe prefix, and
  serializes only post-wrap attention rows. The 520/770-token state gates are
  exact to token-serial prefill.
- **Admission:** one c1 owner retains immutable packed weights. A
  `MapleBatchRunner.from_runner()` owner adds fixed request-local SWA/global K/V
  rings; request-local `KVLiveSpans` and cache views let prompt chunks land
  directly in the physical slot later consumed by decode.
- **Decode:** one active row uses the retained D0 c1 path on that slot's K/V
  view. Two or more active rows use D1's exact c2/c4/c8 affine4 row-reuse path;
  sparse physical masks preserve holes without moving cache state.
- **Reclaim:** first-free slots reset on rollback, cancellation, or completion.
  A retained staggered test completes slot 0 while slot 1 remains live, admits a
  third request into slot 0, and matches all serial trajectories.

`MapleBatchRunner` and `MapleContinuousBatcher` remain useful low-level kernel
and helper harnesses, but public generation now uses the same fixed-slot cache
and c-aware decode machinery. Prompt **storage admission** is packed into those
slots; prompt compute is currently native per request rather than one ragged
multi-request GEMM.

## Interpreting DeepGrove's Apple numbers

DeepGrove's current MLX repository distinguishes its exact and approximate
heads; the model-card headline does not. At
[`mlx-lm-deepgrove@eba96c1`](https://github.com/deepgrove-ai/mlx-lm-deepgrove/tree/eba96c16158f032821b0bf374ea1421cfddef0a9)
the published table is:

| Apple path | Decode | Prefill | Peak |
| --- | ---: | ---: | ---: |
| M4 exact/default | **169 tok/s** | **1075 tok/s** | 6.51 GB |
| M4 `--flash-head` | **218 tok/s** | **1075 tok/s** | 6.69 GB |
| M5 Pro exact/default | **359 tok/s** | **3773 tok/s** | 6.73 GB |
| M5 Pro `--flash-head` | **395 tok/s** | **3857 tok/s** | 6.92 GB |

The **218 tok/s** headline is therefore not dense-head exact decode. FlashHead
scores 4,748 quantized vocabulary centroids, selects 512 clusters of 32 tokens,
and computes exact affine4 logits for those 16,384 candidates (about 10.8% of
the vocabulary) plus three forced control tokens. Greedy output matches the
exact head only when the true argmax is in a probed cluster.

Two upstream implementation details explain the prefill gap more directly:

1. MLX generation evaluates only cache state for discarded non-final prompt
   chunks, so lazy graph elimination does not execute their LM heads; the final
   prompt token alone enters the sampled exact head.
2. Upstream `MapleSwitchGLU` sorts 64 or more routed assignments by expert
   before its switch projections, enabling expert-weight reuse across prompt
   rows.

The upstream table does not publish prompt/generation lengths, repetitions,
software versions, or a correctness protocol, and Apple M4 and Radeon 8060S
are different systems. It is directional evidence, not an apples-to-apples
benchmark. hipEngine's current fixed-token exact c1 A/B candidate is **202.580
tok/s**; its separate cached trace process is **199.293 tok/s**, and its current
category-qualified natural-context row is **153.201 tok/s**. Those distinct
workloads must not be collapsed into one number or compared directly with
Apple's unspecified protocol. All are exact full-head paths, unlike the
approximate 218 headline. After P0+P1+P2,
hipEngine's current **741.890 tok/s** at 320 tokens is about 1.45x below the
published M4 prefill rate, down from a 3.3x gap. Decode is competitive; prefill
remains materially underoptimized. The separate public batching protocol now
reaches **214.788 aggregate tok/s at c8**, but that includes prompt admission and
uses a different workload from either Apple's table or the fixed-token c1 row.

## Current Radeon 8060S / gfx1151 performance

All retained rows use the pinned 2-bit checkpoint, `GPU_MAX_HW_QUEUES=1`, a
clean tracked revision, repeated measurements, actual `rocminfo`/`rocm-smi`
capture, and exact teardown accounting. Full commands and samples are in the
linked artifacts.

### Public native prefill

The M5 protocol measures eight fixed-shape inputs per length—one natural and one
heldout prompt from each of code, general English, general Japanese, and mixed
Japanese/English—after one warmup, with three repetitions.

| Prompt tokens | Native prefill tok/s | Serial reference tok/s | Speedup |
| ---: | ---: | ---: | ---: |
| 128 | **750.854** | 158.602 | **4.734x** |
| 320 | **741.890** | 111.607 | **6.647x** |
| 512 | **754.458** | 86.099 | **8.763x** |

Native sample ranges are 740.203-759.867, 735.195-753.172, and
750.302-758.535 tok/s respectively. P4 changes the retained P2 rows by only
**+0.224%/+0.070%/+0.061%**, passing the <=1% preservation gate while keeping
all **18/18** state hashes and **90/90** positions exact. P2 remains the kernel
optimization that produced the gain over P1; P4 extends its orchestration and
public reach without changing arithmetic.

Evidence:
[`P4 long prefill + public admission`](../benchmarks/results/2026-08-08-gfx1151-maple-p4-long-prefill-public-batch-retained.json)
and [`P2 GQA4`](../benchmarks/results/2026-08-08-gfx1151-maple-p2-gqa4-prefill-retained.json).

### Decode

| Path / workload | Current rate | Scope |
| --- | ---: | --- |
| c1 natural+heldout contexts | **153.201 tok/s** (6.527-ms mean, 6.498-ms median) | clean paired 18-prompt qualification; model load, native prefill, and warmup excluded |
| c1 fixed-token A/B candidate | **202.580 tok/s** (4.936-ms process mean) | four alternating fresh baseline/candidate processes; all candidate processes >201 tok/s |
| c1 cached-trace companion | **199.293 tok/s** (5.018-ms median) | separate clean trace process; 4.550-ms kernels and 271 launches/token |
| c=2, 64 tokens/request | **250.481 aggregate tok/s** | exact row-reuse helper median, 3 repeats |
| c=4, 64 tokens/request | **346.365 aggregate tok/s** | exact row-reuse helper median, 3 repeats |
| c=8, 64 tokens/request | **428.063 aggregate tok/s** | exact row-reuse helper median, 3 repeats; excludes prompt/public scheduling |
| public c1/c2/c4/c8, 64 tokens/request | **123.131/165.697/202.038/214.788 aggregate tok/s** | same public submit/poll protocol; prompt admission and reclaim included |

The D0 one-dispatch router is retained/default. On all natural and heldout
contexts it improves its exact two-dispatch rollback **139.538 -> 145.321
tok/s (+4.14%)**, saves **0.301 ms** at the paired median, and wins
**1,127/1,152** timed pairs. Every **1,296/1,296** continuation token and top
logit is exact; all **36/36** native-start and final state pairs match, the
four-byte counter passes **2,592/2,592** zero checks, and close returns tracked
ownership to zero.

The exact wave32 affine4 head is also retained/default. It improves its group64
rollback **143.679 -> 153.409 tok/s (+6.77%)**, saves **0.442 ms** at paired
median, and wins **1,146/1,152** pairs with the same complete state/counter/
lifecycle gate exact. This is full-vocabulary affine4, not FlashHead.

The final D0 host cleanup snapshots two invariant default-off fusion selectors
once per token instead of reading them in every layer. A clean alternating
fresh-process gate improves **200.279 -> 202.580 tok/s (+1.15%)**, saves **0.076
ms** at paired median, and wins **3/4** pairs; all four candidate processes are
above 201 tok/s. It changes no kernel, launch, pointer, allocation, or math. The
complete current category gate is exact and its 153.201 tok/s candidate is only
**0.14%** below the prior natural-context row.

D1's exact batched affine4 head is retained/default. One block owns one
vocabulary row across all requests, reducing clean selector-unset c2/c4/c8
helper medians **218.818/261.099/299.181 -> 250.481/346.365/428.063 tok/s
(+14.47%/+32.66%/+43.08%)**. Every measured c2/c4/c8 trajectory matches an
independent c1 trajectory. The 18-prompt category/heldout seed gate also passes,
including a sparse final c8 group, and explicit reclaimed-slot reuse is exact.
The head writes logits only and is FP32-bit exact to the original all-row
kernel, so hidden/KV/span state has no new writer. These helper rows exclude
model load and prompt prefill and must not be reported as public throughput.

P4 now measures the actual public submit/poll boundary symmetrically. c1/c2/c4/
c8 reaches **123.131/165.697/202.038/214.788 aggregate tok/s**, so c2/c4/c8 is
**1.346x/1.641x/1.744x** the same public c1 protocol. Timed wall includes prompt
admission, native prefill, exact decode/sampling, output collection, and reclaim;
only model load and one lazy-allocation warmup are excluded. All 15 repeated
18-prompt trajectory sets equal serial. A lone request in the physical-c8 owner
runs at **122.778 tok/s (99.714% of public c1)**, and a staggered third request
reuses slot 0 while slot 1 stays live.

Evidence:
[P4 long prefill + public admission](../benchmarks/results/2026-08-08-gfx1151-maple-p4-long-prefill-public-batch-retained.json),
[D1 batched affine4 row reuse](../benchmarks/results/2026-08-08-gfx1151-maple-d1-batched-affine4-rowreuse-retained.json),
[D0 c1 router qualification](../benchmarks/results/2026-08-08-gfx1151-maple-d0-c1-router-retained.json),
[D0 affine4 qualification](../benchmarks/results/2026-08-08-gfx1151-maple-d0-affine4-wave32-retained.json),
[D0 selector snapshot](../benchmarks/results/2026-08-08-gfx1151-maple-d0-selector-snapshot-retained.json),
[D0 wave32 profile](../benchmarks/results/2026-08-08-gfx1151-maple-d0-wave32-decode-profile.json),
and [M6 helper recertification](../benchmarks/results/2026-08-07-gfx1151-maple-m6-batch-decode-recertified.json).

### Tracked memory

| Resident configuration | hipEngine-owned device memory |
| --- | ---: |
| Exact checkpoint payload alone | **4.944 GiB** |
| Public c1 runner, max context 512 (P2/M5 protocol) | **4.988 GiB** |
| Batch helper c=2 / c=4 / c=8, capacity 66/request | **4.951 / 4.958 / 4.973 GiB** |
| Public fixed-slot c1 / c2 / c4 / c8, default 4K context | **5.115 / 5.180 / 5.310 / 5.571 GiB** |
| After `close()` | **0 bytes / 0 active allocations** |

These are exact process-local hipEngine allocation counters, not sampled
whole-device GTT. They exclude allocations internal to the HIP runtime and
other processes. The low-level batch helper uses much smaller row scratch and
short per-request capacity than either the 256-row public prefill runner or the
public default-4K owner, so those rows are not directly comparable.

## Correctness evidence

The implementation uses several independent gates rather than treating coherent
text as proof of numerical correctness.

| Gate | Result |
| --- | --- |
| Packed NumPy/Torch formula vs hipEngine, 18 positions | max KL **0.013508**, mean KL 0.001679, top-1 **18/18** |
| Pinned HF `trust_remote_code` oracle with matched affine4 endpoints | max KL **0.004719**, mean KL 0.000723, top-1 **18/18** |
| P2/M5 native vs serial, 18 natural+heldout prompts / 90 seed+continuation positions | **18/18** byte-exact final-hidden/normalized/live-KV/span state hashes; max/mean KL **0/0**; top-1/token equality **90/90** |
| D0 one- vs two-dispatch router, 18 natural+heldout prompts / 36 repeated trajectories | **36/36** native-start and final state hashes; **1,296/1,296** tokens/top logits; **2,592/2,592** zero-counter checks |
| D0 wave32 vs group64 affine4 head, same complete protocol | **36/36** native-start and final state hashes; **1,296/1,296** tokens/top logits; **2,592/2,592** zero-counter checks |
| D0 selector-snapshot current production, repeated complete protocol | same **36/36** state pairs, **1,296/1,296** positions, **2,592/2,592** zero-counter checks, and exact teardown |
| P4 520/770-token native vs serial physical-state gate | FP32 top-logit bits, final hidden/norm, every live physical K/V byte, both span owners, and three continuations exact |
| D1/M6 c=2/4/8, sparse/reclaimed slots, and 514-step SWA wrap | FP32 head bits and all generated trajectories exact |
| P4 public c1/c2/c4/c8 + physical-c8 singleton | **15/15** repeated 18-prompt sets, **270/270** trajectories, and **17,280/17,280** generated tokens exact |
| P4 staggered submit/poll reuse | slot 0 reclaimed/reused while slot 1 remains live; all three trajectories exact |
| Public canonical prompt | coherent 37-token answer, real EOS 151645, deterministic resident repeat |
| Lifecycle | every c1/c2/c4/c8/singleton owner returns tracked ownership to zero |

The untouched dense-BF16 endpoint comparison remains an intentionally failed
quantization-quality diagnostic (max KL 0.149840, top-1 16/18). Matching the
packed checkpoint's affine4 embedding and head reduces the implementation gate
to the accepted values above; thresholds were not weakened.

Primary evidence:

- [`maple-ternary2-correctness.json`](../benchmarks/results/2026-08-05-gfx1151-maple-ternary2-correctness.json)
- [`maple-public-e2e-smoke.json`](../benchmarks/results/2026-08-05-gfx1151-maple-public-e2e-smoke.json)
- [P4 long prefill + public admission](../benchmarks/results/2026-08-08-gfx1151-maple-p4-long-prefill-public-batch-retained.json)
- [D1 exact batched affine4 row reuse](../benchmarks/results/2026-08-08-gfx1151-maple-d1-batched-affine4-rowreuse-retained.json)
- [D0 one-dispatch c1 router](../benchmarks/results/2026-08-08-gfx1151-maple-d0-c1-router-retained.json)
- [D0 exact wave32 affine4 head](../benchmarks/results/2026-08-08-gfx1151-maple-d0-affine4-wave32-retained.json)
- [D0 selector snapshot](../benchmarks/results/2026-08-08-gfx1151-maple-d0-selector-snapshot-retained.json)
- [D0 wave32 decode profile](../benchmarks/results/2026-08-08-gfx1151-maple-d0-wave32-decode-profile.json)
- [D0 pre-head decode profile](../benchmarks/results/2026-08-08-gfx1151-maple-d0-decode-profile.json)
- [P3 dense token-tile rejection](../benchmarks/results/2026-08-08-gfx1151-maple-p3-dense-token-tile-rejected.json)
- [P2/M5 GQA4 recertification](../benchmarks/results/2026-08-08-gfx1151-maple-p2-gqa4-prefill-retained.json)
- [P1/M5 expert-major recertification](../benchmarks/results/2026-08-07-gfx1151-maple-p1-expert-major-prefill-retained.json)
- [P0/M5 final-row recertification](../benchmarks/results/2026-08-07-gfx1151-maple-p0-final-row-prefill-retained.json)
- [M6 recertification](../benchmarks/results/2026-08-07-gfx1151-maple-m6-batch-decode-recertified.json)

## Optimization review and next work

Clean cached-only profiles freeze every retained phase: P0 is the immutable P1
baseline, P1 is the immutable P2 baseline, and P2 remains the current prefill
kernel profile after P3's exact tile-16/32 screen regressed and direct BF16 WMMA
failed byte exactness. P4 recertifies the same kernels while extending SWA-wrap
orchestration and public admission. D0 supersedes the old c1 diagnostic with a
clean selector-unset profile; D1 supersedes the corrected pre-row-reuse c8
helper profile:

| Phase | Wall | Kernel | Host gap | Exact LM-head share |
| --- | ---: | ---: | ---: | ---: |
| native prefill320, post-P2 | **439.479 ms/request** | **431.666 ms** | **1.78%** | **0.31%** |
| c1 decode, post-D0 selector snapshot (trace process) | **5.018 ms/token** | **4.550 ms** | **9.32%** | **21.28%** |
| c8 helper decode, post-D1 | **19.296 ms/batch** | **17.305 ms** | **10.32%** | **21.58%** |

### P0 retained: sample only the final prefill row

`prefill_native()` now preserves every batched layer/KV update but executes
final RMSNorm, the exact 151,936-row LM head, and argmax only once on the final
row of the final chunk. The measured 320-token result is **649.280 tok/s**,
within 0.54% of the 645.811 tok/s profile projection and **98.82% faster** than
the corrected 326.573 tok/s baseline.

The public max-context-512 runner falls **5.133 -> 4.988 GiB**, an exact
**148.813-MiB** reduction from deleting the 256-row logit/argmax buffers. The
batch helper retains its separate all-row buffers because every active request
still needs a sampled result.

### P1 retained: true expert-major compact MoE

P1 replaces the row/route gather default with registered stable int32
count/prefix/scatter metadata plus expert-major ternary gate/up/down consumers.
Each consumer stages one expert/output weight row and writes directly back to
original row/route order, preserving the existing SwiGLU and weighted-combine
boundaries. The gather chain remains an explicit rollback.

Qualified throughput is **726.421/679.632/650.745 tok/s**, up
**3.68%/4.67%/5.83%** over P0, with all 18 state hashes and 90 positions exact.
The metadata owner is exactly **45,072 bytes**, moving tracked residency from
5,355,836,776 to **5,355,881,848 bytes** while close still reaches zero.

P1 does not meet the aggressive profile ceiling. The final diagnostic changes
the expert family only **276.150 -> 254.179 ms (1.086x)**; stable metadata costs
0.444 ms and exact 2-/4-lane schedules regress. The measured blocker is now the
scalar ternary unpack/dot/reduction, especially dual gate/up. Reaching the
original **<=97.708-ms** target requires a materially different exact non-WMMA
SIMD consumer rather than more sorting or grouped-lane geometry.

### P2 retained: exact wave32 GQA4 attention

P2 maps each four-query GQA group to one wave32 block, loads each K/V row once,
and emulates every local128 LDS stage plus weighted-value/FMA boundary exactly.
The clean profile cuts attention **63.993 -> 21.916 ms (2.920x)** at unchanged
730 launches and changes profile wall **478.176 -> 439.479 ms (1.088x)**.
Qualified 128/320/512 throughput is **749.175/741.368/754.000 tok/s**, up
**3.13%/9.08%/15.87%** over P1, with the complete state/position/lifecycle gate
exact and no additional persistent memory. Local128 remains the rollback.

### Prioritized exact roadmap

| Priority | Work | Measured rationale and gate |
| ---: | --- | --- |
| **P0 — DONE** | Sample only the final prompt row | Retained at 700.643/649.280/614.874 tok/s, 18/18 byte-exact state hashes, and 148.813 MiB lower residency. |
| **P1 — DONE / SCALAR BLOCKER** | True expert-major compact MoE | Retained at 726.421/679.632/650.745 tok/s and byte-exact; expert family improves 1.086x but misses the 2.826x ceiling, selecting a future exact non-WMMA SIMD ternary consumer rather than more grouping geometry. |
| **P2 — DONE** | GQA/query-row prefill attention | Retained at 749.175/741.368/754.000 tok/s and byte-exact; attention falls **63.993 -> 21.916 ms (2.920x)** with no memory or launch increase. |
| **P3 — DONE / REJECTED** | Retune dense ternary row tiles and test native BF16 WMMA | Tile 8/16/32 are bit-exact, but a counterbalanced natural+heldout screen measures **744.116/731.182/571.923 tok/s** and tile 16/32 lose all 16 pairs. Direct WMMA then changes **106/256 FP32** K16 partials and **43/655,360 BF16** production-shape outputs. All candidate surfaces are removed; tile 8 remains production. |
| **D0 — DONE** | Exact c1 kernel/host work, not graph promotion | The default router passes clean at **+4.14%**, exact wave32 affine4 passes at **+6.77%**, and per-token selector snapshotting improves fresh-process fixed-token A/B **200.279 -> 202.580 tok/s (+1.15%)** with all four candidate processes >201 tok/s. Current natural-context throughput is 153.201 tok/s with the complete exact gate. |
| **D1 — DONE** | c2/c4/c8 exact affine4 row reuse | Retained at **250.481/346.365/428.063 aggregate tok/s (+14.47%/+32.66%/+43.08%)** with exact full logits/trajectories, sparse/reclaimed slots, lifecycle, and a named cached trace; c1 remains on its proven wave32 kernel. |
| **P4 — DONE** | Safe long-prompt and public batch admission | Native prefill is exact at 520/770 tokens; public c1/c2/c4/c8 reaches **123.131/165.697/202.038/214.788 tok/s** with all 15 repeated category/heldout trajectory sets, sparse/staggered reclaim, singleton c8-owner preservation, and lifecycle exact. |

The 200+ decode target is therefore realistic in two distinct forms:

- **Approximate:** a properly quality-gated FlashHead path should directly
  attack the current 0.968-ms c1 exact-head bucket and is the mechanism behind
  DeepGrove's 218 tok/s M4 headline. It must remain opt-in and pass the full
  category/heldout agreement suite.
- **Exact:** the retained fixed-token A/B candidate reaches **202.580 tok/s**
  across four fresh processes, all individually above 201 tok/s. The separate
  trace process is **199.293 tok/s**, so 200 is protocol/noise-sensitive rather
  than a universal floor. The natural-context exact row is 153.201 tok/s.

Evidence:
[`P4 long prefill + public admission`](../benchmarks/results/2026-08-08-gfx1151-maple-p4-long-prefill-public-batch-retained.json),
[`D1 batched affine4 row reuse`](../benchmarks/results/2026-08-08-gfx1151-maple-d1-batched-affine4-rowreuse-retained.json),
[`D0 c1 router`](../benchmarks/results/2026-08-08-gfx1151-maple-d0-c1-router-retained.json),
[`D0 affine4 head`](../benchmarks/results/2026-08-08-gfx1151-maple-d0-affine4-wave32-retained.json),
[`D0 selector snapshot`](../benchmarks/results/2026-08-08-gfx1151-maple-d0-selector-snapshot-retained.json),
[`D0 wave32 profile`](../benchmarks/results/2026-08-08-gfx1151-maple-d0-wave32-decode-profile.json),
[`D0 pre-head profile`](../benchmarks/results/2026-08-08-gfx1151-maple-d0-decode-profile.json),
[`P2 GQA4 prefill`](../benchmarks/results/2026-08-08-gfx1151-maple-p2-gqa4-prefill-retained.json),
[`post-P2 phase profile`](../benchmarks/results/2026-08-08-gfx1151-maple-p2-phase-profile.json),
[`post-P1 phase profile`](../benchmarks/results/2026-08-07-gfx1151-maple-p1-phase-profile.json),
[`P1 expert-major prefill`](../benchmarks/results/2026-08-07-gfx1151-maple-p1-expert-major-prefill-retained.json),
[`P0 final-row prefill`](../benchmarks/results/2026-08-07-gfx1151-maple-p0-final-row-prefill-retained.json),
[`post-P0 phase profile`](../benchmarks/results/2026-08-07-gfx1151-maple-p0-phase-profile.json),
[`corrected pre-P0 phase profile`](../benchmarks/results/2026-08-07-gfx1151-maple-corrected-phase-profile.json),
[`c1 graph review`](../benchmarks/results/2026-08-07-gfx1151-maple-c1-graph-review.json),
and [`MAPLE-PERF.md`](MAPLE-PERF.md). Kernel names and trace resources are in
[`KERNELS.md`](KERNELS.md).

## Reproduction

Download the exact deployment revision into the normal Hugging Face cache:

```bash
hf download deepgrove/maple-preview-2bit-mlx \
  --revision 361db5da5e74ff6fcdd852d478e1f266ce11013a
```

Then run the qualified paths from the repository root:

```bash
# Focused runtime/correctness gates
python3 -m pytest tests/test_maple_runtime.py tests/test_maple_generation.py -q

# M5 public-native prefill recertification
GPU_MAX_HW_QUEUES=1 HIPENGINE_HIP_ARCH=gfx1151 \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt \
HIPENGINE_REQUIRE_CACHED_BUILD=1 python3 scripts/maple_prefill_bench.py \
  --model deepgrove/maple-preview-2bit-mlx --backend hip_gfx1151 \
  --suite benchmarks/prompts/mtpbench-code-general-ja.jsonl \
  --heldout benchmarks/prompts/gdn-prefill-category-heldouts.jsonl \
  --lengths 128,320,512 --repetitions 3 --warmups 1 \
  --continuation-steps 4 --out /tmp/maple-m5.json

# D0 exact c1 production-vs-rollback qualification
GPU_MAX_HW_QUEUES=1 HIPENGINE_HIP_ARCH=gfx1151 \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-maple-hipcc-version.txt \
HIPENGINE_REQUIRE_CACHED_BUILD=1 python3 scripts/maple_c1_bench.py \
  --model deepgrove/maple-preview-2bit-mlx --backend hip_gfx1151 \
  --suite benchmarks/prompts/mtpbench-code-general-ja.jsonl \
  --heldout benchmarks/prompts/gdn-prefill-category-heldouts.jsonl \
  --comparison router --steps 32 --warmup-steps 4 --repetitions 2 \
  --out /tmp/maple-d0-router.json

# D0 exact wave32 head vs group64 rollback qualification
GPU_MAX_HW_QUEUES=1 HIPENGINE_HIP_ARCH=gfx1151 \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-maple-hipcc-version.txt \
HIPENGINE_REQUIRE_CACHED_BUILD=1 python3 scripts/maple_c1_bench.py \
  --model deepgrove/maple-preview-2bit-mlx --backend hip_gfx1151 \
  --suite benchmarks/prompts/mtpbench-code-general-ja.jsonl \
  --heldout benchmarks/prompts/gdn-prefill-category-heldouts.jsonl \
  --comparison affine4_wave32 --steps 32 --warmup-steps 4 --repetitions 2 \
  --out /tmp/maple-d0-affine4.json

# M6+D1 selector-unset fixed-capacity helper recertification
GPU_MAX_HW_QUEUES=1 HIPENGINE_HIP_ARCH=gfx1151 \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-maple-hipcc-version.txt \
HIPENGINE_REQUIRE_CACHED_BUILD=1 python3 scripts/maple_batch_decode_bench.py \
  --model deepgrove/maple-preview-2bit-mlx --backend hip_gfx1151 \
  --suite benchmarks/prompts/mtpbench-code-general-ja.jsonl \
  --heldout benchmarks/prompts/gdn-prefill-category-heldouts.jsonl \
  --steps 64 --repetitions 3 --warmup-steps 8 \
  --natural-gate-steps 8 --out /tmp/maple-m6.json

# P4 public c1/c2/c4/c8 admission + singleton-c8 qualification
GPU_MAX_HW_QUEUES=1 HIPENGINE_HIP_ARCH=gfx1151 \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-maple-hipcc-version.txt \
HIPENGINE_REQUIRE_CACHED_BUILD=1 python3 scripts/maple_public_batch_bench.py \
  --model deepgrove/maple-preview-2bit-mlx --backend hip_gfx1151 \
  --suite benchmarks/prompts/mtpbench-code-general-ja.jsonl \
  --heldout benchmarks/prompts/gdn-prefill-category-heldouts.jsonl \
  --widths 1,2,4,8 --steps 64 --repetitions 3 --warmup-steps 2 \
  --single-active-capacity 8 --out /tmp/maple-p4-public.json
```

## Known limits

- Public sampling is greedy-only; temperature/top-p sampling is not implemented.
- Native prefill is performance-qualified at 128/320/512 and state-qualified at
  520/770; 128K runtime qualification remains separate.
- Public admission is fixed-capacity through c8. It uses sparse physical slots
  rather than K/V compaction, and prompt compute is per request rather than one
  ragged multi-request prefill GEMM.
- The submit/poll path is public in-process generation; an HTTP endpoint remains
  separate server integration work.
- FlashHead is approximate and excluded from the exact default.
- CUDA-peer support, speculative decoding, tensor parallelism, and 128K runtime
  qualification are separate future tracks.
