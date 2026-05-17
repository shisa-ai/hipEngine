# Lessons Learned

This file records hipEngine-specific debugging lessons that are likely to recur.
Keep entries compact, evidence-backed, and actionable. Parent-workspace kernel
R&D notes still belong in `~/amd-gpu-tuning/LESSONS-LEARNED.md`; this file is
for issues observed while integrating stable kernels into hipEngine runtime,
state, and gates.

## 2026-05-15 — Native prefill flakiness can hide in full-attention prefill softmax

### Symptom

After native compact/single-request prefill was enabled and grouped MoE library
loading was fixed, the parent 512/32 fixture became repeat-flaky:

- some runs matched serial resident prefill + decode;
- failing runs diverged after several decode tokens, often producing
  `[1739, 220, 16, 15, 15, 4, 220, 16, ...]` instead of the expected
  `[1739, 220, 16, 15, 15, 15, 15, 15, ...]`;
- failing full-logit gates showed `max_kl≈8.6-9.0` and top-1 agreement around
  `0.485`;
- `HIP_LAUNCH_BLOCKING=1` did not eliminate the flake.

### What did *not* cause it

Targeted probes ruled out several tempting explanations:

- session close/free ordering after removing accidental compiler delays;
- grouped MoE preload/on-demand behavior;
- c=1 MoE vs grouped compact MoE;
- linear-attention state update;
- full-attention KV append content;
- decode state after prefill.

### Localization method

Use targeted, state-family probes rather than guessing:

1. Bisect by `max_layers`; the first pass/fail hidden divergence appeared at
   layer 3, the first full-attention layer.
2. Compare pass/fail runs at that layer. Hidden input, Q/K/V/gate tensors, and
   appended BF16 KV cache were identical.
3. Re-launch `prefill_full_attention_gqa_gate_fp16` twice on identical inputs in
   the same session. The old wrapper produced different `gated_attn` outputs
   (`repeat max abs` roughly `0.05-0.39`).

That localized the nondeterminism to the full-attention prefill softmax kernel
launch, not to runtime state or MoE.

### Fix

`hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.hip` now launches
single-request and varlen prefill GQA gate kernels with a 64-thread block instead
of the old 256-thread block. The wrapper also allocates shared scratch as:

```cpp
max_context_len + threads
```

rather than `max_context_len * 2`, because the kernel needs one score slot per
context token plus one reduction slot per thread. The old formula could
under-allocate short varlen/compact rows.

Commit: `4f252cf kernel: stabilize native prefill attention`.

### Validation evidence

Commands run for the retained fix:

```bash
python3 -m py_compile hipengine/runtime/qwen35_paro_runner.py hipengine/runtime/qwen35_paro.py scripts/qwen35_native_prefill_fixture_gate.py
python3 scripts/smoke.py --mode qwen35-paged-attn-prefill-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt
python3 scripts/smoke.py --mode qwen35-paged-attn-prefill-varlen-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt
for i in $(seq 1 5); do python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/fixture-final-det-$i.json; done
for c in 2 4 8; do python3 scripts/qwen35_batch_packed_prefill_correctness.py --prompt-length 8 --max-layers 40 --batch-size $c --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached --json /tmp/packed-det-c$c.json; done
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/prefill-det-final-512.json
```

Results:

- native fixture gate passed 5/5 repeats;
- max KL stayed around `0.00553-0.00570`;
- top-1 agreement was `100%`;
- compact prompt8 gates still passed for c=2/4/8;
- 512/128 prefill measured `479.755 tok/s`, essentially flat vs the post-preload
  `482.057 tok/s` baseline.

### Checklist for similar bugs

When a native prefill correctness failure is flaky rather than consistently
wrong:

- compare repeated native runs, not only native vs serial;
- checkpoint final-row hidden after each layer to find the first divergent layer;
- at that layer, separately compare layer input, projected Q/K/V/gate, KV cache,
  attention output, MoE input, and MoE output;
- re-launch suspect kernels on identical inputs in the same session;
- verify shared/LDS sizing against both long rows and short varlen/compact rows;
- do not retain throughput improvements until the repeat fixture gate is stable.

--- 

## llama.cpp Vulkan review 

The llama.cpp Vulkan review changed the PARO work from "try random HIP
micro-kernels" to "identify the structural gap, then port the parts that fit
our model, quantization, and memory budget." The biggest retained gains after
that review were:

| Change | Evidence | Lesson |
| --- | ---: | --- |
| Generic non-MoE W4 pack8 layout replacement | 4K/4K decode `81.251 -> 96.494` tok/s (`1.188x`), no peak VRAM increase | Coalesced layout beat dot-intrinsic speculation. |
| Warp-cooperative paged split-K QK dot | 4K/4K decode `101.417 -> 113.603` tok/s (`1.120x`); 32K/128 `63.546 -> 84.289` (`1.326x`); 128K/128 `26.387 -> 42.629` (`1.616x`) | Grid occupancy was not enough; the hot per-token dot also had to be parallel inside the block. |
| Paged split-K address hoist | 32K/128 decode `84.289 -> 92.071` tok/s (`1.092x`); 128K/128 `42.629 -> 51.086` (`1.198x`), no peak VRAM increase | Once math is parallel, repeated page/address work in the V loop can dominate real long-context decode. |
| Grouped-GQA paged split-K producer | Default long-context path: 32K/128 `92.071 -> 102.383` tok/s (`1.112x`); 128K/128 `51.086 -> 56.722` (`1.110x`), no peak VRAM increase | For GQA models, stop rereading the same K/V stream once per Q head; reuse it across the Q-head group. |
| Post-GQA split-cap retune | 128K/128 `56.722 -> 63.738` tok/s (`1.124x`); context-tensor attention `0.803 -> 0.750 ms` default and `0.724 ms` best env; no peak VRAM increase | Retest "flat" geometry knobs after structural changes. A split cap rejected before GQA became correct after the producer grid changed. |
| Short-prefill MoE path split | Quality-correct 512/128 prefill `575.459 -> 1107.491` tok/s (`1.925x`) by using grouped-stacked W4 up to 1024 tokens; 2K stays on device-gather (`1405.727` vs `1197.998`) | If two paths trade places by shape, keep both behind an empirical threshold instead of forcing one universal implementation. |
| Compact WMMA prefill crossover | Compact WMMA vs GEMV-only: 64 tokens `839 vs 740` (`1.13x`), 128 `1285 vs 935` (`1.37x`), 512 `2537 vs 1244` (`2.04x`), 4K `2714 vs 1240` (`2.19x`); 32 tokens still favors GEMV (`498 vs 513`) | Re-measure dispatch thresholds after removing padding overhead; do not let an old negative WMMA result or an old all-size default survive a layout change. |
| Full compact graph/repack line from the post-review baseline to current | 4K/4K decode `71.006 -> 101.211` tok/s (`1.426x`), 4K/128 `76.315 -> 107.703`, 512/128 `81.021 -> 116.721` | Many small structural changes compound if correctness and memory are guarded. |
| Warm-start benchmark policy | 512 prefill `~458 -> ~1930` tok/s; 4K prefill `~2142 -> ~3255` tok/s | Cold first-request timing was mostly JIT/allocator warmup, not steady-state engine speed. |
| W8A16 `lm_head` replacement | 4K/4K decode `76.878 -> 81.251`, peak `22.012 -> 21.531 GiB` | Quantized replacement can improve speed and memory when it replaces BF16 residency. |
| Selected-MoE SiLU/down-rotation fusion | 4K/4K decode `96.624 -> 97.793`; isolated activation/rotation `18.384 -> 6.895 us` | Delete launches and intermediate tensors around hot c=1 glue. |
| Shared-expert W8A16 decode | 4K/4K decode `97.793 -> 98.932`, still under 24GB gate | Dense shared expert was worth quantizing for decode only. |
| Linear-attention A/B projection fusion | 4K/4K decode `98.932 -> 99.966` | Tiny projections are launch-bound; one dense GEMV beat "faster" W8A16 math. |
| Linear-attention QKV/Z pack8 fusion | 4K/4K decode `100.060 -> 100.801`, replay dispatch `939.4 -> 909.4/tok` | Launch fusion helps, but projection fusion without arithmetic/reuse changes is a sub-1% lever. |
| Full-attention Q/K pack8 fusion | 4K/4K decode `100.801 -> 101.211`, replay dispatch now `893.8/tok` | Reuse successful fusion shapes, but use longer traces for launch-count truth when graph capture setup is visible. |
| Full-attention Q/K/V pack8 fusion reject | Correct and graph-safe, but no decode win; diagnostic prefill wiring also regressed same-command prefill (`~905 -> 836/822` tok/s) | Do not widen a fused projection family unless it preserves kernel efficiency or adds real data reuse; graph replay makes pure launch-count wins very small. |
| Qwen RoPE fast-row invalidation | 4K prefill `3231.5` tok/s at `ac7292e` produced repeated `207862` tokens; quality-correct `6d02dc8/current` is `~1955` tok/s and routes to `~213` active experts/layer instead of `~24` | A wrong hidden-state path can look faster by collapsing MoE fanout. Measure route diversity and output sanity before treating prefill speed as real. |
| Corrected W8A8 comparator | Current W8A8 4K/4K is `1851.34 / 97.20`, not the old repeated-token fast row | A speed target with corrupted output is not a target. |

Current state after those keeps: PARO compact decode is `113.60 tok/s` at
4K/4K with `19.86 GiB` peak on the latest sustained warp-split run; the
short-tail retained best is `122.03 tok/s` at 4K/128 after the address-hoist
pass. Grouped GQA is now default for the guarded long-context split path, and
the post-GQA split-cap retune raises the 128K default to 512 splits. 32K/128
is `102.38 tok/s`, above the measured llama.cpp HIP and Vulkan decode-at-depth
rows; 128K/128 is now `63.74 tok/s`, above llama.cpp HIP and `98.9%` of
Vulkan. The sustained 4K/4K row is `116.9%` of the corrected W8A8 decode
comparator and `92.9%` of the llama.cpp Vulkan Q4 decode row, while using far
less memory than W8A8 and fitting the 24GB W4 promotion gate.

The quality-correct short-prefill rerun after the RoPE/RMSNorm gate changed
the prefill baseline. The retained short-path split now gives `1107.49`
prefill tok/s at 512/128. Follow-up MOE2 reconciliation showed the older
`~3.2K tok/s` 4K prefill rows were pre-Qwen-RoPE-fix and invalid: the fast
lineage generated repeated tokens and routed to far fewer experts. Current
quality-correct 4K prefill is the `~1.9K tok/s` class until a high-diversity
bulk W4 MoE prefill kernel replaces the per-active-expert dequant/matmul path.

---

## MOE Optimization 2 (PLAN-MOE2) Lessons

PLAN-MOE2 is the quality-correct PARO compact lane after the Qwen3.5 RoPE fix.
The important reset was not "we lost 3.2K prefill"; it was "the old 3.2K row
was doing much less real MoE work." After the reset, the retained optimization
loop moved the correct 4K/128 prefill path from the `~1.95K tok/s` class to a
repeated mean of `2692.658 tok/s`, while keeping decode near `114 tok/s` and
peak allocation in the `21.5-21.7 GiB` compact class.

Baseline terms used below:

- Clean harness baseline: 4K/128 `1922.884` prefill tok/s, `113.562` decode
  tok/s, `21.559 GiB` peak.
- MOE2 prefill-loop baseline: 4K/128 `1951.761` prefill tok/s.
- Current retained 4K/128 best: `2693.739` and `2691.578` prefill tok/s
  (mean `2692.658`), decode `114.065/114.047` tok/s, `21.671 GiB` peak.
- Current long-context attention row was measured after the grouped-GQA
  contiguous V-loop change, before the later pack-wise W4 dequant and
  grouped-combine reruns. Do not quote the later W4 gains as 32K/128 or
  128K/128 gains until those shapes are rerun.

Ordered by measured impact:

| Rank | Change or discovery | Measured impact | Lesson |
| ---: | --- | ---: | --- |
| 0 | Qwen RoPE quality reset | Old fast lineage: 4K/128 `3231.484` prefill tok/s but repeated `207862` tokens and only `~23.95` active experts/layer. Correct lineage: `~1955` prefill tok/s and `~213.33` active experts/layer. | A wrong hidden-state path can look faster by collapsing MoE fanout. Route diversity is a correctness signal, not just a profiling curiosity. |
| 1 | Total correct MOE2 4K prefill recovery | Loop baseline `1951.761 -> 2692.658` tok/s: `+740.897` tok/s, `+37.96%`. Versus clean harness `1922.884 -> 2692.658`: `+769.774` tok/s, `+40.03%`. Decode stayed neutral-to-slightly-positive (`113.562 -> 114.056` tok/s, `+0.44%`). | The path had real headroom after the reset. Most of it came from W4/MoE structure, not benchmark noise. |
| 2 | Pack-wise W4 dequant | `2299.089 -> 2617.186` prefill tok/s: `+318.097` tok/s, `+13.84%`; decode `114.389 -> 114.080` tok/s; peak unchanged at `21.535 GiB`. | Dequant one AWQ int32 pack per thread and emit eight contiguous columns. Do not launch one scalar output thread that rereads the same packed weight/zero eight times. |
| 3 | Grouped-device-gather cleanup stack | `1951.761 -> 2296.162` prefill tok/s: `+344.401` tok/s, `+17.65%`; decode stayed `~114 tok/s`; peak ended at `21.535 GiB`. | High-diversity MoE prefill was dominated by per-active-expert temporaries and glue launches around the same math. Deleting those costs compounded. |
| 4 | Native weighted index-add fuse | `2141.913 -> 2278.056` prefill tok/s: `+136.143` tok/s, `+6.36%`; peak `21.576 -> 21.550 GiB`. | Fuse `down.float()`, route-weight multiply, and `index_add_` into one native helper before tuning the helper's launch geometry. |
| 5 | Bulk activation and rotation family | `2005.122 -> 2136.602` prefill tok/s across bulk activation, fused SiLU/down-rotation, pair layout, and gate-buffer reuse: `+131.480` tok/s, `+6.56%`. | Per-active-expert SiLU/product/rotation launches are expensive even when the arithmetic is small. Batch the routed rows and keep activation scratch transient. |
| 6 | Grouped-GQA contiguous V-loop attention producer | 32K attention microbench `~0.268 -> 0.216267 ms` (`~19%` latency cut); 128K `~0.95-0.96 -> 0.775317 ms` (`~18-19%`). E2E decode: 32K/128 `94.795 -> 100.433` (`+5.638`, `+5.94%`); 128K/128 `52.712 -> 63.084` (`+10.372`, `+19.68%`). 4K/128 prefill only moved `2296.162 -> 2299.089` (`+0.13%`). | Long-context decode still had address/producer waste. Reuse contiguous physical block strides in the V loop, but keep token-offset writes if graph/scalar validation expects them. |
| 7 | Grouped combine/scatter collapse | Pack-wise dequant baseline `2617.186 -> 2672.604` mean after grouped atomic scatter and token-major non-atomic combine: `+55.418` tok/s, `+2.12%`; peak rose modestly to `21.671 GiB`. | Launch collapse and token-major combine are useful, but after the large W4 dequant win they are polish-scale. Keep them structural and stop when deltas fall into the 1% range. |
| 8 | AWQ pair-pack dequant follow-up | `2672.604 -> 2692.658` mean prefill tok/s: `+20.054` tok/s, `+0.75%`; decode `~113.79 -> 114.056`; peak unchanged at `21.671 GiB`. | After the one-pack dequant rewrite, mapping one thread to two adjacent packs is still a clean W4-layout keep, but it is polish-scale. |
| 9 | Micro cleanup below the launch-storm line | Token-index precompute `+1.21%`; token scratch reuse, in-place scaling, direct buffers, 2D grid, thread tuning, and native-dtype output were each `~0.2-0.5%`. | Small wins are worth keeping when correct and memory-neutral, but they are not a substitute for Path B/A or W4 tile work. Pivot after repeated sub-1% keeps. |

### WMMA GEMM: from "negative result" to mandatory correctness fix

A W4 AWQ WMMA GEMM kernel was implemented using RDNA3's
`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` intrinsic in a separate wave32
compilation unit. The kernel passes graph/scalar/tensorized validation and
uses FP32 accumulation throughout.

The original evaluation called WMMA a negative result based on this table:

| Shape | WMMA on | WMMA off (GEMV) | Delta |
| --- | ---: | ---: | ---: |
| 512/128 | 1216 tok/s | 1300 tok/s | **-6.5%** |
| 1K/128 | 2521 tok/s | 4509 tok/s | **-44.1%** |
| 4K/128 | 2785 tok/s | 4296 tok/s | **-35.2%** |

**This comparison was invalid.** Post-hoc audit of every saved JSON revealed
that the "WMMA off" column at 1K+ had `finite_prefill_logits: false` — the
GEMV path was producing 100% NaN logits. The "4509 tok/s" at 1K was the model
skipping real MoE work because NaN propagation collapsed expert routing.

The corrected comparison, gating on `finite_prefill_logits: true`:

| Shape | GEMV (finite=true) | WMMA (finite=true) | Delta |
| --- | ---: | ---: | ---: |
| 512/128 | ~1230 tok/s | ~2185 tok/s | **+78%** |
| 1K/128 | ~2521 tok/s | ~2619 tok/s | **+4%** |
| 4K/128 | ~2785 tok/s | ~2521 tok/s | **-9%** |

The key conclusion was not that the first WMMA implementation was fastest at
every shape; it was that correctness gates must come before speed comparisons.
The 512 result was already a standout, while the GEMV path produced NaN
non-deterministically (sometimes 512 was finite, sometimes not) because bf16
intermediate accumulation could overflow in certain patterns.

Later compact WMMA removed the padding overhead and changed the dispatch rule:
GEMV is still slightly better at 32 tokens, but compact WMMA is the recommended
grouped-stacked MoE prefill path from 64 tokens onward:

| Tokens | Compact WMMA | GEMV-only | Speedup |
| ---: | ---: | ---: | ---: |
| 32 | 498 | 513 | `0.97x` (GEMV) |
| 64 | 839 | 740 | `1.13x` |
| 128 | 1285 | 935 | `1.37x` |
| 512 | 2537 | 1244 | `2.04x` |
| 4K | 2714 | 1240 | `2.19x` |

Enable with `NANOVLLM_PARO_WMMA_GEMM=1`,
`NANOVLLM_PARO_WMMA_COMPACT=1`, and `NANOVLLM_PARO_WMMA_MIN_TOKENS=64`.

**Lesson**: the original "negative" result compared correct output (WMMA)
against NaN-corrupted fast paths (GEMV). The benchmark harness captured
`finite_prefill_logits` but did not surface it in the summary table or gate
the "Validation" column on it. A performance regression that's actually a
correctness fix looks exactly like a real regression if you only check speed.

**Companion discovery** (still valid): raising `GROUPED_STACKED_MAX_TOKENS`
from 1024 to 4096 was a real win. The grouped-stacked path is genuinely
faster than grouped-device-gather — this was confirmed with WMMA-on
finite-output runs.

### Gate benchmarks on correctness, not just speed (meta-lesson)

This is the second time a "fast" result turned out to be a correctness bug:

1. **Qwen RoPE fix** (rank 0 in the MOE2 table): old fast lineage at 3231
   tok/s was routing to ~24 experts/layer instead of ~213. Speed came from
   doing less MoE work due to collapsed fanout.
2. **GEMV NaN overflow**: GEMV "baseline" at 4509 tok/s (1K) had 100% NaN
   prefill logits. Speed came from NaN propagation collapsing MoE routing and
   skipping real computation.

Both were caught by signals the harness already captured (`generated_sample`
diversity, `finite_prefill_logits`) but did not gate on. Fix applied:

- `bench_paro_native_engine.py` now emits a loud `WARNING` to stderr when
  `finite_prefill_logits` is false, including NaN/Inf counts.
- `run_moe2_baselines.py` now includes a `Finite logits` column in the
  summary table, and `Validation` is `false` if either `finite_prefill_logits`
  or `decode_step_graph_validation` is false. A bold warning is appended to
  the summary when any shape has non-finite logits.

**Rule**: no performance number is meaningful unless `finite_prefill_logits`
is true. Any future sweep must gate on this before comparing speeds. A
speed gain that comes with `finite=false` is a correctness bug, not an
optimization.

The current `AGENTS.md` post-run gates are applicable to all retained PARO rows:
report `finite_prefill_logits`, graph/eager validation, generated-sample match,
`prefill_tok_s`, `decode_tok_s`, `wall_seconds`, peak allocation, allocation
after load, and the exact command/env. Tables that omit correctness and memory
context are only scratch notes, not promotion evidence.

### Expert-sequential GEMV for L2 cache locality (negative result)

Hypothesis: the MoE GEMV kernel at 512 tokens runs over roofline because
blocks from 256 experts interleave on the GPU, thrashing the 6 MB L2 cache.
Changing the grid from `(packs, total_rows)` to `(packs, num_experts)` with a
per-expert token loop should keep each expert's weights in L2 for all its
tokens.

The kernel was implemented (`gemv_awq_expert_seq_dual_pack8_kernel`,
`gemv_awq_expert_seq_pack8_kernel`) and validates correctly. Performance
(all finite=true, comparing against WMMA-on baseline):

| Shape | WMMA baseline | Expert-seq | Delta |
| --- | ---: | ---: | ---: |
| 512/128 | ~2185 tok/s | 890 tok/s | **-59%** |

Root causes of the regression:

1. RDNA3's block scheduler does not guarantee blocks with the same `blockIdx.y`
   run together. Blocks from different experts still interleave, so the
   hoped-for L2 locality does not materialize.
2. The per-expert token loop with `__syncthreads()` barriers serializes work
   that was previously parallel across blocks.
3. With 180K blocks (vs 2.88M in the original), the GPU has far less
   thread-level parallelism to hide memory latency.

**Lesson**: changing grid dimensions does not control block scheduling order on
RDNA3. To guarantee sequential expert processing, you need either cooperative
launch with device-wide sync, a persistent kernel with an atomic work queue,
or multiple kernel launches. The token-loop-per-block pattern is especially
harmful at larger batch sizes where the loop iteration count grows linearly.

### The reset was a correctness fix, not a performance target

The old `~3.2K` PARO 4K prefill row came from a pre-Qwen-RoPE-fix lineage.
It generated repeated tokens and routed to far fewer experts. That made the
MoE path faster for the wrong reason: fewer active experts meant fewer
per-expert dequant/materialize/matmul calls. After the RoPE fix, the same 4K
prompt activated roughly `213` experts/layer, so the grouped-device-gather path
had to execute the real high-diversity workload.

Future prefill rows need at least these gates before they become targets:

- graph/scalar/tensorized decode validation for the standard 4K/128 row;
- finite prefill logits and generated-sample sanity;
- route-diversity sanity when a row is unexpectedly fast;
- peak memory in the compact class unless the row is explicitly diagnostic;
- exact command and env recorded in `WORKLOG.md`.

If a row gets faster by collapsing active experts or changing the seed/final
token, treat it as a quality bug until proven otherwise.

### W4 pack-wise dequant was the biggest single retained win

The old `dequant_awq_pack8_kernel` used one thread per scalar output element.
That meant each group of eight adjacent output columns reread the same packed
AWQ weight and packed zero. The retained kernel changed the unit of work to one
AWQ int32 pack per thread:

- launch over `in_features * out_packed`;
- load `qweight_t` and `qzeros` once;
- unpack all eight nibbles with the existing AWQ lane order;
- read eight contiguous scales;
- write eight contiguous BF16/FP16 dequantized columns.

This did not remove the later `torch.matmul`, and it did not add persistent
expert weight residency. It only stopped doing duplicate dequant memory work
before the matmul. That was enough for the largest individual 4K/128 gain:
`2299.089 -> 2617.186` prefill tok/s (`+13.84%`). A follow-up pair-pack
variant mapped one thread to two adjacent AWQ int32 packs and moved
`2672.604 -> 2692.658` (`+0.75%`) after the grouped-combine work; this is a
good retained W4-layout polish, not a new bottleneck-class change.

Rule: for W4/AWQ pack8 prefill, inspect the producer layout before inventing
WMMA, dp4a, or LDS. If eight scalar threads consume one packed word, the first
fix is usually to make the packed word the work item.

### The grouped-device-gather path is a launch and temporary storm

The current correct path sorts lanes by expert, gathers hidden rows, then loops
on the host over active experts. With correct 4K routing, that is around
`213` active experts/layer. Each active expert still calls
`awq_linear_pack8_prefill()` for gate, up, and down; that helper dequantizes
the expert pack8 W4 weight and then calls `torch.matmul`. The shape is roughly
`213 experts * 3 projections` worth of per-layer host/framework work before
counting activation, rotation, and scatter.

The retained cleanup stack proved where the first layer of waste was:

- scratch reuse and token-index precompute removed repeated metadata
  allocation/division;
- bulk activation replaced per-expert `SiLU(gate) * up` launch patterns;
- fused activation plus down-rotation kept the routed activation in transient
  BF16 scratch;
- direct gate/up output buffers avoided per-expert output tensors followed by
  cat copies;
- native weighted index-add collapsed cast, route-weight multiply, and scatter.

Together these moved `1951.761 -> 2296.162` prefill tok/s (`+17.65%`) without
duplicate weights and without hurting decode. The largest member was native
weighted index-add, `2141.913 -> 2278.056` (`+6.36%`), followed by the bulk
activation family.

Rule: in high-diversity MoE prefill, optimize the routed-row dataflow before
micro-tuning the final helper. Repeated view/cat/cast/scatter work can be a
double-digit E2E prefill tax even when every individual operation looks small.

### Long-context attention work still belongs in MOE2, but keep the scope clear

The grouped-GQA contiguous V-loop change was not a 4K prefill win; it was a
long-context decode win. At 4K/128 it only moved `2296.162 -> 2299.089` prefill
tok/s (`+0.13%`). At 32K and 128K, the same address/producer cleanup mattered:
32K/128 decode improved `94.795 -> 100.433` tok/s (`+5.94%`), and 128K/128
decode improved `52.712 -> 63.084` tok/s (`+19.68%`).

One attempted shortcut failed: skipping token-offset writes for contiguous
chunks produced a 32K graph/scalar mismatch at token 99. Keeping the contiguous
V-loop address path while restoring unconditional token-offset writes preserved
the microbench win and restored validation.

Rule: long-context attention optimizations can be exact and valuable, but they
need the same graph/eager validation as MoE work. Address-path state that looks
debug-only may still be part of the graph/tensorized ABI.

### Path B cannot call kernels from kernels on HIP

The Path B feasibility audit corrected an important design assumption: HIP does
not support dynamic parallelism for this use. A persistent MoE kernel cannot
device-launch the existing `__global__` pack8, activation, or scatter kernels.
Path B must extract or rewrite device-callable inner loops inside one persistent
kernel.

Viable Path B shape:

- reuse existing router/grouping outputs (`counts`, `expert_start`,
  `sorted_lanes`, `sorted_weights`);
- launch a resident persistent grid with `blocks_per_group <= 96` on W7900;
- scan experts in-kernel and skip empty experts;
- extract a device `awq_pack8_dot8` helper with the current vec8 loop, AWQ
  nibble order, group scaling, and BF16/FP16 rounding;
- execute gate/up, barrier, SiLU/down-rotation, barrier, down plus weighted
  scatter, barrier;
- keep scratch in the current transient class (`total_lanes x 512` gate/up/mid
  BF16), with no duplicate expert weights.

Path B's test is simple: if launch elimination with scalar pack8 helpers cannot
move 4K/0 or 4K/128 by at least about `15%`, then launch count is not the main
remaining bottleneck.

### Path A is the fused W4-dequant GEMM escalation

Path A should be built only if Path B's scalar helper extraction is not enough,
or if Path B becomes nearly as large as implementing fused W4-dequant tiles
directly. The goal is to remove both costs that remain in the current path:

- per-active-expert host/framework launches;
- materializing dequantized expert weights before `torch.matmul`.

The concrete Path A target is a persistent expert loop with W4-dequant-to-BF16
or FP16 WMMA tiles for `M>=16` routed prefill work. It should reuse the current
stacked pack8 weights, qzeros, and scales; duplicate BF16/W4/WMMA expert
residency is not promotable. The prototype ladder should start with a
one-expert W4 WMMA GEMM microbench, then grouped gate/up, then full selected
MoE, then 4K/0 and 4K/128 E2E.

Promotion target: at least `2.5K tok/s` 4K/0 or equivalent `>=30%` recovery
from the clean `~1.95K` class, with 4K/128 validation green, decode floor
preserved, compact memory preserved, and no duplicate expert weights. The
pack-wise W4 dequant keep already exceeded the `2.5K` 4K/128 prefill component
target by a simpler layout fix, but Path A still matters if we want to avoid
full expert weight materialization and keep climbing toward the roofline.

### grouped_mm is not a shortcut to Path A (negative result)

`torch.nn.functional.grouped_mm` is available on ROCm 7.13 and functional —
it replaces N per-expert `torch.matmul` calls with a single grouped launch.
Phase 0 microbench showed 8× speedup on pre-dequanted BF16 weights. But
**when the full materialization path is measured, grouped_mm is 16% slower
E2E** (2743 → 2310 tok/s at 4K/128) and uses +1.6 GiB more peak memory.

Root cause: at M=19 per expert, individual matmuls (19×2048×512 = ~40 MFLOPs)
complete in microseconds and pipeline efficiently. With pre-dequanted weights
and no stacking overhead, per-expert matmul loop **ties** grouped_mm
(4.53 ms vs 4.59 ms). Launch overhead is not the bottleneck at this shape.
The regression comes from:

- `torch.stack` or `copy_` overhead: 3.3–4.2 ms to build the [E, K, N] buffer
- Allocator pressure from +850 MiB–1.7 GiB transient BF16 weight tensors
- These costs don't exist in the per-expert path

**Lesson**: microbench results that exclude the full data-movement pattern
(allocation, stacking, copies, cache state) can show wins that reverse at
E2E. The Phase 0 microbench measured "how fast is grouped_mm with pre-filled
weights?" — the real question was "how fast is dequant + materialize + stack +
grouped_mm vs the current pipeline?" Stage measurements in layers: matmul-only
with pre-dequanted weights → add dequant → add stacking → E2E. Each layer
reveals a different bottleneck. Skip one and you chase the wrong optimization.

**What remains viable**: a fused W4 grouped GEMM that never materializes BF16
weights to VRAM at all (reads pack8, dequants in registers, accumulates in
FP32, writes BF16 output only). That removes both the dequant launches AND the
~524 KiB BF16 weight roundtrip per expert. Also: the M crossover sweep at
16K–32K prefill (where M_per_expert climbs into the 100s) may find a regime
where grouped_mm wins on its own merits with a runtime heuristic gate.

See `docs/PATHA-FUSE.md` §13–§15 for the full breakdown.

### Bulk-first policy after the 2.6K recovery

The cleanup loop was successful, but the final deltas showed the boundary:
grouped atomic scatter was `+0.73%`, token-major combine was `+1.38%`, the
AWQ pair-pack dequant follow-up was `+0.75%`, and many other keeps were under
`0.5%`. Those are acceptable structural polishes, not a reason to keep
grinding wrapper/view/thread-count changes.

Future MOE2 loops should be bulk-first:

- choose a named lane that can move headline rows by several percent or change
  the bottleneck class: grouped-GQA producer, Path B persistent MoE dispatch,
  Path A fused W4 MoE GEMM, or W4 pack8 arithmetic/layout;
- allow micro-optimizations only when they unblock, de-risk, or clean up one of
  those lanes;
- pivot after three consecutive non-improving micro attempts, or when retained
  keeps fall into the `~1%` range while larger lanes remain open;
- keep exactness, route sanity, decode floor, and compact memory as hard gates.

---

## Highest-Impact Workflow Lessons

### Audit first, optimize second

The audit is the orientation tool that decides which kernel to touch and what
kind of fix to apply. Before we enforced this, we spent roughly a hundred
iterations micro-optimizing paged attention while it launched on only 16 of 96
CUs, and while the W8A16 linear family was the real majority decode bucket.

The audit asks four questions in this order:

- Which kernels dominate total time? Use `rocprofv3 --kernel-trace`, sum
  `DurationNs` by kernel, and rank.
- Is the hot kernel grid-undersubscribed? `Grid_Size / Workgroup_Size` should
  at least reach the CU count for broad kernels.
- Is the block actually parallel? `if (threadIdx.x == 0)` around a reduction,
  top-k, scan, or loop is a serious bug in a hot decode kernel.
- Are per-thread loops too short for codegen? For
  `for (k = threadIdx.x; k < N; k += blockDim.x)`, `N / blockDim.x < 64`
  usually needs manual vec8-style unrolling.
- Is sync density excessive? `__syncthreads()` inside a token/tile loop can
  poison GPU-wide scheduling even when the kernel itself is not the largest
  bucket.

Measured wins after switching to audit-first:

- Flash-decoding split-K paged attention: about `+17%` 4K/D4K E2E.
- Warp-cooperative paged split-K QK dot: `4K/128` same-build opt-out
  `106.974 -> 119.727` decode tok/s, `32K/128` `63.546 -> 84.289`, and
  `128K/128` `26.387 -> 42.629`.
- Paged split-K address hoist: `32K/128` `84.289 -> 92.071` and `128K/128`
  `42.629 -> 51.086` with unchanged 128K peak memory.
- Grouped-GQA paged split-K producer: default long-context `32K/128`
  `92.071 -> 102.383` and `128K/128` `51.086 -> 56.722`, with unchanged peak
  memory.
- Parallel router top-k: `5.7x` kernel speedup, `+6.6%` E2E.
- W8A16 vec4/vec8 unroll family: about `+54%` E2E in the W8A8 phase.
- Correct GDN barrier work initially appeared to give large gains, but the
  later state-corruption finding changed the lesson: remove barriers only when
  exact recurrent semantics are proven, not merely because timing improves.

### Fast rows are invalid until output sanity proves they are real

The old W8A8 "fast" rows repeated token 0 or relied on GDN no-sync state
corruption. The incorrect value outputs were literal wrong recurrent state, not
rounding noise.

The same pattern recurred in PARO prefill. The `ac7292e` 4K fast row was fast
because incorrect Qwen RoPE settings corrupted hidden states, collapsed MoE
fanout to roughly `24` active experts/layer, and produced repeated token
`207862`. After the `6d02dc8` RoPE fix, the same 4K prompt activates roughly
`213` experts/layer and the grouped-device-gather prefill path must do much
more real work. Existing simple recovery attempts (forcing grouped-stacked
through 4K, 1024-token MoE chunks, duplicated stacked layout, and Python-level
dual gate/up matmul fusion) all regressed. The correct recovery class is a
high-diversity bulk W4 MoE prefill kernel, not restoring the old row.

Retained rows need cheap sanity gates:

- first-token comparison against the known seed row;
- `decode_unique_tokens`;
- graph-vs-eager token agreement when graph replay is involved;
- oracle agreement where available, such as PARO `24/24`;
- state/logit/top-k overlap checks when recurrence or graph state changes.
- calibrated quality checks when promoting a model path, not just top-1:
  one 2026-05-11 smoke found native PARO and llama.cpp Q4 tokenization matched
  exactly on a `docs/ROOFLINE.md` chunk, while PARO PPL was `~1.97e6` versus
  llama.cpp Vulkan Q4_K_M PPL `9.054`. That cross-base number is not a valid
  quantization comparison, but it proves top-1 oracle agreement does not certify
  PPL/KL/logit calibration.
- keep a cheap native-engine logit gate in the same proof script:
  `proof_of_life_native.py --swap-moe` now runs an `n_ctx=128`
  HF-PARO-vs-native smoke that reports mean KL, top-1 agreement, PPL, and the
  first layer whose hidden state drifts. After the RMSNorm and RoPE fixes the
  retained smoke is KL `0.001944` / top-1 `95.24%`; the default gate is
  tightened to KL `<=0.05` and top-1 `>=0.90`.
- model-specific normalization semantics before native-layer promotion:
  Qwen3.5 normal RMSNorm stores an offset parameter and applies
  `norm(x) * (1 + weight)`, while the gated linear-attention norm uses a direct
  scale. The pure-native PARO loader initially used direct weights for both,
  causing layer-0 collapse and native PARO KLD `12.71` vs the HF PARO oracle.
  Materializing effective normal RMSNorm weights recovered the same slice to
  KLD `0.1916` and made layer 0 match HF at FP16-level error.
- model-specific RoPE semantics before full-attention promotion:
  Qwen3.5 full-attention uses partial RoPE (`partial_rotary_factor=0.25`,
  `rotary_dim=64` at `head_dim=256`) and `rope_theta=10000000`. Native PARO
  initially applied full-head theta-10000 RoPE; layer-3 q/k RoPE mean error was
  about `0.71`, which propagated to attention mean error `~0.142`. Carrying
  `rotary_dim` and `rope_theta` from `ModelSpec` reduced q/k RoPE mean error to
  `~2e-4` and the logit smoke to KL `0.001944`.

When these fail, debug persistent buffer/state invalidation and recurrence
math before trusting throughput.

### Long-context split-K needs within-block parallelism, not only enough CTAs

The 32K/128 and 128K/128 profiles showed paged full-attention dominating long
decode. The split-K path already had enough workgroups, but the device-context
kernel assigned each token's 256-wide QK dot to one thread. That made the grid
look populated while the important work inside each block was effectively
serial.

The retained fix adds a Qwen3.5-shape-guarded warp-cooperative split-K
context-tensor kernel: each wave computes one token dot using lane-local BF16
loads and a shuffle reduction, then the block keeps the existing chunk softmax
and value accumulation ABI. A follow-up cleanup reduces max/sum with
wave-partial reductions instead of a 256-thread LDS tree. It defaults on only
for the validated
`16 q-heads / 2 kv-heads / head_dim=256 / block_size=256` shape and can be
disabled with `NANOVLLM_AMD_PAGED_ATTN_WARP_SPLIT_CTX=0`.

Measured result:

| Shape | Before | After | Lift |
| --- | ---: | ---: | ---: |
| 4K/128 | 107.868 tok/s retained baseline; 106.974 same-build opt-out | 119.727 tok/s | `~1.12x` vs opt-out |
| 4K/4K | 101.417 tok/s | 113.603 tok/s | `1.120x` |
| 32K/128 | 63.546 tok/s | 84.289 tok/s | `1.326x` |
| 128K/128 | 26.387 tok/s | 42.629 tok/s | `1.616x` |

Rule: after fixing grid occupancy, audit the work distribution inside the
block. A split-K kernel can still leave a roofline-sized win on the table if a
single thread owns the dot/reduction for each token.

### Hoist repeated address work before redesigning attention

After the QK dot became warp-cooperative, the next 128K profile still showed
the context kernel dominating. The problem was not VGPR pressure or spills:
the hot kernel reported `VGPR=40`, `Scratch=0`, `LDS=0`, and abundant waves.
The remaining cheap waste was address work in the V loop: each output dimension
thread recomputed `logical_block`, `block_offset`, `block_table[...]`, and
`v_offset` for every token, repeating the same page lookup up to 256 times per
token in the Qwen3.5 shape.

The retained fix stores physical token offsets once during the QK pass, then
reuses them in V accumulation. It also adds a contiguous block-table fast path
for single-sequence serving while preserving the paged fallback.

Measured result:

| Shape | Before | After | Lift |
| --- | ---: | ---: | ---: |
| 4K/128 | 119.727 tok/s | 122.034 tok/s | `1.019x` |
| 32K/128 | 84.289 tok/s | 92.071 tok/s | `1.092x` |
| 128K/128 | 42.629 tok/s | 51.086 tok/s | `1.198x` |

Rule: before moving to a larger FlashAttention-style rewrite, inspect the
producer kernel for repeated page-table, stride, and offset calculations that
can be computed once per token/tile. On chiplet RDNA3, reducing scattered
address work can be an E2E win even when raw K/V bytes are unchanged.

### Exploit GQA reuse before changing attention semantics

Qwen3.5 has 16 Q heads and 2 KV heads, so each KV head feeds eight Q heads.
After address hoisting, the remaining exact-attention producer still scanned
the same K/V stream separately for each Q head. The grouped-GQA producer
changed the grid to `(kv_head, split)`, loaded each K/V vector once, computed
the eight Q-head score/output streams sharing that KV head, and preserved the
existing split partial ABI.

Measured result:

| Shape | Before | After | Lift |
| --- | ---: | ---: | ---: |
| 32K/128 | 92.071 tok/s | 102.383 tok/s | `1.112x` |
| 128K/128 | 51.086 tok/s | 56.722 tok/s | `1.110x` |

Rule: in GQA/MQA models, audit whether the kernel rereads K/V once per Q head.
If it does, a grouped producer can be a double-digit long-context win without
changing KV format or model semantics. Keep it shape-gated and fallback-safe:
the current PARO path defaults on only for the guarded long-context split
shape (`num_splits >= 64`) and can be disabled with
`NANOVLLM_AMD_PAGED_ATTN_GQA_GROUPED_CTX=0`. Earlier default-on/profiler
attempts stalled after source churn; clearing the HIP JIT cache before import
resolved the default serving path, so stale extension state remains a first
suspect when a fresh HIP edit hangs.

### Retest geometry knobs after structural producer changes

`NANOVLLM_AMD_PAGED_ATTN_MAX_SPLITS=512` was the right rejection before
grouped-GQA: it was flat to negative on the old per-Q-head producer. After the
producer grid changed to per-KV-head and each split did more useful grouped
work, the same cap became a clear 128K win.

Measured result after grouped-GQA:

| Shape | Before | After | Lift |
| --- | ---: | ---: | ---: |
| 128K context-tensor attention | `0.803 ms` | `0.750 ms` default (`0.724 ms` best env) | `1.07x-1.11x` |
| 128K/128 E2E decode | 56.722 tok/s | 63.738 tok/s | `1.124x` |

Rule: do not permanently discard shape/geometry knobs solely because they were
flat under an older kernel structure. Re-run the cheap sweeps after any change
that alters grid shape, per-block work, or split granularity. Keep the
promotion gate exactness-first: the split-cap 512 promotion passed the 128K
attention fixture and canonical proof before documentation.

### One-pass streaming needs a correctness fixture before E2E promotion

A grouped-GQA online-softmax prototype showed the right speed direction at
128K (`56.594 -> 62.641` decode tok/s with validation skipped), but failed the
exact graph/eager token check at 32K and even at `decode_len=1`. It was
reverted before commit.

Rule: FlashAttention-style rewrites change numerical order and are easy to
mistake for exact speedups. Before wiring another E2E path, build a small
attention-output/logit fixture that compares producer outputs, split partials,
top logits, and greedy tokens against the retained exact path. A fast
online-softmax row without that fixture is an approximate-attention candidate,
not an exact default.

### Warm-up is part of benchmark hygiene

PyTorch JIT extensions and the CUDA/HIP allocator make first-request timing
very different from steady-state timing. PARO 512 prefill reported about
`458 tok/s` cold and about `1930 tok/s` on the next identical pass. 4K prefill
similarly moved from about `2142` to about `3255 tok/s`.

Benchmarks that compare to llama-bench or server-like steady state should
warm kernels and allocator state by default. Keep a `--no-warmup` or explicit
cold-start row for cold-latency work, but do not mix cold and warm rows in the
same speed table.

### Re-audit after every structural change

Graph replay, layout replacement, packed/repacked weights, and fusion all move
the bottleneck. A successful change invalidates yesterday's target list. The
PARO profile moved from dense GEMV and W4 selected paths to generic pack8,
paged attention, rotations, shared expert, router/select, and tiny dense
projections as each layer of overhead was removed.

---

## Layout, Quantization, And Memory Lessons

### For W4 decode, layout comes before dot-product intrinsics

Naive Q8/dp4a or `sudot4` experiments over the wrong layout did not produce the
hoped-for win. The successful W4 change was coalesced pack8 replacement:
ordinary PARO W4 projections moved from the inherited AWQ layout to
`[out/8, in]` pack8 qweights, dropped the original layout, and kept prefill
correct through the pack8-prefill path.

The result was the biggest post-Vulkan code win: 4K/4K decode
`81.251 -> 96.494` tok/s with no peak VRAM increase.

Rule: make per-thread weight reads contiguous and keep activation preparation
cheap before trying dot4. A dot intrinsic cannot rescue strided loads, extra
quantization launches, or poor activation reuse.

### W4 residency is a product requirement

PARO is a W4 quantized 20GB-class model. A default path that needs 34-44 GiB is
a W7900 diagnostic, not a product path. The 24GB compact line was retained
because it replaced layouts instead of duplicating them and stayed around
`21.6 GiB` peak at 4K/4K.

Small persistent scratch buffers are acceptable when measured. Duplicate
weight layouts are different: they need exceptional evidence and must remain
opt-in unless they can be replaced by a compact layout.

### RDNA3 supports mixed `sudot4`, not portable signed `sdot4`

On gfx1100, `__builtin_amdgcn_sdot4` and `__builtin_amdgcn_udot4` require the
unavailable `dot1-insts` target feature. `__builtin_amdgcn_sudot4` uses
`dot8-insts` and does compile; this is the route llama.cpp HIP uses for RDNA3
Q4/Q8 MMVQ.

The practical rules:

- Signed-signed `sdot4` is not portable to W7900.
- Mixed signed/unsigned `sudot4` is available, but the signedness and bias/min
  fold must match the quantization math exactly.
- A compiling builtin is not proof of speed. Inspect ISA for real `v_dot4*`,
  `Scratch_Size == 0`, healthy VGPRs, and benchmark the exact shape.

### RDNA3 INT8 matrix math is not a 2x FP16 path

Both INT8 and FP16 WMMA on RDNA3 are 512 ops/cycle/CU. On W7900, measured
`4096^3` was about `84.8 TFLOPS` BF16 and `75.3 TOPS` INT8 through the tested
PyTorch path. W8A8 wins come from lower memory traffic and better residency,
not a guaranteed 2x compute advantage.

Related rule: RDNA3 has no native FP8 hardware. FP8 decode/dequant on gfx11 is
software bit manipulation, while INT8 has native dot and conversion paths. Use
INT8 for 8-bit KV/cache/storage work on W7900 unless targeting RDNA4+.

---

## llama.cpp Vulkan Lessons

### Do not explain Vulkan wins as backend magic

The useful Vulkan analysis came only after reading the actual llama.cpp HIP and
Vulkan source paths. For Q4_K_M decode on W7900, Vulkan is faster than llama.cpp
HIP at c=1 decode (`122.2` vs `87.6 tok/s` at 4K/4K), but the reason is not
"Vulkan has int-dot and HIP does not" and not "attention is faster."

What survived source checking:

- Both backends pre-quantize `src1` to Q8_1 and both reach RDNA3 dot4-style
  integer dot paths for this shape.
- The key difference is workgroup shape and reduction structure: Vulkan uses a
  64-thread, one-wave64, one-row shape with subgroup reduction; HIP MMVQ uses
  a 256-thread, 8-wave32 block for one row with LDS/barrier reduction.
- The HIP shape wastes many threads on small-K expert-down matvecs. For the
  Qwen3.6 A3B expert-down shape (`ncols=512`), most of the 256 HIP threads do
  no useful loop iterations.
- RADV/ACO appears to schedule this shader shape better than ROCm LLVM-AMDGPU.
  The `-amdgpu-unroll-threshold-local=600` experiment making llama.cpp HIP
  prefill much faster is direct evidence that compiler scheduling/codegen
  matters on gfx1100.
- Vulkan fuses more graph-level glue around MoE/top-k/post-ops. HIP had about
  1600 dispatches/token in the profiled path.
- Vulkan uses better activation-load packing (`block_q8_1_x4`) and less
  repeated address arithmetic in the matvec shader.

The transferable question is always: which structural delta can we port or
test in our runtime?

### What transferred to PARO/HIP

- Coalesced W4 pack8 replacement transferred directly and produced the largest
  post-review decode gain.
- Smaller shape-specific kernels transferred where the win was material:
  selected-down small-K, paged split-K thresholds, and tiny projection fusion.
- Graph/fusion thinking transferred: selected-MoE activation/down-rotation,
  full-attention gate-output fusion, shared expert W8A16, and A/B projection
  fusion all removed launches or intermediates.
- Benchmark policy transferred: use warm-up for llama-bench-style comparisons,
  but keep cold rows when first-request latency matters.

### What did not transfer cleanly

- A direct `sudot4`/Q8 rewrite did not win while layout and activation costs
  were wrong.
- LDS staging and persistent split-K workspaces did not help the current PARO
  path; they moved tiny allocations or added synchronization without improving
  kernel math/layout.
- Multi-step graph segmentation regressed decode; one-step reusable graph
  replay remains the retained graph shape.
- Allocation-only `_out` wrappers were mostly neutral under graph replay for
  both PARO and W8A8. Fusion or shared decode arenas are the right next shape,
  not per-module scratch everywhere.
- W8A16 was not a universal replacement. It won for `lm_head` and shared expert
  but lost on tiny `2048 -> 32` A/B projections.
- Removing recurrence barriers for speed was unsafe. The W8A8 no-sync rows
  were invalid because recurrent state was wrong.

### What hipfire added to the reference picture

The hipfire review was valuable because it is another RDNA3 inference engine
that makes different host/runtime choices. Its gfx1100 HFQ4 GEMV path does not
default to dp4a either: `gemv_dp4a` is enabled by default only for `gfx906`,
while the gfx1100 path uses 32-thread workgroups, `__launch_bounds__(32, 16)`,
packed `uint32_t` nibble loads, and four independent FP32 accumulators for
instruction-level parallelism. That independently corroborates our "layout and
reuse before dot intrinsics" rule.

The transferable hipfire ideas are boundary choices, not a wholesale runtime
port:

- Fuse quantization rotation at a layer/projection-family boundary where the
  rotated activation is reused, as in `fused_rmsnorm_mq_rotate.hip`; do not
  repeat the failed PARO pattern of recomputing a shared rotation per output
  pack.
- Fuse tiny DeltaNet beta/alpha projections into the QKV/Z preamble where
  rotation boundaries allow it, matching `fused_qkvza_hfq4g256.hip`.
- Use `attention_flash_asym3_tile.hip` and `kv_fold_asym3.hip` as source
  templates for a future streaming-attention/KV-quant lane. Asym3 changes KV
  precision and therefore belongs behind quality gates, like INT8 KV or
  Sparge-style sparse attention.

---

## Kernel Shape Rules

### Deep loop unrolling is a major RDNA3 GEMV lever

For loops like `for (k = threadIdx.x; k < N; k += blockDim.x)`, HIP/ROCm did
not automatically expose enough ILP when `N / blockDim.x` was small. Manual
vec8 unrolling was the largest W8A8 decode-era kernel win: vec4 across W8A16
kernels gave roughly `+42%` decode, vec8 added another `+8%`, and related
RMSNorm/GDN unrolls added a few more percent.

Use vec8 as the default hypothesis for simple dot/FMA loops with fewer than
about 64 iterations per thread. Keep a tail loop. Do not assume vec16 is worth
the complexity; our vec16 checks showed diminishing returns.

### WMMA is the wrong default for M=1 decode

WMMA tiles are 16x16x16. Single-token decode has M=1, so a naive WMMA GEMV
wastes most of the output tile. Scalar/vector FMA kernels won on c=1 decode
unless the shape was large enough to fill matrix tiles. Use WMMA for prefill or
batch/multi-token work; route c=1 decode separately.

### Runtime thread-count knobs must honor kernel launch bounds

Task 23's AWQ/GEMV decode audit found a correctness/hygiene bug rather than a
speed win: pack8 wrappers accepted `NANOVLLM_PARO_GEMV_PACK8_THREADS=256` and
`NANOVLLM_PARO_GEMV_SELECTED_PACK8_SMALL_K_THREADS=256`, but the pack8 kernels
are compiled with `__launch_bounds__(128, 4)`. The invalid env values caused
HIP unspecified launch failures during 512/128 decode sweeps. The retained fix
falls back to the existing 128/64 defaults when users request 256.

Rule: every thread/block-size env knob must be cross-checked against the
kernel's `__launch_bounds__`, statically allocated shared memory, and reduction
scratch size. Include at least one smoke for rejected/legacy env values when a
wrapper exposes a tuning knob; failing safely is better than treating an invalid
knob as a benchmark candidate.

### LDS is not free on RDNA3

RDNA3's cache hierarchy is often good enough that LDS staging loses once you
include synchronization and reduced parallelism. Rejected examples include
shared prefill scratch, shared hidden staging, paged-attention K tiling, dynamic
LDS pack8 reductions, and split-K partial workspace rewrites.

Only add LDS when the staged data is reused several times and the staged layout
eliminates a real scatter or uncoalesced-load pattern. Measure E2E, not just a
microbench.

### Vectorized loads plus scalar FMA are a strong baseline

The predictable fast baseline on RDNA3 is often vectorized 128-bit loads
feeding scalar FMA or simple integer dot, with no scratch spills and reasonable
VGPR pressure. Dot intrinsics and LDS are hypotheses to beat that baseline, not
automatic upgrades.

---

## Graph, Dispatch, And Fusion Lessons

### Shape-specific dispatch is worth it when the gap is material

There is no requirement to find one kernel that wins at 512, 4K, 32K, and
128K. Paged/split-K attention, selected-down small-K, short-prompt prefill, and
large-output GEMV all have different bottlenecks.

Keep both paths when the win is large and repeatable, record the threshold, and
avoid extra dispatch branches for tiny or noisy differences.

Full-attention decode is a current example. With graph replay and all quality
gates passing, forcing the direct context-tensor kernel at 4K/128 dropped decode
from `112.83` to `76.07` tok/s, while forcing paged at 512/128 dropped decode
from `115.57` to `113.29` tok/s. The existing
`NANOVLLM_PARO_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT=1024` threshold is the right
shape split: direct remains best at 512, paged/split-K is mandatory by 1K/4K.
This does not explain the remaining Vulkan short-context gap.

Router fusion is the opposite case: the separate top-k kernel is measurable, but
not all launch pairs can be fused safely. The router logits producer uses one
block per expert to keep c=1 grid occupancy high; top-k needs all expert logits
for a token. Without inter-block synchronization, a same-kernel fused top-k
would either be racy or collapse to one block per token, recreating the grid
undersubscription that earlier router work fixed. Prefer thread-count/selection
micro-tuning or a redesigned cooperative producer over a naive logits+top-k
fusion.

### Tiny c=1 projections are often launch-bound

The retained linear-attention A/B fusion won because it changed two
`2048 -> 32` dense GEMVs into one same-input GEMV. A W8A16 retry on the same
tiny shape was slower. For narrow decode projections, first ask whether adjacent
same-input projections or glue kernels can be fused before replacing the dot
loop.

Task 26's rotation/RMSNorm sweep reinforced that fusion only helps when the
fused path keeps the fast layout. `NANOVLLM_PARO_MOE_GATE_UP_ROTATE_FUSED=1`
removed a rotation launch in principle, but fell from `115.53` to `114.42`
decode tok/s because that path gives up the faster repacked/pack8 decode layout.
`NANOVLLM_PARO_BATCH_INPUT_ROTATION=0` regressed to `112.47` tok/s, and
disabling native RMSNorm fell to `105.05` tok/s. Do not promote a fusion that
saves a launch by moving onto a worse memory layout or graph-hostile fallback.

### Output buffers alone are rarely enough under graph replay

Generic W4 `_out` scratch, split-K persistent workspaces, and W8A8 unpacked MoE
`_out` all looked plausible and were correct, but did not improve E2E. Under
graph replay, some allocation cost is already captured or amortized. Output
buffers are worthwhile when they enable fusion, graph capture, or shared arena
ownership; isolated per-module scratch is usually not enough.

### Graph replay must include a state contract

Graph capture is not just launch replay. It must define which device tensors
are mutable, which pointers stay stable, how positions/tokens are updated, and
which recurrent/KV/MoE scratch buffers are invalidated. Repeated token 0 and
seed-token drift are graph/state bugs first, performance bugs second.

---

## Runtime And Library Lessons

### rocBLAS is currently faster than hipBLASLt on this W7900 stack

On W7900/gfx1100 with PyTorch `2.11.0+rocm7.13`, default rocBLAS beat
`TORCH_BLAS_PREFER_HIPBLASLT=1` on tested BF16 GEMM shapes:

| Shape | Default rocBLAS | hipBLASLt preferred |
| --- | ---: | ---: |
| `4096^3` BF16 | `84.8 TFLOPS` | `71.2 TFLOPS` |
| `8192^3` BF16 | `71.0 TFLOPS` | `51.7 TFLOPS` |

Do not cargo-cult Strix Halo or older ROCm guidance onto this W7900 stack.
Re-test when ROCm or PyTorch changes.

### JIT extension caches can silently lie

PyTorch JIT-compiled HIP extensions commonly cache `.so` files under
`~/.cache/torch_extensions/py*_rocm*/`, but project-specific `build_directory`
settings can put them elsewhere. PARO's `paroquant_kernels_v8` extension lives
under `~/.cache/nanovllm_amd/torch_extensions/paroquant_kernels_v8*/...`.
After C++ edit/revert cycles, stale caches can hang, silently cap performance,
or make a reverted kernel look active. A concrete PARO failure mode: the first
native RMSNorm or rotation call stalled indefinitely with GPU use at 0% until
the `~/.cache/nanovllm_amd/...` v8 cache was removed.

After HIP/C++ source churn, clear the matching cache:

```bash
rm -rf ~/.cache/torch_extensions/py*/nanovllm_amd_native_gfx1100_*
rm -rf ~/.cache/nanovllm_amd/torch_extensions/paroquant_kernels_v8*
```

### Deterministic algorithms can be a correctness tool, not a default speed path

`torch.use_deterministic_algorithms(True)` eliminated earlier bulk-torch
prefill + graph replay mismatches, but cost about 17% prefill speed in that
phase. Use it to isolate correctness and as a gate when needed; avoid carrying
it into speed rows unless the current path still needs it.

---

## Speculative Decode Lessons

Speculative decoding needs three ledgers, not one:

- acceptance economics: proposed tokens, accepted draft tokens, correction
  tokens, bonus/root tokens, and committed tokens per iteration;
- verified throughput: same-session AR tok/s, speculative tok/s, verifier time,
  drafter time, proposal/tree time, commit/restore time, and sync counts;
- fallback coverage: how much work silently ran as ordinary AR.

The failure mode we nearly missed: a path can look faster by becoming mostly
AR fallback. DFlash and MTP experiments showed useful harness/runtime lessons,
but acceptance alone did not prove speed. Promote speculative paths only when
verified tok/s improves and the ledger shows real accepted draft/tree work.

---

## How To Use This File

- Before touching a hot kernel, run the audit and identify the true time-share
  bucket.
- Before porting a trick from Vulkan, CUDA, vLLM, or llama.cpp, identify the
  exact structural delta: layout, workgroup shape, compiler path, fusion,
  activation prep, graph behavior, or memory residency.
- Before accepting a speed row, check output sanity.
- When adding a new lesson, put it in the highest-impact section it belongs to
  and include the measured evidence or artifact pointer in `WORKLOG.md`.
