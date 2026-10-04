# The Qwen4Exp prefill gap to gufo is 5.2x, verified same-file and by wall clock, and uniform across families

Date: 2026-10-04
Host: AMD Radeon 8060S Graphics (gfx1151), 40 CUs @ 2900 MHz, AMD RYZEN AI MAX+ 395,
kernel `7.1.6-1-cachyos`, machine-id `55ea6c509d0b49eea8de7094a1023668`.
Peaks: 29.7 TFLOP/s FP32, 59.4 TFLOP/s FP16, ~256 GB/s LPDDR5X.

`2026-10-04-gfx1151-gufo-vs-hipengine-same-file` reports gufo prefill at
4.6x-7.0x hipEngine's on Qwen3.8-Flash-Next. That number drives the whole
prefill campaign, and it was recorded without checking whether gufo's `pp` is a
real full-prompt prefill or a cached-prefix "marginal" rate. This artifact
checks it. **It is real.** The target is 5.78x, not the ~1.7x the older
Vulkan-lane tables implied, and at matched granularity it is **5.77x and
uniform across every kernel family**.

**Tree provenance.** Every hipEngine row here was re-measured with `uv run`,
which imports `/home/lhl/hipEngine`. An earlier revision used
`python3 scripts/...`, which silently imports the separate
`/home/lhl/hipEngine-main` clone; see
`worklog/20261004T030644.952503Z-lhl-python-env-resolves-stale-clone-573911`.
The gufo rows are unaffected - gufo is its own prebuilt binary.

## Check 1: gufo's source

`/home/lhl/gufo-src/src/cli/bench/bench.cpp`, Qwen path, prompt loop:

```cpp
restore_depth();                        // outside the timed region
const auto t0 = std::chrono::high_resolution_clock::now();
prefill();                              // ForwardPromptBatch(prompt_tokens, depth)
                                        //   + hipDeviceSynchronize()
const auto t1 = std::chrono::high_resolution_clock::now();
runs.push_back(static_cast<double>(p_len) / elapsed_sec);
```

The timed region contains the whole `ForwardPromptBatch` for `p_len` tokens and
a device synchronize; `restore_depth()` runs before `t0`. The same file's
validation path cross-checks `ForwardPromptBatch` against per-token
`ForwardToken` under `[Prefill Validation]`. No cache reuse inside the window.

## Check 2: wall clock

`-p 4096 -n 0 -d 0 -r 2` is 3 prefill passes (1 warmup + 2 measured). At the
reported rate each is 2.40 s, so the wall should be load + 7.2 s.

```
$ gufo bench -m <shard1> -p 4096 -n 0 -d 0 -r 2 -v
[Model Load]: 45.2962 s
| Qwen3.8 Flash Next | 103.69 GiB | ROCm (HIP) | pp4096 | 1705.94 ± 0.85 |
WALL 53 s
```

53 s - 45.30 s = 7.7 s against 7.2 s predicted.

## The gap, as one protocol

Full 4096-token synchronized prefill, same host, same file, production profile,
cached build. Prompt token content differs (gufo cycles `(i%1000)+100`,
hipEngine uses the committed fixture) - a same-shape ratio, not a same-prompt one.

| engine | tree | chunk | 4096-token prefill | tok/s |
| --- | --- | ---: | ---: | ---: |
| gufo `98641a6` (prebuilt `8ca63994…`) | gufo binary | - | 2.40 s | **1705.94** |
| hipEngine production | this tree (`uv run`) | 4096 | 13.87 s | 295.40 |
| hipEngine production | this tree (`uv run`) | 1024 | 15.03 s | 272.46 |
| hipEngine production | `hipEngine-main` (stale) | 4096 | 12.42 s | 329.94 |
| hipEngine production | `hipEngine-main` (stale) | 1024 | 14.08 s | 290.47 |

**5.78x** at chunk 4096, against the in-tree rate. The `hipEngine-main` rows are
kept only to show the size of the tree error; they are not this tree's numbers.

In-tree the chunk effect is **+8.41%** (272.46 -> 295.40, three reps per arm,
and the chunk-1024 arm run twice agreeing to 0.12%), and **decode improves
+1.83%** (17.40 -> 17.72). The **-8.9% decode regression** reported from the
earlier chunk sweep came from the `hipEngine-main` arms (19.03 at chunk 1024)
and **does not exist in-tree**.

## The 27B control is a different architecture, not a dense control

Same-file same-host, Qwen3.8-27B `Q4_K_M` (from
`results/2026-10-04-gfx1151-gufo-vs-hipengine-same-file`):

| shape | gufo pp | hipEngine pp (per-shape) | hipEngine pp (1 session) |
| --- | ---: | ---: | ---: |
| 512 | 383.53 | 357.10 | 407.99 |
| 1K | 392.36 | 406.18 | 405.90 |
| 4K | 382.74 | 398.82 | 398.56 |

Within 4% at 1K/4K. But `Qwen3.8-27B-Q4_K_M.gguf` declares architecture
`qwen35` (65 dense layers) while Flash-Next declares `qwen4exp`: these are two
*model implementations*, not two expert counts. What the table establishes is
that the **qwen35 path is at parity with gufo**. It says nothing about whether
the qwen4exp dense projections are fast, and the trace below shows they are not.

An earlier revision of this artifact read that table as a dense-vs-MoE control
and concluded the gap was "MoE-specific". The matched-granularity trace below
refutes that, and the architecture field of the 27B file is why.

## Where gufo's time goes

`rocprofv3 --kernel-trace` over `gufo bench -p 4096 -n 0 -d 0 -r 1`
(raw: `/tmp/gufo-prof/raw_kernel_trace.csv`; gufo held 1675.38 tok/s under the
profiler):

- Total kernel time **4.84 s for 2 prefill runs = 2.42 s per prefill**, across
  **8,475 launches** total (4,237 per prefill).
- Trace span 50.35 s against 45.37 s of model load, so the prefill window is
  ~97% GPU-busy.
- MoE is **one fused family**, `RoutedF16GEMMKernel<WeightType, ...>` - gather,
  quantized-weight GEMM and scatter in one kernel - at **0.793 s per prefill
  across 192 launches** (4 per layer x 48 layers).

| kernel | sec (2 runs) | % | n |
| --- | ---: | ---: | ---: |
| `DenseF16GEMMKernel<256,128,2,8,1,1,false,true,false,false>` (attn proj out) | 0.607 | 12.5 | 144 |
| `RoutedF16GEMMKernel<7,128,48,2,false>` | 0.551 | 11.4 | 166 |
| `RoutedF16GEMMKernel<12,128,64,2,true>` | 0.482 | 9.9 | 90 |
| `RoutedF16GEMMKernel<12,128,128,2,true>` | 0.458 | 9.5 | 98 |
| `HcCombineMoeF16Kernel` | 0.325 | 6.7 | 188 |
| `GdnRowSplitKernel` | 0.307 | 6.3 | 144 |
| `DenseF16GEMMKernel<256,128,2,8,1,1,false,false,false,false>` | 0.276 | 5.7 | 144 |

## Matched granularity: one 4096-row pass, both sides traced

`qwen4exp_profile_gap.py --mode prefill --prefill-chunk-size 4096`
(raw: `/tmp/c4096-trace/raw/trace_kernel_trace.csv`). The trace holds **4
prefill runs**, confirmed structurally rather than assumed: 192 `router_logits`
launches / 4 = 48 = one per layer x 48 layers, and 144 GDN prefill launches / 4
= 36 = the 36 GDN layers.

| bucket | hipEngine s/run | % | n/run | gufo s/run | ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense projections | 6.085 | 43.56 | 854 | 0.878 | **6.93x** |
| routed MoE (main + repair + sched) | 4.429 | 31.71 | 331 | 0.793 | **5.59x** |
| attention | 1.094 | 7.83 | 108 | | |
| GDN | 0.846 | 6.05 | 108 | 0.749 | **4.61x** |
| router | 0.636 | 4.55 | 96 | | |
| other | 0.879 | 6.29 | 1268 | | |
| **total** | **13.969** | | **2766** | **2.42** | **5.77x** |

The last three hipEngine rows and gufo's remainder are not a like-for-like
split, so only the two named buckets carry a per-family ratio; the combined
"everything else" ratio is 4.55x.

Largest hipEngine families at chunk 4096:

| family | s/run | n/run | ms/launch |
| --- | ---: | ---: | ---: |
| `gguf_k_prefill_out_coltile_rowbatch_pipelined` | 2.737 | 134 | 20.4 |
| `q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4` | 2.490 | 96 | 25.9 |
| `gguf_q4_k_selected_dual_wmma_iu8_risk_prefill` (MoE gate/up) | 2.217 | 47 | 47.2 |
| `q5_1_selected_wmma_iu8_risk_prefill` (MoE down) | 1.208 | 43 | 28.1 |
| `qwen4_exp_gdn_prefill` | 0.790 | 36 | 21.9 |
| `qwen35_router_logits_f32_token_tile_dense_exact` | 0.618 | 48 | 12.9 |
| `q8_0_dense_wide` | 0.616 | 264 | 2.3 |
| `q5_1_selected_sparse_exact_repair_row_publish` (MoE down repair) | 0.531 | 43 | 12.3 |
| `dense_gemv` (decode-shaped, running during prefill) | 0.502 | 240 | 2.1 |

Three things fall out that were not visible at chunk 1024:

- **The gap is uniform.** Every bucket is 4.6x-6.9x off, so no single kernel is
  the answer. The widest bucket is the dense projections at 43.6% of prefill
  time, not MoE.
- **It is not launch count.** gufo issues 4,237 launches per prefill against our
  2,766 - **1.53x more**. The difference is per-kernel throughput, which
  refutes the "gufo fuses and we do not" reading of this artifact's first
  revision.
- **240 `dense_gemv` launches run during prefill**, 0.502 s/run (3.6%) -
  decode-shaped kernels on a prefill path.

The chunk-1024 trace
(`results/2026-10-04-qwen4exp-prefill-kernel-breakdown`) reads the same way at
coarser granularity: 16.34 s of kernel time and 21,424 launches per prefill,
with MoE gate/up + down + repair at 38.34% = 6.265 s.

## Derived efficiency on the same arithmetic

Per 4096-token prefill the routed MoE reads ~73 GB of quantized expert weights
and does ~21.8 TFLOP (10 of 512 experts active, `expert_feed_forward_length`
640, `hidden_size` 2560, 48 layers, plus shared expert and router). These are
config-arithmetic estimates, not hardware counters.

| | per-prefill | weight bytes/s | TFLOP/s | % FP16 peak |
| --- | ---: | ---: | ---: | ---: |
| gufo routed MoE | 0.793 s | ~92 GB/s | ~27.5 | ~46% |
| hipEngine MoE (chunk 1024) | 6.265 s | ~11.7 GB/s | ~3.5 | ~6% |

**8x apart on the same math on the same silicon.** gufo is at 36% of memory
bandwidth, we are at 5%. This ratio is for the routed-expert block specifically;
the matched-granularity table above is the overall reading.

## What this changes

- The prefill objective is a **5.2x kernel-throughput gap**, not the ~1.7x the
  older Vulkan-lane tables suggested. Chunk-size tuning and scheduling work are
  real but second-order against it.
- It is **uniform across families** (4.6x-6.9x), so there is no single kernel to
  port. The largest single share is the dense projections at 43.6% of prefill.
- It is **not launch count or scheduling** - gufo issues more launches than we
  do - so the work is per-kernel throughput, and a fused-kernel port is not by
  itself the answer.
- The **qwen35 path (27B) is at parity with gufo**; the qwen4exp path is where
  the work is. That is a different axis from dense-vs-MoE.

## Limits

- Prompt token content differs between engines.
- The per-family bucket boundaries are not identical across the two engines, so
  only the two named buckets and the total carry ratios.
- gufo is a prebuilt binary at `98641a6` with `source.patch` applied; `gufo-src`
  HEAD is one commit ahead. The binary is what was measured.
- TFLOP/s and GB/s are derived from config arithmetic, not hardware counters.
- One host, one model, one quant, one repetition count per engine.
