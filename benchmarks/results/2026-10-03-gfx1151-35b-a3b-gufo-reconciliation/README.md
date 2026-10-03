# gfx1151 35B-A3B: gufo-method reconciliation, host power validation, AR stage attribution

Date: 2026-10-03
Harness: `scripts/qwen35_gguf_bench.py`, plus ad-hoc probes (`route_stage_ab.py`,
`clock_probe.py`, `decode_transient.py`, `chunk_sweep2.py`; not retained in-tree)
Model: `/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (22.66 GB; dense/attention
Q8_0, experts Q4_K/Q5_K, lm_head Q6_K)
Route: shipping AR (`use_wmma_prefill=True`, `use_gemv_decode=True`; harness
`--public-ar-profile`)
Host: AMD Radeon 8060S (integrated, **gfx1151**), Strix Halo, 32 cores,
`tuned` profile `accelerator-performance` / `performance` / EPP `performance`

## Question

Our published gfx1151 Qwen3.6-35B-A3B rows read 1430 tok/s prefill and
54.8 tok/s decode at 4K, against a gufo `Qwen3.6-35B-A3B-Q6dense` scoreboard of
3095 and 68.43 tok/s on the same GPU class. Is that a 2.16x prefill / 1.22x
decode engine deficit, or a protocol and file difference? And where does the
deficit actually live?

## What was measured

**Protocol reconciliation.** gufo's depth `d` is a cached conversation prefix
(a ~d-token user turn plus an 8-token reply) and the timed request is a new
~2048-token turn plus 128 output tokens. The prefix is never timed on either
side, so prefix caching changes what is *paid for*, not what is *measured*.
Their `pp` is new-tokens/s at depth `d`; ours is the average over a 0->D ramp.
Fitting our own 512/4K/32K/64K ladder (1369/1430/1145/936 tok/s) gives ~1380
tok/s marginal at depth 4096, so correcting to their basis widens the prefill
gap from 2.16x to ~2.24x rather than narrowing it.

**Host power state.** `tuned` was inactive and `scaling_governor=powersave`,
which DE-5 showed costs a 3.3x post-prefill decode transient (162 ms vs 49.3 ms
steady, all cores at the 2000 MHz base). After activating
`accelerator-performance` the transient is gone (first-30/last-30 step ratio
**0.996x**) and the published rows improve 1.5-2.0% prefill and 2.7-4.5% decode.

**Decode attribution.** Graph replay vs eager A/B, GPU-busy sampling, and a
GPU-side stage recorder on the shipping route.

**Prefill attribution and chunk sweep.** Shipping-route `rocprofv3
--kernel-trace` over two 4096-token prefills, a compute/memory basis built from
the GGUF tensor shapes, and a controlled chunk sweep.

## Results

### Host power state

50 ms `scaling_cur_freq` sampling across all 32 cores; "max core" is the
highest-frequency core in each sample.

| condition | max-core min/median/max MHz | % pairs at 2000 | % pairs >= 5 GHz |
| --- | --- | ---: | ---: |
| idle 5 s | 2000 / 4114 / 5139 | 87.7 | 1.5 |
| 32-thread CPU burn 5 s | 5103 / 5116 / 5124 | **0.0** | 50.0 |
| during 4K prefill | 2000 / 5111 / 5137 | 88.9 | 3.9 |
| idle 4 s after prefill | 2000 / 4339 / 5138 | 88.2 | 1.6 |

**Methodology correction to DE-5.** 2000 MHz is this APU's *idle parking*
frequency, not a load clamp: under a 32-thread burn **0.0%** of core-samples sit
at base and cores reach 5.1 GHz. "% of core-samples at base" is therefore
misleading; the max-core statistic is the one that shows a package clamp.

Post-prefill decode, 96 steps: per-eighth medians
`17.8 17.7 17.7 17.7 17.7 17.7 17.7 18.2` ms, first-30 17.8 vs last-30 17.8
(**0.996x**), only step 0 elevated (25.68 ms). Host max-core during decode:
median 5124 MHz, 0.4% of samples below 3000 MHz. Against DE-5's 3.3x.

Harness rows, 3 measured runs:

| shape | prefill tok/s (published -> now) | decode tok/s (published -> now) |
| --- | --- | --- |
| 512/128 | 1369.49 -> **1390.25** (+1.5%) | 54.33 -> **56.78** (+4.5%) |
| 4K/128 | 1430.21 -> **1459.48** (+2.0%) | 54.80 -> **56.29** (+2.7%) |

### Decode is kernel-bound

Graph replay vs eager, 4K/128: **56.312 vs 56.293 tok/s** — neutral. Transport
`hipgraph`, `effective_graph_replay_decode` True.

Route x recorder 2x2, 4K/128:

| route | recorder | tok/s | ms/step | GPU busy |
| --- | --- | ---: | ---: | ---: |
| shipping | off | 56.16 | 17.806 | 99% |
| shipping | on | 52.59 | 19.014 | 99% |
| default | off | 34.69 | 28.831 | 72.8% |
| default | on | 52.63 | 19.000 | 95.6% |

99% GPU-busy plus neutral graph replay rules out host-dispatch-bound decode. The
recorder costs 6.8% on the shipping route, so stage times are scaled by
17.806/19.014.

Decode budget, 17.81 ms/step, exact GGUF bytes at top-8/256 (2.446 GiB/token):

| stage | MiB/token | ms/step | GB/s | % of 237 GB/s |
| --- | ---: | ---: | ---: | ---: |
| decode_linear_attn_qkv_gate | 765.0 | 4.032 | 199.0 | 84% |
| decode_ffn_moe_selected | 597.8 | 3.678 | 170.4 | 72% |
| decode_lm_head_sample | 397.9 | 1.757 | 237.4 | 100% |
| decode_linear_attn_ssm_out | 255.0 | 1.448 | 184.7 | 78% |
| decode_full_attn_qkv_head_norm_rope | 210.4 | 1.157 | 190.7 | 80% |
| decode_ffn_moe_shared_gate_up | 87.1 | 0.887 | 103.0 | 43% |
| decode_full_attn_core_gate | 0 | 0.848 | compute | - |
| decode_ffn_moe_router | 81.0 | 0.716 | 118.6 | 50% |
| decode_linear_attn_gdn | 0 | 0.603 | compute | - |
| decode_full_attn_output | 93.5 | 0.491 | 199.6 | 84% |
| decode_ffn_moe_shared_down | 43.6 | 0.465 | 98.2 | 41% |
| decode_ffn_moe_combine | 0 | 0.438 | compute | - |
| decode_ffn_post_norm_residual | 0 | 0.363 | compute | - |
| decode_linear_attn_alpha_beta | 15.0 | 0.284 | 55.4 | 23% |
| decode_linear_attn_conv | 3.8 | 0.184 | 21.4 | 9% |
| remaining 5 stages | 0 | 0.247 | compute | - |

Aggregate 150.1 GB/s of a 237 GB/s ceiling. The four largest weight streams run
at 84-100% of ceiling; the addressable remainder is ~4.35 ms of 17.81 ms (24%):
compute-only stages 2.48 ms, MoE selected 0.54, MoE shared-expert pair 0.67,
router 0.29, small F32 stages 0.37.

### Prefill is a compute-efficiency deficit

Shipping-route trace over two 4096-token prefills: kernel sum 5528 ms for 8192
tokens = 0.675 ms/token against 0.685 ms/token unprofiled = **98.5% kernel
time**. Per token: MoE expert WMMA 0.256 ms (38%), dense Q8_0 WMMA 0.171 (25%),
GDN 0.135 (20%), router 0.031, AOTriton 0.030.

Compute basis from the GGUF shapes: 2997.4 M MACs/token = 5.995 GFLOP/token. At
1446.87 tok/s that is **8.68 TFLOP/s = 14.6% of the 59.4 TFLOP/s FP16 peak**
(40 CU x 512 FLOP/cycle x 2.9 GHz). Memory floor at chunk 512 is 0.187 ms/token,
compute floor 0.101 ms/token, so the step is 3.7x above memory and 6.8x above
compute.

Chunk sweep, 4096 tokens, 3 reps, 3 s settle before each timed prefill, one
session per size with `reset()` between reps:

| linear/MoE chunk | median ms | tok/s | vs 512 |
| ---: | ---: | ---: | ---: |
| 512 (shipping) | 2984 | 1372.6 | - |
| **1024** | **2805** | **1460.5** | **+6.40%** |
| 2048 | 2900 | 1412.6 | +2.91% |

### Dispatch defect

`GGUF_PREFILL_CHUNK_SIZES_BY_GEOMETRY` rewrites linear/MoE chunks in the session
constructor, after the resolver and after any explicit `PrefillConfig`:

* `PrefillConfig(linear_chunk_size=1024, moe_chunk_size=1024)` yields effective
  `(512, 512)`, so `--prefill-linear-chunk-size` / `--prefill-moe-chunk-size`
  are silently ignored for this geometry.
* `prefill_chunk_tuning.chunk_sizes` reports `{linear: 1024, moe: 1024}` while
  the effective `prefill_chunk_sizes` is `{linear: 512, moe: 512}` — the recorded
  tuning metadata does not describe the run.
* `_ARCH_CHUNK_PROFILES["gfx1151"]` (256/256) is unreachable on the GGUF path
  because `resolve_prefill_config_for_sequence` is called without `target_arch`
  (`target_arch: null`).

### External reconciliation

| | gufo Q6dense | hipEngine UD-Q4_K_M |
| --- | --- | --- |
| file | 22.4 GB, dense Q6_K, experts Q4_K/Q5_K, lm_head Q6_K | 22.66 GB, dense Q8_0, experts Q4_K/Q5_K, lm_head Q6_K |
| bytes/token (top-8/256) | 2.126 GiB | 2.446 GiB |
| decode | 68.43 tok/s = 156.2 GB/s | 56.29 tok/s = 147.8 GB/s |
| prefill | 3095 tok/s | 1459 tok/s |
| workload | pp2048/tg128 on a cached prefix | pp{512,4K,32K,64K}/tg128, no prefix |
| KV | F16 prefill + int8 decode copy | BF16 (int8 available, admission-gated) |

llama.cpp on gufo's own file: 55.69 tok/s = 127.1 GB/s.

The 1.22x raw decode gap decomposes into ~1.15x file (their quant streams 13%
fewer bytes/token) and ~1.06x achieved bandwidth.

## Artifacts

| file | contents |
| --- | --- |
| `host-clock-probe.json` | per-condition clock distributions, idle / CPU burn / prefill / post-prefill |
| `decode-transient.json` | 96-step decode series and host max-core during decode |
| `decode-route-stage-ab.json` | route x recorder 2x2 with full stage attribution |
| `decode-graph-replay-ab-4k.json` | graph replay vs eager, 4K/128 |
| `prefill-chunk-sweep-4k.json` | 512/1024/2048 chunk sweep with per-rep clocks |
| `harness-512.json`, `harness-4k.json` | harness summaries under the fixed host state |

## Limits

One host, one model, one quant, one profile, one prompt family. The 237 GB/s
ceiling is inferred from our own `decode_lm_head_sample` stage, not from an
isolated bandwidth microbenchmark. The marginal-prefill figure (~1380 tok/s at
depth 4096) is a fit to our published ladder, not a measurement of gufo's
protocol; measuring that protocol directly is a separate task. The chunk-sweep
ordering was measured with the clock clamp controlled but not eliminated — all
arms ran with the same idle-parking distribution and the same ~5.11 GHz max
core.
