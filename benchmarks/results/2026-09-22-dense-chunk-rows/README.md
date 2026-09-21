# Chunk-size sensitivity of the wide Q8_0 dense prefill route

The prefill protocol runs at a 1024-token chunk (`PrefillConfig.linear_chunk_size`
and `moe_chunk_size`, both 1024; `qwen4exp_profile_gap.py --prefill-chunk-size
1024` mirrors that default rather than overriding it). A `code-p4096` prefill is
therefore four chunk launches of every dense projection in the route.

This packet asks whether those four launches cost more than one. They do:
**1382.2 ms at 1024-row chunks, 1070.4 ms at 2048, 1057.2 ms at 4096 — a 325 ms
saving (23.5%) for identical arithmetic.**

| Chunk rows | Chunks per 4096-token prefill | Route f32in total | vs 1024 |
| ---: | ---: | ---: | ---: |
| 1024 | 4 | 1382.2 ms | — |
| 2048 | 2 | 1070.4 ms | −22.6% |
| 4096 | 1 | 1057.2 ms | −23.5% |

The gain is almost entirely the first halving: per-launch cost is strongly
sub-linear in rows (a launch at 4096 rows costs 53.5 ms against 4 x 16.7 =
67.0 ms for the same work), and it is close to flat between 2048 and 4096, so
there is little to gain past 2048 and the remaining 2x would cost 4x the
activation scratch.

Per-role, at one launch per role (median of 16 event-timed launches,
`--weight-buffers 8` so each launch reads weights the previous launches
evicted). The last column is what four 1024-row launches cost for the same work:

| Role | 1024 rows | 2048 rows | 4096 rows | 4 x 1024 rows |
| --- | ---: | ---: | ---: | ---: |
| `attn_qkv` | 2.644 | 5.270 | 10.354 | 10.574 |
| `attn_q` | 2.816 | 5.332 | 10.435 | 11.264 |
| `attn_gate` | 1.608 | 2.933 | 5.801 | 6.432 |
| `attn_output` | 2.400 | 4.104 | 7.513 | 9.599 |
| `ssm_out` | 2.379 | 4.125 | 7.430 | 9.514 |
| `hc_attn_down` | 1.860 | 2.093 | 4.147 | 7.440 |
| `shared_gate` | 0.256 | 0.271 | 0.726 | 1.023 |
| `shared_down` | 0.158 | 0.289 | 0.728 | 0.631 |
| `attn_k` | 0.248 | 0.259 | 0.712 | 0.991 |
| `attn_v` | 0.249 | 0.258 | 0.737 | 0.998 |

Every role is sub-linear in rows, and the small-weight roles gain the most in
relative terms because the launch itself dominates their 1024-row cost:
`shared_gate` costs 1.024 ms as four launches and 0.726 ms as one, `attn_k`
0.992 against 0.712. The two hyper-connection roles are the largest single
saving at 3.29 ms each per layer (6.59 ms for the pair) and `ssm_out` the next
at 2.08 ms.

## Why this matters beyond the dense route

The routed-MoE main kernels have a *structural* reason to prefer larger chunks
that the dense route does not: their weight traffic is
`sum over experts of ceil(rows_in_expert/16) x bytes_per_tile`, measured at a
constant 104-126 GB/s in
[`../2026-09-22-moe-main-kernel-cost/`](../2026-09-22-moe-main-kernel-cost/README.md).
Per-expert rows grow with the chunk, so the ceil() rounding costs less per row:
the engine's four 1024-token chunk launches carry `grid_y = 1120` tiles each
(4480 tiles per layer), where one 4096-token chunk launch carries 2801. At the
engine's own measured per-launch bandwidth (143 GB/s gate/up, 160 GB/s down)
that is 57.7 -> 36.1 ms gate/up and 34.5 -> 21.6 ms down per layer, i.e.
**1.0 s + 0.6 s = 1.6 s** of a 16.67 s prefill.

That MoE figure is arithmetic on measured per-launch times and the measured
traffic model, not an engine measurement at chunk 4096. Together with this
packet's measured dense saving it puts the chunk lever at **1.7-1.9 s, with no
kernel change at all** — larger than any kernel rewrite currently queued.

## Protocol

```bash
ENV_PREFIX=/home/lhl/miniforge3/envs/therock10-staging-20260828
PY=$ENV_PREFIX/bin/python
SITE=$ENV_PREFIX/lib/python3.12/site-packages
export PATH="$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$SITE/_rocm_sdk_core/lib:$SITE/_rocm_sdk_devel/lib:$SITE/_rocm_sdk_libraries/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HIPENGINE_HIP_ARCH=gfx1151

$PY scripts/qwen4exp_dense_wide_f16_activation.py \
  --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --rows 1024,2048,4096 --repetitions 16 --weight-buffers 8 \
  --output benchmarks/results/2026-09-22-dense-chunk-rows/artifact.json
```

- Host: AMD Radeon 8060S (`gfx1151`), Strix Halo, 125 GB unified memory.
- The route's per-launch times were validated against the engine's own
  role-marked capture to 1.4% in
  [`../2026-09-17-q8-dense-f16-activation/`](../2026-09-17-q8-dense-f16-activation/README.md),
  so the per-launch rows here are engine-representative.
- The three-row totals are the packet's own per-role launch counts
  (`aggregate.per_role`, a 4-chunk prefill at rows=1024) re-weighted by chunk
  count. They are an aggregate of measured per-launch times, not an engine wall
  time.

## What this does not say

- **Not an engine measurement.** No prefill was run at chunk 2048 or 4096. The
  totals are launch-count arithmetic over measured per-launch times.
- **Not free.** A 4096-row chunk needs roughly 4x the prefill activation
  scratch the 1024-row chunk needs. That has not been capacity-probed, and it is
  the first thing to check before attempting the change.
- **Not a global default.** `PrefillConfig`'s 1024 is shared by other lanes, and
  the dense H5120 `MOSTLY_Q4_K_M` geometry has a recorded preference for
  1024-row chunks over 4096 at capacities above 8K. Any change here has to be
  lane-scoped and validated per lane.
- **Not a numerical claim.** Chunk size does not change the per-row arithmetic
  of these projections (the K-loop order per output row is unchanged), so
  identical digests are expected, but that has not been verified by an A/B.
