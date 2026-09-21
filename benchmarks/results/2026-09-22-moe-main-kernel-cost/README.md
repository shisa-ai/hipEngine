# What sets the cost of the routed-MoE main kernels

The gate/up main kernel is the largest single owner in a `code-p4096` prefill
(2710.3 ms of 16673.2 ms) and the down kernel is second (1484.5 ms). Together
they are 25% of the prefill, and both run at a small fraction of the host's
memory roofline: gate/up reads about 2.06 GB of expert weights per launch at
142 GB/s while the host sustains roughly 200 GB/s.

This packet measures what actually sets that cost, on the shipped kernels at the
shipped geometry (512 experts, hidden 2560, ffn 640), and the answer is a single
number: **the number of 16-row tiles the experts' rows are cut into**. The
kernels are weight-traffic-bound at a constant bandwidth, so cost is
`tiles x expert_weight_bytes` and nothing else — not the row count, not the
padding ratio on its own, not the launch geometry.

## The mechanism

Each block owns one 16-row WMMA tile and one 128-column slice, so it reads that
expert's whole `128 x K` weight slice however many rows the tile actually has.
An expert holding 20 rows therefore costs **two** weight reads, not one, and an
expert holding 5 rows costs one. Total traffic per launch is

```
tiles = sum over experts of ceil(rows_in_expert / 16)
bytes = tiles x out_features_total x row_bytes
```

The engine's own capture agrees with that model. Its gate/up launches have
`grid_y = 1120` tiles (`wmma_total_rows = 17920` for 10240 compact rows, a
padding ratio of 1.75 on real routing) and take 14.5 ms; the model says
1120 x 1280 x 1440 B = 2.06 GB, which is **142 GB/s**. The packet's single
10240-row launch, at 109 GB/s, takes 14.8 ms against the engine's 14.4 ms per
launch, and the down kernel 8.9 against 8.6 ms — the packet reproduces the
engine to 2-3% on the same kernel and geometry.

## Measured: row count does not set the cost

`--rows` sweep, `balanced` routing, median of 15 event-timed launches:

| Rows | Rows/expert | Padding | Tiles | Gate/up ms | GB/s | Down ms | GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1280 | 2.5 | 6.40 | 512 | 8.65 | 109 | 5.53 | 114 |
| 2560 | 5.0 | 3.20 | 512 | 8.84 | 107 | 5.78 | 109 |
| 5120 | 10.0 | 1.60 | 512 | 9.11 | 104 | 5.87 | 107 |
| 10240 | 20.0 | 1.34 | 857 | 14.82 | 107 | 8.90 | 118 |
| 20480 | 40.0 | 1.19 | 1525 | 25.45 | 111 | 15.12 | 124 |
| 40960 | 80.0 | 1.09 | 2801 | 45.39 | 114 | 27.31 | 126 |

Eight times the rows cost 5.2x the time, because the tile count only grows from
512 to 2801: below 16 rows per expert every expert needs exactly one tile no
matter how few rows it holds, so a 1280-row launch pays for the same 512 tiles
as a 5120-row one. A gate/up tile reads 1.843 MB (1280 output columns at
1440 B per row), so those 512 tiles are 943.7 MB of weight traffic and the
10240-row launch's 857 tiles are 1.579 GB. The achieved bandwidth sits between 104 and 126 GB/s at every
size, which is what "bound by weight traffic" looks like.

## Measured: routing changes the cost only through the tile count

Same 10240 rows, four routings:

| Routing | Active experts | Rows/expert | Padding | Tiles | Gate/up ms | GB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `balanced` (uneven, all hit) | 512 | 20.0 | 1.34 | 857 | 14.80 | 107 |
| `uniform` (exactly 20 each) | 512 | 20.0 | 1.60 | 1024 | 17.26 | 109 |
| `sparse` (a quarter of experts) | 128 | 80.0 | 1.00 | 640 | 11.05 | 107 |
| `single` (one expert) | 1 | 10240.0 | 1.00 | 640 | 10.85 | 109 |

`uniform` is the *worst* case despite being the most even routing: exactly 20
rows per expert cuts into two 16-row tiles and wastes 12 rows of each, so it
reads 1024 tiles where `balanced` reads 857. `sparse` and `single` both land on
640 tiles and are 27% faster than `balanced` for identical arithmetic. The floor
at this row count is 512 tiles — each of the 512 experts' weights read exactly
once — and no routing with 20 rows per expert reaches it at a 16-row tile.

## What this projects, and what it does not

The lever is the **M-tile height**. With 20 rows per expert, a 32-row tile needs
one tile per expert instead of two, so the traffic halves: 512 tiles instead of
1024-1120, i.e. 943.7 MB instead of 1.89-2.06 GB per launch. At the bandwidth
this packet measures that is 6.6-8.7 ms per launch instead of 14.5, which over
the engine's 188 gate/up launches and 172 down launches is a **projected
1.7-2.3 s of the 16.67 s prefill**.

**That projection is arithmetic on the measured bandwidth, not a measurement.**
No 32-row kernel exists yet; the packet cannot time one. What is measured is the
traffic model, the constant bandwidth, and the engine agreement.

Two other consequences worth recording:

- **The MoE is already chunked at 1024 tokens.** The engine's four gate/up
  launches per layer are the four 1024-token chunks of a 4096-token prompt, not
  four sub-chunks of one layer: each carries the full `grid_y = 1120`. Chunking
  therefore does not multiply the MoE weight traffic here, and the earlier
  reading of "4 launches per layer" as a scheduling defect was wrong.
- **The tile map is near its floor for a 16-row tile.** 1120 tiles for 10240
  rows is 9% above the 1024-tile floor that a perfect 20-rows-per-expert
  partition would give, so there is little padding waste left to reclaim at this
  tile height; the win has to come from the tile height itself.

## Protocol

- Commands: `scripts/qwen4exp_moe_main_kernel_cost.py --rows
  1280,2560,5120,10240,20480,40960 --distribution balanced --repetitions 15`
  and the same with `--distribution balanced,uniform,sparse,single
  --repetitions 11`.
- Kernels: `gguf_q4_k_selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out` and
  `qwen4_exp_q5_1_selected_wmma_iu8_risk_prefill_bf16_bf16_out`, the shipped
  variants, at the shipped geometry and byte layout. `max_risks` is 0, so the
  risk queue is never written; the engine's own capture includes those writes.
- Timing: median of per-launch HIP event pairs after two primed launches.
- Host: AMD Radeon 8060S (`gfx1151`), Strix Halo, 125 GB unified memory.
- Engine comparison: `benchmarks/results/2026-09-17-iu8-repair-unroll/`
  (role-marked `rocprofv3`, `code-p4096`, chunk 1024) and its `role-trace`.

## What this does not say

- **Not a rate.** The weights are synthesized at the real shapes and byte
  layout, so byte counts and the parallelism response are representative and the
  values are not.
- **Not a kernel verdict.** Nothing here says the current kernel is
  well-implemented beyond the fact that its cost is explained by its traffic;
  a better access pattern could still raise the achieved 107-142 GB/s toward the
  host's ~200 GB/s, which this packet does not attempt to bound.
- **Not a claim about the comparator.** The gap to their `expert_gate_up` bucket
  is consistent with a larger M-tile, but their kernel was not inspected here.
- **Not a repair measurement.** The repair passes are a separate packet
  (`../2026-09-17-qwen4exp-iu8-repair-cost-structure/`); this one times the main
  kernels with the queue disabled.
