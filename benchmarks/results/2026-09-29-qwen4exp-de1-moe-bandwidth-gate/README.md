# DE-1 bandwidth-first gate: the routed-expert MoE block is NOT bandwidth-bound

**Date:** 2026-09-29
**Host:** Radeon 8060S (gfx1151), hostname `gfx1151`, machine-id
`55ea6c509d0b49eea8de7094a1023668`
**Model:** Qwen3.8-Flash-Next UD-Q4_K_XL (unsloth split GGUF, 4 shards, 111.3 GB)
**Command:** `python3 scripts/qwen4exp_de1_moe_bandwidth_gate.py --output
benchmarks/results/2026-09-29-qwen4exp-de1-moe-bandwidth-gate/gate.json`
**Artifact:** `gate.json` (this directory) - a derivation, **no GPU execution**
**Tree:** `534d62a6c`

## Why this measurement

`DE-1` is the largest measured opportunity in the ledger: its own attribution
records the routed-expert MoE block at **24.5504 ms/step = 45.28% of decode kernel
time**, more than every other family combined, spread over 48 layers with no
dominant member - so only a whole-family lever can matter.

Its action is:

> An exact-activation variant of the warp256 structure (the structure, not the
> dp4a arithmetic), **measured for bandwidth first**.

This is that first measurement. It is deliberately **cheap and decisive**: if the
family were already at this host's memory roof, no structural change could help
and the row should be parked. It is not.

## Method

**Traffic model.** For a decode GEMV (one token), every weight byte of every
*selected* expert is read exactly once. The working set - 1.5 GB of routed expert
weights for a single step - far exceeds any cache on this host, so each read is a
DRAM read. Per step the MoE block moves:

| component | derivation | bytes/step |
|---|---|---:|
| routed experts | 48 layers x 10 of 512 experts x (gate+up+down) | **1,504.26 MB** |
| shared expert | 48 layers x (gate+up+down) | 250.68 MB |
| router | 48 layers x `ffn_gate_inp` (GEMV over all 512 experts) | 251.66 MB |
| **total** | | **2,006.59 MB** |

Bytes are the exact GGUF payload sizes read from the model's own tensor table
(1224 tensors across 4 shards), not a bits-per-weight estimate.

**Comparator.** `linear:root.lm_head` is a single dense GEMV in the **same
profiled capture**, streaming its own 675.43 MB weight in 2.9798 ms. It measures
the rate this host actually reaches on this workload shape in this run, so it
bounds what the MoE block could reach **without** relying on a datasheet number.

## Result

| family | bytes/step | ms/step | kernels | achieved | vs documented 221 GB/s roof |
|---|---:|---:|---:|---:|---:|
| **`moe:expert_gate`** | **2,006.59 MB** | **24.5504** | **583** | **81.73 GB/s** | **37.0%** |
| `linear:root.lm_head` (comparator) | 675.43 MB | 2.9798 | 1 | **226.67 GB/s** | **102.6%** |

**The same host, in the same decode step, in the same profiled capture, streams a
dense GEMV at 226.67 GB/s while the MoE block manages 81.73 GB/s - a 2.77x gap.**

| derived quantity | value |
|---|---:|
| MoE block floor at the comparator's rate | **8.85 ms** |
| headroom | **15.70 ms/step** |
| headroom as share of the 54.218 ms kernel sum | **29.0%** |
| per-kernel time (583 kernels) | 42.11 us |

**Verdict: not bandwidth-bound.** `DE-1`'s proposed structural change has a real,
quantified target of roughly **15.7 ms/step**, i.e. **~29% of decode kernel time**,
before any arithmetic is touched.

## Why the comparator is the right control

- **Same run, same host, same step.** No cross-session or cross-host inference.
- **It independently validates the roof.** 226.67 GB/s against the documented
  221 GB/s practical read roof is 102.6% - i.e. the documented figure is right and
  the traffic model is sound, since an unsound model would not land on the roof.
- **It is the same *shape* of work** (a decode GEMV streaming a large weight
  matrix), so the comparison isolates the family's own inefficiency rather than
  contrasting different workload classes.
- **It is a single kernel**, whereas the MoE family is 583 - so part of the gap may
  be launch/tail effects rather than pure streaming. That is a hypothesis for
  `DE-1` to test, and it does not change the conclusion that headroom exists.

## What this does and does not establish

**Established:** the routed-expert MoE block runs at 37.0% of the documented roof
while a dense GEMV in the same capture reaches 102.6%; the gap is ~15.7 ms/step.

**Not established:** *which* structure is responsible. The measurement says the
family is not at the roof; it does not say the warp256 variant is the fix, nor
that the gap is launch overhead rather than address-generation, gather
efficiency, tail quantisation across 583 small kernels, or staging traffic. Those
are the next questions, and they are `DE-1`'s.

**Also not established:** any end-to-end speedup. This is a traffic-and-time
derivation from a profiled capture, not a benchmark.

## Limits

- **Derived from a PROFILED capture.** The source artifact's own `limits` state
  that the wall and per-kernel times are *diagnostic, not a performance
  measurement*. Profiling inflates kernel times; the family's true time may be
  somewhat lower than 24.5504 ms, which would *narrow* the gap slightly but
  cannot close a 2.77x ratio.
- **The traffic model assumes each selected expert's full weights are read once
  per token.** True for a GEMV, and the working set far exceeds cache. If the
  kernels additionally stage or dequantize into wider buffers, actual DRAM traffic
  *exceeds* the model - making 81.73 GB/s a **lower bound** on the achieved rate
  and the headroom estimate conservative in the direction that matters least.
- **One case** (`code-p4096`), **one live count** (4097), **one model**, **one
  host**, **one execution profile**.
- The `moe:expert_gate` family name is a **misnomer**: the role wraps the whole
  `run_qwen4_exp_moe` call and merely *names* itself after the gate weight's slot
  (`scripts/qwen4exp_profile_gap.py:97`), so the family covers gate **and** up
  **and** down **and** the shared expert **and** the router. The traffic model
  above counts all of them.

## Reproduce

```bash
TH=/home/lhl/miniforge3/envs/therock10-staging-20260828
$TH/bin/python scripts/qwen4exp_de1_moe_bandwidth_gate.py \
  --output benchmarks/results/2026-09-29-qwen4exp-de1-moe-bandwidth-gate/gate.json
```

## Addendum: identifying the owner kernels from the existing trace

The capture's raw kernel trace (`../2026-09-29-qwen4exp-x4-decode-split/rocprof/x4decode_kernel_trace.csv`,
18,941 dispatches) spans **68 s** of prefill + warmup + measured steps, so it cannot segment a
single decode step without redoing the join that `attribution.json` already performs. Two structural
facts are nevertheless readable from it, and both matter for `DE-1`'s next cut.

**The expert GEMM owner is `gguf_q4_k_selected_dual_wmma_iu8_risk_*`.** The `selected` refers to
*selected experts* (MoE routing), and **`dual` confirms gate and up are fused into a single kernel**.
It appears 188 times in the capture totaling 2,709.02 ms, at **`grid = 1280`, `workgroup = 128`**.

**1,280 is exactly the gate+up output row count for one expert** (640 gate rows + 640 up rows). So the
decomposition is **one workgroup per output row, each reducing over 2,560 input columns** - a
128-thread workgroup spanning the full reduction axis. That is a plausible source of the 2.77x gap
and it is a *structural* property, which is precisely what `DE-1`'s action targets ("the structure,
not the dp4a arithmetic").

The same trace also shows the router as a separate, much wider kernel:
`qwen35_router_logits_f32_token_tile_de*` at `grid = 131072`, `workgroup = 256` (192 dispatches,
617.19 ms) - the 251.66 MB/step router term in the traffic model above.

**Not concluded:** whether the per-row decomposition is the cause. The trace cannot separate
execution from inter-kernel gaps at step granularity here, and no claim is made that changing the
decomposition recovers the gap. This addendum only names the owner kernel, its fusion structure and
its launch geometry, so the next screen starts from the real shape rather than from the family name.

## Addendum 2: the gap is inside the kernels, not between them

The step-segmentation question left open in Addendum 1 is answerable from the capture already in
the tree, because `x4decode_marker_api_trace.csv` carries the `qwen4exp_decode_live4097_rep{0,1,2}`
ROCTX ranges. Segmenting the kernel trace by those ranges gives **exactly 1,771 dispatches per
rep** - matching the attributed kernels/step - and reproduces the attributed kernel sum:

| rep | kernels | kernel sum | span | total gap | **median gap** | max gap |
|---|---:|---:|---:|---:|---:|---:|
| rep0 | 1,771 | 54.721 ms | 227.470 ms | 172.749 ms | **53.60 us** | 7,857.7 us |
| rep1 | 1,771 | 54.671 ms | 225.228 ms | 170.558 ms | **58.46 us** | 1,217.7 us |
| rep2 | 1,771 | 54.173 ms | 81.685 ms | 27.512 ms | **2.28 us** | 514.0 us |

### This answers the launch/tail question

**The per-family figures are sums of kernel *durations*** - the 30 families total exactly 54.218 ms -
so **inter-kernel gaps are not in them.** The MoE block's 24.5504 ms is GPU-resident execution time,
not queueing or launch latency. **Therefore the 2.77x bandwidth gap is inside the kernels, not
between them.** Launch/tail overhead is excluded as the explanation, and `DE-1`'s structural lever
("the structure, not the dp4a arithmetic") is aimed at the right target.

### And it independently corroborates the DE-5 correction

The kernel work is **stable across all three reps** (54.72 / 54.67 / 54.17 ms - a 1.0% spread) while
the step **span** varies 2.8x (227.5 / 225.2 / 81.7 ms). **The wall varies entirely through
inter-kernel gaps, with a median gap that swings 23x between reps (53.60 vs 2.28 us).** A stable
kernel sum under an unstable wall is exactly the signature described in
`../2026-09-29-qwen4exp-de5-host-overhead-correction/README.md`: the harness's step wall is not a
serving-path number, and the engine's kernel work is what it is.

### Limits

Same profiled capture as everything above, and the same one-case/one-model/one-host scope. The gap
statistics describe **this profiled harness**, not the engine's steady-state scheduling: a median
gap of 53.60 us between consecutive kernels is far above a normal device-side kernel-to-kernel
latency, and the reps differ by 23x, so these gaps are dominated by harness/profiler interaction
rather than by engine behavior. **No gap figure here should be read as an engine property.** The
conclusion drawn from them is only the negative one: gaps are *not* what the 24.5504 ms consists of.
