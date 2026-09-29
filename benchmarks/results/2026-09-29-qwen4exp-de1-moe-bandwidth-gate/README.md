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
