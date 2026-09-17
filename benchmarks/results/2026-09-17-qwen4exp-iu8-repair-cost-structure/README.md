# What sets the cost of the iu8 exact repair

The shipped `q4_iu8_exact` MoE route repairs 0.712% of its output elements
through sparse exact-repair kernels, and those kernels cost 1420.9 ms of a
16733.4 ms `code-p4096` prefill. This packet runs the shipped repair kernels on
the admitted shapes and the measured incidence, and varies the three things that
could explain that cost.

It answers a design question, not a rate: **the repair is bound by memory-level
parallelism, not by bytes moved or by grid size.** One unroll pragma on the
runtime-bound inner loop is worth 5.8% on the gate/up repair and 21.2% on the
down repair; a redesign that cuts its traffic 3.3x is 22% slower.

## Protocol

Shipped kernels, shipped geometry (`experts=512`, `hidden=2560`, `ffn=640`,
`rows=10240` = 1024 tokens x top-k 10), synthetic weights at the real byte
layout, risk sets at the measured 0.712% aggregate repair rate
([instrumentation entry](../../../worklog/entries/20260915T111206.049450Z-lhl-journey-risk-repair-instrumentation-de5527.md)).
Median of per-launch HIP event pairs, 15 repetitions, two primed launches.

Weights are synthesized, so values are not representative; byte counts, access
patterns and the parallelism response are. The engine's own capture of the same
kernels agrees with this packet to 0.8% on the gate/up repair
([attribution](../2026-09-17-qwen4exp-shipped-default-attribution/README.md):
3.02 ms per launch against 3.05 here, pre-unroll).

| Artifact | Contents |
| --- | --- |
| `artifact-pre-unroll.json` | Before the unroll pragma: the baseline this unit measures against |
| `artifact.json` | After the pragma, with the grid sweep and the row-bucketed alternative |
| `artifact-clustered.json` | After the pragma, risk set drawn from a small row window (L2 control) |

## What sets the cost

Grid size does not. Same risk count, same shapes, four grid sizes:

| Grid | gate/up repair, 93,342 risks |
| --- | ---: |
| 17920 (shipped) | 3.0465 ms |
| 65536 | 3.0500 ms |
| 262144 | 3.1778 ms |

Bytes moved do not, either. A row-bucketed pair of kernels (`..._risk_bitmap`
plus `..._row_bitmap_repair_bf16`) folds the flat queue into a per-row column
bitmap and stages each row's activations in LDS once, which cuts per-element
traffic from 6560 B to about 2000 B. It is **bit-identical** (see
`tests/test_gpu_qwen4exp_q4_iu8_row_bitmap_repair.py`, which compares both
kernels element by element against a poisoned sentinel) and **22% slower**:

| Path, 93,342 risks | ms | Bytes moved |
| --- | ---: | ---: |
| Per-slot, shipped | 2.8685 | 612 MB |
| Row-bucketed composite | 3.5054 | 186 MB |

Locality barely helps either. Drawing the same 93,342 risks from 1/8 of the
rows, so the activation rows a repair re-reads are L2-resident, gains 13.8%
(2.8685 -> 2.4719 ms). If DRAM bytes were the constraint, removing 85% of the
activation-row traffic would gain far more.

What does help is giving the inner loop enough independent work to issue. Its
trip count is a runtime value (`in_features / 256`), so each iteration's loads
serialize behind one dependent round trip. `#pragma unroll 4` on the three
repair kernels is arithmetic-preserving and worth:

| Kernel | Risks | Before | After | Change |
| --- | ---: | ---: | ---: | ---: |
| `gguf_q4_k_selected_dual_sparse_exact_repair_bf16` | 93,342 | 3.0465 ms | 2.8685 ms | **-5.8%** |
| `q5_1_selected_sparse_exact_repair_row_publish` | 186,685 | 1.6404 ms | 1.2922 ms | **-21.2%** |
| `q8_0_selected_sparse_repair` | engine capture | 134.2 ms | 103.9 ms | **-22.5%** |

## Engine effect

Same protocol as the retained attribution (role-marked `rocprofv3`, one
`code-p4096` prefill, chunk1024, shipped default), source in
`../2026-09-17-iu8-repair-unroll/`:

| Repair kernel | Before | After | Change |
| --- | ---: | ---: | ---: |
| `q5_1_selected_sparse_exact_repair_row_publish` | 659.4 ms | 535.5 ms | -18.8% |
| `gguf_q4_k_selected_dual_sparse_exact_repair_bf16` | 568.2 ms | 538.7 ms | -5.2% |
| `q8_0_selected_sparse_repair` | 134.2 ms | 103.9 ms | -22.5% |
| `gguf_q5_k_selected_dual_sparse_exact_repair_bf16` | 59.1 ms | 61.7 ms | +4.4% |
| **repair total** | **1420.9 ms** | **1239.8 ms** | **-12.7%** |

The main MoE kernels in the same two captures are unchanged
(`gguf_q4_k_selected_dual_wmma_iu8_risk_prefill` 2708.8 -> 2710.3 ms, +0.1%;
`q5_1_selected_wmma_iu8_risk_prefill` 1483.8 -> 1484.5 ms, +0.0%), so the two
captures are comparable and the change is confined to the repair passes.

**The prefill-level effect is not claimed.** The whole-prefill kernel total moved
16733.4 -> 16673.2 ms (-0.36%), which is inside this capture's own drift: the
non-repair kernels summed 1.1% higher in the second capture. The evidence here is
a sub-window reduction in named kernels, measured twice with two independent
instruments.

## What this does not establish

- **Not a wall-clock claim.** No interleaved engine A/B exists for a
  compile-time change; both engine captures are separate profiled runs.
- **Not an incidence result.** The repair rate is taken as measured; this packet
  does not re-derive it, and the down route's true incidence is inferred from its
  engine time (186,685 risks is the 0.712% figure applied to its shape, which
  reproduces the engine's pre-unroll time to 0.3%).
- **Not a traffic model.** The byte counts are the naive per-element sums; the
  clustered control shows the kernel does not behave like a bandwidth-limited
  reader of those bytes.
- **Not a claim that the repair is cheap.** 1239.8 ms to correct 0.7% of
  elements is still 7.4% of the prefill. Reducing the incidence is a numerical
  contract question, not a kernel question, and is untouched here.
