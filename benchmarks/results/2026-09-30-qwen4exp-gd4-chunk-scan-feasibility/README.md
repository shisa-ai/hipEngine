# GD-4 chunked scan: the arithmetic cost is small, so the build is justified

GD-3's screen established that the GDN recurrence is **not** traffic-bound. It did not
establish what a chunked scan would cost. This is that screen, computed from GD-3's own
measured sequential rate and GD-4's already-passed drift screen.

Read-only arithmetic and committed measurements; **no kernel was built and no chunked
schedule was timed.** `performance_claim: false`.

## The sequential route is deeply latency-bound

Production `tiled16`, measured at **4.961 ms per 1K tokens per layer** (GD-3's screen).

| quantity | value |
| --- | ---: |
| MACs per token per v-head | 3 x 128 x 128 = **49,152** |
| FLOPs per 1K tokens per layer | **4.832 GFLOP** |
| achieved | **974 GFLOP/s** |
| as a share of the 61 TFLOP/s fp32 peak | **1.60%** |
| as a share of the 28.5 TFLOP/s measured ceiling | 3.42% |

The three products are `memory = k.S`, `state update = k (x) delta`, `output = q.S`.

## What the chunked scan adds

Two `C x C` dot-product matrices per chunk per head: the **L** matrix of the UT/WY
transform and the **QK** matrix the outputs need.

- per chunk per head: `L: C^2/2 x 128` + `QK: C^2/2 x 128` = **`C^2 x 128` MACs**
- per token per head: `C x 128` MACs
- **ratio to sequential: `C/384`**

The ratio is small because the sequential route already does *three* 128x128 products per
token, not one. (An early pass in this session estimated "16x more FLOPs" by comparing
against a single 128x128 op; that was wrong by the factor of three.)

| chunk | extra FLOPs | total GFLOP/1K/layer | break-even | % of 61 TFLOP/s peak | fp16 state-carry NMSE (measured) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | +8.3% | 5.234 | 1.05 TFLOP/s | 1.7% | 7.961e-10 |
| 64 | +16.7% | 5.637 | 1.14 TFLOP/s | 1.9% | 3.855e-10 |
| **128** | **+33.3%** | **6.442** | **1.30 TFLOP/s** | **2.1%** | **1.824e-10** |
| 256 | +66.7% | 8.053 | 1.62 TFLOP/s | 2.7% | 8.973e-11 |

The NMSE column is from the already-committed drift screen
(`2026-09-30-qwen4exp-gd4-chunk-drift/`), which also found the fp16 numbers are ~1000x
better than the cited reference (PR #91's "NMSE ~2e-7") and that fp16 NMSE **improves with
larger chunks** - fewer boundary roundings. So the numerically better operating point is
also the more expensive one, and **chunk 128 is the balance**.

## Verdict

**Feasible and worth building.** Break-even is **1.05-1.62 TFLOP/s = 1.7-2.7% of the fp32
peak**, against the **974 GFLOP/s (1.60%) the sequential route already reaches**. The
chunked schedule therefore needs only to be roughly **twice** as efficient as a route that
sits at 1.6% of peak, while exposing **C x more parallelism** along the token axis - which
is exactly the axis the sequential route has none of.

**What this does not show.** It does not show the chunked scan wins. It shows the arithmetic
cost is small enough that a win is possible. Whether the schedule reaches the 1.3 TFLOP/s bar
is what the build must demonstrate.

## The counter constraint, re-confirmed

PMC counters cannot be used to establish the mechanism directly.
[`RDNA3-TUNING-GUIDE.md`](../../../docs/RDNA3-TUNING-GUIDE.md) section 4.9 records that every
`rocprofv3` counter collection on this stack **hangs before the profiled application starts**,
and that `timeout` cannot kill it because the SIGTERM handler then blocks forever - wrap such
attempts as `timeout -k 10`. GD-3's screen avoided counters for this reason and was right to.
The mechanism must be established by A/B timing against the roofline bounds.

## Ceiling

The GDN recurrence *is* the GDN family: **194.487 ms of 3,914.468 ms (4.97%)** in GD-3's p512
trace, and the `gdn` role is **844.295 ms of the 15,741.963 ms union (5.36%)** in the X-3
p4096 window. Removing the recurrence entirely is therefore worth **~5% of the request**; a
2x throughput gain is worth ~2.5%.

## Limits

Arithmetic model plus one measured sequential rate. One shape, one host class, no kernel
built, no chunked schedule timed. The 61 TFLOP/s peak is the screen script's W7900 figure; the
28.5 TFLOP/s is GD-3's measured FP32 ceiling with dual issue.
