# GD-3 screened: the premise is stale by 41x and the ceiling is 1.1% of prefill

**Outcome: GD-3's premise is falsified by direct measurement.** The row's
baseline does not describe any route this engine has, and the lever it proposes
is bounded at ~1% of prefill. The row's *mechanism* (latency-bound, not
traffic-bound) is confirmed.

## What the row claimed

> gufo's F32 row-split recurrence stages four tokens at a time through LDS and
> keeps three windows of loads in flight because each step otherwise pays a DRAM
> latency; theirs is **~65 ms per 1K tokens against our 205**

So the row asserts we are **3.15x slower** than gufo and asks to "measure stall
cycles in our register-state kernel first; if latency-bound, add the windowed
prefetch while keeping our per-row reduction order".

## What was measured

`scripts/qwen4exp_gd3_recurrence_screen.py`, gfx1100 (Radeon Pro W7900),
realistic shape (16 k-heads, 48 v-heads, Dk=Dv=128), 5 reps after 1 warmup,
median reported. All four registered recurrence routes, 1K tokens:

| route | ms per 1K tokens | achieved GB/s | % of 221 GB/s roof | x above traffic floor |
| --- | ---: | ---: | ---: | ---: |
| `tiled16` (**production**) | **4.961** | 19.9 | 9.0% | 11x |
| `register_state` | 5.687 | 17.4 | 7.9% | 13x |
| `columnwarps` | 7.610 | 13.0 | 5.9% | 17x |
| `serial_lds` | 39.628 | 2.5 | 1.1% | 88x |

Scaling is exactly linear in tokens (register_state: 2.898 / 5.665 / 11.278 ms
at 512 / 1024 / 2048), so there is no fixed overhead - the cost is per-token.

Floors for the 1K case: traffic **0.448 ms** (99.0 MB), FLOPs **0.053 ms**
(3.22 GFLOP).

## Finding 1: the baseline is stale by 41x

The row's "our 205" does not match **any** route measured here. Even the worst
registered route (`serial_lds`, 39.6 ms) is **5.2x faster** than the row's
baseline, and the production route (`tiled16`, 5.0 ms) is **41x faster**.

**The row's gufo comparison is inverted.** At ~65 ms per 1K tokens, gufo's
figure is **13x slower** than our production route and **11.5x slower** than
`register_state` - we are ahead, not behind. Whatever the row was measuring, it
was not this engine on this hardware.

## Finding 2: the mechanism is confirmed - it is not traffic-bound

Every route sits at **1-9% of the bandwidth roof** and **0.1-1.1% of the FP32
FLOP roof**. A kernel at 9% of the bandwidth roof is not bandwidth-limited, so
the only remaining explanations are latency, occupancy, or issue - which is what
the row's proposed prefetch addresses. Two structural facts support this:

- The register-state kernel holds **128 floats of state per thread** (plus the
  rest of its working set), which is severe register pressure and collapses
  occupancy.
- Its per-token body contains roughly **16 `__syncthreads()`** - two 7-step
  shared-memory reduction ladders plus barriers - while the column-warp sibling
  below it in the same file is explicitly built to have "no `__syncthreads` in
  the token loop".

So the row is right that latency is the constraint, even though its numbers are
wrong.

## Finding 3: the lever is bounded at ~1.1% of prefill

This is the decisive cost gate, and it is an **unconditional upper bound**: it
assumes the recurrence becomes **infinitely fast**.

- One layer's recurrence is 4.961 ms per 1K tokens.
- There are **9 certified GDN layers** (global layers 35-47 → actual GDN layers
  36/37/38/40/41/42/44/45/46), so ~44.6 ms per 1K tokens, or **22.7 ms at p508**.
- Today's p512 prefill is **2014 ms**
  (`benchmarks/results/2026-09-29-qwen4exp-rt1-256-vs-1024/`).

22.7 / 2014 = **1.1%**. So no prefetch, however good, can recover more than about
**1.1% of prefill** - and a realistic 2x on the recurrence recovers ~0.55%.

For context, the last recorded trace put the **whole GDN family** (recurrence,
norms, gates, conv, prepare, snapshots) at **750.68 ms at p508**, against a
6.500 s prefill then (`benchmarks/CHANGELOG.md`, 2026-08-29 gfx1151 peer-GDN
entry). If that family figure is still roughly current, it is now a far larger
share of a 2014 ms prefill than it was of a 6500 ms one - **but that share has
not been re-measured here and is not claimed.** What is claimed is the bound
above, which does not depend on it.

## Disposition

**Close GD-3 as a stale premise.** The baseline it is measured against does not
exist, the direction of its comparison is inverted, and the lever it proposes is
bounded at ~1% of prefill.

**The mechanism is worth keeping, redirected.** The GDN family's cost is
overwhelmingly *outside* the recurrence - the recurrence is ~3% of the family
even on the old 750.68 ms trace. If GDN is to be optimized, the norms, gates,
conv, prepare and snapshot stages are where the time is, and that is a different
row with its own measurement.

## Limits

One host (gfx1100), one shape (Hk16/Hv48/D128), one quant, synthetic inputs,
recurrence kernels only. The gufo and "our 205" figures are the row's own
numbers, not re-derived; they are used only to show the row's comparison does
not survive measurement. Wall times are host-side medians around a device
synchronize, so they include launch overhead - at 5-40 ms per call that is
negligible, but these are not pure kernel durations. No `rocprofv3` counters
were taken, so "latency-bound" is inferred from the distance to both roofs
rather than read from stall counters.
