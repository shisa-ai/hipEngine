# GD-4's chunked WY prefill: built, parent-parity correct, and slower than every route that matters

**Date:** 2026-10-01
**Host:** AMD Radeon 8060S (gfx1151, Strix Halo), backend `hip_gfx1151`
**Artifact:** `artifact.json`

## What was built

The FP32 chunked triangular / Woodbury-Young GDN prefill that the previous entry
recommended as GD-4's first build step:

- `qwen4_exp_gdn_prefill_wy_f32_kernel` in
  `hipengine/kernels/hip_gfx1100/linear_attn/qwen4_exp_gdn.hip`
- 128x32 state tile in LDS instead of registers, chunk 8, 256 threads,
  grid `(num_v_heads, head_v_dim / 32)` = `(48, 4)`
- FP32 operands throughout; the fp16 arm was screened and rejected earlier
  (`2026-09-30-qwen4exp-gd4-wy-chunk-drift`)
- Same two-stage shape as the other prefill owners: recurrence writes raw
  `core`, then `qwen4_exp_gdn_core_norm_gate_f32_kernel` overwrites it in place

The form is the direct lower-triangular one, with the decay-before-projection
convention the production kernels use:

```
P[t]   = prod_{u<=t} decay[u]
R[t,s] = prod_{u=s+1..t} decay[u],  R[t,t] = 1
delta[t] = beta[t] * ( v[t] - P[t]*(k_t . H0)
                       - sum_{s<t} R[t,s]*(k_t . k_s)*delta[s] )
out[t]   = P[t]*(q_t . H0) + sum_{s<=t} R[t,s]*(q_t . k_s)*delta[s]
H1       = P[C-1]*H0 + sum_s R[C-1,s] * outer(k_s, delta[s])
```

The screen (`scripts/qwen4exp_gd3_recurrence_screen.py`) was extended with the
two arms that decide the row: `wave_norm`, the route the `gguf_ud_q4_k_xl`
production lane actually dispatches, and `tiled16_multi`, the multi-column
build that the `gguf_q4_k_m` lane selects.

## Correctness

Parent-parity against the production `tiled16` route, and independent agreement
with an fp64 serial GDN recurrence:

| comparison | max abs | relative to reference max |
| --- | ---: | ---: |
| core vs `tiled16` | 5.96e-07 | 3.8e-7 |
| state vs `tiled16` | 5.59e-09 | 1.9e-7 |
| core vs fp64 serial recurrence | 2.9e-05 | — |

All at fp32 rounding level. The state and core paths are exercised together, so
this is a real parity check rather than a check on a degenerate head.

**One measurement trap worth recording.** The first comparison reported a
full-magnitude mismatch (1.067 against a reference max of 1.563) while the
*state* matched to 5.6e-09. The cause was not the kernel: the production
wrappers launch two kernels, and the second one overwrites `core` in place with
the normalized and gated output, while the new kernel wrote raw `core`. Applying
the same norm/gate stage brought the two into agreement at 7.3e-07. A route that
is not a drop-in for the interface it is compared against will look broken.

## Speed

7 reps after warmup, wall medians including launch overhead, same host and same
session for all seven routes:

| tokens | `wy_chunked` | `tiled16` (non-multi) | `wave_norm` (**UD lane**) | `tiled16_multi` (q4_k_m lane) |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 2.620 ms | 2.177 ms | 2.763 ms | **1.157 ms** |
| 1024 | 6.202 ms | 5.063 ms | 5.413 ms | **3.150 ms** |
| 2048 | 13.545 ms | 11.828 ms | 10.731 ms | **7.741 ms** |

Ratios (first-named route over second):

| comparison | 512 | 1024 | 2048 |
| --- | ---: | ---: | ---: |
| `wy` vs `tiled16` (the screen's original baseline) | 1.20x slower | 1.22x slower | 1.15x slower |
| `wy` vs `wave_norm` (**the UD lane's real route**) | **1.05x faster** | 1.15x slower | 1.26x slower |
| `wy` vs `tiled16_multi` | 2.26x slower | 1.97x slower | 1.75x slower |
| `tiled16_multi` vs `wave_norm` | **2.39x faster** | **1.72x faster** | **1.39x faster** |

The measurement is stable: the largest spread across reps was 6.1%, and most
were under 2.6%. A first 5-arm run (no `wave_norm` / `tiled16_multi`) reproduced
the `tiled16` and `wy_chunked` columns within 1.3%, and the `tiled16_multi`
median at 512 (1.157 ms) matches the promotion artifact's own 1.112 ms.

**The chunked form issues 1.19x the multiply-accumulates of the serial chain,
and the slowdown matches.** Per token per v-head the serial route does three
128x128 products (49,152 MACs). The chunked route keeps those three (key/query
state and the chunk-final state, 49,152) and adds 9,216: the `C x C` key/key and
query/key dots (8,192 - each of the four value-column blocks recomputes them in
full, and half of that square is the discarded upper triangle), the triangular
solve (448) and the output combination (576). 58,368 / 49,152 = **1.19x**, and
the measured slowdown against `tiled16` is 1.15-1.22x. A cooperative
lower-triangle implementation would add only 2,176 MACs (1.04x) - but the
occupancy this was meant to buy is not in the built kernel either (169 VGPR
against the register route's 256, and against the multi-column route's 85).

## The register counts, and which lane each route actually serves

The row's cost gate rests on: *"the recurrence runs at `VGPR_Count` 256 (the
hardware maximum), i.e. ~12.5% occupancy, so the register-file-capacity argument
that makes a chunked/WMMA reformulation attractive is measured rather than
inferred."*

Compiled register counts, re-derived from `.amdhsa_next_free_vgpr` on the
current source (the `rocprofv3`-reported counts are these rounded up to the
allocation granularity):

| kernel | VGPR | LDS | selected by a profile |
| --- | ---: | ---: | --- |
| `..._prefill_f32_kernel<true,true>` (`wave_norm`) | **256** | 0 | **UD-Q4_K_XL production** |
| `..._prefill_f32_kernel<true,false>` (`register_state`) | 256 | 0 | no |
| `..._columnwarps_f32_kernel<4,4>` | 46 | 0 | strict/recovery only |
| `..._prefill_tiled_f32_kernel<16,16,4,1>` | 56 | 16640 | no |
| `..._prefill_tiled_f32_kernel<16,8,4,8>` (`tiled16_multi`) | **85** | 16640 | **q4_k_m production** |
| `..._prefill_wy_f32_kernel` (new) | 169 | 28576 | no |

**The row's premise is not stale. It is true for the lane the trace measured.**
The `gguf_ud_q4_k_xl` production profile applies
`PRODUCTION_ARITHMETIC_RECOVERY_FLAGS`, which zeroes `GDN_COLWARPS_PREFILL`; the
runner then takes the register path, and `GDN_REGISTER_PREFILL=1` +
`GDN_WAVE_NORM=1` (both bound by the production profile) resolve it to
`qwen4exp_sigmoid_wave_norm_prefill` - the `<true,true>` kernel at 256 VGPR. The
family-share trace
(`2026-09-30-qwen4exp-gd4-family-share/rocprof/prefill_kernel_trace.csv`) is
exactly this: **72 dispatches of `<true,true>`, zero tiled dispatches, bound env
`GDN_COLWARPS_PREFILL=0`, model root `UD-Q4_K_XL`**. The 4.97% family share and
the 256-VGPR figure are the UD lane's own route.

**But the register bound is already escaped, by a built and registered route.**
`tiled16_multi` compiles at **85 VGPR** (a third of the register route) and is
**1.39-2.39x faster** than the UD lane's `wave_norm` route at 512/1024/2048. It
is the `gguf_q4_k_m` production selection, and it is gated off on the UD lane by
arithmetic recovery - its arithmetic is fp32-round-off close but not bit-exact
(NMSE 1.8e-14), and the UD lane is the recovery lane. On the `q4_k_m` lane,
where the gate is open, the composition fails the calibrated envelope for a
pre-existing reason unrelated to this route (an attribution control fails the
same scopes with a worse tail: `2026-09-30-qwen4exp-gdn-multicolumn-q4km`).

So GD-4's mechanism is not the cheapest available escape from register pressure,
on either lane: on UD the exact route in use is the one that must be beaten and
a chunked reassociation changes arithmetic, and on q4_k_m a cheaper built route
already exists. The remaining question there is arithmetic admission for an
existing kernel, not a new chunked/WMMA build.

## What this does and does not show

**Does show:** the FP32 chunked WY reformulation of this recurrence is correct,
issues 1.19x the MACs, and is slower than the route it was screened against, the
route the UD production lane runs (at 1K and 2K), and the multi-column route
that already escapes the row's register bound.

**Does not show:** that the recurrence cannot be sped up. The ~5.5% family-share
ceiling is unchanged and independent of which route owns the family. The fp16
WMMA arm was not built - what is dead is the *justification for building it*:
the register-capacity escape it was meant to buy already exists at 85 VGPR and
1.4-2.4x faster, and the fp16 arm adds MX-1-class numerical risk on top.

**Limits:** one host (gfx1151, not the W7900 default), synthetic inputs, wall
medians around a device synchronize, no `rocprofv3` stall counters. The
`wave_norm` / `tiled16_multi` comparison is wrapper-to-wrapper at this one shape
(Hk16/Hv48/Dk=Dv=128); the lanes' own gates are quoted from their artifacts, not
re-run here.
