# GD-4 chunked scan: the build plan, and three constraints that decide whether it works

The feasibility screen priced the arithmetic. Working the design against the validated
reference and the existing ABI surfaced **three constraints the ledger row does not state**,
one of which silently destroys the entire advantage if violated.

**Build plan only. No kernel was written, compiled or timed.** `performance_claim: false`.

## The algorithm, from the validated reference

`scripts/gdn_chunk_drift_screen.py::_chunk_local`, whose direct suffix product is locked by
`tests/test_unit_gdn_chunk_drift_screen.py`:

```
q_scale = rsqrt(max(|q|,1e-6)) * rsqrt(Dk);  k_scale = rsqrt(max(|k|,1e-6))
beta    = sigmoid(beta_logits);  decay = exp(-exp(a_log) * softplus(alpha + dt_bias))
cum     = cumprod(decay);        ratio[i][j] = prod_{k=j+1..i} decay[k]     <- SUFFIX PRODUCT
start_term = cum * (k_norm @ S_0)
L[i][j] = beta_i * (k_norm_i . k_norm_j) * ratio[i][j]        (strictly lower)
rhs = (v - start_term) * beta;   forward-substitute (I + L) delta = rhs
out = (q_norm @ k_norm^T * ratio)_tril @ delta + cum * (q_norm @ S_0)
S_end = S_0 * cum[-1] + (k_norm * ratio[-1,:])^T @ delta
```

then the existing `qwen4_exp_gdn_core_norm_gate_f32_kernel` epilogue, unchanged.

## Constraint 1 — L and QK are value-column independent, so they must be computed cooperatively

L and QK depend only on `k`, `q` and the decay — **not on the value column**. Every warp of a
v-head needs them, so they are shared data. Computing them per-warp redundantly costs **16x**:

| chunk | L+QK MACs/chunk/head | **cooperative** | **per-warp redundant (x16 warps)** |
| ---: | ---: | ---: | ---: |
| 16 | 32,768 | **+4.17%** | **+66.7%** |
| 32 | 131,072 | **+8.33%** | **+133.3%** |
| 64 | 524,288 | +16.67% | +266.7% |

This is the difference between the priced **1.05 TFLOP/s break-even** and a route carrying
**66.7% more arithmetic that must reach ~15 TFLOP/s** — i.e. it converts a **2%-of-peak** bar
into a **25%-of-peak** bar. It must be computed cooperatively into shared memory with a
`__syncthreads()`, not per warp.

## Constraint 2 — the LDS operating point

Staging `kn`, `qn`, `L`, `ratio` and `delta`:

| chunk | LDS | 64 KB/workgroup |
| ---: | ---: | --- |
| 16 | 26.3 KB | fits, comfortable |
| 32 | 56.6 KB | **fits**, 7.4 KB headroom |
| 64 | 129.2 KB | **over** |

**Chunk 32 primary** — it fits, and its fp16 state-carry NMSE (7.961e-10) is better than
chunk 16's, because the drift screen found fp16 NMSE *improves* with larger chunks. **Chunk 16
is the fallback** if LDS-limited occupancy binds.

## Constraint 3 — the block decomposition must change, and this is the structural work

The existing `tiled16` grid is **`(num_v_heads=48, col_blocks=8)`** — 8 blocks per v-head,
each owning 16 of the 128 value columns. Under a chunked scan each of those 8 blocks would
recompute the shared L: **exactly the 8x redundancy constraint 1 forbids.**

Collapsing to one block per v-head removes it, but gives only **48 blocks against the
gfx1100's 96 CUs** — half the device idles.

| option | blocks | L redundancy | device fill |
| --- | ---: | --- | --- |
| one block per v-head, 16 warps x 8 cols | 48 | none | **half idle** |
| two blocks per v-head (64 cols each) | 96 | 2x (+4.17% -> +8.34% at chunk 16) | full |
| **multi-block scan over the chunk axis + state-propagation pass** | many | none | full |

**Filling the device requires chunk-axis parallelism**, i.e. a two-kernel design with a
chunk-level state propagation between groups. **That is the main structural work the row does
not mention** — it is not a one-kernel port of `tiled16`.

## Constraint 4 — the suffix product is an in-kernel requirement, not just a reference one

The kernel's `ratio` matrix **and** the state-carry weights `ratio[-1,:]` must both be direct
suffix products. A ratio of prefix products divides by a geometric decay that underflows to
zero non-simultaneously; while `cum[j]` has underflowed and `cum[i]` has not, the ratio
explodes and the solve diverges — measured max_abs **1.05e1 at chunk 256 against 2.26e-01 at
chunk 64**, larger chunks *worse*, which is the signature. The suffix product never divides
and every factor lies in (0, 1], so entries stay bounded by 1.

## Why this is recorded before the build

The row says the remaining work is "the kernel build". It is, but three of the four
constraints above are invisible from the row and two of them change the kernel's structure
rather than its tuning. A builder who ports `tiled16`'s grid and computes L per warp gets a
kernel that is 66.7% over its arithmetic budget and cannot be diagnosed from timing alone.
