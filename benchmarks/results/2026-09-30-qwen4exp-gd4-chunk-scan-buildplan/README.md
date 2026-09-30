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

---

## Correction: the feasibility artifact's chunk 128 is not buildable, and chunk 32 is right anyway

**Added after the build plan, reconciling it against the already-committed feasibility
artifact.** The two disagreed: the feasibility artifact says *"chunk 128 is the balance"*,
this build plan says chunk 32. The disagreement is real and the LDS arithmetic settles it.

Chunk 128 is **3-5x over the 64 KB per-workgroup limit under every staging choice**:

| staging | max LDS-resident chunk | chunk 128 costs |
| --- | ---: | ---: |
| full fp32 (`kn`, `qn`, `L`, `ratio`, `delta`) | **32** | 320.0 KB — **5x over** |
| fp16 `kn`/`qn`/`L`/`ratio`, fp32 `delta` | **48** | 192.0 KB — **3x over** |

The feasibility artifact picked its operating point on the **numerics axis alone** — fp16
NMSE improves monotonically with chunk size — without checking whether that chunk is
resident. A builder acting on it would have started at chunk 128 and hit a 5x overrun.

**But the reconciliation also shows chunk 32 was the right call on the artifact's own
terms**, because its NMSE advantage does not bind:

| chunk | fp16 NMSE (measured) | break-even | vs the cited reference (2e-7) | LDS-resident |
| ---: | ---: | ---: | ---: | --- |
| **32** | 7.961e-10 | **1.05 TFLOP/s** | **251x better** | **yes** |
| 64 | 3.855e-10 | 1.14 TFLOP/s | 519x better | no |
| 128 | 1.824e-10 | 1.30 TFLOP/s | 1096x better | no |

Chunk 32 has the **lowest break-even in the artifact's own table**, is the only one that is
LDS-resident, and its NMSE is still **251x better than the cited reference**. The extra
orders of magnitude that chunks 64-128 buy are orders of magnitude **nobody asked for** —
the admission question is whether the route clears the reference's 2e-7, and chunk 32
clears it by 2.4 orders.

**Governing operating point: chunk 32**, with chunk 48 available if `delta` is also staged
in fp16 and occupancy does not bind. The feasibility artifact's chunk 128 is superseded on
the feasibility axis, not the numerics axis.

---

## The build cost: five components, one with no precedent in this family

The row reads as "chunked GDN on fp16 WMMA" — i.e. a port of a known technique. Checking what
the tree already has shows it is **not a port**:

| # | component | precedent in tree |
| ---: | --- | --- |
| 1 | **fp16 WMMA GEMM primitive in `linear_attn/`** | **none.** WMMA appears only in the attention family, used inline inside `process_attention_tile` in `attention/laguna_flash_attention_prefill.hip` (intrinsics at lines 230 and 360). It is attention-shaped, not a generic GEMM helper, so it cannot be called as one. |
| 2 | chunk-local scan (cooperative L in LDS, chunk 32, direct suffix products) | none — the family is entirely scalar-FMA |
| 3 | chunk-axis state-propagation pass (the second kernel) | none |
| 4 | variant registration with `tiled16` as the strict fallback | the registry pattern exists |
| 5 | **a new GPU test envelope** | **conflicts with the existing one** — see below |

### Component 5 is a contract change, not just a test

`tests/test_gpu_qwen4_exp_gdn_hip.py::test_qwen4_exp_gdn_prefill_is_exact_to_serial_decode`
holds every existing prefill route **exact to serial decode**. A chunked scan **cannot**
satisfy that by construction: it reassociates the recurrence's floating-point sums, which is
the entire point. The drift screen already measured what the reassociation costs (fp32 NMSE
~5-7e-14, indistinguishable from round-off; fp16 state carry 7.961e-10 at chunk 32).

So component 5 is not "add a test" — it is **adding a second, envelope-based prefill contract
alongside the exact one**, which is the kind of change `docs/EXECUTION-PROFILES.md` governs
rather than a local test edit.

### What this means for the estimate

**Five components, one of which has no precedent in the directory and one of which changes a
test contract rather than adding to it.** This is a multi-session build, not a single
iteration. Stated so the row is not started under a "port it" estimate — the arithmetic and
numerics are now fully specified, and the remaining risk is concentrated in component 1
(bringing WMMA into a scalar-FMA family) and component 3 (the second kernel's state handoff).
