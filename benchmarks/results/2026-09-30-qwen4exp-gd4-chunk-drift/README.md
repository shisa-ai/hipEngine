# GD-4: does a chunked GDN scan drift? Screen says no, and here is the hazard it does have

Ledger row `GD-4` proposes replacing the sequential GDN prefill recurrence with a chunked
scan on fp16 WMMA (halo-box PR #91: fused conv/SiLU, q/k norm, gated RMS norm). Its own
action is explicit that this needs a **numerical/state screen first**, because the
recurrence carries state across the whole prompt and a chunked schedule changes where that
state is carried and in what order the per-token terms are summed.

This packet is that screen. **No kernel was built.**

## Why the question is well posed

The recurrence is affine in the state. Writing `D_t = prod_{i<=t} d_i`:

```
d_t     = exp(-exp(a_log) * softplus(alpha_t + dt_bias))
delta_t = beta_t * (v_t - k_t.(S_0 D_t) - sum_{j<t} (k_t.k_j)(D_t/D_j) delta_j)
S_t     = S_0 D_t + sum_{j<=t} (k_j (x) delta_j)(D_t/D_j)
o_t     = q_t . S_t,  then RMS-norm and gate
```

The first two lines are the UT/WY transform: a lower-triangular solve, **exactly**
equivalent to the sequential recurrence. So a chunked scan is not an approximation, and
every figure below is floating-point reordering cost rather than modelling error.

The screen reuses `replay_gdn` from `scripts/gguf_gdn_replay_compare.py` as the sequential
reference, so it and the existing GPU-verification harness agree on the arithmetic by
construction rather than by transcription.

## Method

`scripts/gdn_chunk_drift_screen.py --tokens 4096 --chunks 32 64 128 256 --tails 4095 4097 4100`

Qwen4Exp's GDN shape (`Hk16/Hv48/Dk=Dv=128`, the shape GD-3's screen used), fp32 throughout
except where the state carry is deliberately narrowed to fp16. Three effects are separated:
summation order inside a chunk, state-carry precision at chunk boundaries, and tail lengths
that are not multiples of the chunk size.

**RED test first.** Before any drift number was trusted, the chunked form had to converge to
the sequential one as the chunk grows. It does: at T=8/32/64/128/256 the fp32 NMSE is
4.6e-14 to 7.4e-14 with no growth in T, and smaller chunks are marginally better than larger
ones. A form that is algebraically identical and correctly ordered cannot do anything else,
so this is the check that the derivation is right.

## Result: 4K tokens

| chunk | state | max_abs | rms_abs | nmse | p99_rel |
| ---: | --- | ---: | ---: | ---: | ---: |
| 32 | fp32 | 8.166e-06 | 1.137e-07 | **4.558e-14** | 6.853e-06 |
| 64 | fp32 | 1.347e-05 | 1.422e-07 | 7.123e-14 | 9.902e-06 |
| 128 | fp32 | 1.347e-05 | 1.420e-07 | 7.109e-14 | 9.902e-06 |
| 256 | fp32 | 1.359e-05 | 1.420e-07 | 7.101e-14 | 9.898e-06 |
| 32 | fp16 | 7.703e-03 | 1.503e-05 | 7.961e-10 | 2.836e-04 |
| 64 | fp16 | 4.256e-03 | 1.046e-05 | 3.855e-10 | 1.294e-04 |
| 128 | fp16 | 3.041e-03 | 7.195e-06 | 1.824e-10 | 5.207e-05 |
| 256 | fp16 | 3.041e-03 | 5.046e-06 | 8.973e-11 | 2.369e-05 |

Reference output rms is 0.5327, so an fp32 max_abs of 1.4e-05 is ~2.6e-05 relative.

## Result: chunk/tail boundaries (chunk 128, fp32 state)

| tokens | max_abs | nmse | p99_rel |
| ---: | ---: | ---: | ---: |
| 4095 | 1.645e-05 | 7.357e-14 | 9.705e-06 |
| 4096 | 1.359e-05 | 7.101e-14 | 9.898e-06 |
| 4097 | 2.170e-05 | 7.264e-14 | 9.745e-06 |
| 4100 | 1.603e-05 | 7.171e-14 | 9.782e-06 |

**No boundary effect.** A partial final chunk costs nothing measurable, and the three
non-multiple lengths sit inside the spread of the exact-multiple case.

## What this says about the row

1. **A chunked reformulation does not drift.** At fp32 state carry the error is
   indistinguishable from fp32 round-off (NMSE ~5-7e-14) and is flat in the chunk size.
   The recurrence's affine structure is what makes this true, and it is why the row's own
   reference could claim a small NMSE at all.
2. **An fp16 state carry costs ~4 orders of magnitude and is still tiny** - NMSE 9e-11 to
   8e-10. Notably it *improves* with larger chunks, the opposite of a naive
   longer-horizon-is-worse intuition, because fewer boundary roundings occur. Even the
   worst case is **~1000x better than the row's cited reference figure** (halo-box PR #91's
   "NMSE ~2e-7 against the sequential kernel"), which is what an fp16 WMMA path would be
   measured against.
3. **The row's stated blocker - "numerical/state screen first" - is cleared.** The screen
   was the gate on this row, and it passes. The remaining work is the kernel build, not
   more numerics.

## The hazard the screen did find, and it is not the one the row expected

The first revision of the screen computed `D_t/D_j` as a **ratio of prefix products**,
`cum[t] / max(cum[j], tiny)`. That form is numerically unsafe, and it failed the RED test
loudly: max_abs 1.05e1 at chunk 256 against 2.26e-01 at chunk 64 - **larger chunks getting
worse**, which is the signature of an explosion rather than of accumulation.

The mechanism: `cum` decays geometrically and underflows to zero, but not simultaneously
across tokens. While `cum[j]` has underflowed and `cum[i]` has not, the clamped ratio
`tiny/1e-30` becomes enormous, the triangular matrix `L` blows up, and the solve diverges.
The floor on the denominator does not help - it is what creates the blow-up.

The fix is a **direct suffix product**: `R[i, j] = prod_{k=j+1..i} decay[k]`, walking
backwards from `i`. It never divides, and each factor is a decay in (0, 1], so every entry
is bounded by 1 and underflows monotonically to zero. That is the form in the committed
script, and it is the form any kernel implementing this must use.

**This is the one thing in this packet a kernel author most needs**, and it is exactly the
class of finding the screen existed to produce: the chunked scan is algebraically safe, and
the way to implement it is not the obvious way.

## Limits

- Synthetic inputs at the right shape and scale, not captured model activations. The screen
  measures reordering and carry precision, which is what the row asked for; it does not
  measure drift under the real activation distribution.
- One shape (Qwen4Exp's GDN), one seed, fp32 arithmetic with an fp16 carry simulation.
  The fp16 arm simulates the state-carry rounding, not a full fp16 WMMA operand path.
- No timing claim. This packet says nothing about whether the chunked scan is faster; that
  is what the ~5.5% ceiling in `GD-4` and a kernel build are for.

## Files

- `drift.json` - the full result set, machine-readable.
- `scripts/gdn_chunk_drift_screen.py` - the screen.
- Sequential reference: `scripts/gguf_gdn_replay_compare.py::replay_gdn`.
