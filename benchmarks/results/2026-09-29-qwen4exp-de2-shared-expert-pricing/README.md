# DE-2 shared expert: name-to-tensor mapping and cost pricing

Resolves the mapping the DE-2 row names as its blocker - *"A name-to-tensor mapping (or a scoped
marker around the shared-expert call) is what this row still needs before its action can be
screened"* - and prices the block from it. Correctness/pricing evidence; `performance_claim: false`.

- **Script:** `scripts/qwen4exp_de2_shared_expert_pricing.py`
- **Model:** unsloth Qwen3.8-Flash-Next `UD-Q4_K_XL`, 4 shards, 1,224 tensors
- **Source:** read-only GGUF header parse; no GPU, no model load

## The mapping

The shared expert is **192 tensors = 48 layers x 4**, and it is a **Q8_0 block, not Q4_K**:

| tensor | shape | type | bytes/layer | count |
|---|---|---:|---:|---:|
| `ffn_gate_shexp.weight` | [2560, 640] | **Q8_0** | 1,740,800 | 48 |
| `ffn_up_shexp.weight` | [2560, 640] | **Q8_0** | 1,740,800 | 48 |
| `ffn_down_shexp.weight` | [640, 2560] | **Q8_0** | 1,740,800 | 48 |
| `ffn_gate_inp_shexp.weight` | [2560] | F32 | 10,240 | 48 |

Types across the 192: **144 Q8_0 + 48 F32**. There are no Q4_K or Q5_1 shared-expert tensors.

**Per-step weight traffic: 251,166,720 B = 251.17 MB** (5,232,640 B/layer x 48).

This independently reproduces the 250.68 MB figure the family traffic model already carried, from the
actual payload sizes rather than a bits-per-weight estimate.

**Why this matters for the row's action:** the shared expert being uniformly Q8_0 is exactly why the
row cites gufo's `qfn_mmq_q8_0_dense_vec_preq` ("one fused gated Q8_0 GEMV over activations
quantized once ... with two weight pointers"). The fused-gate+up action is a Q8_0 fusion, and it is a
different quantization class from the routed experts, which are Q4_K (gate+up) and Q5_1 (down).

## The pricing

The block cannot be priced from its own marker, but it is **bounded above by the measured non-routed
remainder** of the moe family:

```
moe family                      24.5504 ms/step
- routed gate+up                 14.0126   (gguf_q4_k_selected_dual_silu_logical128_t64)
- routed down                     5.4440   (q5_1_selected_weighted_sum_logical256_t64)
= non-routed remainder            5.0938 ms/step
```

That remainder contains the shared expert **and** the router GEMV **and** the top-k select **and** the
combine/norm/misc kernels. It moves 251.17 MB (shared) + 251.66 MB (router) = **502.82 MB**, so it
runs at **>= 98.71 GB/s**.

Bracketing the shared expert's own cost:

| basis | rate | shared expert cost | % of 54.218 ms decode kernel sum |
|---|---:|---:|---:|
| in-block comparator (router dense GEMV) | 184.66 GB/s | **1.360 ms/step** | 2.51% |
| non-routed remainder's implied rate | 98.71 GB/s | **2.544 ms/step** | 4.69% |

**So the shared expert costs between 1.36 and 2.54 ms/step.**

## The row's 6.19 ms/token cannot be right

The historical figure is **larger than the measured cost of the entire non-routed block** - shared
expert plus router plus select plus combine plus every other non-routed kernel, all of it, is
5.0938 ms/step. A component cannot exceed its container.

## What this means for the row's action

The action - screen a fused shared gate+up with unchanged multi-plane/per-output arithmetic - remains
valid and is now **priced**:

- **Hard ceiling:** a *perfect* (zero-time) shared expert recovers **at most 2.544 ms/step = 4.69%**
  of the decode kernel sum. That is the entire prize for this row.
- **Realistic:** the fusion targets gate+up, which is 2/3 of the traffic (167.44 MB of 251.17 MB). At
  a 2x improvement on that portion the saving is ~0.45-0.85 ms/step, i.e. **~0.8-1.6%**.
- **Ranking:** DE-1's routed gate+up owner is 14.013 ms/step with a measured 10.091 ms/step of
  headroom. DE-2's ceiling is 2.544 ms/step. **DE-1 is roughly 4x the prize of DE-2's entire
  ceiling**, which is consistent with the ledger ranking routed experts above the shared expert.

## Limits

- **The GGUF mapping is exact** (parsed payload sizes), but the **cost split within the remainder is
  not measured** - the bracket is a bound, not an attribution. The shared expert's true cost inside
  [1.36, 2.54] ms/step is unresolved without a scoped marker or a name-to-kernel join, which this
  capture cannot supply: all 5,313 decode-rep kernels carry renamed role names
  (`qwen4exp_role:moe:layers.N.expert_gate`), and no `shared_*` marker exists in it.
- **The 184.66 GB/s comparator** is the router dense GEMV from the same step, itself derived from a
  profiled capture whose per-kernel times are diagnostic rather than a serving measurement.
- **One case, one live count, one model, one host** (gfx1151).
- **No speedup claimed and no kernel built.** This is a pricing result.
