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

## Addendum 1: RETRACTED - the kernel named here is the PREFILL owner, not the decode owner

> **RETRACTED (`20260929T`, see Addendum 3).** The kernel identified below is real and its
> geometry is correct, but it **does not execute during decode**. All 192 of its dispatches in
> this capture fall **before** the measured decode repetitions; **zero** fall inside them. It was
> found by ranking total time across the whole 68 s trace, which is prefill-dominated, so it is
> the **prefill** owner of the routed gate+up work. The decode owner is a different kernel.
> Addendum 3 replaces this section. The text is kept so the error and its cause stay legible.

The capture's raw kernel trace
(`../2026-09-29-qwen4exp-x4-decode-split/rocprof/x4decode_kernel_trace.csv`, 18,941 dispatches)
spans **68 s** of prefill + warmup + measured steps, so it cannot segment a single decode step
without redoing the join that `attribution.json` already performs. One structural fact is
nevertheless readable from it.

**The expert GEMM owner is `gguf_q4_k_selected_dual_wmma_iu8_risk_*`.** The `selected` refers to
*selected experts* (MoE routing), and **`dual` confirms gate and up are fused into a single kernel**.
It appears 188 times in the capture totaling 2,709.02 ms, at **`grid = 1280`, `workgroup = 128`**.
**This is the PREFILL owner.** Its `grid = 1280` geometry describes prefill, and Addendum 3 shows
the decode owner has a different name and a different shape.

**1,280 is exactly the gate+up output row count for one expert** (640 gate rows + 640 up rows). So the
decomposition is **one workgroup per output row, each reducing over 2,560 input columns** - a
128-thread workgroup spanning the full reduction axis. **This describes the prefill kernel.** That is a plausible source of the 2.77x gap
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

## Addendum 3: the decode owner, and the correction of Addendum 1

`scripts/qwen4exp_de1_moe_per_kernel.py` (artifact `per_kernel.json`) splits the `moe:expert_gate`
family into its raw kernels and charges each one the traffic it actually moves. It joins two
artifacts: the harness renames every **decode** dispatch to its semantic role, so
`rocprof/x4full_kernel_trace.csv` carries the attribution but not the raw name; and
`rocprof/x4full_results.db` maps `Kernel_Id` back to the mangled raw kernel name. The decode window
comes from the `qwen4exp_decode_live4097_rep{0,1,2}` ROCTX ranges, and every figure is divided by the
three reps.

**The join reproduces the family total exactly - 24.550 ms/step - which is the check that it is
correct rather than merely plausible.** It also reproduces the published kernel sum: 162.654 ms over
the three reps = 54.218 ms/step.

### The correction

**Addendum 1 named the wrong kernel.** All 192 dispatches of `selected_dual_wmma_iu8_risk_*` fall
**before** the decode repetitions and **zero** fall inside them, as do all 430 `q5_1_selected` and all
192 `router_logits` dispatches. Those are **prefill** kernels. The error's cause is worth recording:
Addendum 1 ranked kernels by total time across the whole 68 s capture, and the measured decode reps
are only 0.68 s of that, so a prefill-dominated ranking cannot identify a decode owner. The decode
kernels are also **not identifiable by raw name in that file at all** - all 5,313 decode dispatches
carry renamed role names, and all 13,628 prefill dispatches carry raw names.

### The decode owner

| raw kernel | bucket | ms/step | kernels/step | us/kernel | bytes/step | **GB/s** | **% roof** |
|---|---|---:|---:|---:|---:|---:|---:|
| `gguf_q4_k_selected_dual_silu_logical128_t64` | routed gate+up | **14.013** | 47 | 298.14 | 888.83 MB | **63.43** | **28.7%** |
| `q5_1_selected_weighted_sum_logical256_t64` | routed down | 5.444 | 43 | 126.59 | 615.42 MB | 113.06 | 51.2% |
| `dense_gemv_out_kernel<float,float,false>` | router GEMV | 1.363 | 96 | 14.20 | 251.66 MB | **184.66** | **83.5%** |
| `qwen35_router_select_kernel` | top-k select | 0.331 | 48 | 6.90 | ~0 | - | - |

Charged **20.819 ms/step**; **3.731 ms/step (15.2%)** is reported unattributed rather than
distributed across the charged kernels.

### Two in-block comparators, and a corrected router term

**The family figure of 81.73 GB/s is a blend of kernels running at 63.43, 113.06 and 184.66 GB/s.**
The gate+up GEMV is simultaneously the largest owner (57.1% of the family) and the least efficient.

**The router term in the family-level model is wrong, and this artifact corrects it.**
`qwen35_router_select_kernel` is the **top-k selection over 512 logits**, not the router GEMV: it
takes 0.331 ms/step, so charging it the router's 251.66 MB would imply **760 GB/s, 3.4x the
documented roof** - an impossible rate that the family-level model absorbed because it charges the
router's bytes to the family rather than to a kernel. The actual router GEMV is
`dense_gemv_out_kernel<float,float,false>`, the only fp32 dense GEMV in the block, which matches
`ffn_gate_inp` exactly (5,242,880 B/layer = 512 x 2560 at 4 B/weight). At 1.363 ms/step it reaches
**184.66 GB/s = 83.5% of roof** - a **second comparator inside the same block**, and a stronger one
than `linear:root.lm_head` because it is the same role family, the same step, and the same shape
class (a dense GEMV over a single token).

### Headroom, now localized

At the `linear:root.lm_head` comparator rate (226.67 GB/s) the gate+up GEMV's 888.83 MB would take
**3.921 ms** instead of 14.013 ms - a **10.091 ms/step** saving, **18.6% of the 54.218 ms kernel
sum**. That is larger than the 15.70 ms/step the family-level gate attributed to the whole block,
because the family figure was diluted by the router's impossible bytes.

### Limits

Same profiled capture, one case, one live count, one model, one host, as above. The gate+up and
down mappings are high confidence - the kernel names carry both the quant (`q4_k`, `q5_1`) and the
role (`selected_dual_silu`, `selected_weighted_sum`), and the per-step dispatch counts (47 and 43
over 48 layers) match one dispatch per layer, with the shortfall expected from this model's
**dynamic** quantization (layer tensor sizes are not uniform). The router-GEMV identification is
medium confidence: it rests on the kernel being the only fp32 dense GEMV in the block and on
`ffn_gate_inp` being exactly fp32, not on a name match. **No timing claim is made** - the artifact
inherits the source capture's own limit that its per-kernel times are diagnostic.

## Addendum 4: the production decode path is already the exact variant - and the dp4a revert does not cover it

Addendum 3 measured the decode owner as `gguf_q4_k_selected_dual_silu_logical128_t64` - the
**non-dp4a** variant. That raised an obvious question: which variant does the production profile
actually select? The answer changes what `DE-1`'s action should target.

### The capture is the production profile

`scripts/qwen4exp_context_decode_profile.py` builds its generator with
`_make_generator(factory_args, "production")`, so the capture is the **production** profile. Its
report labels it `"execution_profile": "production"`.

### Production selects the exact variant, via the recovery block

`hipengine/generation/qwen4_exp_profiles.py` defines two selection sets:

- **`_strict_selections()`** selects `selected_dual_silu_logical128_t64` (exact activation).
- **`_production_selections()`** selects `selected_dual_q8_1_dp4a_silu_logical128_t64` (dp4a) with
  `evidence=_DECODE_EVIDENCE` - but for this model's quant (`gguf_ud_q4_k_xl`) the call site passes
  **`recovery=True`**, and the recovery block then applies:

  ```python
  replacements = {
      "selected_dual_q8_1_dp4a_silu_logical128_t64_gemv_bf16_bf16_out":
          "selected_dual_silu_logical128_t64_gemv_bf16_bf16_out",
  }
  ```

  under `_ARITHMETIC_RECOVERY_EVIDENCE`. **So production decode runs the exact variant**, which is
  why the trace shows `gguf_q4_k_selected_dual_silu_logical128_t64_kernel`. **Addendum 3's 63.43 GB/s
  is therefore a production-path measurement, not a strict-profile artifact.**

### The warp256 structure exists only as dp4a

`DE-1`'s action is "an exact-activation variant of the warp256 structure (the structure, not the
dp4a arithmetic)". The tree contains:

- `gguf_q4_k_selected_dual_q8_1_dp4a_silu_warp256_gemv_bf16_bf16_out` (gate/up)
- `qwen4_exp_q5_1_selected_weighted_sum_warp256{,_fast}_bf16_bf16_out` (down)

with docstring: *"One block per output column with 32\*rows threads; warp w serves expert row w
(rows <= 12); lanes stride the 4-element groups by 32 ... Warp shuffle reductions, no barriers."*

**Every warp256 variant is dp4a. No exact-activation warp256 exists.** So the row's action is a real,
unbuilt unit - and the recovery's replacement of dp4a-logical128 with exact-logical128 does **not**
cover it.

### What the recovery evidence actually covers

`_ARITHMETIC_RECOVERY_EVIDENCE` is `benchmarks/results/2026-09-14-q8-prefill-numerics/artifact.json`.
That artifact's failing candidate is `production_q8_fallback`, mechanism *"strict Q8 selected-down
and dense Q8 MMQ fallback; **other production paths unchanged**"*, and it failed on **task quality,
not numerics**: the candidate produced *"Llama-2-7B uses GQA, reducing KV cache size by 4x compared
to MHA"*, a claim absent from the strict arm. Its `task_review.status` is
`rejected_for_default_promotion`.

**That candidate did not change the decode linear.** Yet the recovery's `replacements` dict sweeps
the decode dp4a linear into the revert. The recovery's own gate then passes
(`conservative_recovery.status: passed`, `task_gate: passed_exact`, 18/18) - which shows the
conservative configuration works, **not** that the decode dp4a linear fails.

### The decode dp4a linear has its own passing acceptance

`_DECODE_EVIDENCE` is
`benchmarks/results/2026-08-29-gfx1151-qwen38-flash-next-production-dp4a-safe43-decode.json`:

- `status: **accepted_explicit_production_profile**`
- `numerical_gate.mean_kl: **2.72e-4**`, `max_kl: **5.77e-3**`

For comparison, the warp256 dp4a variant failed admission at **max KL 0.121-0.431** - roughly 20-75x
worse. **These are different arithmetic outcomes and should not be treated as one class.**

### What this does and does not establish

**Establishes:** the production decode path is the exact variant; the warp256 structure has never
been built with exact activations; and the decode dp4a linear was reverted by a conservative sweep
whose triggering candidate explicitly left it unchanged, despite having its own accepted production
evidence.

**Does NOT establish** that re-admitting the decode dp4a linear is safe today, or that it would pass
the current production envelope. The recovery was a deliberate conservative response to a real task
failure, and reverting non-exact owners broadly was a reasonable choice at the time. **No speedup is
claimed here**, and the row's 2.0x/5.8x figures belong to the warp256 variant, not to the logical128
dp4a variant that was reverted.

**Two candidate next steps follow, and they are different units:** (1) re-run the profile gate on the
decode dp4a logical128 owner under the current production envelope - a gate run, no kernel work; and
(2) build the exact-activation warp256 variant the row actually asks for - real kernel work. Neither
is licensed by this addendum.

## Addendum 5: the authoritative family split, and the shared-expert upper bound

Joining the harness's renamed decode dispatches to `rocpd_info_kernel_symbol` also yields the full
family breakdown, which sums **exactly** to the published kernel sum:

| family | ms/step | kernels/step |
|---|---:|---:|
| **moe** | **24.550** | 583 |
| **linear** | **23.166** | 545 |
| qsa_decode | 2.621 | 120 |
| gdn | 2.401 | 72 |
| gr_read | 0.935 | 292 |
| decode_boundary | 0.508 | 151 |
| ple | 0.036 | 8 |
| **total** | **54.218** | **1,771** |

### A correction this forces

`gguf_k_pack8_prefill_out_kernel<float,float,8>` totals **23.997 ms/step across 497 kernels/step**,
of which **22.184 ms/step is the `linear` family** and only 1.812 ms/step is inside `moe`. So it is
predominantly **DE-3's** owner, not a shared-expert kernel. An earlier draft of this artifact
considered it a shared-expert candidate; the family split excludes that.

### The shared-expert upper bound (for DE-2)

Inside `moe`, the two routed owners are named and charged to exact GGUF payloads:

- routed gate+up `gguf_q4_k_selected_dual_silu_logical128_t64` — **14.013 ms/step** (888.83 MB)
- routed down `q5_1_selected_weighted_sum_logical256_t64` — **5.444 ms/step** (615.42 MB)

**Therefore the entire remainder of the moe family — shared expert gate/up/down AND the router GEMV
AND the top-k select AND the combine/norm/misc kernels — is 24.550 - 14.013 - 5.444 = 5.093
ms/step.** The shared expert moves 250.75 MB/step and the router 251.66 MB/step, so that remainder
moves **~502.41 MB in ≤5.093 ms → ≥98.6 GB/s**.

**DE-2 records the shared block alone at 6.19 ms/token, which is larger than the measured cost of
shared + router + every other non-routed kernel in the block.** That does not prove the record wrong —
the two figures come from different days, models and compositions — but it does mean the record
cannot be used as-is to size a screen.

### What is still missing for DE-2

Which specific kernels serve `ffn_gate_shexp` / `ffn_up_shexp` / `ffn_down_shexp` versus the router.
The per-kernel names do not say, and the candidate kernels
(`gguf_k_pack8_prefill_out_kernel` 1.812, `gguf_k_selected_prefill_out_kernel` 0.871 + 0.474,
`dense_gemv_out_kernel` 1.363) are not resolvable by name alone. **A name-to-tensor mapping, or a
scoped marker around the shared-expert call, is what the row still needs.**

Shared work *is* present in the capture: `shared_gate_combine_out_kernel` runs **48 times/step
(1 per layer) at 0.067 ms/step**, inside `moe:layers.N.expert_gate`. The earlier note that the shared
expert "has NO role in the decode capture at all" is correct about **role names** but too strong as a
conclusion — it is located, and bounded above as above.
