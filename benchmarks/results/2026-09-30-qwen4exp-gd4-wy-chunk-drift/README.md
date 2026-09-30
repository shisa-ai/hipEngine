# GD-4 screen (correct form): the WY reassociation is numerically free — the entire cost is the fp16 operand class

**Status: SCREEN COMPLETE.** This replaces
`benchmarks/results/2026-09-30-qwen4exp-gd4-chunk-scan-drift/`, which modelled **gated linear
attention** and was retracted — GDN is the gated **delta** rule.

## Question

`GPF-8` built a correct chunkwise/WY GDN kernel (**byte-exact at one token**) and still failed
the model gate at **KL 0.056522**, at FP32. Two possibilities: the WY reassociation itself is
numerically dangerous, or GPF-8's KL came from something else. This separates them by measuring
the WY form's fp32 drift against the serial form's fp32 drift on identical inputs, both scored
against the float64 oracle.

## Result — sequence 512, Hk16/Hv48/Dk=Dv=128, beta∈[0.1,0.9], decay∈[0.7,0.999]

| variant | out NMSE | state NMSE | max abs dout | vs serial fp32 |
| --- | ---: | ---: | ---: | ---: |
| serial fp32 (production form) | 5.359e-14 | 3.169e-15 | 3.725e-08 | 1.00x |
| wy fp32 chunk 8 | 4.265e-14 | 1.116e-14 | 2.608e-08 | 0.80x |
| wy fp32 chunk 16 | 3.856e-14 | 1.117e-14 | 2.421e-08 | 0.72x |
| wy fp32 chunk 32 | 3.610e-14 | 1.121e-14 | 2.235e-08 | 0.67x |
| wy fp32 chunk 64 | 3.477e-14 | 1.106e-14 | 2.235e-08 | **0.65x** |
| serial **fp16 operands** | 1.297e-07 | 4.423e-08 | 3.744e-05 | 2.4e6x |
| **wy fp16 operands chunk 64** | **8.637e-08** | **6.093e-10** | 2.446e-05 | 1.6e6x |
| wy fp16 operands chunk 128 | 8.640e-08 | 6.093e-10 | 2.446e-05 | 1.6e6x |

Oracle cross-check (both float64, committed CPU reference): `NMSE(wy8, serial)` out
**1.556e-14**, state **6.768e-15**. My fp32/f64 port of the WY form matches the committed oracle
at **6.4e-16** in both single-chunk and multi-chunk (carry-exercised) configurations.

## Finding 1 — the reassociation is numerically free

**In fp32 the WY form drifts at 0.65–0.80x the serial form's own fp32 drift**, and it gets
*better* as the chunk grows. A chunk size of 64 adds nothing measurable over chunk 8 or over the
serial chain.

**So the WY reassociation cannot explain GPF-8's KL 0.056522.** That failure must come from
something else — most plausibly the **direct-conv Q/K/V input path** GPF-8 bundled with the
recurrence change. One-token byte-exactness argues the input path is arithmetically equivalent at
`t=0`, but it does not establish equivalence over a long sequence, and the artifact never
separated the two by measurement.

## Finding 2 — the entire cost is the fp16 operand class, and chunking *reduces* it

`serial fp16 operands` costs **1.297e-07** out NMSE. `wy fp16 chunk 64` costs **8.637e-08** —
**0.67x the serial fp16 form**, i.e. chunking with fp16 operands is *more* accurate than the
serial chain with fp16 operands, not less.

The state result is stronger: serial fp16 state NMSE **4.423e-08** against wy fp16 chunk 64's
**6.093e-10** — **73x better**. That is structural: the WY form computes the chunk's state
contribution as one sum, where the serial chain applies 512 successive fp16 roundings to the
carried state.

## What this means for GD-4

1. **The numerical risk GD-4 was carrying is not the chunking.** It is exactly the risk of
   running the recurrence with **fp16 operands**, which is `MX-1`'s F16-operand class — and
   `MX-1`'s F16 route failed the `prefill_last` scope while sitting 8.0–9.5x *below* the BF16
   floor already in the path.
2. **GPF-8 already demonstrated the speed at FP32**: 210.501 ms → 47.491 ms on the 30-layer 512
   GDN family, **4.43x**, at **48 VGPR** (against the production route's 256). The structural win
   — chunk-level matmuls, lower register pressure — is what bought that, **not** the operand
   format.
3. **So the recommended build is the chunked WY form with FP32 operands.** It keeps GPF-8's
   measured 4.43x, and this screen shows its arithmetic drift is *below* the current production
   path's. fp16 WMMA adds a speed increment whose size is unmeasured here while adding the entire
   numerical risk.
4. **Keep the existing input path.** Since the reassociation is free in fp32, GPF-8's model-gate
   failure is evidence against the *direct-conv* change, not against chunking.

## Limits

- **Synthetic inputs and a NumPy model of the arithmetic, not the real kernel.** This measures
  arithmetic drift only.
- **Drift is not KL.** GPF-8's own numbers are the caution: a primitive inside frozen tolerance
  still failed the model gate. The `prefill_last` scope has now failed two non-exact routes, so
  the fp16 arm must be gated, not assumed.
- **The 4.43x is GPF-8's, on Qwen3.6-35B-A3B `q4_k_m`** at C=8, not on qwen4exp at
  Hk16/Hv48/Dk=Dv=128.
- **`np.tril` on a 3-D array masks the last two axes, not the first two.** The first version of
  this screen used it for a `[t,s]` causal mask and silently masked `[s,h]` instead, producing a
  chunk-size-independent state error of 0.9263. Fixed with an explicit `np.where` mask on
  `[t,s]`. The self-check against the committed oracle is what caught it, and it is now part of
  the script.

## Artifacts

- `screen.json` — full payload.
- Harness: `scripts/qwen4exp_gdn_wy_chunk_drift_screen.py`.
- Worklog: `worklog/entries/20260930T16*-lhl-qwen4exp-gd4-wy-chunk-drift*.md`.
