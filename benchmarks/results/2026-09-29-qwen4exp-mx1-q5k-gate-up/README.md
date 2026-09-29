# MX-1: the Q5_K gate/up layer is admissible, closing the named coverage gap

MX-1's own limit said: *"Part C covers 13 of the 48 gate/up layers (all Q4_K); the two
Q5_K gate/up tensors (layer 2 and one other) are not in the real-activation sample."*

**Measured. Layer 2 is the only Q5_K gate/up layer (47 Q4_K + 1 Q5_K across the 48
layers), and it is admissible.**

| quantity | value |
|---|---:|
| layer | 2 |
| quant | Q5_K |
| experts measured | 310 of 512 (the ones that received tokens) |
| rows | 10,240 (full) |
| `swiglu_error_f16_vs_f32` rms_abs | **1.0677e-05** |
| `swiglu_error_bf16_floor_vs_f32` rms_abs | **6.7016e-05** |
| ratio (rms) | **0.159** -> **6.3x below the floor** |
| max_abs, candidate vs floor | 3.317e-04 vs 3.362e-03 -> **10.1x below** |
| `candidate_below_bf16_floor` | **true** |
| activation overflow | 0 (max magnitude 7.4375) |

The Q4_K layers measured earlier span ratios 0.092 to 0.179 (5.6x to 10.9x below the
floor). **Layer 2's 0.159 sits inside that range**, near the upper end but comfortably
below the floor. So the Q5_K quant is not an outlier: the F16-operand route behaves the
same way there as on Q4_K.

Command (reproduces the artifact):

```
python3 benchmarks/results/2026-09-29-qwen4exp-mx1-f16-operand/qwen4exp_mx1_f16_operand_packet.py \
  --output /tmp/mx1-layer2-q5k.json --skip-part-b \
  --gate-up-capture /tmp/q4k-gate-up-input-general_en-p4096.npz \
  --layers 2 --max-experts 512
```

## A limitation this run exposed, which applies to Part C as a whole

The capture holds **one** activation block - `x_bits` is `(10240, 2560)`, with a single
`expert_start` array and one `case_id`. It is not per-layer. So **every layer Part C
measures is measured against the same MoE block input**, including this run's layer 2.

That makes Part C's figures a **shared-activation operand-error budget**, not a
per-layer-exact measurement. It does not invalidate the budget - the question is how F16
operand rounding compares to the BF16 rounding already in the path, and that is dominated
by the weights and the activation *distribution*, both of which the capture represents -
but the per-layer numbers should not be read as if each layer saw its own hidden state.
This is recorded here because the existing artifact does not state it.

## What this changes

- MX-1's ledger row currently says *"Gate/up has weight coverage only (no real gate/up
  input capture yet)"*. That is **stale**: Part C exists in the artifact and does use a
  real gate/up input capture. The row's real remaining gap was the narrower one quoted
  above, and it is now closed.
- The Q5_K coverage limit in the artifact is closed for layer 2. The artifact's phrase
  "and one other" is wrong: there is only one Q5_K gate/up layer.

## Limits

- **One layer, one capture, one prompt** (`general_en-p4096`), one model, one host.
- Layer 2 measured 310 of 512 experts; the rest received no tokens in this capture.
- The shared-activation limitation above.
- This is an **operand-error budget**. It does not replace the production mean/p95/p99/max
  KL, top-1, determinism, isolation or task gate that any candidate built on it must pass.
  The candidate build and the production gate remain open - MX-1 is not promoted here.
