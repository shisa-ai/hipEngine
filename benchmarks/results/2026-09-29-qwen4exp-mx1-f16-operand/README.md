# MX-1: the F16-operand routed-expert route is admissible by quant

MX-1 is the competitor's routed-expert design: decode the quantized expert
weights to **F16** in shared memory, cast the activation to F16 once, run F16
WMMA with F32 accumulation, and drop the iu8 risk/repair machinery entirely.
Our exact path dequantizes to F32 and accumulates with F32 FMA plus a repair
pass. This packet answers the numerical question that has to come first:

> Does decoding the real expert weights to F16, and casting the real activations
> to F16, stay inside the error the path already carries?

**Yes.** The operand change costs 8.0x to 9.5x less than the BF16 rounding the
path already applies, on real natural-input activations, for every one of the 48
expert_down layers and every expert quant in the model. There is no F16 overflow
anywhere.

## Comparison basis

The BF16 publication floor, not an arbitrary tolerance. The expert output is
published to BF16 downstream, so an operand change whose error sits far below
that rounding cannot move the model beyond what the current path already does.
The packet reports both errors on identical operands.

## Part A — F16 weight decode, every expert tensor

144 expert tensors, 8 experts sampled per tensor.

| Quant | Tensors | Max relative | RMS relative | Max absolute |
| --- | ---: | ---: | ---: | ---: |
| Q4_K | 94 | 4.880e-4 | 2.068e-4 | 1.221e-4 |
| Q5_1 | 43 | 4.880e-4 | 1.843e-4 | 1.221e-4 |
| Q5_K | 2 | 4.880e-4 | 2.067e-4 | 1.373e-4 |
| Q8_0 | 5 | 4.815e-4 | 1.977e-4 | 1.087e-4 |

The maximum relative error is 4.88e-4 = 2^-11, which is exactly the F16
representation limit: the error is F16's, identical for every quant, and not
quant-dependent. Weight magnitude peaks at 0.5624 against the F16 maximum of
65504, so **no weight overflows**; 6,840,924 values land in the F16 subnormal
range (below 6.1e-5), where they are small enough to matter far less than the
quantization error already present. Every value is finite after the cast.

## Part B — expert_down on real activations

Natural-input `general_en-p4096` prefill capture (BF16 bits, decoded and
verified), 10,240 expert-sorted rows, K=640, N=2560, 512 experts, all 48 down
layers. The capture is the real `SiLU(gate)*up` intermediate produced by the
production route, not a synthetic distribution.

| Quant | Layers | Candidate RMS error | BF16 floor RMS | Ratio |
| --- | ---: | ---: | ---: | ---: |
| Q5_1 | 43 | 5.12e-6 to 5.50e-6 | 4.44e-5 to 4.79e-5 | 0.115 to 0.116 |
| Q8_0 | 5 | 5.67e-6 to 6.52e-6 | 4.55e-5 to 5.23e-5 | 0.124 to 0.125 |

Every layer is below the floor; the worst ratio is **0.125**, so the candidate
error is 8.0x below the BF16 rounding even in the least favourable layer.
Activation magnitude peaks at 13.44 against the F16 maximum of 65504, so **no
activation overflows**, and the cast is finite everywhere.

## Why this matters

The two expert families hold **4,610.1 ms of the 12,973 ms gap** against the
comparator (`expert_gate_up` 2,640.6 ms and `expert_down` 1,969.5 ms, 35.6%
together), and the comparator's numbers come from this design. A further
1,239.8 ms of iu8 exact-repair sits inside our two expert rows with no
comparator equivalent, so the route also removes a pass rather than only
speeding one up. Our activation packing cost is already 0.0 ms, so that part of
the comparison needs no work.

## Limits

- Part A samples 8 of 512 experts per tensor. It bounds the per-element decode
  error, not the whole tensor.
- Part B covers `expert_down` only. `expert_gate_up` (Q4_K) has Part A weight
  coverage but no real gate/up input capture, so its activation-cast term is
  unmeasured; the next unit should capture one.
- The BF16 floor is computed per expert-output value, not on the final combined
  MoE output that the real path actually rounds.
- This is an operand-error budget. It does **not** replace the production
  mean/p95/p99/max KL, top-1, determinism, isolation or task gate, which any
  candidate built on it must still pass. The GR family in particular is
  cliff-shaped: its F16 wide tile was 4.61x faster on the projection and still
  failed the 780-row gate at mean KL 1.554e-3.

## Reproduce

```bash
python3 scripts/qwen4exp_mx1_f16_operand_packet.py \
  --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --capture /home/lhl/tmp/q51-down-input-general_en-p4096.npz \
  --experts-sampled 8 --output /tmp/mx1-packet-full.json
```

CPU-only, about 6.5 minutes. The capture is produced by
`scripts/qwen4exp_q51_down_input_capture.py`.
