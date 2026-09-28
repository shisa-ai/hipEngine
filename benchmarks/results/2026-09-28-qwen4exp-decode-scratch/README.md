# Q5_1 decode owner across first scratch allocation

The current Q5_1 down/combine owner shows no large persistent slowdown after
the first scratch-using GDN dispatch in this screen. No runtime change or
optimization is proposed by this result.

- Model: unsloth Qwen3.8-Flash-Next `UD-Q4_K_XL`, real
  `blk.0.ffn_down_exps.weight`, checked Q5_1 and shape512/2560/640.
- Victim: logical256/t64 weighted-sum kernel, top-k10, in640/out2560,
  synthetic BF16 activations and deterministic synthetic routes.
- Trigger: wave-norm GDN, tokens64/Hk16/Hv48/Dk128/Dv128, FP32 inputs/state.
- Host: Framework machine `55ea6c509d0b49eea8de7094a1023668`, Radeon 8060S/gfx1151,
  TheRock 10 / HIP 7.15.26333.
- [Artifact](artifact.json): actual loaded library hashes, weight hash, full
  event samples, output hashes, clock snapshots and compact profiler evidence.

Four serial fresh processes each run three victim warmups, 31 before samples,
one GDN trigger (or victim sham), and 31 after samples. No GDN kernel runs
before the before-window. The GDN strict-reference check runs after both
windows to avoid contaminating the baseline.

| Arm | Before median ms | After median ms | After / before |
| --- | ---: | ---: | ---: |
| sham | 0.099507 | 0.097022 | 0.975027 |
| scratch | 0.097984 | 0.097103 | 0.991009 |
| scratch | 0.097824 | 0.097183 | 0.993447 |
| sham | 0.095058 | 0.094978 | 0.999158 |

Victim outputs match byte-for-byte before/after and across all four processes.
GDN output and state match the existing strict parent. Boundary snapshots
show SCLK2900/MCLK1000 MHz throughout; continuous clocks are not recorded.
These are repeatability checks on existing arithmetic, not numerical admission.

A separate cached profiler run confirms six zero-scratch victim calls before
the first 1,638,400-byte scratch allocation, followed by one 36-byte GDN call
and three zero-scratch victim calls on queue1. The strict GDN oracle then runs
outside the timing windows. Thus the intended first-scratch ordering is
observed rather than assumed.

This covers one Q5_1 layer, not gate/up or Q8_0 decode victims, dense-wide as the
first trigger, every route, or serving latency. X-5 remains scoped evidence,
not universal scratch-throttle clearance.

## Commands

Use the exact environment exports from the
[scratch-growth packet](../2026-09-28-qwen4exp-scratch-growth/README.md).

```bash
MODEL=/home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL
i=0
for arm in sham scratch scratch sham; do
  "$ENV_PREFIX/bin/python" scripts/qwen4exp_decode_scratch_probe.py \
    --model-root "$MODEL" --arm "$arm" --repeats 31 \
    --output "/tmp/qwen4exp-scratch-audit/decode-order-$i-$arm.json" || exit
  i=$((i+1))
done
rocprofv3 --kernel-trace --scratch-memory-trace --output-format csv \
  --output-directory /tmp/qwen4exp-scratch-audit/decode-order-trace -- \
  "$ENV_PREFIX/bin/python" scripts/qwen4exp_decode_scratch_probe.py \
  --model-root "$MODEL" --arm scratch --repeats 3 \
  --output /tmp/qwen4exp-scratch-audit/decode-order-profiled.json
python3 -m py_compile scripts/qwen4exp_decode_scratch_probe.py
```
