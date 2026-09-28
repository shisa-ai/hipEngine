# Ordered scratch-growth screen

A dense-wide trigger that grows queue scratch did not produce a large persistent
slowdown on the tested scratch-enabled GDN victim. This is a diagnostic screen,
not an optimization result or full X-5 clearance.

Physical host: Framework `55ea6c509d0b49eea8de7094a1023668`, Radeon 8060S/gfx1151,
TheRock 10 / HIP 7.15.26333. Synthetic Qwen3.8-Flash-Next GDN geometry:
64 tokens, Hk16/Hv48/Dk128/Dv128, FP32 inputs/state. The dense trigger uses
synthetic Q8_0 weights and FP32 input, rows257/K256/N256. No model is loaded.
Libraries and hashes are recorded in [artifact.json](artifact.json).

## Results

Four fresh processes ran serially, with 31 victim samples before and after
one trigger, after three victim warmups. Scratch arms use dense-wide; sham
arms launch GDN again. State reset is outside HIP-event timing.

| Arm | Before median ms | After median ms | After / before |
| --- | ---: | ---: | ---: |
| sham | 0.333824 | 0.332503 | 0.996043 |
| scratch | 0.332741 | 0.331899 | 0.997469 |
| scratch | 0.332779 | 0.332299 | 0.998558 |
| sham | 0.329573 | 0.330899 | 1.004023 |

All GDN outputs and states match byte-for-byte before/after and across the four
processes. Dense-wide passes its CPU F16-operand reference check:
max absolute error 3.7104e-6 against reference scale 9.2502 (bound 1e-3 × scale).
No new numerical implementation is being admitted by this check.
Clock snapshots at the four phase boundaries all report active SCLK 2900 MHz
and MCLK 1000 MHz; these are not continuous clock measurements.

A separate cached-build profiler run confirms queue 1 ordering: six wave-norm
GDN calls with 36-byte scratch, one dense-wide call with 92-byte scratch, then
three GDN calls. The first six include three warmups. Scratch allocation grows
from 1,638,400 bytes before the first GDN to 3,932,160 bytes before dense-wide.
The profiled dense-wide duration is 31.859 µs; GDN durations are 322.682–327.371 µs.
These traced durations are engagement evidence, not the unprofiled timing sample.

## Limits

This tests scratch growth after an already-scratch-enabled baseline, not a
pristine-queue first-scratch transition. It covers only one GDN victim and one
small synthetic dense trigger, not representative decode GEMVs or complete
model geometry. The complete prefill/decode code-object metadata census also
remains open. No claim is made that the external scratch-throttle mechanism is
absent in every configuration.

## Commands

```bash
ENV_PREFIX=/home/lhl/miniforge3/envs/therock10-staging-20260828
SITE=$ENV_PREFIX/lib/python3.12/site-packages
export PATH="$SITE/_rocm_sdk_core/bin:$SITE/_rocm_sdk_devel/bin:$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$SITE/_rocm_sdk_core/lib:$SITE/_rocm_sdk_devel/lib:$SITE/_rocm_sdk_libraries/lib"
export HIPENGINE_HIP_ARCH=gfx1151 HIPENGINE_REQUIRE_CACHED_BUILD=1
export HIPENGINE_COMPILER_VERSION_FILE=/tmp/qwen4exp-ledger-20260928-current/hipcc-version.txt
export OPENBLAS_NUM_THREADS=1
i=0
for arm in sham scratch scratch sham; do
  "$ENV_PREFIX/bin/python" scripts/qwen4exp_scratch_order_probe.py \
    --arm "$arm" --victim wave-norm --trigger dense-wide --tokens 64 --repeats 31 \
    --output "/tmp/qwen4exp-scratch-audit/dense-order-$i-$arm.json" || exit
  i=$((i+1))
done
rocprofv3 --kernel-trace --scratch-memory-trace --output-format csv \
  --output-directory /tmp/qwen4exp-scratch-audit/dense-order-trace -- \
  "$ENV_PREFIX/bin/python" scripts/qwen4exp_scratch_order_probe.py \
  --arm scratch --victim wave-norm --trigger dense-wide --tokens 64 --repeats 3 \
  --output /tmp/qwen4exp-scratch-audit/dense-order-profiled.json
```
