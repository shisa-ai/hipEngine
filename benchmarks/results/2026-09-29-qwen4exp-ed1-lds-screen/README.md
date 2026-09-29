# Q5_1 cooperative raw-weight LDS staging: negative initial screen

Neither prototype improves expert-down main-kernel time on this workload.
The candidate code, test and screen are archived in `candidate.patch`; no
runtime path or default changed. **ED-1 remains open for real-population
qualification or a materially different staging schedule.** This screen does
not establish that all LDS staging designs lose.

## Measurement

Physical host `gfx1151`, machine ID `55ea6c509d0b49eea8de7094a1023668`,
AMD Radeon 8060S, TheRock `therock10-staging-20260828`. Qwen3.8-Flash-Next
UD-Q4_K_XL, real `blk.0.ffn_down_exps.weight` Q5_1 weights; synthetic BF16
activations and balanced synthetic expert populations. Shape: 10,240 compact
rows, 512 experts, K=640, N=2560. Tensor hash and exact populations are in
`artifact.json`. Fifteen HIP-event samples follow two warmups per arm.

| Main kernel | Median ms | Change vs j16 | LDS bytes | VGPRs | Scratch bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing j16 | 9.147 | — | 16,384 | 192 | 0 |
| Existing j32 | 12.646 | +38.26% | 32,768 | 208 | 0 |
| Raw-weight staged j32 | 18.312 | +100.20% | 57,344 | 248 | 0 |
| Raw-weight staged j16 | 28.919 | +216.17% | 40,960 | 248 | 0 |
| Existing j16, repeated last | 9.132 | -0.16% | 16,384 | 192 | 0 |

Timing suppresses risk-index writes (`max_risks=0`) and excludes sparse repair,
packing and engine work. Correctness launches use full risk capacity. These
are main-kernel measurements, not complete-owner or engine rates.

Separate cache-only `rocprofv3` tracing confirms both staged symbols execute
without scratch spills. Timed-dispatch median `GCEA_RDRAM_SIZE_REQ_sum`:
j16 37,504,723; j32 23,153,164; staged j32 22,477,041; staged j16 23,058,797.
Staging reduces this counter only 2.92% relative to unstaged j32 while raising
resource use and time. Counter units are uncalibrated and include all reads;
these are **not isolated weight-byte measurements**. Lower occupancy or LDS
access cost is a plausible explanation, not a measured causal attribution.

## Correctness and limits

- RED before implementation: missing staged symbol.
- Three GPU fixtures pass exact raw output and sorted risk-set comparison to
  j16 and j32; staged j32 also repeats. K/N cases are 32/17, 288/129, 640/256;
  expert populations include empty experts and 16/32-row boundaries.
- The real-weight screen matches every output bit and all 390,121 risk indices
  across five arms, including the final j16 repeat.
- CPU-reference outer, repair-chain, model numerical/task and real-routing
  gates were not run. This is not production qualification.
- No gain was found, so these prototype runtime paths were removed. Reproduce
  in a separate checkout if needed; the patch includes the test and harness.

## Commands

From the candidate base recorded in `artifact.json`, apply `candidate.patch`
in a separate checkout. The environment was:

```bash
E=/home/lhl/miniforge3/envs/therock10-staging-20260828
S=$E/lib/python3.12/site-packages
export PATH="$S/_rocm_sdk_devel/bin:$S/_rocm_sdk_core/bin:$E/bin:$PATH"
export HIP_PATH="$S/_rocm_sdk_devel"
export LD_LIBRARY_PATH="$S/_rocm_sdk_core/lib:$S/_rocm_sdk_devel/lib:$S/_rocm_sdk_libraries/lib"
export HIPENGINE_HIP_ARCH=gfx1151
export HIPENGINE_COMPILER_VERSION_FILE=/tmp/qwen4exp-ledger-20260928-current/hipcc-version.txt
# The version file was prepared outside profiling; do the same when reproducing.
flock -n /tmp/hipengine-gfx1151-benchmark.lock "$E/bin/python" -m pytest -q tests/test_gpu_qwen4exp_q51_iu8_lds.py
flock -n /tmp/hipengine-gfx1151-benchmark.lock "$E/bin/python" scripts/qwen4exp_q51_lds_screen.py --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL --output /tmp/qwen4exp-ed1-screen-j16.json
export HIPENGINE_REQUIRE_CACHED_BUILD=1
flock -n /tmp/hipengine-gfx1151-benchmark.lock "$E/bin/rocprofv3" --kernel-trace --pmc GCEA_RDRAM_SIZE_REQ_sum -f csv -o /tmp/qwen4exp-ed1-final-trace -- "$E/bin/python" scripts/qwen4exp_q51_lds_screen.py --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL --repetitions 3 --output /tmp/qwen4exp-ed1-final-profiled.json
```
