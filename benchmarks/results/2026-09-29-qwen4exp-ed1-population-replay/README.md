# Q5_1 staging replay on 48 measured router populations

Both raw-weight LDS staging prototypes are slower than the existing j16
kernel on **all 48 populations**. No runtime paths or defaults change; the
replay implementation and test are archived in `candidate.patch`.

## Protocol and result

Qwen3.8-Flash-Next UD-Q4_K_XL, real layer0 Q5_1 weights, synthetic BF16
activations, rows10240/experts512/K640/N2560. The populations were measured
from synthetic token IDs4000..5023, not natural prompts; see the
[fixture provenance](../2026-09-29-qwen4exp-population-recovery/README.md).
Every population uses the same layer0 weights. Capture ordinal is not used
as a layer identity.

Physical host gfx1151, machine ID55ea6c509d0b49eea8de7094a1023668,
Radeon8060S, TheRock10 staging20260828. Five HIP-event samples after two
warmups per arm; fixed order j16/j32/stagedj32/stagedj16/j16. Each population
runs in a fresh process with the benchmark lock held across the full sweep.

| Variant | Mean of 48 case medians (ms) | Change vs j16 | Cases faster than j16 |
| --- | ---: | ---: | ---: |
| Existing j16 | 7.888 | — | — |
| Existing j32 | 11.220 | +42.24% | 0/48 |
| Staged j32 | 14.864 | +88.44% | 0/48 |
| Staged j16 | 22.246 | +182.01% | 0/48 |
| Existing j16 repeated last | 7.887 | -0.016% | 30/48 |

The final j16 repeat differs from its first measurement by -0.55% to +0.84%
per case. All 240 correctness launches agree with their case's j16 output
bits and sorted risk set. Three ragged/tail GPU fixtures also pass.

Timing disables risk-index writes and excludes repair and engine work.
These are diagnostic main-kernel times, not engine speed or complete-owner
qualification. No CPU-reference outer, model numerical/task or mixed-quant
quality gate was run. The unchanged kernel source has cache-only resource and
DRAM-counter evidence in the [initial screen](../2026-09-29-qwen4exp-ed1-lds-screen/README.md);
no new counters were collected on this replay.

## Reproduce

Apply this directory's `candidate.patch` to the source revision recorded in
`artifact.json` in a separate checkout. Use the exact TheRock environment
setup from the initial screen linked above, including the precomputed compiler
version file and `HIPENGINE_REQUIRE_CACHED_BUILD=1` after cache warmup.

```bash
python3 benchmarks/results/2026-09-29-qwen4exp-population-recovery/verify.py
"$E/bin/python" -m pytest -q tests/test_gpu_qwen4exp_q51_iu8_lds.py
exec 9>/tmp/hipengine-gfx1151-benchmark.lock
flock -n 9
for i in $(seq 0 47); do
  "$E/bin/python" scripts/qwen4exp_q51_lds_screen.py \
    --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
    --populations benchmarks/results/2026-09-29-qwen4exp-population-recovery/populations.json \
    --capture-index "$i" --repetitions 5 \
    --output "/tmp/qwen4exp-ed1-population-replay/capture-$i.json"
done
```

Per-case results/commands, population hash, source, weight hash and host
metadata are preserved in `artifact.json`. These two prototypes have no
measured win to promote. ED-1 remains open only for a materially different
staging design or natural-input qualification; do not rerun these unchanged
prototypes as an untested approach.
