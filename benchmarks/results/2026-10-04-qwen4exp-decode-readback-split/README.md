# The per-step token readback costs nothing, and `device_drain_ms = 0.0` does not mean host-bound

This directory corrects the host/device attribution published for
`2026-10-04-qwen4exp-thermal-protocol-and-llamacpp-basis`. **The measurement in
that directory stands; its interpretation of the host/device split does not.**

## What was claimed

`scripts/qwen4exp_decode_host_bound_probe.py` split the decode loop into a host
window and a trailing device drain and reported `device_drain_ms = 0.0` at
p512, p1024 and p4096, concluding `host_bound: true`: *"the GPU completed all
128 queued steps before the host finished enqueuing them, so it is idle for the
entire decode loop."*

## Why that inference is invalid

`runner.step()` ends with a **blocking** device-to-host copy of the argmax
result (`_finish_lm_head` -> `copy_device_to_host`). The host window therefore
already contains the device wait, and a trailing drain of zero is a tautology,
not a measurement. Two independent checks:

- The canonical bench's `--capture-mode compact` removes the explicit per-step
  `device_synchronize()` and still reports `decode_device_drain_ms_per_step =
  0.000`, because the blocking readback supplies the synchronization instead.
  See `../2026-10-04-qwen4exp-decode-capture-mode/`.
- Removing the readback is what makes the drain meaningful, and then it is not
  zero: **0.343 ms/step (p512) and 0.575 ms/step (p4096)**.

## The counter-rotated A/B

The first readback measurement was confounded: the two arms ran at different
sequence positions on one growing sequence, so they differed in KV state and in
which PLE rows had to be faulted in. This probe alternates arms on **one
generator with a `reset()` + `prefill()` before every pass**, so both arms start
from the identical position, KV state and PLE row set, and it counter-rotates
(readback / no_readback / readback / no_readback):

| case | arm | wall ms/step | tok/s | trailing drain over 128 steps |
| --- | --- | ---: | ---: | ---: |
| code-p512 | readback | 54.687 / 54.670 | 18.286 / 18.292 | 0.000 ms/step |
| code-p512 | no_readback | 55.230 / 55.091 | 18.106 / 18.152 | **0.343 ms/step** |
| code-p4096 | readback | 84.618 / 84.588 | 11.818 / 11.822 | 0.000 ms/step |
| code-p4096 | no_readback | 85.761 / 85.789 | 11.660 / 11.657 | **0.575 ms/step** |

**Delta (readback - no_readback): -0.482 ms/step = -0.87% at p512 and -1.172
ms/step = -1.37% at p4096.** Dropping the blocking readback makes the loop
*slightly slower*, by exactly the extra trailing drain the readback was hiding.
The readback is inside `max(host enqueue, device work)`, not on top of it.

The two arms also produce the same result twice each (0.03% and 0.04% within
arm), so this instrument resolves well below the effects it was built to test.

## What replaces the claim

At both depths the host enqueue and the device execution are within ~1% of each
other, and the loop wall is their maximum. **A host-only or device-only change
is therefore capped at roughly half its nominal saving**, which is what the
canonical bench's non-kernel remainder shows directly:

| case | decode wall ms/step | profiled kernel sum ms/step | kernel sum growth | wall growth |
| --- | ---: | ---: | ---: | ---: |
| code-p512 | 47.66 | 46.19 | - | - |
| code-p4096 | 57.69 | 47.13 | **+2.0%** | **+21%** |

The wall grows 10.04 ms/step where the device kernel sum grows 0.94 ms/step.
Independently, removing 2.3 ms/step of host work (the build-resolution memo)
moves p512 by +0.02% and p4096 by +2.53% against a 4.0% prediction - nothing to
gain where the step is device-limited, most of the prediction where it is not.
**The decode rate at depth is set by the growing non-kernel remainder, and that
remainder is host-side.**

## Limits

- The probe's absolute rates (18.29 tok/s at p512, 11.82 at p4096) exceed the
  canonical bench's (21.0 and 17.4) in *step time*: 54.7 vs 47.7 ms/step and
  84.6 vs 57.7 ms/step. The probe never warms the PLE page cache and it runs
  both cases in one process, so its host cost is inflated. **The probe's
  within-arm deltas are sound; its absolute levels are not bench levels.**
- Because the probe's host cost is inflated, its "the device keeps up" result
  describes the probe, not the bench. The bench-side host/device question needs
  the readback removed in the bench itself.
- `wall - kernel_sum` is a difference of two differently-measured quantities:
  the kernel sum comes from a profiled capture whose own note records that the
  profiler serializes launches and inflates absolute times. Read the 46.19 /
  47.13 figures as the device floor and the wall growth as the finding.
- One model, one quant, one host (`gfx1151`, machine-id
  `55ea6c509d0b49eea8de7094a1023668`), one execution profile.

## Files

- `counter-rotated.json` - the four-pass A/B, both cases.
- `probe.json` - the confounded first attempt, kept because its absolute levels
  are what the correction is measured against.
