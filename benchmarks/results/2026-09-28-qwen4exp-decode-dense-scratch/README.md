# Q5_1 decode after dense-wide first scratch

No large persistent slowdown appears when a 92-byte private-segment dense-wide
kernel first allocates scratch before subsequent Q5_1 down/combine decode.
This is a diagnostic screen, not a performance improvement or qualification.

The [artifact](artifact.json) records four serial fresh-process runs on
Framework machine `55ea6c509d0b49eea8de7094a1023668`, Radeon 8060S/gfx1151,
TheRock 10 / HIP 7.15.26333. The victim uses real layer0 Q5_1 down weights from
unsloth Qwen3.8-Flash-Next `UD-Q4_K_XL`, 512 experts, top-k10, in640/out2560,
with synthetic deterministic BF16 activations/routes. The trigger uses synthetic
Q8_0 dense-wide inputs, rows257/K256/N256. Loaded library and weight hashes,
event samples and clock snapshots are in the artifact.

| Arm | Before median ms | After median ms | After / before |
| --- | ---: | ---: | ---: |
| sham | 0.098305 | 0.097142 | 0.988169 |
| scratch | 0.096421 | 0.095940 | 0.995011 |
| scratch | 0.095339 | 0.095619 | 1.002937 |
| sham | 0.097744 | 0.096661 | 0.988920 |

Each process performs three victim warmups, 31 before samples, one trigger
(or victim sham), and 31 after samples. Trigger buffers are prepared without
executing the trigger before the baseline. Victim outputs are byte-identical
before/after and across the four processes. Dense output matches the FP16
CPU reference with maximum absolute error 4.489e-6 (reference scale 8.7387).
The check occurs after both timing windows.

A separate cached profiler capture verifies six zero-scratch Q5_1 calls,
the first scratch allocation of 3,932,160 bytes, one 92-byte dense-wide call,
then three zero-scratch Q5_1 calls, all on queue1. It executes no GDN kernel.
This excludes an earlier GDN allocation as a confounder for this capture.

Limits: one Q5_1 layer and synthetic routes, boundary-only clock telemetry,
no gate/up or Q8 decode victims, and no serving-wall measurement. X-5 is not
universally cleared. Runtime arithmetic and defaults are unchanged.

## Reproduction

Use the environment exports from the
[scratch-growth packet](../2026-09-28-qwen4exp-scratch-growth/README.md).
Run the commands in the [GDN first-trigger packet](../2026-09-28-qwen4exp-decode-scratch/README.md)
with `--trigger dense-wide`, replacing output prefixes `decode-order` with
`decode-dense` (including the profiler directory). Keep process order
`sham scratch scratch sham`, 31 repeats unprofiled and 3 profiled. The artifact
preserves every child's argument vector. Validation also includes:

```bash
python3 -m py_compile scripts/qwen4exp_decode_scratch_probe.py
```
