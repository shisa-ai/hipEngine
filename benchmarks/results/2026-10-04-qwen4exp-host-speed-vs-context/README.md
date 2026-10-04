# The host CPU is flat with context, so the decode wall's growth is not host work

`DE-11` prices the decode step's non-kernel remainder at 1.47 ms/step at p512
and 10.56 ms/step at p4096, and two ablations of comparable host work at p4096
measured **+2.53%** (the build-resolution memo) and **-0.45%** (the layer-scope
env memo). Neither matches the 10.04 ms/step the wall grows. This directory
measures the mechanism the profile implied and refutes it.

## What was tested

The canonical bench's `--host-profile` shows every host function's call count
growing 2-4% from p512 to p4096 while its tottime grows ~25%, which reads as
"the same Python calls cost more at depth". On a unified-memory APU the obvious
cause would be the GPU's larger working set starving the CPU's own cache and
bandwidth, which would make host work expensive in proportion to context and
explain why host ablations do not pay.

`scripts/qwen4exp_host_speed_vs_context_probe.py` tests that directly. It runs a
**fixed, context-independent host workload** inside the decode loop - one block
per step, 32 iterations of eight `os.environ.get` calls through the same
four-frame path the profile blames - and reports what that block costs. The
workload cannot depend on sequence length, so if its cost rises with context the
CPU is slower; if it is flat, the CPU is not the explanation.

## Result

| case | host burn us/step | vs idle GPU | wall ms/step |
| --- | ---: | ---: | ---: |
| (GPU idle, no loop) | 67.499 | 1.000 | - |
| code-p512 | 70.816 | 1.049 | 79.829 |
| code-p1024 | 70.912 | 1.051 | 81.156 |
| code-p4096 | **71.478** | 1.059 | **109.763** |

**The same fixed host workload costs 70.816 -> 71.478 us/step, +0.93%, while the
wall grows +37.5%.** Running under GPU load costs ~5% against an idle GPU, and
that penalty is flat with context. **The "the CPU gets slower as context grows"
hypothesis is refuted.**

## What that leaves

With the host CPU's speed flat and its call counts up only 2-4%, host Python work
cannot produce a +10.04 ms/step wall growth. Combined with the device kernel sum
growing only **+2.0%** across the same range, the growth is in the **gaps between
kernels or in device-side work outside kernel durations** - not in the host and
not in kernel bodies. The earlier `X-4` measurement is consistent with this: the
kernel sum is stable to 1.0% while the step span varies 2.8x, and the median
inter-kernel gap swings 23x across repetitions (53.60 / 58.46 / 2.28 us).

**A second conclusion, about instruments.** The `--host-profile` diff that
motivated this test is not usable for it: cProfile's own inflation grew from
5.9% at p512 (50.44 profiled against 47.64 measured) to 8.7% at p4096 (61.33
against 56.40), which inflates every function's tottime roughly in proportion
and produces the "25% more time for 3% more calls" pattern. **cProfile call
counts and caller attribution are exact; cProfile tottime is not a basis for
ranking host work on this path.** Ranking needs ablation, which is what the two
memo A/Bs did.

## Limits

- One host (`gfx1151`, machine-id `55ea6c509d0b49eea8de7094a1023668`), one
  model, one quant, production profile, chunk 1024, 64 transitions, one case per
  generator, one process.
- The probe uses `runner.step`'s defaults, so its `drain_ms_per_step` is 0.0 by
  construction (the step's blocking token readback already synchronizes). The
  burn figure is unaffected: it is timed around the inserted block, not around
  the step.
- Its absolute wall (79.8 / 81.2 / 109.8 ms/step) is well above the canonical
  bench's (47.7 / - / 57.7) because it runs all three cases in one process and
  never warms the PLE page cache. Read the burn ratios, not the walls.
- The conclusion is a negative one: it excludes host Python as the cause. It does
  not yet attribute the growth to a specific gap.
