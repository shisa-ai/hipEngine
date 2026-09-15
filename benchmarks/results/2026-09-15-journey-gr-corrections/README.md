# GR Correction Operand Screen

Two in-tree diagnostic variants at `a0d301f9d`, Framework machine
`55ea6c509d0b49eea8de7094a1023668`, Radeon8060S/gfx1151,
Flash-Next UD-Q4_K_XL/BF16. Neither changes a production default.

- **Compensated block accumulation:** GR-down sampled FP64-relative MSE
  improves16.9-67.4x over the original IU8 projection, and is0.18-0.48x
  the strict projection MSE in all six captured roles.
- **Fourth activation-residual plane:** GR-up sampled MSE improves1.76-4.60x.
  It does not materially help GR-down; compensation has little effect on GR-up.

Both use the same model-produced inputs and raw Q8 weights as the original
replay, verified by hashes. Layers0/23/47, both GR roles, three geometry-
selected activation rows and64 columns are sampled. Full model control
logits and sampled recurrent state match, with zero tracked teardown.
These MSE ratios are not a model-quality or throughput certificate.

The compensated analytic cancellation fixture reduces absolute error from
0.155273 to below0.005. Ten GPU tests pass, including FP64-reference tails
and deterministic repeats. A cached-only trace observes both specializations:
P4 uses224VGPR/19456LDS, compensated176VGPR/14848LDS, both with0scratch.
Trace durations are smoke evidence, not performance comparisons.

The variants have separate gfx1151 registry keys and registered strict
coltile/fused-GR fallbacks. Profile-owned diagnostic selectors are cleared
by normal binding. Full model quality and operation-complete cost are next;
the original independently failing GR paths are not re-enabled.

```bash
.venv/bin/python benchmarks/results/2026-09-15-journey-gr-corrections/assemble.py \
  --raw-root /tmp/hipengine-journey-execute-20260914 \
  --trace /tmp/hipengine-gr-corrections-trace/gfx1151/3188036_kernel_trace.csv
```
