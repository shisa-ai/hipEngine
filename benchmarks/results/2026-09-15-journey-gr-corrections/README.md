# GR Correction Operand Screen

Two in-tree diagnostic variants developed at `a0d301f9d`, Framework machine
`55ea6c509d0b49eea8de7094a1023668`, Radeon8060S/gfx1151,
Flash-Next UD-Q4_K_XL/BF16. Both are now removed after failed model gates;
neither changed a production default.

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

The variants had separate gfx1151 registry keys and registered strict
coltile/fused-GR fallbacks. The rejected specializations, exports, registry
entries, selectors and candidate-only tests are removed. Historical commands
require their pinned revisions. The original failing GR paths remain off.

## Compensated GR-Down Model Gate

Clean `99e21a0ac`, all12 canonical cases,64 teacher-forced transitions,
three repeats/780 rows, with6912 actual corrected registry calls:

| Metric | Compensated | Limit |
| --- | ---: | ---: |
| Mean KL | 0.00131602 | <=0.001 |
| p95 KL | 0.00666774 | <=0.005 |
| Maximum KL | 0.05029259 | <=0.05 |
| Top-1 | 769/780 (98.590%) | >=99% |

**Not promoted.** Determinism, state/metadata/finiteness and teardown pass,
but mean/p95/max/top1 fail. Better agreement with sampled FP64 projections
does not imply acceptable drift against the model's strict execution.
No task or performance run was spent on this failed correction.

## P4 GR-Up Model Gate

Clean `b1302fc31`, the same full protocol and6912 corrected registry calls:

| Metric | P4 | Limit |
| --- | ---: | ---: |
| Mean KL | 0.00161375 | <=0.001 |
| p95 KL | 0.00759287 | <=0.005 |
| Maximum KL | 0.06809635 | <=0.05 |
| Top-1 | 770/780 (98.718%) | >=99% |

**Not promoted.** Determinism, state/metadata/finiteness and teardown pass;
mean/p95/max/top1 fail. No task or performance run follows this failure.
Both corrections improved sampled FP64 MSE but failed the actual production
envelope. This closes these two variants, not all GR optimization.

The affected kernel/runtime/registry/gate files are restored byte-for-byte
to pre-experiment `4e6de95b2`. Independent test cleanup and registration
repairs are preserved. Parent-order work must avoid repeating the unchanged
September6 `(row_batch,hidden_tile)=(8,1)` loss and `(2,4)` inconclusive/loss;
larger accumulator tiles would be a distinct, untested hypothesis.

```bash
.venv/bin/python benchmarks/results/2026-09-15-journey-gr-corrections/assemble.py \
  --raw-root /tmp/hipengine-journey-execute-20260914 \
  --trace /tmp/hipengine-gr-corrections-trace/gfx1151/3188036_kernel_trace.csv
```
