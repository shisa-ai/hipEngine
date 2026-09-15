# Identical-Input GR Projection Replay

Source `72cbc2f56`, Framework machine `55ea6c509d0b49eea8de7094a1023668`,
Radeon 8060S/gfx1151, Flash-Next UD-Q4_K_XL/BF16. One canonical Japanese
512-token prompt, current production arithmetic, chunk1024.

The model keeps its original outputs. Replays use separate buffers for
strict-order and IU8 projections on the same inputs. Both GR roles at
layers0/23/47 are captured. FP64 checks sample three geometry-selected rows
and64 evenly spaced output columns; full GPU output arrays supply direct
parent/candidate error metrics.

## Findings

- In all six GR-up captures, strict-order projection followed by standalone
  sigmoid/mean is exactly equal to the existing fused output. The split
  epilogue is not an error source in these samples.
- GR-down IU8 sampled MSE versus raw-weight FP64 is **7.84-32.07x** the
  strict projection's MSE, and **24.66-198.17x** the CPU-estimated
  activation-reconstruction-only MSE.
- GR-up IU8 sampled MSE is **1.15-1.75x** the CPU-estimated
  reconstruction-only MSE. That makes activation precision a stronger
  candidate here than for GR-down.
- Full model final logits and sampled recurrent state match the
  uninstrumented control. Tracked allocations close to zero.

The reconstruction is a CPU simulation of the three-plane residual process,
not captured GPU quantizer output. Ratios of MSE are not an additive error
decomposition. Sampling does not establish a universal error bound or a
full-model precision guarantee.

## Next Corrections

Test compensated block accumulation for GR-down and a fourth residual
activation plane for GR-up, first on these same operands and then under the
complete production numerical/task and performance gates. These are
hypotheses, not measured improvements. The original independent GR failures
remain binding; no default changes or throughput claims result from this replay.

```bash
.venv/bin/python benchmarks/results/2026-09-15-journey-gr-operands/assemble.py \
  --capture /tmp/hipengine-journey-execute-20260914/resume-gr-operands.json
```
