# Isolated GR-Down Numerical Gate

Source `b20439483`, Framework machine `55ea6c509d0b49eea8de7094a1023668`,
Radeon 8060S/gfx1151, Flash-Next UD-Q4_K_XL/BF16, chunk1024.
Only GR-down IU8 is restored; GR-up, generic dense Q8 and MMQ are off.

| Metric | Measured | Required |
| --- | ---: | ---: |
| Mean KL | 0.00131282 | <=0.001 |
| p95 KL | 0.00683522 | <=0.005 |
| p99 KL | 0.01772116 | <=0.02 |
| Maximum KL | 0.03944422 | <=0.05 |
| Top-1 | 769/780 (98.590%) | >=99% |

**Not promoted.** All12 canonical512/1K/4K cases and64 teacher-forced decode
transitions completed, with three deterministic repeats, finite state,
matching ownership metadata and zero tracked allocations after teardown.

Actual direct calls total6912:1152 at `(rows,K,N)=(512,10240,320)` and5760
at `(1024,10240,320)`. These match96 GR reads per chunk over the matrix.
The independent failure complements the GR-up result; neither relies on
attributing the earlier combined-arm failure by flag name.

The next correction needs identical-input comparison of raw Q8 weights,
activation reconstruction and accumulation at the projection boundary.
No task/performance run was spent after this binding numerical failure.
Maximum-KL and leaf-tolerance passes do not override failed mean/p95/top1.

```bash
.venv/bin/python benchmarks/results/2026-09-15-journey-gr-down/assemble.py \
  --capture /tmp/hipengine-journey-execute-20260914/resume-gr-down-depth.json
```
