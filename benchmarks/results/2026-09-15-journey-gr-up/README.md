# Isolated GR-Up Numerical Gate

Source `bb42afe94`, Framework machine `55ea6c509d0b49eea8de7094a1023668`,
Radeon 8060S/gfx1151, Flash-Next UD-Q4_K_XL/BF16, chunk1024.
The only restored arithmetic is GR-up IU8; dense Q8, GR-down and MMQ are off.

| Metric | Measured | Required |
| --- | ---: | ---: |
| Mean KL | 0.00140243 | <=0.001 |
| p95 KL | 0.00740151 | <=0.005 |
| p99 KL | 0.01635711 | <=0.02 |
| Maximum KL | 0.04356255 | <=0.05 |
| Top-1 | 772/780 (98.974%) | >=99% |

**Not promoted.** All12 canonical512/1K/4K cases,64 teacher-forced decode
transitions and three repeats completed. Determinism, state ownership/
finiteness and teardown pass, but the numerical envelope fails.

The actual direct wrapper executed6912 times:1152 at `(rows,K,N) =
(512,320,10240)` and5760 at `(1024,320,10240)`. These counts match96 GR reads
per prefill chunk across the full matrix. They are not selector-cache misses
or calls to an unused registry compatibility wrapper.

The existing leaf fixture passes its host-reference tolerance; that does not
override the full-model failure. No task/performance run was spent after the
binding numerical failure. Next correction work should capture actual
normalized inputs and compare the projection and sigmoid/mean boundaries.
This result does not reject every GR fusion or independently implicate GR-down.

```bash
.venv/bin/python benchmarks/results/2026-09-15-journey-gr-up/assemble.py \
  --capture /tmp/hipengine-journey-execute-20260914/resume-gr-up-depth.json
```
