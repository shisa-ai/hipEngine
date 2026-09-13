# Journey Production Baseline: Numerical Gate Failed

September 14, 2026 JST, Framework gfx1151, Qwen3.8-Flash-Next UD-Q4_K_XL,
BF16 KV, chunk1024, context2051. No runtime overrides or new optimization.

Full18 prompt/heldout suite, 32 teacher-forced decode steps plus prefill last
per prompt: 594 rows, three production repeats, two free32 trajectories.

| Gate | Result |
| --- | --- |
| Overall mean / p95 / p99 KL | 0.000625 / 0.002651 / 0.008139 |
| Maximum KL | **0.054642 > 0.05: failed** |
| Top1 | 590/594 (99.327%) |
| Prefill-last mean / p95 KL | **0.001475 / 0.007070: failed** |
| Japanese category mean KL | **0.001149: failed** |
| Determinism / state-repeat / teardown | Passed |
| Free32 trajectories | 14/18 strict-identical; four deterministic differences need task review |

The maximum occurs on `heldout_general_ja_speculative`, teacher step12.
This is an incumbent profile failure, not a candidate regression. No threshold
is widened and no new arithmetic candidate is promotable from this packet.
Exact-preserving owner experiments may continue; they cannot certify the
incumbent's numerical envelope.

`artifact.json` includes command, source, profile manifests, host, recomputed
four-shard sampled fingerprint, category/shape/transition metrics and outliers.
The full raw JSON remains at the hashed path in that artifact. Regenerate:

```bash
python3 benchmarks/results/2026-09-14-journey-production-baseline/assemble.py \
  --raw /tmp/hipengine-journey-execute-20260914/production-baseline.json
```
