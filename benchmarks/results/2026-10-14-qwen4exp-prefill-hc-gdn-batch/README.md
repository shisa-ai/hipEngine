# Qwen4Exp prefill: hyper-connection, GDN and expert-glue kernels (2026-10-14)

`qwen4_exp_gguf` / `hip_gfx1151` / `gguf_ud_q4_k_xl`, `production` profile,
Radeon 8060S / gfx1151, host `gfx1151` (machine
`55ea6c509d0b49eea8de7094a1023668`), BF16 KV, chunk 4096, greedy decode with
compact capture, warm PLE cache, tracked-clean `21177f07f`, measured
2026-10-14 UTC.

## Rates

Canonical 12-case protocol (one warmup, three repetitions), weighted rate
per prompt length, against the previous baseline on the same host and
protocol
(`../2026-10-14-qwen4exp-prefill-exact-kernels-and-ladder/`, 2026-10-13):

| Prompt tokens | Prefill tok/s before | Prefill tok/s now | Change | Decode tok/s now |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 779.4 | **1028.5** | +32.0% | 20.88 |
| 1024 | 977.9 | **1276.5** | +30.5% | 20.12 |
| 4096 | 1190.3 | **1562.2** | +31.2% | 20.53 |

Second fixture (`benchmarks/fixtures/qwen4exp_canonical_ar_p1024_p8192_p16384.json`):

| Prompt tokens | Prefill tok/s before | Prefill tok/s now | Change | Decode tok/s now |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 978.7 | **1280.4** | +30.8% | 20.13 |
| 8192 | 1158.8 | **1526.6** | +31.7% | 20.48 |
| 16384 | 1133.1 | **1475.5** | +30.2% | 20.37 |

All 24 cases are deterministic across repetitions. Output token ids changed
in 9 of 12 cases of each fixture, which is expected: two of the changes
below change arithmetic. `scripts/qwen4exp_rate_regression_check.py` passes
on both fixtures (`regression-check-canonical.json`,
`regression-check-ladder.json`).

External engines on the same GGUF and host, measured 2026-10-12 and not
rerun: gufo `v0.12.0` 1095.0 / 1482.2 / 1736.4 and halo-box `6aa81b88e`
759.3 / 1109.1 / 1446.4 prefill tok/s at 512 / 1024 / 4096 tokens. hipEngine
is now at 0.94x / 0.86x / 0.90x of gufo and ahead of halo-box at every
length. Neither external engine has been measured at 8192 or 16384 tokens.

Commands:

```bash
python scripts/qwen4exp_canonical_ar_bench.py hipengine --model-root <UD-Q4_K_XL> \
  --prefill-chunk-size 4096 --capture-mode compact --ple-cache-mode warm \
  --execution-profile production --output hipengine-production.json
# and the same with
#   --fixture benchmarks/fixtures/qwen4exp_canonical_ar_p1024_p8192_p16384.json
```

## What changed

Each row is one commit. "Same-process" numbers are prefill tok/s on
`code-p4096` with the change off and on inside one process (identical
residency and thermal state), not the canonical protocol.

| Change | Arithmetic | Same-process `code-p4096` |
| --- | --- | ---: |
| Hyper-connection read projections launched directly on F16 (norm publishes F16 at every layer and row count; K10240/N320 down on the 128x128 tile; up reads an activation converted once) | bit-identical | 1176.4 -> 1224.4 |
| Pipelined vector staging for the small-matrix F16 GEMM (router logits, indexer) | bit-identical | 1224.4 -> 1235.8 |
| 48-column GDN gate projections on that GEMM instead of the scalar FP32 GEMV | F16 operands | 1235.8 -> 1283.9 |
| Sigmoid gated mean folded into the hyper-connection up GEMM (gate logits never written) | bit-identical | 1283.9 -> 1309.4 |
| Routed gate/up reads token rows through the lane map (no grouped activation copy) | bit-identical | 1309.7 -> 1338.1 |
| Hyper-connection read without the FP32 normalized plane (norm stores F16 and `inv_rms`; consumers recompute) | bit-identical | 1320.5 -> 1382.1 |
| One shared F16 image of the hyper-connection output for every F16 projection that reads it | bit-identical | 1390.8 -> 1440.1 |
| 128-row blocks and one-branch staging for the fused up kernel | bit-identical | 1440.1 -> 1470.3 |
| Eight-column BF16 weighted lane sum | bit-identical | 1470.3 -> 1473.4 |
| Row-split GDN prefill recurrence in 128-thread blocks (ported from gufo, MIT) | FP32 summation order | 1478.5 -> 1529.8 |
| Long-K F16 wide launches on the 128x128 tile built without CU mode (GDN output projection) | bit-identical | 1520.7 -> 1535.1 |
| GDN norm/gate epilogue publishes the F16 image of its output | bit-identical | 1535.1 -> 1554.6 |
| Dense wide launches of 257-1024 rows use a build without CU mode | bit-identical | `code-p512` 981.2 -> 1020.5, `code-p1024` 1256.0 -> 1270.1 |

Rollback switches and removal conditions for every flag are in
`docs/REFACTOR.md`.

Negative results, not kept: two bit-exact relayouts of the tile-16 GDN
recurrence (four lanes per column, eight lanes per two columns) were slower
than the body they would replace (8.5 and 9.6 ms against 8.0 ms, recurrence
plus epilogue, 4096 tokens, 48 heads); wider K10240/N320 tiles (320x64,
160x128) were slower than 128x128 (1.93 and 1.57 against 1.45 ms); builds
without CU mode did nothing for the hyper-connection elementwise and GDN
conv kernels in a same-process A/B; adding CU mode to the routed-expert
kernels cost 1.7% at 4096 tokens.

## Kernel time per `code-p4096` prefill

`rocprofv3 --kernel-trace`, one measured prefill, milliseconds, kernels
grouped by role with one rule set for both hipEngine columns:

| Family | 2026-10-13 | Now | gufo `v0.12.0` |
| --- | ---: | ---: | ---: |
| Routed experts | 696 | 699 | 812 |
| Dense projections (GDN in/out and gates, attention q/k/v/o, shared expert) | 874 | 659 | — |
| Hyper-connection read and write (norm, down, up, gated mean, inject, write) | 934 | 544 | — |
| GDN recurrence, conv, norm/gate | 349 | 271 | 208 |
| Attention, block scoring, indexer | 233 | 211 | 113 |
| Router, expert reduce, lane sum, BF16 conversions | 309 | 167 | — |
| Per-layer embedding and other | 14 | 14 | — |
| **Total** | **3409** | **2564** | **2373** |

The gufo column lists only the families whose boundaries match; its total is
from `../2026-10-12-qwen4exp-cross-engine-pass1/`. `attribution-code-p4096.json`
is the per-kernel table for the "now" column.

## Correctness

Layer2 production gate on this commit (`gate.json`; 18 prompts, 450
teacher-forced rows, 24 decode steps, three candidate repeats, `strict`
teacher):

| Metric | Limit | 2026-10-13 plan | This commit |
| --- | ---: | ---: | ---: |
| mean KL | 1e-3 | 6.493e-4 | 6.295e-4 |
| p95 KL | 5e-3 | 3.452e-3 | 3.630e-3 |
| p99 KL | 2e-2 | 7.889e-3 | 8.495e-3 |
| max KL | 5e-2 | 1.186e-2 | 1.034e-2 |
| top-1 | 0.99 | 448/450 | 448/450 |
| lowest scope top-1 | 0.97 | 0.99 | 0.99 |
| strict-exact tasks | review | 15/18 | 13/18 |

Hard gates pass, repeats are bit-identical, state and lifecycle gates pass.
The gate status is `requires_task_review`, as it was for the plan it
replaces: free-running greedy continuations diverge from `strict` on five
prompts instead of three. The same three scopes are listed under
`scope_kl_advisories` (`mixed_ja_en` mean 1.108e-3 / p95 5.636e-3,
`prefill_last` and `prefill_to_c1`), which the profile contract reports
without binding.

18-prompt cross-engine battery (`quality-score.json`): 13/18 strict-exact
(14/18 before), aligned top-1 agreement 1.00, all repeats exact, aligned
top-20 mean KL 0.076 (0.049 before).

Long-context gate (`gate-long-context.json`; four prompts of 2,000 to 2,600
tokens, 100 rows): mean / p95 / p99 / max KL 2.16e-3 / 9.59e-3 / 3.02e-2 /
3.23e-2, top-1 97/100 (2.10e-3 / 1.28e-2 / 1.61e-2 / 2.06e-2 and 99/100 for
the 2026-10-13 plan). It is outside the envelope, as every plan that is not
bit-identical to `strict` has been at this length, including a control that
only reorders one FP32 summation; how long prompts are gated is still an
open owner decision (`docs/EXECUTION-PROFILES.md` section 6.1).

## Files

- `hipengine-production.json`, `hipengine-production-p1024-p8192-p16384.json`:
  canonical and ladder runs; the regression baselines named in
  `benchmarks/HARNESSES.md`.
- `regression-check-canonical.json`, `regression-check-ladder.json`.
- `gate.json`, `gate-long-context.json`, `quality-hipengine-production.json`,
  `quality-score.json`.
- `attribution-code-p4096.json`.
