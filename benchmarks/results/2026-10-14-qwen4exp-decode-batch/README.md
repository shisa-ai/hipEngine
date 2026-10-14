# Qwen4Exp decode: row-split GDN step, warp-per-expert MoE owners, fused one-row read (2026-10-14)

`qwen4_exp_gguf` / `hip_gfx1151` / `gguf_ud_q4_k_xl`, `production` profile,
Radeon 8060S / gfx1151, host `gfx1151` (machine
`55ea6c509d0b49eea8de7094a1023668`), BF16 KV, chunk 4096, greedy decode with
compact capture, warm PLE cache, tracked-clean `1b7fb672d`, measured
2026-10-14 UTC.

## Rates

Canonical 12-case protocol (one warmup, three repetitions, 128 decode
transitions per case), weighted rate per prompt length, against the
prefill-batch baseline on the same host and protocol
(`../2026-10-14-qwen4exp-prefill-hc-gdn-batch/`):

| Prompt tokens | Decode tok/s before | Decode tok/s now | Change | Prefill tok/s now |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 20.88 | **23.40** | +12.1% | 1029.0 |
| 1024 | 20.12 | **22.47** | +11.7% | 1284.0 |
| 4096 | 20.53 | **22.99** | +12.0% | 1562.9 |
| 8192 | 20.48 | **22.92** | +11.9% | 1528.1 |
| 16384 | 20.37 | **22.81** | +12.0% | 1476.7 |

The 8192/16384 rows are the second fixture
(`benchmarks/fixtures/qwen4exp_canonical_ar_p1024_p8192_p16384.json`); its
1024-token cases measure 22.48. All cases are deterministic;
`scripts/qwen4exp_rate_regression_check.py` passes on both fixtures
(`regression-check-canonical.json`, `regression-check-ladder.json`).
Prefill is unchanged.

External engines on the same GGUF and host, measured 2026-10-12 and not
rerun: gufo `v0.12.0` decodes at 28.1 and halo-box `6aa81b88e` at 25.1 tok/s
at p4096; hipEngine is at 0.82x and 0.92x of them.

## What changed

Decode tok/s on `code-p512` / `code-p4096`, flags set before the generator
is built (the per-layer read and MoE segments are captured as HIP graphs on
first use, so a flag changed later in the same process is not seen):

| Change | Arithmetic | `code-p512` | `code-p4096` |
| --- | --- | ---: | ---: |
| Row-split GDN decode step (512-thread block per head, four lanes per column, state read and written once) | FP32 reduction order | 20.87 -> 21.47 | — |
| Warp-per-expert Q5_1 routed down projection (vectorized half-block loads) | FP32 reduction order | 21.47 -> 22.65 | — |
| Warp-per-expert DP4A routed gate/up | reduction order | 22.65 -> 23.00 | 20.53 -> 22.57 (all three) |
| One-row hyper-connection read in three launches instead of six | bit-identical | 22.99 -> 23.43 | 22.58 -> 23.00 |

Per-kernel decode time on a `rocprofv3` trace (32 steps after a 512-token
prompt): GDN decode 66 -> 27.5 us per layer, routed down 124 -> 73 us,
routed gate/up 128 -> 114 us; a decode step is about 38 ms of kernel time
over about 1,450 launches (42.7 ms over 1,741 before).

Not kept:

- The DP4A gate/up owner at all 48 layers instead of 43: +2.2% decode, but
  the layer2 gate measured top-1 445/450 against the 0.99 floor
  (`gate-b.json`); reverted.
- A two-launch read that rebuilt the norm in every block: bit-identical and
  1.4% slower end to end.

## Correctness

Layer2 production gate on `1b7fb672d` (`gate.json`; 18 prompts, 450
teacher-forced rows, 24 decode steps, three candidate repeats, `strict`
teacher). `gate-a.json` is the same gate on `542787507`, before the
bit-identical read fusion; both record the same candidate logits digest.

| Metric | Limit | Prefill batch (`21177f07f`) | This commit |
| --- | ---: | ---: | ---: |
| mean KL | 1e-3 | 6.295e-4 | 6.364e-4 |
| p95 KL | 5e-3 | 3.630e-3 | 3.637e-3 |
| p99 KL | 2e-2 | 8.495e-3 | 8.069e-3 |
| max KL | 5e-2 | 1.034e-2 | 9.880e-3 |
| top-1 | 0.99 | 448/450 | 446/450 |
| lowest scope top-1 | 0.97 | 0.99 | 0.987 (`code`) |
| strict-exact tasks | review | 13/18 | 13/18 |

Hard gates pass, repeats are bit-identical, state and lifecycle gates pass;
status `requires_task_review` as before. Top-1 has the least margin: 446/450
is two rows above the floor, and the 48-layer DP4A variant that would have
taken one more row was rejected for that reason.

Task review. The five battery prompts whose 32-token greedy continuation
differs from `strict` were read side by side (decoded text in
`task-review-divergences.txt`). The battery feeds raw `[id]` / `role:`
sections without a chat template, so the first tokens decide between closing
the tag, opening a `<think>` block or starting a new turn, and that is where
four of the five split (`general_ja_plan` at token 0, `heldout_general_en_quant`
at 3, `heldout_mixed_review` at 4, `code_markdown_table` at 6); the fifth
(`general_en_plan`, token 6) is a paraphrase of the same sentence. Every
production continuation is coherent and on-task, with no repetition,
truncation or wrong-language output. My reading is that these are near-tied
decisions at the format boundary, not quality regressions; the gate's
`requires_task_review` is answered by this review rather than by a code
change.

18-prompt cross-engine battery (`quality-score.json`): 13/18 strict-exact,
aligned top-1 agreement 1.00, all repeats exact, aligned top-20 mean KL
0.077 (0.076 before).

Long-context gate (`gate-long-context.json`): mean / p95 / p99 / max KL
2.13e-3 / 8.28e-3 / 3.11e-2 / 3.46e-2, top-1 98/100 (2.16e-3 and 97/100
before); diagnostic, as recorded for the prefill batch.

## MTP verification cost

`mtp-batched-verify-cost.json`: after a 512-token prompt and 32 greedy steps,
the last eight target tokens re-verified from a device snapshot, k rows in
one pass through the prefill chunk owners plus one batched head, against a
serial decode step of 45.97 ms (Python step loop included):

| Rows | Pass ms | Pass / step | Per token ms | Argmax equals serial |
| ---: | ---: | ---: | ---: | --- |
| 1 | 42.5 | 0.92 | 42.5 | 1/1 |
| 2 | 81.3 | 1.77 | 40.6 | 2/2 |
| 4 | 90.6 | 1.97 | 22.7 | 4/4 |
| 6 | 99.6 | 2.17 | 16.6 | 6/6 |
| 8 | 108.7 | 2.36 | 13.6 | 8/8 |

A device snapshot plus rollback costs 2.5 ms. At four rows a pass costs two
decode steps, and rejecting part of a block means rolling back and replaying
the accepted prefix, so batched verification only beats autoregressive
decode at high draft acceptance. The two-row pass costing almost two steps
points at small-row kernel selection rather than at the weight traffic.
