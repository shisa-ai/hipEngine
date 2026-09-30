# What the gate's `prefill_last` row measures

`shape=prefill_last` / `transition=prefill_to_c1` is `trajectory[0]`:
`runner.prefill(prompt_ids, capture_logits=True)`'s own last-token logits
(`scripts/execution_profile_q8_wmma_prefill_layers_gate.py`, `_strict_trajectory`).
It is the only row in the protocol that reports the prefill's own output.

Two artifacts disagreed about it by five orders of magnitude:

| run | route | prompt tokens | `prefill_last` mean | `c1`/`steady` mean |
|---|---|---|---|---|
| canonical 12-case, at HEAD | `dense_wide` | 512 / 1024 / 4096 | **4.4145e-09** | 4.4225e-04 |
| canonical 12-case, at HEAD | `wmma` | 512 / 1024 / 4096 | **4.4145e-09** | 4.4225e-04 |
| long-form 4-case heldout | `dense_wide` | 424 / 438 / 508 / 512 | **1.4390e-03** | 3.9168e-04 |

Both at 16-47, both with the route engaged, both deterministic. `benchmarks/
results/2026-09-29-qwen4exp-prefill-scope-discrepancy/README.md` recorded the
same inversion for `wmma` on the admission suite and left it open: *"Why the
heldout prefill row moves. Recorded as an open question with a named candidate
explanation and a test that would settle it."* The candidate explanation there
(dispatch order, the exact leaf failing the `prefill_` prefix predicate) was
falsified the same day by the engagement recorder, which showed the route
engaging in **both** protocols.

## The measurement

`scripts/qwen4exp_prefill_last_row_probe.py`, one process, production profile,
`prefill_chunk_size=1024`, three repeats per arm, both arms built from the same
generator and differing only in `HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE` +
`..._DENSE_WIDE_LAYERS=16-47`:

| subject | tokens | same-arm repeats | route effect on `prefill_last` | route effect on the first decode step |
|---|---|---|---|---|
| `canonical` `code-p512` | 512 | bit-identical | KL **1.3277e-12** | KL 2.1902e-05 |
| `admission` `code_merge_intervals` | 40 | bit-identical | **0** (bit-identical) | **0** (bit-identical) |
| `admission` `heldout_code_rate_limiter` | 55 | bit-identical | **0** (bit-identical) | **0** (bit-identical) |
| `longform` `long_mixed_ja_en_acceptance` | 512 | bit-identical | KL **1.0885e-03** | KL 9.6647e-06 |
| `longform` `long_general_ja_retention_review` | 424 | bit-identical | KL **1.5955e-04** | KL 1.7305e-06 |

The probe reproduces both recorded artifacts from one process: `code-p512` at
1.3e-12 against the canonical artifact's 4.4e-09, and `long_mixed_ja_en_
acceptance` at 1.1e-03 against the long-form artifact's 1.439e-03 four-case mean.

The two admission cases are a negative control. `dense_wide` declines at
`rows <= 256`, so the route cannot run there and the probe correctly measures
nothing. The 40/55-token zeros are the route's row floor, not evidence about
arithmetic.

## The answer

**The route changes the prefill's own logits by the same amount in every
protocol. What differs is where that change lands.**

| subject | `prefill_last` KL | `prefill_last` max abs logit delta |
|---|---|---|
| `canonical` `code-p512` | 1.3277e-12 | **4.0587e-01** |
| `longform` `long_mixed_ja_en_acceptance` | 1.0885e-03 | **4.4500e-01** |
| `longform` `long_general_ja_retention_review` | 1.5955e-04 | **6.2800e-01** |

The max absolute logit delta is the same order in all three - the route replaces
the exact chain's f16 operands at 16-47 and moves individual logits by ~0.4-0.6
regardless of prompt. KL is a distributional measure: on the canonical prompts
those deltas fall where the softmax is flat, and on the long-form prompts they
fall near the argmax.

So neither of the two readings in the record survives:

- The canonical `prefill_last` row being flat is **not** evidence that the route
  contributes nothing measurable to the prefill. Its logits move by 0.4. The row
  is a distributional measure of a per-logit change, and the canonical fixture's
  twelve prompts happen to hide it.
- The long-form `prefill_last` failure is **not** a defect in the prefill-to-c1
  transition. There is no transition mechanism here: the row is the prefill's own
  output, it is deterministic, and the same route moves it in every protocol.

`prefill_to_c1` and `prefill_last` are two labels on the same four rows - the
identical mean, p95 and max in the long-form artifact's `by_scope` show that -
so the long-form run's `scope_failures` list counts one scope twice. It is not
two independent failures.

## Why the long-form scope exceeds its limit, and why that is the gate working

The scope is 4 rows, one per prompt, and `kl_mean_max` is 1e-3. There is no
averaging across the 96 steady rows to absorb a single sensitive prompt. The
long-form suite's prompts are natural long-form tasks whose next-token
distribution is sharper at the transition than the canonical suite's, so the
same route arithmetic that reads as 1e-9 on the canonical suite reads as 1.4e-3
here.

That is a real finding about the route at 16-47, and it is why the long-form
protocol exists: a per-scope limit on a 4-row scope is more sensitive than the
same limit on a 1536-row scope, which is the intended behaviour. It is also the
first measurement that attributes the sensitivity to the route's f16 operands
rather than to a transition, a decode-path selector, or nondeterminism.

## Established / not established

Established:

1. The `prefill_last` row is bit-deterministic across repeats in both arms, in
   every protocol tested. The movement is not nondeterminism.
2. The route moves the prefill's own last-token logits by ~0.4-0.6 max abs in
   every protocol where it can engage.
3. The canonical flatness and the long-form failure are the same route effect
   measured against distributions of different sensitivity.
4. `dense_wide` declines below 257 rows, so no short-prompt protocol can measure
   this route's prefill.

Not established:

5. Whether the ~0.4 max abs logit delta is dominated by the f16 operand
   conversion, the wide-row staging order, or both. The route's own recorded
   figure is 2.08e-4 relative against the exact reference, against 1.5e-7 for the
   coltile family; this packet measures the per-logit delta and does not
   decompose it.
6. Whether a route whose per-logit delta is ~0.4 can satisfy a 1e-3 mean-KL
   limit on a 4-row scope at every prompt. One natural long-form suite fails it.
   That is a scope-and-threshold question for the qualification packet, not a
   kernel question, and it is not decided here.

## Reproduce

```
bash scripts/run_qwen4exp_prefill_last_row_probe.sh
```

`benchmarks/results/2026-09-30-prefill-last-row-probe/artifact-prefill-last-row-probe.json`
