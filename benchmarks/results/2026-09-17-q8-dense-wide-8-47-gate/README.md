# Wide-row Q8_0 dense prefill at layers 8-47: gate result

**Verdict: fail.** Extending the wide-row dense route eight layers deeper than the
promoted scope (16-47) overshoots the calibrated tail limits while the mean is
still inside them. The passing boundary lies between 8 and 16; this run is the
failing half of the bisection.

Same protocol as the promoted 16-47 gate: Qwen4Exp UD-Q4_K_XL canonical
exact-token fixture, 12 cases (4 categories at 512/1024/4096), 128 decode
transitions each, production profile, chunk 1024, three candidate repeats, one
warmup after the selector flip.

| Measure | 8-47 | Limit | Headroom | 16-47 (promoted) |
| --- | ---: | ---: | ---: | ---: |
| Rows | 1548 | | | 1548 |
| Mean KL | **8.44e-4** | 1e-3 | 1.18x | 3.80e-4 |
| p95 KL | **4.07e-3** | 5e-3 | 1.23x | 1.76e-3 |
| p99 KL | 1.12e-2 | 2e-2 | 1.79x | 5.42e-3 |
| Max KL | **7.30e-2** | 5e-2 | **0.68x — fail** | 1.53e-2 |
| Top-1 agreement | **1532/1548 = 0.98966** | 0.99 | **fail by one row** | 0.99354 |
| Verdict | **fail** | | | pass |

`hard_gates_passed: false`, `requires_outlier_review: true`,
`eligible_for_automatic_admission: false`, one row over the 2e-2 review boundary,
scope failures on category `code`, shape `c1`, transition `steady`.

## The failure is a tail failure

The mean and p95 stay inside their limits, and every per-category mean does too,
but with almost no room left:

| Category | Rows | Mean KL | p95 KL | Max KL | Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `code` | 387 | 8.04e-4 | 3.62e-3 | **7.30e-2** | 0.9871 |
| `general_en` | 387 | **9.79e-4** | **4.97e-3** | 1.30e-2 | 0.9845 |
| `general_ja` | 387 | 8.69e-4 | 4.91e-3 | 1.62e-2 | 0.9974 |
| `mixed_ja_en` | 387 | 7.25e-4 | 3.46e-3 | 1.99e-2 | 0.9897 |

The single outlier is `code-p4096` at `teacher_step` 1 (`kl 0.0730`), and the
top-1 miss is by exactly one row: 0.99 of 1548 is 1532.5, so 1533 matches are
required and 1532 were obtained. All 16 top-1 mismatches are inside the 239-row
flip-eligible set and none outside it, and the 12 prefill-last rows are exact
(mean KL 3.76e-8), so the drift is decode-transition drift on generated text,
not a prefill defect.

`general_en` at 9.79e-4 mean is the binding constraint: it is 2% inside the
limit, so this scope fails on the tail while the mean sits at the edge. There is
no arithmetic change here that would recover a 2x margin — the route is a
retiling of the f16 WMMA arithmetic (see below), and the drift is the f16
accumulation itself over 40 layers of a 48-layer stack.

## Why the route cannot go all the way to 0-47

The wide route is a retiling of the f16 WMMA dense route, not a reassociation.
Both routes emit the same candidate logits at layers 16-47
(`candidate_logits_sha256 65370710c0b9c40f` in both gate artifacts, identical
top-1 agreement 0.99354), so the WMMA arm's `0-47` failure (mean KL 1.099e-3)
transfers to this route rather than being escaped by it. What is open is only how
deep the passing scope reaches.

## Protocol and provenance

- Command: `scripts/execution_profile_q8_wmma_prefill_layers_gate.py --route
  dense_wide --layers 8-47 --decode-steps 128 --repeat-runs 3
  --prefill-chunk-size 1024`
- Route surface: post-binder `HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE` +
  `HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS=8..47`, with
  `HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS` cleared in both arms. Teacher: the exact
  coltile chain, `coltile8_rowbatch4_f32_f32_out`.
- `timing_protocol: none_full_logits_only_v1`; the artifact carries no timing and
  `performance_claim` is false by construction.

## What this does not say

- **Not a rejection of the wide route.** Layers 16-47 stay promoted; this is the
  scope boundary, not a regression.
- **Not a task-quality or long-form-factual certificate**, and it certifies no
  scope other than 8-47 and no quant other than `gguf_ud_q4_k_xl`.
- **Not evidence about 12-47 on its own.** That arm was run separately and also
  fails, on the `general_en` category
  ([artifact](../2026-09-17-q8-dense-wide-12-47-gate/README.md)). Together the two
  runs bound the passing scope to at most a few layers below 16; neither decides
  13-47 or 14-47.
