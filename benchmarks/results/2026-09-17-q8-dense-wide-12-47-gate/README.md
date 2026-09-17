# Wide-row Q8_0 dense prefill at layers 12-47: gate result

**Verdict: fail, on one category.** The global envelope passes with 1.5-2.2x
headroom, but the `general_en` category exceeds the same mean and p95 limits, so
the run is not admissible. Together with the 8-47 failure this bounds the passing
scope to at most a few layers below 16.

Same protocol as the promoted 16-47 gate: Qwen4Exp UD-Q4_K_XL canonical
exact-token fixture, 12 cases (4 categories at 512/1024/4096), 128 decode
transitions each, production profile, chunk 1024, three candidate repeats, one
warmup after the selector flip.

| Measure | 12-47 | Limit | Headroom | 16-47 (promoted) |
| --- | ---: | ---: | ---: | ---: |
| Rows | 1548 | | | 1548 |
| Mean KL | **6.47e-4** | 1e-3 | 1.54x | 3.80e-4 |
| p95 KL | **3.61e-3** | 5e-3 | 1.38x | 1.76e-3 |
| p99 KL | 9.01e-3 | 2e-2 | 2.22x | 5.42e-3 |
| Max KL | 2.25e-2 | 5e-2 | 2.22x | 1.53e-2 |
| Top-1 agreement | 1533/1548 = 0.99031 | 0.99 | at the floor | 0.99354 |
| Verdict | **fail** | | | pass |

`hard_gates_passed: false`, one scope failure: category `general_en`. Two rows
over the 2e-2 review boundary, all 15 top-1 mismatches inside the 192-row
flip-eligible set and none outside it.

## The binding category

| Category | Rows | Mean KL | p95 KL | Max KL | Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `code` | 387 | 4.81e-4 | 3.09e-3 | 1.10e-2 | 0.9845 |
| `general_en` | 387 | **1.089e-3 — fail** | **6.06e-3 — fail** | 2.25e-2 | 0.9793 |
| `general_ja` | 387 | 5.76e-4 | 3.53e-3 | 1.41e-2 | 1.0000 |
| `mixed_ja_en` | 387 | 4.44e-4 | 2.23e-3 | 9.55e-3 | 0.9974 |

The global top-1 is exactly at the floor (0.99 of 1548 is 1532.5, so 1533
matches are the minimum and 1533 were obtained), which leaves no room for the
category-level failure to be absorbed.

## The scope boundary, measured

Three scopes, same fixture and protocol, per-category means:

| Scope | ALL | `code` | `general_en` | `general_ja` | `mixed_ja_en` |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16-47 | 3.80e-4 | 4.10e-4 | **5.24e-4** | 3.40e-4 | 2.45e-4 |
| 12-47 | 6.47e-4 | 4.81e-4 | **1.089e-3** | 5.76e-4 | 4.44e-4 |
| 8-47 | 8.44e-4 | 8.04e-4 | 9.79e-4 | 8.69e-4 | 7.25e-4 |

Drift grows with the number of layers carried on f16 arithmetic, and
`general_en` is the most sensitive category at every scope. The promoted 16-47
scope already spends about 40% of that category's headroom (5.24e-4 of 1e-3), and
four more f16 layers exhaust it. `general_en` is not monotone between 12-47 and
8-47 (1.089e-3 against 9.79e-4) while the aggregate is, so per-category values
carry a spread of roughly 15%; the trend, not any single category value, is the
finding.

**Consequence:** the wide route cannot reach the ~2.5 s that a full 0-47
extension would be worth. The reachable extension is at most a few layers, worth
at most ~0.4 s by the per-layer costs, and it would sit at the numerical limit
with a real chance of failing its own gate. The route's arithmetic is the f16
WMMA arithmetic (`dense_wide256` is a retiling of it, not a reassociation — both
emit `candidate_logits_sha256 65370710c0b9c40f` at 16-47), and that arithmetic is
what the WMMA `0-47` arm already failed at (mean KL 1.099e-3).

## Protocol and provenance

- Command: `scripts/execution_profile_q8_wmma_prefill_layers_gate.py --route
  dense_wide --layers 12-47 --decode-steps 128 --repeat-runs 3
  --prefill-chunk-size 1024`
- Route surface: post-binder `HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE` +
  `HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS=12..47`, with
  `HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS` cleared in both arms. Teacher: the exact
  coltile chain, `coltile8_rowbatch4_f32_f32_out`.
- `timing_protocol: none_full_logits_only_v1`; no timing, `performance_claim`
  false by construction.

## What this does not say

- **Not a rejection of the wide route.** Layers 16-47 stay promoted.
- **Not a task-quality or long-form-factual certificate**, and it certifies no
  scope other than 12-47 and no quant other than `gguf_ud_q4_k_xl`.
- **Not a statement that 13-47 or 14-47 would pass.** They were not run. The
  measured facts are that 8-47 and 12-47 fail and 16-47 passes with 2.6x
  headroom.
