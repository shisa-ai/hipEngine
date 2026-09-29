# QA-8: split-K sparse attention is already implemented, promoted, and exact

**Date:** 2026-09-29
**Host:** Radeon 8060S (gfx1151), hostname `gfx1151`, machine-id
`55ea6c509d0b49eea8de7094a1023668`
**Model:** Qwen3.8-Flash-Next UD-Q4_K_XL
**Evidence source:** the two promotion artifacts already in-tree, plus source
inspection at `7ebd2f68b`; **no new measurement was run**
**Tree:** `7a3c9d05e`

## The question

`QA-8` is filed as *"A historical unmeasured alternative to one-wave attention,
not a demonstrated successor to today's H256 route"*, with the action:

> After `X-3`, screen one real 24-query/2-KV-head/D256 packet against the current
> owner, charging partial-output scratch and merge. Preserve selected positions
> and tails; changed softmax/reduction association is T2 and needs §3.1. Stop if
> the complete owner loses.

`X-3` is CAPTURED at `2789c19d6`, so the dependency is satisfied.

## Verdict

**Already implemented, already the production owner, already promoted with
measured whole-model evidence, and exact rather than T2. Close `QA-8`.**

The row's own premise - that this is "a historical unmeasured alternative" that
is "not a demonstrated successor to today's H256 route" - is out of date on both
counts.

## The split exists, and it is the shape the row describes

`hipengine/kernels/hip_gfx1100/attention/qwen4_exp_qsa.{hip,py}` carries an
**ordered three-pass** family:

| pass | kernel | role |
|---|---|---|
| 1 | `qsa_sparse_attention_paged_bf16_ordered_scores_f32_kernel` | scores |
| 2 | `..._ordered_coefficients_...` | softmax normalization (the merge) |
| 3 | `..._ordered_values_...` | value accumulation |

The pass-1 kernel's decomposition is the decisive line:

```cpp
const int64_t q_head        = blockIdx.x;
const int64_t selected_index = blockIdx.y;   // one block per selected position
const int64_t dimension      = threadIdx.x;
```

**`grid = (query_heads, selected_count)` - the selected-position reduction is
split across blocks, one block per position.** That is precisely the
split-K decomposition `QA-8` asks to screen, and the three passes are the
"partial-output scratch and merge" the row asks to charge for:

- **Scratch is real and is charged**: the host wrapper
  `qwen4_exp_qsa_sparse_attention_paged_bf16_ordered_f32` takes explicit
  `scores_ptr` and `coefficients_ptr` arguments - materialized intermediate
  buffers sized `query_heads x selected_count` floats each, not hidden.
- **The merge is real**: pass 2 performs the normalization that a partial-softmax
  merge would perform.
- **Positions are preserved**: the kernel iterates `selected_index` over exactly
  `selected_count`, so there is no tail assumption to violate - any count gets one
  block per position. The v2 wrapper bounds it explicitly
  (`selected_count <= 4096`, `head_dim == 256`, divisible GQA).

**The production route is the stricter version of the proposal.** A classic
split-K merges partial max/sum, which changes the reduction association. The
ordered route instead materializes *all* scores and normalizes globally, so it
**charges more scratch than a partial-merge split would** and buys exactness with
it. The row's cost concern is therefore already paid, at a higher rate than the
proposal implies.

## It is the production owner, not an alternative

`hipengine/generation/qwen4_exp_profiles.py` selects it for the c1 decode scopes,
with the single-pass kernel demoted to fallback:

```python
_selection(
    "qsa_sparse_attention",
    "qwen4exp_c1_h256_indexed_sparse_decode",
    "strict_ordered_three_pass_spans",   # production
    "strict_spans",                      # strict fallback
    "bf16_kv",
    evidence=_QSA_ORDERED_DECODE_EVIDENCE,
),
_selection(
    "qsa_sparse_attention",
    "qwen4exp_c1_h256_indexed_sparse_decode_v2",
    "strict_ordered_three_pass_v2_spans",  # production
    "strict_spans",                        # strict fallback
    "bf16_kv",
    evidence=_QSA_ORDERED_DECODE_V2_EVIDENCE,
),
```

Both variants also appear in the production **restore** set
(`restored_qsa`), so they are not merely registered - they are explicitly
restored as production. **The 24-query/2-KV-head/D256 packet `QA-8` names is
already served by the split.**

## The measured evidence, already recorded

### `2026-09-02-gfx1151-qwen38-flash-next-p6-qsa-ordered-decode.json`

`status: accepted`, `decision.promoted: true`, `decision.reason`:

> Exact T0 route improves complete p4096 decode wall in every counterbalanced
> pair, traces the expected kernels, preserves full logits and IDs, and passes
> steady lifecycle/teardown.

| field | value |
|---|---|
| `same_session_whole_model_ab.aggregate_ratio_mean` | **1.173** |
| `same_session_whole_model_ab.aggregate_ratio_95ci` | **[1.1699, 1.1762]** |
| `same_session_whole_model_ab.all_12_pairs_candidate_faster` | **true** |
| `same_session_whole_model_ab.all_cross_route_logits_exact` | **true** |
| `same_session_whole_model_ab.all_route_repeats_exact` | **true** |
| `aggregate_bound_ms_per_token` -> `candidate_ms_per_token` | **93.912 -> 80.061** |
| `protocol` | four canonical p4096 categories, 128 transitions, three counterbalanced pairs per category after warmup |
| `memory_after_close` | `active_allocations: 0`, `current_allocated_bytes: 0` |

The 95% CI lies entirely above 1.0, all 12 pairs are faster, and the measurement
is whole-model wall (not a kernel microbenchmark), which is exactly the "complete
owner" accounting the row demands.

### `2026-09-08-framework-qwen4exp-qsa-ordered-v2-production.json`

The v2 (de-latencied passes) promotion records
`within_mode_deterministic: true`, `cross_mode_output_exact: true`,
`mismatched_case_ids: []`.

## The T2/§3.1 concern does not bind

`QA-8` warns that "changed softmax/reduction association is T2 and needs §3.1".
That gate would apply to a *partial-merge* split whose association differs from
the strict route. **The promoted ordered route does not change the result:
`all_cross_route_logits_exact` and `cross_mode_output_exact` are both true, and
`mismatched_case_ids` is empty.** It is an exact (T0) route that happens to use a
split decomposition, so it clears the arithmetic bar by construction rather than
by profile calibration. No §3.1 pass is owed.

## What this does not close

- **Prefill.** The ordered family has a multi-row entry point
  (`qwen4_exp_qsa_sparse_attention_paged_bf16_ordered_rows_f32`), but the
  *selected* prefill variant for the rows path remains `strict_rows_spans`. Whether
  a split prefill owner would win is **not** measured here and is not what `QA-8`
  asked. `QA-2` remains the open prefill-side arithmetic question.
- **A different split shape.** Whether some other decomposition (different pass
  count, different scratch layout, partial merge with exact renormalization)
  would beat the promoted ordered route is a *new* candidate, not `QA-8`. It would
  need its own row and its own whole-model A/B.
- **Any new timing.** None was taken. Every number above is quoted from artifacts
  that were already committed.

## Limits

**No new measurement.** This closure rests on source inspection at `7ebd2f68b`
plus two pre-existing in-tree promotion artifacts. The performance figures are
those artifacts' own records, on their own protocol and hardware, and are
reproduced here rather than re-verified. The claim that the pass-1 grid splits
the position reduction is a direct reading of `blockIdx.y = selected_index` in the
kernel source; the claim that scratch is charged is a reading of the host
wrapper's `scores_ptr`/`coefficients_ptr` parameters. Neither was re-measured.

## Reproduce

```bash
# the split, in source
sed -n '383,400p' hipengine/kernels/hip_gfx1100/attention/qwen4_exp_qsa.hip
# the selection (production vs fallback)
sed -n '444,458p' hipengine/generation/qwen4_exp_profiles.py
# the promotion record
python3 -c "import json;d=json.load(open('benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p6-qsa-ordered-decode.json'));print(d['decision']);print(d['same_session_whole_model_ab'])"
```
