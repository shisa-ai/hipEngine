# The `wmma` gate's two arms do not differ only in the WMMA route

Found while trying to explain the prefill-scope discrepancy recorded in
`../2026-09-29-qwen4exp-x7-category-heldout/README.md`. **The discrepancy is
explained by this, but the finding is larger than the discrepancy: the arms are
not isolating the route they claim to measure.**

## The env var gates more than the WMMA dispatch

`HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS` reaches `launch_gguf_linear`, where
`use_wmma` is derived from it and then used for decisions that have nothing to do
with the WMMA prefill route:

```python
use_wmma = _resolve_use_wmma_prefill(use_wmma_prefill)
wmma_filter = _wmma_prefill_weight_filter.get()
if use_wmma and wmma_filter is not None:
    use_wmma = bool(wmma_filter(weight))
f_rowtile = (not use_wmma) and _resolve_use_q4k_rowtile(None)     # <- decode path
if (_native_batch_decode_session_enabled and (2 <= rows <= 8)
        and not use_wmma and ...):                                 # <- decode path
```

and further down the same call it also gates `enabled=not use_wmma`,
`use_rowtile=f_rowtile` and `use_rowtile=f_rowtile and not use_q4_pack8_wmma`.

So setting or clearing the variable changes **at least four** selectors, several
of which apply at row counts the WMMA route itself refuses. The route declines at
`rows <= 1`, but `f_rowtile` and the rows-2-8 batch-decode session are *decode*
decisions and move anyway.

The gate's recorded surface is `post-binder HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS`,
which reads as "this run measures the WMMA route". It does not. The measured KL
is the WMMA route **plus** `f_rowtile` **plus** the batch-decode session.

## Why this explains the prefill/decode asymmetry

The canonical run reports `shape=prefill_last` at `4.41452e-09` and
`shape=c1`/`steady` at `3.82518e-04`. That asymmetry is what a decode-path side
effect looks like: a selector that only bites at rows <= 8 leaves the prefill
output identical and moves the decode steps.

**This is the leading explanation, not a verified mechanism.** What is verified
is that the arms differ in more than the WMMA route; the mapping from that to the
exact per-scope numbers is inferred.

## The WMMA route's own gate

`_dispatch_can_use_wmma_prefill` requires
`dispatch.key.variant.startswith("prefill_")`:

| variant | rows=1 | rows=64 | rows=512 | rows=1024 |
|---|---|---|---|---|
| `coltile8_rowbatch4_f32_f32_out` (the exact leaf) | False | False | False | False |
| `prefill_q8_0_f32_f32_out` | False | **True** | **True** | **True** |
| `prefill_f32_f32_out` | False | **True** | **True** | **True** |
| `gemv_q8_0_f32_f32_out` | False | False | False | False |

So whether the route engages depends on which variant the dispatch carries at the
moment `_wmma_prefill_dispatch` runs, which is a dispatch-order property and not
something the gate's env surface controls or records.

## Consequence for the certification

The 2026-09-16 certification and its bit-identical HEAD re-run both measure a
compound. They are reproducible and they are not meaningless - but **the drift
they report cannot be attributed to the WMMA route alone**, and the gate's
surface string says it can. Re-attributing it needs either a run that isolates
`use_wmma` from `f_rowtile` and the batch-decode session, or a per-selector
ablation. Neither exists yet.

This is the same class as the two false passes and the qualification-claim
over-reach already recorded in this packet: a green number whose scope is
narrower than its label.
