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

## Confirmed on an isolated case: the drift is entirely on the decode path

`artifact-canonical-codep512-d1.json` runs a **single** canonical case
(`code-p512`) for **one** decode step, so the post-prefill row and the decode row
are each one row with no averaging:

```
strict_logits_sha256    = 93042f1a4d162b7568a2...
candidate_logits_sha256 = 1b4e5c0fa5d3bd266c4a...     (different: the arms do differ)
```

| row | rows | mean KL | max |
|---|---|---|---|
| `shape=prefill_last` / `transition=prefill_to_c1` | 1 | **5.8014e-12** | 5.8e-12 |
| `shape=c1` / `transition=steady` | 1 | **1.0184e-04** | 1.02e-4 |

The post-prefill output is identical to f32 precision - 5.8e-12 is rounding
noise - and the entire drift sits on the single decode step. The prediction was
made before the run: if the arms differ in decode-path selectors rather than in
the WMMA prefill route, the prefill row must be flat and the decode row must not
be. It is.

So **the canonical `wmma` gate does not measure a prefill route.** Its recorded
`candidate_chain` is `wmma_prefill_f32_f32_out` and its surface is
`post-binder HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS`, but the quantity that moves is
a decode-path selector reached through `use_wmma` (`f_rowtile`, the rows-2-8
native batch-decode session, `use_rowtile`). The prefill route contributes
nothing measurable here.

## The heldout run does the opposite, and that is unexplained

| run | prefill row | decode rows |
|---|---|---|
| canonical (512 tok, 1 or 128 steps) | **5.8e-12** | 1.02e-04 |
| heldout (39-71 tok, 24 steps) | **2.06179e-03** | 1.99214e-04 |

In the heldout run the prefill row moves **ten times more** than the decode rows.
That cannot be the same decode-path effect, so something engages on the heldout
prefill that does not engage on the canonical one - despite both having
`rows > 1` and the same `in_features`.

The candidate explanation is dispatch order: `_dispatch_can_use_wmma_prefill`
requires the variant to start with `prefill_`, and the exact leaf
`coltile8_rowbatch4_f32_f32_out` fails that at every row count. If the canonical
prefill arrives carrying the coltile leaf and the heldout prefill arrives
carrying a `prefill_*` variant, the route engages in one and not the other. **Not
verified.** It is testable by recording the variant the dispatch actually carries
at `_wmma_prefill_dispatch` for both protocols, which neither run does today.

## What is established, and what is not

Established:

1. The gate's arms differ in at least four selectors, not one (source of
   `launch_gguf_linear`).
2. On the canonical configuration the post-prefill output is bit-identical and
   the whole drift is on the decode path, confirmed on an isolated single case.
3. Therefore the canonical `wmma` gate does not measure the route its name,
   surface and `candidate_chain` claim.

Not established:

4. Why the heldout prefill row moves. Recorded as an open question with a named
   candidate explanation and a test that would settle it.
