# RT-1: the full prefill suite, chunk 256 -> 4096

**Suite-wide result: 12 of 12 cases faster at chunk 4096, every one completely
separated, `token_id` identical 12 of 12 - and `logits_sha256` identical in
ZERO of 12.**

This is the evidence that decides whether RT-1's single-case `code-p4096` result
generalizes. It does.

## Protocol

All 12 fixture cases (`qwen4exp_canonical_ar_p512_p1024_p4096.json`, sha256
`42b562bd…`) - four categories x three prompt sizes - at chunk 256 and chunk 4096,
three repetitions each, `--mode prefill --execution-profile production`,
`--require-cached-build`, same host, one run. 24 invocations, all exit 0.

## Result

| case | category | 256 median (s) | 4096 median (s) | delta | % | separated | token_id | logits | peak delta |
|---|---|---:|---:|---:|---:|:--:|:--:|:--:|---:|
| code-p512 | code | 2.8405 | 2.0399 | -0.8006 | **-28.19%** | yes | same | DIFF | +0.77 GB |
| code-p1024 | code | 5.8054 | 3.7731 | -2.0323 | **-35.01%** | yes | same | DIFF | +1.24 GB |
| code-p4096 | code | 23.8169 | 14.8605 | -8.9563 | **-37.61%** | yes | same | DIFF | +6.20 GB |
| general_en-p512 | general_en | 2.8111 | 1.9983 | -0.8127 | **-28.91%** | yes | same | DIFF | +0.77 GB |
| general_en-p1024 | general_en | 5.6192 | 3.7405 | -1.8787 | **-33.43%** | yes | same | DIFF | +1.24 GB |
| general_en-p4096 | general_en | 23.2947 | 14.8005 | -8.4942 | **-36.46%** | yes | same | DIFF | +6.20 GB |
| general_ja-p512 | general_ja | 2.7876 | 1.9862 | -0.8014 | **-28.75%** | yes | same | DIFF | +0.77 GB |
| general_ja-p1024 | general_ja | 5.6060 | 3.7301 | -1.8759 | **-33.46%** | yes | same | DIFF | +1.24 GB |
| general_ja-p4096 | general_ja | 22.9992 | 14.8572 | -8.1420 | **-35.40%** | yes | same | DIFF | +6.20 GB |
| mixed_ja_en-p512 | mixed_ja_en | 2.8567 | 2.0087 | -0.8479 | **-29.68%** | yes | same | DIFF | +0.77 GB |
| mixed_ja_en-p1024 | mixed_ja_en | 5.7465 | 3.7806 | -1.9659 | **-34.21%** | yes | same | DIFF | +1.24 GB |
| mixed_ja_en-p4096 | mixed_ja_en | 23.6567 | 14.8569 | -8.7998 | **-37.20%** | yes | same | DIFF | +6.20 GB |

"Separated" is the meaningful test: the 4096 arm's **slowest** repetition is faster
than the 256 arm's **fastest**. All 12 pass.

**The effect is monotone in prompt size and flat across category.** By size:
p512 -28.2/-28.9/-28.7/-29.7%, p1024 -35.0/-33.4/-33.5/-34.2%, p4096
-37.6/-36.5/-35.4/-37.2%. No category is an outlier; the spread within a size is
under 1.6 percentage points.

Peak allocation delta is **+0.77 GB at p512, +1.24 GB at p1024, +6.20 GB at
p4096** - it tracks `min(chunk, max_sequence_length)` as the workspace model
predicts, and is identical across categories at each size.

## Numerics: the arithmetic changes in every case

`token_id` is identical in **12/12**. `logits_sha256` differs in **12/12**.

So chunk 256 -> 4096 is an **arithmetic** change for every case in the suite, not a
scheduling change. It must be described that way and cannot be certified by a
bit-identity argument.

What certifies it is the canonical numerical gate at chunk 4096, which was run
over this same 12-case fixture (`gate-wmma-16-47-chunk4096.json`):
`canonical_numerical_qualification_claim: true`, with `kl_mean 0.00038`,
`kl_p95 0.00176`, `kl_p99 0.00542`, `kl_max 0.0153` against thresholds
0.001 / 0.005 / 0.02 / 0.05, and `top1` above the 0.99 / 0.97-per-scope floor.

**Stated precisely, because the distinction matters:** that artifact records
`profile_qualification_claim: false`, with blockers `category-heldout numerical
evidence`, `strict-teacher and applicable BF16-relative evidence`, `exact
control/isolation/graph/lifecycle evidence`, `task-quality evidence`, and
`candidate route engagement and repair-oracle coverage`. Those are **route-level
(X-6/X-7) blockers that were already open before this work** - they are not
introduced by the chunk size and are not cleared by it. The chunk-4096
configuration passes the **canonical numerical** qualification; it does not carry
a full profile qualification, and this packet does not claim one.

## Promotion decision: NOT a blanket default flip

**No default changed.** The reason is concrete, not "needs more evidence":

`DEFAULT_MAX_PREFILL_CHUNK_TOKENS` is a **global** default governing every model
the engine serves, and the prefill workspace is `min(chunk_size,
max_sequence_length)` for every one of them. Raising it to 4096 multiplies the
prefill workspace on **every model and every host**, while the +6.20 GB admission
cost above is measured for **one model on one host**. On a host where the model
currently fits at chunk 256 with little headroom, a blanket 4096 turns a working
configuration into a possible OOM. That is a real regression risk introduced by
generalizing beyond the measurement, and it is the blocker.

Two things this packet establishes that bound the decision:

1. **The historical short-request objection is dead.** The sibling packet shows a
   512-token request is unaffected in both time and memory, because the workspace
   is already shape-aware through `max_sequence_length`. So the only remaining
   constraint on a larger chunk is **peak memory**, not short-request latency.
2. **There is no scoped hook to promote through.** This model has no entry in a
   geometry-keyed chunk table (the `GGUF_PREFILL_n` pattern used for the
   Qwen3.5 H2048-MoE geometry), and the runner has no memory-aware admission
   guard. So the choice today is a global flip or new mechanism, and neither is
   supported by one model's measurement.

**The concrete next step, if this is pursued**, is a memory-bounded chunk
selection: choose the largest chunk the host can actually admit, falling back
toward 256, which gets the -28% to -38% where it fits and cannot OOM where it does
not. That is a real change with its own test and gate burden, not a constant edit.

## Limits

- **One host** (gfx1151 APU, unified memory, 120 GB GTT). The admission cost is
  host-specific.
- **One model** (Qwen3.8-Flash-Next UD-Q4_K_XL).
- **Prefill only.** This parameter does not affect decode.
- The gate cited is the canonical **numerical** qualification, not a full profile
  qualification.
