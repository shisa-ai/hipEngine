# `dense_wide` 16-47 re-gate at HEAD — the live default's certificate is stale, and the envelope still passes

**Measured 2026-09-30 at `cac0f1da5`.** Full canonical protocol: 12 cases, 128
decode steps, 3 repeat runs, prefill chunk 1024. Artifact:
`artifact-dense-wide16-47-chunk1024-at-head.json`.

This is the corrected action item from `20260930T111452`: `dense_wide`, not
`wmma`, is the route live at 16-47
(`PRODUCTION_Q8_DENSE_WIDE_PREFILL_LAYERS`), and its 2026-09-16 certificate
carried teacher hash `550bb9b8`, which `20260930T110625` showed no longer
reproduces at HEAD.

## Result

| | 2026-09-16 certificate (`1f06cd5e3`) | HEAD (`cac0f1da5`) | |
| --- | --- | --- | --- |
| teacher hash | `550bb9b8…` | **`294e6c08d5b2c315…`** | moved |
| candidate hash | `65370710…` | **`10fd0ae50b6eb9ce…`** | moved |
| `kl_mean` | 3.79552e-4 | 4.38817e-4 | +15.6% |
| `kl_p95` | 1.761933e-3 | 2.360965e-3 | +34.0% |
| `kl_p99` | 5.424248e-3 | 5.843182e-3 | +7.7% |
| `kl_max` | 1.530440e-2 | 2.3411946e-2 | +53.0% |
| top-1 | 0.993540 | 0.994186 | +0.06pp |
| rows | 1548 | 1548 | same |
| rows over review | 0 | **1** | +1 |
| hard gates | passed | **passed** | |

**The certificate no longer reproduces, and the finding now covers the live
production default rather than a retired route.**

**The envelope still passes every hard gate**, with the tightest three holding
2.1x-2.3x headroom (mean 4.39e-4 against 1e-3, p95 2.36e-3 against 5e-3, max
2.34e-2 against 5e-2), and top-1 is **better** at HEAD (0.9942 vs 0.9935). One
row now crosses the review boundary where none did. As with the `wmma` finding,
this is a **review requirement, not a demonstrated quality failure**.

## The teacher hash was predicted exactly

`20260930T111452` predicted the teacher hash at HEAD would be `294e6c08…` **by
identity rather than by running the gate**: both the `wmma` and `dense_wide`
gates clear every selector for the teacher arm, so both teachers are the same
exact chain. The measured value is `294e6c08d5b2c315…`.

The candidate hash is likewise identical to the `wmma` gate's candidate at HEAD
(`10fd0ae5…`), confirming the documented byte-identity of the two routes
persists across the decode-path changes that moved the baseline.

## A real gap: the false-pass detector does not cover this route

The gate reports `candidate_route_engaged: False` and `teacher_route_engaged:
False`, with no selector differing between arms:

```
native_batch_decode_session: t=0.0000 c=0.0000 differs=False
q4k_rowtile:                 t=1.0000 c=1.0000 differs=False
wmma_prefill:                t=0.0000 c=0.0000 differs=False
```

**That `False` is a coverage artifact, not evidence that the route failed to
engage.** `wmma_prefill_dispatch_recorder` records exactly two things
(`hipengine/runtime/gguf_linear.py`):

- `selector: "wmma_prefill"` — the WMMA route, via
  `_record_wmma_prefill_dispatch`; and
- collateral selectors that `use_wmma` moves but that are not a route, via
  `_record_selector` (`q4k_rowtile`, `native_batch_decode_session`, …).

**The `dense_wide` route has no recorder.** So the engagement mechanism added in
`056e6fbd8` — the one that caught the WMMA false pass — cannot see the route that
is the **live production default at 16-47**. The gate has a row floor for it
(`ROUTE_ROW_FLOORS["dense_wide"]`) but no dispatch recorder.

This is the same failure mode class the campaign has been hunting, in the one
place where the detector is blind. It is recorded in `docs/REFACTOR.md`.

## What this does and does not establish

- **Does establish:** the live default's certificate is stale (both hashes
  moved); the full envelope at HEAD still passes every hard gate with >2x
  headroom; the two routes remain byte-identical in candidate logits at HEAD.
- **Does not establish:** that the `dense_wide` route ran in this gate. The
  tracker cannot answer that. The drift measured here is real and the envelope
  passes, but **attribution to `dense_wide` specifically rests on the documented
  byte-identity with the `wmma` route**, which does have a recorder.
- **Not measured:** a `dense_wide` run with a working recorder, which requires
  adding one.

## Reproduce

```bash
HIPENGINE_HIP_ARCH=gfx1151 uv run --no-sync python \
  scripts/execution_profile_q8_wmma_prefill_layers_gate.py \
  --model-root <Qwen3.8-Flash-Next-UD-Q4_K_XL>/UD-Q4_K_XL \
  --route dense_wide --layers 16-47 --decode-steps 128 --repeat-runs 3 \
  --prefill-chunk-size 1024 --output <artifact>.json
```
