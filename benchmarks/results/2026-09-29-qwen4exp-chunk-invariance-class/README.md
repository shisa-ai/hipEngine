# The chunk-invariance class is {1024, 2048, 4096} - chunk 256 is outside it

RT-1's retained claim is that chunk size "changes scheduling, not arithmetic", evidenced by
`token_id` and `logits_sha256` being identical across its arms (1024 vs 4096). Reading the
artifacts across all three RT-1 packets shows that claim is **correct for chunks >= 1024,
and false for chunk 256.**

## The measurement

Twelve canonical cases, comparing the published `logits_sha256` across chunk sizes. All
figures are read from existing RT-1 artifacts; nothing was re-run.

| comparison | cases identical | cases differing |
|---|---:|---:|
| **chunk 1024 vs chunk 4096** | **12 of 12** | 0 |
| chunk 256 vs chunk 4096 | 0 of 12 | **12** |
| `token_id` across all three chunks | **12 of 12** | 0 |

An independent three-point run on `code-p4096` (its own packet) gives the same answer:

| chunk | `logits_sha256` | `token_id` |
|---|---|---|
| 1024 | `7270a972da43ff46...` | 248068 |
| 2048 | `7270a972da43ff46...` | 248068 |
| 4096 | `7270a972da43ff46...` | 248068 |

So the invariance class is exactly **{1024, 2048, 4096}**, and **256 is a different
arithmetic**.

## Why this matters

1. **RT-1's claim is scoped, not wrong.** Its arms were 1024 and 4096, and those are
   identical on all 12 cases. The claim should be read as "within the >= 1024 class".
2. **`token_id` is invariant across all four chunk sizes** (256 included) while logits are
   not. So the divergence at 256 is a **sub-argmax** difference - it does not change the
   chosen token on these cases, but it is a real arithmetic difference that could change a
   near-tie.
3. **Chunk 256 is not numerically interchangeable with 1024+.** Any comparison that mixes
   the two - a baseline measured at 256 against a candidate at 1024, or a gate run at 256
   quoted for a 1024 deployment - is comparing two configurations. This is the same
   protocol-confound class the external review checklist (C1-C6) warns about.
4. **It explains why the X-6 gate can be expected to pass at chunk 4096**: the certification
   was taken at chunk 1024, and 1024 vs 4096 is bit-identical on all 12 cases. That is a
   prediction, and the chunk-4096 gate run tests it directly.

## Likely mechanism (not established)

Chunk size sets the per-chunk row count, and several Qwen4Exp kernels are selected by
row/M thresholds (the dense wide-256 route, the coltitle/rowbatch families, the staged
routes). A threshold between 256 and 1024 rows would select a different kernel - and
therefore a different reduction order - at chunk 256. **This is a hypothesis, not a
measurement**; no dispatch trace was taken for this packet.

## Limits

- **Attribution from existing artifacts, not a new run.** The twelve comparisons come from
  three prior packets; the hashes are theirs.
- One model, one quant, one host, one execution profile. The class boundary could differ on
  another shape or a different kernel-selection table.
- `logits_sha256` equality proves bit-identical published logits **for the compared tensor
  as the harness hashes it**; it is not a proof about intermediate tensors.
- Only chunk sizes 256, 1024, 2048 and 4096 were compared. The class boundary between 256
  and 1024 is not narrowed further.
- The mechanism paragraph is explicitly a hypothesis.
