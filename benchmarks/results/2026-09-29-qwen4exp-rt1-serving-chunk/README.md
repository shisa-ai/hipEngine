# RT-1: the serving-relevant chunk comparison (256 -> 4096)

**The win against the serving default is far larger than the win against the
benchmark protocol's value.**

RT-1's earlier packet measured 1024 -> 4096 (-4.94%). That was then qualified:
1024 is the *benchmark protocol's* chunk size, while the serving path's default is
**256** (`EngineLoopConfig.max_prefill_chunk_tokens`). This packet measures the
comparison that actually governs a promotion.

## Result: `code-p4096`, three repetitions per arm

| chunk | samples (s) | median (s) | peak allocation |
|---:|---|---:|---:|
| 256 | 23.8427, 23.8886, 31.5102 | 23.8886 | 83,198,793,992 B |
| 4096 | 14.7967, 14.6882, 14.6832 | **14.6882** | 89,403,335,432 B |

**Median delta: -9.2004 s (-38.51%).** Min-to-min: -9.1595 s (-38.42%).

## Separation

The arms **do not meaningfully overlap**: the 4096 arm's *slowest* repetition
(14.7967 s) is faster than the 256 arm's *fastest* (23.8427 s) by 38%. A naive
"do the ranges overlap" test returns true only because the 256 arm's outlier
extends its range upward, which makes the comparison *stronger*, not weaker.

Honest note on the 256 arm: its third repetition (31.5102 s) is a 1.32x outlier
against its own median, and inflates its within-arm stdev to 3.6037 s against the
4096 arm's 0.0524 s. The |median delta| is 2.55x the larger stdev - weaker
separation by that statistic than the 5.70x recorded for 1024 -> 4096. The
conclusion does not rest on the median, though: min-to-min is -38.42% and the
comparison of worst-of-4096 to best-of-256 is unambiguous.

## Numerics: the chunk size is NOT invariant here

| | 256 | 4096 |
|---|---|---|
| `token_id` | 248068 | 248068 (**identical**) |
| `logits_sha256` | `e717076f…` | `7270a972…` (**differs**) |

So 256 -> 4096 **changes the arithmetic** relative to today's serving default.
This is not a regression against the numerical gate: the X-6 production-profile
gate passes at chunk 4096 and hashes bit-identically to the chunk-1024
certification, so 4096 is the *certified* configuration. But it does mean this is
an arithmetic change, not a pure scheduling one, and it must be described that way.

Note this is a different invariance result from the 1024 <-> 4096 comparison in
the sibling packet, where the two arms hashed *identically*. The route is
invariant between 1024 and 4096 but not between 256 and 4096. Both statements are
measured; neither should be generalized to "chunk size does not affect numerics".

## Memory

Peak grows 83,198,793,992 -> 89,403,335,432 B = **+6.20 GB**. On this host GTT is
120 GB and 121 GB of RAM is available, so 89.40 GB leaves **30.6 GB of headroom
(25.5%)**. This is an admission cost paid on long prompts by the prompt that
receives the win, not a cost imposed on short ones.

## Limits - what this packet does NOT establish

- **One case, one category** (`code-p4096`). A claim retained on a single fixed
  prompt is not retainable. The 12-case, 4-category suite run is the evidence that
  would generalize this; until it lands, the -38.51% is a single-case result.
- **Prefill only.** Decode is unaffected by this parameter.
- **One host** (gfx1151 APU, unified memory). The +6.20 GB admission cost is
  host-specific and would be a real blocker on a discrete-memory device.
- The 256 arm carries an outlier; the arm was not re-run to replace it.

**No default changed.** The serving default is still 256.
