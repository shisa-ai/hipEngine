# Full-attention context route: 1.74x on the largest prefill launch

**Host.** Physical host `55ea6c509d0b49eea8de7094a1023668`, AMD Radeon 8060S,
gfx1151, TheRock ROCm 10.0.0, `HIPENGINE_HIP_ARCH=gfx1151`.

**Model and workload.** `unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL`, BF16 KV,
block_size 256, prompt processing of a 4096-token prompt in 1024-token chunks,
`scripts/qwen4exp_profile_gap.py --mode prefill --case-id code-p4096
--repetitions 3 --prefill-chunk-size 1024`.

## Result

The full-attention decode-context leaf runs three times per attention layer on
this workload: 1024 rows at `max_context_len` 1024, 1024 rows at 2048, and 3
rows at 2051. Two changes that alter no floating-point operation move them from
67.740 to 40.766 ms per layer, a 1.662x speedup, and the prefill wall time from
16.6865 to 16.3325 s. Across all four mtp-bench categories at p4096 the category
mean moves from 248.109 to 253.350 tok/s (+2.11%), with every category improving.

| Launch | Shipped (ms) | Promoted (ms) | Speedup |
| --- | ---: | ---: | ---: |
| rows=1024, `max_context_len`=1024 | 12.413 | 8.715 | 1.424x |
| rows=1024, `max_context_len`=2048 | 54.731 | 31.562 | 1.734x |
| rows=3, `max_context_len`=2051 | 0.596 | 0.489 | 1.219x |
| per layer | 67.740 | 40.766 | 1.662x |
| 12 layers | 812.9 | 489.2 | 1.662x |

| Whole-prefill metric | Shipped | Promoted | Change |
| --- | ---: | ---: | ---: |
| Wall median (s) | 16.6865 | 16.3325 | -354.0 ms (-2.12%) |
| Tokens/s | 245.415 | 250.813 | +2.20% |

Both arms were measured in one session, alternating by an environment switch, so
the comparison is not exposed to session drift.

### All four categories, p4096

The same A/B repeated across every mtp-bench category at p4096, three
repetitions per case. Every category improves, in a band narrow enough that no
single case carries the result:

| Category | Shipped (tok/s) | Promoted (tok/s) | Change |
| --- | ---: | ---: | ---: |
| code | 246.552 | 253.033 | +2.63% |
| general_en | 248.581 | 252.846 | +1.72% |
| general_ja | 249.248 | 253.567 | +1.73% |
| mixed_ja_en | 248.056 | 253.953 | +2.38% |
| **category mean** | **248.109** | **253.350** | **+2.11%** |

## What changed

`qwen35_paged_full_attn_decode_context_tensor_batch_kernel` gains two levers:

1. **The per-token physical-offset table is int32 instead of int64.** It is the
   dominant term in the dynamic shared-memory request, so this drops the request
   from 25,607 to 17,223 bytes at `max_context_len` 2048 and lifts residency
   from 2 to 3 blocks per CU.
2. **The QK phase scores two tokens per warp iteration.** Each iteration was one
   serial dot-product plus five-step-shuffle chain; the second token gives the
   scheduler an independent chain to interleave.

The int32 table is used only when the whole offset range provably fits, decided
from the KV cache element count the caller supplies. Otherwise the int64 table
is used and the two-token unroll still applies.

## Correctness

The route is **byte-identical** to the shipped one, not merely within an
envelope: `maxdiff = 0.000e+00` over the full 1024 x 24 x 256 output, and
`tests/test_gpu_fullattn_context_promoted_route.py` asserts equality across three
shapes and all three capacity choices. Each token keeps its original dimension
order and reduction tree, the per-warp max fold is order-independent, and the
separate exp/sum pass keeps its own token-to-warp map, so the softmax
denominator is untouched. No KL or top-1 gate applies, because no floating-point
operation changed.

That is what separates this result from the MoE router work on the same campaign:
a router change of one ulp can flip which of 512 experts runs and move a model
logit by O(1), which is why a 2.16x kernel win there could not be promoted. This
change cannot move a logit at all.

## Isolating the levers

A standalone replay of the captured launch, 15 repetitions each, all
byte-identical:

| Variant | ms | Speedup |
| --- | ---: | ---: |
| shipped `<true,2>` geometry | 54.667 | 1.000x |
| int32 offsets | 38.992 | 1.403x |
| two-token QK unroll | 43.420 | 1.260x |
| both | 31.381 | 1.743x |
| `VALUE_VECTOR`=1 | 67.663 | 0.809x |
| `VALUE_VECTOR`=1 + int32 | 46.071 | 1.188x |

The two levers compose near-multiplicatively (1.403 x 1.260 = 1.768 measured
against 1.743), which indicates they relieve independent constraints.

`VALUE_VECTOR`=1 is recorded because it falsifies a plausible-looking hypothesis:
with `VALUE_VECTOR`=2 and `head_dim`=256 the value loop skips threads 128..255
entirely, so half the block idles through the whole value phase. Putting all 256
threads on one dimension each is byte-exact and 19% **slower**, so that phase is
not short of threads.

## Artifacts

- `promoted-route.json` - machine-readable form of the tables above.
- `scripts/qwen4exp_fullattn_launch_probe.py` - logs the real launch geometry,
  live-count distribution, and per-launch time, and can dump one launch's exact
  inputs.
- `scripts/qwen4exp_fullattn_prefill_microbench.py` - replays a dumped launch and
  checks byte equality before reporting timing.
