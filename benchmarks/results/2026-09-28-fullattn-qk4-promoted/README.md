# Full-attention context leaf: four-token QK unroll

**Result: 54.611 ms -> 28.563 ms (1.912x) on the p4096 prefill launch, and
+2.65% prefill throughput across all four mtp-bench categories.** Byte-identical
to the shipped route, so no numerical gate applies.

Host: `55ea6c509d0b49eea8de7094a1023668` (AMD Radeon 8060S, gfx1151, TheRock ROCm
10.0.0). Model `unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL` (UD-Q4_K_XL), BF16 KV,
block size 256, 1024-token prefill chunks.

## What changed

This unit adds one lever to the route promoted on 2026-09-27
(`../2026-09-27-fullattn-context-promoted/`) and removes one that did not work.

| lever | state |
| --- | --- |
| int32 per-token offset table | carried over (shared memory 26,663 -> 18,471 B, residency 2 -> 3 blocks/CU) |
| two-token QK unroll | carried over |
| **four-token QK unroll** | **added** |
| per-block base table | **implemented, measured, rejected, removed** |

## Measured

Replay of the captured production launch (`rows=1024`, `max_context_len=2048`,
`block_size=256`, 24 query heads, 2 KV heads, head dim 256, 37,761,024
`(row, head, token)` triples), 15 repetitions, byte-equality asserted before any
timing:

| variant | shared bytes | blocks/CU | ms | vs shipped | bit-exact |
| --- | ---: | ---: | ---: | ---: | --- |
| shipped fixed256 leaf | 18,471 | 3 | 54.611 | 1.000x | reference |
| per-token int32, QK=2 | 18,471 | 3 | 31.292 | 1.744x | yes |
| **per-token int32, QK=4 (promoted)** | **18,471** | **3** | **28.522** | **1.913x** | **yes** |
| per-token int64, QK=4 | 26,663 | 2 | 39.554 | 1.380x | yes |
| per-block base table, QK=4 | 10,423 | 6 | 40.384 | 1.352x | yes |
| per-block base table, QK=2 | 10,423 | 6 | 42.513 | 1.285x | yes |

The four-token unroll is 1.097x over the two-token unroll at identical shared
memory, so that gain is instruction-level parallelism rather than occupancy.

## The rejected lever

Replacing the per-token offset table with one int64 element offset per logical
block - rebuilt in-loop from an incrementally tracked `(logical block, offset in
block)` pair - cuts the shared-memory request from 18,471 to 10,423 bytes and
doubles residency from 3 to 6 blocks per CU. It measured **1.41x slower**.

The table is not overhead. It is a precomputed address stream: reading the next
address from shared memory lets the hardware issue independent loads. The
incremental tracker instead puts the address on a serial recurrence that every
load depends on, and in a latency-bound kernel that costs more than the table
ever did. The int64 variant's 1.38x regression at only one fewer block per CU is
the same effect read the other way, which is why the int32 table sits between two
worse extremes rather than at one end.

This falsifies the estimate in the prior unit, which named this as the next lever
and predicted 1.2-1.4x on top of the then-promoted route. The estimate was wrong
in sign and is not retained.

## End-to-end

Same-session A/B at p4096 across all four mtp-bench categories, 3 repetitions per
case, legacy arm selected by `HIPENGINE_QWEN4_EXP_FULLATTN_CONTEXT_LEGACY=1`:

| category | legacy tok/s | promoted tok/s | delta |
| --- | ---: | ---: | ---: |
| code | 246.540 | 252.835 | +2.55% |
| general_en | 247.942 | 256.542 | +3.47% |
| general_ja | 250.862 | 256.457 | +2.23% |
| mixed_ja_en | 248.605 | 254.432 | +2.34% |
| **mean** | **248.487** | **255.066** | **+2.65%** |

Every category improves. The prior unit measured +2.11% on the same protocol, so
this lever contributes about half a point of the total. The `code` case wall
median moves 16.6015 s -> 16.2027 s.

## Correctness gate

`tests/test_gpu_fullattn_context_promoted_route.py` (4 passed). The promoted route
is byte-identical to the shipped route over the full 1024x24x256 production
output (`maxdiff = 0.000e+00`), because neither lever changes a floating-point
operation: the four-token unroll keeps each token's dimension order and reduction
tree, and the per-warp max fold stays order-independent.

## Reproduce

```bash
# microbench over the captured launch
scripts/qwen4exp_fullattn_prefill_microbench.py --replay /tmp/fullattn-launch.npz \
  --repetitions 15 --capacities auto

# same-session end-to-end A/B, one category
HIPENGINE_QWEN4_EXP_FULLATTN_CONTEXT_LEGACY=1 \
  scripts/qwen4exp_profile_gap.py --model-root <gguf> --mode prefill \
  --case-id code-p4096 --repetitions 3 --prefill-chunk-size 1024
```

## Files

- `promoted-route.json` - this artifact.
- `hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.hip` - the kernel,
  including the comment at the precompute loop recording why the offset table is
  per-token.
- `docs/REFACTOR.md` - the rejected-lever record and the measurement-only unroll
  selector's removal trigger.
