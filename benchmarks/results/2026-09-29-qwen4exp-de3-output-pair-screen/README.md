# DE-3 screening: output-pairing is already applied for q8_0 decode, and exhausted

DE-3's next step is "screen row-pairing on the largest exact GEMV (attn_qkv)",
from halo-box PR #56's "two output rows per block in MMVQ, bit-exact, +12% TG on
the 27B".

**Screening result: the q8_0 decode path already computes 8 output features per
block - four times the proposed two - and a sweep of alternative output-tile widths
at equal work shows the current tiling is the best of them. Every alternative is
slower.**

## The lever is already applied

`launch_gguf_k_pack8_gemv_out` (the kernel the decode census shows at rows=1):

```cpp
const int64_t out_packed = out_features >> 3;   // 8 output features per block
const dim3 grid(static_cast<unsigned int>(out_packed), static_cast<unsigned int>(rows));
```

So `attn_qkv` decode (10240 output features) already runs 1280 blocks of 8 packed
output features each. DE-3's proposal is two per block; the path is at eight.

The `pair` kernels that implement DE-3's idea in this tree
(`gguf_q5_k_pair_pack8_decode_out_kernel`, `gguf_q6_k_pair_pack8_decode_out_kernel`)
exist for **q5_k and q6_k only** - there is no q8_0 pair kernel, because q8_0
already packs wider than a pair.

## And widening further does not help

Sweep at `attn_qkv` (2560 -> 10240), rows=1, median of 40 reps after 8 warmups,
device-synchronized both sides. **All variants compute the same 10240 output
columns**, so the comparison is at equal work:

| variant | out cols/block | us | vs baseline |
|---|---:|---:|---:|
| **`pack8` (current decode path)** | **8** | **63.70** | **1.00x** |
| `coltile4_rowbatch8` | 4 | 110.52 | 0.58x |
| `coltile8_rowbatch4` | 8 | 80.47 | 0.79x |
| `coltile8_rowbatch8` | 8 | 110.23 | 0.58x |
| `coltile16_rowbatch2` | 16 | 77.01 | 0.83x |
| `coltile16_rowbatch4` | 16 | 211.93 | 0.30x |
| `coltile32_rowbatch1` | 32 | 133.69 | 0.48x |

There is no monotonic benefit in the output-tile width: narrower (4) is worse,
wider (16, 32) is worse, and the best alternative is 0.83x - still 17% slower than
the kernel already in use.

This is consistent with the independent MT-1 measurement, where the same
`coltile8_rowbatch4` kernel at rows=1 was **0.80x** the decode kernel.

## What this screening does not settle

- **The alternatives carry rowbatch machinery that is dead weight at rows=1.**
  They are prefill-shaped kernels, so this sweep **bounds** the output-tile axis
  rather than isolating it. A purpose-built output-pair kernel with no row
  dimension is not what was measured here, and the q5_k/q6_k `pair` kernels are the
  closer structural analogue.
- **Kernel-level only.** No end-to-end TG measurement, no attention or MoE, and
  the 24.73 ms/token family figure was not re-attributed.
- **Not a bit-exactness result.** This is a timing screen; the coltile variants'
  bit-exactness is covered separately by
  `test_q8_0_f32_rowbatch32_matches_scalar_prefill_bits`.
- One host, one quant, one shape, one width.

## A corrected test, recorded

The first version of this sweep compared the coltile variants at
`out_features=32` against the baseline at `out_features=10240` - different total
work - and produced meaningless ratios (0.00x). It was discarded and re-run at
equal work. The table above is the corrected measurement.

## Limits

- `attn_qkv` is DE-3's largest owner, but the row covers the whole linear family.
- The +12% on the 27B is an external figure for a different engine, quant, and
  model, and **no rate was transferred** from it.
