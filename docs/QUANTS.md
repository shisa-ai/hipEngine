# GGUF Quantization Portfolio — Qwen3.5 Evidence and Laguna S 2.1 Plan

_Status: planning reference for the Laguna S 2.1 port. Laguna file inventories are
header measurements; quality expectations are inferred from Qwen3.5 and must be
validated on Laguna. Last updated: 2026-07-22._

This document answers four questions:

1. Which GGML weight types hipEngine can execute natively today?
2. Where did Unsloth's Qwen3.5 Q2/Q3/Q4 quality cliffs occur?
3. Which Laguna S 2.1 GGUFs are the best targets for gfx1151/120 GiB, one
   W7900/48 GiB, and a W7900 + RX 7900 XTX/72 GiB system?
4. How much BF16 K/V context fits in the memory left after weights?

It complements [`GGUF.md`](GGUF.md), [`KVCACHE.md`](KVCACHE.md), and
[`TENSOR_PARALLEL.md`](TENSOR_PARALLEL.md). It is not a performance or Laguna
quality claim.

## Evidence boundaries

Keep these evidence classes separate:

- **Measured Laguna inventory:** GGUF headers at
  [`unsloth/Laguna-S-2.1-GGUF@99d7f9a`](https://huggingface.co/unsloth/Laguna-S-2.1-GGUF/tree/99d7f9a1251bd4d925cac85cf64ffba7189338c2).
  All shards were scanned; each recipe declares 814 tensors and
  117,561,977,600 parameters.
- **Measured Qwen quality:** Unsloth's published
  [Qwen3.5 GGUF analysis](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks),
  with tensor inventories checked against
  [`unsloth/Qwen3.5-35B-A3B-GGUF@bc014a1`](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/tree/bc014a17be43adabd7066b7a86075ff935c6a4e2).
- **Calculated capacity:** exact weight-file sizes and ideal BF16 K/V payload
  math. Runtime scratch, replacement-layout growth, page rounding, allocator
  fragmentation, graph buffers, and server overhead are not included unless a
  row explicitly applies a planning reserve.
- **Inferred Laguna quality:** Qwen's ordering is a porting prior only. Laguna
  needs its own held-out coding/tool-use suite, KL/top-1 gate, and long-context
  gate before any recipe is promoted.

Unsloth also cautions that perplexity and KLD depend on the calibration corpus
and do not always predict real-world evaluation ordering. Use them to identify
cliffs and sensitive tensor families, not as substitutes for Laguna evaluation.

## Decisions at a glance

- **gfx1151 / nominal 120 GiB:** implement per-tensor dispatch rather than a
  Q4-only model path. Prefer `UD-Q4_K_XL` over `UD-Q4_K_M`; target
  `UD-Q5_K_XL` as the quality/performance sweet spot; use `UD-Q6_K_XL` as the
  highest-quality practical resident model when its bandwidth and context cost
  are acceptable.
- **Single W7900 / 48 GiB:** `UD-Q2_K_XL` is the preferred compact target.
  `UD-IQ3_XXS` is a possible higher-quality short-/medium-context target but
  needs two new native formats and has much less runtime headroom.
- **W7900 + 7900 XTX / 72 GiB:** `UD-Q3_K_XL` is the preferred quality target.
  It uses tensor types already present in hipEngine. It requires real weight
  partitioning; moving only K/V to the second card cannot make a 50.4 GiB
  weight set fit on the W7900.
- **Do not plan on Q4 at 72 GiB:** `UD-Q4_K_XL` leaves only 3.6 GiB aggregate
  before per-device scratch and K/V.
- **Do not plan on Q8 at 120 GiB:** `UD-Q8_K_XL` is 119.3 GiB before any runtime
  allocation.
- **Best new low-bit kernel return:** implement `IQ2_XS` first to unlock
  `UD-Q2_K_XL`. Implementing `IQ2_XXS` + `IQ2_S` first buys a smaller but much
  lower-quality model.

## Current hipEngine GGML type coverage

The distinction is native compressed execution, not merely parsing the type id.

This table covers the whole engine, so a type listed as native may still be
unavailable on a narrower route. What each type does on the dense model route
specifically — read as quantized, converted to dense BF16, or refused at load — is
measured per file in [`UD-QUANTS.md`](UD-QUANTS.md).

| GGML type | Current status | Relevant execution role |
| --- | --- | --- |
| `F32`, `BF16` | Native | Norm/router/dense weights and unquantized fallback |
| `Q8_0` | Native | Dense, shared-expert, embedding/head, and selected-MoE routes |
| `Q4_K`, `Q5_K`, `Q6_K` | Native | Dense and selected-MoE routes; raw and replacement layouts |
| `IQ3_XXS`, `IQ4_XS` | Native, selected-MoE focused | Routed expert gate/up/down; Laguna uses these only in rank-3 expert tensors |
| `Q3_K` | Native selected-MoE | Implemented for Qwen NextN, but absent from the Laguna S 2.1 files |
| `IQ4_NL`, `MXFP4` | CPU dequant only | No retained native compressed execution kernel |
| `IQ1_S`, `IQ1_M` | Layout only | CPU dequant and native execution missing |
| `IQ2_XS` | Native selected decode/prefill + CPU dequant | Laguna K=3072,N=1024 gate/up primitives are exact; full model validation remains open |
| `IQ2_XXS`, `IQ2_S` | Layout only | CPU dequant and native execution missing |
| `IQ3_S` | Layout only | CPU dequant and native execution missing |

The `IQ4_NL` values embedded in the `IQ4_XS` implementation are its codebook;
they are not a standalone `IQ4_NL` kernel.

## Qwen3.5 quality reference

### Published quant results

Lower is better. The reported-GB column is copied from Unsloth's analysis and
is used only for relative quality/size comparisons, not Laguna capacity.

| Quant | Reported GB | PPL | KLD 99.9% | Mean KLD |
| --- | ---: | ---: | ---: | ---: |
| `IQ2_XXS` | 9.09 | 7.7160 | 4.2221 | 0.1846 |
| `Q2_K_XL` | 12.04 | 7.0438 | 2.9092 | 0.0970 |
| `IQ3_XXS` | 13.12 | 6.7829 | 1.5296 | 0.0501 |
| `IQ3_S` | 14.13 | 6.7715 | 1.4193 | 0.0457 |
| `Q3_K_M` | 15.54 | 6.7320 | 0.9726 | 0.0324 |
| `Q3_K_XL` | 16.06 | 6.7245 | 0.9539 | 0.0308 |
| `MXFP4_MOE` | 18.17 | 6.6000 | 0.7789 | 0.0272 |
| `Q4_K_M` | 18.49 | 6.6053 | 0.5478 | 0.0192 |
| `Q4_K_L` | 18.82 | 6.5905 | 0.4828 | 0.0150 |
| `Q4_K_XL` | 19.17 | 6.5918 | 0.4097 | 0.0137 |
| `Q5_K_XL` | 23.22 | 6.5489 | 0.2360 | 0.0069 |
| `Q6_K_S` | 26.56 | 6.5456 | 0.2226 | 0.0065 |
| `Q6_K_XL` | 28.22 | 6.5392 | 0.1437 | 0.0041 |
| `Q8_K_XL` | 36.04 | 6.5352 | 0.1033 | 0.0026 |

Relative to `Q4_K_M`, Qwen's compact tiers had:

| Quant | PPL delta | KLD 99.9% ratio | Mean-KLD ratio |
| --- | ---: | ---: | ---: |
| `IQ2_XXS` | +16.82% | 7.71x | 9.61x |
| `Q2_K_XL` | +6.64% | 5.31x | 5.05x |
| `IQ3_XXS` | +2.69% | 2.79x | 2.61x |
| `IQ3_S` | +2.52% | 2.59x | 2.38x |
| `Q3_K_M` | +1.92% | 1.78x | 1.69x |
| `Q3_K_XL` | +1.80% | 1.74x | 1.60x |

### Where the cliffs occurred

Current Qwen3.5 headers make the recipe names less misleading:

| Qwen recipe | Routed expert gate/up | Routed expert down | Consequence |
| --- | --- | --- | --- |
| `UD-IQ2_XXS` | `IQ2_XXS` | `IQ2_S` | Most aggressive compression and largest measured quality loss |
| `UD-IQ2_M` | `IQ2_XXS` | `IQ3_XXS` | Protects the more-sensitive down projection |
| `UD-Q2_K_XL` | mostly `IQ2_XS` | mostly `IQ3_XXS` | Better Q2 gate/up plus protected down |
| `UD-IQ3_XXS` | `IQ2_S` | `IQ3_XXS` | Still predominantly 2-bit despite the filename |
| `UD-IQ3_S` | `IQ2_S` | `IQ3_S` | Still predominantly 2-bit gate/up |
| `Q3_K_M` | `IQ3_XXS` | `IQ4_XS` | First listed tier with genuinely 3-bit gate/up and 4-bit down |
| `UD-Q3_K_XL` | same basic pattern plus selective promotions | same | Small M-to-XL gain |

The largest quality improvements were therefore not generic filename changes:

| Step | Reported size delta | PPL change | KLD 99.9% change | Mean-KLD change |
| --- | ---: | ---: | ---: | ---: |
| `IQ2_XXS -> Q2_K_XL` | +32.5% | -8.71% | -31.1% | -47.5% |
| `Q2_K_XL -> IQ3_XXS` | +9.0% | -3.70% | -47.4% | -48.4% |
| `IQ3_XXS -> Q3_K_M` | +18.4% | -0.75% | -36.4% | -35.3% |
| `Q3_K_M -> Q3_K_XL` | +3.3% | -0.11% | -1.9% | -4.9% |
| `Q4_K_M -> Q4_K_XL` | +3.7% | -0.20% | -25.2% | -28.6% |
| `Q4_K_XL -> Q5_K_XL` | +21.1% | -0.65% | -42.4% | -49.6% |
| `Q5_K_XL -> Q6_K_XL` | +21.5% | -0.15% | -39.1% | -40.6% |

Unsloth's broader tensor sweep also identified attention projections, hybrid
`ssm_out`, and `ffn_down_exps` as sensitive. Gate/up experts tolerated 3-bit
better than 2-bit. This supports three planning rules:

1. Treat routed gate/up at 2-bit as the primary Q2 quality cliff.
2. Protect expert down at least one tier above gate/up.
3. Expect Q4+ upgrades to show more in tail/logit fidelity than in perplexity.

### MXFP4 versus Q4_K

MXFP4 has a theoretical storage advantage (`4.25` bpw versus `4.50` bpw for
Q4_K), and may have a speed advantage on hardware with native FP4 execution.
It did not establish a quality advantage in Unsloth's Qwen study:
`MXFP4_MOE` had nearly identical PPL but approximately 42% worse mean and
99.9%-tail KLD than `Q4_K_M`. Unsloth explicitly reported MXFP4 as unusually
poor on several sensitive tensor families and retired it from mixed Q2/Q3/Q4
recipes.

For Laguna, `MXFP4_MOE` is only 1.9 GiB (2.8%) smaller than `UD-Q4_K_M` and
would require a new native kernel on RDNA3. It is a low-priority target unless a
native-FP4 backend demonstrates a measured throughput advantage.

## Laguna S 2.1 GGUF inventory

Sizes are the sum of all shards in GiB. Effective bpw includes the complete
mixed recipe and small file overhead:

```text
effective bpw = total GGUF file bytes * 8 / 117,561,977,600 parameters
```

### Type-complete with current native formats

"Type-complete" means no new GGML codec/kernel family is required. Laguna model
mapping, shape coverage, correctness, and performance still require validation.

| Laguna quant | GiB | Effective bpw | Actual GGML types |
| --- | ---: | ---: | --- |
| `UD-Q3_K_M` | 50.3 | 3.676 | `IQ3_XXS`, `IQ4_XS`, `Q8_0`, `Q6_K`, `F32` |
| `UD-Q3_K_XL` | 50.4 | 3.681 | `IQ3_XXS`, `IQ4_XS`, `Q8_0`, `Q6_K`, `F32` |
| `UD-Q4_K_S` | 63.9 | 4.667 | `Q4_K`, `Q8_0`, `Q6_K`, `F32` |
| `UD-Q4_K_M` | 68.1 | 4.976 | `Q4_K`, `Q5_K`, `Q8_0`, `Q6_K`, `F32` |
| `UD-Q4_K_XL` | 68.4 | 4.994 | `Q4_K`, `Q5_K`, `Q8_0`, `Q6_K`, `F32` |
| `UD-Q5_K_S` | 77.0 | 5.625 | `Q5_K`, `Q8_0`, `Q6_K`, `F32` |
| `UD-Q5_K_M` | 81.8 | 5.979 | `Q5_K`, `Q6_K`, `Q8_0`, `F32` |
| `UD-Q5_K_XL` | 82.0 | 5.993 | `Q5_K`, `Q6_K`, `Q8_0`, `F32` |
| `UD-Q6_K` | 91.2 | 6.663 | `Q6_K`, `Q8_0`, `F32` |
| `UD-Q6_K_XL` | 99.7 | 7.287 | `Q6_K`, `Q8_0`, `F32` |
| `Q8_0` | 116.4 | 8.508 | `Q8_0`, `F32` |
| `UD-Q8_K_XL` | 119.3 | 8.718 | `Q8_0`, `BF16`, `F32`; no `Q8_K` tensors |
| `BF16` | 219.0 | 16.005 | `BF16`, `F32` |

For Laguna specifically, `Q4_K_M -> Q4_K_XL` costs only 0.3 GiB. XL promotes
about 1.61B expert parameters from Q4_K to Q5_K and one approximately 0.31B
parameter tensor from Q6_K to Q8_0. It uses no new format, so XL should be the
default Q4 quality target once the M path is correct.

### Requires additional native tensor types

Missing share is percentage of Laguna parameters stored in formats without a
native compressed execution path.

| Laguna quant | GiB | Effective bpw | New native types required | Missing share |
| --- | ---: | ---: | --- | ---: |
| `UD-IQ1_S` | 31.4 | 2.298 | `IQ1_S`, `IQ2_XXS` | 64.4% |
| `UD-IQ1_M` | 33.2 | 2.425 | `IQ1_M`, `IQ2_XXS` | 64.4% |
| `UD-IQ2_XXS` | 34.6 | 2.531 | `IQ2_XXS`, `IQ2_S` | 64.4% |
| `UD-IQ2_M` | 34.7 | 2.536 | `IQ2_XXS`, `IQ2_S` | 64.4% |
| `UD-Q2_K_XL` | 37.0 | 2.701 | `IQ2_XS`; no actual `Q2_K` tensors | 63.0% |
| `UD-IQ3_XXS` | 41.2 | 3.013 | `IQ2_S`, `IQ3_S` | 95.2% |
| `UD-IQ3_S` | 45.1 | 3.296 | `IQ2_S`, `IQ3_S` | 64.4% |
| `UD-IQ4_XS` | 53.6 | 3.917 | `IQ3_S` | 63.0% |
| `UD-IQ4_NL` | 54.7 | 3.998 | `IQ3_S`, native `IQ4_NL` | 95.2% |
| `MXFP4_MOE` | 66.2 | 4.837 | native `MXFP4` | 63.0% |

Dynamic recipe names are not tensor inventories. Notably, Laguna
`UD-Q2_K_XL` has no Q2_K, `UD-Q3_K_*` has no Q3_K, and `UD-Q8_K_XL` has no
Q8_K.

## Hardware sweet spots

### gfx1151 / nominal 120 GiB

| Target | Weight GiB | Nominal headroom | Role | Planning verdict |
| --- | ---: | ---: | --- | --- |
| `UD-Q4_K_XL` | 68.4 | 51.6 | Throughput/headroom | Preferred Q4; virtually free upgrade over M |
| `UD-Q5_K_XL` | 82.0 | 38.0 | Quality/performance balance | **Primary sweet spot** |
| `UD-Q6_K` | 91.2 | 28.8 | Higher quality, more context than Q6 XL | Useful intermediate |
| `UD-Q6_K_XL` | 99.7 | 20.3 | Highest practical quality | Fits nominally; gate runtime peak and target context |
| `UD-Q8_K_XL` | 119.3 | 0.7 | Near-Q8 quality | Does not leave a viable runtime envelope |

If "120 GB" is decimal rather than 120 GiB, available binary capacity is only
about 111.8 GiB. Recompute the table from the allocator-reported usable domain
before selecting Q6 XL.

Higher-bit expert weights increase resident size and the bytes read for each
active expert. Qwen's Q4 XL -> Q5 XL and Q5 XL -> Q6 XL improvements were large
in KL but small in PPL; benchmark Laguna wall time and agentic quality rather
than assuming the larger file is automatically the best product choice.

### Single W7900 / 48 GiB

| Target | Weight GiB | Nominal headroom | Planning verdict |
| --- | ---: | ---: | --- |
| `UD-IQ2_XXS` / `UD-IQ2_M` | 34.6 / 34.7 | 13.4 / 13.3 | Fits, but inferior quality/implementation return |
| `UD-Q2_K_XL` | 37.0 | 11.0 | **Preferred single-card target**; implement `IQ2_XS` |
| `UD-IQ3_XXS` | 41.2 | 6.8 | Possible higher-quality target; runtime/context constrained |
| `UD-IQ3_S` | 45.1 | 2.9 | Too tight without measured very-low scratch overhead |
| `UD-Q3_K_M/XL` | 50.3 / 50.4 | Does not fit | Requires weight partitioning/offload |

### W7900 + RX 7900 XTX / nominal 72 GiB

Multi-GPU support is currently a plan, not a retained hipEngine product path.
The first TP plan in [`TENSOR_PARALLEL.md`](TENSOR_PARALLEL.md) replicates K/V
on each rank, so aggregate free memory cannot be treated as one K/V pool.
Contiguous layer/pipeline sharding instead naturally partitions K/V by layer.

| Hypothetical layer-sharded target | Weight GiB | Aggregate headroom | Planning verdict |
| --- | ---: | ---: | --- |
| `UD-Q3_K_M` | 50.3 | 21.7 | Good |
| `UD-Q3_K_XL` | 50.4 | 21.6 | **Preferred two-card target** |
| `UD-Q4_K_S` | 63.9 | 8.1 | Very tight after per-device runtime allocations |
| `UD-Q4_K_XL` | 68.4 | 3.6 | Not viable for a normal runtime |

For an asymmetric 48+24 GiB pair, assign weights with per-device scratch and
K/V headroom in mind, not merely a 2:1 byte split. Co-locate each layer's K/V
with that layer's weights. A design that leaves weights on the W7900 and stores
all K/V remotely on the XTX adds PCIe traffic to every attention step and does
not solve Q3 XL's greater-than-48-GiB weight footprint.

## Laguna BF16 K/V capacity

### Model facts and formula

The pinned Laguna configuration defines:

- 48 attention layers: 12 global and 36 sliding-window;
- sliding window 512;
- 8 K/V heads;
- head dimension 128;
- BF16 K and V, 2 bytes each;
- maximum context 1,048,576 tokens.

One layer stores:

```text
8 KV heads * 128 values * 2 (K,V) * 2 bytes = 4,096 bytes/token
```

Therefore one sequence requires:

```text
C <= 512:
  KV(C) = C * 48 layers * 4,096 bytes
        = C * 192 KiB

C > 512, with physical sliding-window eviction:
  KV(C) = (12 * C + 36 * 512) * 4,096 bytes
        = C * 48 KiB + 72 MiB
```

Without physical sliding-window eviction, all 48 layers grow and the first
formula remains active: 1M would require 192 GiB instead of about 48.1 GiB.
Laguna admission must therefore prove that old sliding-layer K/V is actually
reclaimed, not merely masked from attention.

### K/V footprint by context

| Context tokens | BF16 K/V per sequence |
| ---: | ---: |
| 512 | 96 MiB |
| 4,096 | 264 MiB |
| 16,384 | 840 MiB (0.820 GiB) |
| 32,768 | 1.570 GiB |
| 65,536 | 3.070 GiB |
| 131,072 | 6.070 GiB |
| 262,144 | 12.070 GiB |
| 524,288 | 24.070 GiB |
| 1,048,576 | 48.070 GiB |

### Context capacity for a K/V budget

These are payload-only maxima for one sequence, with physical sliding-window
eviction and no page/allocator overhead.

| K/V budget | Maximum context tokens |
| ---: | ---: |
| 1 GiB | 20,309 |
| 2 GiB | 42,154 |
| 4 GiB | 85,845 |
| 6 GiB | 129,536 |
| 8 GiB | 173,226 |
| 12 GiB | 260,608 |
| 16 GiB | 347,989 |
| 20 GiB | 435,370 |
| 24 GiB | 522,752 |
| 28 GiB | 610,133 |
| 32 GiB | 697,514 |
| 38 GiB | 828,586 |
| 40 GiB | 872,277 |
| 48 GiB | 1,047,040 |
| 48.071 GiB | 1,048,576 (model maximum) |

For `N` equal-length requests above 512 tokens, divide the K/V budget per
request before inversion:

```text
context/request = floor((total_KV_bytes / N - 72 MiB) / 48 KiB)
```

For example:

| Total K/V budget | c=1 | c=2 per request | c=4 per request | c=8 per request |
| ---: | ---: | ---: | ---: | ---: |
| 8 GiB | 173,226 | 85,845 | 42,154 | 20,309 |
| 16 GiB | 347,989 | 173,226 | 85,845 | 42,154 |

Prefix sharing can reduce physical storage when requests share a prefix; the
simple table assumes no sharing.

### Weight-headroom scenarios

The 4/8 GiB reserves below are planning scenarios, not measured Laguna runtime
requirements. They represent all non-weight, non-K/V allocations. Replace them
with an allocator peak once the Laguna runner exists.

| Hardware / quant | Nominal headroom | Max context after 4 GiB reserve | After 8 GiB reserve |
| --- | ---: | ---: | ---: |
| W7900 `UD-Q2_K_XL` | 11.0 GiB | 151,381 | 64,000 |
| W7900 `UD-IQ3_XXS` | 6.8 GiB | 59,630 | none |
| W7900 `UD-IQ3_S` | 2.9 GiB | none | none |
| 72 GiB layer-sharded `UD-Q3_K_XL` | 21.6 GiB | 382,941 | 295,560 |
| 72 GiB layer-sharded `UD-Q4_K_XL` | 3.6 GiB | none | none |
| 120 GiB `UD-Q4_K_XL` | 51.6 GiB | 1,038,301 | 950,920 |
| 120 GiB `UD-Q5_K_XL` | 38.0 GiB | 741,205 | 653,824 |
| 120 GiB `UD-Q6_K` | 28.8 GiB | 540,228 | 452,846 |
| 120 GiB `UD-Q6_K_XL` | 20.3 GiB | 354,542 | 267,161 |
| 120 GiB `UD-Q8_K_XL` | 0.7 GiB | none | none |

Capacity is not throughput. At long context, global-attention K/V reads grow by
48 KiB per historical token per decode token across the 12 global layers.
Attention bandwidth can dominate even when allocation succeeds.

## Implementation order

### P0 — Finish the type-complete quality ladder

1. Finish Laguna `UD-Q4_K_M` on gfx1151 with per-tensor GGML dispatch.
2. Admit `UD-Q4_K_XL`; it uses the same formats and costs only 0.3 GiB more.
3. Validate `UD-Q5_K_XL` and `UD-Q6_K_XL` through existing Q5_K/Q6_K/Q8_0
   dense and selected-MoE families.
4. Validate `UD-Q3_K_XL` through existing IQ3_XXS/IQ4_XS selected-MoE kernels
   and use it as the two-card target.

Do not register a model-wide recipe by assuming every tensor has that recipe's
name. Dispatch from each tensor's actual GGML type and role.

### P1 — Add the best single-W7900 low-bit format

Implement `IQ2_XS` first. It unlocks `UD-Q2_K_XL`, whose remaining types are
already covered, and Qwen evidence gives it a much better quality prior than
`IQ2_XXS`.

Required surface:

- **Done:** CPU dequant/oracle, with a pinned independent llama.cpp fixture;
- **Done:** selected single and dual-SiLU decode for rank-3 routed gate/up;
- **Done:** exact grouped/rowbatch compact prefill plus a correctness-gated WMMA
  diagnostic at Laguna K=3072;
- **Not currently required:** weighted selected-down IQ2_XS; the pinned
  `UD-Q2_K_XL` inventory places IQ2_XS in routed gate/up, not down;
- **Partial:** synthetic tensor-role materialization tests pass, but the exact
  Laguna model mapping and model-level KL/top-1 gate remain open;
- **Partial:** rocprof symbol/resource proof is complete; model wall/memory and
  long-context evidence remain open.

These completed primitive bullets do **not** make the Laguna recipe supported;
the acceptance gates below still require the model-level work.

### P2 — Optional single-card Q3/IQ4 ladder

Implement `IQ2_S` and `IQ3_S` together if `UD-IQ3_XXS`, `UD-IQ3_S`, or
`UD-IQ4_XS` is strategically important. For Laguna, `UD-IQ3_XXS` requires both
formats and places 95.2% of parameters in currently unsupported types. The
engineering scope is therefore much larger than its filename suggests.

### P3 — Low-priority formats

- `IQ2_XXS` + `IQ2_S`: unlocks the smallest Q2 models but has a weaker quality
  prior than `IQ2_XS`/Q2 XL.
- `IQ1_S`/`IQ1_M`: capacity-first only; validate real agentic utility before
  investing.
- `IQ4_NL`: CPU dequant exists, but Laguna's recipe also requires `IQ3_S`.
- `MXFP4`: pursue only with a hardware/backend throughput case.

## Acceptance gates

A Laguna recipe is supported only after all of the following:

1. Header inventory is pinned to a model revision and every tensor role resolves
   through the four-axis registry.
2. Every new compressed kernel passes its CPU-reference fixture and repository
   correctness gate (KL <= 0.05 and top-1 agreement >= 90%).
3. Full-model testing covers the complete coding/general-English/general-Japanese/
   mixed-language prompt categories rather than a single prompt.
4. Long-context tests include at least one point beyond the 512-token sliding
   window and prove physical sliding-K/V reclamation.
5. Memory evidence records weight allocations, replacement-layout delta,
   persistent scratch, K/V payload and metadata, allocator peak, and usable
   device-domain headroom.
6. Multi-GPU claims identify TP versus layer/pipeline sharding, K/V replication
   versus partitioning, per-rank peaks, transfer/collective cost, and the
   correctness gate.
7. Performance rows state model revision, exact quant, hardware, context,
   concurrency, command, wall result, and correctness result.

## Reproduction notes and sources

Laguna inventory methodology:

1. List every GGUF shard through the Hugging Face tree API at revision
   `99d7f9a1251bd4d925cac85cf64ffba7189338c2`.
2. HTTP-range read each shard's GGUF header.
3. Parse every tensor descriptor, combine split shards, reject duplicate tensor
   names, and sum logical parameters by GGML type.
4. Sum API-reported shard bytes and convert decimal bytes to GiB with `2^30`.

Primary sources:

- [Laguna S 2.1 model card](https://huggingface.co/poolside/Laguna-S-2.1)
- [Pinned Laguna configuration](https://huggingface.co/poolside/Laguna-S-2.1/blob/179ee67cf0fff5391c67fe1a392ea849fa6d643f/config.json)
- [Pinned Unsloth Laguna GGUF repository](https://huggingface.co/unsloth/Laguna-S-2.1-GGUF/tree/99d7f9a1251bd4d925cac85cf64ffba7189338c2)
- [Unsloth Qwen3.5 GGUF quality analysis](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks)
- [Pinned Qwen3.5-35B-A3B GGUF repository](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/tree/bc014a17be43adabd7066b7a86075ff935c6a4e2)
- hipEngine format/oracle definitions: `hipengine/quant/gguf.py`
- hipEngine quant plugins: `hipengine/quant/gguf_k.py`,
  `hipengine/quant/gguf_q4_k.py`, `hipengine/quant/gguf_t16.py`, and
  `hipengine/quant/gguf_x8.py`
- Native selected-IQ kernels:
  `hipengine/kernels/hip_gfx1100/quant/gguf_iq_gemv.{py,hip}` and
  `gguf_iq_selected_prefill.{py,hip}`
