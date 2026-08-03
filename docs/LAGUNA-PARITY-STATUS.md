# Laguna W7900 Prefill Parity Status

- **Status date:** 2026-08-03
- **Implementation state:** **TABLED**
- **Production path:** H8B scoped activation-pack reuse, **440.893 tok/s**
- **Matched llama.cpp HIP comparator:** **690.791 tok/s**
- **Current gap:** llama.cpp HIP is **1.566801×** faster

This is the canonical pause and resumption handoff for the Laguna S 2.1
UD-Q2_K_XL prefill campaign on W7900/gfx1100. Future implementation is paused.
Do not select an H8R micro-target or salvage H8C-H8Q. Any later resumption must
start from the highest-impact component gaps and satisfy the admission rules
below before source work.

## Apples-to-apples protocol

The comparison uses:

- AMD Radeon Pro W7900, gfx1100, one device and one HIP queue;
- `/models/gguf/Laguna-S-2.1-UD-Q2_K_XL.gguf`, SHA-256
  `8fe1170f012723f6f7d6c9b08d8f928b0b3d8bffc32926f33a930148a1d62679`;
- the same fixed natural 512-token stream, token-stream SHA-256
  `512cbfe87d3e7cef4af9c935b907a5d24e3a73d4234f1d0d7fcc902f2562527a`;
- context admission C4096, direct matrix M512, attention M128, FlashAttention,
  and BF16 K/V;
- cached compiler-free measured requests and one last-row first-token output;
- next token 2930 for both engines.

The user-reported llama.cpp **714.07 tok/s** is the random/synthetic default
`pp512` result. It is useful context but is **not** substituted for the matched
natural-token comparator. The apples-to-apples comparator is **690.791 tok/s**.
See the [H8B production packet](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-scoped-activation-pack-reuse-production.json)
and the [matched llama.cpp attribution packet](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-llamacpp-prefill-matched-attribution.json)
for commands, revisions, samples, traces, and source hashes.

## Final status table

All component values are representative single-request kernel sums in
milliseconds under the matched natural C4096/M512 protocol. `Remaining gap` is
`current H8B - llama.cpp HIP`; a negative value means hipEngine is ahead for
that bucket.

| Component | Campaign start | Current H8B | llama.cpp HIP | Remaining gap |
| --- | ---: | ---: | ---: | ---: |
| Q5 projections | 1,270.458 ms | 230.429 ms | 58.314 ms | **172.115 ms** |
| IQ3/IQ4 expert-down | 557.091 ms | 273.163 ms | 153.860 ms | **119.303 ms** |
| Attention | 488.304 ms | 115.349 ms | 21.512 ms | **93.837 ms** |
| Q6 projections | 157.073 ms | 73.320 ms | 14.668 ms | **58.652 ms** |
| Gate/up | 460.143 ms | 401.403 ms | 397.805 ms | **3.598 ms** |
| Remaining | 68.623 ms | 52.755 ms | 67.849 ms | **−15.094 ms** |
| **Kernel sum** | **3,001.692 ms** | **1,146.420 ms** | **714.008 ms** | **432.411 ms** |
| **Wall throughput** | **169.516 tok/s** | **440.893 tok/s** | **690.791 tok/s** | **1.566801× behind** |

Measured campaign totals:

- wall throughput improved **169.516→440.893 tok/s (+160.089%)**;
- representative kernel sum fell **3,001.692→1,146.420 ms (−61.808%)**;
- the remaining measured kernel gap is **432.411 ms**;
- H8B uses **2,155 application dispatches**, already fewer than llama.cpp's
  **2,824** in the matched trace, so launch count is not the primary blocker.

## Retained gain timeline

| Checkpoint | Throughput | Gain from prior checkpoint |
| --- | ---: | ---: |
| Campaign start | 169.516 tok/s | — |
| H6Q | 390.947 tok/s | +221.431 tok/s (**+130.625%**) |
| H6Z | 423.233 tok/s | +32.286 tok/s (**+8.258%**) |
| H7U | 437.189 tok/s | +13.956 tok/s (**+3.298%**) |
| H8A | 440.353 tok/s | +3.164 tok/s (**+0.724%**) |
| H8B | 440.893 tok/s | +0.539 tok/s (**+0.122%**) |
| H8C-H8Q | 440.893 tok/s | **0 retained gain** |

The campaign's large early gains are real. The recent trend is equally clear:
post-H8B work produced no production throughput improvement.

## What the post-H8B work established

Fifteen consecutive hypotheses were screened without a retained production
win. The gates prevented incorrect, benchmark-specific, or physically invalid
changes from landing, but this ladder reached severe diminishing returns.

| Targets | Area | Binding result |
| --- | --- | --- |
| H8C | Shared-expert Q5 dual consumer | complete both-clock timing rejection |
| H8D | Q6 F32 SGEMM complete class | rejected before production qualification |
| H8E | Alternative packed-F32 attention algorithms | complete quality rejection |
| H8F | Resident shared-Q5 F32 cache | 1K/4K transfer rejection |
| H8G | Existing global qrow6 transfer | direct transfer rejection |
| H8H | Attention+softplus dual publication | first-object runtime-resource rejection |
| H8I | Stream-ordered Q5 partitions | complete both-clock timing rejection |
| H8J-H8M | IQ3 occupancy, rowbatch, and codebook variants | physical or all-layer timing rejection |
| H8N | Q5 twin-team weight staging | 0/6 roles win both clocks; weighted wall **+63.059%** slower |
| H8O | Low-priority MoE branch overlap | **−0.4765%, 0/7 wins** |
| H8P | Q5 signed-int16 power-of-two plane | analytically impossible on actual weights |
| H8Q | Q6 int16-product/F32-scale plane | 15/15 correctness pass; all consumer VGPR ceilings fail |

H8Q is the final closed target. Its transient implementation was exact, but
consumer metadata VGPR **169/136/169** exceeded frozen ceilings
**160/128/160**. It has no timing result. Candidate source and RED were removed;
production remains H8B. See the [H8Q rejection packet](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-q6-int16-product-plane-physical-rejected.json).

## Remaining leverage, ranked

| Priority | Component | Current / llama.cpp | Gap | Share of net kernel gap |
| ---: | --- | ---: | ---: | ---: |
| 1 | Q5 projections | 3.952× | **172.115 ms** | **39.804%** |
| 2 | IQ3/IQ4 expert-down | 1.775× | **119.303 ms** | **27.590%** |
| 3 | Attention | 5.362× | **93.837 ms** | **21.701%** |
| 4 | Q6 projections | 4.999× | **58.652 ms** | **13.564%** |
| 5 | Gate/up | 1.009× | 3.598 ms | 0.832% |
| — | Remaining | 0.778× | −15.094 ms | −3.491% |

The top three components account for **89.095%** of the net kernel gap.
Gate/up and the remaining bucket are closed for parity purposes and must not be
used as restart targets.

## Mandatory resumption policy

A later campaign must start from the H8B production source and fresh matched
traces. It must not continue the H8 letter ladder by default.

1. **Reproduce before changing source.** Re-run the H8B and pinned llama.cpp HIP
   commands with the same model, natural token fixture, C4096/M512 shape, BF16
   K/V, one queue, and cached builds. If the component table moves materially,
   replace this report before selecting work.
2. **Select an algorithm/dataflow transfer, not a local tweak.** Audit the actual
   llama.cpp kernel family and trace beside hipEngine, including raw bytes,
   activation format, tile ownership, reuse, wave/workgroup geometry,
   occupancy, and complete call coverage. Source resemblance without matched
   execution evidence is insufficient.
3. **Require large modeled leverage before implementation.** The default target
   admission floor is a plausible **≥50 ms request-level kernel saving or ≥5%
   end-to-end throughput** under the matched workload. A resumed phase should
   aim to remove at least **25% of the 432.411-ms total kernel gap
   (108.103 ms)**. Anything smaller requires explicit user approval.
4. **Attack in gap order.** Start with Q5, then IQ3/IQ4 expert-down, then
   attention; treat Q6 as fourth priority. Do not spend implementation time on
   gate/up, dispatch-count reduction, or the already-ahead remaining bucket.
5. **Use complete-family gates immediately.** A Q5/Q6 target covers every owned
   production role; an IQ target covers all actual selected-expert layers; an
   attention target covers global and SWA shapes required by the proposed
   ownership. No favorable role/layer/prompt/length subset can justify a keep.
6. **Preserve correctness and anti-gaming rules.** New kernels still require KL
   ≤0.05 and top-1 ≥90% versus CPU reference/fixture gates, plus complete
   lifecycle/state checks. Never tune token IDs, routes, or the fixed prompt.
7. **Stop adjacent salvage ladders.** If a high-level candidate fails its frozen
   correctness, physical, or complete timing gate, publish the rejection and
   reprofile. Do not follow it with a sequence of geometry/codebook/resource
   micro-variants unless new trace evidence changes the dominant premise.

### High-leverage source-transfer questions

These are questions to answer from source and traces before coding, not approved
implementations:

- **Q5:** Can hipEngine adopt a complete raw-quantized, activation-quantized,
  tensorized MMQ dataflow comparable to llama.cpp's Q5_K/Q8_1 path while passing
  the full quality lane, instead of further tuning expanded-F32 consumers?
- **IQ3/IQ4 down:** Can the all-layer selected-expert path move to a complete
  source-MMQ/tensorized ownership model, rather than another codebook,
  occupancy, or output-partition micro-variant?
- **Attention:** Can full-M512 query tiling, head grouping, and stream-K/fixup
  ownership be transferred behind `KVLiveSpans` with the complete quality gate,
  instead of another qrow, prefetch, or publication tweak?
- **Q6:** After the top three, can a complete dequantize-plus-BLAS or tensorized
  raw-Q6 route reproduce llama.cpp's family-level advantage without reopening
  H8Q's compressed transient plane?

## Clean handoff boundary

- H8B measured source checkpoint: `6b9411b1527db9dad7b750da99be9a57a8e6b125`
- H8B production publication: `172c38103`
- H8B candidate-seam cleanup: `f0069b89a`
- H8Q final rejection/removal: `f219af660`
- H8Q executable/package surfaces: **absent**
- H8Q RED: **absent**
- Next Laguna implementation target: **none selected**

The detailed experiment ledger remains in [LAGUNA-prefill.md](LAGUNA-prefill.md).
The canonical benchmark rollup remains in [benchmarks/README.md](../benchmarks/README.md).
