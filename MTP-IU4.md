# MTP IU4 verifier research plan

_Status: R0–R2/B1–B4/R6–R8 complete. Single-request B1–B3 IU4 remains
rejected on representation economics. Dense-prefill gate/up passes 2.67–3.10×;
the immutable original-product full-FFN route reaches 2.61–3.35× leaf and wins
complete prefill against both authorities, but fails both binding distribution
gates. It is retained explicit/quality-traded only; decode stays strict fallback._

_Branch: `mtp-iu4` (renamed from `mpt-iu4` 2026-08-24 — original uploader typo)_

_Last updated: 2026-08-24 (gfx1100 port screen §4.2.3)_

## Verdict

Native packed IU4 is real on this gfx1151 host. The in-tree instruction screen
measures **109.715 TOPS U4×S4 IU4 WMMA**, versus **55.015 TOPS U8×S8 IU8
WMMA**, **55.066 TFLOP/s FP16 WMMA**, and **55.381 TFLOP/s BF16 WMMA**. The
candidate lane is therefore about **1.98–2.00×** the other WMMA types, and the
compiled ISA contains `v_wmma_i32_16x16x16_iu4`.

That does **not** imply a 2× faster hipEngine B3 verifier. The first roofline
screen says the opposite: single-request B1–B4 verification is a tiny-M,
weight-streaming regime. For ideal packed W4 weights,

```text
OI_weight ≈ 2*M*K*N / (K*N/2 bytes) = 4*M ops/byte.
```

At physical `M=4`, this is only **16 ops/B**, or **3.536 TOPS** at the
established 221 GB/s practical gfx1151 read roof. The measured U4×S4 compute
roof is 109.715 TOPS. Even after charging the 16-row WMMA tile for 12 padded
rows, its useful `M=4` arithmetic roof is about **27.43 TOPS**, nearly the same
as the measured **28.25 TOPS** U8×S8 DOT4 roof used by the current verifier.
The 2× IU4 instruction density is largely consumed by tile underfill before
memory traffic, activation packing, correction, and epilogues are counted.

The current rows-3 Qwen3.8 gate/up owner reinforces this conclusion. It streams
a 100,270,080-byte actual gate/up pair in about **0.463 ms**, equivalent to
**216.6 GB/s**, already near the 221 GB/s practical read roof. A clean signed-I4
companion for that pair would be about 89,407,488 bytes including one FP32 scale
and one I32 sum per output, so its impossible-best read floor is about
**0.405 ms**. The leaf's bandwidth-only ceiling is therefore roughly **1.14×
before U4 activation packing**—not 2×.

R1 then measured the complete exact transactional target window. At physical
M=4, the 64 gate/up calls own **35.576 ms of 89.348 ms kernel time and 148.655
ms target wall** (39.82% / 23.93%). They stream 6.417 GB at an aggregate **180.4
GB/s**. Even an impossible 221-GB/s S4 family with zero activation-pack cost
would lower the target window by only **1.070×**; deleting gate/up entirely
would bound it at **1.314×**. The one-layer experiment remains worthwhile, but
its system ceiling is now measured rather than inferred.

R2 confirms the bound. The operation-complete layer-0 candidate—including U4
packing, native IU4 WMMA, I32 correction/scales, BF16 gate/up publication, and
SiLU—moves M2/M3/M4 from **0.459/0.473/0.553 ms to 0.584/0.591/0.601 ms**:
**0.787×/0.800×/0.920×**, with 0/15, 0/15, and 1/15 paired wins. It first wins
at M5 (1.195×), reaches 1.598× at M8, and reaches 3.190–4.463× at M16–128.
Activation packing is only 9.7–11.8 µs; tiny-M failure is the padded IU4 core,
not packing.

The naive research sidecar is also not quality-qualified. It re-quantizes the
authoritative Q4_K_S view rather than original BF16/F16 weights; one long-K
symmetric S4 scale per output has about **15.9% weight NRMSE**, and the actual
gate/up output is about **28.5–29.4% NRMSE** versus the current exact owner.
Intermediate-channel softmax KL/top-1 are reported only for localization and are
not the model full-logit gate.

**Decision:** reject native IU4 for single-request B1–B3 and do not allocate the
5.329-GiB gate/up companion there. Retain the rebuilt gfx1151 dense-prefill
family and the immutable 7.99-GiB Kairic FFN product as an **explicit,
quality-traded T3 configuration only**. It improves complete prefill against
both tested authorities and passes five Q4_K_S long tasks, but arbitrary-context
logits fail the frozen distribution envelope against both Q4_K_S and the matched
Kairic authority. Omitted/default profile remains unchanged and c1 decode falls
back to the authority's exact owner.

### Verdict correction (2026-08-23 review)

The rejection above stands, but R2's stated mechanism was wrong and its wide-M
rows are not usable evidence. **The R2 candidate core is implementation-limited,
not representation-limited.** See §3.1. In summary:

- The candidate core sustains **150–155 GB/s** at M2–M16 while its own control,
  on the same host, same layer pair, and same launch, sustains **218.3 GB/s**.
  It runs **30% below the 221 GB/s roof the control demonstrably reaches**.
- It executes **9.6–12.5 TOPS** against the measured 109.715-TOPS IU4 roof —
  **9–11% of peak**. It is therefore neither compute-bound nor at the memory
  roof, so it never entered the regime this document's roofline reasons about.
- Projected onto the control's own demonstrated 218.3 GB/s, the candidate would
  reach **0.419 ms inclusive**, i.e. **1.10×/1.13×/1.32× at M2/M3/M4** — wins,
  not the recorded 0.787×/0.800×/0.920× losses.

That projected win is **exactly the 1.12× byte ratio** (89,407,488 S4 bytes
versus 100,270,080 Q4_K_S bytes) and matches §3's own 1.14× bandwidth-only
ceiling. So the corrected tiny-M finding is sharper than the original:

> At tiny M the only available lever is **bytes, not ops**. The 2× arithmetic
> lane contributes nothing, because both the current owner and any fixed
> candidate are pinned at the weight-bandwidth roof.

Feeding the optimistic 1.32× through R1's measured 23.93% B3 gate/up wall share
yields **1.06× on the target window** in exchange for 5.329 GiB resident, 15.9%
weight NRMSE, U4 activations, and a full T3 quality campaign. The go/no-go
result is unchanged; only its justification is corrected.

**The wide-M rows cut the other way.** R2's M64–M128 results (3.190×–4.463×) are
**understated**, measured with the same 1-accumulator core at 11% of the IU4
roof. They are not a floor on what a properly blocked IU4 kernel can do, and
they remain the wrong control (decode rowtile8, not bulk prefill WMMA). The
prefill screen in §4.2 is consequently *under*-argued, not over-argued.

### R6/R7 result (2026-08-23)

The corrected wide-M thesis now has direct in-tree evidence. R6 pairs adjacent
K16 fragments into aligned K32/dwordx4 U4/S4 tiles and replaces the one-chain
core with a 16-accumulator-per-wave, M256-blocked IU4 WMMA owner. Four waves
share one 8-KiB staged S4 slab across 256 physical rows. The cache-only object
is local128/VGPR192/LDS8192 with 32 reported scratch bytes and an upper-bound
eight VGPR-limited waves/SIMD; a scratch-free local-array rewrite was rejected
after regressing the short R7 screen by 8.0–16.1%. M128 executed arithmetic
moves **12.54 → 64.39 TOPS (5.13×)** versus R2.

R7 then compares operation-complete pack+projection+correction+BF16+SiLU on
both sides against the missing B3 current prefill owner:

| M | Current pack8/F16 ms | IU4/S4 ms | Speedup | Paired wins | IU4 executed TOPS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 3.818 | 1.431 | **2.668×** | 15/15 | 64.96 |
| 128 | 3.940 | 1.455 | **2.708×** | 15/15 | 64.13 |
| 256 | 4.152 | 1.534 | **2.707×** | 15/15 | 60.96 |
| 512 | 8.198 | 2.640 | **3.105×** | 15/15 | 70.03 |

All candidate outputs are finite and repeat-bit-deterministic, and tracked
allocation returns 0→0. Applying the fresh M512 leaf delta to 64 gate/up layers
and the prior Q4_K_S 396.09-tok/s pp512 packet gives an **inferred**, not
measured, family share of 40.59% and a projected 1.380× complete-prefill gain
(396.09→546.44 tok/s). R8 below supersedes this projection with a measured
complete-model result.

This is a **T3 speed screen only**. The sidecar still comes from re-quantized
dequantized Q4_K_S at ~15.9% weight NRMSE, not original weights. No model-logit,
BF16-relative, category/task, down-projection, manifest, lifecycle-at-scale, or
runtime-route gate has passed. The speed result therefore authorizes R8 work,
not product retention.

### Marginal value of tiny-M U4/S4 (2026-08-23 review 2)

The measured DOT8 M4 gate/up win of **1.230×** is real, but it is **not
1.230× of representation value**. Most of it is recovery of a bandwidth deficit
in the *current exact owner* that is available with no sidecar at all.

R1 measures the same 6.417-GB gate/up sweep twice: **30.162 ms @ 212.76 GB/s at
B1/M2** and **35.576 ms @ 180.4 GB/s at B3/M4**. Identical bytes, 17% worse
bandwidth. So:

| Path | Saves at B3 | % of 111.071-ms cycle | Cost |
| --- | ---: | ---: | --- |
| **Exact M4 schedule fix** (reach the owner's own M2 bandwidth) | **5.414 ms** | **4.87%** | none — exact, 0 added bytes, no quality gate |
| DOT8 U4/S4 sidecar @ 1.230× | 6.652 ms | 5.99% | +5.33 GiB, T3, unqualified re-Q4 |
| **Marginal value of the sidecar** | **1.238 ms** | **1.11%** | the entire T3 campaign |

**81.4% of the DOT8 win is something the exact owner should already have.**
The honest standalone case for a tiny-M U4/S4 representation is therefore about
**1.1% of a B3 cycle**, not 6%. Any summary quoting ~6% for the sidecar is
comparing it against a control that is itself known to be leaving 4.87% on the
table, and overstates the marginal case by roughly 5×.

This does not change the tiny-M rejection — it makes it decisive. It also
re-ranks the work: **R9 item 2 (exact M4 gate/up) is not a smaller version of
the sidecar path, it is 81% of the same prize with none of the cost**, and it
must be measured *before* any tiny-M representation is re-screened, or the
representation will keep being credited with the schedule's deficit.

The same correction applies to the M2/M3 rows the coder reports (0.952× loss,
1.065× win). Those track the control's bandwidth deficit almost exactly — at
M2 the owner is already at 212.8 GB/s and there is nothing to recover, so the
sidecar loses; at M4 the owner is 17% down and the sidecar wins. **The sidecar's
measured speedup is a function of the control's inefficiency, not of M.**

## 1. Scope and non-goals

The initial target is Qwen3.8-27B Q4_K_S on `hip_gfx1151`:

- physical target rows `M=B+1`, especially `M=2,3,4,5`;
- H=5,120 and FFN=17,408;
- FFN gate/up first, then down only if gate/up survives;
- the existing Q4_K_S model, BF16 KV, MTP proposal, attention/state, acceptance,
  transaction, and commit machinery remain authoritative;
- current exact qmicro Q8_1×Q4 DOT4/DP4A owners remain the strict fallback.

This document does **not** claim:

- a single-request B1–B3 operation-complete IU4 win;
- a verifier or MTP throughput win;
- production-profile admission;
- exact trajectory preservation after a new S4 weight quantization;
- that Kairic's package throughput is comparable to hipEngine's B3 result; or
- that a full sidecar should be resident by default.

## 2. Evidence already in hipEngine

### 2.1 Current exact B3 is verifier-bound

The retained exact native B3 artifact
[`2026-08-17-gfx1151-qwen38-27b-q4ks-exact-native-b3.json`](benchmarks/results/2026-08-17-gfx1151-qwen38-27b-q4ks-exact-native-b3.json)
reports:

| Quantity | Measured value |
| --- | ---: |
| True AR | 13.273 tok/s |
| Native B3 | 24.193 tok/s / 1.823× AR |
| Cycles | 87 |
| Target rows | 335 / **3.851 rows per cycle** |
| Proposal | 0.206 s total / 2.37 ms per cycle |
| Target verify | 9.663 s total / **111.07 ms per cycle** |
| Target-verify share of recorded decode stages | **97.41%** |
| Tracked peak | 16.914 GiB |

The retained gate/up route is
`dense_dual_q8_1x2_rowtile8_dp4a_bf16_bf16_out`. It quantizes activations into
two Q8_1 planes, shares each qmicro-Q4 traversal across rows, and preserves the
serial-c1 DOT4/FMA/BF16/SiLU order. The rows-3 profiler sample is
0.462–0.465 ms for one actual layer pair, local size 128, 120 VGPR, 512 B LDS,
and zero scratch.

This establishes that target verification is the right system component, but it
does not establish that target projection arithmetic is compute-bound.

### 2.2 New comparative instruction screen

Reproduce with:

```bash
HIPENGINE_HIP_ARCH=gfx1151 \
python3 scripts/mtp_iu4_roofline.py \
  --iterations 65536 --samples 9 --warmups 3 \
  --output benchmarks/results/2026-08-23-gfx1151-mtp-iu4-instruction-roofline.json
```

Hardware/software: host `gfx1151`, AMD Radeon 8060S, 40 CU, wave32, 2.9 GHz
maximum device property, TheRock HIP 7.15 / clang 23, native `gfx1151` code
object. One wave32 workgroup runs 2/4/8 independent accumulator chains. HIP
events time one register-resident kernel launch. The table selects the strongest
**median**, not the best sample.

| Arithmetic lane | Best chains | Median | Relative to U8×S8 WMMA |
| --- | ---: | ---: | ---: |
| FP16 WMMA, F32 accumulate | 2 | 55.066 TFLOP/s | 1.001× |
| BF16 WMMA, F32 accumulate | 8 | 55.381 TFLOP/s | 1.007× |
| S8×S8 WMMA, I32 accumulate | 8 | 55.035 TOPS | 1.000× |
| U8×S8 WMMA, I32 accumulate | 8 | **55.015 TOPS** | 1.000× |
| S4×S4 WMMA, I32 accumulate | 8 | 109.660 TOPS | 1.993× |
| U4×S4 WMMA, I32 accumulate | 8 | **109.715 TOPS** | **1.994×** |
| U4×U4 WMMA, I32 accumulate | 8 | 109.901 TOPS | 1.998× |
| U8×S8 vector DOT4 | 8 | **28.252 TOPS** | — |
| U4×S4 vector DOT8 | 8 | **56.830 TOPS** | — |

The WMMA lanes sustain about 92–93% of the 59.392/118.784 theoretical peaks;
selected WMMA timing stdev is below 0.70%. Signedness is performance-neutral within
about 0.2%: the important distinction is representation/correction, not the
sign flag.

The build's saved assembly contains all expected instructions:

- `v_wmma_f32_16x16x16_f16`;
- `v_wmma_f32_16x16x16_bf16`;
- `v_wmma_i32_16x16x16_iu8`;
- `v_wmma_i32_16x16x16_iu4`;
- `v_dot4_i32_iu8`; and
- `v_dot8_i32_iu4`.

A cache-only `rocprofv3 --kernel-trace` sees all 27 lane/chain dispatches. The
selected 8-chain IU4 body uses 72 VGPR, no LDS, and no scratch. Full samples,
commands, compiler/source hashes, ISA counts, and profiler resources are in
[`2026-08-23-gfx1151-mtp-iu4-instruction-roofline.json`](benchmarks/results/2026-08-23-gfx1151-mtp-iu4-instruction-roofline.json).
The artifact is deliberately `diagnostic_only` and `performance_claim=false`
because it excludes the operation and was measured from the disclosed shared
dirty worktree.

## 3. Tiny-M roofline

Assume a packed S4 weight matrix, packed U4 activations, one weight read per
physical verifier batch, and ignore metadata/output traffic for the optimistic
first bound:

```text
ops                 = 2*M*K*N
weight bytes        = K*N/2
weight-only OI      = 4*M ops/B
memory roof         = OI * 221 GB/s
measured IU4 roof   = 109.715 TOPS
compute break-even  = 109715/221 / 4 ≈ M=124
```

| Physical M | Weight-only OI | 221 GB/s roof | Useful IU4 WMMA roof after 16-row fill | Current DOT4 roof | First-order limit |
| ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 8 ops/B | 1.768 TOPS | 13.71 TOPS | 28.25 TOPS | memory |
| 3 | 12 ops/B | 2.652 TOPS | 20.57 TOPS | 28.25 TOPS | memory |
| 4 | 16 ops/B | 3.536 TOPS | 27.43 TOPS | 28.25 TOPS | memory |
| 5 | 20 ops/B | 4.420 TOPS | 34.29 TOPS | 28.25 TOPS | memory |
| 8 | 32 ops/B | 7.072 TOPS | 54.86 TOPS | 28.25 TOPS | memory |
| 16 | 64 ops/B | 14.144 TOPS | 109.72 TOPS | 28.25 TOPS | memory |
| 32 | 128 ops/B | 28.288 TOPS | 109.72 TOPS | 28.25 TOPS | crossover for DOT4 |
| 64 | 256 ops/B | 56.576 TOPS | 109.72 TOPS | 28.25 TOPS | IU4 can matter |
| 96 | 384 ops/B | 84.864 TOPS | 109.72 TOPS | 28.25 TOPS | mixed |
| 128 | 512 ops/B | 113.152 TOPS | 109.72 TOPS | 28.25 TOPS | compute |

Activation reads, output writes, scales/sums, imperfect coalescing, quantization,
and epilogues lower these roofs. The useful-WMMA column is itself optimistic: it
only charges padded rows and assumes no register/LDS packing cost.

### Consequence for B3

At average B3 `M=3.851`, a native 16-row IU4 tile exposes only about 24% useful
rows. Its useful raw arithmetic ceiling is approximately the existing DOT4
ceiling, while both ceilings remain far above the weight-bandwidth roof. IU4 can
still win by reducing qmicro metadata decode, corrections, or resident bytes,
but that is an operation-design win—not the 2× instruction ratio.

For the measured gate/up pair:

| Quantity | Current qmicro Q4 | Ideal S4 companion |
| --- | ---: | ---: |
| Pair bytes, one layer | 100,270,080 | 89,407,488 incl. scale+sum |
| Measured/read floor | 0.463 ms measured | 0.405 ms at 221 GB/s |
| Effective current read rate | 216.6 GB/s | — |
| Optimistic leaf ceiling | — | about **1.14×** before pack/correction |

The former single-layer projection was 29.6 ms of a 111.1 ms retained-suite
verify pass. R1 supersedes that estimate with a cache-only exact native trace:
30.162/30.653/35.576 ms of gate/up core at B1/B2/B3. The B3 S4 family read floor
is 25.892 ms at 221 GB/s, so its impossible zero-pack saving is 9.684 ms of the
profiled 148.655-ms target window (1.070×). The original 1.14× screening ceiling
saved only 4.37 ms (about 2.9%); the rebuilt R6/B4 body supersedes that leaf
implementation and supports the more specific projection below.

#### Current answer: does IU4 make the exact B3 verifier faster?

**Not with any retained runtime product. Mechanically, a compact U4/S4 DOT8
leaf can improve the dominant M4 gate/up family by enough to project about a 6%
complete B3 gain.** This is a representation/operation effect, not the 2× IU4
WMMA lane:

- The retained exact B3 suite averages `M=3.851`; its unprofiled target verify is
  **111.071 ms/cycle**, complete cycle wall is **114.023 ms**, and it publishes
  **2.7586 visible outputs/cycle** at **24.193 tok/s**.
- R6/B4 uses U4/S4 **DOT8**, not 16-row IU4 WMMA, at M2–M16. Against the exact
  qmicro owner it measures **0.952×/1.065×/1.230×/1.501×** at M2/M3/M4/M5.
  M2 loses; M3 is a small win; M4 is the relevant full B3 shape.
- Applying only the measured M4 1.230× leaf ratio to R1's 35.576-ms gate/up
  family gives **28.914 ms**, a 6.662-ms saving. Holding acceptance and every
  other stage fixed projects **107.361 ms/cycle and about 25.69 tok/s
  (1.062×)**. This is an optimistic Amdahl projection, not a measured verifier
  result: the natural suite contains M2/M3 tails and no model-wide tiny-M route
  was run.
- The existing Kairic `iu4_s4_kairic_ffn_v1` product admits only physical
  **M96–M2048**. B3 M2–M4 therefore takes strict fallback and receives **zero
  IU4 acceleration** from the current product.
- The only measured tiny-M sidecar re-quantizes dequantized Q4_K_S rather than
  original weights, adds 5.329 GiB for gate/up, and is not model-quality
  qualified. The immutable original Kairic PFS also fails both Q4_K_S and
  matched-authority full-model distribution gates. Neither can be promoted to
  MTP.

Therefore IU4 is neither the current answer to the full **1.474×** target-pass
tax nor a 2× B3 opportunity. It is a plausible **~5–6%** exact-B3 mechanism only
if a new original-weight tiny-M product passes the complete T3 gate. The first
next work must optimize and attribute the exact qmicro verifier without changing
representation; only then is the incremental residency/quality cost of an S4
product interpretable.

### 3.1 Implementation-quality control (2026-08-23 review)

R2 compared a naive candidate against a fully tuned control and attributed the
gap to the representation. The artifact's own fields refute that attribution:

| M | Control ms | IU4 ms | **IU4 GB/s** | **Control GB/s** | IU4 executed TOPS | Row util |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.459 | 0.584 | **153.9** | **218.3** | 9.82 | 0.125 |
| 3 | 0.473 | 0.591 | **153.5** | 212.0 | 9.79 | 0.188 |
| 4 | 0.553 | 0.601 | **152.6** | 181.4 | 9.73 | 0.250 |
| 5 | 0.682 | 0.571 | 152.7 | 147.1 | 9.74 | 0.312 |
| 8 | 0.958 | 0.600 | 154.2 | 104.6 | 9.84 | 0.500 |
| 16 | 1.913 | 0.600 | 150.2 | 104.9 | 9.59 | 1.000 |
| 32 | 3.874 | 1.104 | 161.4 | 103.5 | 10.30 | 1.000 |
| 64 | 7.961 | 1.991 | 183.7 | 100.8 | 11.72 | 1.000 |
| 96 | 11.977 | 2.880 | 189.8 | 100.5 | 12.11 | 1.000 |
| 128 | 16.374 | 3.669 | 196.5 | 98.0 | 12.54 | 1.000 |

Two diagnostics dominate. First, **candidate core time is flat at ~0.59 ms from
M2 through M16** while row utilization rises 8×; the kernel is insensitive to
the very axis the tiny-M thesis is about. Second, the candidate never approaches
either roof: 153 GB/s against a 221 GB/s memory roof, and 9.6–12.5 TOPS against
a 109.715 TOPS arithmetic roof.

The cause is visible in the retained source and the retained rocprof block, and
is a register-blocking deficit rather than anything about U4/S4:

| Property | R2 candidate `iu4_s4_dual_silu_bf16_kernel` | Production owner `gguf_q4_k_pack8_dual_wmma_prefill_silu_bf16_kernel` |
| --- | --- | --- |
| Source | `hip_gfx1151/quant/iu4_s4_sidecar.hip:143` | `hip_gfx1100/quant/gguf_q4_k_prefill.hip:515` |
| Accumulators per wave | **1** (`i32x8_t acc`) | **16** (`acc_a[2][4]` + `acc_b[2][4]`) |
| Workgroup | 64 threads / 2 waves | 128 threads / 4 waves |
| VGPR (measured) | **24** | register-blocked |
| LDS (measured) | **1 KiB** (epilogue staging only) | **32 KiB** weight staging union |
| Operand staging | none; every block re-reads A from global | LDS-staged weights, reused fragments |
| N-tile per block | 1 (16 columns) | 2 (32 columns) |
| Dependent WMMA chain | **320-deep, single chain** | 16 independent chains |

The consequences compound:

1. **One accumulator chain.** R0's own instruction screen established that
   **8 independent chains** are required to reach 109.715 TOPS; a single chain
   exposes the full WMMA latency chain 320 times per block.
2. **No activation reuse.** With one 16-column N-tile, all 1,088 x-blocks
   re-read the whole packed activation tile: about **44.6 MB of redundant
   logical traffic against 89.4 MB of weights**, roughly a 50% overhead.
3. **Shallow memory-level parallelism.** 24 VGPR and no K-unroll leave only
   about two 8-byte loads in flight per wave, and `lane16 = lane & 15` means
   lanes 16–31 duplicate lanes 0–15, so each 32-lane load instruction fetches
   only 128 unique bytes.

The M32–M128 bandwidth climb (161→196 GB/s) is itself confirmation: the only
thing that changes there is `grid.y` growing 2→8, adding the block-level
parallelism the kernel fails to express within a block.

**Rule adopted from this review:** a candidate kernel may not be compared
against a tuned production owner until it reaches a comparable fraction of the
roof that binds it. Any future IU4 leaf must report candidate-versus-control
effective bandwidth and percent-of-arithmetic-roof in its artifact, and a
candidate below ~85% of its binding roof is a **kernel result, not a
representation result**.

## 4. General U4 applicability beyond MTP

The measured 2× instruction rate is not MTP-specific. It is a gfx1151
arithmetic capability that can serve any contraction whose operands can be
represented as packed 4-bit lanes. The practical question is narrower: **which
current hipEngine stages are limited by FP16/BF16/IU8 arithmetic rather than by
weight/KV traffic, launch overhead, tile underfill, packing, or corrections?**

### 4.1 Conditions required to realize the 2× arithmetic lane

Relative to the measured 55.015–55.381 TOPS/TFLOP/s IU8/BF16 WMMA roofs, the
109.715-TOPS U4×S4 roof can approach a 2× kernel gain only when all of the
following hold:

1. the control is compute-bound at the lower WMMA roof;
2. physical work fills the 16-row WMMA tile and supplies enough independent
   output tiles to occupy the GPU;
3. packed U4 activations and S4 weights are already available, or their packing
   is amortized inside the measured operation;
4. I32 zero-point correction, scales, BF16/F32 publication, and fused epilogues
   do not become the new bottleneck; and
5. the U4/S4 representation passes its T3 model-quality gate.

For an optimistic packed-W4 linear using the measured 221-GB/s read roof,
`OI≈4*M` ops/B. The measured BF16/IU8 WMMA roof crosses that memory roof around
`M≈63`; IU4 crosses around `M≈124`. This creates four distinct regimes:

| Physical rows | First-order interpretation |
| ---: | --- |
| `M<16` | WMMA underfill plus weight bandwidth; the 2× instruction rate is mostly inaccessible. |
| `M≈16–63` | Better row reuse, but both old and IU4 lanes are still principally weight-bandwidth limited. |
| `M≈64–123` | The current BF16/IU8 lane may become compute-bound, but IU4 often moves the operation back under the memory roof; expect less than 2× unless layout/dequant work is also removed. |
| `M>=124` | Both lanes can be compute-bound in the weight-only model, so a well-tiled operation has the clearest chance to approach the measured 1.99× arithmetic ratio. |

These are screening thresholds, not dispatch thresholds. Activation/output
traffic, metadata, selected-expert fragmentation, imperfect reuse, and
application layouts move the real crossover. The measured U4×S4 DOT8 lane also
has a genuine **56.830 versus 28.252 TOPS** advantage over U8×S8 DOT4, but a
vector kernel benefits only if it is arithmetic-bound; replacing DOT4 in a
streaming c=1 GEMV does not remove its weight-bandwidth limit.

### 4.2 Current hipEngine opportunity ranking

| Surface | Current limiting evidence | U4 applicability | Priority |
| --- | --- | --- | --- |
| **Dense bulk prefill FFN and projections** | The Qwen3.8 gfx1151 campaign identifies prefill as compute-bound: roughly 370–380 tok/s at 512/1K/4K. The current dense Q4_K_S gate/up prefill owner reconstructs FP16 fragments, uses F16 WMMA with F32 accumulation, and publishes BF16 boundaries. | **Strongest current candidate.** Rows 128–512 fill IU4 tiles and exceed the weight-only crossover. A clean offline S4 product plus U4 activation packing could replace reconstruction and use the 2× lane. | **1** |
| **Packed concurrent decode / verification** | Retained gfx1151 rowtiles own rows2–8; larger widths decompose into <=8-row chunks and reread weights. There is no exact single-sweep M16+ batch owner. | Promising at sustained packed `M>=16`, strongest at `M>=64–128`; first build the R10 control. R2's M16–128 repeated-rowtile rows are not that control. | **2** |
| **Dense/shared-expert MoE prefill** | Large grouped token sets can be WMMA-heavy, but selected tokens fragment across experts and often leave each expert at small M. | Useful for shared experts or grouped experts only when the measured per-expert row histogram fills tiles. Do not infer eligibility from total prompt rows. | **3** |
| **K/V projection linears** | These are ordinary weight contractions: large-M during prefill, M1 during serial decode. | Same disposition as other projections: plausible for bulk prefill, not for c1 merely because the output roles are K and V. | **3** |
| **Prefill attention `QK^T` / `PV` tiles** | Attention can expose large matrix tiles, but current wall also includes layout, masks, softmax, and KV movement; no retained profile establishes these contractions as an IU4-addressable compute bottleneck. | Technically possible, but requires quantized Q/K and possibly probabilities/V plus float softmax boundaries. Treat as a separate quality-sensitive attention product after linear prefill. | **4** |
| **Long-context KV-cache decode** | Primarily streams K/V and performs reductions at tiny query M. Existing INT8-KV evidence is model-specific, and lower precision faces a harder quality gate. | U4 may help by reducing cache bytes, not primarily through 2× TOPS. A K4/V8 leaf using U4×S4 for `QK^T` is more defensible than immediately quantizing both K and V to four bits. | **capacity/bandwidth lane** |
| **LM head, embeddings, norms, RoPE, softmax, sampler, GDN/state recurrence** | Singleton/low-reuse weight reads, reductions, elementwise work, or sequential state traffic dominate; several are not matrix contractions. | Poor match for the WMMA advantage. DOT8 may be usable in isolated reductions, but there is no current compute-bound, quality-qualified owner that predicts a material request-level gain. | **defer** |

The dense-prefill classification and routing context are recorded in the
[Qwen3.8 gfx1151 campaign](docs/QWEN38-27B-GFX1151-CAMPAIGN.md) and its
[`R6` compact artifact](benchmarks/results/2026-08-18-gfx1151-qwen38-27b-r6-int8-activation-prefill.json).
The F16 instruction is explicit in
[`gguf_q4_k_prefill.hip`](hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_prefill.hip).
The later c=N rowtile retention supersedes that campaign's original c8 WMMA
fallback observation; R10 states the remaining single-sweep control gap.

The strongest answer to “where are we currently compute-bound?” is therefore
**dense prompt prefill**, especially FFN gate/up/down and large dense
projections. That is also the surface for which the current R2 artifact must
not be reused as proof: its M16–128 control intentionally repeats the decode
rowtile8 owner. A prefill decision requires an operation-complete comparison
against `pack8_dual_wmma_prefill_bf16_bf16_out` and the corresponding current
bulk down/projection owners at actual prompt shapes.

### 4.2.1 Why prefill is the compound case (2026-08-23 review)

This section previously ranked prefill first without quoting a single prefill
number, which left the top-ranked opportunity unquantified. Two structural
facts, both checkable in tree, make the case concrete:

- **Prefill is not weight-bandwidth-bound.** At the campaign's ~399 tok/s
  512-shape prefill, the 16.12 GB model file is swept once per 512-token
  forward pass, i.e. on the order of **12.5 GB/s against a 221 GB/s roof**.
  Unlike every decode/verify shape in this document, prefill has bandwidth
  headroom to spare and is limited by arithmetic and dequantization work.
- **IU4 removes a stage the current owner cannot avoid.** The retained owner
  `gguf_q4_k_pack8_dual_wmma_prefill_silu_bf16_kernel` reconstructs Q4_K
  subblocks into FP16 in a 32-KiB LDS union (`scale*q - min` per value) before
  any WMMA issues. **A packed S4 companion deletes that decode entirely** —
  nibbles feed `v_wmma_i32_16x16x16_iu4` directly — *and* doubles the
  arithmetic roof from 55.066 to 109.715.

That compound effect (dequant elimination **plus** 2× arithmetic), not the
instruction ratio alone, is the most plausible explanation for Kairic's
2.52–3.48× at M96–512 in §5. It is also the only surface where all five
§4.1 conditions can hold at once.

The corresponding warning is that **R2's kernel is not a candidate for this
comparison as written.** Per §3.1 it has 1 accumulator per wave versus the
prefill owner's 16, no operand staging versus 32 KiB, and 24 VGPR. Against a
tuned bulk-prefill owner it would plausibly *lose* despite holding a 2×
arithmetic roof. The R6/R7 result above closes that implementation prerequisite.

### 4.2.2 Disposition of the DOT8 lane

The measured **56.830 versus 28.252 TOPS** U4×S4 DOT8 advantage deserves an
explicit disposition rather than the single dismissive sentence in §4.1,
because DOT8 is the one lane with **zero tile-underfill exposure** and is
therefore the natural tiny-M shape: it would inherit the existing rowtile8
owner's demonstrated 218 GB/s memory behavior and swap only the arithmetic.

**Disposition: rejected for tiny M, on the §3.1 economics rather than on
speed.** B4's actual DOT8/S4 port reaches **0.471/0.436/0.438 ms core** and
**189.7/205.2/204.3 GB/s** at M2/M3/M4. Inclusive M2 still loses (0.952×),
while M3/M4 reach only 1.065×/1.230×. This validates the bandwidth thesis but
not the optimistic ~0.41-ms floor: the only available lever remains the 1.12×
byte ratio, and the complete target-window ROI cannot justify 5.329 GiB plus a
T3 campaign. DOT8 is worth revisiting only if a future low-M shape is
arithmetic-bound — none is currently identified. Recorded so this option is
not re-derived later.

### 4.2.3 gfx1100 port screen (2026-08-24)

The whole gfx1151 plan leaves one question open: does the IU4 capability
exist on RDNA3, or is it a gfx1151/RDNA3.5 artifact? We now have a direct
answer from the same in-tree instruction screen run on both gfx1100 devices
(W7900 and 7900 XTX). The ISA transfer is confirmed and the headline 2×
lane transfers; the tiny-M DOT8 advantage does **not**.

| Arithmetic lane (best median) | gfx1151 Radeon 8060S | W7900 gfx1100 | 7900 XTX gfx1100 |
| --- | ---: | ---: | ---: |
| FP16 WMMA | 55.066 TFLOP/s | 126.9 | 140.1 |
| BF16 WMMA | 55.381 TFLOP/s | 133.3 | 141.6 |
| IU8 WMMA (`v_wmma_i32_16x16x16_iu8`) | 55.015 TOPS | 122.1 | 134.3 |
| **IU4 WMMA (`v_wmma_i32_16x16x16_iu4`)** | **109.715 TOPS** | **242.0** | **267.0** |
| **IU4/IU8 WMMA ratio** | **1.994×** | **1.981×** | **1.989×** |
| DOT4 (`v_dot4_i32_iu8`) | 28.252 TOPS | 52.3 | 56.3 |
| DOT8 (`v_dot8_i32_iu4`) | 56.830 TOPS | 61.2 | 66.2 |
| **DOT8/DOT4 ratio** | **2.012×** | **1.188×** | **1.169×** |

All six target instructions compile **and execute** on gfx1100
(`isa_all_expected_seen: true` in both artifacts):
`v_wmma_i32_16x16x16_iu4` and `v_dot8_i32_iu4` are present in the gfx1100
ISA, so this is not a gfx1151-only instruction set. Same harness and
parameters as the gfx1151 row (65536 iters, 9 samples, 3 warmups, best
median). Artifacts:
[`W7900`](benchmarks/results/2026-08-24-gfx1100-w7900-mtp-iu4-instruction-roofline.json)
and
[`7900 XTX`](benchmarks/results/2026-08-24-gfx1100-7900xtx-mtp-iu4-instruction-roofline.json).

**The prefill thesis transfers.** IU4 WMMA sustains **1.981×/1.989×** IU8 on
gfx1100, matching gfx1151. gfx1100's absolute IU8 roof is already ~2.3×
gfx1151's (122–134 vs 55 TOP/s), so IU4 doubles an already-higher ceiling.
The compute-bound dense-prefill surface (§4.2, R7/R8/R13) is therefore the
right place to look on gfx1100: the 2× lane plus Q4→FP16 dequant removal
should transfer. That transfer is now measured (2026-08-24): the R7 prefill
gate/up A/B gives **2.858×/2.893×/2.920×/3.227× at M64/128/256/512 on the
W7900** and **3.100×/3.028×/3.107×/3.454× on the 7900 XTX**, both 15/15
paired wins, candidate at 61–69% of the measured gfx1100 IU4 roof.
[`W7900 A/B artifact`](benchmarks/results/2026-08-24-gfx1100-qwen38-iu4-s4-r7-prefill-gate-up-ab.json)
and
[`7900 XTX A/B artifact`](benchmarks/results/2026-08-24-gfx1100-7900xtx-qwen38-iu4-s4-r7-prefill-gate-up-ab.json).
This is a T3 speed screen only (`model_logit_quality_gate_run: false`): the
S4 sidecar is the same re-quantized-from-Q4_K_S representation that fails the
full-model distribution gate on gfx1151, so gfx1100 prefill IU4 has the same
unqualified-T3 status and is **not** a production path. It also confirms the
§3.1 rule: the candidate reports 61–67% of its binding roof, so the leaf is a
representation+2×-lane result, not an unblocked-kernel artifact.

**The tiny-M DOT8 thesis is weaker on gfx1100, not stronger.** gfx1151's 2×
DOT8-over-DOT4 (56.83 vs 28.25) is largely a gfx1151 DOT4 under-measurement:
gfx1151 DOT4 runs at ~half its WMMA roof. On gfx1100, DOT4 is already
efficient (~45% of the 122–134 IU8 roof), so the DOT8 lane adds only
**1.169–1.188×**. Since B1–B3 verifier decode is a DOT-instruction,
memory-bound regime anyway (§3), the only remaining lever is the 1.12× byte
ratio, and the marginal-value argument from the Verdict review applies *more*
strongly here: the DOT8 arithmetic advantage that partially motivated the
gfx1151 B4 leaf is largely absent on gfx1100. That is now measured directly
(2026-08-24, tiny-M gate/up leaf on gfx1100 W7900, rows 2–128, 15/15 samples):
1.438×/1.271×/1.956×/2.295× at M2/M3/M4/M5 — which *looks* like a pass
(`tiny_m_gate_passed`) but is a **control-deficit + cache artifact**: the exact
rowtile8 control reaches only 269–502 GB/s (31–58% of the 864 GB/s W7900
memory roof, tuned for gfx1151 and unsaturating on gfx1100), while the
candidate reports 891–1458 GB/s (103–169% of roof), which is impossible from
HBM and means L2/cache-assist or under-measured work. The mechanical speedup
tracks the control's deficit (doc review 2), not IU4 representation value;
the representation is unchanged and still fails the T3 distribution gate.
[`leaf artifact`](benchmarks/results/2026-08-24-gfx1100-w7900-qwen38-iu4-s4-b4-tiny-m-gate-up-leaf.json)
**Do not pursue a tiny-M B1–B3 IU4 sidecar on gfx1100**; the exact-M4 qmicro
bandwidth fix (R9 item 2) stays the decode lever and is representation-free.

### 4.3 KV-cache use is a different thesis

A native four-bit cache path is feasible in principle:

```text
Q_BF16 -> dynamic asymmetric U4
K_cache -> symmetric/grouped S4
QK^T    -> IU4 I32 dot + zero correction + FP32 scale
mask/softmax -> FP32/BF16
PV      -> initially retain INT8 or BF16 V
```

For serial decode, vector `v_dot8_i32_iu4` or packing independent
heads/requests into WMMA rows is more natural than padding one query to M16.
The first experiment should be **K4/V8**, preserving float mask/softmax and the
existing `KVLiveSpans` ownership contract. K4/V4 should follow only if K4/V8
passes complete attention-output and model-quality gates. Any retained format
must be a registered `KVStorageView`/`KVCacheBackend` product with payload,
scale, zero, lifecycle, fallback, and capacity accounting—not a dtype branch in
the attention kernel.

This lane is lower priority for exploiting the 2× arithmetic result. Long
context is usually cache-bandwidth-bound, so its success criterion is fewer
bytes at equal quality and lower complete attention wall. It may be valuable on
a capacity-constrained 24-GiB device, but gfx1151's 128-GiB unified memory and
the W7900's existing 256K INT8 reach weaken the capacity case. The repository's
model-dependent INT8-KV outcomes also warn against assuming that plain S4 will
qualify; Hadamard/group scaling, variance normalization, or mixed K4/V8 may be
required.

### 4.4 Recommended non-MTP experiment order

R7/R8 complete the original gate/up and full-FFN speed screens; both current
representations fail production quality. Continue in this order:

1. **Exact expanded-F16 prefill control (R11):** isolate dequant removal from
   the 2× IU4 lane against the current bulk owner.
2. **New original-weight S4 revision (R12):** only after the exact control and a
   predeclared full-model quality gate; do not salvage the immutable failed PFS.
3. **Packed c>N linear owner (R10):** only after a clean single-sweep native
   batch baseline exists; profile physical-M histograms and distinguish
   concurrent-request throughput from single-request latency.
4. **Grouped/shared-expert MoE:** proceed only where actual per-expert rows are
   large enough to fill IU4 tiles.
5. **K4/V8 attention leaf:** pursue for measured long-context bandwidth or
   capacity pressure, not as an automatic consequence of the TOPS result.

A raw 2× instruction ratio is the admission signal for these screens, not a
performance claim. Retention still requires operation-complete timing, current
owner comparison, full-model Amdahl impact, representation bytes, and the
applicable T3 quality/task/lifecycle gates.

## 5. What Kairic establishes—and what it does not

The immutable ROCmFPX release inspected here is
`ciru-ai/ROCmFPX@e97b32468509270bd15e891973f985b04fe999d7`
(tag/branch `kairic-edge-qwen38-27b-v1`). Its CK patch adds the GFX11 IU4 WMMA
selector and passes two packed I32 VGPRs per operand to
`__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32`. Clang's builtin prototype is:

```c
_ExtVector<8, int> __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
    bool a_sign, _ExtVector<2, int> a,
    bool b_sign, _ExtVector<2, int> b,
    _ExtVector<8, int> c, bool clamp);
```

Kairic's current public evidence is strongest at wider M:

| Physical M | Compact control | Inclusive IU4 FFN | Speedup |
| ---: | ---: | ---: | ---: |
| 96 | 2.928 ms | 1.162 ms | 2.52× |
| 128 | 3.700 ms | 1.219 ms | 3.04× |
| 256 | 6.706 ms | 1.999 ms | 3.35× |
| 512 | 13.222 ms | 3.801 ms | 3.48× |

Those native values include input packing, gate/up, activation-and-pack, down,
and BF16-to-F32 output. A separate exact-M65 serving A/B reports 48.73→52.57
tok/s (+7.89%) and 39.47→37.06 s wall (-6.11%). These rows support a wide-M
route, not a hipEngine B3 prediction.

The release runner explicitly sets:

```text
PROMPTFORGE_ENABLE_IU4_DECODE_M2_M5=0
PROMPTFORGE_ENABLE_SMALLM_IU4=1
```

In that source, “small M” means **96–512**, while the separately named decode
route means **2–5**. Therefore the recommended package does not establish a
qualified M2–M5 IU4 target verifier. It does establish that the instruction,
packing stack, signed-I4 companions, and served routing can work end to end.

Current companion sizes are also larger than the earlier figures in the prompt:

| Companion | Current release size |
| --- | ---: |
| FFN | 7.99 GiB |
| GDN | 1.88 GiB |
| GDN output | 0.70 GiB |
| Total companions | **10.57 GiB** |
| GGUF + companions | **26.05 GiB** |

The 9.13 GiB/46.3% statement on the card compares companions with a matched
8-bit companion inventory; it is not the total IU4 companion byte count.

## 6. Representation and correction math

### 6.1 Clean S4 companion

For one activation row and one output channel:

```text
a_k ≈ s_a * (q_a,k - z_a),     q_a in [0, 15]
w_k ≈ s_w * q_w,k,             q_w in [-8, 7]
```

Then:

```text
a·w ≈ s_a*s_w * (sum(q_a*q_w) - z_a*sum(q_w)).
```

The IU4 instruction computes `sum(q_a*q_w)` with unsigned A / signed B flags.
The offline sidecar stores `s_w` and `sum(q_w)` per output channel (or per
chosen K segment). Dynamic activation packing publishes packed nibbles, `s_a`,
and `z_a` per row/segment. Correction and scale should be fused into the BF16
producer epilogue; gate and up should retain the declared BF16 boundaries before
SiLU unless the T3 quality campaign explicitly qualifies a different boundary.

A single K segment minimizes metadata/correction but maximizes quantization
error. More segments improve quality while increasing sums/scales, correction
work, and fragment scheduling. This is a quality/performance parameter, not a
benchmark-specific knob.

### 6.2 Direct use of current Q4_K/qmicro nibbles

This avoids a large companion but is not the same formula. Q4_K reconstructs
subblocks with affine scale/min terms. A direct U4×U4 route needs both weight-sum
and activation-sum corrections per Q4_K scale/min group, conceptually:

```text
w_k = d*s_g*q_w,k - dmin*m_g
sum(a_k*w_k) = s_a * [
    d*s_g*(sum(q_a*q_w) - z_a*sum(q_w))
  - dmin*m_g*(sum(q_a) - z_a*K_g)
].
```

The exact grouping and rounding must follow the current qmicro CPU/reference
contract. This path saves persistent bytes but adds metadata loads, groupwise
correction, and packing/layout work—the very costs a clean S4 sidecar avoids.
It is a useful ceiling experiment, not the default first implementation.

## 7. Sidecar memory model

For Qwen3.8 H=5,120, I=17,408, 64 layers, packed 4-bit weights plus one FP32
scale and one I32 sum per output channel:

| Scope | Weight+metadata estimate | Added tracked peak from 16.914 GiB |
| --- | ---: | ---: |
| Gate/up only | **5.329 GiB** | about 22.243 GiB |
| Down only | **2.659 GiB** | about 19.573 GiB |
| Full FFN | **7.988 GiB** | about 24.902 GiB |
| Kairic full FFN+GDN+output set | 10.57 GiB | about 27.48 GiB, before integration differences |

These are payload estimates, not whole-process GTT/RSS or transient peak. They
explain Kairic's 7.99 GiB FFN file and correct the earlier “~6 GiB FFN” premise.
A one-layer research sidecar is only about 85.3 MiB for gate/up and 42.5 MiB for
down, so the leaf gate can be run without committing a model-wide allocation.

## 8. Proposed architecture

A retained implementation must follow the existing plugin and fallback design:

1. **Quant/product identity.** A signed-I4 companion is a new model
   representation/product configuration, not a hidden variant of Q4_K_S.
   Resolve it through quant/model capability metadata such as
   `gguf_q4_k_s+iu4_s4_ffn_v1`.
2. **Four-axis kernel key.** Register leaves under
   `(hip_gfx1151, layer-role, iu4_s4_sidecar, variant)`. Do not add backend or
   quant branches to generic engine/model dispatch.
3. **Raw pointers.** Device kernels take packed weight/activation pointers,
   scale/sum/zero planes, dimensions, and output pointers—never framework
   tensors.
4. **Strict fallback.** Every IU4 gate/up/down composite names the current exact
   qmicro/Q4/Q5/Q6 chain as its registered fallback. M=1, misses, unsupported
   roles, failed sidecar validation, and unqualified contexts stay current.
5. **Role/shape routing.** Route by model/backend/representation/role/physical
   M and execution profile. Start experimental and fail closed; do not infer
   eligibility from requested MTP depth alone.
6. **Immutable sidecar manifest.** Bind model SHA-256, tensor name/shape/source,
   quantizer version/calibration, segment size, packed layout, scale/sum offsets,
   byte counts, and payload hashes. Reject partial or mismatched companions.
7. **No CK dependency.** Invoke the already proven Clang builtin directly in
   hipEngine's HIP infrastructure. Kairic's CK patch is a register/layout
   reference, not a runtime dependency.

## 9. Execution-profile classification

A Kairic-style S4 companion changes weight representation. Under
[`docs/EXECUTION-PROFILES.md`](docs/EXECUTION-PROFILES.md), it is **T3**:
representation/algorithm change. It is an explicit experimental product
configuration and is not admitted by the current initial `production` profile
campaign.

A lossless repack of existing Q4_K values with exact reconstruction could be T0
or T2 depending on boundaries. Dynamic U4 activation quantization with unchanged
weights is at least T1. Combining dynamic U4 with newly quantized S4 weights is
still T3.

No arithmetic class relaxes control semantics. Request/slot/token/position,
`KVLiveSpans`, Conv/GDN/KV state ownership, masks, acceptance, commit/rollback,
graph bucket, lifecycle, and provenance remain exact.

## 10. Phased experiment plan

### R0 — instruction capability (complete)

- [x] Compile direct FP16/BF16/IU8/IU4 WMMA and DOT4/DOT8 on gfx1151.
- [x] Measure 2/4/8 dependency chains with HIP events.
- [x] Verify saved ISA and cache-only rocprof dispatch/resources.
- [x] Retain a compact diagnostic artifact.

Result: the U4×S4 instruction lane is real and about 2× IU8/FP16 WMMA.

### R1 — current verifier attribution (complete)

- [x] Profile exact transactional native B1/B2/B3 target windows with cached
  builds and the retained FP32 recurrent-state MTP contract.
- [x] Filter `qwen36_dense_mtp_target_verify_*` ROCTX windows from the final
  child process; do not profile a nested parent harness.
- [x] Record physical-M distributions, calls/time, current variants, qmicro
  bytes/effective bandwidth, and gate/up Amdahl bounds.
- [x] Retain a diagnostic artifact with exact model/raw-trace hashes and no IU4
  or product throughput claim.

| Budget | Physical M histogram | Target wall / cycle | Kernel / cycle | Gate/up core / cycle | Gate/up share kernel / wall | Effective gate/up BW | Ideal zero-pack S4 target ceiling |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 | M2×4 | 137.455 ms | 79.980 ms | 30.162 ms | 37.71% / 21.94% | 212.8 GB/s | 1.032× |
| B2 | M3×2, M2×1 | 121.835 ms | 81.497 ms | 30.653 ms | 37.61% / 25.16% | 209.4 GB/s | 1.041× |
| B3 | M4×2 | 148.655 ms | 89.348 ms | 35.576 ms | 39.82% / 23.93% | 180.4 GB/s | 1.070× |

Each target cycle has 64 qmicro Q8_1x2 rowtile8 gate/up launches. Q8_1x2
activation packing adds only 0.107–0.119 ms/cycle in this trace. The remaining
large projection families are the 212-call Q4 rowtile family (27.99–29.14 ms),
the 60-call Q5 col4 family (9.76–11.03 ms), Q6 tail calls, GDN/Conv, attention,
norms, copies, and the output head; exact details are in the compact artifact.

The trace is deliberately a one-prompt attribution diagnostic. All AR/B1/B2/B3
greedy outputs and GPU/CPU acceptance agree, but its tok/s and absolute wall are
not MTP economics claims. The first preflight used the newer gfx1151 Q4_K_S
FP16-state default and faulted in the incompatible MTP chain-journal path; the
retained run explicitly uses `HIPENGINE_GGUF_FP16_RECURRENT_STATE=0`, matching
the exact B3 state contract.

### R2 — one-layer operation-complete leaf (complete; tiny-M rejected)

Implemented an original, direct-builtin gfx1151 research family under
`hipengine/kernels/hip_gfx1151/quant/iu4_s4_sidecar.{hip,py}`, format helpers
in `hipengine/quant/iu4_s4.py`, and independent arithmetic oracles in
`hipengine/kernels/cpu_reference/iu4_s4.py`:

1. per-output symmetric S4 pack from explicitly labeled dequantized Q4_K_S;
2. dynamic asymmetric U4 pack for M2/3/4/5/8/16/32/64/96/128;
3. coalesced `[N16,K16,N,8]` / padded `[M16,K16,M,8]` native WMMA tiles;
4. direct U4×S4 `__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32`;
5. I32 zero-point correction plus FP32 scales; and
6. BF16 gate/up publication followed by SiLU×up, with the exact qmicro
   Q8_1x2 rowtile8 owner as control/fallback.

The first row-major storage smoke reached only about 42 GB/s and was replaced
before retention by the coalesced tile view. Final actual layer-0 results use a
100,270,080-byte Q4_K_S gate/up source pair, 89,407,488-byte S4 candidate pair,
three warmups, 15 counterbalanced HIP-event pairs, and runtime-equivalent
rowtile8 chunking for the current owner:

| Physical M | Current exact inclusive | IU4 inclusive | Speedup | Paired wins | IU4 core effective BW | Output NRMSE vs current |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.459 ms | 0.584 ms | **0.787×** | 0/15 | 153.9 GB/s | 28.82% |
| 3 | 0.473 ms | 0.591 ms | **0.800×** | 0/15 | 153.5 GB/s | 28.74% |
| 4 | 0.553 ms | 0.601 ms | **0.920×** | 1/15 | 152.6 GB/s | 28.58% |
| 5 | 0.682 ms | 0.571 ms | 1.195× | 15/15 | 152.7 GB/s | 28.46% |
| 8 | 0.958 ms | 0.600 ms | 1.598× | 15/15 | 154.2 GB/s | 28.87% |
| 16 | 1.913 ms | 0.600 ms | 3.190× | 15/15 | 150.2 GB/s | 29.42% |
| 32 | 3.874 ms | 1.104 ms | 3.509× | 15/15 | 161.4 GB/s | 28.96% |
| 64 | 7.961 ms | 1.991 ms | 3.998× | 15/15 | 183.7 GB/s | 29.06% |
| 96 | 11.977 ms | 2.880 ms | 4.158× | 15/15 | 189.8 GB/s | 29.14% |
| 128 | 16.374 ms | 3.669 ms | 4.463× | 15/15 | 196.5 GB/s | 29.00% |

The current M16+ control is intentionally the runtime's repeated rowtile8 decode
owner, not bulk-M512 WMMA prefill. Therefore the wide rows establish a packed
verification/decode crossover, not a prompt-prefill speed claim. R2 stops after
layer 0 because every native B1–B3 shape fails the inclusive speed gate and the
naive re-Q4 sidecar is numerically far from qualified; layers 8/63 and a
model-wide allocation cannot change that go/no verdict.

> **Superseded in part by the 2026-08-23 review (§3.1).** The tiny-M go/no
> verdict survives on representation economics, but **every timing row in this
> table is a lower bound produced by a 1-accumulator, 24-VGPR, 64-thread
> candidate running at 9–11% of the IU4 arithmetic roof and 70% of the memory
> roof.** The M2–M4 losses are implementation-scoped and reverse under a
> properly blocked kernel; the M16–M128 wins are understated by an unknown but
> large factor. **Do not cite any row of this table as a bound on IU4** — cite
> it only as the measured behavior of the R2 reference implementation. R6
> in §13 replaces it.

CPU and GPU nibble/sign/correction tests pass. The cache-only trace sees the
operation-complete core at local64, grid-x 69,632, grid-y 1/2/4/6/8, 24 VGPR,
1 KiB LDS, and zero scratch; saved-code disassembly contains two static
`v_wmma_i32_16x16x16_iu4` sites. Exact commands, samples, hashes, resources,
and sidecar error are in the compact R2 artifact.

### R3 — numerical qualification

Add independent CPU-reference and fixtures before model promotion:

- packed S4 round trip, sums, and segment metadata;
- U4 min/max/constant-row/saturation/zero-point cases;
- all U4/S4 sign and lane-layout cases against I32 matmul;
- correction identity on hand-checkable matrices;
- BF16 boundary and fused/unfused parity where declared;
- actual layer gate/up/down errors for rows 2–5 and transition shapes;
- finite outputs and deterministic same-schedule repeats; and
- registered fallback/manifest resolution and negative-path tests.

Because S4 is T3, leaf KL/top-1 alone cannot promote it. The complete target
model needs strict-teacher full-vocabulary mean/p95/p99/max KL, >=99% top-1,
category/shape/transition reporting, BF16-relative deltas where available,
neighbor isolation, state/KV ownership, and task non-inferiority. The broad
KL<=0.05/top-1>=90% CPU-reference check remains only the outer smoke floor.

### R4 — bounded target integration

Only after R2/R3 pass:

- materialize gate/up companions for all 64 layers;
- add explicit experimental representation selection and route telemetry;
- keep M=1 and unsupported M on the current owner;
- first test c=1 B1/B2/B3/B4, then packed concurrent verifier V buckets;
- record pack/core/correction/epilogue and fallback counters in the target
  window; and
- run graph/eager and reject/partial/full acceptance transactions.

Do not extend to down, GDN, or output projection until the complete gate/up
route reduces measured target wall enough to justify 5.329 GiB.

### R5 — full MTP economics and product decision

A speed claim requires the complete
`benchmarks/prompts/mtpbench-code-general-ja.jsonl` suite plus heldouts, a true
same-protocol no-MTP AR baseline, complete MTP cycle wall, acceptance/density,
all category ratios, tracked and whole-process memory, teardown, and the T3
quality packet. Measure c=1 and packed-concurrency lanes separately.

Promotion questions:

- Does target-verify wall improve after packing and corrections?
- Does complete MTP/true-AR improve in every required category?
- Does a deeper B become economical once row reuse changes?
- Is gate/up-only the best bytes/performance point?
- Is the same sidecar useful for prefill/concurrent verification, or is it dead
  weight outside one route?
- Does gfx1100 independently qualify, or remain unverified?

## 11. Go/no-go gates

| Gate | Pass condition | Current status |
| --- | --- | --- |
| Native instruction | Expected IU4 ISA; >=1.8× matched IU8/FP16 median; stable samples | **Pass**: 1.98–2.00× |
| Tiny-M useful arithmetic | Account for `M/16` tile utilization, not raw TOPS | **Warning**: M4 ≈ DOT4 roof |
| **Candidate implementation quality** | Candidate reaches >=85% of its binding roof before any comparison against a tuned owner | **Pass for B1–B3 shapes / partial through M16**: rebuilt DOT8 reaches 85.8%/92.9%/92.5% at M2/M3/M4 and 91.1%/90.7% at M5/M8; M16 is 83.8%. The wide WMMA core raises M128 executed arithmetic 12.54→64.39 TOPS and passes the direct R7 current-owner gate. |
| Actual-weight tiny-M gate/up | Operation-complete M2–5 beats current owner with pack included and passes declared leaf numerics | **Fail, representation-scoped**: B4 is 0.952×/1.065×/1.230×/1.501× at M2/M3/M4/M5. M2 still loses with the core at 189.66 GB/s; M3–M5 win. The all-shape gate fails and the tiny-M decision stays closed. |
| Actual-weight dense-prefill gate/up | Operation-complete M64/128/256/512 beats the current bulk owner | **Pass speed screen, rejected production T3**: 2.668×/2.708×/2.707×/3.105×, all 15/15 pairs. The original PFS full FFN later passes complete-wall speed but fails both full-model distribution gates. |
| **Tiny-M representation economics** | S4 bytes buy enough target wall to justify the companion | **Fail, representation-scoped (decisive)**: the entire available win is the 1.12× byte ratio → 1.06× target window for +5.329 GiB and a T3 campaign |
| Family attribution | Measured all-layer gate/up share and effective BW justify expected target-wall delta | **Pass, bounded**: 21.9–25.2% wall; ideal zero-pack S4 ceiling 1.032–1.070× |
| Memory ROI | Measured complete target/cycle gain justifies +5.329 GiB gate/up (or +7.988 GiB FFN) | **Split**: fail for c1 B1–B3; prefill product gains 1.44–1.47× complete wall for +7.99 GiB but remains explicit due quality. |
| T3 quality | Full strict-teacher/determinism/isolation/BF16-relative/task packet passes | **Fail against both authorities**: Q4_K_S teacher mean/max KL 0.02119/0.08754 and 88.89% top-1; matched Kairic teacher mean/max KL 0.08159/0.28923 and 77.78% top-1. The matched padded 50-row suite is 100% top-1 but mean/max KL 0.003768/0.06360. Q4_K_S long-task 5/5 non-inferiority and deterministic leaf pass; BF16-relative evidence remains unavailable and cannot rescue either binding failure. |
| MTP economics | Full category suite beats same-protocol true AR and current B3 without category regression | **No retainable run**: current PFS admits M96+, so exact B3 M2–M4 is strict fallback. The research M4 DOT8 leaf projects about 1.062× complete B3, but its re-Q4 representation is unqualified and M2 loses. |
| Lifecycle | Sidecar load/fallback/close reaches zero tracked allocation and bounded process memory | **Model-wide pass**: explicit 8,576,827,392-byte product, M96–2048 route telemetry, M<96/c1 fallback, and tracked teardown 0→0. |

The decisive tiny-M gate is **representation economics plus quality**, not lack
of a mechanical M4 win. R6/B4 reaches a properly roofed 1.230× at M4, but that
projects only about 1.062× complete B3 for +5.329 GiB and a new T3 product; M2
still loses and the available representations fail model-level qualification.

A failed current tiny-M product does not invalidate IU4 generally. Exact B3
attribution/optimization is now first priority. Wider packed verification may
reopen IU4 only after one honest single-sweep native M>=16 control exists;
M96–2048 prompt work remains the demonstrated speed regime, with production
blocked by quality.

## 12. Reproduction command sequence

> The first two commands reproduce retained R0/R1 evidence. The leaf command now
> runs the rebuilt R6/B4 implementation; the old R2 one-chain timing requires
> its recorded commit/artifact. See **§13 Punchlist** for current status.

```bash
# 1. Re-run/inspect the instruction screen.
python3 -m pytest -q tests/test_mtp_iu4_roofline.py
HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/mtp_iu4_roofline.py \
  --iterations 65536 --samples 9 --warmups 3 \
  --output /tmp/mtp-iu4-roofline.json

# 2. Current exact target attribution is retained in the R1 artifact.
# It used the final qwen36_dense_gguf_suite.py child with ROCTX markers,
# cached builds, native B1-B3, and the required FP32-state rollback.

# 3. Re-run the rebuilt operation-complete layer-0 gate (no model route):
HIPENGINE_HIP_ARCH=gfx1151 PYTHONPATH=. \
python3 scripts/qwen38_iu4_s4_gate_up_leaf.py \
  --model /models/gguf/Qwen3.8-27B-Q4_K_S.gguf --layer 0 \
  --rows 2,3,4,5,8,16,32,64,96,128 \
  --warmups 3 --samples 15 --burst 1 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --output /tmp/mtp-iu4-b4.json
```

The current lineage command is mechanically blocked by the already documented
missing `/home/lhl/amd-gpu-tuning/reference/atlas` checkout. No external kernel
was ported in R0; Kairic/Clang were used only as read-only references.

## 13. Punchlist (2026-08-23 review handoff)

Ordered work list for the kernel lane. **B** items are backfill — evidence that
should already exist or corrections to landed material. R6–R8 and B1–B4 are
complete. R9 is now the next priority; representation work may not skip its
exact control.

### Dependency order

```text
B1,B2,B3  (complete)
   |
   v
R6  core rebuild  ------> B4 tiny-M verdict (complete; current product closed)
   |
   +--> R7 prefill gate/up --> R8 full FFN (complete; quality rejected)
   |
   +--> R9 exact B3 attribution/owner optimization
          |
          +--> R10 true single-sweep packed verifier control --> IU4 screen
          |
          +--> R11 exact expanded-F16 prefill control
                   |
                   +--> R12 new original-weight product, if justified
```

### B — backfill

**B1. Artifact schema: mandatory implementation-quality fields — complete.**
Every IU4 artifact must carry, per shape: candidate effective GB/s, control
effective GB/s, candidate percent-of-arithmetic-roof, candidate
percent-of-memory-roof, and the binding roof named explicitly. R2's artifact has
the raw numbers but no derived comparison, which is how the §3.1 defect survived
review. Add to `scripts/qwen38_iu4_s4_gate_up_leaf.py` emission.
*Accept:* re-emitting the R2 artifact shows `binding_roof="memory"`,
`percent_of_binding_roof≈0.70`.

**B2. Retained rocprof block must include occupancy-relevant resources — complete.**
The R2 profiler block records VGPR/LDS/scratch but no waves-per-SIMD or
achieved-occupancy estimate, and nothing flagged that 24 VGPR on a WMMA kernel
is anomalous. Add a derived `accumulators_per_wave` and
`vgpr_anomaly` heuristic (a WMMA kernel under ~64 VGPR is almost certainly
unblocked).
*Accept:* R2 re-emission flags the core.

**B3. Missing prefill baseline — complete.** Measured the current
`pack8_dual_wmma_prefill_silu_bf16` owner alone on actual layer-0 gate/up at
M=64/128/256/512/1024 with 3 paired warmups, 15 counterbalanced HIP-event
activation pairs, and the 133,693,440-byte pack8 cold-pool payload. The body
owns 256 rows per block, so M64/M128 are deliberately charged for the padded
WMMA work and M512/M1024 for two/four complete weight sweeps:

| M | Median ms | Effective payload GB/s | Executed TFLOP/s | F16 WMMA roof |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 3.869 | 34.56 | 23.59 | 42.84% |
| 128 | 3.937 | 33.95 | 23.18 | 42.09% |
| 256 | 4.170 | 32.06 | 21.89 | 39.75% |
| 512 | 8.240 | 32.45 | 22.15 | 40.23% |
| 1024 | 16.234 | 32.94 | 22.49 | 40.84% |

Outputs are finite, activation-sensitive, repeat-bit-deterministic, and tracked
allocation tears down 0→0. This validates the prefill classification: the
control is nowhere near the 221-GB/s weight roof and reaches only about 40–43%
of the measured 55.066-TFLOP/s F16 WMMA roof while also paying Q4_K→FP16
reconstruction. **This is the control R7 must beat.** Artifact:
[`2026-08-23-gfx1151-qwen38-pack8-gate-up-prefill-control.json`](benchmarks/results/2026-08-23-gfx1151-qwen38-pack8-gate-up-prefill-control.json).

**B4. Honest tiny-M re-measure — complete.** Re-ran the existing R2 leaf
protocol after R6. The bandwidth-oriented DOT8 rowtile reaches
189.66/205.23/204.34/201.23 GB/s at M2/M3/M4/M5, so this is no longer the R2
one-chain implementation result. Inclusive current→candidate is
**0.4594→0.4828 ms (0.952×)**, **0.4732→0.4445 (1.065×)**,
**0.5513→0.4480 (1.230×)**, and **0.6734→0.4487 (1.501×)**. The projection is
therefore corrected: M2 still loses, M3/M4 win less than the inferred
1.13×/1.32×, and M5 wins more. This does not reopen the tiny-M decision, which
still fails its all-shape gate and representation ROI. Artifact:
[`2026-08-23-gfx1151-qwen38-iu4-s4-r6-b4-gate-up-leaf.json`](benchmarks/results/2026-08-23-gfx1151-qwen38-iu4-s4-r6-b4-gate-up-leaf.json).

### R6 — candidate core rebuild — complete

Rebuild `iu4_s4_dual_silu_bf16_kernel` to production blocking standards. The
reference for structure is the in-tree owner
`gguf_q4_k_pack8_dual_wmma_prefill_silu_bf16_kernel`
(`hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_prefill.hip:515`) — same tiling
philosophy, different arithmetic. Required changes:

1. **Multiple independent accumulators.** Target 8–16 per wave via an
   `OUT_TILES × ROW_TILES` register tile. R0 established 8 chains as the
   requirement for the 109.715-TOPS roof. Budget ~64–128 VGPR; current 24 is
   the defect signature.
2. **Widen the N-tile per block** to 64–128 output columns so each activation
   fragment is amortized across 4–8 weight tiles, cutting the ~44.6 MB
   redundant activation traffic to 6–12% of weight traffic.
3. **LDS-stage the packed activation tile.** For M≤16 the entire A operand is
   ≤40 KiB; load once per workgroup and read from LDS in the K loop. Frees the
   global load pipeline for weights exclusively.
4. **Wider loads and K-unroll.** Move to 16-byte (`dwordx4`) loads consuming two
   k-tiles, unrolled 4–8 deep so many requests are in flight per wave. Note the
   `lane16 = lane & 15` duplication is inherent to the w32 WMMA ABI and is not
   itself the bug — shallow MLP is.
5. **Preserve the declared numerics exactly.** I32 zero-point correction, FP32
   scales, BF16 gate/up publication before SiLU, and the registered strict
   qmicro fallback are unchanged. This is a scheduling/blocking rewrite, not an
   arithmetic change; the CPU-reference oracles in
   `hipengine/kernels/cpu_reference/iu4_s4.py` must pass untouched.

*Accept:* candidate reaches **>=85% of its binding roof** at M≤16 (i.e. >=188
GB/s, expected ~0.42 ms) **and** materially raises executed TOPS at M=128 from
the current 12.54. Existing `tests/test_iu4_s4_sidecar.py` green with no oracle
edits. Deterministic-bits repeat preserved. Fresh rocprof kernel-trace showing
the expected VGPR/LDS profile.

*Anti-goal:* do not tune against layer 0 or the fixed R2 shape list; the kernel
must be shape-general. No hardcoded M, N, or layer constants.

**Result.** The implementation is shape-general through M1024 and preserves the
unchanged CPU oracles. The best schedule differs from the literal first design:
whole/chunked activation LDS staging and 64–256-column tiny-M WMMA reduced
block-level memory parallelism and measured ~0.59–0.98 ms, so tiny M uses the
already-screened DOT8 arithmetic lane solely to finish B4. Wide M uses paired-K
16-byte loads, 16 accumulators/wave, 32 output columns, 256 rows/block, and an
8-KiB shared S4 slab. The final cache-only trace is DOT8 local128/VGPR48/80/144,
LDS512/scratch0 and wide WMMA local128/VGPR192/LDS8192/scratch32, with no
low-VGPR anomaly. M128 executed arithmetic rises 5.13× over R2. Existing
CPU/GPU nibble, correction, BF16-boundary, small/bulk operation, determinism,
and fallback tests pass without oracle edits.

### R7 — dense prefill gate/up A/B — speed screen complete

The §4.4 step-1 experiment, now with a real control. Compare the R6 kernel
against B3's measured `pack8_dual_wmma_prefill_silu_bf16` at M=64/128/256/512,
operation-complete on both sides (IU4 side includes U4 activation packing,
correction, scales, BF16 publication, SiLU).

*Accept:* a compact artifact with per-shape speedup, paired wins, both sides'
percent-of-roof, and the family Amdahl share of a complete prefill pass. A win
here is the first genuine evidence that the 2× lane is reachable in hipEngine.

*Caveat to carry:* the S4 companion is still re-quantized from the dequantized
Q4_K_S view (15.9% weight NRMSE). R7 is a **speed screen only**. Any retention
requires the R3 numerical packet and an original-weight/offline-optimized S4
product per §8.

**Result.** Passes the mechanical speed screen at M64/128/256/512 by
**2.668×/2.708×/2.707×/3.105×**, with 15/15 candidate wins at every shape.
Both sides are operation-complete and cold-pool sized. The inferred pp512
family projection is 1.380×, but it is not a complete-model measurement.
Artifact:
[`2026-08-23-gfx1151-qwen38-iu4-s4-r7-prefill-gate-up-ab.json`](benchmarks/results/2026-08-23-gfx1151-qwen38-iu4-s4-r7-prefill-gate-up-ab.json).

### R8 — original-product full prefill FFN — complete, quality-traded

R8 uses the immutable published Kairic FFN product instead of extending the
15.9%-NRMSE re-Q4 proxy. `iu4_s4_kairic_ffn_v1` fail-closes SHA-256
`adcbb90a…ba2b`, `PFSIU4F` header/table/file bytes, all 384 role/dtype/shape/
offset records, and the release's block-Hadamard seeds. The explicit session
selector is `iu4_ffn_pfs_path`; gfx1151 Qwen3.8-27B H5120/I17408/L64 admits
Q4_K_S or the coherent Kairic authority, and only physical M96–M2048 uses the
product. Every miss, including c1 decode and canonical short category prompts,
stays on the model's strict owner. The authority adapter expands ROCmFP4 exactly
to BF16 and ROCmFP6 exactly to F32 (49.579 GiB); it is a matched denominator,
not compressed-native execution.

The layer-0 Q5-down full-FFN leaf is operation-complete on both sides:

| M | Current Q4 gate/up + Q5 down | PFS IU4 full FFN | Speedup | Paired wins | Output top-1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 5.833 ms | 2.239 ms | **2.605×** | 15/15 | 100% |
| 128 | 5.986 ms | 2.291 ms | **2.613×** | 15/15 | 100% |
| 256 | 8.508 ms | 2.541 ms | **3.348×** | 15/15 | 100% |
| 512 | 15.882 ms | 4.813 ms | **3.300×** | 15/15 | 100% |

Layer-local hidden-vector KL is diagnostic but bounded (mean 0.000292–0.000334,
max 0.00140–0.00352). Cache-only tracing confirms Hadamard packs at local256,
VGPR48/120, LDS4096, scratch0 and IU4 gate/down at local128, VGPR216, LDS8192,
scratch0; raw trace SHA-256 is `273e22bf…90a7`.

The same-resident complete-model p512/p1K/p4K screen moves
**384.70/384.58/359.25 → 567.92/563.05/518.07 tok/s
(1.476×/1.464×/1.442×)**, all 5/5 pairs. Candidate route counts are 64/64/128
full-FFN layer launches and teardown is exact. The full 8,576,827,392-byte
execution payload loads in 47.5 seconds on this host.

Promotion nevertheless **fails** against the available Q4_K_S authority. The
varied p512 prefill flips top-1 with KL 0.08754; its nine-row strict-teacher
trajectory is mean/max KL **0.02119/0.08754** and **88.89%** top-1, failing the
frozen 0.001/0.05/99% envelope. The full canonical category suite padded to
p512 is much stronger — 50 teacher rows, **100% top-1**, p95/p99/max KL
**0.004255/0.015595/0.015673** — but mean KL is **0.001124**, narrowly above
the 0.001 gate. Repeated 512/1K/4K rows stay exact-top1 and low KL, which
demonstrates why a sole repeated-token gate would have been unsafe. All ten
canonical short prompts are exact through explicit M<96 fallback.

The bounded 4K retrieval/multihop/aggregation/long-doc/code task suite passes
**5/5 control and 5/5 candidate**, with candidate wall lower on every row;
free-running IDs/output lengths differ and the task packet is non-inferiority,
not arithmetic certification. BF16-relative evidence remains unavailable.

The public Kairic authoritative GGUF supplies the exact matched denominator:
hipEngine parses CIRU `Q4_0_ROCMFP4` ID 100 and `Q6_0_ROCMFPX` ID 102 and
materializes lossless expanded fallbacks. That matched packet also rejects the
PFS. Its varied-p512 nine-row teacher trajectory is mean/max KL
**0.08159/0.28923** and **77.78%** top-1. The canonical ten-prompt/five-step
padded suite remains 100% top-1 but records mean/p95/p99/max KL
**0.003768/0.01761/0.05677/0.06360**, failing mean, tails, and the absolute
ceiling. Canonical M40–71 prompts fall back exactly, 10/10 through four tokens.
The candidate nevertheless moves matched-authority p512/p1K/p4K prefill
**29.57/31.70/33.69 → 70.27/72.72/75.67 tok/s
(2.377×/2.294×/2.246×)**, all 5/5 pairs, with exact teardown. Speed cannot
compensate for the binding correctness failure.

**Disposition:** keep the explicit product/runtime and strict fallback as a
quality-traded research configuration; do not expose it as `production`,
omitted/default, or decode behavior. Artifact set:
[`full-FFN leaf`](benchmarks/results/2026-08-23-gfx1151-qwen38-kairic-pfs-iu4-r8-full-ffn-q5-leaf.json),
[`Q4_K_S rejected distribution + complete-prefill gate`](benchmarks/results/2026-08-23-gfx1151-qwen38-kairic-pfs-iu4-r8-full-model-gate.json),
[`accepted Q4_K_S long-task gate`](benchmarks/results/2026-08-23-gfx1151-qwen38-kairic-pfs-iu4-r8-long-task-gate.json), and
[`matched-authority rejection`](benchmarks/results/2026-08-23-gfx1151-qwen38-kairic-authority-pfs-iu4-full-model-gate.json).

### R9 — exact current-B3 verifier tax — next priority

Before another representation change, close the attribution gap on the exact
Q4_K_S B3 owner. The retained direct suite measures **75.343 ms** per true-AR
forward and **111.071 ms** per B3 target pass (**1.474×**), but the often-quoted
21.723-ms "non-kernel gap" subtracts R1's **profiled** 89.348-ms kernel sum from
the **unprofiled** target wall. R1's profiled wall is 148.655 ms, and B3 already
submits one reusable native graph containing about 1,201 kernel nodes. The
21.723 ms is an inference, not a measured Python-launch bucket.

Ordered work:

1. **Matched attribution.** Capture AR/B1/B3 in one final child and one timing
   environment. Separate selected-region wall, graph submit/replay, device
   kernel spans, synchronization/readback, graph-node bubbles, accept/commit,
   and scheduler residual. Report physical M and bytes per family. Do not mix
   profiled kernel sums with unprofiled host wall.
2. **Exact M4 qmicro gate/up.** Treat R1's measured M2→M4
   **30.162→35.576 ms** and **212.8→180.4 GB/s** as a bounded 5.414-ms target.
   The current M4 body reaches VGPR144 versus VGPR112 at M2. Screen only
   representation-preserving register/row ownership or staging changes that
   retain one weight sweep and are BF16-bit identical to c1 arithmetic. The
   full possible recovery is about 5% of complete B3.

   **This is the highest-value exact item in the campaign and it must run before
   any tiny-M representation is re-screened.** Per the marginal-value table in
   the Verdict, it is **81.4% of the entire measured DOT8 M4 win** at zero bytes
   and zero quality risk; the sidecar's marginal contribution above it is only
   **1.238 ms (1.11% of a B3 cycle)**. Running the representation screen first
   credits the sidecar with this schedule deficit and has already produced one
   over-stated ~6% figure. Record VGPR/occupancy and effective GB/s per M so the
   deficit's mechanism (register pressure at VGPR144, row-ownership, or staging)
   is named rather than inferred.
3. **Remaining exact families and graph nodes.** In the same R1 trace B1→B3
   total kernel growth is 9.368 ms; after gate/up, the directly measured
   remainder is only **3.954 ms**, not the residual 8.6-ms estimate. Audit Q4
   projections, Q5 down, GDN/state boundaries, copies, and exact Qwen3.6 N1R
   fusions missing from Qwen3.8. Prefer exact fusion/node removal over another
   sidecar. Skip Q6 root initially: `output.weight` is about **1.043 GB** and
   4.757 ms implies roughly **219 GB/s**, already at the practical roof.
4. **Full exact economics gate.** Every retained candidate reruns true AR and
   B1/B2/B3 together over the complete train/heldout/category suite with exact
   IDs, GPU/CPU acceptance, transaction/KV/state ownership, deterministic
   repeats, memory, and teardown. Public `auto` remains AR until its independent
   real-world qualification passes.

### R10 — true packed/concurrent IU4 verifier

The old claim that current c8 falls to prefill WMMA is stale: retained gfx1151
rowtile work now owns rows2–8 and larger widths decompose into <=8-row chunks.
That still is not an honest wide-M IU4 control because it rereads weights across
chunks. Build and qualify one exact single-sweep native control first, record
actual verifier physical-M histograms, then compare operation-complete
U4/S4 pack+projection+correction+boundary work. R2's historical M16–128 rows and
prefill R7 do not substitute for this control.

**Target M32/64/128, not M16.** "M16+" understates the crossover and would point
the screen at a shape where the answer is predetermined. §3's own weight-only
table, at one sweep and the 221-GB/s roof:

| Physical M | Memory roof | DOT4 roof | Binding constraint |
| ---: | ---: | ---: | --- |
| 16 | **14.14 TOPS** | 28.25 TOPS | **memory** — both arithmetic lanes are 2–8× above it |
| 32 | 28.29 TOPS | 28.25 TOPS | DOT4 crossover |
| 64 | 56.58 TOPS | 28.25 TOPS | arithmetic in play; IU4 can matter |
| 124 | 109.62 TOPS | 28.25 TOPS | full 2× IU4 lane reachable |

At M16 a single-sweep verifier is **still purely memory-bound**, so IU4 buys only
the 1.12× byte ratio there — the same non-result as tiny M. The IU4 *arithmetic*
lane cannot contribute until roughly **M>=32**, and the full 2× not until
**M≈124**. Since the chunked owner rereads weights across chunks, its effective
OI is lower still and memory binds even harder. Scope the control accordingly and
state the concurrency this implies: a B3 verifier reaches M32 only at about
**c8**, and M64+ at **c16+**. If the measured physical-M histogram never reaches
M32 under realistic concurrency, R10 closes negative without a kernel being
written — establish the histogram first.

### R11 — exact expanded prefill control

Materialize the current Q4_K_S dequantized FFN values in the same F16 boundary
the existing pack8 owner feeds to WMMA, then compare no-dequant dense F16
against current pack8/F16 and IU4 at M64/128/256/512. This isolates dequant
removal from the 2× arithmetic lane. Do not assume a BF16 sidecar is T0: prove
primitive and full-model output bytes under the declared schedule. Report the
roughly 31.9-GiB (34.2-GB) FFN residency and complete-wall/lifecycle result;
retain only an exact or fully profile-qualified explicit product.

### R13 — layer-selective routing to recover the R8 prefill gate

**Added by the 2026-08-23 review 2. This is the highest-expected-value open item
in the campaign** and requires no new weights, no new quantizer, and no change
to the immutable Kairic PFS.

R8 measured a **1.476× complete-model p512 prefill** (384.70→567.92 tok/s) and
then failed promotion. But the failure is narrow and non-uniform, and nobody has
localized it:

| Packet | top-1 | mean KL | max KL | Against gate |
| --- | ---: | ---: | ---: | --- |
| Canonical category suite, 50 teacher rows | **100%** | **0.001124** | 0.015673 | fails mean by **12.4%**; tails pass comfortably |
| Varied p512, 9-row trajectory | 88.89% | 0.02119 | 0.08754 | fails everything |

The **5.6× max-KL spread** between those two packets is the finding. Diffuse
quantization noise does not behave that way; concentrated error does. The
canonical packet is *one metric, 12% over, with passing tails* — that is a
routing-policy distance from the gate, not a representation dead end.

Standard mitigations, none of which have been tried here, and all of which
operate on the existing product by **routing fewer layers to it**:

1. **Per-layer sensitivity sweep.** Measure teacher KL with exactly one layer on
   IU4 and the rest exact, for all 64 layers. Produces a sensitivity ranking, and
   a per-layer KL contribution budget. This is the prerequisite measurement for
   everything below and is cheap — 64 short runs, no new kernels.
2. **Top-k exact retention.** Keep the k most sensitive layers on the strict
   Q4_K_S owner. In 4-bit practice the first and last one to three layers usually
   dominate. Speedup scales roughly as `1 - k/64`, so k=8 retains ~1.40× of the
   measured 1.476×. If the sensitivity ranking is concentrated, k=4 may suffice.
3. **Role-selective precision.** `down` is typically the most error-sensitive FFN
   tensor. Route gate/up to IU4 and keep down exact, then measure. R8's leaf
   already separates the Q5 down control, so the comparison is nearly free.

*Accept:* the canonical packet clears mean KL <=0.001 with tails and top-1
unchanged, **and** the varied-p512 trajectory clears its envelope, **and** the
retained speedup is stated with the exact `k` and role policy that produced it.
A policy that passes canonical but not varied p512 is **not** a pass — the
varied-packet failure is the more informative of the two and must be explained,
not routed around.

*Anti-goal:* this is layer/role **routing** policy over an immutable product. It
is explicitly **not** re-tuning weights against the evaluated prompts, and the
selection must be made on a calibration split and confirmed on heldout
categories, or it is benchmark gaming under `AGENTS.md` and non-retainable.

*Relationship to "not scheduled".* This does not reopen "further salvage of the
immutable Kairic PFS" — that entry closes *changing or re-tuning the product*.
R13 changes only which layers the runtime sends to it, which is ordinary
execution-profile routing and already has a registered strict fallback.

### Priority note (2026-08-23 review 2)

R9 is currently labelled "next priority". Weigh that against measured ceilings:

| Item | Status | Ceiling | Distance to landing |
| --- | --- | ---: | --- |
| **R13** prefill layer-selective | quality-blocked | **~1.40–1.48× prefill** | one metric, 12% over |
| R9 exact verifier tax | unmeasured | ~5% of B3 decode | full attribution campaign |
| R10 packed verifier | unscoped | 0 below M32 | needs histogram first |
| R12 new S4 product | not started | ~1.1% marginal at tiny M | full new campaign |

R13 is worth roughly an order of magnitude more than R9 and is closer to done.
R9 item 2 remains mandatory before any tiny-M representation screen, but that is
a *sequencing* constraint on R12, not a reason to rank the whole of R9 above
R13. **The correct read of R8 is that the prefill thesis was validated and only
the routing policy is missing.**

The one input that should reorder this is workload mix: these ceilings assume
prefill and decode both matter. If the target deployment is decode-dominated
chat, R9's 5% applies to the dominant phase and R13's 1.4× to the minor one.
State the assumed mix when picking, rather than leaving it implicit.

### R12 — new original-weight product only after R9/R11

Do not rerun or tune around the immutable Kairic PFS's failed distribution
packet. A new original-weight S4 product may reopen tiny-M MTP only if it has an
immutable manifest, M2–M4 kernels, strict fallback, a model-quality oracle, and
a predeclared complete B3 T3 gate. Compare its projected/measured gain against
the exact R9 owner and charge all resident bytes. No current sidecar becomes a
runtime/default MTP route.

### Explicitly not scheduled

- **K4/V8 attention leaf.** Deprioritized: a bytes/capacity thesis, not a TOPS
  thesis, and current KV targets are already met on both a 128-GiB unified host
  and the W7900's 256K INT8 reach. Revisit only under measured capacity or
  long-context bandwidth pressure.
- **Current re-Q4 tiny-M runtime integration.** Closed by M2 loss,
  representation economics, and absent model-quality qualification. **Decisively
  closed by marginal value**: 81.4% of the measured 1.230× M4 DOT8 win is a
  bandwidth deficit in the exact owner that R9 item 2 can recover for free, so
  the representation's own contribution is **1.238 ms = 1.11% of a B3 cycle**.
  Keep B4 as the mechanical leaf; do not route it. Only the new-product R12
  gate can reopen the decision, and only after R9 item 2 has established the
  corrected exact baseline.
- **Further salvage of the immutable Kairic PFS.** Closed by both binding
  full-model distribution failures. A new product revision is a new campaign.
- **Direct Q4_K/qmicro nibble reuse (§6.2).** Remains a useful ceiling
  experiment, but adds groupwise correction and metadata traffic to a surface
  that is already bandwidth-bound at tiny M and dequant-bound at prefill — where
  a clean S4 companion is strictly better.

## References

- Instruction diagnostic:
  [`benchmarks/results/2026-08-23-gfx1151-mtp-iu4-instruction-roofline.json`](benchmarks/results/2026-08-23-gfx1151-mtp-iu4-instruction-roofline.json)
- Current exact verifier attribution:
  [`benchmarks/results/2026-08-23-gfx1151-qwen38-mtp-native-verifier-attribution.json`](benchmarks/results/2026-08-23-gfx1151-qwen38-mtp-native-verifier-attribution.json)
- Operation-complete R2 leaf:
  [`benchmarks/results/2026-08-23-gfx1151-qwen38-iu4-s4-gate-up-leaf.json`](benchmarks/results/2026-08-23-gfx1151-qwen38-iu4-s4-gate-up-leaf.json)
- B3 current prefill control:
  [`benchmarks/results/2026-08-23-gfx1151-qwen38-pack8-gate-up-prefill-control.json`](benchmarks/results/2026-08-23-gfx1151-qwen38-pack8-gate-up-prefill-control.json)
- R6/B4 corrected core and tiny-M result:
  [`benchmarks/results/2026-08-23-gfx1151-qwen38-iu4-s4-r6-b4-gate-up-leaf.json`](benchmarks/results/2026-08-23-gfx1151-qwen38-iu4-s4-r6-b4-gate-up-leaf.json)
- R7 dense-prefill A/B:
  [`benchmarks/results/2026-08-23-gfx1151-qwen38-iu4-s4-r7-prefill-gate-up-ab.json`](benchmarks/results/2026-08-23-gfx1151-qwen38-iu4-s4-r7-prefill-gate-up-ab.json)
- R8 original-product full-FFN leaf:
  [`benchmarks/results/2026-08-23-gfx1151-qwen38-kairic-pfs-iu4-r8-full-ffn-q5-leaf.json`](benchmarks/results/2026-08-23-gfx1151-qwen38-kairic-pfs-iu4-r8-full-ffn-q5-leaf.json)
- R8 full-model distribution/performance gate:
  [`benchmarks/results/2026-08-23-gfx1151-qwen38-kairic-pfs-iu4-r8-full-model-gate.json`](benchmarks/results/2026-08-23-gfx1151-qwen38-kairic-pfs-iu4-r8-full-model-gate.json)
- R8 long-task gate:
  [`benchmarks/results/2026-08-23-gfx1151-qwen38-kairic-pfs-iu4-r8-long-task-gate.json`](benchmarks/results/2026-08-23-gfx1151-qwen38-kairic-pfs-iu4-r8-long-task-gate.json)
- Current exact B3:
  [`benchmarks/results/2026-08-17-gfx1151-qwen38-27b-q4ks-exact-native-b3.json`](benchmarks/results/2026-08-17-gfx1151-qwen38-27b-q4ks-exact-native-b3.json)
- gfx1151 roofline: [`docs/ROOFLINE-gfx1151.md`](docs/ROOFLINE-gfx1151.md)
- Kernel catalog: [`docs/KERNELS.md`](docs/KERNELS.md)
- Testing contracts: [`docs/TESTING.md`](docs/TESTING.md)
- Execution profiles: [`docs/EXECUTION-PROFILES.md`](docs/EXECUTION-PROFILES.md)
- Kairic model card:
  <https://huggingface.co/jcbtc/Qwen3.8-27B-IU4-Kairic-Edge>
- Kairic immutable source:
  <https://github.com/ciru-ai/ROCmFPX/tree/kairic-edge-qwen38-27b-v1>
- Kairic CK patch at the inspected release:
  `patches/composable-kernel-gfx1151-iu4.patch`
- Clang AMDGPU builtin reference:
  <https://clang.llvm.org/docs/AMDGPUBuiltinReference.html>
