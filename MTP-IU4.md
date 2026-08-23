# MTP IU4 verifier research plan

_Status: instruction screen and current-verifier attribution complete; no runtime IU4 route or sidecar is implemented_

_Branch: `mpt-iu4`_

_Last updated: 2026-08-23_

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

**Decision:** continue the research, but do not allocate a full 5.33–7.99 GiB
sidecar yet. The next gate is a one-layer, operation-complete, actual-weight
`M=2..5` experiment that includes packing, IU4 core, correction, BF16
publication, and the gate/up SiLU boundary. Native IU4 becomes a stronger
candidate for packed concurrent verifier rows (`V>=16`, preferably `V>=32`) and
for `M>=96` prompt/verification batches than for single-request B3.

## 1. Scope and non-goals

The initial target is Qwen3.8-27B Q4_K_S on `hip_gfx1151`:

- physical target rows `M=B+1`, especially `M=2,3,4,5`;
- H=5,120 and FFN=17,408;
- FFN gate/up first, then down only if gate/up survives;
- the existing Q4_K_S model, BF16 KV, MTP proposal, attention/state, acceptance,
  transaction, and commit machinery remain authoritative;
- current exact qmicro Q8_1×Q4 DOT4/DP4A owners remain the strict fallback.

This document does **not** claim:

- an operation-complete IU4 kernel win;
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
profiled 148.655-ms target window (1.070×). Applying the leaf's more conservative
1.14× ceiling saves only 4.37 ms (about 2.9%).

## 4. What Kairic establishes—and what it does not

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

## 5. Representation and correction math

### 5.1 Clean S4 companion

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

### 5.2 Direct use of current Q4_K/qmicro nibbles

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

## 6. Sidecar memory model

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

## 7. Proposed architecture

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

## 8. Execution-profile classification

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

## 9. Phased experiment plan

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

### R2 — one-layer operation-complete leaf

Build only layer 0 (then 8 and 63) gate/up companions:

1. offline S4 pack from an explicitly named source (prefer original BF16/F16
   weights; re-quantizing Q4_K_S must be labeled as such);
2. dynamic asymmetric U4 pack for `M=2,3,4,5,8,16,32,64,96,128`;
3. direct `__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32` core;
4. I32 correction plus FP32 scale;
5. gate/up BF16 publication and SiLU; and
6. exact current owner as control/fallback.

Use actual 100+ MiB cold-weight pools, counterbalanced order, at least 15 timed
samples, and include pack+core+correction+SiLU. Report useful TOPS and effective
GB/s separately. A core-only win cannot pass R2.

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

## 10. Go/no-go gates

| Gate | Pass condition | Current status |
| --- | --- | --- |
| Native instruction | Expected IU4 ISA; >=1.8× matched IU8/FP16 median; stable samples | **Pass**: 1.98–2.00× |
| Tiny-M useful arithmetic | Account for `M/16` tile utilization, not raw TOPS | **Warning**: M4 ≈ DOT4 roof |
| Actual-weight gate/up | Operation-complete M2–5 beats current owner with pack included and passes declared leaf numerics | Not run |
| Family attribution | Measured all-layer gate/up share and effective BW justify expected target-wall delta | **Pass, bounded**: 21.9–25.2% wall; ideal zero-pack S4 ceiling 1.032–1.070× |
| Memory ROI | Measured complete target/cycle gain justifies +5.329 GiB gate/up (or +7.988 GiB FFN) | Not run |
| T3 quality | Full strict-teacher/determinism/isolation/BF16-relative/task packet passes | Not run |
| MTP economics | Full category suite beats same-protocol true AR and current B3 without category regression | Not run |
| Lifecycle | Sidecar load/fallback/close reaches zero tracked allocation and bounded process memory | Not run |

A failed tiny-M gate does not invalidate IU4 generally. It redirects the lane to
`M>=32` packed verification and `M>=96` prompt work, where the roofline and
Kairic evidence are much stronger.

## 11. Immediate next command sequence

```bash
# 1. Re-run/inspect the instruction screen.
python3 -m pytest -q tests/test_mtp_iu4_roofline.py
HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/mtp_iu4_roofline.py \
  --iterations 65536 --samples 9 --warmups 3 \
  --output /tmp/mtp-iu4-roofline.json

# 2. Current exact target attribution is retained in the R1 artifact.
# It used the final qwen36_dense_gguf_suite.py child with ROCTX markers,
# cached builds, native B1-B3, and the required FP32-state rollback.

# 3. Next: build one layer-0 gate/up S4 companion and measure M=2..128 with
# actual weights, inclusive activation pack/correction/SiLU, before allocating
# all 64.
```

The current lineage command is mechanically blocked by the already documented
missing `/home/lhl/amd-gpu-tuning/reference/atlas` checkout. No external kernel
was ported in R0; Kairic/Clang were used only as read-only references.

## References

- Instruction diagnostic:
  [`benchmarks/results/2026-08-23-gfx1151-mtp-iu4-instruction-roofline.json`](benchmarks/results/2026-08-23-gfx1151-mtp-iu4-instruction-roofline.json)
- Current exact verifier attribution:
  [`benchmarks/results/2026-08-23-gfx1151-qwen38-mtp-native-verifier-attribution.json`](benchmarks/results/2026-08-23-gfx1151-qwen38-mtp-native-verifier-attribution.json)
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
