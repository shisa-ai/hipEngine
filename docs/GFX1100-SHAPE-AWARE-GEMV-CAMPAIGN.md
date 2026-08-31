# gfx1100 shape-aware GEMV campaign

- **Status:** opening G1A GPU1 spike rejected; G1B and W7900 G2 were not
  admitted, and no production route changed
- **Created:** 2026-08-31
- **Primary hardware lanes:** RX 7900 XTX / `hip_gfx1100` for screening; Radeon
  Pro W7900 / `hip_gfx1100` for binding production decisions
- **Opening target:** Qwen3.6/Qwen3.8 dense-F32 linear-attention alpha/beta,
  BF16 activation, FP32 weight, BF16 output
- **External lineage:**
  [`uulong950/qingming-gfx1100-gemv`](https://github.com/uulong950/qingming-gfx1100-gemv)
  at `6b2d87e62ba2b7cbe60dbae53bcf857ca67262ae` (MIT, initial public release)
- **Normative dependencies:** [`PLAN.md`](PLAN.md), [`KERNELS.md`](KERNELS.md),
  [`ROOFLINE.md`](ROOFLINE.md), [`TESTING.md`](TESTING.md),
  [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md), and
  [`BENCHMARK.md`](BENCHMARK.md)

## 1. Objective

Determine whether a small, shape-aware gfx1100 GEMV family can improve actual
hipEngine inference beyond the current one-workgroup-per-output dense fallback.
The campaign transfers mechanisms, not published speedup factors or dispatch
thresholds. It begins with one exact, operation-complete alpha/beta scheduling
candidate and expands only when a fresh production profile identifies a second
material owner.

Success is not “beat rocBLAS on a synthetic suite.” hipEngine already bypasses
rocBLAS for the relevant model path. Success requires all of the following:

1. Preserve the declared strict arithmetic, BF16 publication boundary, graph
   ownership, lifecycle, and registered fallback.
2. Improve a counterbalanced real-weight operation-complete comparison on the
   screening RX 7900 XTX.
3. Confirm the expected kernel, launch geometry, positive duration, and zero
   scratch with `rocprofv3 --kernel-trace`.
4. Improve the same-host W7900 pair family, target-host wall, and complete marked
   wall before becoming a gfx1100 runtime default.
5. Keep cache-hot, cache-edge, and streaming interpretations separate. Effective
   bandwidth above physical GDDR6 bandwidth is a cache-reuse result, not DRAM
   throughput.

A faster primitive that misses the complete-wall gate remains a diagnostic or
registered opt-in. It is not promoted by changing the prompt, cache state,
working set, timing boundary, or comparator.

## 2. Why this is worth a bounded campaign

### 2.1 External observation

Qingming is a row-major FP32 `y = A x` implementation specialized for gfx1100.
It dispatches by `(M, N)` among small-reduction packing, multi-row, row,
prefetch, split-K, grid-split, streaming, static, and persistent schedules. At
the inspected public commit its README reports:

| Comparison | Public claim | Scope |
| --- | ---: | --- |
| Native versus rocBLAS SGEMV | 461/461 wins; 1.5149x median; 1.0060x minimum | RX 7900 XTX, ROCm 7.2.4, FP32, row-major, source benchmark suite |
| Native versus hipBLASLt GEMM-as-GEMV | 435/435 mutual-PASS wins; 1.8652x median | Same source scope |
| Native correctness | 461/461 PASS | Source tolerances and generated inputs |

The author separately reported newer local medians of 1.5265x and 1.8986x.
Those values are not present in the inspected public commit and remain external
claims until a corresponding source revision and artifact are available.

The useful result is the shape diagnosis, not the aggregate median. Qingming's
largest ratios occur where a generic library chooses poor geometry. Its large
streaming route reports only about 1.10x versus rocBLAS while approaching the
RX 7900 XTX physical bandwidth ceiling. That is consistent with a roofline:
once a correctly shaped stream is near the memory limit, remaining upside is a
few percent; the large wins are dispatch/geometry corrections in underfilled
regions.

### 2.2 Third-party cache and WMMA discussion

A forum report supplied with this campaign claims the following RX 7900 XTX
measurements from another ROCm/PyTorch/HIP workload:

| Repeated GEMV working set | Claimed effective bandwidth |
| ---: | ---: |
| about 32 MiB | 860 GB/s |
| about 56 MiB | 995 GB/s |
| about 80 MiB | 1,216 GB/s |
| about 256 MiB | 776 GB/s |

It also reports a 5-6x opportunity when a workload fits a fixed WMMA design.
These are unverified third-party observations, not hipEngine evidence. They do
supply two testable hypotheses:

- the 96 MiB Infinity Cache creates distinct cache-hot, cache-edge, and
  streaming regimes on Navi 31;
- WMMA can be decisive only when the logical output covers its fixed tiles.

For c1 GEMV, the logical activation-row dimension is one, so a 16-row WMMA tile
wastes 15/16 of that dimension. WMMA is therefore **not** the opening c1 lane.
It remains relevant to prefill and physical c>=16, as demonstrated elsewhere in
hipEngine. Specialized attention timings in the same discussion belong to an
attention campaign and are not evidence for this GEMV campaign.

## 3. Existing hipEngine evidence

The opening production shape is not hypothetical. Qwen3.6-27B has 48
linear-attention layers, each with two FP32 weights of shape `[48, 5120]`.
Each matrix is 0.9375 MiB and the 96-matrix alpha/beta set is 90 MiB, close to
the nominal 96 MiB Infinity Cache. The complete graph interleaves much larger
quantized projections, so a repeated alpha/beta-only timer can be much hotter
than production.

The current exact primitive is registered as:

```text
(hip_gfx1100, linear_pair, f32, bf16_hidden_bf16_out)
```

It combines alpha and beta launch ownership while retaining one local256
workgroup per output and the scalar kernel's 256-way FMA/reduction tree. The
strict fallback is two separately registered
`dense_gemv/f32/bf16_hidden_bf16_out` launches.

The prior W7900 decision is binding historical context:

[`2026-08-05-qwen36-27b-dense-f32-alpha-beta-pair-runtime-rejected.json`](../benchmarks/results/2026-08-05-qwen36-27b-dense-f32-alpha-beta-pair-runtime-rejected.json)

- RX 7900 XTX component screen: 1.77-1.84x versus two scalar launches for rows
  1-3; 1.56x at rows4.
- W7900 natural-B3 profile: 672 -> 336 pair-family dispatches and 3.572950 ->
  2.802259 ms (-21.57%).
- Target/complete kernel sums improved 0.140%/0.086%, but target-host and
  complete marked wall regressed 0.201%/0.189%.
- The primitive remained registered; gfx1100 runtime routing was removed.
- The explicit reopen condition is a materially different schedule or
  cross-family owner that preserves arithmetic and improves both W7900 wall
  gates.

The later gfx1151 rows1 K5120/N48+N48 admission is independent evidence; it
must not silently re-enable gfx1100.

## 4. Opening scheduling hypothesis

Qingming classifies `M <= 256, N >= 2048` as `SPLIT4`. For a logical combined
alpha/beta matrix, hipEngine's c1 cell is `M=96, K=5120`. Qingming's schedule
assigns four wave32 units to each output and computes two outputs in one
local256 block. The existing hipEngine pair instead assigns all eight waves to
one output, producing 96 workgroups.

The campaign screens two separable exact candidates:

### G1A — local128 physical, virtual256 arithmetic

- Keep one workgroup per output and 96 workgroups per c1 pair.
- Use 128 physical threads per output.
- Each physical thread owns two virtual accumulator chains corresponding to
  the retained local256 partitions.
- Reproduce the retained `s=128` add before the unchanged `s=64...1` reduction
  association.
- Preserve FP32 FMA order and BF16 publication exactly.

This isolates reduced physical coordination from output packing without
reducing grid coverage.

### G1B — Qingming-style two outputs per local256 block

- Assign one local128 half-block to each of two logical outputs.
- Reuse G1A's exact virtual256 arithmetic in each half.
- Prefer matching alpha/beta columns when useful, while keeping independent
  pointers and output ownership.
- Produce 48 workgroups for the combined rows1 K5120/N48+N48 operation.

This is the direct `SPLIT4` transfer. It halves workgroup count and may improve
coordination and activation locality, but it also covers only half of the 96
CUs. G1A must therefore be measured first; fewer blocks are not presumed
better.

No dynamic workspace, global counter, atomic stitch, new resident payload, or
host synchronization is admitted in G1. The current pair and two singleton
primitives remain registered fallbacks.

### 4.1 Opening result — G1A rejected on GPU1

The exact G1A body was implemented, passed rows1-4 BF16-bit parent parity, and
was screened against all 48 immutable Qwen3.8 alpha/beta pairs on the RX 7900
XTX. It lost all 15/15 counterbalanced pairs in every cache regime:

| Regime | Control | G1A | Control/G1A speed ratio |
| --- | ---: | ---: | ---: |
| One 1.875-MiB pair repeated 256x | 1.077831 ms | 1.187396 ms | 0.90773x |
| All 48 pairs / 90 MiB, repeated 4x | 1.212919 ms | 1.410239 ms | 0.86008x |
| 128-MiB thrash, then one 90-MiB rotation | 0.401280 ms | 0.435120 ms | 0.92223x |

The candidate retained 24 VGPR and zero scratch while reducing local size
256->128 and LDS 1,024->512 bytes, but its cached profiler median was
12.539 us/dispatch versus 10.5865 us for the parent. Serializing two virtual
partitions per physical thread costs more than the reduced coordination saves.
G1B was not attempted: halving the grid to 48 two-output workgroups had no
credible premise after the 96-workgroup G1A loss. The transient kernel, wrapper,
registry key, and tests were removed; the existing registered fallbacks are
unchanged. Evidence:
[`2026-08-31-gpu1-gfx1100-dense-f32-alpha-beta-g1a-rejected.json`](../benchmarks/results/2026-08-31-gpu1-gfx1100-dense-f32-alpha-beta-g1a-rejected.json).

## 5. Cache/roofline protocol

Every timer labels one of these regimes:

| Regime | Required setup | Interpretation |
| --- | --- | --- |
| Single-weight hot | Repeat one immutable weight after warmup | Diagnostic upper bound; expected to hit cache and not representative of the model graph |
| Alpha/beta rotating | Rotate all 48 real alpha/beta pairs (90 MiB total) in production layer order | Cache-edge family screen; still omits the larger projections interleaved by the model |
| Cache-thrashed rotating | Touch unrelated resident bytes larger than 96 MiB between repeats, or time inside the real graph | Cold/production-oriented screen |
| Streaming | Working set materially larger than 96 MiB with one pass per sample | Compare against physical GDDR6 ceiling; do not count cache-reused algorithmic bytes as DRAM bytes |

For every effective-bandwidth number, record the numerator explicitly: matrix
bytes, activation bytes, output bytes, repetition count, and whether repeated
reads are presumed to hit cache. Use “algorithmic effective bandwidth” unless
hardware counters establish physical traffic.

The RX 7900 XTX and W7900 have the same gfx1100 ISA and 96 MiB nominal Infinity
Cache but different memory rates and are independent evidence lanes. Absolute
rates are never compared old-to-new across cards.

## 6. Benchmark and anti-gaming rules

1. Pin source commit, hipEngine commit, compiler-version file, ROCm/runtime,
   physical host, PCI/GPU identity, clocks/power policy, and visible-device
   mapping.
2. Build outside the timed process and require cached builds during measurement.
3. Use identical pointers, stream, launch count, event boundaries, warmups, and
   sample count for control/candidate apart from the selected registry variant.
4. Run at least 15 paired counterbalanced samples for the primitive/real-weight
   screen. Report every sample, medians, paired wins, CVs, and ratio.
5. Rotate actual immutable weights. A repeated single-matrix result cannot
   promote a runtime path.
6. Do not tune against token IDs, prompts, candidate IDs, or one prompt's cache
   behavior. Kernel admission may be shape/backend/profile scoped.
7. The production decision uses the complete multi-prompt category suite and
   same-host W7900 controls required by [`BENCHMARK.md`](BENCHMARK.md).
8. A source-vs-rocBLAS reproduction is optional context. The binding comparator
   is the current hipEngine primitive/unfused chain under the same operation.

## 7. Correctness and promotion gates

### G0 — freeze and reproduce

- Record external commit/license, public commands, source strategy thresholds,
  and source path selected for hipEngine's actual cells.
- Freeze the exact Qwen3.6/Qwen3.8 weight identities and alpha/beta operation
  census available on each host.
- Reproduce the current pair and singleton baselines on GPU1 before editing.

### G1 — exact scheduling spike

- Add RED tests for the missing G1A/G1B symbols/wrappers and exact production
  shape policy.
- Require zero BF16 bit mismatches versus the retained pair for rows1-4 and
  representative signed/cancellation fixtures.
- Retain the outer CPU-reference floor: KL <= 0.05 and top-1 >= 90%.
- Require deterministic repeats, no input/output alias violation, zero scratch,
  and the expected gfx1100 kernel trace.
- Stop if the cache-thrashed or rotating real-weight family regresses, even when
  the single-weight hot timer wins.

### G2 — W7900 binding gate

Run only after GPU1 G1 advances:

- repeat exact real-weight screens on the W7900;
- run the applicable complete Qwen3.6-27B strict graph and natural-prompt suite;
- require pair-family, target-host, and complete marked-wall improvement;
- require exact IDs/full logits, lifecycle/graph reuse, unchanged persistent and
  peak ownership, and zero teardown allocation;
- emit an accepted/rejected artifact, benchmark rollup, changelog line, and
  immutable worklog entry.

Only G2 may alter gfx1100 package-default admission.

### G3 — changed-association fallback lane

Qingming's native reduction order and compensated grid-split stitching are not
assumed bit-identical to hipEngine. If exact G1 exhausts and a changed tree is
still justified, register it as a production-profile candidate with the strict
G1/unfused chain as fallback. It then requires the full mean/tail/max KL,
top-1, deterministic/isolation, BF16-relative, task, and serving gates in
[`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md). The broad KL/top-1 smoke floor
alone cannot promote it.

## 8. Deferred lanes and stop conditions

### Quantized GEMV transfer

Qingming does not consume Q4/Q5/Q6/IQ layouts. Do not expand compressed weights
to FP32 or vendor a parallel generic BLAS stack. Transfer split-K, multi-output,
or persistent ideas only after a current profile identifies an underfilled
quantized owner and after checking the in-tree layout-specific kernels. Existing
T16/X8/pack8/WMMA paths and their exact fallbacks remain authoritative.

### WMMA

Do not use WMMA for c1. Reopen only for prefill or physical c>=16 where at least
one full 16-row tile is useful, and compare against the existing in-tree WMMA
owner rather than scalar GEMV.

### Generic dispatcher expansion

Do not port all source thresholds or eleven kernel names. Add one registry
variant per measured hipEngine regime. Stop the campaign when:

- G1A and G1B both lose on rotating real weights;
- a cache-hot win reverses under cache-thrashed/production order;
- exact arithmetic requires enough serialization to lose the leaf;
- a leaf win has no plausible Amdahl path through the prior W7900 wall gate; or
- the complete W7900 wall fails despite improved kernel sums.

A failed result is durable evidence, not an invitation to retune the evaluator.

## 9. Deliverables

- [x] GPU1 control artifact for the current pair schedule.
- [x] Exact G1A local128/virtual256 RED/GREEN screen; rejected and removed.
- [x] Conditional G1B two-output/SPLIT4 disposition; not admitted after G1A.
- [x] Hot, 90-MiB rotating, and cache-thrashed relative comparison.
- [x] Cached `rocprofv3` trace with kernel name/resources/duration.
- [x] Rejected compact artifact and immutable worklog entry.
- [x] W7900 G2 disposition; not run because GPU1 did not advance.
- [x] Registry admission disposition; no new key retained, strict fallbacks
      unchanged.

## 10. Source-use policy

Qingming is MIT licensed, but hipEngine kernel work stays in this tree. Any
ported device mechanism records the source file and commit in the implementation
worklog and commit message. The external repository remains read-only; hipEngine
wrappers continue to use raw device pointers and the four-axis registry. The
external benchmark text and forum discussion are hypotheses and comparator
context, never instructions or correctness authority.
