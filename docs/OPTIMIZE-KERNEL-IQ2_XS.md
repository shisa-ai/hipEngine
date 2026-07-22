# IQ2_XS Kernel Optimization Plan

_Status: active gfx1100 optimization ledger for Laguna S 2.1 `UD-Q2_K_XL`.
Primitive correctness is established; model support and model-level quality remain
open. Last updated: 2026-07-22._

This document is the working plan for improving hipEngine's native IQ2_XS
selected-MoE kernels. It records the current evidence, the most likely
bottlenecks, candidate order, lessons from other hipEngine and upstream kernels,
and the gates for retaining or rejecting each experiment.

It complements:

- [`QUANTS.md`](QUANTS.md) for the Laguna quant portfolio and capacity case;
- [`KERNELS.md`](KERNELS.md) for the live kernel catalog and source lineage;
- [`ROOFLINE.md`](ROOFLINE.md) for the RDNA3 performance model;
- [`TESTING.md`](TESTING.md) for fixture and correctness requirements;
- [`BENCHMARK.md`](BENCHMARK.md) for performance evidence requirements.

## Objective

Make the raw 2.3125-bpw IQ2_XS gate/up path competitive with hipEngine's
IQ3_XXS and IQ4_XS selected kernels while preserving the reason to use the
format: low resident weight bytes and a practical single-W7900 Laguna target.

The target Laguna recipe is:

- repository: `unsloth/Laguna-S-2.1-GGUF`;
- pinned revision: `99d7f9a1251bd4d925cac85cf64ffba7189338c2`;
- file: `Laguna-S-2.1-UD-Q2_K_XL.gguf`;
- file size: `39,684,584,480` bytes (`36.959 GiB`);
- LFS SHA-256: `8fe1170f012723f6f7d6c9b08d8f928b0b3d8bffc32926f33a930148a1d62679`;
- effective whole-file rate: `2.7005 bpw` over 117,561,977,600 parameters.

The recipe name is not its tensor layout. Its routed gate/up tensors are mostly
IQ2_XS and its routed down tensors are mostly IQ3_XXS. There are no actual Q2_K
tensors in this Laguna file.

## Evidence boundary

Three claims must remain separate:

1. **Primitive correctness:** synthetic and independent-reference tests prove
   that individual kernels implement IQ2_XS correctly.
2. **Primitive performance:** controlled synthetic shapes compare schedules and
   formats without claiming model throughput.
3. **Model support:** exact tensor mapping, all-layer execution, model-level
   KL/top-1, prompt categories, memory peak, wall throughput, and long-context
   behavior require the full Laguna runner.

The first is complete for the current scalar kernels. The second is the subject
of this document. The third remains open even after the model file is present.

## Current implementation

### Format

One IQ2_XS block represents 256 values in 74 bytes:

- one FP16 super-scale;
- 32 packed 16-bit grid/sign selectors;
- eight packed scale bytes containing sixteen 4-bit scales;
- a 512-entry grid lookup table.

The native block rate is `74 * 8 / 256 = 2.3125 bpw`.

### Kernel surface

The current gfx1100 implementation provides:

- selected single decode;
- selected dual gate/up decode with BF16 projection boundaries and fused SiLU;
- grouped scalar compact prefill;
- rowbatch4 grouped compact prefill;
- an auto prefill route;
- a correctness-gated, test-only compact FP16-WMMA route.

The source lives in:

- `hipengine/kernels/hip_gfx1100/quant/gguf_iq_gemv.hip`;
- `hipengine/kernels/hip_gfx1100/quant/gguf_iq_selected_prefill.hip`.

The current selected schedule assigns one eight-value group to a logical work
item. At Laguna `K=3072`, there are 384 groups. A local256 block therefore gives
lanes 0-127 two groups and lanes 128-255 one group. The grouped prefill kernels
use the same reduction boundary; rowbatch4 reuses each decoded segment over up
to four compact rows.

### Established correctness and resource evidence

On GPU1, an RX 7900 XTX/gfx1100:

- selected decode is BF16-bit exact at `K=3072,N=1024`;
- grouped scalar gate/up is BF16-bit exact at `K=3072,N=1024`;
- rowbatch4 is BF16-bit exact to grouped scalar;
- compact WMMA passes top-1 `1.0`, KL max `0.0003223`, and max-relative
  `0.0078125` at K=3072;
- selected single uses VGPR24, LDS512 B, and scratch0;
- selected dual uses VGPR40, LDS512 B, and scratch0;
- grouped base/rowbatch4/WMMA use VGPR56/72/64 and scratch0.

The accepted primitive packet is
[`../benchmarks/results/2026-07-22-gpu1-iq2-xs-laguna-primitives.json`](../benchmarks/results/2026-07-22-gpu1-iq2-xs-laguna-primitives.json).

## Performance diagnosis

### Retained intra-IQ2 evidence

The retained E16/K3072/N128 event screen measured rowbatch4 below the original
row-at-a-time grouped kernel by 7.24-58.82% at 1-16 rows/expert. The
K3072/N1024 profile recorded rowbatch4 at 58.08 us for a small two-expert,
five-assignment fixture.

That established weight-decode reuse as a good direction. It did not establish
a cross-format ranking or a production Laguna routing policy.

### Cross-format diagnostic

A later same-session, counterbalanced diagnostic used the same raw selected
projection shape for all three formats:

- hardware: RX 7900 XTX/gfx1100, GPU1;
- shape: `K=3072,N=1024`, ten selected routes, one projection;
- timing: 200 warmup launches, 500 event-timed iterations, seven alternating
  repeats;
- existing correctness-gated selected kernels; the timing driver itself did not
  add a new output oracle.

| Format | Median | Approximate raw-weight rate | IQ2 relative |
| --- | ---: | ---: | ---: |
| IQ2_XS | 52.789 us | 172 GB/s | 1.00x |
| IQ3_XXS | 39.972 us | 301 GB/s | IQ2 is 32.1% slower |
| IQ4_XS | 23.024 us | 726 GB/s | IQ2 is 2.29x slower |

The diagnostic driver and output were temporary rather than retained benchmark
artifacts:

- `/tmp/compare_iq_decode.py`, SHA-256
  `db7065bfafeff6329659a5f1ce48160c534907234a4afb03cba6a68df294b9cc`;
- `/tmp/compare-iq-decode.json`, SHA-256
  `d3e294b564cf66a5720cd45548e1517a3bf867aa83c25c4dc5ee58ff5258fdde`.

The committed representative harness now supersedes this temporary driver for
future comparisons; this table remains useful historical cross-format context.

A second diagnostic compared the fastest exact scalar or rowbatch leaf at
`E=16,K=3072,N=128` with equal counts per expert:

| Rows/expert | IQ2_XS | IQ3_XXS | IQ4_XS |
| ---: | ---: | ---: | ---: |
| 1 | 20.452 us | 12.289 us | 13.705 us |
| 2 | 22.584 us | 16.232 us | 15.866 us |
| 4 | 25.977 us | 19.954 us | 21.984 us |
| 8 | 48.428 us | 37.399 us | 34.793 us |
| 16 | 97.458 us | 74.761 us | 59.216 us |

This used five alternating 200-iteration event samples after warmup. The driver
and result SHA-256 values were
`1090e383038c43464d27ae6097eb922cacdec6cf3f500548869024aa13727ebc`
and `f869226677be9e9ca507d59385e48c926b4b855ed645ca25b32949d190cc23bc`.
It remains diagnostic-only; the committed harness below is the canonical IQ2
baseline.

### Representative E256 baseline

Task #22 added `scripts/iq2_xs_tuning_bench.py` with sustained warmup,
counterbalanced order, rotating decode routes, full E256/K3072/N1024 shapes,
and representative balanced/hot/Zipf prefill distributions. The accepted
baseline is
[`../benchmarks/results/2026-07-22-gpu1-iq2-xs-laguna-tuning-baseline.json`](../benchmarks/results/2026-07-22-gpu1-iq2-xs-laguna-tuning-baseline.json).

At top-10 decode, rotating distinct experts measure 57.881 us selected-single
and 106.136 us fused dual-SiLU. A fixed hot set is 51.781/92.138 us and one
repeated expert is 46.496/83.122 us, confirming that repeated cache-hot routing
is 20-22% faster than the cold/distinct control and cannot select a production
candidate by itself. Fused dual correctness is BF16-bit exact to selected-single
gate/up plus the rounded SiLU boundary over 10,240 elements.

For prefill, rowbatch4 versus scalar is distribution-dependent at short prompts:

- 16 balanced tokens: `1.764 -> 1.875 ms` (+6.30%, reject rowbatch4);
- 16 hot/Zipf: -17.09%/-21.33%;
- 32 balanced/hot/Zipf: -8.35%/-20.20%/-29.92%;
- 64 tokens: -30.71% to -42.44%;
- 128 tokens: -42.66% to -48.34%;
- 512 tokens: -54.08% to -57.46%.

The full-shape rowbatch output is BF16-bit exact to grouped scalar over 327,680
elements. This baseline confirms that the K3072 unconditional rowbatch policy is
not optimal for sparse balanced routing; Task #25 must use per-expert population
rather than one global width rule.

### Clock-ramp finding

The original sequential event screen warmed each approximately 20-40 us leaf
only three times. A longer alternating screen found the one-row IQ2 scalar leaf
at 20.452 us and rowbatch4 at 21.451 us: rowbatch4 was 4.9% slower in that
corner, rather than 7.24% faster. The first scalar sample was much slower before
clocks rose.

Do not change the production policy from this E16 uniform fixture alone. It does
show that future screens need sustained warmup and counterbalanced ordering,
and that unconditional K3072 rowbatch4 needs re-evaluation on E256 sparse and
hot-expert distributions.

### Device-code finding

The most actionable finding is in the current gfx1100 HSACO. The source maps
one two-bit selector with nested conditionals:

```cpp
code == 0 ? 8.0f : (code == 1 ? 25.0f : (code == 2 ? 43.0f : 0.0f))
```

The compiler emits repeated `v_cmpx`, EXEC-mask manipulation, and
`s_cbranch_execz` sequences for every unrolled value. The dual kernel repeats
the decode independently for gate and up. The random selectors make waves
execute multiple divergent arms instead of selecting one uniform arm.

The packed table contains 4,096 selectors with this census:

- code 0: 2,114;
- code 1: 1,142;
- code 2: 840;
- code 3: zero.

The exact branchless mapping for the valid table is therefore:

```text
magnitude = 8 + 17 * code + (code >> 1)
```

It yields exactly 8, 25, and 43. Sign application can also use integer
XOR/subtract or float sign-bit manipulation without changing accumulation
order. This is the first implementation candidate.

### Bottleneck conclusion

IQ2 moves fewer raw bytes than IQ3 or IQ4 but achieves much less effective
raw-weight throughput. VGPR and scratch are already healthy. The first-order
problem is dequant instruction/control-flow efficiency, not resident bytes or
register spilling.

## Priority list

Expected impacts are hypotheses and apply to the named leaf, not to full-model
throughput.

| Priority | Candidate | Target | Expected leaf potential | Risk |
| ---: | --- | --- | ---: | --- |
| 0 | Representative, counterbalanced benchmark and policy screen | both | prevents false wins; small immediate policy gain | low |
| 1 | Exact branchless magnitude/sign decode | both | 20-50% | low |
| 2 | Q8_1 activation plus raw-IQ2 `sudot4` | decode/small batch | 1.5-3x | medium; approximate |
| 3 | 16-/32-value tasks, wider loads, geometry sweep | both | 10-30% | medium |
| 4 | Adaptive rowbatch1/2/4/8 | prefill | 15-40% at populated experts | low-medium |
| 5 | Tile two output columns while sharing activations | decode | 5-20% | medium |
| 6 | IQ2-specific integer MMQ/WMMA | large prefill | 1.5-3x | high |
| 7 | Wave-uniform address, reduction, and codegen cleanup | both | 2-10% | low |
| 8 | Fuse Q8_1 quantization into its producer | Q8_1 path | several us/layer | medium |

Priority number is execution guidance, not a promise to keep a candidate.

## Candidate details and precedent

### P1 — Exact branchless magnitude/sign decode

Replace the nested magnitude conditional with the exact arithmetic mapping.
Keep the current per-j accumulation order and the final block scale multiply.
Use an explicit branchless sign transform if the compiler still emits divergent
control flow.

Required evidence:

- RED static/device-code test that detects the old divergent decode pattern or
  pins the branchless helper contract;
- all existing CPU-oracle and selected/grouped exact gates;
- extracted HSACO showing the repeated selector branches disappeared;
- representative event and rocprof comparison;
- no scratch and no unexpected VGPR cliff.

Why first: it is format-specific, exact, small in scope, and the baseline HSACO
shows the pathology directly.

**Retained result (2026-07-22):** the exact arithmetic magnitude plus sign-bit
OR removes 16/32 selector `v_cmpx` operations from selected single/dual and
64/32 from grouped scalar/rowbatch4. Instruction counts fall
`502/784/1305/1026 -> 406/595/951/841`. Rotating-distinct selected single/dual
improve `57.881/106.136 -> 49.200/78.784 us` (-15.00/-25.77%); all hot/repeated
decode controls and all 30 representative prefill leaves improve. Scalar
prefill gains 21.21-32.26% and rowbatch4 gains 13.90-27.21%. Full-shape fused
and grouped checks remain BF16-bit exact. Allocated VGPR rises from
24/40/56/72 to 40/64/80/88 for selected-single/dual/grouped/rowbatch4, but all
leaves remain scratch-free and the complete matrix is non-regressive, so the
win is retained. Evidence:
[`../benchmarks/results/2026-07-22-gpu1-iq2-xs-branchless-decode.json`](../benchmarks/results/2026-07-22-gpu1-iq2-xs-branchless-decode.json).

### P2 — Q8_1 plus `sudot4`

The pinned llama.cpp implementation already contains the intended IQ2_XS
vector dot:

- `ggml-cuda/mmvq.cu` selects `vec_dot_iq2_xs_q8_1`;
- `ggml-cuda/vecdotq.cuh` expands two packed IQ2 selectors into signed byte
  vectors, performs `ggml_cuda_dp4a`, and applies IQ2/Q8 scales;
- gfx1100 maps the mixed signed/unsigned dot to
  `__builtin_amdgcn_sudot4` / `v_dot4_i32_iu8`.

Port the math and dataflow, not llama.cpp's host/runtime abstractions. Quantize a
BF16 activation row once into caller-owned Q8_1 scratch and reuse it across
selected experts, output columns, gate, and up.

Useful hipEngine precedent:

- raw selected Q4_K: 0.946 -> 0.357 ms, 2.65x, with 0.0025 ms activation
  quantization;
- raw selected Q5_K: 0.0916 -> 0.0395 ms, 2.32x;
- raw selected Q6_K: 0.0419 -> 0.0259 ms, 1.62x;
- all emitted profiler-visible dot kernels; Q4 ISA contained
  `v_dot4_i32_iu8`.

Cautionary precedent:

- the T16 Q4 split path gained only 1.04x after quantization overhead;
- a callable c1 fused-SiLU dp4a diagnostic regressed and stayed off;
- a Q5 T16 path improved its leaf but regressed end-to-end.

For IQ2 the dequant control-flow burden is larger, so this remains a strong
candidate. It is nevertheless a separate relaxed-math lane until Laguna
model-level quality passes.

### P3 — Pair shared-scale groups and use wider loads

Adjacent IQ2 eight-value groups share one scale nibble. Evaluate:

- one 16-value logical task: 192 tasks at K3072;
- one 32-value logical task: 96 tasks at K3072;
- aligned 32-/64-bit loads for adjacent packed selectors;
- local sizes 96/128/192/256 where mechanically valid.

The 16-value candidate can load and convert the shared scale once for two
selectors. It also removes the current 384-on-256 second-iteration imbalance.
The 32-value candidate shares more block metadata but increases per-thread work
and can lose memory-level parallelism.

Preserve the single primitive and dual/unfused equivalence. A changed reduction
association is not bit-exact by assumption; it must pass the full primitive
correctness gate before timing matters.

### P4 — Adaptive row batching

Add or evaluate rowbatch2 and rowbatch8 alongside scalar and rowbatch4. Prefer a
block-uniform decision from `end - begin`, so one launch can handle a mixed
routing distribution without a host scalar read:

- one row: scalar;
- two to three rows: rowbatch2 candidate;
- four to seven rows: rowbatch4;
- eight or more rows: rowbatch8 candidate.

Useful hipEngine precedent:

- IQ2 rowbatch4 already reduces its scalar leaf by roughly 30-67% in the
  counterbalanced diagnostic from two to sixteen equal rows/expert;
- IQ3 rowbatch4 improved production micro medians by 13.43-15.74% and cut its
  4K IQ3 family by 12.12%;
- exact Q8 prefill row reuse cut the Q8 family by 62.80% and total kernel sum by
  34.81%;
- Q6_K T16 rowtiling changed four/six-row head work by 2.57x/3.48x.

Monitor VGPR carefully. Rowbatch8 doubles the gate/up accumulator set and must
stay scratch-free.

### P5 — Two-output selected decode tile

A block that computes two adjacent output columns can share BF16 activation
loads and conversions while retaining independent weight streams and
accumulators. Start at tile2; do not jump to tile4 or tile8.

Cautionary precedent: the IQ4 weighted-down tile4 candidate looked 39.61% faster
in a repeated cache-hot microbenchmark but regressed the real 37-layer family by
12.50%. Its fourfold smaller grid lost cold, distinct-weight latency hiding.
An IQ2 tile is retainable only after a cold/distinct-expert production-shape
trace, not a repeated one-weight microbenchmark.

### P6 — Integer MMQ/WMMA for populated prefill

The current direct raw-IQ FP16 WMMA path is a correctness diagnostic, not a
production schedule. Sparse expert padding made related raw-IQ WMMA experiments
catastrophically slow, including a sampled Q3 path that collapsed to
0.370 tok/s.

A future IQ2 path should instead follow the pinned llama.cpp IQ2 MMQ concepts:

- Q8_1 activation tiles;
- IQ2 expanded into packed integer fragments;
- integer dot/WMMA accumulation;
- scalar or rowbatch fallback below a measured per-expert population threshold.

This lane starts only after scalar/rowbatch tuning and only if the representative
routing harness shows enough experts at 16 or more rows to amortize tiles.

### P7 — Codegen and reduction cleanup

Low-risk checks after the large decode change:

- mark wave-uniform block indices with `__builtin_amdgcn_readfirstlane`;
- hoist row/block bases and shared scale addresses;
- confirm byte loads combine into aligned wider global loads;
- inspect `s_waitcnt`, vector 64-bit address chains, LDS, barriers, and VGPR;
- test reductions only when the expected association is explicit.

Useful precedent: one wave-uniform IQ3 block-base annotation removed two
per-lane `v_mad_u64` chains, reduced allocated VGPR 48 -> 40, and cut the real
IQ3 family by 2.05% with exact outputs.

### P8 — Fuse Q8_1 quantization after it wins

Do not build fusion around an unproven dot path. If P2 wins, first keep a
caller-owned sidecar and prove it is reused for gate and up. Then consider
fusing activation quantization into the preceding norm/router producer to remove
one launch and one activation read.

## Recommended execution order

### Phase A — trustworthy baseline

1. Land a committed representative benchmark harness.
2. Measure current selected single, selected dual, grouped scalar, rowbatch4,
   and test-only WMMA with sustained warmup and alternating order.
3. Correct policy only from E256 representative routing, not E16 uniform counts.

### Phase B — exact scalar path

4. Implement branchless magnitude/sign decode.
5. Gate exactness and inspect HSACO before timing.
6. Evaluate 16-/32-value task width and local geometry.
7. Add adaptive rowbatch2/4/8 and retain only measured thresholds.
8. Evaluate output tile2 with a cold, distinct-weight production trace.

### Phase C — approximate integer path

9. Port caller-owned Q8_1 plus raw IQ2 `sudot4`.
10. Require primitive KL/top-1, ISA proof, and quantizer-inclusive timing.
11. When the Laguna runner is ready, run model-level KL/top-1 and held-out
    prompt categories before enabling it by default.

### Phase D — large-prefill path

12. Measure expert population distributions from real Laguna prompts.
13. Implement integer MMQ only if enough work occupies complete tiles.
14. Keep scalar/rowbatch fallback for sparse experts.

## Representative benchmark matrix

### Decode

Primary shape:

```text
E=256, K=3072, N=1024, top_k=10, x_rows=1
```

Measure both selected single and fused dual gate/up. Route cases:

- ten unique experts;
- repeated/hot experts;
- expert IDs including 0 and 255;
- invalid-ID correctness control;
- rotating distinct expert sets across timed iterations to avoid a
  repeated-weight cache-hot illusion.

Report:

- event median and all samples;
- raw weight bytes and approximate effective GB/s;
- VGPR, LDS, scratch, workgroup and grid;
- quantizer time separately and inclusive for Q8_1 candidates;
- BF16 mismatch or KL/top-1 result.

### Prefill

Primary shape:

```text
E=256, K=3072, N=1024, top_k=10
```

Synthetic token counts: 16, 32, 64, 128, and 512. Routing distributions:

- balanced/uniform;
- hot-expert skew;
- Zipf-like skew;
- sparse short prompt;
- held-out deterministic seeds.

A smaller `E=16,N=128` screen may reject obviously bad candidates quickly, but
cannot select production policy.

### Timing discipline

- Warm the GPU for a sustained interval, not three microsecond-scale launches.
- Alternate candidate/control order across at least five samples.
- Use HIP events for leaf timing and rocprof only after prebuilding the cache.
- Separate repeated cache-hot screens from cold/distinct-weight traces.
- Do not claim a model speedup from a primitive event ratio.

## Retention gates

### Exact candidates

Retain only when:

- the independent CPU oracle still passes;
- selected single, dual/unfused, grouped, and rowbatch boundaries pass;
- BF16-bit exactness is preserved where the candidate claims exactness;
- every representative shape is non-regressive or dispatch policy excludes the
  losing region;
- scratch remains zero;
- expected kernel symbols appear in a cache-only trace.

### Approximate candidates

At minimum:

- KL <= 0.05 and top-1 agreement >= 90% on primitive fixtures;
- finite outputs across scale and routing edge cases;
- quantization cost included in performance;
- unfused fallback remains registered;
- eventual Laguna model-level KL/top-1 and prompt-category gate before default
  promotion.

### Performance evidence

Every retained performance row records exact quant, shape, hardware, command,
result, correctness, source revision, and artifact. Any model-level acceptance
uses the full prompt categories and held-outs required by
[`BENCHMARK.md`](BENCHMARK.md); no prompt-conditioned or token-conditioned path
is valid.

## What not to chase first

Do not spend early iterations on:

- WMMA for c=1 decode;
- wave64 as a generic fix;
- LDS staging of the IQ2 grid table;
- generic compiler-unroll flags without a code-object hypothesis;
- a fully expanded persistent IQ2 sidecar that destroys the 2.3125-bpw
  bandwidth advantage;
- tile4/tile8 output kernels before tile2 survives a cold production trace;
- repeated one-weight microbenchmarks as promotion evidence;
- model-, token-, route-, or prompt-specific hardcoding.

These either conflict with the measured regime or repeat rejected hipEngine
experiments documented in [`ROOFLINE.md`](ROOFLINE.md) and
[`LESSONS-LEARNED.md`](LESSONS-LEARNED.md).

## Full-model readiness

The pinned model is available outside the repository at:

```text
/models/gguf/Laguna-S-2.1-UD-Q2_K_XL.gguf
```

Model weights and partial downloads are never committed. It was verified with:

```bash
stat -c '%s %n' /models/gguf/Laguna-S-2.1-UD-Q2_K_XL.gguf
sha256sum /models/gguf/Laguna-S-2.1-UD-Q2_K_XL.gguf
```

Expected size and SHA-256 are listed in the objective section. File presence is
not model support. The other integration lane must still complete exact model
mapping, all-tensor inventory resolution, runtime allocation, full execution,
and state/KV behavior before kernel model-level gates begin.

## Active lane ledger

| Lane | State | Exit condition |
| --- | --- | --- |
| Representative benchmark | complete | committed E256 harness and accepted diagnostic baseline |
| Branchless exact decode | complete | exact, branch-free selector decode; retained across full matrix |
| Group width/geometry | queued | best exact/correctness-gated mapping selected or all rejected |
| Adaptive rowbatch | queued | measured sparse/balanced/hot policy |
| Output tile2 | queued | cold production-shape non-regression |
| Q8_1 `sudot4` | queued | primitive gate, ISA proof, inclusive win; model gate later |
| Integer MMQ | deferred behind scalar work | populated-expert crossover and retained full-shape win |
| Laguna model validation | blocked on integration | all-tensor runner plus model-level gates |

## Source lineage

Pinned references:

- llama.cpp HIP `1ebf790cda38d827559548f67b0469189690cc8c`:
  `ggml/src/ggml-common.h`, `ggml/src/ggml-quants.c`,
  `ggml/src/ggml-cuda/vecdotq.cuh`, `ggml/src/ggml-cuda/mmvq.cu`, and
  `ggml/src/ggml-cuda/mmq.cuh`;
- qwen-kernel `52e240f9c6d91750d0e5e692976cfb67fd9bc603`:
  grouped IQ3/IQ4 scheduling and compact expert-major precedent;
- hipEngine retained IQ3/IQ4/Q4/Q5/Q6 experiments cited through
  [`KERNELS.md`](KERNELS.md) and benchmark artifacts.

Before each source port, rerun `scripts/check_lineage.py` for the exact parent
file and inspect drift rather than assuming this plan remains current.
