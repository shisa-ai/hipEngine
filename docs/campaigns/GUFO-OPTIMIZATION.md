---
status: current
owns: The gufo source review, ranked adoption candidates, the DFlash2 adoption assessment, and the correctness-gated revisit queue for gfx1151/gfx1100 optimization work.
---
# gufo optimization campaign

**Evidence status:** read-only source review completed 2026-09-24. No GPU runs
were performed (the device is in use). Every hipEngine number below is cited
from an existing scoreboard artifact; every gufo number comes from their
published benchmark docs. Kernel claims cite file anchors in the gufo clone at
`~/gufo-src` (external read-only reference, commit `9cad139`, 2026-09-24) and in
this tree. Nothing here is a new measurement or a new claim.

## 1. What gufo is

gufo (github.com/gufo-org/gufo) is a C++20/HIP inference engine built only for
AMD Strix Halo: Ryzen AI MAX+ 395, Radeon 8060S `gfx1151`, up to 128 GiB
unified memory. Nix-pinned toolchain, ROCm 7.2.3, GGUF-only, a small curated
model list (Qwen3.8-27B, Qwen3.8-Flash-Next, DeepSeek V4 Flash, plus audio,
image, and video models), OpenAI-compatible C++ server. hipEngine is credited
in their README as a reference project ("torch-free HIP execution, and native
speculative-cycle work").

Their measurement and promotion rules live in `docs/PERFORMANCE.md` in the
clone. Rules worth noting for this campaign: explicit Wave32 compilation for
`gfx1151` with a stated prohibition on inheriting launch geometry from another
target without a fresh measurement; hipBLASLt/rocBLAS kept as the baseline for
supported matrix shapes; decode GEMV judged on bytes read before peak matrix
throughput; in-kernel dequantization with a stated ban on materializing
dequantized weights in global memory; promotion requires focused correctness,
full-logit envelope, greedy token parity, state/rollback correctness, and a
measured workload gain outside noise.

## 2. Comparable published numbers

Same iGPU class (Radeon 8060S `gfx1151`); different protocols, listed below the
table. hipEngine rows are 2026-09-20 scoreboard entries; gufo rows are their
2026-09-22/23 published pages.

| Workload | gufo | hipEngine | Notes |
| --- | ---: | ---: | --- |
| Qwen3.8-27B Q4 AR decode | 12.37 tok/s | 12.15 tok/s | Effective parity; decode is weight-bandwidth-bound on both sides |
| Qwen3.8-27B Q4 AR prefill | 656.33 tok/s | 404.5 tok/s | gufo leads ~1.6x; protocol mismatch (see caveats) |
| Qwen3.8-27B speculative, mixed/natural prompts, c1 | 26–27 tok/s (DFlash2) | 21.0–25.3 tok/s (MTP) | Near parity on the workload that matters; hipEngine range spans its measured MTP lanes |
| Qwen3.8-27B speculative, repetitive output, c1/c8 | 63–70 / up to 123 tok/s | no row | Acceptance-friendly workload; hipEngine has no matching row |
| Qwen3.8-Flash-Next Q4 prefill | 1,628.52 tok/s | 153.96 tok/s (p512) | Same artifact (`UD-Q4_K_XL`); largest gap |
| Qwen3.8-Flash-Next Q4 AR decode | 26.04 tok/s | 19.38 tok/s (p512) | Same artifact |

Protocol caveats, required before any of these becomes a claim:

- gufo measures over HTTP at pp2048/tg128 with depth sweeps; hipEngine rows
  are resident-harness snapshots at 512/1K/4K prompts (27B) and a 2026-09-05
  screening baseline flagged at >2% per-case CV (Flash-Next).
- gufo's 27B target is Unsloth `Q4_K_XL`; hipEngine's is `Q4_K_M`
  (a `Q4_K_S` file is a separate lane). Flash-Next uses the identical
  `UD-Q4_K_XL` artifact on both sides.
- gufo headline peaks include repetitive output and sum per-request decode
  rates at concurrency; hipEngine headline rates are natural-length suites.
- A matched-protocol pp2048/tg128 run through `hipengine serve` on the
  Framework Desktop is required before publishing any gap figure (§7, item 1).

## 3. Ranked adoption candidates from gufo's source

Ranked by expected gain against the two measured gaps (27B prefill ~1.6x,
Flash-Next prefill ~10x) divided by adoption cost. All are design patterns;
none has been implemented here yet.

### A. Fused norm/activation → Q8_1 quantization epilogues (HIGH)

gufo's `src/models/qwen/hip/kernels/prefill_quant_gemm.hip` fuses the passes
that feed its quantized prefill GEMMs, with an explicit bit-identity argument
for each:

- `opt-c173` (line 39): residual add + RMSNorm + tiled Q8_1 quantization in
  one pass, staging in LDS. Removes a 42 MB + 21 MB per-layer write/read pair
  at batch 2048; 221 MB → 137 MB per norm. The reduction tree and epsilon are
  identical to the unfused kernel, so the FP32 norm values are bit-identical;
  only the quantizer input changes from a BF16 round-trip to the FP32 value.
- `opt-c174` (line 129): SSM post-norm + SiLU gate writing Q8_1 directly,
  214 MB → 114 MB per layer at batch 2048, same butterfly reduction and
  epsilon as the unfused kernel.
- `opt-c192` (line 272): SwiGLU written straight from a blocked GEMM's FP32
  accumulator into the next GEMM's Q8_1 activation, removing 143 MB stores +
  143 MB loads per layer at batch 2048. Enabled by two shape facts (a wave's
  row subtiles align to 32-element quantization blocks; LDS staging aligns one
  token's rows to consecutive lanes).

hipEngine's prefill chain still launches separate quantization passes:
`hipengine/runtime/gguf_linear.py` imports `gguf_q4_k_quantize_bf16_q8_1`,
`gguf_q8_1_d4s4_f32_quantize_bf16`, `gguf_q8_0_mmq128_quantize_bf16_d4x3`,
and related standalone kernels, then calls the MMQ/WMMA GEMMs.

Why this fits our contracts: each fusion preserves the reduction tree and the
exact FP32 value handed to quantization, so it belongs to the strict
bit-identical fusion class under `docs/EXECUTION-PROFILES.md`, not the
tolerance-traded production class. It satisfies the "fused kernels require a
strict unfused fallback" invariant naturally (the current unfused chain is the
fallback). Expected gain targets the compute/memory round-trip share of the
27B prefill gap; Flash-Next's chain would benefit the same way.

**Measured on-route status (2026-09-24, gfx1151, Qwen3.8-27B `Q4_K_M`,
512/128 sweep).** On the route this model selects, none of the three
standalone quantize passes above execute: a full-kernel trace of the 512-row
prefill dispatchs no activation-quantize kernel at all (the only name
matching `quantize` is the weight-side `gguf_q5_k_t16_dequantize_f16_tile_octet`,
8.2 ms/pass). Activations stay BF16 straight from the fused gate/up SiLU
GEMM epilogue (`gguf_q4_t16_dense_dual_wmma_prefill_silu`) into the next
GEMM, so c192's pass elimination is already structural on this route — one
standalone pass fewer than gufo's Q8_1 chain. The weight-side passes that do
run (`bf16_to_f32`, 4.2 ms/pass, feeding the hipBLAS `Cijk` GEMMs at 36.7
ms/pass) sit in a different accumulation-order class, so folding or flipping
them is a production-profile change, not a strict-class one. The remaining
strict-class fold — the c173 norm-into-GEMM-prologue analog — saves at most
12.58 MB per norm site (a BF16 output write plus its GEMM read at 512 x
6,144 x 2 B); even counting every one of the 129 norm dispatches per pass
gives ~1.6 GB, ~1.9 ms at 864 GB/s, a 0.14% ceiling against the +0.31%
needed to clear the 382.12041 baseline (380.9576 same-day stock). §7 item 3
cannot move the 512-row metric on this route; the batch-2048 shapes where
gufo's 221 MB → 137 MB per-norm regime applies are carried as revisit queue
item 9.

### B. Shape-routed W8A8 blocked-WMMA prefill GEMM for Flash-Next (HIGH)

gufo's Flash-Next prefill engine is a shape-dispatched W8A8 blocked-WMMA GEMM
family:

- `src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp:4492` —
  `W8A8Gemm` picks per shape: a wave64 four-row-group instantiation only for
  `batch >= 1024 && m == 2560 && k == 6144` ("preserving every K32
  accumulator update"), a 128-token macro tile as the throughput config, a
  64-token variant for short chunks, and 64-row tiles for narrow projections
  so they still fill the device.
- `w8a8_wave64.hip:6` instantiates their generic
  `WKQuantA8BlockedWmmaGEMMKernel<128, 128, 2, 4, 1, kQ8_0, 64>` with a comment
  that each model keeps "its qualified schedule".
- `src/models/qwen/hip/kernels/attention_wmma.hip:1-13` — masked prefill
  attention on WMMA cores, wave32 fragment layout verified against a
  double-precision CPU reference, three ablation-derived design choices.
- Flash-Next also has its own SSM prefill/decode family
  (`ssm_row_split.hip`, `ssm_recurrence.hip`, `prefill_ssm.hip`,
  `batched_ssm.hip` in `src/models/qwen/hip/kernels/`).

hipEngine's equivalent lever exists but is gated: the Qwen4Exp (= Qwen3.8
Flash-Next) grouped prefill candidates raise warm repeated-token 512 prefill
from **8.67 to 211.76 tok/s** and corrected natural-suite wall by **6.60x**,
but default off because they fail the production numerical gate (687-row
packet: mean/p95/max KL 0.01280/0.05553/0.82237, top-1 94.47%) —
`docs/REFACTOR.md` "2026-08-27 Qwen4Exp grouped prefill performance
candidates"; still env-gated in current code
(`hipengine/runtime/qwen4_exp_runner.py:3687,4493`).

The adoption lesson is the fidelity strategy, not the tile sizes: gufo keeps
accumulator order stable per qualified shape and proves every fusion
bit-identical, which is how a WMMA-class prefill kernel stays inside a
greedy-parity envelope. Our blocked candidates reassociate and drift. The
revisit path for our candidates is therefore accumulator-order-preserving
tiling plus the same bit-identical fusion proofs, evaluated against the
`docs/EXECUTION-PROFILES.md` §6 production gate — not a gate relaxation.

Open question to resolve before design work: whether gufo's W8A8 route runs
the `UD-Q4_K_XL` weights as-is or converts per-layer weight types at load
(`weights.cpp:123-144` accepts `Q8_0`/`BF16`/`F16`/`F32` and `Q5_1`/`Q8_0`
lists per layer group). Their `docs/models/qwen3.8-flash-next/README.md` says
original-model parity is unqualified; `QUALITY.md` for that model governs.

### C. Adaptive speculative length controller with an offline cost model (MEDIUM-HIGH)

`src/models/qwen/dflash_policy.hpp` — `DFlashLengthController` chooses a draft
block length per cycle by maximizing expected emitted tokens per unit of
work:

- offline measured cost tables per width and cohort size (1, 2, 3-4, 5-6,
  7-8 concurrent requests), never request timings;
- a censoring-aware mean-acceptance estimator (a saturated block probes upward
  instead of being treated as the true mean);
- a position-dependent context multiplier past 2,048 tokens;
- greedy cohorts priced with "wider-verification cost".

hipEngine's speculative lanes use fixed block sizes (DFlash2 B3 measured
optimum; MTP with budget/admission controls in the concurrency campaigns).
The expected-tokens-per-work model above is directly portable to both MTP
block/budget selection and any DFlash2 revival, and needs only offline cost
tables plus acceptance history we already record.

### D. Draft-private Q8_0 copy of the target LM head (MEDIUM, DFlash2-scoped)

`src/models/qwen/hip/kernels/dflash_kernels.hip:16-21`: the DFlash-2 draft
head is the target LM head, which for their Q8 target stays BF16 at 2.54 GB —
half of everything the draft graph reads. Drafting is purely weight-bandwidth
bound and only steers which tokens the target verifies, so a draft-private
Q8_0 copy cannot change an emitted token (the verifier still runs the
target's head).

This addresses hipEngine's measured DFlash2 deficit head-on: the drafter
forward + select costs ~96 ms/cycle at roughly one-fifth of achievable
bandwidth for its 3.584 GiB residency
(`docs/campaigns/QWEN38-27B-DFLASH2-CAMPAIGN.md`). Applicable only if DFlash2
revives (§5); see the assessment in §4.

### E. hipBLASLt offline plan tuning with a persistent plan database (MEDIUM-LOW)

gufo ships `tune_hipblaslt` (offline plan tuning),
`src/core/hip/hipblaslt_plan_database.cpp` (file-backed, locked,
`STRIXLT`-tagged plan store) and `GUFO_HIPBLASLT_PLAN_CACHE` replay at
benchmark time. hipEngine selects "the measured gfx1151 fast heuristic,
clamped to availability" for its hipBLASLt problems
(`hipengine/core/hipblaslt.py:286`) — one hardcoded heuristic rather than a
per-shape tuned set. Gain applies only where BLAS routes run (dense projections,
lm_head, the rocBLAS prefill families), not to the custom MMQ quantized paths.

### F. Fresh wave-geometry measurement for gfx1151 (MEDIUM, measurement-first)

`hipengine/kernels/hip_gfx1151/` is empty; `hipengine/kernels/backends.py:198`
treats `gfx1100` and `gfx1151` as peer backends, so every kernel running on the
8060S is architected for the gfx1100 lineage. gufo compiles Wave32 explicitly
for gfx1151 and forbids inheriting geometry without a fresh measurement
(`docs/PERFORMANCE.md` "HIP"). One concrete in-tree signal: the
`HIPENGINE_QWEN4_EXP_Q5_1_WAVE64` reduction improves warm decode by 6–8% but
defaults off on production KL (0.002565/0.007202) — a geometry experiment
blocked by gates, not by absence of headroom. First step when the GPU is free:
record the effective wave size our gfx1151-qualifying kernels actually compile
to, then treat geometry as a per-kernel ablation under the existing gates.

**Measured on-route status (2026-09-24, gfx1151, Qwen3.8-27B `Q4_K_M`,
512/128).** First ablation executed under this item. The gfx1151 router's
measured owner bands stop at 384 rows (`SHARED2_384_MAX_ROWS`), so the
512-row production cell reached the 48-column shared-B base purely by
fallthrough — and that family is 337.9 ms, 25.4% of prefill, in the same-day
kernel trace. A/B of the same-`.hip` siblings at rows >= 385 (base wrapper
delegation, all five arms with identical `correctness_sanity` blocks):

| rows >= 385 owner | prefill tok/s (mean of 3) | vs base |
| --- | ---: | ---: |
| shared-B 48-col base (stock) | 381.40 | — |
| **w64 64-col** | **395.44** | **+3.7%** |
| shared_b3w8r3 | 340.66 | -10.7% |
| shared_b2w2 | 351.55 | -7.8% |
| shared_b2w4 | 359.71 | -5.7% |

The base leaf (`hipengine/kernels/hip_gfx1100/quant/gguf_k_t16_selected_prefill.py`)
now delegates to w64 past 384 rows. The remaining §3F work is the per-kernel
effective wave-size record for the rest of the gfx1151-qualifying set; the
`hipengine/kernels/hip_gfx1151/` host router itself was not edited (outside
the loop's declared scope), so its bands still name the base past 384 — the
decision lives one level down until a router-band update is scoped.

### G. Measurement tooling and MALL discipline (LOW cost, enables everything above)

From `docs/PERFORMANCE.md` and their tool tree:

- A roofline calibrator reporting sustained WMMA INT8/BF16/FP16, VALU FP32
  FMA, LDS, and DRAM rates with checksummed live work and hoisting-detection
  guidance.
- `tools/prof/isa_mix.py` — per-kernel instruction mix (matrix/VALU/LDS/DMEM/
  wait) with a documented caution that epilogue instructions must not be
  attributed to the K loop without an ablation.
- `tools/prof/prof.py` — pipeline-stage rollup, GPU-busy-vs-wall-span, and
  largest-idle-gap attribution to the dispatch on either side (idle share =
  host/launch-bound diagnosis).
- 32 MB MALL harness rule: a decode-bandwidth harness must size its working
  set to the model's whole per-token weight footprint and time a full token
  pass; looping one shape reports 400–860 GB/s for a kernel that sustains
  124 GB/s in the model. They also measured `__builtin_nontemporal_load`
  costing more than half of achieved bandwidth on every streaming kernel tried.
- `GUFO_DISPATCH_TELEMETRY=1` emits semantic dispatch decisions as JSONL —
  the same observability our "selected variant and fallback reason must be
  observable" rule requires.

hipEngine already has rocprofv3 recipes in `docs/OPTIMIZATION.md` §9 and a
legacy MALL/cache-hint investigation (worklog entry
`20260629T014516.000000Z-legacy-2026-06-29-mall-cache-hint-investigation-non-tem-3dae39.md`,
pre-cutoff legacy entry).
What to adopt is the MALL sizing rule for decode-GEMV harnesses and the
idle-gap attribution step as standard benchmark protocol.

### H. Host/runtime design — study only (LOW)

gufo's host rules (RAII everything, all model/KV/graph/scratch resources
allocated before serving, no general heap allocation in decode dispatch, no
process-wide mutex held across device waits, mapped GGUF shards registered
with HIP plus a bounded async upload pipeline in `src/core/hip/weight_upload.hpp`)
are sound but partially native to a C++ runtime. hipEngine's Python host with
ctypes, session pools, PM4/HIP-graph decode, and radix prefix cache keeps the
torch-free invariant either way. Adopt the checklist items that map to our
host (pre-serve allocation, upload pipelining, lock audit) without treating a
C++ rewrite as in scope — it would be an architectural change, not an
optimization.

## 4. DFlash2 adoption assessment

**Question:** would adopting gufo's DFlash2 approach be worthwhile?

**What hipEngine already has.** A complete DFlash2 chain: NumPy drafter
oracle, native grouped dynamic-conv + top-16 selector kernels, native drafter
forward, GGUF tap capture, chain-batched B-exact verifier, AR-exact on every
measured prompt. It is a registered diagnostic, not a default: the measured
best point (B3) is 8.85 tok/s = 0.66x AR, against exact native MTP B3 at
23.85 tok/s = 1.7845x AR. The 2026-08-22 attribution correction in
`docs/campaigns/QWEN38-27B-DFLASH2-CAMPAIGN.md` established that drafting
quality is at parity with MTP (2.80 vs 2.85 accepted tokens/cycle) and the
entire 2.7x deficit is cost: a ~96 ms drafter forward running at about a fifth
of achievable bandwidth, a verify path with different amortization, and a
rowtile admission cliff (`_PACK8_ROWTILE_MAX_ROWS = 4`) that forces the
shallow chain. The reverted rowtile-8 experiment (620 → 310 ms, AR-divergent
on `code_lru_cache`, never root-caused) remains the campaign's top open item.

**What gufo gets from DFlash2 on the same model and iGPU.** 26–27 tok/s mixed
at depth 0 (single user), 63–70 tok/s repetitive single-user, 123 tok/s summed
at 8 concurrent requests, with an adaptive length controller (§3C) and the
draft-private Q8 head (§3D). Their verification preserves AR output in greedy
mode; their QUALITY.md states sampled DFlash2 need not match AR's same-seed
sequence (numerical-tolerance gates, not bit-identical sampler parity).

**Comparison.** On mixed/natural prompts — the workload comparable to our
suites — gufo's DFlash2 (26–27 tok/s) is at parity with our MTP lane
(21.0–25.3 tok/s across measured lanes). The large gufo headline numbers come from repetitive-output
and summed-concurrency workloads where acceptance is high; our MTP would also
benefit from repetitive output, but no matching row exists, so no such gain is
claimed here.

**Verdict.**

1. **Do not displace MTP with DFlash2.** Our speculative lane already matches
   gufo's best speculative mode on comparable prompts, and MTP stays the
   promoted path (1.8–2.1x AR on measured lanes).
2. **DFlash2 adoption as a project is not worthwhile in the near term.** It
   exists, is AR-exact, and its deficit is cost, not quality — but every cost
   fix (rowtile-8 root cause, drafter bandwidth, verify amortization) must be
   paid before it competes with our own MTP, let alone adds to it.
3. **Revisit triggers:** (a) a target model with a DFlash2 draft but no MTP
   sidecar where block drafting beats single-token proposal; or (b) the named
   cost fixes become cheap — the draft-private Q8_0 head (§3D) is the
   cheapest single win and is provably token-preserving, so it is the first
   thing to try if DFlash2 work restarts.
4. **Port the adaptive length controller (§3C) to MTP first** — it improves
   the lane that is already on by default.

## 5. Correctness-gated revisit queue

Optimizations in this tree that exist but are not default because they miss a
numerical gate or an envelope, plus open performance debt worth tracking.
Sources: `docs/REFACTOR.md`, `docs/EXECUTION-PROFILES.md`, campaign docs,
scoreboard artifacts. The gate itself is not up for renegotiation —
`docs/EXECUTION-PROFILES.md` §6 defines the production envelope; what is
revisited is the design, per §3B.

| # | Item | Measured upside | Why it is not default | Revisit cost | Recommendation |
| --- | --- | --- | --- | --- | --- |
| 1 | Qwen4Exp grouped prefill candidates (`HIPENGINE_QWEN4_EXP_GROUPED_MOE_PREFILL`, `Q5_1_WMMA`, `Q8_0_GROUPED_WMMA`, `Q4_TILE_M/N`) | 8.67 → 211.76 tok/s warm p512; 6.60x natural-suite wall | Fails production gate: 687-row KL mean/p95/max 0.01280/0.05553/0.82237, top-1 94.47% (`docs/REFACTOR.md` 2026-08-27) | Expensive: new accumulator-order-preserving layout plus gate reruns | **Top priority.** This is the Flash-Next prefill lever; §3B names the fidelity strategy |
| 2 | Qwen4Exp chunked prefill (`prefill_chunked`, size-2 candidate) | 1.210 s vs 2.193 s serial on a 9-token smoke | Size-9 rejected at KL_serial 0.09754; full multi-prompt + 2,052-row QSA transition gate not passed | Medium: needs the gate run on real workloads | Re-run the gate once item 1's arithmetic is settled; chunking and layout interact |
| 3 | `HIPENGINE_QWEN4_EXP_Q5_1_WAVE64` decode reduction | +6–8% warm decode | Production KL 0.002565/0.007202 | Medium: exact-repack/reduction successor, or geometry retune that stays in envelope | Tie to §3F wave-geometry audit; bisection flag exists |
| 4 | DFlash2 rowtile-8 (reverted) | verify 620 → 310 ms for 8 rows | AR-divergent on `code_lru_cache`, never root-caused (`QWEN38-27B-DFLASH2-CAMPAIGN.md`) | Medium: root-cause first | Only if DFlash2 revives (§4 trigger); otherwise stays documented |
| 5 | UD raw-IQ four-quant integer MMQ prefill | 171.9 vs current W4A16 150.3 tok/s (~+14%) | Route is "not admissible under the production envelope" (`benchmarks/README.md` raw-IQ section) | Medium: envelope admissibility analysis per §6, then gates | Revisit if the UD 27B lane matters; the per-tensor repack fix already captured most of the gap (22.1 → 150.3) |
| 6 | Wide-Q6 shared4 verifier candidate (`HIPENGINE_GGUF_VERIFY_WIDE_Q6_SHARED4`) | W1 verifier shapes at R20/R24/R32 | Default-off pending complete C6/C8 strict-teacher, determinism, task, and performance gates (`docs/REFACTOR.md` 2026-09-01) | Medium: run the named gates | Cheap to finish — the gates are already specified; run or reject |
| 7 | Paged suffix prefill route | 6.5 ms/token vs 0.39 ms/token slot-local on a gapped placement (0.8B probe) | Not correctness-gated; open performance debt with a partial gather-route fix (`docs/REFACTOR.md` 2026-09-19, "the real prize") | Medium | Worth its own unit once prefix-cache traffic is profiled; several-times-slower-per-token on every backend |
| 8 | Device top-512 QSA selector | Replaces host NumPy exact selection at 262K scale | Gated on exact GPU selector matching Transformers/llama indices at 2,052/4K/16K/64K/262K plus isolation gates | Medium | Revisit with Flash-Next long-context work (item 1) |
| 9 | §3A norm/activation epilogue fusions at batch-2048 shapes (gufo c173/c174) | gufo: 221 MB → 137 MB per norm at batch 2048 | No measurable share at 512 rows on the selected-WMMA route: zero activation-quantize dispatchs in the 2026-09-24 trace, and a 0.14% theoretical ceiling on the norm fold versus the +0.31% acceptance gap (§3A status note) | Medium: 2048-row trace plus the norm-side fusion behind the strict gate; the norm kernel lives outside the Sep-24 loop scope (`kernels/hip_gfx1100/norm/`) | Revisit with the matched-protocol §7.1 run at larger shapes |

**Settled decisions this campaign does not reopen:**

- Packed verifier model graphs — rejected by measurement (0.19–0.62% gains,
  19.53% recapture regression; artifact
  `benchmarks/results/2026-09-02-w7900-q4km-k3-packed-verifier-graphs-rejected.json`).
- `llama-compat` MTP-2 — retained as an explicit accuracy-traded opt-in by
  decision; the promoted MTP path is exact.
- GPF-9C (llama.cpp recurrence schedule) — rejected on frozen-512 speed, not
  correctness (legacy worklog 2026-07-15).
- Non-temporal load hints — gufo independently measured them harmful on
  gfx1151; treat as closed unless a new MALL finding reopens it.

## 6. Explicit non-goals

- No C++ host rewrite; the Python host and torch-free runtime invariant stand.
- No GGUF-only restriction; hipEngine's model/quant plugin surface stays.
- No relaxation of `docs/EXECUTION-PROFILES.md` gates. gufo's own promotion
  rules (greedy parity + full-logit envelope) are broadly the same shape as
  our strict/production split; the difference is that their retained kernels
  prove bit-identity per fusion, which is the strategy §3B adopts.
- No benchmark row is produced by this document; §7 queues them.

## 7. Execution queue (when the GPU is free)

1. **Matched-protocol 27B prefill A/B:** run hipEngine through `hipengine
   serve` at pp2048/tg128, depths 0/4K/16K, greedy, on the Framework Desktop
   against `Q4_K_M`, and record the gufo-protocol row (their `Q4_K_XL` artifact
   difference noted). This converts the ~1.6x table figure into a claimable
   number or retires it.
2. **Flash-Next profile attribution:** stage-rollup profile (§3G idle-gap
   attribution) of our p512 prefill to split GEMM / SSM / attention / host
   shares before touching kernels; then start item 1 of §5 with §3B's
   fidelity strategy.
3. **Bit-identical prefill epilogue fusions** (§3A) on the 27B Q4 prefill
   chain, strict class, unfused fallback already registered. *(2026-09-24:
   measured no-target on the selected-WMMA route at 512 rows — no
   activation-quantize passes run, and the norm-fold ceiling is 0.14% versus
   the +0.31% acceptance gap; see the §3A status note and revisit queue
   item 9.)*
4. **Wave-geometry + MALL audit** (§3F, §3G): effective wave size per
   gfx1151-qualifying kernel; decode harness working set sized to per-token
   weight footprint.
5. **Adaptive length controller port** (§3C) to MTP, offline cost tables from
   existing acceptance telemetry.
6. **DFlash2 stays parked** per §4; triggers named there.