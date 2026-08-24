# oMLX Ideas Campaign

- **Status:** source audit complete; experiments not started
- **Created:** 2026-08-25
- **Requested filename:** `OLMX-IDEAS.md` (the project reviewed is spelled **oMLX**)
- **Primary target:** Qwen3.6 dense and MoE MTP on `hip_gfx1100` / Radeon Pro W7900
- **Secondary target:** independently qualified `hip_gfx1151`; no result transfers across hardware lanes
- **Authority:** [`PLAN.md`](PLAN.md), [`KERNELS.md`](KERNELS.md),
  [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md), [`TESTING.md`](TESTING.md),
  [`BENCHMARK.md`](BENCHMARK.md), [`MTP.md`](MTP.md), and
  [`MTP-FIX.md`](MTP-FIX.md) remain normative

This document records a source-level review of
[`jundot/omlx`](https://github.com/jundot/omlx) and defines a falsifiable
hipEngine campaign. It is not a benchmark result, a promise that an Apple
kernel will transfer to RDNA3, or authorization to bypass the existing strict
fallback, registry, transaction, or evidence contracts.

## 1. Executive decision

The prior review was directionally useful, but it overstated how much is new to
hipEngine and misstated several source details.

The most important conclusions are:

1. **oMLX does not fuse the router softmax.** Its fast router reuses the stock
   composed softmax output and fuses only top-k selection plus selected-score
   renormalization. Reusing the exact selection key is the parity lesson.
2. **oMLX's verify QMM is narrower than “M=2..8.”** Runtime admission is affine
   Q4/Q8, group size 32/64/128, `M=3..6`, `K % 64 == 0`, `N % 4 == 0`, and
   `N >= 16384`. `M=2`, `M=7`, and `M=8` fall back. Its generated Metal is
   lane-strided scalar FMA, not a WMMA implementation.
3. **A dedicated small-row quantized target path is not greenfield here.**
   hipEngine already has exact Q4/Q5/Q6 T16 rowtiles through rows 2..8 where
   qualified, small-row pair/triple kernels, quant-specific LM-head rowtiles,
   target graphs, and direct selected-state commit. The useful experiment is a
   **morphology audit against those owners**, especially the huge-vocabulary
   LM head—not “add skinny-M GEMM” generically.
4. **Two meaningful runtime ideas were omitted:** safe online draft-depth
   selection and streaming prompt priming. hipEngine already has evidence and
   failed prototypes for adaptive budgets, so a retry must use independent
   per-budget graph buckets rather than padded max-shape rows. hipEngine already
   primes the NextN head, but currently retains a full prompt-hidden slab and
   serially catches up the draft head after target prefill; streaming the same
   exact fold may reduce startup wall and transient memory.
5. **oMLX's verify-attention split was omitted.** It collapses a per-row
   head-dim-256 attention loop into causal multi-row vector-kernel chunks. The
   principle corroborates hipEngine's existing small-B verifier-attention work;
   it is not a new primitive family here.
6. **The local oMLX snapshot is stale in a correctness-relevant way.** It is
   `f244ac0842b54a079eac8ff16a15a922a9855daa` from 2026-08-23, while remote
   `main` was `404c059f442fbb09a8a7690789dcf2d80c82b7a3` when reviewed.
   Post-snapshot commits `522e7279` and `5fb02019` repair fused-GDN fallback
   state/offset and sink-entry corruption. Never copy that local integration
   wrapper as-is.
7. **The “~166 tensors at 5-bit” oQ4e description is not a format contract and
   was not found in this snapshot.** oQ/oQe builds a model-specific,
   sensitivity- and byte-budget-driven plan. Any count is artifact-specific.

### Campaign priority

| Rank | ID | Experiment | Decision |
| ---: | --- | --- | --- |
| 0 | `OI-0` | Re-profile the current real verifier by width, shape, and quant family | Required before code |
| 1 | `OI-1` | Small-M morphology audit, beginning with the quantized LM head | Highest-value kernel experiment if `OI-0` confirms it |
| 2 | `OI-2` | Safe adaptive B1/B2/B3 policy over independent exact buckets | Reopen only after transition RED gates |
| 3 | `OI-3` | Streaming exact NextN prompt catch-up | Startup/memory experiment, not an acceptance claim |
| 4 | `OI-4` | Pre-norm versus post-norm draft hidden | Explicit speculative-policy experiment; full suite required |
| 5 | `OI-5` | GDN prework boundary fusion | Profile-triggered only; much of the chain is already fused |
| 6 | `OI-6` | Long-K attention dispatch bounding | Watchlist; Apple preemption motivation does not transfer directly |
| 7 | `OI-7` | Sensitivity-driven mixed quant / MTP-head protection | Separate model-artifact campaign, not a runtime optimization |

## 2. Review provenance and evidence strength

### 2.1 Sources inspected

The local checkout `/home/lhl/omlx` was treated as a read-only reference.

| Item | Value |
| --- | --- |
| Repository | `https://github.com/jundot/omlx.git` |
| Local snapshot | `f244ac0842b54a079eac8ff16a15a922a9855daa` |
| Snapshot subject | `fix(dflash): honor repetition context size (#3011)` |
| Snapshot time | 2026-08-23T20:40:47+09:00 |
| Local clone | shallow, one local commit |
| Remote `main` observed during review | `404c059f442fbb09a8a7690789dcf2d80c82b7a3` |
| License | Apache-2.0 |

A shallow clone can still be cited at its exact HEAD. What it cannot provide is
local ancestry or introduction history. GitHub's commit-by-path API supplied
that history without modifying the read-only clone, so unshallowing was not
needed for this audit.

Primary source surfaces inspected:

- `/home/lhl/omlx/omlx/patches/qwen35_*.py`
- `/home/lhl/omlx/omlx/patches/mlx_lm_mtp/qwen35_model.py`
- `/home/lhl/omlx/omlx/patches/mlx_lm_mtp/batch_generator.py`
- `/home/lhl/omlx/omlx/patches/mlx_lm_mtp/prompt_priming.py`
- `/home/lhl/omlx/omlx/patches/mlx_lm_mtp/cache_rollback.py`
- `/home/lhl/omlx/omlx/custom_kernels/qwen35_prefill/`
- `/home/lhl/omlx/omlx/oq.py`
- the corresponding local tests under `/home/lhl/omlx/tests/`

No Apple hardware was available. No oMLX benchmark or test was run. Performance
numbers below are therefore **donor-reported** source comments or commit
messages, not measurements reproduced by hipEngine.

### 2.2 Evidence labels used below

| Label | Meaning |
| --- | --- |
| `A` | Current source plus a directly relevant local test was inspected. |
| `B` | Current source and introduction/fix commit metadata were inspected. |
| `C` | The number or conclusion appears only in a source comment/docstring or commit message and was not reproduced. |
| `D` | Applicability to hipEngine is an inference to be tested. |

### 2.3 Useful oMLX lineage anchors

These are provenance anchors, not hipEngine parent kernels.

| Feature | oMLX commit anchor |
| --- | --- |
| Native Qwen3.5/3.6 prefill QMM, FA-256, GDN, weighted sum | `d42528afd426f50ee11fbba5243ab004010a8b52` |
| Depth-k MTP, adaptive controller, one-sync verify, verify QMM from MTPLX | `8a9b1972c9b52e1e73a3fa7f76b3c13f894f66d2` |
| Long-K FA-256 chunk/LSE fold | `6d1617495c754873bdad3942e63af5dfd876d56e` |
| MoE gate/up fusion | `5a39ba3a9c28bd8125aa8502d710b89e874efb46` |
| Prompt priming | `27d973efb0d6ce913d0fb64a41b368754d7b4963` |
| Verify GDN prework and verify-attention split | `293d697c2d5a773225891636af75a2b8dd2b8d3f` |
| MoE router top-k fusion | `a9de32ca6f60121b34552e8a3e7102f650df4a3f` |
| Post-snapshot GDN pre-state/offset fallback repair | `522e727937f6dc1abc70b33c324b506d684ad68e` |
| Post-snapshot GDN sink-entry fallback repair | `5fb02019826470a16a02db05bb0c1a8a4d0ba16a` |

If implementation copies source rather than only the idea, preserve the
Apache-2.0 notices and the more specific donor attribution in
`qwen35_verify_qmm.py` (MTPLX / Youssof Altoukhi) and
`qwen35_gdn_prework.py` (mlx-serve and mlxfast-challenge). The hipEngine commit
must name the exact donor file and commit.

## 3. Claim-by-claim sanity check

| Prior claim | Verdict | Corrected reading |
| --- | --- | --- |
| oMLX is an MLX-backed serving server | Confirmed (`A`) | Its custom Qwen work is spread across model patches, MTP scheduler/cache code, and native Metal extensions—not only two files/directories. |
| The router collapses softmax + top-k + gather + renormalize into one kernel | **Incorrect** (`A`) | Gate projection and full softmax remain composed. The fused kernel receives that exact softmax output and replaces top-k selection plus selected-score renormalization. Source says the old post-gate chain is five tiny ops and reports ~51 us/layer versus ~5 us fused; those timings are donor comments (`C`). |
| Reusing composed softmax output makes expert selection bit-identical | Qualified (`A`) | The local test checks selected **sets** over random/tie cases and pins MLX's observed highest-index tie behavior. Scores are allowed to differ by reduction order. This is not a universal tie rule. |
| Gate/up concatenation is bit-identical and removes one routed launch | Confirmed in donor scope (`A/B`) | The test checks decode, sorted prefill, VLM verify, dense and affine-quantized `SwitchGLU` outputs with `array_equal`. oMLX drains the MLX pool per fused layer to bound load-time transient memory. hipEngine already has pair and whole-FFN owners. |
| Weighted sum consumes sorted output without scatter | Confirmed (`A`) | Prefill only, default `>=1024` tokens, top-k 6/8, no sharding, no target verify. The test allows max absolute difference `<=2e-2`; do not call it bit-exact. |
| Native affine prefill QMM has “about ten” tile variants | Confirmed with precision (`A`) | There are ten `(BM,BK,BN)` geometries per dtype/bit family for group 64 and again for group 128, bits 2/4/5/6/8. The Python route defaults to `>=2048` rows, while Q8 defaults to `>=16384`. This is MLX affine packing, not GGUF T16. |
| Verify skinny-M QMM handles the MTP dead zone | Confirmed but narrower (`A/B/C`) | Runtime admission is `M=3..6`; split-K uses 2 or 4 simdgroups and the huge-N path uses eight independent simdgroups with four columns each. `N>=100000` selects the huge-N morphology. `N>=16384` is the global route floor. |
| The verify QMM maps directly to HIP/WMMA | Overstated (`A/D`) | The scheduling morphology maps; the implementation does not. It uses generated scalar FMA loops over MLX affine words. hipEngine needs quant-specific Q4/Q5/Q6 GGUF T16 math and must compare against existing rowtiles. |
| Verify QMM numerics are exact | Incorrect (`A`) | The source explicitly allows BF16 tail-ULP drift and occasional greedy divergence. No direct `vk_qmm` test exists in the local test tree. Donor end-to-end claims do not replace a hipEngine RED gate. |
| MTP GDN verify runs the full window once and replays only on reject | Confirmed (`A`) | It stores pre-forward state references and projected inputs, commits full accept without replay, and replays the kept prefix on partial accept. hipEngine's transaction journal is stronger in the qualified paths because it materializes per-row state and directly commits the selected row without accepted-prefix replay. |
| Fused GDN prework replaces roughly ten small ops | Confirmed primitive; stale integration (`A/B/C`) | The local primitive test is bit-exact at S=3/4/5/7/9. The local integration wrapper predates two upstream fallback fixes and can reuse mutated Conv state, double-advance offsets, or duplicate rollback sink entries after failure. |
| FA-256 chunks K and folds partials with LSE | Confirmed (`A/B`) | It is an Apple command-buffer robustness fix, auto-calibrated around a donor 10 ms dispatch target, with a 2 GiB temporary partial-slab cap. It is not evidence that AOTriton needs the same policy on ROCm. |
| Verify attention has no additional idea | **Missed feature** (`A/B`) | `qwen35_verify_sdpa_split.py` groups causal verify rows within MLX's `q_len * gqa <= 32` vector-kernel budget instead of issuing one SDPA per row. Donor tests cover q_len 2/4/5/6/7/9 and KV 512/2048. |
| oQ4e is 4-bit g64 with ~166 selected 5-bit tensors | Unsupported as a general claim (`A`) | oQ/oQe uses model-specific sensitivity, protection floors, and a byte-budget allocator. oQ4 targets about 4.6 bpw; boosted tensor counts vary by model and calibration. oQe adds GPTQ-style optimization and imatrix-aware clipping, not a fixed 166-tensor recipe. |
| A shallow clone prevents commit lineage | Incorrect | Exact snapshot citation was always possible. Introducing/fix commits were recoverable through remote path history without changing the clone. |

## 4. Complete technique inventory and hipEngine disposition

### 4.1 MoE and projection paths

| Technique | oMLX source | Evidence | hipEngine mapping | Disposition |
| --- | --- | --- | --- | --- |
| Top-k + selected softmax/renorm fusion | `patches/qwen35_moe_router.py` | `A/B/C` | `kernels/hip_gfx1100/moe/router.{hip,py}` already has deterministic block-parallel selection and cooperative router+top-k+shared variants; proposer specialization is already retained. | Covered. Add only tie-contract tests where missing; do not import MLX's highest-index rule. |
| Reuse exact composed selection key | same | `A` | CPU Qwen oracle uses stable lower expert IDs on ties; HIP reduction also keeps the incumbent lower index. | Keep the **principle**, not donor tie semantics. Any alternate key must be compared to the current strict key before selection. |
| Gate/up output-axis packing | `patches/qwen35_moe_gate_up.py` | `A/B` | Dense/selected pair kernels and whole selected FFN megakernels already reuse activations and contract launches. | Covered and often exceeded. |
| Per-layer pool drain after weight rewrite | same | `B` | hipEngine uses bounded materialized layouts rather than MLX post-load concatenation. | No direct action. Revisit only if a new materializer temporarily duplicates a large weight family. |
| Scatter-free sorted weighted sum | `patches/qwen35_moe_weighted_sum.py`; native primitive in `qwen35_prefill.cpp` | `A` | `fused/paro_combine`, selected T16 weighted variants, and whole FFN owners avoid the same expanded scatter where qualified. | Covered. |
| Shape-first fast rejection in hot wrappers | all qwen35 prefill patches; `af91f86` | `B/C` | `REFACTOR.md` RF-1 already requires cold-resolved immutable plans; verifier object/lookup/scratch caches have retained wins. | Cross-link RF-1. Do not add more runtime env reads in this campaign. |

### 4.2 Quantized matrix paths

| Technique | oMLX source | Evidence | hipEngine mapping | Disposition |
| --- | --- | --- | --- | --- |
| Steel tiled affine prefill QMM | `patches/qwen35_q4_mlp.py`; `custom_kernels/qwen35_prefill/csrc/qwen35_qmm.metal` | `A/B` | hipEngine already has Q4/Q5/Q6 T16 WMMA/rocBLAS prefill with shape-scoped policies and strict fallbacks. | Covered; donor thresholds reinforce measured, quant-specific admission. |
| M-templated verify accumulators | `patches/qwen35_verify_qmm.py` | `A/B` | Existing exact Q4/Q5/Q6 rowtiles are templated through rows 2..8 where qualified. | Covered conceptually. Audit morphology, not existence. |
| K split across 2/4 wave groups with LDS reduction | same | `A/D` | Some hipEngine projections remain one-wave rowtiles; split-K changes association unless designed carefully. | `OI-1` production/T2 candidate after exact baseline. |
| Barrier-free multi-wave output tiles for huge N | same | `A/D` | Most relevant to the ~248K-vocabulary quantized LM head. Current Q6 FP32 rowtiles have quant/backend-specific row caps and chunking. | `OI-1` first microbenchmark target. |
| Route floor for large N only | same | `A/C` | hipEngine already uses capability maps and primitive max-row metadata. | Use measurements; never copy `N>=16384` as a universal threshold. |
| Verify-only routing scope | thread-local arming in oMLX | `A` | hipEngine should resolve a verifier registry variant/manifest, not monkey-patch a global linear class. | Principle retained through plugin dispatch. |

### 4.3 MTP model, scheduler, and state

| Technique | oMLX source | Evidence | hipEngine mapping | Disposition |
| --- | --- | --- | --- | --- |
| Depth-k head chaining with returned hidden | `mlx_lm_mtp/qwen35_model.py`; `batch_generator.py` | `A/B` | Native NextN proposal and B1/B2/B3 target graphs are implemented. | Covered. |
| Project only needed logits rows (`logits_keep`) | `qwen35_model.py` | `A` | hipEngine proposer and verifier have streaming top-1 / bounded result paths and avoid full-logit D2H outside diagnostics. | Covered and exceeded. |
| Full-window GDN verify, rejection-only replay | `qwen35_model.py` | `A/C` | Exact Conv/GDN chain journals plus selected-state commit avoid replay in retained transaction paths. | Covered and exceeded. |
| Rotating-cache undo log scoped to MTP | `cache_rollback.py` | `A` | Scheduler-owned KV transaction/journal and `KVLiveSpans` are normative. | Covered architecturally. Preserve the lesson that post-mutation fallback must restore every surface or fail the request. |
| One host synchronization per cycle | `batch_generator.py` | `A/B` | N3P proposal completion event plus N2 target graph can retire under one target synchronization. | Covered in admitted graph scopes. |
| In-graph greedy and stochastic acceptance | `batch_generator.py` | `A/B` | Greedy GPU acceptance is implemented; public non-greedy MTP currently rejects/falls back. | Stochastic support is out of this campaign unless separately requested. |
| Sharper stochastic draft sampler with exact residual correction | same | `A/C` | Sampling-policy change is T3 and the current product path is greedy-only. | Deferred. Do not mix with kernel work. |
| Online adaptive depth with EMA cost/acceptance and B0 parking | same | `A/B/C` | hipEngine has fixed-budget sweeps, oracle headroom, and failed dynamic prototypes; current independent B1/B2/B3 graph buckets create a safer prerequisite than the old padded-row design. | `OI-2`, with no true-AR parking in the first slice. |
| Prompt priming during target prefill | `prompt_priming.py` | `A/B/C` | hipEngine already primes shifted NextN state exactly, but stores all target hidden rows and catches up one row at a time after target prefill. | `OI-3` streaming implementation candidate. |
| Post-norm target hidden for the draft head | `prompt_priming.py` | `A/C` | hipEngine currently captures pre-output-norm target hidden as part of its strict GGUF/llama contract. | `OI-4` explicit provider-policy A/B only. |
| Per-key MTP norm-convention repair | `qwen35_model.py` | `A/C` | GGUF conversion/loading has a different weight convention and existing model-hash/quality gates. | No direct runtime action. Useful only in a future converter audit. |

### 4.4 Attention and GDN paths

| Technique | oMLX source | Evidence | hipEngine mapping | Disposition |
| --- | --- | --- | --- | --- |
| Verify-row causal SDPA grouping | `patches/qwen35_verify_sdpa_split.py` | `A/B/C` | Small-B full-attention batching, causal row limits, shared-page attention, and split-K batch variants already exist. Long strict fallback deliberately executes rows serially after a proven rounding drift. | Covered principle. Re-profile before any new attention kernel. |
| Fused GDN verify prework | `patches/qwen35_gdn_prework.py` | primitive `A`; integration caveat `B` | HIP Conv already fuses depthwise Conv+SiLU; GDN kernels fuse Q/K normalization, recurrence, output RMSNorm, gate, optional BF16 cast, and snapshot. They remain separate launches. | `OI-5` only if fresh profile shows boundary cost. |
| Blocked sequential GDN prefill | `patches/qwen35_gdn_chunked.py`; `custom_kernels/qwen35_prefill/gdn.py` | `A/C` | Compact-peer/segmented GDN prefill and exact fallbacks are heavily qualified in-tree. | Covered. |
| Long-K FA-256 chunks plus LSE fold | `patches/qwen35_fa256_attention.py`; `qwen35_prefill.cpp`; shared Steel header | `A/B` | AOTriton owns long prefill; native decode/DMS attention already uses split producers plus deterministic online-softmax reducers. | `OI-6` watchlist only. |
| Auto-calibrated dispatch-wall budget | `qwen35_fa256_attention.py` | `A/B/C` | HIP does not share macOS's IOGPU interactivity threshold. Runtime self-calibration also complicates deterministic manifests. | Do not port generically. Use fixed artifact-qualified buckets if a ROCm problem is measured. |
| Ragged decode threadgroup-limit probe | `patches/qwen35_ragged_decode.py` | `A` | Metal-specific safety wrapper; HIP launchers already validate supported shapes/threads. | Not applicable. |

### 4.5 Quantization strategy

| Technique | oMLX source | Evidence | hipEngine mapping | Disposition |
| --- | --- | --- | --- | --- |
| Position/sensitivity-driven byte-budget allocation | `oq.py`; `docs/oQ_Quantization.md` | `A/C` | hipEngine primarily consumes fixed PARO/GGUF artifacts and separately gates quant quality. | `OI-7`, separate artifact campaign. |
| MTP-head calibration pass | `oq.py` | `A/B` | Relevant only if hipEngine produces quantized NextN artifacts. | Deferred but worth preserving. |
| Full-precision fusion projection and minimum 4-bit MTP head | `oq.py`; `qwen35_model.py` | `A/B/C` | Could improve proposal acceptance with small size cost, but changes the model artifact. | T3 artifact experiment, not a kernel default. |
| Weighted clipping / GPTQ-style optimization | `oq.py` | `A/C` | Quality/size work, not a direct same-artifact speed optimization. | Out of the initial campaign. |

## 5. Campaign rules

Every experiment below follows these rules in addition to the repository-wide
instructions.

1. **One physical host per comparison.** Never compare donor Apple numbers with
   W7900 numbers or transfer absolute rates between gfx1100 and gfx1151.
2. **One model artifact per A/B.** A runtime/kernel candidate uses the same file,
   quant, KV format, prompts, and execution profile as its control.
3. **True AR denominator.** MTP speedup always uses the true no-MTP AR path from
   the same protocol. Verifier `off`/B0 diagnostics are not denominators.
4. **Full suite, never one prompt.** Use all categories in
   `benchmarks/prompts/mtpbench-code-general-ja.jsonl` plus heldouts and the
   applicable long/task gates. Fixed-prompt tuning is invalid.
5. **Strict fallback remains registered.** Every fused or T1/T2 candidate names
   the exact current primitive/chain fallback before implementation.
6. **Profile class is declared first.** T0 must pass its exact/parent-parity RED
   contract. T1/T2 must pass the complete production envelope. T3 is an
   explicit provider/model-policy experiment and is never relabeled as a
   production arithmetic optimization.
7. **Transactional ownership is exact.** Request, position, parent row,
   `KVLiveSpans`, Conv/GDN state, target/draft KV, acceptance, commit, rollback,
   cursor, output room, and lifecycle checks are binding for every candidate.
8. **No post-launch retry on uncertain mutation.** A HIP/C-ABI failure after
   possible submission fails the owned transaction and invalidates the
   affected graph/session as required. It does not silently execute the old path
   a second time.
9. **Cold-resolved dispatch.** New retained behavior enters through the
   `(backend, layer, quant, variant)` registry and immutable model/session plan.
   Do not add global monkey patches, process-wide “armed” flags, or repeated hot
   environment reads.
10. **Evidence rollup only after retention.** Rejected diagnostics get a compact
    artifact/worklog entry but no topline. A retained performance win updates
    `benchmarks/README.md`, `benchmarks/CHANGELOG.md`, and a compact artifact.

## 6. `OI-0` — establish the current verifier baseline

**Goal:** determine whether the next recoverable full-cycle milliseconds are in
quantized projections/LM head, full attention, GDN, proposal, transaction glue,
or another family.

This phase changes no runtime code.

### Required lanes

Keep these separate:

- W7900 / Qwen3.6-27B dense `Q4_K_M`, exact/default B1/B2/B3;
- W7900 / Qwen3.6-35B-A3B `UD-Q4_K_M`, explicit exact and, when relevant,
  separately labeled `llama-compat`;
- gfx1151 only after an independent baseline on that physical host.

The current scoreboard is a reference, not the new baseline: W7900 dense 27B
reports 29.457 true AR and 60.929 B3 MTP; W7900 35B-A3B reports 96.75 true AR
and 122.67 explicit accuracy-traded MTP-2. Rerun on the current commit before
using either as a campaign denominator.

### Measurements

1. Run the canonical true-AR/category and MTP/category protocol from
   [`benchmarks/MTP.md`](../benchmarks/MTP.md) and [`BENCHMARK.md`](BENCHMARK.md).
2. Profile the **final verifier child** with
   `scripts/gguf_mtp_verifier_rocprof.py`; never wrap the economics parent in
   `rocprofv3`.
3. Record per B1/B2/B3:
   - complete cycle wall and target-only wall;
   - proposal, accept/commit/repair, and synchronization/readback windows;
   - LM-head family and shape;
   - Q4/Q5/Q6 projection families and actual `(M,K,N)`;
   - full-attention producer/reducer family and context;
   - Conv/GDN family;
   - kernel count, copies, synchronizations, VGPR/LDS/scratch for top families;
   - acceptance and visible tokens/cycle by category.
4. Reconcile at least 90% of complete target wall before selecting a kernel
   family. An unreconciled residual remains an upper bound, not “Python time.”

### Exit decision

- Proceed to `OI-1` only if the LM head or another small-M quantized projection
  family has material recoverable wall under the real verifier.
- If attention or GDN dominates, reorder `OI-5`/`OI-6` using the measured
  result.
- If the candidate width is already using one exact weight-once rowtile and is
  near its measured roof, do not create a duplicate kernel merely because the
  oMLX morphology differs.

## 7. `OI-1` — small-M quantized projection morphology audit

**Hypothesis:** for selected verifier shapes, an oMLX-inspired output-tiled
multi-wave or split-K schedule can beat the current exact T16 rowtile/LM-head
owner by improving occupancy and scheduler behavior while reading each weight
plane no more often.

### 7.1 Start with the LM head

The donor's multi-simdgroup path is most compelling at huge N. hipEngine's
quantized vocabulary projection is therefore the first candidate, not every
linear layer.

Audit exact live shapes and current owners before coding:

- model/quant/layout (`Q4_K_M`, standard or planar Q6 T16, etc.);
- verifier rows B1/B2/B3 => `M=2/3/4`, then diagnostic rows through 8;
- hidden K and vocabulary N;
- current primitive's `_hipengine_max_rows` and backend package cap;
- whether the current path writes full logits, row top-1, or a bounded accept
  payload;
- whether one launch already reads the head once across all rows.

### 7.2 Candidate ladder

Implement one rung at a time:

1. **T0 exact row-shared candidate:** M-templated row accumulators and
   output-column tiling while preserving each row's existing K traversal,
   dequant order, FP32 accumulation, output boundary, and lower-ID tie rule.
2. **T0 exact multi-wave output partition:** independent waves own disjoint N
   tiles and full K; no cross-wave reduction. This is the closest HIP analogue
   to the donor huge-N path and should be attempted before split-K.
3. **T2 split-K candidate:** 2/4 waves own K partitions and reduce through LDS.
   Declare the reduction association change and use the production-profile
   gate; do not call it strict.
4. **WMMA candidate only when justified:** GGUF dequant/layout and M<=8 may make
   scalar/dot4 rowtiles better. WMMA is not implied by the donor source.

Do not combine LM-head fusion, acceptance logic, and a new projection schedule
in the first RED/GREEN unit. First prove the projection bytes; then separately
consider a top-1/accept epilogue if the current path still materializes data the
transaction never consumes.

### 7.3 RED and primitive gates

Before implementation, add fixtures for:

- every claimed M (minimum 2/3/4; extend through 8 only if routed);
- exact real K/N and tail/alignment cases;
- standard and planar layouts only when independently claimed;
- random inputs plus near-tie logits;
- current primitive bytes/top-1 as the strict parent;
- lower token ID on exact ties;
- invalid-layout/width fail-closed behavior.

A strict candidate requires bit equality. A T2 candidate requires full-logit
teacher rows and the binding execution-profile mean/p95/p99/max KL, top-1,
determinism, isolation, task, and lifecycle gates.

### 7.4 Performance gate

- Matched microbenchmark current owner versus candidate on the real weight.
- Cached `rocprofv3` proves the expected kernel name, launch geometry, positive
  duration, VGPR/LDS/scratch, and no hidden fallback.
- Complete B1/B2/B3 target wall improves or is at least non-regressive in every
  routed scope.
- Complete category + heldout MTP economics are non-regressive against the
  current verifier and improve against the same true-AR denominator.
- Any exact, measured, same-suite non-regressive win is retained in its exact
  shape/backend scope; there is no arbitrary minimum percentage.

### 7.5 Stop conditions

Stop or narrow scope if:

- the candidate rereads weights more often than the current rowtile;
- a “multi-wave” path is only multiple current GEMVs under a new wrapper;
- split-K reduction fails the production envelope;
- primitive speed does not survive target wall;
- one width wins but another loses and dispatch cannot exclude the loser
  exactly;
- registry or profile provenance is missing.

## 8. `OI-2` — safe adaptive B1/B2/B3 policy

**Hypothesis:** a content-agnostic online controller can select among exact
B1/B2/B3 buckets using observed conditional acceptance and measured cycle wall,
improving full-suite economics over the best fixed budget without prompt- or
token-specific policy.

This idea is corroboration, not a blank slate. [`MTP.md`](MTP.md) already
records:

- fixed-budget B1/B2/B3 sweeps;
- about 1.027x total-time oracle headroom from a prohibited prior-outcome prompt
  map;
- a whole-cycle confidence gate that was exact but economics-negative;
- a padded max-shape active-budget path that was slow and non-exact;
- a live B1->B2/B3 prototype that faulted/hung and was removed;
- no exactly replayable simple ladder policy in the retained traces.

The old failure means **do not** port oMLX's controller directly into the current
loop.

### 8.1 Prerequisite: variable-budget transition contract

First create a RED-only transition matrix over independent fixed buckets:

- B1->B1, B1->B2, B1->B3;
- B2->B1, B2->B2, B2->B3;
- B3->B1, B3->B2, B3->B3;
- reject, every partial accept, and full accept at each edge;
- page/context bucket transitions, output tails, reset, cancellation, and
  subsequent AR health;
- proposal cache, target graph, Conv/GDN journal, full-attention KV, selected
  hidden, positions, and result payload.

Use a distinct cached graph/scratch owner per `(B, context bucket, manifest)`.
Never emulate B1 with inactive rows in a B3 graph. Do not add the controller
until this matrix is exact and teardown-clean.

### 8.2 Controller model

A first controller may copy the donor's **policy shape**, not its constants:

```text
p[j]      = EMA of conditional acceptance at depth j
t[B]      = EMA of complete cycle wall when B ran
visible(B)= 1 + p[0] + p[0]p[1] + ...
score(B)  = visible(B) / t[B]
```

Requirements:

- inputs are only online aggregate acceptance, cycle wall, current qualified
  bucket availability, output room, and lifecycle state;
- no prompt names, categories, text, token IDs, or heldout outcomes;
- bounded exploration, hysteresis, and deterministic update order;
- controller state is request-owned and emitted in diagnostics without prompt
  content;
- first slice chooses only B1/B2/B3. It does **not** park to verifier B0.

True AR parking is a later stage. It requires a proved exact MTP->AR handoff,
proposer/head-history maintenance or explicit invalidation, and safe re-entry.
A verifier `B0` timing row is not true AR and cannot drive the decision.

### 8.3 Gates

- Transition matrix above passes before economics.
- Fixed-seed controller repeats choose the same B sequence and output IDs.
- Full category + heldout suite remains exact to the declared strict teacher.
- Per-category and aggregate economics beat the best fixed policy or the
  controller is rejected; categories cannot compensate for one another.
- Compare against true AR from the same protocol.
- No dynamic graph capture occurs after a proposal is in flight.

## 9. `OI-3` — streaming exact NextN prompt priming

**Hypothesis:** consume target hidden chunks into the NextN cache as target
prefill progresses, preserving the current shifted-prompt contract while
removing the full `prompt_len * hidden` temporary and the post-prefill serial
catch-up loop.

This is not the donor's acceptance rescue: hipEngine already primes the draft
head. The current implementation in `runtime/qwen35_gguf_mtp.py` captures all
target trunk-hidden rows, then calls the draft executor once per prompt token.
The expected benefits are startup wall, overlap opportunity, and transient
memory only.

### 9.1 Exact streaming contract

For prompt tokens `t[0..P-1]` and target hidden rows `h[0..P-1]`, preserve the
current sequence exactly:

- draft row 0 consumes token `t[0]` with zero hidden;
- draft row `i>0` consumes token `t[i]` with target hidden `h[i-1]`;
- the final pending target hidden is retained for the activation seam if the
  existing contract needs it;
- head positions/cache offsets match the non-streaming control.

Implement a batch/chunk sink owned by the target prefill plan. Do not add a
scalar-only engine-side prompt path or a model-global context shared across
requests.

### 9.2 RED gates

- one-shot versus chunk partitions including 1, 2, boundary-1, boundary,
  boundary+1, and ragged final chunks;
- warm prefix / nonzero starting offset;
- cancellation between chunks;
- request interleaving and owner mismatch fail closed;
- exact draft KV/cache offsets and bytes;
- exact first proposal, first target verify, full generated IDs, and acceptance;
- zero leaked temporary bytes after close.

### 9.3 Performance/memory gate

Measure 512/4K/16K prompts on the same W7900/model:

- target prefill tok/s;
- total MTP prefill/activation wall and TTFT;
- peak tracked/HIP/whole-device memory;
- decode and acceptance unchanged after activation.

Retain if exact and non-regressive, even if the primary win is bounded memory or
startup wall rather than steady decode.

## 10. `OI-4` — draft hidden pre-norm versus post-norm

**Hypothesis:** feeding consistently post-final-norm target hidden to the NextN
head may improve proposal agreement/acceptance, as reported in oMLX, without
changing target semantics.

This is an explicit speculative-provider policy (`T3`), not an execution-profile
implementation change. The model was converted and qualified around a current
pre-output-norm/llama-compatible contract, so donor comments are insufficient.

### Candidate rules

- Add an immutable provider manifest field such as
  `draft_hidden_variant=pre_output_norm|post_output_norm`.
- Apply the same variant during prompt priming/catch-up, steady proposal seed,
  accepted-row reseed, rollback/repair, and any AR handoff.
- Do not change target hidden/state commit or target logits.
- Keep the current variant as strict fallback/control.

### Gates

- Direct CPU/small fixture proves the chosen normalization and no accidental
  double norm.
- Full multi-prompt + heldout suite reports draft acceptance by depth/category,
  visible tokens/cycle, target verify rows, complete MTP wall, and true-AR
  ratio.
- Greedy generated IDs and target acceptance agree with the strict target for
  the complete claimed horizon.
- Long context, tails, reset, cancellation, and subsequent AR health pass.
- Retain only if same-suite economics or quality improves without a failed
  category. Do not tune the choice per prompt.

## 11. `OI-5` — profile-triggered GDN prework boundary fusion

**Hypothesis:** if fresh profiling shows a material boundary cost between the
existing Conv+SiLU producer and GDN Q/K normalization/recurrent consumer, a
registered composite may reduce launches and intermediate traffic.

Do not describe this as porting oMLX's full prework kernel. hipEngine already
fuses more downstream math inside GDN and has exact chain journals/snapshot
writers. The candidate boundary and grid geometry are different.

### Preconditions

- `OI-0` attributes a material target-wall share to the Conv/GDN boundary, not
  just recurrence arithmetic.
- Current two-kernel producer/consumer bytes and launch duration are known.
- A fused grid can preserve state order and has enough cross-head/channel
  parallelism without excessive LDS/VGPR.

### Correctness requirements

- RED fixture covers rows 2..8, B1/B2/B3 transaction outcomes, state snapshots,
  BF16 handoff, and exact selected commit.
- Strict candidate preserves all existing rounding points and bytes.
- A changed SiLU/RMS/reduction implementation is T1/T2 and requires the full
  production gate.
- The old Conv and GDN primitives stay registered as the strict unfused chain.
- Failure handling follows the post-snapshot oMLX lesson: never fall back with a
  mutated Conv state, advanced cursor, or duplicate journal entry.

Stop if launch removal raises complete target wall, as several prior in-tree
fusion experiments have done.

## 12. `OI-6` — long-context attention watchlist

The donor chunks key ranges because a long Metal dispatch can trigger macOS
IOGPU interactivity demotion/termination. That mechanism is not established on
Linux/ROCm W7900.

hipEngine already has:

- AOTriton causal head-dim-256 prefill;
- native split-K decode and verifier attention;
- deterministic partial-max/sum reducers;
- grouped-GQA KV reuse;
- context-bucketed target graphs and bounded workspaces.

Open this experiment only if a current same-host profile shows one of:

- an individual AOTriton/native prefill dispatch violates an operational wall
  or watchdog bound;
- long-context prefill throughput has a dispatch-duration cliff not explained
  by ordinary O(N^2) work;
- AOTriton cannot serve a required shape and a custom kernel is approved;
- partial-slab memory or scheduling can be improved by a measured K partition.

If triggered, reuse the standard online-softmax merge equations already present
in hipEngine rather than copying Metal. Bucket sizes must be artifact-qualified
and deterministic; runtime self-calibration does not become an unrecorded
variant selector.

## 13. `OI-7` — separate quantization-artifact campaign

oQ's strongest transferable ideas are quality/size policy rather than immediate
runtime speed:

- calibration-driven sensitivity instead of a fixed layer-position rule;
- byte-budgeted bit allocation;
- MTP-head calibration coverage;
- protecting fusion projections that collapse acceptance when quantized;
- a minimum precision floor for the small draft head;
- per-group weighted clipping/GPTQ optimization.

hipEngine consumes PARO/GGUF artifacts and has strict quant-quality protocols.
Adopting these ideas requires a separate converter/artifact proposal with:

- source and output model hashes;
- exact per-tensor quant/layout manifest;
- model size/bpw and resident-memory accounting;
- BF16-relative and same-quant task gates;
- full MTP acceptance/economics plus true AR;
- loader/registry support without hot-path quant branches.

There is no generic “raise 166 tensors to 5-bit” experiment. Tensor count and
bits must come from the declared model's measured plan.

## 14. Campaign scorecard

Update this table as atomic units land. A blank metric is not a pass.

| ID | State | Host/model/quant | Strict/quality gate | Primary metric | Decision artifact/worklog |
| --- | --- | --- | --- | --- | --- |
| `OI-0` | pending | W7900 dense 27B first | existing exact category/heldout gate | reconciled target/cycle wall by family | — |
| `OI-1` | blocked by `OI-0` | exact shape selected by profile | T0 exact or T2 full production gate | primitive + target + full-suite MTP/AR | — |
| `OI-2` | blocked by transition RED | W7900 dense B1/B2/B3 | exact variable-budget transaction matrix | controller versus best fixed B and true AR | — |
| `OI-3` | pending after baseline | W7900 dense prompt 512/4K/16K | exact prompt-head cache and generated IDs | TTFT/prefill wall/peak memory | — |
| `OI-4` | pending | W7900 dense | explicit T3 full category/heldout/long gate | acceptance and complete MTP/AR | — |
| `OI-5` | blocked by profile trigger | shape selected by `OI-0` | exact chain or full production gate | Conv/GDN boundary and target wall | — |
| `OI-6` | watchlist | long prefill only | AOTriton/native parity + task gate | dispatch wall/throughput/memory | — |
| `OI-7` | separate campaign | model artifact TBD | BF16-relative quant/task/MTP gate | quality/size/speed | — |

## 15. Final review conclusions

- The prior review's highest-value intuition—small-M verifier specialization—is
  sound, but hipEngine has already implemented the basic rowtile idea. The next
  test is output/split morphology on profiled real shapes, especially LM head.
- The most valuable omitted oMLX runtime ideas are adaptive per-sequence budget
  selection and streaming prompt priming. Both need hipEngine-specific
  transaction work; neither is a drop-in port.
- The verify-attention, one-window GDN, one-sync cycle, scatter-free MoE, router,
  and gate/up lessons are already represented in hipEngine, often with stronger
  exact transaction ownership.
- The donor's Apple-specific long-dispatch and NAX/ANE choices are not RDNA3
  evidence.
- Donor performance comments and commit messages are hypotheses only until the
  exact hipEngine protocol measures them on the named physical host.
- Before any implementation, re-check remote oMLX history for the exact donor
  path and run `python3 scripts/check_lineage.py --kind kernel --diff stat` for
  the in-tree family. The reviewed local snapshot must not be treated as latest.
