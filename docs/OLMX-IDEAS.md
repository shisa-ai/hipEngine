# oMLX Ideas Campaign

- **Status:** gfx1151 `OI-0`/`OI-1` complete; `OI-2` transitions pass, controller rejected
- **Created:** 2026-08-25
- **Requested filename:** `OLMX-IDEAS.md` (the project reviewed is spelled **oMLX**)
- **Primary target:** Qwen3.x MTP on `hip_gfx1151` / Radeon 8060S; `OI-0` starts with Qwen3.8-27B `Q4_K_M`
- **Secondary target:** independently qualified `hip_gfx1100`; no result transfers across hardware lanes
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
   **morphology audit against those owners**, not “add skinny-M GEMM”
   generically. `OI-0` later selects the standard-Q4 pair/single rowtiles over
   the huge-vocabulary LM head on the active gfx1151 artifact.
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
| 0 | `OI-0` | Re-profile the current real verifier by width, shape, and quant family | Complete on gfx1151 |
| 1 | `OI-1` | Small-M morphology audit, beginning with standard-Q4 dense pair/single rowtiles | Selected by `OI-0` |
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
| Barrier-free multi-wave output tiles for huge N | same | `A/D` | Most relevant in isolation to the ~248K-vocabulary quantized LM head. Current Q6 FP32 rowtiles have quant/backend-specific row caps and chunking. | Second rung after `OI-0` measured only 4.5-4.8% target-wall share. |
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

- Radeon 8060S/gfx1151 / Qwen3.8-27B dense `Q4_K_M`, exact/default
  B1/B2/B3—the first active lane;
- Radeon 8060S/gfx1151 / Qwen3.6-35B-A3B `UD-Q4_K_M`, explicit exact and,
  when relevant, separately labeled `llama-compat`;
- gfx1100 only after an independent baseline on that physical host.

The exact first-lane file is
`/models/gguf/Qwen3.8-27B-Q4_K_M.gguf`, SHA-256
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
The current local `/models/gguf/Qwen3.6-27B-Q4_K_M.gguf` has no trailing NextN
block and cannot supply a B1/B2/B3 lane; do not silently substitute it. Historical
scoreboard rows are references, not new baselines. Rerun on the current commit
before using any rate as a campaign denominator.

### Measurements

1. Run the canonical true-AR/category and MTP/category protocol from
   [`benchmarks/MTP.md`](../benchmarks/MTP.md) and [`BENCHMARK.md`](BENCHMARK.md).
2. Profile the **final verifier child**, never the economics parent. For the
   dense lane run `scripts/qwen36_dense_gguf_suite.py --limit 1
   --roctx-markers` directly under `rocprofv3` after an unprofiled cache warmup;
   `scripts/gguf_mtp_verifier_rocprof.py` remains a MoE-oriented reference.
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

### gfx1151 result — 2026-08-25

The clean `f1c16ebbb` Qwen3.8-27B `Q4_K_M` natural25 run is exact for all ten
prompts and all B1/B2/B3 GPU/CPU acceptance decisions:

| Route | tok/s | versus true AR | Draft acceptance | Target share of decode |
| --- | ---: | ---: | ---: | ---: |
| true AR | 11.7119 | 1.0000x | — | — |
| B1 | 17.1878 | 1.4675x | 86.26% | 91.15% |
| B2 | 20.0909 | 1.7154x | 72.00% | 86.97% |
| B3 | **21.0620** | **1.7983x** | 63.10% | 83.78% |

B3 wins aggregate and every budget beats AR in every category. B2 nevertheless
beats B3 in `general_en` (**19.826 versus 18.059 tok/s**), preserving a real
but bounded `OI-2` premise.

Cached native-gfx1151 B1/B2/B3 traces reconcile **100%** of each target marker
as device-busy intervals plus measured internal queue gaps and pre/post device
margins. Kernel sums explain 91.28%/82.51%/81.27%; internal single-stream queue
gaps explain 8.16%/17.03%/18.23%. Standard-Q4 dense pair plus single rowtiles
own **53.01%/46.27%/45.06%** of target wall. The Q6 LM-head float rowtile is
only **4.84%/4.61%/4.52%**, GDN is 2.27%/2.74%/3.10%, and full attention is
0.55%/0.52%/0.53%.

Decision: open `OI-1` on the standard-Q4 rows 2/3/4 owners, not the LM head.
Do not open `OI-5` or `OI-6`; neither profile trigger fired. Evidence:
[`2026-08-25-gfx1151-qwen38-omlx-oi0-baseline.json`](../benchmarks/results/2026-08-25-gfx1151-qwen38-omlx-oi0-baseline.json).

## 7. `OI-1` — small-M quantized projection morphology audit

**Hypothesis:** for selected verifier shapes, an oMLX-inspired output-tiled
multi-wave or split-K schedule can beat the current exact T16 rowtile/LM-head
owner by improving occupancy and scheduler behavior while reading each weight
plane no more often.

### 7.1 Start with the profiled standard-Q4 owners

`OI-0` overrules the pre-profile LM-head guess. The first candidates are the
standard-Q4 `K=5120, N=17408` gate/up+SiLU pair and the live standard-Q4 single
projection shapes at verifier rows 2/3/4. Together they own 45-53% of target
wall. The current pair is a 64-thread/two-wave weight-per-wave rowtile; the
single path is a 32-thread/eight-column rowtile. Both already reuse weights
across rows, so a candidate needs a genuinely different output/wave morphology,
not another wrapper around the same work.

The Q6 `K=5120, N=248320` target LM head remains a second-rung audit. It already
uses one exact weight-once float rowtile per target cycle and owns only 4.5-4.8%
of target wall.

Audit before coding:

- exact standard-Q4 pair and single `(M,K,N)` call frequencies and shape roles;
- verifier rows B1/B2/B3 => `M=2/3/4`, including B3's physical tail row;
- current rowtile K ownership, per-thread FMA order, ordered merge, BF16
  gate/up+SiLU boundary, lower-ID tie rule where applicable, VGPR/LDS/scratch;
- existing gfx1151 shape policies and strict primitive-chain fallback;
- the prior gfx1151 row8 two-wave retention and native Q8_1x2 split-weight
  rejection, so this experiment does not repeat either mechanism blindly.

### 7.2 Candidate ladder

Implement one rung at a time:

1. **T0 exact pair output subdivision:** assign disjoint column groups to
   independent gate/up wave pairs while preserving every historical K subset,
   per-thread FMA stream, wave reduction, ordered merge, BF16 gate/up round
   trips, and SiLU/store boundary. Screen the real pair before routing.
2. **T0 exact single two-wave output tile:** adapt the retained physical-row8
   idea only if rows2/3/4 preserve their existing one-wave arithmetic and the
   actual live shapes beat the current WG32 owner. Do not infer transfer from
   row8.
3. **T2 reassociated candidate:** a wave may own full K only under the complete
   production-profile gate. A naive full-K wave does not preserve the current
   four-subset ordered merge and must not be labeled strict.
4. **WMMA candidate only when justified:** GGUF dequant/layout and M<=8 may make
   scalar/dot4 rowtiles better. WMMA is not implied by the donor source, and
   prior gfx1151 WMMA geometry failures remain binding evidence.
5. **LM-head rung only after Q4:** revisit output partition/top-1 epilogues only
   if the Q4 ladder is exhausted and a fresh profile still gives the head
   enough complete-wall leverage.

Do not combine LM-head fusion, acceptance logic, and a new projection schedule
in the first RED/GREEN unit. First prove the Q4 pair/single bytes and actual
weight leaves, then route only an exact gfx1151 shape winner.

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

### 7.5 gfx1151 T0 result — 2026-08-25

The first exact pair-output subdivision (four waves owning gate/up four-column
halves) is rejected. It is BF16-bit exact over layers 0/8/63 and rows 2/3/4,
but loses all nine actual cases; its weighted pair wall is **1.02655x** the
parent.

The existing exact two-wave/16-column single-projection body transfers only in
narrow scopes:

- full-attention Q `K=5120, N=12288`: rows 2/3/4;
- recurrent QKV `K=5120, N=10240`: rows 3/4.

Every other measured role/row remains on the WG32/eight-column parent. The
selected key is `(hip_gfx1151, linear, gguf_q4_k_t16_v1,
dense_rowtile16_w2_bf16_bf16_out)`; the parent `dense_rowtile_bf16_bf16_out`
is its strict fallback. The selected/fallback manifest SHA-256 is
`90f0a8585617d3c4e4fa2ccd17ab9b53f0aba4ebc5fabdb282505e8850515fbf`.

The cached B3 trace names 256 selected calls at WG64, VGPR 120/144, zero LDS,
and zero scratch. The physical standard-Q4 single family improves
**150.960→148.358 ms (-1.724%)** and total target kernels improve
**690.081→686.661 ms (-0.496%)**. Target-marker wall is flat within changed
queue gaps. Against the immediate same-host hot control, complete B1/B2/B3 are
**17.092→17.115 (+0.130%)**, **19.960→20.008 (+0.241%)**, and
**20.947→21.040 tok/s (+0.440%)**. Corresponding MTP/own-AR ratios improve
**+0.224%/+0.336%/+0.535%**, with every train/heldout/category ratio positive;
all IDs and GPU/CPU acceptance decisions remain exact.

Decision: retain this gfx1151-only T0 scope. Do not route the rejected pair
candidate or any losing single shape. Evidence:
[`2026-08-25-gfx1151-qwen38-omlx-oi1-q4-two-wave-retained.json`](../benchmarks/results/2026-08-25-gfx1151-qwen38-omlx-oi1-q4-two-wave-retained.json).

### 7.6 Stop conditions

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

### 8.4 gfx1151 result — 2026-08-25

The request-owned variable-budget transition contract passes on the exact
Qwen3.8-27B `Q4_K_M` lane. The deterministic schedule
`1,1,2,1,3,2,2,3,3,1` exercises all nine directed budget edges on the complete
natural25 suite. Real cycles cover B1 reject/full, B2 reject/every partial/full,
and B3 reject/every partial/full. All generated IDs, GPU/CPU acceptance,
transaction stage reconciliation, and teardown remain exact. The provider owns
three independent `(slot,budget)` proposal graphs; target scheduler rows own
three distinct shape buckets. Cancellation before a cycle chooses no budget or
launches no proposal/target mutation.

The content-agnostic EMA controller is rejected. It explores B1/B2/B3, estimates
per-depth conditional acceptance and per-budget complete cycle wall, and scores
expected visible tokens/wall with 2% hysteresis. It receives no prompt text,
token IDs, categories, or heldout outcomes. Nevertheless:

- primary adaptive **21.089 tok/s** versus fixed B3 **21.211** (**-0.577%**);
- train **-1.047%**, code **-3.916%**, general English **-7.769%** versus each
  scope's best fixed budget;
- independent matched repeat adaptive **20.832** versus B3 **21.197 tok/s
  (-1.724%)**;
- budget choices, accepted counts, and generated IDs repeat exactly on all ten
  prompts, so this is an economics rejection rather than nondeterminism.

Retain the explicit sequence/policy hook only as default-off transition and
negative-policy infrastructure. Do not wire adaptive depth into public
generation. Evidence:
[`2026-08-25-gfx1151-qwen38-omlx-oi2-adaptive-rejected.json`](../benchmarks/results/2026-08-25-gfx1151-qwen38-omlx-oi2-adaptive-rejected.json).

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

Measure 512/4K/16K prompts on the same gfx1151 host/model:

- target prefill tok/s;
- total MTP prefill/activation wall and TTFT;
- peak tracked/HIP/whole-device memory;
- decode and acceptance unchanged after activation.

Retain if exact and non-regressive, even if the primary win is bounded memory or
startup wall rather than steady decode.

### 9.4 gfx1151 result — retained

The target prefill plan now exposes a request-owned hidden-chunk sink. The sink
carries one BF16 target row across chunks, appends the exact shifted token/hidden
pairs through the existing NextN block on the target stream, and skips the
prompt predictions' discarded output norm/LM-head scoring. This is target-hidden
streaming, not output streaming or a second compute stream. The old full-slab
capture remains available for diagnostics but is no longer used by MTP prompt
admission.

Same-host Qwen3.8-27B `Q4_K_M` results (cached/prewarmed kernels, BF16 KV):

| Prompt | Pure target prefill | Full-slab MTP TTFT | Streaming MTP TTFT | Wall delta | Tracked transient | HIP/whole-device transient |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 54.141 tok/s | 13.079 s | **10.356 s** | **-20.82%** | 5,253,120 -> **10,240 B** | 6 MiB -> **0** |
| 4K | 53.510 tok/s | 105.741 s | **85.574 s** | **-19.07%** | 41,953,280 -> **10,240 B** | 40 MiB -> **0** |
| 16K | 40.393 tok/s | 535.653 s | **467.949 s** | **-12.64%** | 167,782,400 -> **10,240 B** | 160 MiB -> **0** |

All shape rows preserve the first token and exact target/draft cursors. The
complete ten-prompt category/heldout B1/B2/B3 gate preserves greedy IDs,
acceptance, and GPU/CPU acceptance; teardown returns tracked memory to zero.
Retain the streaming sink as the sole MTP admission path. Evidence:
[`2026-08-25-gfx1151-qwen38-omlx-oi3-streaming-prompt-priming.json`](../benchmarks/results/2026-08-25-gfx1151-qwen38-omlx-oi3-streaming-prompt-priming.json).

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

### 10.1 gfx1151 result — post-norm rejected

An immutable `pre_output_norm|post_output_norm` target-to-draft policy now covers
both prompt chunks and every steady proposal. Target commit/rollback stays
pre-output-norm, while deeper draft chaining continues from the NextN head's own
post-norm output; direct fixtures prove one target norm rather than an accidental
double application. The strict/default policy remains `pre_output_norm`.

The complete category/heldout run is exact, but post-norm is not a universal
win:

| Budget | Pre-norm | Post-norm | Speed delta | Acceptance change | Decision |
| ---: | ---: | ---: | ---: | --- | --- |
| B1 | 17.143 | 17.328 tok/s | +1.08% | 113/131 -> 113/129 | reject: train -0.53%, Japanese -3.68% |
| B2 | 20.020 | 20.733 tok/s | +3.56% | 144/200 -> 145/195 | scoped repeat rejects: heldout -0.71%, Japanese -4.50% |
| B3 | 21.052 | 20.710 tok/s | **-1.62%** | 159/252 -> 158/255 | reject |

The dedicated B2 repeat confirms its aggregate signal (+2.63%) and improved
proposal efficiency, but it still fails the no-regressed-category rule. Do not
select hidden convention by prompt or budget. Retain the explicit policy only as
default-off diagnostic infrastructure for future model artifacts; preserve the
pre-norm fallback/default. Evidence:
[`2026-08-25-gfx1151-qwen38-omlx-oi4-postnorm-rejected.json`](../benchmarks/results/2026-08-25-gfx1151-qwen38-omlx-oi4-postnorm-rejected.json).

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
the active Linux/ROCm gfx1151 host.

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
| `OI-0` | complete | gfx1151 Qwen3.8-27B `Q4_K_M` | exact 10-prompt category/heldout gate | AR 11.712; B3 21.062; 100% target-timeline reconciliation | [`artifact`](../benchmarks/results/2026-08-25-gfx1151-qwen38-omlx-oi0-baseline.json) |
| `OI-1` | T0 retained; further rungs optional | gfx1151 Q4 attn-Q rows2-4 + recurrent-QKV rows3-4 | strict parent-bit exact | B1/B2/B3 +0.130%/+0.241%/+0.440%; Q4 family -1.724% | [`artifact`](../benchmarks/results/2026-08-25-gfx1151-qwen38-omlx-oi1-q4-two-wave-retained.json) |
| `OI-2` | transition retained; controller rejected | gfx1151 dense B1/B2/B3 | all 9 edges + every per-budget outcome exact | adaptive 21.089 vs fixed B3 21.211 (-0.577%); repeat -1.724% | [`artifact`](../benchmarks/results/2026-08-25-gfx1151-qwen38-omlx-oi2-adaptive-rejected.json) |
| `OI-3` | retained exact | gfx1151 Qwen3.8 dense 512/4K/16K | shifted cache/cursors + full category/heldout IDs/acceptance exact | TTFT -20.82%/-19.07%/-12.64%; prompt slab -> one 10,240-B row | [`artifact`](../benchmarks/results/2026-08-25-gfx1151-qwen38-omlx-oi3-streaming-prompt-priming.json) |
| `OI-4` | policy retained; post-norm rejected | gfx1151 Qwen3.8 dense B1/B2/B3 | full category/heldout IDs and target acceptance exact | B3 -1.62%; B2 aggregate +2.63% but heldout/Japanese regress | [`artifact`](../benchmarks/results/2026-08-25-gfx1151-qwen38-omlx-oi4-postnorm-rejected.json) |
| `OI-5` | not triggered | gfx1151 GDN 2.27-3.10% of target | profile trigger failed | no implementation | [`OI-0`](../benchmarks/results/2026-08-25-gfx1151-qwen38-omlx-oi0-baseline.json) |
| `OI-6` | not triggered | gfx1151 attention 0.52-0.55% of target | profile trigger failed | no implementation | [`OI-0`](../benchmarks/results/2026-08-25-gfx1151-qwen38-omlx-oi0-baseline.json) |
| `OI-7` | separate campaign | model artifact TBD | BF16-relative quant/task/MTP gate | quality/size/speed | — |

## 15. Final review conclusions

- The prior review's highest-value intuition—small-M verifier specialization—is
  sound, but hipEngine has already implemented the basic rowtile idea. Fresh
  gfx1151 evidence selects exact output/wave morphology on the standard-Q4
  pair/single rows 2/3/4; the LM head is a second rung.
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
