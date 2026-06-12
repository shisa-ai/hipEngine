# hipEngine MTP Native Implementation Plan

> Status (2026-06-12): shared ABI, local PARO+MTP-BF16 weights, persistent
> native proposal, exact B=3 chain verification, verify graph replay, draft vocab
> cap, and device expert dispatch are landed. The current W7900/gfx1100 35B-A3B
> MTP sprint baseline remains **0.758x AR** at **27.8 ms/cycle**. The current
> exact 9-prompt D32 best is now **0.927x AR** with `chain_attn_mode=decode_batched`,
> `graph_mode=off`, MTP verify canonicalize skip default-on, and fused
> 256-expert/top-8 proposer router top-k+softmax plus linear/full-attn
> shared-down combine, route-batched proposer expert, and linear shared
> SiLU+down-rotate plus linear A/B separate-output dual dense and one-split
> direct-gate full-attention decode default-on:
> **19.33 ms/cycle** wall, **16.05 ms/cycle** verifier time, and
> **1.25 ms/cycle** proposal/update time.
> This is the sister document to [`DFLASH.md`](DFLASH.md). MTP must reuse the
> shared native verifier/commit infrastructure from DFlash, not fork a separate
> c=1 native-loop tuning lane.

> **Top priority for the next push:** MTP break-even sprint. Hold every exact
> same-suite improvement, use `0.758x / 27.8 ms` as the locked sprint baseline,
> and push the active `0.927x / 19.33 ms` row below true break-even and ultimately
> above `1.0x`. See ["Next Push: 35B MTP Break-Even Sprint"](#next-push-35b-mtp-break-even-sprint-2026-06-11).

## Next Push: 35B MTP Break-Even Sprint (2026-06-11)

Locked baseline:

- Model/workload: Qwen3.6-35B-A3B-PARO packed trunk + MTP-BF16 sidecar,
  W7900/gfx1100, stable quicksort B=3 chain path.
- Runtime config: `proposal_impl=persistent_device`,
  `chain_attn_mode=batched`, verifier graph `auto`, draft vocab cap `32768`,
  device expert dispatch, exact chain verifier; branching tree default-off.
- Current speed: **`83.4 tok/s` MTP vs `~110 tok/s` AR = `0.758x`**.
- Current wall: **`27.8 ms/cycle = 22.0 ms verify + 5.8 ms proposer/draft`**.
- Wall milestone from the original sprint: **`<21.5 ms/cycle`**, requiring
  about **6.3 ms/cycle** off the locked row.

Current best retained stack after the 2026-06-12 host/cache/proposer-router/reduced-DAG cleanup:

- Runtime config: `proposal_impl=persistent_device`,
  `chain_attn_mode=decode_batched`, verifier graph `off`, MTP
  canonicalize-after-verify skip default-on, fused 256-expert/top-8 proposer
  router top-k+softmax default-on, route-batched proposer expert loop
  default-on, linear shared SiLU+down-rotate fused default-on, linear-attn and
  full-attn shared-down+combine fused default-on, linear A/B separate-output
  dual dense GEMV default-on for small-batch rows, one-split direct-gate
  full-attention decode default-on, draft vocab cap `32768`, device expert
  dispatch, exact chain verifier; branching tree default-off.
- Current speed: **`0.927x` AR** on the 9-prompt D32 suite, exact `9/9`, with
  identical accepted lengths and active budgets relative to the prior
  linear A/B dual-separate current-best row.
- Current wall: **`19.334 ms/cycle = 16.053 ms verify + 1.248 ms proposal/update`**.
- Wall milestone status: crossed (`19.334 ms < 21.5 ms`). True `>1.0x`
  break-even at the current visible-token density is stricter: with
  `2.012` visible tokens/cycle and `9.030 ms/token` AR, wall would need to be
  about **`18.2 ms/cycle`**, or visible density must rise to at least
  **`2.14` visible tokens/cycle** at the current wall. The next work must keep
  grinding wall while saving acceptance-density policy work for the endgame.

Tree status is explicit: the B=3 gated tree path is exact and graph-replayed, but
negative (`0.61x` vs chain `0.76x`) because it spends budget on a depth-1 sibling
and caps depth where the chain often accepts 3. Keep tree off until the verifier
wall is lower or acceptance depth changes.

Planning estimates below are not performance claims until artifacted with exact
command, hardware, workload shape, and correctness gate.

Current priority order after folding in the external review:

| Rank | Work item | Expected wall effect | Risk / readiness | Live-plan correction |
| --- | --- | ---: | --- | --- |
| Done | Specialized proposer router top-k / fused router-topk-softmax | -1.31 ms/cycle wall retained | Exact D32 suite positive | `HIPENGINE_MTP_PROPOSER_ROUTER_TOPK_FUSED=1` fuses the proposer router's 256-expert/top-8 `topk_rows_i32` + softmax path into one exact kernel with the same descending-value/lower-index tie order. Exact `9/9`, identical accepted lengths/active budgets, ratio `0.8244x -> 0.8806x`, wall `21.686 -> 20.379 ms/cycle`, proposal/update `1.974 -> 1.460 ms/cycle`. Proposer marker profile moved router family `1.714 -> 0.373 ms/cycle`, total proposer kernel `4.676 -> 3.379 ms/cycle`, host window `5.578 -> 4.248 ms/cycle`, and calls `181.5 -> 178.4/cycle`. |
| Done | Linear-attn shared-down+combine parallel epilogue | -0.11 ms/cycle wall retained | Exact D32 suite positive | `HIPENGINE_LINEAR_SHARED_DOWN_COMBINE_FUSED=1` fuses the linear-attn shared-down output-tiled W4 GEMV with selected/shared gate residual combine while preserving the old FP16 rounding points. Exact `9/9`, identical accepted lengths/active budgets, ratio `0.8843x -> 0.8859x`, wall `20.315 -> 20.204 ms/cycle`, verify `16.523 -> 16.402 ms/cycle`. Quicksort verify profile removed 30 combine launches/pass (`942 -> 912`), moved kernel `12.871 -> 12.781 ms/pass`, host `16.753 -> 16.565 ms/pass`, and `moe_combine` `0.124 -> 0.032 ms/pass`. |
| Done | Full-attn shared-down+combine parallel epilogue | -0.05 ms/cycle wall retained | Exact D32 suite positive | `HIPENGINE_FULL_SHARED_DOWN_COMBINE_FUSED=1` reuses the exact shared-down output-tiled W4 + selected/shared gate residual combine epilogue in the full-attention C-dispatch path. Exact `9/9`, identical accepted lengths/active budgets, ratio `0.8894x -> 0.8910x`, wall `20.158 -> 20.110 ms/cycle`, verify `16.366 -> 16.311 ms/cycle`. Quicksort verify profile removed the remaining 10 combine launches/pass (`912 -> 902`), moved kernel `12.789 -> 12.714 ms/pass`, host `16.596 -> 16.482 ms/pass`, and `moe_combine` `10 -> 0 calls/pass`. |
| Done | Linear shared SiLU+down-rotate fusion | -0.05 ms/cycle wall retained | Exact D32 suite positive | `HIPENGINE_LINEAR_SHARED_SILU_ROTATE_FUSED=1` routes the linear-attn C dispatcher through the existing exact `silu_mul_pair_rotate_out_fp16` kernel for shared expert down input, replacing `silu_mul_separate_out_fp16 + paro_rotate1_fp16` while preserving the FP16 activation rounding point. Exact D32 `9/9`, identical accepted lengths/active budgets, ratio `0.9173x -> 0.9194x`, wall `19.547 -> 19.496 ms/cycle`, verify `16.278 -> 16.217 ms/cycle`; quicksort verify profile removed 30 launches/pass (`902 -> 872`), moved kernel `12.714 -> 12.700 ms/pass`, and host `16.482 -> 16.359 ms/pass`. |
| Done | Linear A/B separate-output dual dense GEMV | -0.04 ms/cycle wall retained | Exact D32 suite positive | `HIPENGINE_LINEAR_AB_DUAL_SEPARATE=1` collapses the verifier linear-attn A/B FP16 dense projections from two `dense_gemv_out_fp16` launches into one separate-output dual GEMV while preserving the old per-output FP16 accumulation/store layout. Exact D32 `9/9`, identical accepted lengths/active budgets, ratio `0.9201x -> 0.9240x`, wall `19.480 -> 19.440 ms/cycle`, verify `16.197 -> 16.155 ms/cycle`; quicksort profile removed 30 launches/pass (`872 -> 842`), moved dense GEMV `0.209 -> 0.125 ms/pass`, total kernel `12.680 -> 12.636 ms/pass`, and host `16.311 -> 16.220 ms/pass`. Default-on for `1 < tokens <= HIPENGINE_SMALL_BATCH_DECODE_THRESHOLD`; opt out with `HIPENGINE_LINEAR_AB_DUAL_SEPARATE=0`. |
| Done | Full-attn decode_batched one-split direct gate | -0.106 ms/cycle wall retained | Exact D32 suite positive | `HIPENGINE_QWEN35_DECODE_BATCHED_DIRECT_GATE=1` detects the current `decode_batched` `num_splits=1` full-attention rows and replaces `split_k_ctx_tensor_gqa_batch + reduce_gate_batch` with one direct gated GQA decode kernel. Focused GPU test is bit-identical to the retained split+reduce path. Exact D32 `9/9`, identical accepted lengths/active budgets, ratio `0.9240x -> 0.9273x`, wall `19.440 -> 19.334 ms/cycle`, verify `16.155 -> 16.053 ms/cycle`; quicksort profile removed 10 reduce launches/pass (`842 -> 832`), moved decode-attention `0.429 -> 0.416 ms/pass`, total kernel `12.636 -> 12.609 ms/pass`, and host `16.220 -> 16.085 ms/pass`. Default-on for `num_splits == 1`; opt out with `HIPENGINE_QWEN35_DECODE_BATCHED_DIRECT_GATE=0`. Artifact: `benchmarks/results/2026-06-12-hipengine-mtp-direct-gate-retained.json`. |
| 1 | Full-layer reduced-DAG batching for the non-MoE layer surround | -1.5 to -3.0 ms if real DAG nodes disappear | Medium; M13.C/M14 patterns prove the dispatch mechanics | This is the useful version of "extend the C dispatcher." A C-only loop around the same launches is already measured parity; the next unit must remove launches, fills, copies, Python/ctypes round trips, or per-pass object/pointer rebuilds outright. The shared-down+combine epilogues are retained reduced-DAG micro-slices, but they are not enough; continue with broader layer-surround batching/composites rather than stopping at pairwise fusions. |
| 2 | M12.7 proposer graph subgraph design | 0 ms retained from whole-body capture; subgraph TBD | Medium/high; whole-body shape no-held | Post-route-batching proposer marker profile shows `proposer_all` is `~3.54 ms/cycle` host for `~3.02 ms/cycle` kernel over `92` launches/cycle. Capture can still recover some launch/ctypes overhead, but it cannot remove the expert dense, LM-head, attention, or dense BF16 GPU work. **2026-06-12 audit:** naive direct capture is not a speed row: `NativeMtpChainProposer.advance()` bakes `key_cache_dst`, `value_cache_dst`, and attention `context_len` from absolute `cache_len`, while the harness alternates result-producing advances with LM-head/readbacks and state-only repair advances. The current HIP wrapper only exposes capture/instantiate/launch, not graph-node parameter updates, so a useful M12.7 must use fixed base pointers plus device-read live metadata. First prerequisite: add graph-safe proposer KV writes, either by teaching the QKV/rotary producers to write `key_cache_base + device_cache_slot * kv_features` and `value_cache_base + device_cache_slot * kv_features` directly or by adding one indexed K/V copy kernel that consumes a device slot scalar; pair that with bucketed attention's device live-context scalar. Otherwise exact-cache-length graphs will mostly miss and may freeze stale cache slots. A focused live-context probe using the existing DFlash bucketed attention kernel stayed exact on quicksort but no-held as a standalone slice (`proposal/update +0.011 ms/cycle`) because it adds a live-context H2D and still leaves dynamic KV write destinations unsolved; see `benchmarks/results/2026-06-12-hipengine-mtp-proposer-bucketed-attention-nohold.json`. The fixed-base indexed K/V producer slice now exists behind `HIPENGINE_MTP_PROPOSER_INDEXED_KV_WRITE=1` and is exact, but it is not a standalone speed row: exact D32 off/on moved ratio `0.9237x -> 0.9215x`, wall `19.376 -> 19.412 ms/cycle`, verify `16.092 -> 16.125 ms/cycle`, and visible density stayed `2.012/cycle`. A direct current-stack bucketed-attention smoke also matched the locked accepted trace, but private-stream HIP graph capture of the same body changed proposer accepted lengths even when recapturing every advance (`[3,3,2,0,2,0,0,1,3,0,2,0,2] -> [1,0,0,0,0,0,1,0,...]`), and default-stream capture is rejected by HIP. Experiment code was removed; do not retry M12.7 as a whole-body HIP graph until the proposer body is split into a capture-safe stream-honoring subgraph or we add graph node parameter updates. Artifacts: `benchmarks/results/2026-06-12-hipengine-mtp-proposer-indexed-kv-write-nohold.json`, `benchmarks/results/2026-06-12-hipengine-mtp-proposer-graph-capture-nohold.json`. A narrower route-batched selected-expert-only graph replay also stayed exact (`9/9`) but no-held: same-session graph-on/off kept identical density, wall moved `19.376 -> 19.346 ms/cycle` from verifier noise, while the directly relevant proposal/update metric regressed `1.2492 -> 1.2550 ms/cycle`; prototype code removed. Artifact: `benchmarks/results/2026-06-12-hipengine-mtp-proposer-route-expert-graph-nohold.json`. |
| Done | Route-batched proposer expert loop | -0.44 ms/cycle wall retained | Exact D32 suite positive | `HIPENGINE_MTP_PROPOSER_ROUTE_BATCHED_EXPERT=1` batches all top-8 routed expert gate/up GEMVs, route-major SiLU, down GEMVs, and ordered route accumulation while preserving scalar route accumulation order. Exact D32 `9/9`, identical accepted lengths/active budgets, ratio `0.8939x -> 0.9135x`, wall `20.045 -> 19.604 ms/cycle`, proposal/update `1.455 -> 1.244 ms/cycle`; proposer profile calls `178.4 -> 92.0/cycle`, kernel `3.379 -> 3.018 ms/cycle`, host `4.248 -> 3.543 ms/cycle`. |
| 4 | Acceptance-density diagnostics and policy | Ratio lever, not pure wall | Low/medium; save policy changes until fast path stabilizes | Add per-position acceptance histograms, B=4/B=5 economics, vocab-cap rejection diagnostics, and adaptive B/AR fallback gates after the wall path stabilizes. At the current wall, break-even needs about `+0.17` visible tokens/cycle, so this is co-equal with the final wall cuts but should not distract from reduced-DAG/proposer work in the next implementation unit. First diagnostic should be B=4 with per-depth counters; promote higher B only if the added row improves suite ratio, not just accepted-token count. |
| 5 | Verifier/proposer inter-cycle overlap design | 0 ms retained for naive side-stream shape; still possible only with broader update redesign | Medium/high; dependencies are clean enough to test, but update-side sync cost dominates today | **No-hold 2026-06-12 for naive side stream.** `HIPENGINE_MTP_OVERLAP_VERIFY_COMMIT_PROPOSER=1` stayed exact `9/9` with identical acceptance and did hide part of verify/commit (`16.166 -> 16.028 ms/cycle`), but proposal/update grew more (`1.243 -> 1.438 ms/cycle`) and aggregate wall/ratio regressed (`19.441 -> 19.506 ms/cycle`, `0.9216x -> 0.9184x`). Keep default off; revisit only after proposer update removes its final sync/readback cost or after a broader graph/update design changes the stream balance. Artifact: `benchmarks/results/2026-06-12-hipengine-mtp-verify-commit-proposer-overlap-nohold.json`. |
| Closed | Per-layer memset/fill/copy enumeration | 0 ms retained on current stack | Low if a future profile reintroduces a live bucket | Current retained profiles show no verifier fill family and only `2` `runtime_copy` launches/pass at about `0.0067-0.0069 ms/pass`, so this is not an active speed lever. Reopen only from a fresh profile with a material live fill/copy bucket plus local write-before-read proof. |
| 7 | Multi-stream overlap after DAG reduction | -1.0 to -3.0 ms possible | High; direct branch-overlap attempt regressed all-cycle wall | Do not retry on the current launch-heavy graph. Revisit only after ranks 1-6 shrink synchronization and graph-capture overhead. |
| Hold | New selected-GEMV/shared rotate design | Only if it avoids prior redundant rotate/barrier costs | Template exists, but the measured M13.B family is no-hold | The colleague ranking is directionally right about launch-count compression, but M13.B.1/B.2/B.3 already closed the current rotate-fusion designs as exact-but-negative. A 2026-06-12 full-attn C-dispatch shared rotate-once/no-reset diagnostic also no-holded (row below), so reopen only for a materially different selected/parallel design that avoids barrier spin and redundant per-output work. |
| Done | Scratch-cache generation stamp | Retained default-on | Exact D32 same-suite positive | `HIPENGINE_VERIFY_SCRATCH_GENERATION_STAMP=1` stores verifier scratch cache entries with a generation bumped by `_clear_verify_scratch_caches()` and uses the generation match to skip workspace-pointer revalidation on hits. Exact `9/9`, identical accepted lengths/active budgets, wall `25.7085 -> 25.5955 ms/cycle`, verify `20.5460 -> 20.4342 ms/cycle`, ratio `0.7145x -> 0.7252x`; graph-auto profile kept `932` calls/pass and moved host `18.322 -> 18.298 ms/pass`; graph-off host `32.659 -> 31.971 ms/pass`. |
| Done | MTP graph-off canonicalize-after-verify skip | Retained default-on for the MTP harness | Exact D32 same-suite positive | `HIPENGINE_MTP_SKIP_CANONICALIZE_AFTER_VERIFY=1` keeps verifier-shaped scratch live between MTP graph-off verify cycles. Exact `9/9`, identical accepted lengths/active budgets, graph-off batched wall `37.207 -> 24.076 ms/cycle`, verify `32.069 -> 18.933 ms/cycle`, ratio `0.4969x -> 0.7730x`; rocprof calls and kernel stayed flat (`932/pass`, ~`14.33 ms/pass`) while host moved `32.505 -> 18.272 ms/pass`. |
| Done | Current-stack `decode_batched` with graph-off skip | Current best retained MTP verifier mode | Exact D32 same-suite positive | After canonicalize skip made graph-off competitive, `chain_attn_mode=decode_batched` is exact `9/9` and improves over graph-off batched skip with identical accepted lengths/active budgets: ratio `0.7730x -> 0.8252x`, wall `24.076 -> 21.661 ms/cycle`, verify `18.933 -> 16.511 ms/cycle`; profile calls `932 -> 942`, kernel `14.330 -> 12.922 ms/pass`, host `18.272 -> 16.849 ms/pass`. |
| No-hold | `decode_batched` verifier HIP graph for `num_splits=1` | 0 ms retained | Exact but profile-negative | Opt-in prototype allowed graph capture for the current one-split `decode_batched` verifier bucket and keyed the graph by split count. Quicksort `graph_mode=validate` and `graph_mode=auto` both stayed exact with the locked accepted trace, and auto replayed after the first capture miss. Focused verifier profile no-held it: calls/pass stayed `832`, while kernel regressed `12.606 -> 12.674 ms/pass` and host marker `16.153 -> 16.182 ms/pass`. Prototype code removed; keep `decode_batched` graph-off. Artifact: `benchmarks/results/2026-06-12-hipengine-mtp-decode-batched-verify-graph-nohold.json`. |
| No-hold | `decode_batched + HIPENGINE_SELECTED_MOE_DOWN_STAGED=1` compound | 0 ms retained | Exact but aggregate-negative | Retested after graph-off became current best. Exact `9/9` with unchanged accepted lengths/active budgets, but slower than `decode_batched + graph_off + skip`: ratio `0.8252x -> 0.8204x`, wall `21.661 -> 21.763 ms/cycle`, verify `16.511 -> 16.628 ms/cycle`, cycle cost `2.411 -> 2.425`. Keep selected-down staged opt-in only. |
| No-hold | Thread-0 shared-down + shared-gate/residual combine epilogue | 0 ms retained | Exact but kernel-negative | Prototype removed 30 linear-attn combine launches/pass and kept quicksort exact with identical accepted lengths, but the fused output-tiled shared-down kernel grew from `0.348 -> 0.814 ms/pass`; net profile regressed calls `942 -> 912/pass`, kernel `12.860 -> 13.255 ms/pass`, host `16.832 -> 17.076 ms/pass`. Superseded by the retained parallel-epilogue implementation above; do not reintroduce the serial shape. |
| No-hold | Final RMSNorm FP16-to-BF16 fused cast | 0 ms retained | Exact but suite wall/verify-negative | Replacing verifier final `paro_rmsnorm_out_fp16 + fp16_to_bf16` with one exact BF16-writing kernel removed one launch in the quicksort profile (`902 -> 901 calls/pass`) and nudged profile kernel/host (`12.7144 -> 12.7105 ms/pass`, `16.4816 -> 16.4393 ms/pass`), but the exact 9-prompt D32 suite regressed wall and verify with identical acceptance (`20.087 -> 20.119 ms/cycle`, verify `16.298 -> 16.329 ms/cycle`). Experiment code removed; do not promote without a new same-suite positive design. |
| No-hold | LM-head thread-count sweep (`HIPENGINE_QWEN35_LM_HEAD_THREADS`) | 0 ms retained | Exact quicksort but profile-negative | Current-stack quicksort D32 stayed exact at `256` and `512` threads with the locked accepted trace, but both worsened the retained verifier profile before a suite run was justified. Baseline `128` from the retained `0.924x` profile: `842` calls/pass, `12.636 ms/pass` kernel, `16.220 ms/pass` host marker, W8A16 LM-head `1.450 ms/pass`. `256` threads: `13.427 ms/pass` kernel, `16.973 ms/pass` host marker, W8A16 `2.251 ms/pass`; `512` threads: `15.737 ms/pass` kernel, `19.267 ms/pass` host marker, W8A16 `4.572 ms/pass`. Keep `128`. Artifact: `benchmarks/results/2026-06-12-hipengine-mtp-lmhead-thread-sweep-nohold.json`. |
| No-hold | FP16 dense A/B WMMA verifier projection | 0 ms retained | Exact but suite wall/verify-negative | Before the dual-separate path was promoted, `HIPENGINE_VERIFY_DENSE_GEMV_WMMA=1` routed the linear-attn A/B FP16 dense verifier projections through the existing WMMA dense GEMV path. Quicksort and same-session 9-prompt D32 both stayed exact with unchanged visible/accepted density, but the suite regressed on every prompt: ratio `0.9236x -> 0.8807x`, wall `19.460 -> 20.374 ms/cycle`, verify `16.176 -> 17.086 ms/cycle`, and cycle cost `2.153 -> 2.256` AR tokens. Keep default off; on the current stack this diagnostic also requires `HIPENGINE_LINEAR_AB_DUAL_SEPARATE=0` to bypass the retained path. Artifact: `benchmarks/results/2026-06-12-hipengine-mtp-dense-gemv-wmma-nohold.json`. |
| No-hold | Current-stack W4 all-sites mask recheck | 0 ms retained | Exact suite but no live profile-path win | `HIPENGINE_W4_MULTI_ROW_PACK8_SITES=all` now passes quicksort and exact D32 `9/9`, including the formerly fragile `translation` prompt, with unchanged accepted/visible density. The 9-prompt sample moved ratio `0.9251x -> 0.9289x`, wall `19.345 -> 19.285 ms/cycle`, and verify `16.063 -> 16.004 ms/cycle`, but this is not retained because the focused quicksort verifier profile shows no launch/path movement: `832.0` calls/pass both ways, `w4_single_gemv` unchanged (`1.337118 -> 1.337121 ms/pass`), and total kernel/host slightly worse (`12.6277 -> 12.6357 ms/pass`; host window `0.177350 -> 0.177554 s` over 11 passes). Keep the default safe mask; treat `all` as a diagnostic only. Artifact: `benchmarks/results/2026-06-12-hipengine-mtp-w4-all-sites-current-nohold.json`. |
| No-hold | Full-attn Q/K/V mixed triple projection | 0 ms retained | Exact but spills/regresses profile | A diagnostic `HIPENGINE_FULL_QKV_TRIPLE_PROJECT=1` prototype collapsed the current full-attn Q/K decode-dual projection plus V single projection into one mixed split W4 launch. Focused GPU tests were bit-exact against the current chain and quicksort D32 kept the locked accepted trace, but the clean sequential profile no-held it: calls/pass dropped `832 -> 822`, yet kernel/host regressed `12.623 -> 14.533 ms/pass` / `16.137 -> 18.053 ms/pass`. The monolithic row-loop kernel raised the W4 single bucket `1.328 -> 3.423 ms/pass` with `vgpr 256` and `scratch 336 B`. Prototype code removed; do not retry this exact three-output shape unless the implementation avoids the register/scratch blow-up. Artifact: `benchmarks/results/2026-06-12-hipengine-mtp-full-qkv-triple-project-nohold.json`. |
| No-hold | Full-attn shared rotate staged C-dispatch | 0 ms retained | Exact quicksort but profile-negative | Prototype wired the keyed HBM-staged shared gate/up rotate into the full-attn C dispatcher only, preserving the retained linear-attn C path and testing the rotate-once/no-barrier-reset shape. Quicksort stayed exact with identical accepted lengths, and the profile removed the intended 10 full-attn shared rotate launches/pass (`842 -> 832` calls/pass, `moe_paro_rotate_in` `160 -> 150`), but staged dual GEMV cost outweighed the launch saving: verifier kernel `12.636 -> 12.697 ms/pass`, host marker `16.220 -> 16.236 ms/pass`, and `w4_dual_gemv` `1.834 -> 1.950 ms/pass`. Prototype code removed. Artifact: `benchmarks/results/2026-06-12-hipengine-mtp-full-shared-rotate-staged-nohold.json`. |
| No-hold | Full-attn shared gate/up split output-tiled C-dispatch | 0 ms retained | Non-exact | Prototype routed the full-attn C dispatcher through split output-tiled dual W4 plus separate-output `silu_mul_pair_rotate_out_fp16`, mirroring the retained linear-attn split-output path. The integration is not numerically equivalent to the packed full-attn `gemv_awq_dual_pack8_transposed_fp16 + silu_mul_dual_rotate_out_fp16` chain: quicksort D32 failed exact AR at output index 9 (`156973 -> 149315`) and accepted lengths were all zero. Experiment code removed. Artifact: `benchmarks/results/2026-06-12-hipengine-mtp-full-shared-gate-up-split-output-tiled-nohold.json`. |
| No-hold | Linear-attn out-proj f32 rotate + W4 GEMV staged keyed | 0 ms retained | Real MTP smoke hung | Prototype combined `paro_rotate1_f32_to_fp16` staging with the following transposed pack8 W4 GEMV for `linear_attn.out_proj`, preserving the FP16 rotated staging point. Focused GPU comparison against the old two-kernel chain passed bitwise, but the real B=3 quicksort D32 verifier smoke hung with GPU busy, memory idle, and no JSON after about two minutes. Prototype code removed; do not retry this same-kernel producer/consumer spin-barrier topology. Artifact: `benchmarks/results/2026-06-12-hipengine-mtp-linear-out-rotate-gemv-staged-nohold.json`. |
| No-hold | C-dispatch keyed cooperative MoE router | 0 ms retained | Exact but profile-negative | Prototype merged verifier MoE router logits+select into one keyed cooperative kernel behind `HIPENGINE_MOE_C1_ROUTER_KEYED=1`. Unit comparison and quicksort exact smoke held identical accepted lengths, and rocprof removed 40 launches/pass (`902 -> 862`), but router kernel time grew `0.502 -> 0.877 ms/pass`, total verifier kernel grew `12.714 -> 13.080 ms/pass`, and host window grew `16.482 -> 16.717 ms/pass`. Experiment code removed; do not retry this single-kernel topology unless selection is parallelized without serializing all tokens in the final block. Artifact: `benchmarks/results/2026-06-12-hipengine-mtp-moe-router-keyed-nohold.json`. |
| No-hold | Force grouped compact MoE at B=3 verifier rows | 0 ms retained | Exact quicksort but profile-negative | `HIPENGINE_VERIFY_MOE_GROUPED_MIN_TOKENS=2` forces the current `rows=4` verifier MoE through grouped compact instead of the selected c1 C-dispatch path. Quicksort D32 stayed exact with identical accepted lengths, but profile regressed calls/pass `872 -> 1232`, kernel `12.680 -> 14.155 ms/pass`, and host marker `16.311 -> 19.210 ms/pass`; grouped metadata/combine overhead outweighed any selected-path benefit. Keep the B=3 selected c1 path and do not retest grouped at this shape without a new grouping kernel. Artifact: `benchmarks/results/2026-06-12-hipengine-mtp-grouped-min2-nohold.json`. |
| No-hold | Proposer shared gate/up dual dense | 0 ms retained; opt-in diagnostic only | Exact but repeated full-suite wall/ratio-negative | `HIPENGINE_MTP_PROPOSER_SHARED_GATE_UP_DUAL=1` replaces the proposer's two shared expert BF16 dense gate/up launches with one `dense_dual_gemv_out_bf16_wmma` launch. The direct proposer marker moved in the expected direction (`92.0 -> 88.7` launches/cycle, kernel `3.018 -> 2.913 ms/cycle`, host `3.543 -> 3.464 ms/cycle`) and repeated exact D32 A/B pairs kept identical acceptance while proposal/update improved by about `0.020 ms/cycle`, but total wall regressed in both pairs (`+0.043` and `+0.015 ms/cycle`) and ratio regressed (`-0.0006x`, `-0.0049x`). Keep default-off/lazy opt-in; do not count it as a retained speed row unless a broader proposer fusion removes the end-to-end regression. Artifact: `benchmarks/results/2026-06-12-hipengine-mtp-proposer-shared-gate-up-dual-nohold.json`. |
| No-hold | Proposer shared-down+accumulate WMMA epilogue | 0 ms retained | Non-exact full MTP smoke | Prototype fused the proposer's shared expert down WMMA dense store with `sigmoid(shared_gate) * shared_down` accumulation into the FP32 MoE accumulator, preserving the local BF16 rounding point. The isolated epilogue matched the old `dflash_dense_bf16_to_bf16_wmma + mtp_accumulate_sigmoid_gate_bf16_to_f32` chain bitwise at the real `rows=1,in=512,out=2048` shape, but the full B=3 D32 exact smoke failed at output index 9 (`156973 -> 149315`) and accepted zero draft tokens over all 31 cycles. Prototype code removed; do not retry as a direct epilogue swap without a full-proposer hidden/logit comparator. Artifact: `benchmarks/results/2026-06-12-hipengine-mtp-proposer-shared-down-accum-fused-nohold.json`. |
| No-hold | Proposer QKV projection + query/gate split | 0 ms retained | Exact but suite wall/ratio-negative | Prototype fused the proposer's `dflash_qkv_proj_bf16_mixed + mtp_split_q_gate_f32_bf16` sequence so the Q projection wrote query FP32 and gate BF16 directly. Focused HIP tests and a real-shape synthetic probe were bitwise-equal, and quicksort plus the 9-prompt D32 suite stayed exact with identical acceptance. The profiler removed the expected split launches (`91.82 -> 88.73` proposer calls/pass) and slightly reduced host marker (`3.537 -> 3.518 ms/pass`), but kernel time was flat/slightly worse (`2.977 -> 2.980 ms/pass`) and the full suite regressed: ratio `0.9264x -> 0.9217x`, wall `19.357 -> 19.399 ms/cycle`, proposal/update `1.247 -> 1.261 ms/cycle`. Prototype code removed; do not retry this local QKV+split shape without a broader proposer fusion that pays end-to-end. Artifact: `benchmarks/results/2026-06-12-hipengine-mtp-proposer-qkv-split-q-gate-nohold.json`. |
| Done | Verifier MLP scratch policy alignment | Retained default-on | Exact D32 same-suite positive | `HIPENGINE_VERIFY_MLP_SCRATCH_POLICY_ALIGNED=1` makes verifier MLP scratch reservation use the same c1/grouped threshold as the chain/tree t-loop MoE path and keys the cache by expected policy. Exact `9/9`, identical accepted lengths/active budgets, wall `26.309 -> 25.690 ms/cycle`, verify `21.176 -> 20.523 ms/cycle`, ratio `0.7003x -> 0.7172x`; graph-auto profile kept `932` calls/pass and moved host `18.314 -> 18.246 ms/pass`; graph-off host `32.445 -> 32.273 ms/pass`. |
| Done | Resident state/cache Tensor view caching | Retained default-on | Exact D32 same-suite positive | `HIPENGINE_RESIDENT_TENSOR_VIEW_CACHE=1` caches `_slot_linear_state`, `_slot_full_cache`, and `_full_cache_all_slots` non-owning Tensor views with explicit invalidation. Exact `9/9`, identical acceptance, wall `26.642 -> 26.426 ms/cycle`, verify `21.506 -> 21.278 ms/cycle`, ratio `0.6924x -> 0.6986x`; graph-off host control `32.52 -> 31.70 ms/pass`. |
| Done | `single_linear_out` / `single_full_v` exact multi-row routing | Already retained | Default-on after exact D32 gates | Not pending next work; keep the retained paths and remove their opt-outs after the follow-up defaults-only gates in `docs/REFACTOR.md`. |

Current profile triage after the `0.927x / 19.33 ms` row:

- The retained `decode_batched + graph_off` verifier profile refreshed on
  2026-06-12 after the direct-gate full-attention decode slice is down to
  `832` launches/pass, `12.609 ms` kernel/pass, and `16.085 ms` host marker/pass
  on the quicksort D32 B=3 shape. The top buckets are now real compute:
  selected gate/up W4 `1.900 ms/pass`, shared/dual W4 `1.836 ms/pass`, GDN
  `1.768 ms/pass`, LM-head W8A16 `1.445 ms/pass`, single W4
  `1.334 ms/pass`, and selected down W4 `1.204 ms/pass`. Full-attention
  decode is now a single direct-gate launch per full-attn layer at this
  one-split shape (`10` calls/pass, `0.416 ms/pass`).
- Do not spend the next implementation unit on fill/copy cleanup unless a fresh
  profile shows a live bucket again: current `runtime_copy` is only
  `2` launches/pass and `0.0068 ms/pass`, and `fillBufferAligned` is effectively
  absent from the verifier window.
- The reviewer-flagged linear-state commit item is already the retained chunked
  one-launch path. It is still visible (`~0.20 ms/pass`), but prior 32 KiB
  chunk tuning only saved about `2 us` in-kernel and worsened the total profile,
  so further commit work belongs with inter-cycle overlap or a broader commit
  design, not as a standalone hot item.
- The old full-attn KV/rotate flags
  (`HIPENGINE_PARO_FULL_ATTN_KV_PACK8_FUSED`,
  `HIPENGINE_PARO_ROTATE_DUAL_PACK8_FUSED`) have rejected artifacts and should
  not be re-run as generic "maybe" checks. Reopen only if the callsite and
  shape differ from those artifacts.
- Forcing grouped compact MoE at the B=3 verifier shape is also no-held:
  `HIPENGINE_VERIFY_MOE_GROUPED_MIN_TOKENS=2` kept quicksort exact but increased
  calls/pass `872 -> 1232`, kernel `12.680 -> 14.155 ms/pass`, and host marker
  `16.311 -> 19.210 ms/pass`.
- The remaining high-leverage wall path is still real reduced-DAG work:
  remove named projection/router/MoE/attention launches or fuse structural
  composites. A C wrapper around the same launches is not enough.
- Fresh reduced-DAG audit from the `842` calls/pass profile:
  - Do **not** treat the separate GDN finalize launch as an easy local epilogue
    fuse. The retained chain GDN kernel gets its speed from `VTILE=4`, where
    each block owns only four `dv` columns, while
    `qwen35_gdn_tree_rmsnorm_gate_finalize` needs the full `head_v_dim=128`
    row reduction for RMS/gate. A one-kernel version would need a different
    whole-row/inter-block design and must beat the current `VTILE=4 + finalize`
    profile, not just remove 30 launches/pass on paper.
  - Do **not** route the current B=3 linear-attn QKV/Z site through the legacy
    `HIPENGINE_PARO_ROTATE_DUAL_PACK8_FUSED` staged kernel. That kernel writes
    the old concatenated `[qkv,z]` row layout and is only wired for token-1
    deferred rotation; the exact rows>1 verifier path writes separate
    contiguous `qkv` and `z` buffers through the split-dual decode kernel. A
    real retry requires a new keyed-barrier, separate-output rotate-staged dual
    W4 ABI plus exact kernel tests before any MTP smoke/profile.
  - 2026-06-12 no-hold: that separate-output ABI was prototyped and removed.
    Wrapper dispatch, HIP build, and a small synthetic GPU comparison against
    `paro_rotate2_fp16 + gemv_awq_dual_pack8_multi_row_decode_split_transposed_fp16`
    passed, but the full B=3 QKV/Z smoke hung with GPU busy before writing JSON.
    The same-kernel spin barrier is unsafe at the real `1536` pack x `4` row
    grid because consumer GEMV blocks can occupy scheduler slots while producer
    rotate blocks still need to run. Reopen only with a scheduling-safe design,
    not another global in-kernel producer/consumer wait.
  - 2026-06-12 no-hold: the narrower linear-attn `out_proj`
    `f32 -> fp16 PARO rotate + transposed W4 GEMV` staged-keyed retry was also
    prototyped and removed. A focused GPU test matched the old two-kernel chain
    bitwise, but the real B=3 quicksort D32 smoke hung with GPU busy, memory
    idle, and no JSON after about two minutes. Do not spend more time on this
    same in-kernel producer/consumer wait topology; a useful reduced-DAG retry
    needs a scheduling-safe multi-kernel/subgraph design or a single kernel that
    does not wait on blocks that may not have been scheduled yet.
  - Do **not** chase a "full-attn version" of the retained linear shared
    SiLU+down-rotate fusion. The 10 full-attention verifier layers already use
    the small-batch `silu_mul_dual_rotate_out_fp16` shared path; the promoted
    linear slice only fixed the 30 linear-attention C-dispatch path that still
    had `silu_mul_separate_out_fp16 + paro_rotate1_fp16`.
  - 2026-06-12 no-hold: also do **not** chase a full-attn split-output version
    of the retained linear shared gate/up output-tiled route. The split GEMV
    primitive is exact, but replacing the packed full-attn shared path with
    split outputs plus `silu_mul_pair_rotate_out_fp16` failed exact AR on
    quicksort D32; code removed.
  - Do **not** spend MTP sprint time on
    `HIPENGINE_LINEAR_GDN_PREFILL_ROTATE_FUSED`. That diagnostic is a prefill
    tail fusion rejected in `docs/OPTIMIZE.md`; the current chain verifier uses
    the retained decode-tail `paro_rotate1_f32_to_fp16` slice already.

### Acceptance-Density Endgame

This is the place to bank policy ideas while keeping the current implementation
push focused on wall-clock reductions. At the current B=3 density, wall cuts
alone need about `18.2 ms/cycle` (`2.012` visible tokens/cycle at the current
AR denominator); if the next reduced-DAG/proposer units leave the row near
`19-20 ms/cycle`, acceptance density becomes the co-equal lever. Policy changes
come after the wall-cut list, but the diagnostic counters should be designed now
so B sweeps, vocab-cap tests, fallback, and tree/sibling retests all use the
same evidence.

| Rank | Diagnostic / policy | Why it matters | First gate |
| --- | --- | --- | --- |
| A0 | Acceptance-density instrumentation pass | Do this before changing policy so B sweeps, vocab-cap tests, adaptive fallback, and tree/sibling retests all use the same evidence. | Add per-depth accept histograms, rejection reason counters, target top-1 ids for rejected rows, cap-representability flags, per-prompt zero-accept streak counters, and the first-rejected-depth reason to the exact 9-prompt D32 output. Keep any extra D2H/logging behind a diagnostics flag. This is diagnostics only: no policy change, no speed row, and no promotion until a later same-suite A/B improves `actual_ratio`. |
| A1 | B=4/B=5 economics with per-position acceptance histograms | At `19.334 ms/cycle`, the current `2.012` visible tokens/cycle still needs about `2.14` visible tokens/cycle for AR break-even. Higher B is the simplest way to buy visible tokens if deeper positions still accept often enough, and each added draft row shares fixed per-cycle overhead including host orchestration, verifier setup, and final LM-head/sample work. | Add per-depth accept counters across the exact 9-prompt D32 suite, then run B=4 first and B=5 only if the depth-3/depth-4 histogram justifies it. Promote only if exact AR holds and suite `actual_ratio` improves after the added verifier row cost; the marginal row must add visible tokens faster than AR's `~0.111 tokens/ms`. A useful precondition is depth-3 acceptance around the `25-30%` range or better, but the same-suite ratio is the final gate. |
| A2 | Draft vocab-cap rejection census (`32768 -> 65536/full`) | A correct target token outside the draft cap is a guaranteed rejection. If cap-caused first rejections are nontrivial, raising the cap may buy density for proposer-only LM-head cost. | Instrument rejected depths with target top-1 token id, proposed id, cap status, and whether the target token would have been representable at `65536` or full vocab. Count first rejections where the target top-1 token at the rejected depth is out of the current cap separately from genuine prediction misses; do not infer this from the proposed id alone. If cap-caused misses are around `5-10%` of rejections or more, A/B cap `32768`, `65536`, and full vocab for exactness, acceptance, proposer wall, and total ratio; promote only if recovered acceptance pays the larger proposer LM-head cost. |
| A3 | Adaptive B / AR fallback for zero-accept streaks | Some prompts or phases pay the full verify wall to accept no draft tokens. Dropping to B=1 or AR after repeated zero-accept cycles can raise suite average without changing kernel math. | Exact output must remain identical. Gate on a fixed online policy, 9-prompt D32 suite, and per-prompt reporting; do not use future prompt knowledge or hide prompt-level regressions behind the average. |
| A4 | Tree or rejection-boundary sibling retest | Full B=3 tree is exact but negative on the current stack. Lower wall does not by itself make tree overhead cheaper, so only revisit if histograms show first-rejection cases a sibling can recover. | Revisit only after the reduced-DAG/proposer wall path stabilizes. Prefer a chain-plus-one-sibling-at-first-rejection diagnostic before reopening full tree search; treat the possible `+0.3-0.5` visible tokens/cycle as a hypothesis, not a claim. Compare against chain at the same B and report added rows, acceptance lift, wall, and ratio. |
| A5 | Relaxed speculative sampling | This is the known theoretical acceptance ceiling, but it changes the exact top-1 accept contract and needs distribution access from both models. | Out of scope for the exact-default sprint. Treat as explicit opt-in quality tier, never as a default speed row; it needs a separate accept/reject kernel and distribution-read cost model. |

For B sweeps, use the ratio gate rather than intuition: the current row is
`2.012 / 19.334 = 0.1041` visible tokens/ms, while AR is about
`1 / 9.030 = 0.1107` tokens/ms. A larger B row must either beat the current
suite ratio immediately or show enough per-depth acceptance to justify the extra
verify row after the next wall-cut unit lands. As a concrete marginal rule, if
B=4 adds roughly `3-4 ms/cycle`, it needs about `0.33-0.44` additional visible
tokens/cycle just to match the AR denominator. The stable quicksort trace is a
useful wiring smoke only, not a policy oracle: its B=3 accepted lengths
`[3,3,2,0,2,0,0,1,3,0,2,0,2]` reach depth 3 on `4/13` cycles, which is enough
to justify instrumenting B=4 but not enough to promote it. The A1 decision must
come from D32 suite histograms and same-suite wall/ratio, not the quicksort
trace.

For vocab-cap work, count representability at the first rejected depth before
running a cap sweep. If target top-1 tokens outside the current `32768` cap are
not a visible share of rejections, larger caps are likely just proposer LM-head
cost. If they are, compare `32768`, `65536`, and full vocab on exactness,
acceptance density, proposer wall, and total ratio in the same artifact.

Relaxed speculative sampling is deliberately excluded from this sprint's default
lane. It can preserve a target distribution with acceptance probability
`min(1, p_target(x) / p_draft(x))`, but it requires distribution access and a
new accept/reject kernel rather than the current exact top-1 equality gate.

Meta-bias for this sprint: prefer launch-count reduction, host round-trip
compression, graph capture, and C-side batching only when it removes actual DAG
nodes. VTILE=8 GDN, fused LM-head, refreshed RMSNorm+rotate, and current
selected-rotate experiments all showed that isolated kernel-body wins are mostly
noise or negative on this stack.

| Priority | Lever | Target saving | First action | Gate | Status / notes |
| --- | --- | ---: | --- | --- | --- |
| P0 | Re-artifact locked baseline | 0 ms | Rerun exact B=3 chain graph-auto + cap32768 + device-expert-dispatch config and emit a compact artifact. | Exact same-session AR; ratio within noise of `0.758x`; wall near `27.8 ms`. | **Done 2026-06-11.** Fresh audit row: `84.314` vs `111.769 tok/s` = `0.754x`; accepted lengths unchanged. Keep `0.758x / 27.8 ms` as the sprint's best locked baseline. |
| P0 | Current verify profile refresh | Diagnostic | Run `scripts/mtp_verifier_rocprof.py` on the locked config with callsite/family rollup. | Family split reconciles with `22.0 ms` verify wall; no unexpected fallback kernels. | **Done 2026-06-11.** Post-warmup verify profile: `19.73 ms/pass` host window, `15.33 ms/pass` kernel, `972` calls/pass. |
| P1 | Finish M16.4 dual output-tiling | -0.46 ms/pass kernel / -0.57 ms/pass host measured; default suite -0.61 ms/cycle verify | Add a split-output output-column-tiled dual W4 kernel for prefill-style ABIs, then route only exact-suite-safe sites. | Byte-exact W4 gates; same-session exact B=3 smoke; rocprof shows `w4_dual_prefill_smallbatch` and remaining single-prefill shrink. | **Promoted default-on 2026-06-11.** Split-output kernel is byte-exact and C-dispatch-routed for the prompt-suite-safe linear-attn shared gate/up site. Same-suite D32 off/default A/B with cast+rotate held exact `9/9`, identical acceptance, speed `0.652x -> 0.671x` AR, cycle wall `28.43 -> 27.83 ms`, verify `22.98 -> 22.37 ms`; opt out with `HIPENGINE_W4_DUAL_OUTPUT_TILED_SPLIT_PREFILL=0`. |
| P1 | Remove glue launches / copy-cast floor | -1.5 to -2.2 ms full target; -0.27 ms/pass host measured for first slice; default suite -0.61 ms/cycle verify | Fuse or alias producer outputs into the next RMSNorm/rotate/GEMV inputs; eliminate pure `copyBuffer`/format-cast nodes from verifier hot path. | RED layout/lifetime tests; exact B=3 smoke; launch count and cycle wall both drop. | **First slices promoted 2026-06-11.** Linear-attn out-proj `f32_to_fp16 + paro_rotate1` fusion is byte-exact vs the old chain and removes 30 launches/pass (`972 -> 942`), with neutral kernel time and lower host overhead. Same-suite D32 off/default A/B is exact `9/9` and non-regressive on every prompt; opt out with `HIPENGINE_LINEAR_OUT_CAST_ROTATE_FUSED=0`. Current graph-auto suite also proves the earlier selected-down staged path is now slower because of capture-safe barrier/fill overhead; flipping `HIPENGINE_SELECTED_MOE_DOWN_STAGED` to opt-in keeps exact `9/9`, identical acceptance, and moves cycle wall `27.648 -> 27.408 ms/cycle` plus verify `22.377 -> 22.131 ms/cycle`. A follow-up graph-off current-best compound retest is also no-hold: `decode_batched + graph_off + skip` with `HIPENGINE_SELECTED_MOE_DOWN_STAGED=1` stays exact but regresses ratio `0.8252x -> 0.8204x`, wall `21.661 -> 21.763 ms/cycle`, and verify `16.511 -> 16.628 ms/cycle`. The verifier accept summary now packs seven tiny D2H reads into one int32 payload by default (`HIPENGINE_VERIFY_ACCEPT_PACKED_PAYLOAD=0` restores the old path): same-tree D32 A/B stayed exact `9/9` with identical accepted lengths/active budgets and moved cycle wall `27.279 -> 27.122 ms/cycle`, verify `22.162 -> 21.997 ms/cycle`. The C dispatcher now also routes linear-attn shared-down rows through the existing W4 output-tiled policy instead of a stale fused-prefill call: exact locked D32, identical acceptance, `w4_single_prefill_smallbatch` `0.350 -> 0 ms/pass`, verifier kernel `14.594 -> 14.538 ms/pass`, host marker `18.621 -> 18.538 ms/pass`. The M12.6 `single_linear_out` W4 multi-row site is now default-on after the no-env stack repeatedly failed exact AR on `translation` without it; patched default is exact `9/9`, improves prompt-suite wall `27.122 -> 26.921 ms/cycle` vs the prior exact suite, and moves the locked profile kernel `14.538 -> 14.496 ms/pass`. The M12.6 `single_full_v` site is now default-on too after a fresh no-env A/B stayed exact `9/9` and moved wall `27.001 -> 26.946 ms/cycle`, verify `21.890 -> 21.817 ms/cycle`, and locked profile kernel `14.496 -> 14.448 ms/pass`. **No-hold:** fusing full-attention `qwen35_split_qgate_fp16 + fp16_to_f32(key)` into `qwen35_split_qgate_fp16_key_f32` is bit-exact and removes 10 launches/pass in the locked profile (`932 -> 922`), nudging host `18.269 -> 18.234 ms/pass`, but two exact 9-prompt D32 A/B pairs with identical acceptance regressed average wall/verify (`26.925 -> 27.010 ms/cycle`, `21.754 -> 21.828 ms/cycle`), so `HIPENGINE_FULL_QKV_SPLIT_KEY_FUSED` stays opt-in. Continue with higher-reach glue: remaining fill/copy elimination, rotate launch consolidation, and overlap. |
| P1 | Cache invariant verifier scratch objects, tensor lookups, resident views, MLP policy, and scratch generation | -0.394 ms/cycle retained for scratch objects; -0.019 ms/cycle retained for tensor lookup cache; -0.217 ms/cycle retained for resident views; -0.619 ms/cycle retained for MLP scratch policy; -0.113 ms/cycle retained for scratch generation stamp | Cache fixed-shape scratch dataclasses keyed by `(layer_id, rows)`, memoize immutable model tensor lookups by raw caller name on each decode state, cache resident non-owning state/cache Tensor views, make verifier MLP scratch reservation match the live c1/grouped MoE threshold, and replace hot cache-hit workspace revalidation with a generation stamp. | No math changes. Exact quicksort + exact 9-prompt D32 with identical acceptance; graph-auto and graph-off profile controls; wall and host-window evidence. | **Scratch objects promoted default-on 2026-06-11.** `HIPENGINE_VERIFY_SCRATCH_CACHE=1` caches fixed-row linear-attention and MLP scratch objects with workspace-pointer validation and invalidates on prefill restore / graph-cache invalidation. Exact D32 off/on stayed `9/9` with identical visible/accepted cycle aggregates (`2.01185` / `1.01185`), while wall moved `27.0958 -> 26.7015 ms/cycle`, verify `21.9328 -> 21.5511 ms/cycle`, proposal/update `1.9880 -> 1.9725 ms/cycle`, and actual ratio `0.6860x -> 0.6987x`. Graph-auto rocprof showed the expected small replay-time host change (`18.290 -> 18.275 ms/pass`, calls unchanged at `932/pass`) because steady graph replay skips most Python rebuild work; graph-off control proved the raw host win (`33.469 -> 32.988 ms/pass`). **Weight tensor lookup cache promoted default-on 2026-06-12.** `HIPENGINE_WEIGHT_TENSOR_LOOKUP_CACHE=1` avoids repeated prefix normalization and allocation unwraps; exact D32 off/on stayed `9/9` with identical visible/accepted cycle aggregates, wall `26.6621 -> 26.6433 ms/cycle`, verify `21.5290 -> 21.4984 ms/cycle`, and actual ratio `0.69160x -> 0.69200x`. Graph-off control showed raw host `34.757 -> 32.288 ms/pass`; graph-auto was neutral/noisy (`18.218 -> 18.236 ms/pass`). **Resident Tensor view cache promoted default-on 2026-06-12.** `HIPENGINE_RESIDENT_TENSOR_VIEW_CACHE=1` caches `_slot_linear_state`, `_slot_full_cache`, and `_full_cache_all_slots` views with invalidation on reset/sequence-capacity/state-cache rebuild. Exact D32 off/on stayed `9/9` with identical visible/accepted aggregates and identical per-prompt accepted lengths, while wall moved `26.6424 -> 26.4259 ms/cycle`, verify `21.5059 -> 21.2785 ms/cycle`, and actual ratio `0.69239x -> 0.69857x`; graph-off host control moved `32.52 -> 31.70 ms/pass`, graph-auto was neutral/noisy (`18.235 -> 18.244 ms/pass`). **Verifier MLP scratch policy promoted default-on 2026-06-12.** `HIPENGINE_VERIFY_MLP_SCRATCH_POLICY_ALIGNED=1` reserves c1 scratch for locked B=3 rows until `_verify_moe_grouped_min_tokens()` just like the t-loop MoE path, and keys the verifier MLP scratch cache by policy. Exact D32 opt-out/default A/B stayed `9/9` with identical accepted lengths/active budgets, wall `26.3089 -> 25.6898 ms/cycle`, verify `21.1757 -> 20.5228 ms/cycle`, and ratio `0.7003x -> 0.7172x`; graph-auto profile kept `932` calls/pass and moved host `18.314 -> 18.246 ms/pass`, graph-off host `32.445 -> 32.273 ms/pass`. **Scratch generation stamp promoted default-on 2026-06-12.** `HIPENGINE_VERIFY_SCRATCH_GENERATION_STAMP=1` stores cache entries with a generation bumped by `_clear_verify_scratch_caches()` and skips `_workspace_tensor_matches` on generation-matched hits. Exact D32 opt-out/default A/B stayed `9/9` with identical accepted lengths/active budgets, wall `25.7085 -> 25.5955 ms/cycle`, verify `20.5460 -> 20.4342 ms/cycle`, and ratio `0.7145x -> 0.7252x`; graph-auto profile kept `932` calls/pass and moved host `18.322 -> 18.298 ms/pass`, graph-off host `32.659 -> 31.971 ms/pass`. Opt out with `HIPENGINE_VERIFY_SCRATCH_CACHE=0`, `HIPENGINE_WEIGHT_TENSOR_LOOKUP_CACHE=0`, `HIPENGINE_RESIDENT_TENSOR_VIEW_CACHE=0`, `HIPENGINE_VERIFY_MLP_SCRATCH_POLICY_ALIGNED=0`, or `HIPENGINE_VERIFY_SCRATCH_GENERATION_STAMP=0`. Deeper raw-pointer structs are only worth doing as part of full-layer reduced-DAG batching. |
| P1 | Pack verifier dynamic metadata H2D | -0.029 ms/cycle wall measured | Pack per-cycle token/position/context metadata into one int64 H2D copy and unpack device-side. | Exact 9-prompt D32 with identical acceptance; rocprof confirms the unpack kernel; DFlash shared-path smoke passes. | **Promoted default-on 2026-06-11.** `HIPENGINE_VERIFY_PACK_DYNAMIC_METADATA=1` replaces five tiny dynamic metadata H2D submissions with one packed copy plus `unpack_verify_chain_dynamic_metadata_i64_kernel`. Same-session D32 A/B stayed exact `9/9` with identical accepted lengths/active budgets and moved actual ratio `0.68417x -> 0.68898x`, cycle wall `27.02196 -> 26.99252 ms/cycle`, verify `21.87984 -> 21.85918 ms/cycle`, and proposal/update `1.95733 -> 1.95069 ms/cycle`. A 27B dense DFlash D16 one-prompt shared-verifier smoke passed. Opt out with `HIPENGINE_VERIFY_PACK_DYNAMIC_METADATA=0`. |
| P1 | Chunk linear-state commit copies | -0.048 ms/pass commit kernel / -0.021 ms/cycle suite verify measured | Split the fused linear-state commit copy into 64 KiB chunks so large recurrent-state rows use multiple CTAs per layer/family. | Exact quicksort; exact 9-prompt D32 with identical acceptance; rocprof commit kernel and verifier sub-window improve; DFlash shared-path smoke passes. | **Promoted default-on 2026-06-11.** `HIPENGINE_LINEAR_STATE_COMMIT_CHUNKED=1` keeps the M12.4 one-launch pointer-table ABI but changes the copy grid from one CTA per `(layer,family)` row to 64 KiB chunks. Rocprof moved `linear_state_pair_commit` `0.250 -> 0.203 ms/pass`, total verifier kernel `14.395 -> 14.341 ms/pass`, and host marker `18.301 -> 18.263 ms/pass`. The D32 prompt suite stayed exact `9/9` with identical accepted lengths/active budgets and moved verify `21.8518 -> 21.8308 ms/cycle`; whole-cycle wall was neutral/noisy (`26.9920 -> 27.0228 ms/cycle`) because proposal/update moved the other way. A 32 KiB chunk trial made the commit kernel only ~2 us faster but worsened total profile, so 64 KiB is retained. Opt out with `HIPENGINE_LINEAR_STATE_COMMIT_CHUNKED=0`. |
| P1 | M13.B.1+B.2 selected-GEMV/shared rotate fusion audit | 0 ms retained | Confirm whether the review item is fresh work or an already-measured rotate path. | Evidence must identify a live `GEMV -> rotate` pair that is not covered by M13.B.1/B.2/B.3, M15.4, or the promoted linear-out cast+rotate slice. | **Closed/no-hold from existing evidence.** M13.B.1 already added the transposed selected-dual rotate extern/wrapper and stayed exact, but launch savings were overwhelmed by redundant in-LDS rotate work: kernel time/pass `17.32 -> 29.76 ms`, `moe_gate_up_dual_gemv` `1.86 -> 14.21 ms`, suite cycle cost `3.613 -> 3.658`. M13.B.2 shared-expert staged rotate also stayed exact but net launch delta was zero because a barrier memset replaced the saved rotate launch. M13.B.3 staged selected gate/up was later exact but still kernel-negative. Do not reopen this class without a new design that rotates once per row and avoids per-launch barrier/reset overhead. |
| P1 | Full-layer reduced-DAG batching, not a C-only loop | -1.5 to -3.0 ms estimated if launches/fills/copies disappear | Audit the non-MoE layer surround and pick one launch-removing unit first. Use C-side batching only as a vehicle for fewer real DAG nodes, not as a wrapper around the same `hipLaunchKernelGGL` sequence. | Exact 9-prompt D32 with same acceptance; host marker, call count, and named launch families fall; graph-auto and graph-off controls do not regress. | **Second design priority after host-cache.** The external ranking is right to bias toward dispatch/host compression, but M14.dispatch.1/M16.2 already show pure C-side reissue of the same kernels is parity. The next work must remove launches or synchronization/fill/copy nodes: a layer-surround megakernel, a structural MoE/attention composite, or a proven write-before-read fill elimination. Do this before another multi-stream overlap attempt. |
| P1 | Producer-side RMSNorm+rotate re-test | 0 ms retained; prior target was -0.5 to -1 ms | Re-test the existing M15.4 producer-side input RMSNorm + PARO rotate2 fusion on the current P1-default stack before spending new kernel time here. | Bit-exact candidates/AR output; launch count, verifier kernel time, and host window all improve before promotion. | **No-hold refreshed 2026-06-11.** `HIPENGINE_FUSED_RMSNORM_ROTATE=1` stayed exact, but reduced calls only `943.0 -> 915.9/pass` while worsening verifier kernel `13.41 -> 14.09 ms/pass` and host window `18.45 -> 19.05 ms/pass`; keep default-off. The live MTP sidecar uses RoPE RMS+rotary rather than PARO rotates, so the next retained levers are overlap and device-resident proposer/update work. |
| P1 | Current-stack `decode_batched` graph-off validation | -2.415 ms/cycle vs graph-off batched skip | Retest the old quicksort-positive full-attention decode kernel path after graph-off canonicalize skip removes the host penalty. | Exact D32 9-prompt suite; identical acceptance vs graph-off batched skip; profile kernel/host both drop. | **Promoted 2026-06-12.** `chain_attn_mode=decode_batched` still requires `graph_mode=off`, but graph-off is now competitive after MTP skips post-verify canonicalization. Exact `9/9`, identical accepted lengths/active budgets, ratio `0.7730x -> 0.8252x`, wall `24.076 -> 21.661 ms/cycle`, verify `18.933 -> 16.511 ms/cycle`, profile calls `932 -> 942/pass`, kernel `14.330 -> 12.922 ms/pass`, and host `18.272 -> 16.849 ms/pass`. Use this as the current exact B=3 verifier baseline. |
| P2 | Multi-stream overlap spike | 0 ms retained; prior target was -1 to -3 ms | Prototype verifier-layer dispatch on 2/4 streams with event dependencies around independent W4/MoE/GDN work. | Microbench shows measurable overlap before runtime integration; exact smoke after integration. | **No-hold C-dispatch branch split 2026-06-11.** Selected/shared MoE branch overlap inside the real C dispatcher stayed exact only after restricting the gate to verifier-sized batches; the naive version also touched linear-attn prompt prefill and changed the AR continuation. Verifier-only graph-auto D32 kept locked acceptance but worsened target all-cycle wall `26.10 -> 28.29 ms/cycle` and verify `0.266 -> 0.295 s`, despite steady markers improving `22.60 -> 21.30 ms`. Graph-off also worsened (`37.77 -> 44.22 ms/cycle`). Do not retain the code; revisit only with a graph-capture/amortization design that preserves all-cycle D32 economics. |
| P2 | Device-resident proposer chain advance | -0.5 to -1.5 ms | Use the graph-safe device expert dispatch to keep proposer update/advance in device-resident batches. | Candidate token sequence identical to baseline; cycle wall improves. | **First slices promoted default-on 2026-06-11.** Persistent proposer now skips discarded expert-topk host reads, skips intermediate lm-head/argmax for update-only accepted-token state advances, and skips the final draft snapshot save because the live proposer state already is that snapshot. Same-suite D32 9-prompt off/default A/B is exact `9/9` with identical acceptance and visible tokens; read/result skip moved actual speed `0.664x -> 0.670x` AR, cycle wall `27.94 -> 27.68 ms`, proposal/update `2.145 -> 2.052 ms`. Snapshot skip refresh stayed exact `9/9`, skipped `142` final snapshot saves, and moved cycle wall `27.676 -> 27.648 ms` plus proposal/update `2.052 -> 2.045 ms`; actual MTP/AR ratio was flat within run noise (`0.6701 -> 0.6699`). Per-advance token/position scalar H2D copies are now stream-ordered `memcpy_async`, preserving exact `9/9` while moving cycle wall `27.408 -> 27.253 ms/cycle` and proposal/update `2.035 -> 1.999 ms/cycle`; AR-normalized ratio is flat within noise (`0.67845 -> 0.67830`). The persistent chain now also skips the unused lm-head top-1 logit-value D2H while still reading the required token id, improving the D32 suite to `0.6876x` with cycle wall `27.201 ms/cycle` and proposal/update `1.973 ms/cycle`. The proposer token+position metadata H2D is now packed into one 16-byte copy per advance (`HIPENGINE_MTP_PROPOSER_PACK_TOKEN_POSITION=0` restores the old two-copy path); same-tree D32 A/B stayed exact `9/9` and nudged wall `26.922 -> 26.869 ms/cycle` plus proposal/update `1.9766 -> 1.9758 ms/cycle`, with AR-normalized ratio noisy/down because the AR control was faster. Route 0 now initializes the FP32 MoE accumulator in the first accumulation kernel instead of launching a standalone memset (`HIPENGINE_MTP_PROPOSER_ROUTE0_ACCUM_INIT=0` restores the old path); same-suite D32 A/B stayed exact `9/9`, identical acceptance, wall `27.081246 -> 27.079143 ms/cycle`, and proposal/update `1.96299 -> 1.95303 ms/cycle`, while ratio was noisy/down because AR changed. K rotary and V projection producers now write directly into the sidecar cache slots (`HIPENGINE_MTP_PROPOSER_DIRECT_KV_WRITE=0` restores the old temp-buffer plus two-D2D-copy path); exact quicksort and exact D32 `9/9` held identical acceptance, proposal/update moved `1.9955 -> 1.9801 ms/cycle` with 8/9 prompts improving, and total wall was flat/noisy-negative because verify moved independently. Opt out of the read/result/snapshot/logit-value skips with `HIPENGINE_MTP_PROPOSER_SKIP_UNUSED_READS=0`. **No-hold:** partial-accept replay removed more D2D snapshot saves (`285 -> 148`) but worsened proposal/update `2.035 -> 2.340 ms/cycle` and cycle wall `27.408 -> 27.680 ms/cycle`, so the experiment code was removed. Device-chain candidate buffering stayed exact and ran for all `142` draft cycles, but regressed speed `0.6876x -> 0.6795x`, cycle `27.201 -> 27.244 ms`, and proposal/update `1.973 -> 1.978 ms`; that experiment code was also removed. Continue with real batched proposer/update work rather than per-depth token-pointer chaining. |
| P1 | Proposer router top-k specialization | -1.31 ms/cycle wall retained | Fuse the generic single-row router `topk_rows_i32` + softmax path for the actual 256-expert/top-8 sidecar shape. | Exact quicksort + D32 suite: identical candidate sequence, accepted lengths, active budgets, and top-k order/tie behavior; proposer marker profile shows router family falls. | **Promoted default-on 2026-06-12.** `HIPENGINE_MTP_PROPOSER_ROUTER_TOPK_FUSED=1` is exact `9/9`, identical accepted lengths/active budgets, ratio `0.8244x -> 0.8806x`, wall `21.686 -> 20.379 ms/cycle`, proposal/update `1.974 -> 1.460 ms/cycle`. Proposer profile moved router family `1.714 -> 0.373 ms/cycle`, kernel `4.676 -> 3.379 ms/cycle`, host `5.578 -> 4.248 ms/cycle`, calls `181.5 -> 178.4/cycle`. |
| P2 | M12.7 graph-capture proposer loop | 0 ms retained from whole-body capture; subgraph design still TBD | Capture only after the body is split into a stream-safe fixed-address subgraph or HIP graph node parameter updates are available. | Exact 9-prompt D32 with same accepted lengths; proposal/update must drop; no capture-freeze or stale scalar state across prompts. | Profile-bounded and currently no-held as a whole-body graph. Post-route-batching proposer all-region host/kernel gap is about `0.53 ms/cycle`; capture cannot remove the `~3.02 ms/cycle` proposer GPU work. A bucketed-attention live-context-only probe stayed exact but regressed quick-smoke proposal/update by `+0.011 ms/cycle`; indexed K/V producer variants stayed exact but no-held as a standalone speed row. A follow-up whole-body graph diagnostic using fixed-address indexed K/V plus bucketed attention changed the quicksort accepted trace under private-stream capture, even when recapturing every advance, and default-stream capture is rejected by HIP. Do not retry M12.7 as a whole-body HIP graph around `advance_with_previous_hidden()`; use only a smaller capture-safe subgraph or graph-node parameter updates. |
| Closed | Eliminate per-layer memset/fill launches | 0 ms retained on current stack | Reopen only if a fresh trace shows a material live fill/copy family. | RED lifetime/unit test plus exact 9-prompt D32; launch family disappears or shrinks; no stale-data failures under graph replay. | The current verifier trace makes this stale as a sprint item: `fillBufferAligned` is absent and `runtime_copy` is only `2` calls/pass at about `0.0067-0.0069 ms/pass`. Do not spend wall-cut time here without new profile evidence. |
| P3 | Revisit top-k/tree after wall cut | Acceptance margin | Re-run gated tree/top-k only after verify wall is materially lower or a better acceptance head is available. | Tree beats chain on the same wall and same prompt suite. | Current B=3 tree is default-off negative. |
| No-go | p_min / whole-pass persistent / selected-FFN megakernel / exact PARO rotate-pair fusion / fused verifier LM-head / wider GDN dv-tiling / routed-expert WMMA proposer dense / launch-only full-QKV split+key-cast fusion / accept-position fused state update / final RMSNorm+cast micro-fusion | 0 ms | Do not spend sprint time here unless new evidence changes the bottleneck. | n/a | Measured neutral, negative, or non-exact on this stack. Includes p_min/sync trims, consumer-side rotate fusions, refreshed M15.4 producer-side RMSNorm+rotate2, and the current-stack `HIPENGINE_PARO_FFN_MEGAKERNEL=1` recheck: it fired but failed exact AR at token index 9 (`156973` vs `149315`), so it stays default-off. The existing `HIPENGINE_DFLASH_VERIFY_FUSED_LM_HEAD=on` path is exact on locked quicksort but no-hold for MTP: calls/pass `935 -> 934`, kernel `14.594 -> 15.236 ms/pass`, host `18.621 -> 19.234 ms/pass`; it removes one argmax launch but makes the W8A16 body slower (`1.4435 -> 2.0953 ms/pass`). GDN `VTILE=8` stayed exact and halved chain blocks (`1024 -> 512`) but raised VGPR (`64 -> 80`) and regressed locked GDN `1.7559 -> 1.7597 ms/pass` plus total kernel `14.5941 -> 14.6308 ms/pass`. Routed-expert WMMA BF16 dense for the sidecar proposer was exact on quicksort and `9/9` prompts, but regressed D32 wall `27.011 -> 27.149 ms/cycle` and proposal/update `1.980 -> 2.068 ms/cycle`; experiment code removed. Full-QKV split+key-cast fusion is bit-exact and profile-positive on launch count, but repeated exact prompt-suite A/B regressed wall/verify, so it is opt-in diagnostic only. Fusing the resident base-slot position/context update into the packed accept kernel removed one scalar launch/pass (`932 -> 931`) and stayed exact, but profile host window and 9-prompt wall/verify both regressed (`27.038 -> 27.135 ms/cycle`, verify `21.884 -> 21.969 ms/cycle`), so `HIPENGINE_VERIFY_ACCEPT_UPDATES_POSITION` stays opt-in. Final RMSNorm+cast micro-fusion stayed exact and removed one profile launch, but same-suite D32 wall/verify regressed; experiment code removed. |

Additional no-hold from the 2026-06-11 proposer pass: fusing the sidecar
shared-gate FP32 accumulation and FP32-to-BF16 finalize into one helper stayed
exact on quicksort and exact `9/9` on the D32 prompt suite with identical
acceptance, but regressed aggregate wall `27.0015 -> 27.1124 ms/cycle` and
actual ratio `0.6840x -> 0.6816x`; the proposal/update nudge was only
`1.95247 -> 1.95093 ms/cycle`. Experiment code was removed. Do not re-open this
single-row fuse unless it is part of a broader proposer graph/batch design.

Live-tree corrections to the external review ranking:

- `single_linear_out` and `single_full_v` exact W4 multi-row routing are already
  default-on and measured above, so they are not next work.
- The rotate-fusion candidate from the review maps to already-measured M13.B
  paths. Reopen only for a genuinely new design, not the selected-dual
  transposed entry point, no-held producer RMSNorm+rotate2, or already-promoted
  linear-out cast+rotate slice.
- The selected-GEMV/shared rotate audit is already closed by M13.B.1/B.2/B.3
  no-holds. The next bias is launch-count and round-trip compression, but not
  a broad native host-loop rewrite: M14.dispatch.1 and M16.2 show that reissuing
  the same kernels from C is parity. The useful version is reduced-DAG batching
  that removes launches/fills/copies outright, then proposer graph capture and
  low-risk fill removal. Multi-stream overlap should wait until the DAG is
  smaller; the direct branch-overlap spike already regressed all-cycle wall.
- The latest host-overhead review found two numerically-identical caching targets
  outside MoE: fixed-shape verifier scratch object reuse and non-MoE
  weight-pointer hoisting. The scratch-object slice and a first model tensor
  lookup cache are now retained default-on; only build deeper raw pointer structs
  if they simplify a reduced-DAG/full-layer dispatcher.
- The same review's `single_linear_out` and `fillBufferAligned` notes were
  checked against the current locked profile: `w4_single_prefill_smallbatch`
  is gone (`single_linear_out`/`single_full_v` are already default-on), and
  current raw traces show `fillBufferAligned=0`. Keep both as stale-profile
  corrections unless a fresh profile reintroduces those buckets.
- The 2026-06-12 review's `decode_batched` item was first no-held against the
  old graph-off baseline, then promoted after the MTP-only canonicalize skip
  removed the graph-off host penalty. This remains the current verifier
  baseline: exact `9/9` with `chain_attn_mode=decode_batched`, graph `off`,
  wall `21.661 ms/cycle`, verify `16.511 ms/cycle`, and ratio `0.8252x`
  before the fused proposer router and shared-down+combine epilogue are layered
  on top.
- The same review's `decode_batched + HIPENGINE_SELECTED_MOE_DOWN_STAGED=1`
  compound has now been checked against that current best and no-held: exact
  `9/9` with identical accepted lengths/active budgets, but ratio regressed
  `0.8252x -> 0.8204x`, wall `21.661 -> 21.763 ms/cycle`, and verify
  `16.511 -> 16.628 ms/cycle`.
- The same review's final-norm BF16-output item is stale: BF16 RMSNorm output
  kernels/wrappers already exist in the runtime. The GDN tile-left item is also
  stale for the suggested VTILE path: `VTILE=8` was exact but no-held on
  2026-06-11 because total verifier kernel time regressed.
- The 2026-06-12 host-cache review's state/cache view idea is promoted
  default-on: `_slot_linear_state`, `_slot_full_cache`, and
  `_full_cache_all_slots` reuse non-owning views with explicit invalidation on
  session reset and any resident state/cache rebuild.
- The same review's `_workspace_tensor_matches` bypass is promoted default-on
  as a generation-stamp optimization, not as deleted validation. The scratch
  cache generation bumps whenever verifier scratch caches are cleared, and
  generation-matched hits can skip workspace-pointer revalidation.
- The follow-up scratch review found one concrete policy mismatch and it is now
  promoted default-on: verifier MLP scratch reservation in the runner follows
  the same c1/grouped threshold as the actual chain/tree t-loop MoE path
  (`_verify_moe_grouped_min_tokens()`, default `16`) and keys cached scratch by
  expected policy. Exact D32 improved wall `26.309 -> 25.690 ms/cycle` with
  identical acceptance.
- The MTP-only graph-off `_canonicalize_decode_scratch()` skip is promoted
  default-on in the MTP harness via `HIPENGINE_MTP_SKIP_CANONICALIZE_AFTER_VERIFY`.
  It is narrow: AR/c1 handoff paths still can request canonical scratch, while
  steady MTP verify cycles keep verifier-shaped scratch live.
- Shared-down + shared-gate/residual combine remains a reduced-DAG candidate
  only with a parallel epilogue. The first exact prototype removed the 30/pass
  linear-attn combine launches but serialized selected accumulation in one
  thread per output pack, regressing kernel `12.860 -> 13.255 ms/pass`; do not
  repeat that shape.
- Proposer route batching is now promoted default-on. The retained design uses
  route-batched gate/up expert GEMV, route-major SiLU, route-batched down GEMV,
  and one ordered accumulation kernel, preserving the scalar route accumulation
  order while cutting proposer launches nearly in half.
- Final RMSNorm+cast fusion is checked/no-hold on the current best stack. It
  removed one profile launch and stayed exact, but the 9-prompt D32 suite
  regressed wall/verify, so the experiment was removed rather than retained as
  another opt-in flag.

External review ranking, folded into the live plan:

| Review item | Disposition in this sprint |
| --- | --- |
| Fixed-shape scratch object + non-MoE weight-pointer caching | Scratch-object cache and the first model tensor lookup cache are promoted default-on after exact D32 evidence plus graph-off host controls. Deeper raw pointer structs should be folded into reduced-DAG/full-layer batching, not pursued as another standalone cache micro-slice. |
| Current-stack `decode_batched` validation | Promoted 2026-06-12 after canonicalize skip made graph-off competitive. Exact `9/9` with identical acceptance, ratio `0.7730x -> 0.8252x`, wall `24.076 -> 21.661 ms/cycle`, verify `18.933 -> 16.511 ms/cycle`; current verifier baseline is `decode_batched + graph_off + MTP canonicalize skip`. |
| `decode_batched` verifier graph capture | Checked/no-hold 2026-06-12. The one-split graph prototype validated and replayed exactly on quicksort, but focused profile showed no launch reduction and slight kernel/host regression (`832` calls/pass both ways, kernel `12.606 -> 12.674 ms/pass`, host `16.153 -> 16.182 ms/pass`); code removed. |
| `decode_batched + selected-down staged` compound | Checked 2026-06-12 after graph-off became current best; exact but no-hold. `HIPENGINE_SELECTED_MOE_DOWN_STAGED=1` regressed ratio `0.8252x -> 0.8204x`, wall `21.661 -> 21.763 ms/cycle`, verify `16.511 -> 16.628 ms/cycle`; keep staged down opt-in only. |
| Resident state/cache Tensor view caching | Promoted default-on 2026-06-12. Exact D32 `9/9`, identical accepted lengths, wall `26.642 -> 26.426 ms/cycle`, verify `21.506 -> 21.278 ms/cycle`, ratio `0.6924x -> 0.6986x`; graph-off host `32.52 -> 31.70 ms/pass`. |
| Scratch cache generation stamp | Promoted default-on 2026-06-12. Exact D32 `9/9`, identical accepted lengths/active budgets, wall `25.7085 -> 25.5955 ms/cycle`, verify `20.5460 -> 20.4342 ms/cycle`, ratio `0.7145x -> 0.7252x`; graph-auto host `18.322 -> 18.298 ms/pass`, graph-off host `32.659 -> 31.971 ms/pass`. |
| Verifier MLP scratch policy mismatch | Promoted default-on 2026-06-12. Exact D32 `9/9`, identical accepted lengths/active budgets, wall `26.309 -> 25.690 ms/cycle`, verify `21.176 -> 20.523 ms/cycle`, ratio `0.7003x -> 0.7172x`; graph-auto host `18.314 -> 18.246 ms/pass`, graph-off host `32.445 -> 32.273 ms/pass`. |
| Graph-off canonicalize-after-verify skip | Promoted default-on in the MTP harness. Exact `9/9`, identical accepted lengths/active budgets, graph-off batched wall `37.207 -> 24.076 ms/cycle`, verify `32.069 -> 18.933 ms/cycle`; profile proves this is host scratch canonicalization, not kernel math (`932/pass`, kernel ~`14.33 ms/pass` unchanged). |
| M13.B.1+B.2 selected-GEMV/shared rotate fusion | Closed/no-hold from prior exact evidence. Reopen only with a new rotate-once/no-barrier-reset design. |
| Full-layer C-dispatcher | Accepted only as reduced-DAG batching. Do not build a C-only wrapper that launches the same kernels. |
| Shared-down + shared-gate/residual combine | Parallel-epilogue retry promoted default-on 2026-06-12. Exact D32 `9/9`, identical accepted lengths/active budgets, ratio `0.8843x -> 0.8859x`, wall `20.315 -> 20.204 ms/cycle`, verify `16.523 -> 16.402 ms/cycle`; profile removes 30 combine launches/pass (`942 -> 912`) and moves kernel `12.871 -> 12.781 ms/pass`. The earlier thread-0 epilogue remains no-held and should not be restored. |
| Linear shared SiLU+down-rotate fusion | Promoted default-on 2026-06-12 through the existing exact pair-rotate kernel. Exact D32 `9/9`, identical accepted lengths/active budgets, ratio `0.9173x -> 0.9194x`, wall `19.547 -> 19.496 ms/cycle`, verify `16.278 -> 16.217 ms/cycle`; profile removes 30 launches/pass (`902 -> 872`) and moves host `16.482 -> 16.359 ms/pass`. |
| Linear A/B separate-output dual dense GEMV | Promoted default-on 2026-06-12 for small-batch rows. Exact D32 `9/9`, identical accepted lengths/active budgets, ratio `0.9201x -> 0.9240x`, wall `19.480 -> 19.440 ms/cycle`, verify `16.197 -> 16.155 ms/cycle`; profile removes 30 launches/pass (`872 -> 842`) and moves host `16.311 -> 16.220 ms/pass`. |
| Full-attn `decode_batched` one-split direct gate | Promoted default-on 2026-06-12 for the current `num_splits=1` full-attention verifier shape. Focused GPU comparison is bit-identical to split+reduce; exact D32 `9/9`, identical accepted lengths/active budgets, ratio `0.9240x -> 0.9273x`, wall `19.440 -> 19.334 ms/cycle`, verify `16.155 -> 16.053 ms/cycle`; profile removes 10 reduce launches/pass (`842 -> 832`) and moves host `16.220 -> 16.085 ms/pass`. |
| Linear QKV/Z separate-output rotate-staged dual W4 | Checked/no-hold 2026-06-12. The prototype built and passed small synthetic exactness, but the real B=3 verifier smoke hung with GPU busy due to the same-kernel producer/consumer spin barrier at the large QKV/Z grid; code removed. Reopen only with a scheduling-safe topology, not another keyed spin barrier. |
| Linear out-proj f32 rotate + W4 GEMV staged keyed | Checked/no-hold 2026-06-12. The focused GPU comparison passed bitwise against `paro_rotate1_f32_to_fp16 + gemv_awq_pack8_multi_row_transposed_fp16`, but real B=3 quicksort D32 hung with GPU busy and no JSON; code removed. Same scheduling hazard as the QKV/Z staged split. |
| `single_linear_out` exact multi-row path | Already default-on after current-stack 9-prompt exactness; not pending. |
| Specialized proposer router top-k | Promoted default-on 2026-06-12 for the actual 256-expert/top-8 sidecar shape. Exact D32 `9/9`, identical accepted lengths/active budgets, ratio `0.8244x -> 0.8806x`, wall `21.686 -> 20.379 ms/cycle`; profiler router family `1.714 -> 0.373 ms/cycle`. |
| M12.7 proposer graph capture | Keep only as a subgraph/design item. Post-route-batching proposer all-region has about `0.53 ms/cycle` host gap above kernel time. Live-context bucketed attention and fixed-address indexed K/V producers are individually exact, but the whole-body graph capture around `advance_with_previous_hidden()` no-held because private-stream capture changed the accepted trace and default-stream capture is rejected by HIP. Do not retry the same full-body shape. |
| Batch proposer expert loop | Promoted default-on 2026-06-12. Exact D32 `9/9`, identical accepted lengths/active budgets, ratio `0.8939x -> 0.9135x`, wall `20.045 -> 19.604 ms/cycle`, proposal/update `1.455 -> 1.244 ms/cycle`; proposer profile calls `178.4 -> 92.0/cycle`, kernel `3.379 -> 3.018 ms/cycle`, host `4.248 -> 3.543 ms/cycle`. |
| Verifier/proposer stream overlap | Plausible but dependency-risky; audit accepted-token, hidden-row, proposer-repair, and commit-buffer dependencies before prototyping events. |
| Per-layer memset/fill elimination | Closed on the current stack. Current retained profiles have no verifier fill family and only `2` `runtime_copy` calls/pass at about `0.0067-0.0069 ms/pass`; reopen only from fresh live-bucket evidence plus local write-before-read proof. |
| Multi-stream overlap | High-upside but blocked until the verifier DAG is smaller; prior branch-overlap attempt regressed all-cycle wall. |
| Final RMSNorm+cast fusion | Checked/no-hold 2026-06-12. Exact `9/9` with identical acceptance and one fewer profile launch, but same-suite wall/verify regressed (`20.087 -> 20.119 ms/cycle`, verify `16.298 -> 16.329 ms/cycle`); experiment code removed. |
| C-dispatch keyed cooperative MoE router | Checked/no-hold 2026-06-12. Exact quicksort smoke and unit comparison held, but the single-kernel keyed topology removed 40 launches/pass while regressing router kernel time (`0.502 -> 0.877 ms/pass`), total verifier kernel (`12.714 -> 13.080 ms/pass`), and host window (`16.482 -> 16.717 ms/pass`); experiment code removed. |

Break-even accounting: the retained stack has already moved the exact 9-prompt
D32 row from the locked sprint baseline `27.8 ms/cycle` to the current
`19.334 ms/cycle`, crossing the original `<21.5 ms` wall milestone. The row is
still `0.927x` AR because visible-token density is only about `2.012/cycle`.
At the current AR denominator (`9.030 ms/token`), true break-even at unchanged
acceptance would require about `18.2 ms/cycle`; at the current wall, it needs at
least `2.14` visible tokens/cycle. The next push must keep every retained wall
win while continuing reduced-DAG/proposer work and saving acceptance-density
policy for the endgame. The D32 9-prompt off/on A/B confirms the stacked P1
gates are exact and worth keeping by default,
and the first P2
proposer skip removes another `0.26 ms/cycle`; the follow-on final-snapshot skip
removes dead D2D snapshot work and trims proposal/update by another
`0.007 ms/cycle`. Disabling the superseded selected-down staged path by default
removes another `0.240 ms/cycle` on the exact D32 prompt suite by avoiding the
graph-era barrier/fill cost. Stream-ordering the proposer token/position scalar
H2D copies removes another small `0.155 ms/cycle` wall slice and `0.036
ms/cycle` proposal/update slice while keeping the ratio flat within run noise.
Skipping the unused lm-head top-1 logit-value D2H in the persistent chain adds
another exact proposer slice: `27.253 -> 27.201 ms/cycle`,
`1.999 -> 1.973 ms/cycle` proposal/update, and `0.6783x -> 0.6876x`.
Packing the persistent proposer's per-advance token+position metadata into one
16-byte H2D copy keeps exact `9/9` and removes one tiny H2D submission per
advance. Same-tree D32 opt-out/default A/B moved wall
`26.922 -> 26.869 ms/cycle` and proposal/update
`1.9766 -> 1.9758 ms/cycle`; the AR-normalized ratio moved
`0.6911x -> 0.6875x` because the packed row's AR control was faster, so track
this as a retained micro wall/proposer slice rather than a ratio win.
Packing the verifier accept payload adds a small exact verifier/host slice on
top: the same-tree opt-out/default A/B preserved accepted lengths and active
budgets while moving cycle wall `27.279 -> 27.122 ms/cycle` and verify
`22.162 -> 21.997 ms/cycle`. The AR-normalized ratio only moved
`0.6800x -> 0.6805x` because the same-session AR denominator changed, so track
this as retained cycle-wall progress rather than a headline ratio jump.
Packing verifier dynamic metadata adds another exact verifier/host slice: the
same-suite opt-out/default A/B replaced five tiny per-cycle H2D submissions with
one packed H2D plus an unpack kernel, kept accepted lengths and active budgets
identical, and moved cycle wall `27.02196 -> 26.99252 ms/cycle`, verify
`21.87984 -> 21.85918 ms/cycle`, proposal/update
`1.95733 -> 1.95069 ms/cycle`, and actual ratio `0.68417x -> 0.68898x`.
Rocprof confirms the unpack kernel runs; a 27B dense DFlash D16 one-prompt
shared-verifier smoke passed.
No-hold: trying to make that packed H2D stream-ordered from a persistent host
buffer stayed exact `9/9` with identical acceptance, but regressed cycle wall
`26.9165 -> 27.0911 ms/cycle`, verify `21.7799 -> 21.9470 ms/cycle`, and
actual ratio `0.68634x -> 0.68161x`; experiment code removed. Keep the retained
sync packed copy unless a broader metadata/graph design changes the lifetime and
ordering model.
Chunking the fused linear-state commit copy adds a verifier sub-window slice on
top of that stack. The same-suite D32 A/B kept accepted lengths and active
budgets identical, moved verify `21.8518 -> 21.8308 ms/cycle`, and the locked
rocprof pass moved the commit kernel `0.250 -> 0.203 ms/pass`; whole-cycle wall
was neutral/noisy (`26.9920 -> 27.0228 ms/cycle`) because proposal/update moved
the other way. Retain it as a real commit/verifier improvement, not a headline
ratio row.
Routing the C-dispatch linear shared-down projection through the existing W4
output-tiled policy adds another exact kernel-choice slice on the locked
profile: the residual 30/pass `awq_fusedw4_prefill` bucket disappears, verifier
kernel time moves `14.594 -> 14.538 ms/pass`, and the host marker window moves
`18.621 -> 18.538 ms/pass`.
Promoting the M12.6 `single_linear_out` W4 multi-row site fixes a current-stack
exactness hole and keeps the profiler moving: the no-env default before this
site repeatedly failed exact AR on the `translation` prompt (first mismatch
`5494 -> 72931` at output index 6), while the patched default is exact `9/9`.
Against the previous exact D32 suite row, aggregate cycle wall moves
`27.122 -> 26.921 ms/cycle`, verify `21.997 -> 21.799 ms/cycle`, and actual
ratio `0.6805x -> 0.6898x`; the locked quicksort profile moves kernel
`14.538 -> 14.496 ms/pass` and host marker `18.538 -> 18.497 ms/pass`.
Promoting the M12.6 `single_full_v` W4 multi-row site adds the other remaining
single-output site to the default exact mask. The retained no-env default is
exact `9/9`; against a fresh same-session no-env baseline before the mask
change, wall moves `27.001 -> 26.946 ms/cycle`, verify
`21.890 -> 21.817 ms/cycle`, and actual ratio `0.6872x -> 0.6901x`. The locked
quicksort profile moves kernel `14.496 -> 14.448 ms/pass`, host marker
`18.497 -> 18.465 ms/pass`, and `w4_single_gemv` `1.457 -> 1.399 ms/pass`.
Caching fixed-shape verifier scratch objects is the first retained host-cache
slice from the external review. The default cache stays exact `9/9` and keeps
visible/accepted cycle aggregates identical while moving exact D32 wall
`27.0958 -> 26.7015 ms/cycle`, verify `21.9328 -> 21.5511 ms/cycle`,
proposal/update `1.9880 -> 1.9725 ms/cycle`, and actual ratio
`0.6860x -> 0.6987x`. The graph-auto profile only moves host
`18.290 -> 18.275 ms/pass` because steady graph replay skips most Python
scratch rebuild work; graph-off control shows the raw host effect
`33.469 -> 32.988 ms/pass`. Keep the default-on cache. Resident state/cache
Tensor view caching, verifier MLP scratch policy alignment, and scratch-cache
generation stamps are now also promoted default-on after exact D32 evidence.
Deeper raw-pointer structs should be folded into reduced-DAG or full-layer
dispatcher work rather than chased as standalone cache polish.
The GDN follow-up tried widening dv-tiling from `VTILE=4` to `VTILE=8`; exactness
held, but the locked profile worsened because the narrower grid did not offset
extra register pressure. Keep the current `VTILE=4` GDN default.
The fused proposer router then removes the biggest proposer family without
changing acceptance: exact `9/9`, identical accepted lengths/active budgets,
ratio `0.8244x -> 0.8806x`, wall `21.686 -> 20.379 ms/cycle`, verify neutral
at `16.537 -> 16.578 ms/cycle`, and proposal/update
`1.974 -> 1.460 ms/cycle`. The quicksort proposer profile shows the cause:
router top-k+softmax collapses `1.714 -> 0.373 ms/cycle`, total proposer kernel
`4.676 -> 3.379 ms/cycle`, and proposer host window `5.578 -> 4.248 ms/cycle`.
The parallel shared-down+combine epilogue is the first retained reduced-DAG
slice on top of that row: exact `9/9`, identical acceptance, ratio
`0.8843x -> 0.8859x`, wall `20.315 -> 20.204 ms/cycle`, verify
`16.523 -> 16.402 ms/cycle`, and quicksort verify launches `942 -> 912/pass`.
Partial-accept proposer replay is not one of them: it stayed exact and cut
snapshot saves, but the 9-prompt D32 suite regressed aggregate wall and
proposal/update time, so do not re-add that flag without a new batching design.
Device-chain candidate buffering is also not enough by itself: it kept deeper
draft argmax ids on device and read candidate ids once per cycle, but the
9-prompt suite regressed aggregate speed `0.6876x -> 0.6795x`, cycle wall
`27.201 -> 27.244 ms/cycle`, and proposal/update `1.973 -> 1.978 ms/cycle`.
Do not re-add that flag without a broader batched proposer design that removes
more work than the extra D2D stores and final candidate-id read add back.

### P0 Refresh Artifacts (2026-06-11)

Locked rerun:

- Artifact:
  [`2026-06-11-hipengine-mtp-b3-locked-baseline.json`](../benchmarks/results/2026-06-11-hipengine-mtp-b3-locked-baseline.json)
- Command shape: stable quicksort prompt, D32, B=3, `persistent_device`,
  `chain_attn_mode=batched`, verifier graph `auto`, draft vocab cap `32768`,
  `HIP_VISIBLE_DEVICES=0`, `HIPENGINE_HIP_ARCH=gfx1100`.
- Result: exact same-session AR, accepted lengths
  `[3,3,2,0,2,0,0,1,3,0,2,0,2]`; AR `111.769 tok/s`, MTP
  `84.314 tok/s`, ratio `0.754x`.
- Timing nuance: all-cycle `decode_seconds / cycles` is `29.19 ms/cycle`
  because cycle 1 includes graph bucket/capture work (`77.1 ms`). Cycle markers
  after the first cycle average `23.44 ms`; after two cycles, `23.31 ms`.
  Verifier time averaged over all cycles is `21.75 ms/cycle`.

Verifier profile refresh:

- Artifact:
  [`2026-06-11-hipengine-mtp-b3-locked-rocprof.json`](../benchmarks/results/2026-06-11-hipengine-mtp-b3-locked-rocprof.json)
- `scripts/mtp_verifier_rocprof.py` now accepts `--graph-mode`, so the profile
  matches the locked verifier graph `auto` config instead of profiling a
  graph-off surrogate.
- Post-warmup marker slice (`--steady-state-skip 2`): `11` verifier passes,
  `19.73 ms/pass` host window, `15.33 ms/pass` GPU kernel time, `972`
  kernel calls/pass.

| Family | Calls/pass | ms/pass |
| --- | ---: | ---: |
| `native_prefill_attention` | 10 | 1.884 |
| `moe_gate_up_dual_gemv` | 40 | 1.880 |
| `linear_attention_gdn_decode` | 30 | 1.749 |
| `moe_down_gemv` | 40 | 1.513 |
| `w8a16_linear` | 1 | 1.442 |
| `w4_dual_gemv` | 50 | 1.392 |
| `w4_single_gemv` | 60 | 1.250 |
| `moe_paro_rotate_in` | 190 | 0.924 |
| `w4_dual_prefill_smallbatch` | 30 | 0.917 |
| `other` | 125 | 0.611 |

Quick scoped A/B (not promoted): enabling only the historically-risky
`shared_gate_up` M12.6 site on top of the safe multi-row mask stayed exact on
the quicksort prompt and moved the locked smoke from `84.31 -> 85.15 tok/s`
and verifier `21.75 -> 21.34 ms/cycle`. That is only about `0.4 ms/cycle`, and
the broader prompt suite previously found fragile sites, so do not default it on
without a 9-prompt exact-suite pass. Treat it as evidence that the next M16.4
packet is useful but not sufficient for break-even.

### Graph-Off Canonicalize Skip + Decode-Batched Current Best (2026-06-12)

Artifacts:

- Canonicalize control:
  [`D32`](../benchmarks/results/2026-06-12-hipengine-mtp-canonicalize-after-on-graphoff-9prompt-d32.json),
  [`rocprof`](../benchmarks/results/2026-06-12-hipengine-mtp-canonicalize-after-on-graphoff-rocprof.json)
- Canonicalize skip:
  [`D32`](../benchmarks/results/2026-06-12-hipengine-mtp-canonicalize-after-skip-graphoff-9prompt-d32.json),
  [`rocprof`](../benchmarks/results/2026-06-12-hipengine-mtp-canonicalize-after-skip-graphoff-rocprof.json)
- Current best:
  [`decode_batched + skip D32`](../benchmarks/results/2026-06-12-hipengine-mtp-canonicalize-skip-decode-batched-9prompt-d32.json),
  [`decode_batched + skip rocprof`](../benchmarks/results/2026-06-12-hipengine-mtp-canonicalize-skip-decode-batched-rocprof.json)
- No-hold compound:
  [`decode_batched + staged-down D32`](../benchmarks/results/2026-06-12-hipengine-mtp-decode-batched-staged-down-on-9prompt-d32.json)

Implementation:

- Added `canonicalize_after` to chain/tree verify commit entry points and wired
  the MTP harness to pass `False` by default through
  `HIPENGINE_MTP_SKIP_CANONICALIZE_AFTER_VERIFY=1`.
- The default is intentionally MTP-scoped. Real AR/c1 handoff can still request
  canonical scratch by leaving `canonicalize_after=True`.

Measurement:

- Graph-off batched canonicalize control vs skip: exact `9/9`, identical
  accepted lengths/active budgets, ratio `0.4969x -> 0.7730x`, wall
  `37.207 -> 24.076 ms/cycle`, verify `32.069 -> 18.933 ms/cycle`.
- Rocprof for that A/B proves the win is host-side: calls/pass remain `932`,
  kernel remains ~`14.33 ms/pass`, and host moves `32.505 -> 18.272 ms/pass`.
- With graph-off fixed, `decode_batched` becomes current best: exact `9/9`,
  identical accepted lengths/active budgets versus graph-off batched skip, ratio
  `0.7730x -> 0.8252x`, wall `24.076 -> 21.661 ms/cycle`, verify
  `18.933 -> 16.511 ms/cycle`, profile kernel `14.330 -> 12.922 ms/pass`, host
  `18.272 -> 16.849 ms/pass`.
- Retesting the old staged selected-down path on top of this current best stayed
  exact but did not pay: `HIPENGINE_SELECTED_MOE_DOWN_STAGED=1` moved ratio
  `0.8252x -> 0.8204x`, wall `21.661 -> 21.763 ms/cycle`, verify
  `16.511 -> 16.628 ms/cycle`, and cycle cost `2.411 -> 2.425`.

Decision: promote the MTP canonicalize skip default-on and use
`chain_attn_mode=decode_batched`, graph `off` as the current exact B=3 MTP
verifier baseline, with `HIPENGINE_SELECTED_MOE_DOWN_STAGED=1` still no-held.
After the fused proposer router, route-batched proposer expert loop, linear
shared SiLU+down-rotate fusion, linear A/B dual-separate dense projection, and
linear/full parallel shared-down+combine epilogues land on top, the current
exact B=3 MTP row is now `0.927x` at `19.334 ms/cycle`; the original
`<21.5 ms` wall milestone is crossed, but the true `>1.0x` target still needs
broader reduced-DAG/proposer work and later acceptance-density uplift.

### M16.4 Split-Output Follow-Up (2026-06-11)

Artifacts:

- Exact smoke:
  [`2026-06-11-hipengine-mtp-m16.4-dual-split-output-tiled-cdispatch-smoke.json`](../benchmarks/results/2026-06-11-hipengine-mtp-m16.4-dual-split-output-tiled-cdispatch-smoke.json)
- Verifier profile:
  [`2026-06-11-hipengine-mtp-m16.4-dual-split-output-tiled-cdispatch-rocprof.json`](../benchmarks/results/2026-06-11-hipengine-mtp-m16.4-dual-split-output-tiled-cdispatch-rocprof.json)

Implementation:

- Added `gemv_awq_dual_pack8_output_tiled_split_transposed_fp16`, a
  split-output variant of the existing dual output-column-tiled pack8 GEMV.
  The RED gate compares it byte-for-byte against the packed output-tiled dual
  kernel and passes rows `{2,4,8}` across the existing fp16 dual shapes.
- Routed the linear-attention shared gate/up C dispatcher through that symbol
  for `tokens in {2,4,8}`. The promoted default is on for the proven
  `shared_gate_up` site; opt out with
  `HIPENGINE_W4_DUAL_OUTPUT_TILED_SPLIT_PREFILL=0`.

Measurement:

- Exact quicksort smoke: AR `112.109 tok/s`, MTP `87.537 tok/s` (`0.781x`),
  accepted lengths unchanged `[3,3,2,0,2,0,0,1,3,0,2,0,2]`, verifier
  `270.6 ms` over 13 cycles vs locked `282.8 ms`.
- Post-warmup profile (`11` verifier passes): host `19.73 -> 19.16 ms/pass`,
  kernel `15.33 -> 14.87 ms/pass`, calls/pass unchanged at `972`.
- Family delta: `w4_dual_prefill_smallbatch` is eliminated (`30` calls/pass,
  `0.917 ms/pass`), replaced by `30` output-tiled dual calls inside
  `w4_dual_gemv` at `0.438 ms/pass`; net kernel saving is `0.459 ms/pass`.

Decision: promoted default-on after the D32 9-prompt off/on suite held exact
with identical acceptance and a positive wall delta. This is useful headroom,
not the break-even move by itself; the next higher-yield work remains broader
glue launch removal and multi-stream overlap.

### P1 Linear Out Cast+Rotate Slice (2026-06-11)

Artifacts:

- Exact smoke:
  [`2026-06-11-hipengine-mtp-p1-linear-cast-rotate-fused-smoke.json`](../benchmarks/results/2026-06-11-hipengine-mtp-p1-linear-cast-rotate-fused-smoke.json)
- Verifier profile:
  [`2026-06-11-hipengine-mtp-p1-linear-cast-rotate-fused-rocprof.json`](../benchmarks/results/2026-06-11-hipengine-mtp-p1-linear-cast-rotate-fused-rocprof.json)
- Stacked with M16.4:
  [`smoke`](../benchmarks/results/2026-06-11-hipengine-mtp-p1-stacked-split-output-cast-rotate-smoke.json),
  [`rocprof`](../benchmarks/results/2026-06-11-hipengine-mtp-p1-stacked-split-output-cast-rotate-rocprof.json)

Implementation:

- Added `paro_rotate1_f32_to_fp16`, a single-output PARO rotate kernel that first
  rounds each FP32 input element to FP16 exactly like `f32_to_fp16`, then runs the
  same FP16 rotate body. The RED gate compares raw FP16 output bits against
  `f32_to_fp16 + paro_rotate1_fp16` for rows `{1,2,4}`.
- Routed `project_linear_attention_out_fp16(..., tokens>1)` through this kernel
  by default. AR `tokens=1` keeps the old `cast -> rotate` chain; opt out with
  `HIPENGINE_LINEAR_OUT_CAST_ROTATE_FUSED=0`.

Measurement:

- Isolated exact quicksort smoke: AR `110.761 tok/s`, MTP `86.045 tok/s`
  (`0.777x`), accepted lengths unchanged. This smoke was run after rebuilding
  the touched rotate library, so the retained decision uses the profile deltas.
- Isolated post-warmup profile (`11` verifier passes): calls/pass `972 -> 942`,
  host `19.73 -> 19.45 ms/pass`, kernel `15.33 -> 15.31 ms/pass`.
  The removed `f32_to_fp16` launch was `30 calls/pass` at `0.050 ms/pass`; the
  fused rotate costs `30 calls/pass` at `0.158 ms/pass`, so kernel-side saving is
  nearly neutral and the win is mostly host launch count.
- Stacked with M16.4 split-output: exact quicksort smoke AR `112.087 tok/s`,
  MTP `87.725 tok/s` (`0.783x`), verifier `269.6 ms` over 13 cycles. Profile
  host `19.02 ms/pass`, kernel `14.86 ms/pass`, calls/pass `942`.

Decision: promoted default-on after the D32 9-prompt off/on suite held exact
with identical acceptance and a positive wall delta. This is a clean launch-count
cleanup and useful evidence for the glue bucket, but it is not a major
break-even lever. The next P1 work should target the larger capture-safe
barrier/fill bucket, remaining rotate consolidation that does not repeat
rotation per output tile, or a P2 overlap/proposer win.

### Stacked P1 9-Prompt Exactness Check (2026-06-11)

Artifacts:

- Off baseline:
  [`2026-06-11-hipengine-mtp-p1-off-9prompt-d32.json`](../benchmarks/results/2026-06-11-hipengine-mtp-p1-off-9prompt-d32.json)
- Env-on stacked check:
  [`2026-06-11-hipengine-mtp-p1-stacked-9prompt-d32.json`](../benchmarks/results/2026-06-11-hipengine-mtp-p1-stacked-9prompt-d32.json)
- No-env default-on verification:
  [`2026-06-11-hipengine-mtp-p1-defaulton-9prompt-d32.json`](../benchmarks/results/2026-06-11-hipengine-mtp-p1-defaulton-9prompt-d32.json)

Command shape: W7900/gfx1100, Qwen3.6-35B-A3B-PARO packed trunk +
MTP-BF16, D32, B=3, `persistent_device`, `chain_attn_mode=batched`, verifier
graph `auto`, draft vocab cap `32768`, with both
`HIPENGINE_W4_DUAL_OUTPUT_TILED_SPLIT_PREFILL=1` and
`HIPENGINE_LINEAR_OUT_CAST_ROTATE_FUSED=1`.

Stacked result:

- Correctness: exact same-session AR on all `9/9` prompts.
- Aggregate by-prompt mean: actual decode speedup `0.666x` AR, observed cycle
  speedup `0.672x`, visible tokens/cycle `2.023`, acceptance rate `0.355`.
- Timing: cycle wall `27.77 ms/cycle`, verify `22.31 ms/cycle`,
  proposal/update `2.14 ms/cycle`, AR decode `9.01 ms/token`.
- Best acceptance prompt: `code_python`, `0.949x` AR with `3.10` visible
  tokens/cycle.
- Worst prompt: `long_code_review`, `0.408x` AR, driven by `45.02 ms/cycle`
  wall and only `1.88` visible tokens/cycle.

Off/on A/B:

- Off baseline exact `9/9`, same visible tokens/cycle and acceptance rate.
- No-env default-on exact `9/9`, same visible tokens/cycle and acceptance rate.
- Actual speed: `0.652x -> 0.671x` AR (`+2.96%` relative).
- Cycle cost: `3.128 -> 3.049` AR-token equivalents.
- Cycle wall: `28.43 -> 27.83 ms/cycle` (`-0.60 ms/cycle`).
- Verify wall: `22.98 -> 22.37 ms/cycle` (`-0.61 ms/cycle`).
- Every prompt was non-regressive; per-prompt speed deltas ranged from `+0.4%`
  to `+3.8%` relative in the env-on A/B with identical acceptance; the no-env
  default verification also held exact `9/9`.

Decision: promote both P1 gates to default-on. They are not enough to make the
prompt suite positive, but they are exact, same-suite non-regressive, and recover
about `0.6 ms/cycle`; leaving them opt-in would throw away real break-even
progress. The locked sprint baseline remains `0.758x / 27.8 ms`; the next work
must keep this default headroom and reduce the verifier/proposer wall further.

## Thesis

Qwen3.5/Qwen3.6 MTP has the same core bottleneck that blocked the Python
DFlash harness: it can improve accepted-token density, but it cannot beat AR if
target verification remains a sequence of near-c=1 decode steps plus rollback
or accepted-prefix re-forward.

The hipEngine MTP path should therefore be a thin speculative-draft plugin on
top of the same native infrastructure planned in `DFLASH.md`:

- `DraftBatch` candidate metadata;
- `TargetVerifyBatch` / verifier-internal root row materialization;
- `KVLiveSpans` with `span_role="verify_chain"`;
- transactional KV/state scratch and commit;
- GPU target top1/accept summaries;
- graph buckets keyed by verify shape;
- exact greedy equality against MTP-disabled AR.

MTP differs from DFlash only in the **proposal provider**. DFlash uses a
separate block-draft model plus draft context KV. MTP uses target-attached MTP
weights to propose a short chain. The verifier, accept, commit, graph, and
benchmark contracts should be shared.

For the current `dflash` branch, the DFlash-built verifier is now exact enough
for MTP integration work: chain B+1 verification, GPU accept summaries,
transactional state/KV commit, batched full-attention verification, and DDTree
verification are all landed as shared infrastructure.  It is **not** yet a speed
win over serial c=1: the true-batched chain verifier is 6-8% faster than
`c1_loop` at B=2/4 but remains `2.0-5.0x` slower than serial c=1, and the first
real branching DDTree proposer beats chain/tree baselines at B=2/4/8 but still
loses to serial.  MTP speed work therefore remains blocked on the same verifier
row-cost wall.

What is now landed for MTP (2026-05-19):

- provider-neutral chain metadata in `hipengine.speculative.chain`:
  `ChainDraftRequest`, `ChainDraftCompiler`, and `compile_chain_draft()`;
- `hipengine.speculative.mtp` with `MtpProposalContext`, `MtpDraftProvider`,
  `Qwen35MtpDraftProvider`, `MtpChainCompiler`, and `compile_mtp_chain()`;
- target-attached Qwen3.5/Qwen3.6 `mtp.*` metadata/loading in
  `hipengine.loading.mtp`, including validation and `load_qwen35_mtp_bf16_weights()`;
- `scripts/mtp_chain_e2e_bench.py`, a readiness diagnostic that records the MTP
  chain ABI and refuses to fake a speed row when tensors are missing.

The original shisa packed PARO target snapshot
`501ef8635e5cfb5a7497d232358ca8d1afc0c66e` contains `0/19` expected `mtp.*`
tensors; the retained artifact
`benchmarks/results/2026-05-19-hipengine-mtp-chain-readiness-missing-tensors-diagnostic.json`
records that initial `blocked_missing_mtp_tensors` state.

A local bring-up artifact now exists at
`/models/hipengine/Qwen3.6-35B-A3B-PARO-full4096-e5-packed-MTP-BF16`.  It reuses
(symlinks) the packed PARO trunk and adds `mtp-bf16.safetensors` generated from
`Qwen/Qwen3.6-35B-A3B-FP8` `mtp.safetensors`: BF16 tensors are copied as-is,
FP8 block-128 projection/expert tensors are dequantized to BF16, and per-expert
`gate_proj`/`up_proj` are fused into the hipEngine runtime layout
`mtp.layers.0.mlp.experts.gate_up_proj`.  The retained assembly diagnostic is
`benchmarks/results/2026-05-19-hipengine-qwen36-paro-mtp-bf16-assembly-diagnostic.json`.
Validation now sees all `19/19` required tensors.  Native MTP proposal bring-up
has started: `hipengine_mtp_fuse_inputs_f16_bf16` covers token-embedding +
target-hidden pre-fc RMSNorm/concat, and `scripts/mtp_input_fc_smoke.py` applies
`mtp.fc` with the existing BF16 dense GEMV.  A full one-layer torch-reference
proposal smoke also exists in `scripts/mtp_torch_proposal_smoke.py`; it runs
MTP attention, MoE/shared expert, final `mtp.norm`, shared lm-head/top1, and
emits candidate-only `DraftBatch` rows for `verify_chain_bulk_and_commit`.

The native proposal-chain smoke is now `scripts/mtp_native_decode_step_smoke.py`.
It reuses native BF16 dense/QKV/GQA/lm-head kernels plus MTP-specific helper
kernels for zero-centered RMSNorm, q/gate split, BF16 gate multiply, router
softmax, MoE accumulation, and FP32-to-BF16 finalization.  On gfx1151 with the
assembled artifact, `--draft-budget 2 --torch-compare` produced candidate chain
`[12, 4773]` in both native and torch-reference paths and emitted verifier rows
`[root, d1, d2]` with parent rows `[-1, 0, 1]`.  The same smoke can now consume a
BF16 target hidden row captured by `Qwen35ParoResidentSession.step_with_hidden_taps`
using `--target-hidden-source target_session`; that path produced native/torch
candidate chain `[27399, 220]`.  This remains a diagnostic smoke, not a speed row:
selected expert ids are host-orchestrated.  The first shared-verifier E2E smoke
is `scripts/mtp_chain_e2e_smoke.py`; it feeds native MTP candidate rows into
`Qwen35ParoResidentSession.verify_chain_bulk_and_commit` and matched exact AR on
a 3-token gfx1151 smoke.  A persistent native provider now lives in
`hipengine/speculative/mtp_native.py`; it keeps MTP weights/cache resident and
uses device-resident target hidden rows.  On the stable quicksort prompt with
B=5 it matched exact AR and accepted all proposed draft tokens for the 8-token
sample (`accepted_lengths=[5,1]`), but still lost the decode speed gate
(`31.34 tok/s` vs AR `52.98 tok/s`) because the current bulk verifier is slower
than serial AR and expert dispatch still copies selected expert ids to host.  A
follow-up single-chain linear-attention verifier t-loop now avoids parent-row
global state reloads while still materializing row states for exact partial
accepts; the same B=5 sample stayed exact with `linear_attn_mode=chain_tloop` and
measured `33.13 tok/s` vs AR `52.63 tok/s`, so the speed blocker moved from
linear-state materialization specifically to the broader target-verifier
launch/row-cost wall.

A subsequent verifier graph-capture experiment fixed two blockers:
1. `_verify_capture_staging_tensor` rejected `width <= 0`, breaking graph paths
   where hidden-tap capture is empty (`capture_layer_ids=()`).
2. `_should_use_chain_tloop_linear_verify` disabled `chain_tloop` whenever
   `graph_mode != "off"`, forcing the slower `tree_tloop` fallback during capture.

After those fixes, B=2 and B=3 graph capture with `chain_tloop` validates
exactly against the direct path.  Measured on the stable quicksort prompt with
32 decode tokens and `graph_mode=auto`:

| mode | AR decode tok/s | MTP decode tok/s | verify sec/14 cycles | avg ms/cycle |
|---|---|---|---|---|
| graph=off | 61.5 | 28.6 | 0.806 | 57.6 |
| graph=auto | 61.2 | 26.1 | 0.912 | 65.1 |

Graph replay is **not faster**; it is slightly slower.  The verifier per-cycle
execution time (~57-65 ms for B=2) is dominated by GPU kernel work, not CPU
launch overhead.  Capture/validation overhead for new buckets (`rows=3`,
`rows=2`) negates any replay savings when buckets change frequently.  Even with
13 replays and 1 capture over 14 cycles, the total verify time is higher than
the direct path.  This means graph capture alone will not close the MTP/AR gap.
The remaining speed work must attack kernel execution cost (fused verifier layers,
batched row processing) rather than launch overhead.

### GPU-fast token accept

`verify_chain_bulk_and_commit` and `verify_tree_bulk_and_commit` historically
read target top1 tokens back to CPU (`_read_verify_top1`), ran the CPU oracle
`batch.accept_from_top1()`, and then compared with a GPU-side accept-summary
kernel (`dflash_accept_chain_i32`) for validation.  The GPU kernel already
computes the full acceptance result (commit row, commit token, next token,
full-accept flag); the CPU path is redundant overhead.

A GPU-fast accept path was added:
- `TargetAcceptSummary.from_gpu_payload(batch, gpu_payload)` reconstructs
  accepted tokens by walking `batch.parent_rows` from the commit row back to
  the root, producing a semantically identical result to the CPU oracle.
- `_verify_gpu_accept_enabled()` checks `HIPENGINE_VERIFY_GPU_ACCEPT`.
  - `0` / `false` / `no` / `off` → disabled (default).
  - `1` / `true` / `yes` / `on` → fast path: skip CPU top1 read and CPU
    accept computation; trust the GPU accept-summary kernel directly.
  - `validate` → run both paths and compare; fall back to CPU on mismatch.
- When enabled, `verify_chain_bulk_and_commit` reads only the small GPU accept
  payload (`~28 bytes/request`) instead of the larger top1 buffer
  (`rows * 8 bytes`), and avoids the CPU tree-walk entirely.

Validation on the stable quicksort prompt with B=2, 32 decode tokens:
- `HIPENGINE_VERIFY_GPU_ACCEPT=validate` → exact AR match, all cycles
  `gpu_accept_match_cpu=True`.
- Speed impact is **neutral to noise**: the CPU top1 read + accept walk is
  `<0.1 ms/cycle`, negligible next to the `~60 ms/cycle` verifier forward pass.
  The fast path is architecturally cleaner but does not measurably improve
  throughput.  The dominant verifier cost remains the forward-pass kernel
  execution, not host-side acceptance logic.

## Alignment with existing hipEngine design

hipEngine already has the right abstract contracts:

| Existing design piece | MTP use |
| --- | --- |
| `hipengine.speculative.DraftBatch` | Carries candidate draft rows, depths, request ids, and optional tree parents. MTP emits chain candidates; it does **not** include the already-committed root row. |
| `hipengine.speculative.DraftModel` | MTP head/plugin implements `propose(...)`. It is model-attached, not an external draft model. |
| `hipengine.speculative.Verifier` | Shared target verifier verifies a `DraftBatch` by internally materializing `[root, candidates...]`. |
| `hipengine.speculative.AcceptResult` | Stores accepted counts/tokens plus optional transaction, selected-row, and target next-token provenance per request; should be extended later with compact device-summary provenance. |
| `KVLiveSpans(span_role="verify_chain")` | Full-attention verify rows write into scratch/journal spans, not canonical KV. |
| `KVPolicy.begin_transaction/commit/rollback` | Accepted prefix commit and rejected suffix discard. |
| Graph bucket key `(mode, draft/tree shape, C, context bucket, ...)` | MTP fixed-depth buckets such as B=1/2/3/5. |

Design clarification shared with `DFLASH.md`: `DraftBatch.candidate_tokens` are
candidate rows only. The target verifier builds a runtime `VerifyBatch` with a
root row at slot 0 and candidate rows at slots `1..N-1`. Docs and tests should
not encode the root token as a `DraftBatch` candidate.

## Prior W7900/Quark evidence from `~/amd-gpu-tuning`

Source plan: `~/amd-gpu-tuning/PLAN-MTP.md`.

The rows below are prior W7900/gfx1100 + Quark/W8A8/BF16-MTP evidence. They are
useful for verifier break-even math, but they are not a baseline for the current
packed `gfx1151` DFlash lane. gfx1151 has a higher compute-per-byte balance than
W7900 (roughly 48% of W7900 compute but ~30% of its memory bandwidth), so bytes
are more expensive and native row reuse may matter even more. Do not promote an
MTP speed claim until it is re-measured on the packed target with the shared
native verifier.

Retained native MTP rows show useful correctness and acceptance, but not speed:

| Evidence | Result |
| --- | --- |
| Best B=5 native-loop / target-graph-replay row | exact same-session equality, MTP `83.88 tok/s` vs AR `120.04 tok/s` = `0.699x` |
| Acceptance | accepted `20/29` drafts, average committed output `2.91` tokens/iteration |
| Scalar syncs | still ~60 scalar D2H reads in retained row |
| Longer B=5/B=6 windows | around `82 tok/s` vs `~120 tok/s`; accepted depth rose but speed did not |
| True bulk torch verifier | argmax-correct but far too slow; torch grouped MoE dominated profile |
| Profile orientation | W8A16 linear/MoE/lm-head family ~51.5% kernel time; full attention ~1.9% |

Budget reminder from the parent plan:

```text
T_iter <= A_out * T_AR / target_speedup
```

At AR `~120 tok/s`, `T_AR ~= 8.33 ms`. With `A_out ~= 2.91`, a `1.10x`
speedup needs the full MTP iteration under `~22 ms`. Current correct rows are
closer to `~35 ms/iteration`. The missing piece is not another MTP policy
sweep; it is a faster exact target verifier and cheap state commit.

## MTP model boundary

The Qwen3.5/Qwen3.6 MTP module is target-attached:

```text
input:  RMSNorm(token_embedding) + RMSNorm(target_hidden)
        -> concat
        -> mtp.fc
        -> one decoder layer (full attention + MoE, same family as target)
        -> mtp.norm
        -> shared target lm_head
output: next-token logits / top1
```

Local bring-up artifact from the parent workspace:

```text
/models/qwen36-quant/Qwen3.6-35B-A3B-Quark-W8A8-INT8-MTP-BF16
```

That artifact hardlinks the Quark W8A8 target shards and adds BF16 `mtp.*`
tensors. It is a bring-up artifact, not the current DFlash target layout. The
`dflash` branch should not hardcode it into benchmark paths; the target baseline
is the shisa packed PARO model, and the DFlash drafter is
`z-lab/Qwen3.6-35B-A3B-DFlash`. MTP metadata/loading should be revisited after
the shared verifier exists and should validate whatever packed-target-attached
MTP artifact we actually retain.

Important external facts:

- vLLM's `Qwen3_5MultiTokenPredictor` defines the same boundary:
  normalized embedding + normalized target hidden -> `mtp.fc` -> one decoder
  layer -> final norm -> shared lm-head.
- vLLM warns that using one MTP layer for `num_speculative_tokens > 1` repeats
  the same predictor and can reduce acceptance. hipEngine should measure small
  depths first rather than assuming deeper is better.
- llama.cpp-style speculative decode verifies `[last, draft0, ...]` in one
  target batch and crops/restores to the committed prefix. hipEngine should do
  the same logically, but with transactional scratch/commit instead of a
  measured accepted-prefix re-forward.

## Shared infrastructure with DFlash

MTP should not create a parallel verifier stack. The following dependencies are
shared and should land through `DFLASH.md` first:

1. **Native `TargetVerifyBatch`.**
   A fixed-shape batch of root + candidate rows with token ids, positions,
   parent/depth metadata, output hidden/final rows, logits/top1 buffers, and
   per-layer state scratch.

2. **Exact selectable target state.**
   For every verify row the target forward must expose enough state to commit
   that row without target re-forward:
   - full-attention K/V rows;
   - linear-attention Conv/GDN state;
   - hidden taps / final hidden needed by the next proposal step;
   - target top1/logit summary.

3. **Transactional KV/state commit.**
   Rejected rows never touch canonical KV/state. Accepted rows are copied or
   journal-committed through the scheduler-owned transaction.

4. **GPU target top1 and accept summary.**
   MTP must not copy full logits to host per row. It needs a compact summary:
   accepted depth, committed token ids, first correction/bonus token, and graph
   validation/status flags.

5. **Graph bucket discipline.**
   Fixed MTP depths are graph shapes. Initial buckets should be small:
   `B={1,2,3}` and only then `B=5` if the measured split supports it.

6. **Measurement schema.**
   Same fields as DFlash: exact equality, finite logits, AR tok/s, spec tok/s,
   target verify rows/time, draft/proposal time, commit/replay time, rows/output,
   accepted-depth histogram, scalar D2H count, graph status, and peak memory.

## MTP runtime flow

For one request, with chain draft depth `B`:

```text
1. AR or previous commit has produced the current root token and target hidden.
2. MTP propose step uses (root embedding, target hidden, MTP KV/state if any)
   to emit draft token d1.
3. For depth > 1, either repeat the MTP predictor using the newly proposed token
   and MTP state, or use an explicitly supported multi-depth MTP path. Measure;
   do not assume repetition is profitable.
4. MTP plugin returns DraftBatch candidates [d1, d2, ... dB] with depths 1..B.
5. Shared verifier internally builds [root, d1, ... dB] and runs target verify.
6. Device accept compares target top1 at row i to candidate d{i+1}.
7. Commit root + accepted draft prefix + correction/bonus, selecting target
   state from the verified accepted row.
8. Expose the committed target hidden/final row for the next MTP proposal.
```

For c>1, the same flow is row-mapped by `request_id` and physical batch slot;
MTP-specific code must not assume one global scalar root or one global scalar
position.

## Phased implementation plan

### Phase M0 — Shared verifier baseline

The shared verifier/accept/commit baseline has landed for correctness and ABI
reuse, but not for speed promotion:

- native topk=1 chain `TargetVerifyBatch` exists;
- exact selectable per-row target state exists;
- accepted-prefix target re-forward is avoided in native commit paths;
- GPU target top1 + chain accept summary exists;
- batched chain and tree verifier variants exist.

The unresolved M0 speed gate remains: the verifier is still slower than serial
c=1 on chain-shaped work.  MTP integration may proceed as metadata/proposal
bring-up, but retained MTP speed claims must wait until either verifier row cost
falls or MTP acceptance density clearly overcomes it.

### Phase M1 — Model metadata and native MTP proposal oracle

Goal: prove hipEngine can load and execute the MTP proposal module outside the
main generation loop.

- Add model-plugin metadata for target-attached MTP tensors.
- Extend loading/materialization for `mtp.*` tensors from the bring-up artifact.
- Implement or port MTP proposal kernels in parent-compatible order:
  - embedding/root input preparation;
  - hidden/embedding RMSNorm;
  - `mtp.fc`;
  - one MTP decoder layer;
  - `mtp.norm`;
  - shared lm-head + GPU top1/topk.
- Compare native MTP top1/topk against the parent Python/native-loop harness on
  fixed hidden/token fixtures.
- No speed claim yet; this is a proposal correctness gate.

### Phase M2 — DraftModel plugin and DraftBatch chain output

Goal: wire MTP as a `DraftModel` provider without touching target commit logic.

- Implement `MtpDraftProvider.propose(...) -> DraftBatch`.
- Emit candidate tokens only; do not include the root token in `DraftBatch`.
- Fill `request_ids`, `candidate_tokens`, `parent_positions`, `draft_depths`,
  `row_to_request`, `active_mask`, and `mode="verify_chain"`.
- Add CPU fixtures for forced accept patterns at B=1/2/3/5.
- Add telemetry: proposal depth, draft time, proposal top1/top2 margin where
  available, and MTP KV/state bytes.

### Phase M3 — Shared verifier integration

Goal: make MTP consume the same exact verifier as DFlash.

- Feed the MTP `DraftBatch` into the shared chain verifier.
- Use verifier-internal root row + candidates for target logits/state.
- Use the shared chain accept kernel or a small MTP-specialized variant if the
  summary layout differs.
- Commit selected target state without target re-forward.
- Ensure the accepted row exposes target hidden/final hidden for the next MTP
  proposal.
- Correctness gate: exact greedy equality vs MTP-disabled AR on fixed prompts,
  including reject-at-depth-0, partial accept, and full accept cases.

### Phase M4 — Fixed-depth graph buckets and small-depth sweep

Goal: find the useful MTP depth after verifier economics improve.

Initial buckets:

| Bucket | Purpose |
| --- | --- |
| B=1 | baseline one MTP proposal; should be cheap and exact |
| B=2 | likely first real speed candidate if verifier row cost is low |
| B=3 | check acceptance-depth tradeoff |
| B=5 | compare to parent retained B5 evidence only after B1-B3 are understood |

Report for every bucket:

- same-session AR tok/s;
- MTP tok/s and vs AR;
- average committed output tokens/iteration;
- accepted-depth histogram;
- target verify rows/output and verify eta;
- MTP proposal time;
- accept/commit/host overhead;
- scalar/vector D2H count;
- peak memory and MTP KV/state bytes;
- graph direct vs replay validation.

### Phase M5 — MTP cache/state and long-context gate

Goal: avoid short-context wins that collapse with prompt length.

- Keep MTP KV/state append-only where possible; do not rebuild prompt-side MTP
  cache in the measured loop.
- Measure prompt lengths `512`, `4096`, `16384`, `32768`, and `65536` where
  memory permits after a short-context row wins.
- Classify regressions as verifier cost, MTP prompt-cache rebuild, MTP KV memory
  growth, acceptance/position shift, or target attention/KV cost.

### Phase M6 — Policy, top-k rescue, and tree variants

Only after M3/M4 show the verifier is cheap and proposal cost is not dominant:

- root top-k / margin guard;
- adaptive depth cap;
- target-guided top-k oracle for upper bounds;
- MTP tree/branching experiments using `mode="verify_tree"` and the DDTree
  compiler/accept infrastructure from `DFLASH.md`.

Branching must remain parent-linked and prefix-closed: no accepted path may use a
state row whose ancestors were not accepted.

## Correctness gates

MTP has stricter continuation requirements than a pure draft-only provider:

- exact greedy output equality vs MTP-disabled AR;
- target verifier rows finite;
- MTP proposal logits finite;
- accepted prefix state equals serial target state at the first token after the
  commit;
- target hidden/final hidden selected for the accepted row matches serial c=1;
- rejected suffix does not leak into target KV, linear-attention state, MTP KV,
  or output ring;
- disabling MTP produces byte-identical deterministic output to the normal
  target path on fixtures.

Layer-ladder debug is required before promotion: on a mismatch, compare serial
c=1 vs bulk verify row at each layer boundary and report the first failing
layer/family.

## Promotion gates

A retained hipEngine MTP row must satisfy:

| Gate | Requirement |
| --- | --- |
| Correctness | exact same-session AR equality, finite target/MTP logits, no state leak |
| Speed | >1.10x same-session AR on a short retained prompt before policy work |
| Verifier | target verify eta low enough to explain the speedup; no accepted-prefix re-forward |
| Accounting | rows/output, accepted histogram, proposal/verify/commit split, D2H count |
| Memory | MTP weights + target + target KV + MTP state/KV under the active gate |
| Artifacts | compact JSON under `benchmarks/results/`, rollup/changelog updated per `docs/BENCHMARK.md` |

If MTP does not beat AR after the shared verifier is fast, run a fresh split
before optimizing: proposal cost may dominate, or acceptance may be insufficient
for the measured prompt class.

## M12 — true batched verifier loop pivot (2026-05-21)

M7.C.6 answered the pushback question: llama.cpp can get benefits from MTP-2
and MTP-3 because its target-verifier cycle is cheap in **AR-token-equivalent**
units. hipEngine's current verifier is exact but too expensive per cycle. The
new primary metric is therefore not an isolated kernel family millisecond; it is

```text
cycle_cost_ar_tokens = avg_mtp_verify_cycle_wall_ms / ar_decode_ms_per_token
```

A fixed-depth chain with candidate budget `B` can beat AR only when

```text
avg_visible_tokens_per_verify_cycle > cycle_cost_ar_tokens
```

where `avg_visible_tokens_per_verify_cycle = 1 + avg_accepted_drafts`. A 1.5×
row requires `avg_visible_tokens_per_verify_cycle / cycle_cost_ar_tokens >= 1.5`.
llama.cpp's MTP-2/3 wins imply a cycle cost around **~2 AR-token equivalents**;
our first M12 sweep is far above that.

### M12.0 economics baseline (single-run diagnostic)

Artifact: `benchmarks/results/2026-05-21-hipengine-mtp-verifier-economics-m12.json`.
Command:

```bash
python3 scripts/mtp_verifier_economics.py \
  --prompt-tokens-file /tmp/quicksort-prompt-tokens.txt \
  --decode-tokens 32 \
  --candidate-budgets 1,2,3,5 \
  --runs 1 \
  --raw-root /tmp/hipengine-mtp-economics-m12-b1-b5 \
  --out benchmarks/results/2026-05-21-hipengine-mtp-verifier-economics-m12.json
```

`performance_claim=false`; this is a planning diagnostic. Exact-AR-match passed
for every row.

| B | AR tok/s | MTP tok/s | MTP/AR | avg visible tokens/cycle | cycle wall ms | cycle cost (AR tok) | verify ms/cycle | perfect-accept ceiling | required accept rate for 1× |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64.4 | 31.5 | 0.489 | 1.55 | 49.6 | **3.20** | 38.0 | 0.63× | 220% |
| 2 | 63.9 | 28.8 | 0.451 | 2.21 | 77.6 | **4.96** | 57.5 | 0.61× | 198% |
| 3 | 50.7 | 28.2 | 0.557 | 2.38 | 85.5 | **4.33** | 62.1 | 0.92× | 111% |
| 5 | 51.8 | 21.6 | 0.416 | 2.82 | 132.6 | **6.87** | 96.8 | 0.87× | 124% |

Interpretation:

- B=1/B=2 are structurally impossible to beat AR with the current loop: even
  impossible acceptance rates would be needed for 1×.
- B=3 is close only in the **perfect acceptance** ceiling (4 visible tokens /
  4.33 AR-token cost = 0.92×). With observed acceptance it is 0.56×.
- B=5 currently gets more accepted tokens, but the cycle cost scales upward
  faster than useful output (6.87 AR-token cost for 2.82 visible tokens/cycle).
- Therefore the next target is not “try B=7” or “shave 1 ms from LM head”; it is
  to make the verifier cycle cost **sublinear in rows**. The M12 success gate is
  `cycle_cost_ar_tokens <= 2.5` as the first milestone and `<= 2.0` for
  llama.cpp parity.

### M12 design contract — make verifier cycles llama.cpp-shaped

M12 is a verifier-loop rebuild, not another isolated kernel-family cleanup. The
contract is: one speculative cycle should look like a small batched target decode
with GPU-resident control metadata, not `B+1` decode rows glued together by host
bookkeeping.

Definitions used by every M12 decision:

```text
B                         = candidate draft budget
rows                      = B + 1 verifier rows (root + candidates)
C_B                       = cycle_cost_ar_tokens for budget B
A_B                       = avg accepted draft tokens/cycle
E_B                       = avg visible tokens/cycle = 1 + A_B
observed_speedup_vs_ar    = E_B / C_B
perfect_accept_ceiling    = (B + 1) / C_B
break_even_gate           = E_B > C_B
1.5x_gate                 = E_B / C_B >= 1.5
```

**Target ratios:**

| Milestone | Required verifier shape | Promotion meaning |
|---|---|---|
| M12-alpha | `C_3 <= 3.0` and `C_5 <= 4.0`; B-scaling slope from B=3→5 <= 0.5 AR-token/extra row | Current loop is no longer structurally impossible at B=3/B=5 under high acceptance. |
| M12-beta | `C_3 <= 2.5`, `C_5 <= 3.5`, and first exact B=7 economics row | Small-B verifier cost is close enough that adaptive B and acceptance quality can determine wins. |
| llama.cpp parity | `C_3 <= 2.0` and `C_5 <= 2.5`; any B=7 row must have `C_7 <= 4.0` and sublinear scaling | Verifier cycle cost is in the same regime as llama.cpp MTP-2/3. |
| speed-row promotion | `E_B / C_B > 1.0` over >=3 runs with exact AR equality; 1.5× row needs `E_B / C_B >= 1.5` | May be promoted to a benchmark rollup speed claim. |

#### Explicit go/no-go math for B=3/B=5/B=7

Use this table before spending implementation time on a budget-specific path.
`C target @ observed E` is the cycle-cost ceiling if acceptance does not improve;
`C target @ perfect 1.5×` is the absolute ceiling for a 1.5× row even with perfect
acceptance. B=7 is not yet measured; it must be added by M12.1/M12.2 before any
B=7 optimization is retained.

| Budget | Current measured `E_B` | Current `C_B` | Current `E_B/C_B` | `C_B` target for 1× at observed `E_B` | `C_B` target for 1.5× at observed `E_B` | `C_B` target for 1.5× at perfect accept | Go / no-go rule |
|---:|---:|---:|---:|---:|---:|---:|---|
| B=3 | 2.38 | 4.33 | 0.55× | <= 2.38 | <= 1.59 | <= 2.67 | **Go** only after M12.2/M12.3 can drive `C_3 <= 2.5` or acceptance improves to `A_3 >= C_3 - 1`. **No-go** for standalone B=3 speed claims while `C_3 > 4.0` because even perfect acceptance cannot beat AR. |
| B=5 | 2.82 | 6.87 | 0.41× | <= 2.82 | <= 1.88 | <= 4.00 | **Go** if reshaped verifier shows `C_5 <= 4.0` (perfect 1.5× becomes possible) and observed `E_5/C_5` is trending upward. **No-go** if `C_5 > 6.0` after M12.2 because even perfect acceptance cannot beat AR. |
| B=7 | TBD | TBD | TBD | `<= E_7` | `<= E_7/1.5` | <= 5.33 | **Measure first.** For `C_7=2.5`, 1.5× needs `A_7 >= 2.75` (39% draft acceptance). For `C_7=4.0`, 1.5× needs `A_7 >= 5.0` (71%). **No-go** if first exact B=7 row has `C_7 > 8.0` (perfect cannot beat AR) or if B=5→7 scaling remains >0.5 AR-token per extra row. |

The B=7 row is deliberately conditional. It is the likely budget that can win if
acceptance is decent, but trying it before M12 lowers the cycle-cost slope just
repeats the B=5 failure mode: more accepted tokens with even more verifier cost.

#### Required measurements for every M12 implementation step

Every M12 subtask must produce or update an economics artifact with these fields:

1. `C_B`, `E_B`, `E_B/C_B`, perfect-accept ceiling, and required acceptance for
   1×/1.5× for B=3 and B=5; B=7 once M12.1 supports the row count reliably.
2. Per-cycle timeline split that reconciles with `cycle_marker_ns`:
   draft build, metadata writes, target forward, LM-head/top1, accept read/CPU
   oracle, linear-state commit, proposer repair, final stream sync.
3. GPU-event sub-splits for target forward once available: full-attn layers,
   linear-attn chain-tloop layers, target MoE/shared expert, LM head/top1.
4. Kernel-family rollup from `scripts/mtp_verifier_rocprof.py` for any retained
   kernel/layout change, tied to the same B/prompt/decode workload.
5. Acceptance provenance: accepted lengths, active budgets, target top1 rows (or
   fused top1 payload), correction/bonus token, and exact AR output tokens.

Minimum command set for retained diagnostics:

```bash
python3 scripts/mtp_verifier_economics.py \
  --prompt-tokens-file /tmp/quicksort-prompt-tokens.txt \
  --decode-tokens 32 \
  --candidate-budgets 3,5 \
  --runs 3 \
  --out benchmarks/results/<date>-hipengine-mtp-m12-<subtask>-economics.json

# Add B=7 once the specific subtask claims B=7 support:
python3 scripts/mtp_verifier_economics.py \
  --prompt-tokens-file /tmp/quicksort-prompt-tokens.txt \
  --decode-tokens 32 \
  --candidate-budgets 7 \
  --runs 3 \
  --out benchmarks/results/<date>-hipengine-mtp-m12-<subtask>-b7-economics.json
```

A performance row is not promoted unless exact AR equality passes on every run
and `E_B/C_B > 1.0`. Single-run rows remain `performance_claim=false` planning
diagnostics.

#### Architectural changes required by M12

1. **Verifier control plane:** keep chain metadata, candidate tokens, parent rows,
   active masks, top1/correction payload, and commit selection GPU-resident across
   a cycle. Host can launch the cycle and read one compact result, but it should
   not build/repair per-row decisions with synchronous D2H top1 reads.
2. **Small-B full-attention verifier primitive:** replace the default
   `_run_full_attention_chain_c1_loop` row loop with a verifier-specific small-B
   path. The existing `_run_full_attention_chain_batched` proves the ABI, but its
   prefill-style kernels have too much fixed overhead. M12.2 needs decode-shaped
   row batching: Q/K/V projection, K/V append, attention, output projection, MoE,
   and residual for all rows in one layer pass.
3. **Verifier LM-head + accept fusion:** replace full `rows × vocab` logits
   materialization with a streaming W8A16 row top1 / candidate-check kernel. The
   kernel should output the exact top1 token per verifier row (debug/validation),
   accepted length, correction token, bonus token, and matched-mask payload.
4. **Layer-level target MoE primitive:** turn the current row-batched but
   launch-fragmented `run_moe_c1_fp16(tokens=rows)` chain into a verifier-layer
   primitive with an ids-tensor ABI. This is the M12 version of the old M7
   selected-expert work.
5. **Proposer handoff cleanup:** after target forward is no longer dominant,
   remove serial c=1 proposer repair from the critical path: GPU top1 for draft
   tokens, snapshot ring/copy elision where possible, and optional graph replay
   for the draft MTP block.
6. **Graph replay only after reshaping:** graph capture/replay is an amplifier,
   not the fix. Re-enable it per `(B+1, kv_bucket)` only after M12.2–M12.4 reduce
   kernel time and launch count enough that graph replay improves `C_B`.

#### Correctness gates

- **Primary gate:** exact AR token equality for the full decode, same prompt and
  same decode length, with accepted-token provenance recorded.
  - *Strict Exact-AR vs Tolerant Validation:* If the verifier uses slightly different math (e.g. direct BF16 RMSNorm instead of FP16+Cast), a 1-ULP float difference can flip the top-1 argmax between two closely-scored tokens. If the verifier disagrees with the pure AR model on the top-1 choice, it will either reject a draft token that pure AR would have picked, or accept a draft token pure AR would have rejected. **This is a true output token change.** Because greedy MTP commits whatever the verifier top-1 demands, relaxing math equality directly changes the generated text. Our `exact_ar_match` gate strictly forbids this to guarantee bit-for-bit identical outputs to the baseline.
- **Behavior-preserving rewrites:** accepted lengths and top1 rows must match the
  previous implementation at the same B unless the subtask intentionally changes
  policy (for example adaptive B). If policy changes, output tokens must still
  match AR and the artifact must explain accepted-length differences.
- **Kernel gates:** any new HIP verifier kernel must pass fixture tests against
  the CPU reference / existing runtime path, then an MTP exact-AR smoke, then a
  `rocprofv3 --kernel-trace` smoke showing the expected kernel names.
- **Numerical gate:** final verifier top1 rows must be identical to the baseline
  row-wise argmax for fixture prompts. KL/top-1 aggregate gates are acceptable for
  intermediate math tests, but a retained verifier-loop change must preserve final
  greedy top1 provenance exactly on the smoke fixture.
- **Fallback gate:** every risky M12 path ships behind an env flag or existing
  mode switch until it has a retained economics artifact. The legacy c1 loop / LM
  head path must remain available for bisection.

### Current loop map — where the cycle is not llama.cpp-shaped

Code audit (2026-05-21, task #2 / M12 map): the local model has 40 target
layers: 30 `linear_attention` layers and 10 `full_attention` layers (`config.json`
`text_config.layer_types`, full attention every 4th layer). In chain mode the
verifier rows are `[root, d1, d2, ...]` with parent rows `[-1, 0, 1, ...]`.
Default benchmark mode is `chain_attn_mode=c1_loop`, `graph_mode=off`,
`HIPENGINE_VERIFY_CHAIN_LINEAR_TLOOP=on`, and GPU accept disabled unless
`HIPENGINE_VERIFY_GPU_ACCEPT` is set.

#### Host / proposer side

| Stage | Code | B handling today | Serial work / sync |
|---|---|---|---|
| Prompt handoff | `scripts/mtp_chain_e2e_smoke.py:349-369`, `NativeMtpChainProposer.prefill_from_target_hidden_rows` | Prompt hidden taps are captured from the target into one BF16 buffer, but proposer prefill advances one prompt token at a time. | `prefill_from_target_hidden_rows` loops over prompt tokens and calls `advance` per row. Outside steady-state cycle, but it proves the proposer is not a graph/batch API yet. |
| Draft construction | `scripts/mtp_chain_e2e_smoke.py:404-412` | Candidate list is built on the host. | One `save_state(0)`, then `for draft_idx in range(1, active_budget): proposer.advance_with_previous_hidden(...); save_state(draft_idx)`. This is serial in draft depth. Result-producing advances still read argmax/logit to the host; discarded expert-topk metadata is skipped by default. |
| Target batch metadata | `_target_batch` / `TargetVerifyBatch.from_draft` (`scripts/...:150-159`, `interfaces.py:159-196`) | The metadata object is row-batched: one root row plus B candidate rows, parent chain encoded as row indices. | Pure host construction. No per-row target forward here, but row topology is fixed before kernels run. |
| Proposer repair after accept | `scripts/mtp_chain_e2e_smoke.py:448-456` | Uses the accepted count to restore a saved proposer snapshot, optionally advance through the last accepted draft, then advance once on the bonus/correction token. | Serial: `restore_state` does D2D memcpy, and update-only accepted-token advances are still c=1 MTP blocks. Default-on proposer skip removes discarded lm-head/argmax and expert-topk host reads for those update-only advances, but the chain is not batched/device-looped yet. This contributes to cycle wall but not `verify_seconds`. |

#### Target verifier entry / metadata / commit

| Stage | Code | B handling today | Serial work / sync |
|---|---|---|---|
| Metadata copies | `verify_chain_bulk_and_commit` → `_write_verify_chain_metadata` (`qwen35_paro_runner.py:2394`, `3498-3569`) | True row metadata: tokens, positions, parent rows, depths, active mask, context counts, and a tiled block table are copied as `[rows]` / `[rows, blocks]` arrays. | Several small host→device copies per cycle. Not the main cost, but it prevents the whole cycle from being purely GPU-resident. |
| Target forward launch | `_launch_verify_chain_forward_accept` (`qwen35_paro_runner.py:2833-2937`) | Embedding lookup is truly batched over `rows`; then the code loops through all 40 layers once. Layer loop is required by model topology. | The question is inside each layer: linear layers are partly row-batched; full-attention layers are row-serial by default. |
| Accept payload read | `_launch_verify_accept_summary` + `_read_verify_accept_payload` (`3571-3684`) | GPU accept summary kernel is row-aware and request-aware. | `_read_verify_accept_payload` always synchronizes and copies 7 tiny fields D2H. If `HIPENGINE_VERIFY_GPU_ACCEPT` is unset, `verify_chain_bulk_and_commit` also reads all row top-1 values and runs the CPU accept oracle (`2434-2458`). GPU-fast accept removes the CPU oracle/top1 read but not the forward cost. |
| State commit | `_commit_bulk_linear_states`, `_set_slot_position`, final sync (`2467-2469`, `3862-3880`, `4631-4639`) | Commits the selected row's linear states into the resident slot; full-attention K/V rows are left in cache and the slot position selects the accepted prefix. | Host loop over 30 linear-attention layers, with 2 D2D copies per layer (conv + recurrent state), then one position kernel and final stream sync. This is small compared with target forward but is explicitly serial by layer. |

#### Target forward: true batched vs serial by component

| Component | Code | B handling today | Verdict |
|---|---|---|---|
| Row embedding | `embedding_lookup_batch_fp16_i64` (`qwen35_paro_runner.py:2833-2844`) | One launch over `rows`. | **True batched**. |
| Linear-attention layer wrapper | `run_linear_attention_moe_chain_tloop_layer_fp16` (`qwen35_paro.py:4453-4588`) | Called once per linear layer with `tokens=rows`. | **Partly batched**: no host row loop; many small per-layer kernels. |
| Linear Conv/GDN recurrence | `qwen35_linear_attn_chain_conv_decode_fp16_tloop`, `qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_tloop_fp16` (`4515-4549`) | One Conv t-loop launch and one GDN t-loop launch per linear layer. The chain dependency is carried inside the kernel using `chain_conv_state` / `chain_recurrent_state`. | **Correct batched-chain shape** for the recurrence. It is still serial in depth logically, but not a host/per-row launch loop. This is the best-shaped part of the current verifier. |
| Linear-attention projections / output / norms | `input_rmsnorm_fp16`, `rotate_linear_attention_inputs_fp16`, `project_linear_attention_qkv_z_fp16`, `project_linear_attention_ab_fp16`, `project_linear_attention_out_fp16`, `post_attention_add_rmsnorm_fp16` (`4489-4566`) | One call per primitive with `tokens=rows`. M7.C.6 fixed the QKV/Z small-batch dispatch alias by splitting dual GEMV into two single GEMVs for `rows <= threshold`. | **Row-batched but launch-fragmented**. Good enough for correctness; not a llama.cpp-like fused graph. |
| Target MoE selected experts | `run_moe_c1_fp16` → `route_moe_topk_shared_fp16`, `selected_moe_gate_up_pack8_fp16`, `activate_rotate_moe_down_fp16`, `selected_moe_down_pack8_fp16`, shared expert, combine (`5723-5747`) | Runs once per layer with `tokens=rows`. Selected-expert GEMV kernels consume device `selected_experts` and `rows = tokens * num_experts_per_tok`; no host loop over experts in the target verifier. | **Row-batched but not layer-fused**. The old M7 work belongs here as M12.4: one verifier-layer selected-expert primitive should reduce launch count and improve B scaling. |
| Shared expert W4 path | `shared_expert_paro_w4_fp16` (`4999-5204`) | At `tokens>1`, uses prefill-style W4 kernels for gate/up and down; M7.C.6 did not alter this safe-but-noisy site. | **Batched, but not small-B optimal**. Potential sub-primitive, secondary to full-attn serial row loop and LM head. |
| Full-attention default path | `_run_full_attention_chain_c1_loop` (`qwen35_paro_runner.py:2939-2987`) | Explicit `for row, position in enumerate(positions)` over verifier rows. Each row calls `run_full_attention_moe_c1_layer_fp16(tokens=1)` and then copies the row output. | **Serial by verifier row**. For B=3 this is 10 full-attn layers × 4 row-layer invocations = 40 c=1 layer runs per cycle. This is the clearest non-llama.cpp shape and the M12.2 target. |
| Full-attention batched alternative | `_run_full_attention_chain_batched` (`2988-3127`) | One pass over `rows`: batched RMSNorm, rotate, QKV, batch K/V append, prefill GQA gate with per-row causal limit, O projection, post-norm, and `run_moe_c1_fp16(tokens=rows)`. | **True row-batched**, but historical diagnostics found it slower at small B because it reuses prefill-style kernels with high fixed overhead. M12.2 should not simply flip this default; it needs a small-B full-attn verifier primitive. |
| LM head + top1 | `_sample_verify_rows_from_hidden` (`3571-3618`) | Final norm and cast are row-batched; `w8a16_linear_bf16_f32_out` materializes full `rows × vocab` logits; `argmax_f32_rows_i32` reduces each row. | **True row-batched but verifier-inefficient**. It scales with `rows * vocab` and writes a logits slab even though accept only needs top-1 / next-token provenance. This is M12.3. |
| Accept summary | `dflash_accept_chain_i32` (`3627-3651`) | One GPU kernel over rows/requests. | **True batched and likely tiny**. Not a priority except to keep GPU-fast accept enabled after M12.3. |

#### Immediate ordering from the map

1. **M12.1 timeline split** should instrument the existing boundaries exactly as
   above: draft build, target forward, LM-head/top1, accept read/CPU oracle,
   linear-state commit, proposer repair, and final sync. The sums must reconcile
   with `cycle_marker_ns`.
2. **M12.2 first implementation target:** replace `_run_full_attention_chain_c1_loop`
   with a small-B row-batched full-attention verifier primitive. The existing
   `_run_full_attention_chain_batched` proves the ABI/topology, but the kernel
   shape must avoid prefill fixed overhead.
3. **M12.3 second implementation target:** fuse verifier LM head + top1/accept so
   the verifier does not materialize `rows × vocab` logits.
4. **M12.4 third target:** convert the target MoE path from “row-batched but many
   small primitive launches” into a verifier-layer primitive. This is where the
   earlier M7 selected-expert work belongs.
5. **Proposer handoff is not free:** the persistent proposer is resident, and
   the 2026-06-11 proposer skip removed discarded expert-topk reads plus
   update-only lm-head/top1 reads by default, but draft build and repair are
   still serial c=1 MTP advances. Measure this separately before assuming the
   target verifier alone explains the AR-token-equivalent cycle cost.

### M12 implementation track (Updated for W7900 Phase)

Based on the 32 optimization iterations on `dflash`, the orchestration/Python side has been pushed to a diagnostic ceiling of `C3 = 2.82x`. To break the `< 2.0x` floor, we must move to deep kernel fusion and graph capture.

| # | Sub-task | Goal / acceptance gate | Status |
|---|---|---|---|
| M12.1 | **Graph capture for batched mode** | Re-enable HIP Graph capture for `chain_attn_mode="batched"` (chain shape is deterministic per cycle). Gate: `C_3` cycle wall drops by 15-20%. | **Done (no perf win)** 2026-05-22 W7900: `chain_attn_mode='batched' + graph_mode!='off'` now allowed; cache key extended with `(chain_attn_mode, linear_attn_mode)`; replay launches on the caller's stream. Validated exact-AR. Cycle-2+ wall unchanged (~33.3 ms with graph=auto vs graph=off) because ROCm 7.x `hipGraphLaunch` per-node overhead on our 1,840-kernel DAG matches direct ctypes overhead. The Python round-trip *is* removed; it just isn't the bottleneck at this kernel count. Real wins require M12.2 + M12.4 to cut the kernel count first. Artifact: `benchmarks/results/2026-05-22-hipengine-mtp-m12.1-w7900-graph-capture-diagnostic.json`. |
| M12.2 | **Verifier LM-head weight sharing** | Replace the verifier LM-head path that streamed the W8A16 weight matrix once per row with a multi-row weight-sharing GEMV feeding the existing top-1 reduction. Gate: final top1 rows identical, LM-head timeline drops. | **Done** 2026-05-22 W7900: exact AR preserved on the stable quicksort gate; MTP throughput 66.33 → 68.77 tok/s (+3.7% over M12.4+M12.5), verify time 26.45 → 25.17 ms/cycle. |
| M12.3 | **Selected-expert mul_mat_id consolidation / staged-down slice** | Original charter expected per-expert GEMV launches, but M13.A found the verifier already uses one ids-tensor selected GEMV per layer/op. The narrower slice was the down-side staged kernel: fuse selected `silu_mul_dual_rotate_out + selected_pack8_gemv` while retuning selected verifier GEMVs to 64 threads. Gate: exact MTP/DFlash smoke plus rocprof launch-count and kernel-time reduction vs prior default. | **Landed 2026-06-03, superseded/no-held 2026-06-11 through 2026-06-12.** The staged path passed exact smokes and moved B=3/decode4 verifier calls/pass `1019.00 -> 989.33` (-2.9%) and kernel time/pass `15.369 -> 14.949 ms` (-2.7%) vs the then-prior default. Current graph-auto D32 suite is faster with the unfused fallback because the capture-safe barrier/fill cost dominates (`27.648 -> 27.408 ms/cycle`), so `HIPENGINE_SELECTED_MOE_DOWN_STAGED=1` is opt-in only. A current-best graph-off `decode_batched + skip` compound retest also stayed exact but regressed wall (`21.661 -> 21.763 ms/cycle`) and verify (`16.511 -> 16.628 ms/cycle`). Artifacts: `benchmarks/results/2026-06-03-hipengine-mtp-selected-moe-down-staged-default.json`, `benchmarks/results/2026-06-12-hipengine-mtp-decode-batched-staged-down-on-9prompt-d32.json`. The larger few-hundred-launch goal remains outside this item because the original per-expert-launch premise is false. |
| M12.4 | **Device-resident accept summary → device-resident state commit** | Replace the 60 D2D `hipMemcpy` calls in `_commit_bulk_linear_states` with one indexed-copy kernel keyed off `commit_rows[0]` (already produced by `dflash_accept_chain_i32` on device). Folds the commit into the captured graph. | **Done** 2026-05-22 W7900: fused multi-layer linear-state commit landed; exact AR preserved; MTP throughput +~1% on the stable quicksort gate. |
| M12.5 | **Invariant cycle metadata cache** | Cache the verifier-chain metadata that is invariant for a fixed `(B, base_slot)` and refresh only the dynamic token/position buffers per cycle. | **Done** 2026-05-22 W7900: H2D metadata copies 11 → 5 for the common path; exact AR preserved; +~1% MTP throughput on top of M12.4. |
| M12.6 | **Small-B W4 verifier GEMV** | Dedicated multi-row pack8 W4 kernels share weights across `B+1 <= 8` verifier rows. Full all-site enablement improved the stable quicksort prompt but changed verifier numerics enough to fail exact AR on the llama.cpp-compatible translation prompt. | **Partial / gated** 2026-05-22 W7900, updated 2026-06-11: the FMA row-loop kernel now half-rounds dequantized FP16 weights to better match the stock WMMA prefill path, and default enables the prompt-suite-safe sites (`full_qk`, `linear_qkv_z`, `dense_gate_up`, `single_full_o`, `single_shared_down`, `single_dense_down`, `single_linear_out`, `single_full_v`). `single_linear_out` was promoted after the current D32 9-prompt suite passed exact and the pre-promotion no-env stack failed `translation` without it; `single_full_v` was promoted after a fresh same-session default A/B and profile both moved down. `HIPENGINE_W4_MULTI_ROW_PACK8_SITES=all` remains available for risky perf experiments; `shared_gate_up` is handled by the M16.4 split-output output-tiled path instead of M12.6. |
| M12.7 | **GPU-resident proposer loop** | Proposer's 3 AR steps are still serial c=1 advances; the 2026-06-11 default-on skip trims discarded host reads/results but does not batch or graph-loop the proposer. Graph-capture the proposer loop or fuse into one C++ step to drop toward ~3 ms. Gate: exact accepted-token provenance unchanged. | Pending |
| M12.8 | **Adaptive B / fallback policy** | After kernel/graph work drops `C_3` below 2.0, add policy to dynamically choose B=1 or pure AR fallback based on proposer confidence to rescue low-acceptance cycles. | Pending |

### llama.cpp PR #21845 follow-up: small-column verifier GEMV audit (2026-06-07)

Review note: [`ggml-org/llama.cpp#21845`](https://github.com/ggml-org/llama.cpp/pull/21845)
optimizes SYCL MTP verifier mat-vec by handling several RHS columns in one
kernel (`ncols <= 8`) instead of launching one quantized GEMV per verifier
column and rereading weights each time.  Treat this as a **checklist**, not a
code port: hipEngine's HIP/gfx1100 kernels, PARO layouts, and exact-AR gates are
different, and the useful invariant is the shape (`small rows share one weight
stream`) rather than the SYCL implementation.

Action items to double-check before the next M12/M13 perf push:

1. **Audit single-row gates.** Search every verifier-hot `rows == 1`,
   `tokens == 1`, `ncols == 1`, and optimized-layout/reorder eligibility gate.
   For MTP/DFlash verifier shapes, the default question should be whether the
   exact path can support `2 <= rows <= 8`, not whether the path is single-row.
2. **Trace dispatch coverage for `rows=B+1`.** For B=1/2/3/5/7, confirm which
   QKV/O, linear-attention, selected/shared/dense MoE, and LM-head projections
   hit read-once-weight multi-row kernels versus row-wise or prefill fallback
   kernels.  Add callsite markers for the W4 projection bucket before assigning
   a profile row to shared-expert, dense, or full-attention work.
3. **Finish exact W4 multi-row coverage deliberately.** M12.6 already proved the
   weight-sharing idea but only the prompt-suite-safe site mask is enabled by
   default.  `single_linear_out` and `single_full_v` joined the default mask on
   2026-06-11 after current-stack 9-prompt exact gates. Treat
   `shared_gate_up` as superseded by the M16.4 split-output output-tiled route
   unless a new M12.6-specific reason appears.  Keep
   `HIPENGINE_W4_MULTI_ROW_PACK8_SITES=all` as an experiment until every added
   site has its own 9-prompt exact evidence.
4. **Keep LM-head as the reference success pattern.** M12.2 is the local analog
   of the llama.cpp PR: one verifier LM-head launch streams W8A16 weights once
   for all small rows and feeds the existing top-1 reduction.  New projection
   work should match that evidence standard: exact final top1 rows, reduced
   launches/weight traffic, and a retained rocprof/economics artifact.
5. **Do not promote a pure launch win.** The 2026-06-07 family rollup still shows
   host/launch/D2H residual plus W4 projection time as the wall.  A retained
   change must reduce verifier ms/cycle or `C_B`, not just move launches between
   buckets.

Promotion rule: no MTP speed row is accepted until the economics artifact shows
`avg_visible_tokens_per_verify_cycle / cycle_cost_ar_tokens > 1.0` on the same
prompt/workload, with exact AR equality and accepted-token provenance preserved.

## M13 — launch-count + host-dispatch consolidation (2026-05-23)

### M13.0 framing

M12.3 was originally chartered as "selected-expert mul_mat_id consolidation" with
the headline gate "1,840 calls/pass → a few hundred". The M13.A audit below
shows that line is **already false today**: the selected-expert MoE GEMVs are
already a single ids-indexed mul_mat_id launch each (`gemv_awq_selected_dual_pack8_transposed_*`
and `gemv_awq_selected_pack8_transposed_*` at `runtime/qwen35_paro.py:4791`
and `:4872`), one per (layer, op). The 40+40 = 80 selected-MoE GEMV launches/pass
in the M12.6c rocprof is literally one per layer per MoE op. So the original
M12.3 framing — "collapse 138 per-expert launches into one ids-tensor GEMV" —
refers to a state of the code that no longer exists.

The **actual** remaining structural fat is not the MoE GEMVs themselves. It is
the ~10-launch surround around each GEMV (RMSNorm → rotate → GEMV → RoPE/q+k
norm → KV append → attention → rotate → GEMV → add+RMSNorm → router → selected
gate_up → SiLU+rotate → selected down → shared expert → combine) × 40 layers,
which is what produces the 1052-launch DAG. M12.1's HIP graph capture sits
idle because at 1052 nodes the per-node `hipGraphLaunch` overhead on ROCm 7.x
is comparable to the per-launch ctypes cost.

M13 reframes the kernel-count attack accordingly:

- **M13.A** (this section, source-only audit): build a static per-layer
  launch table for one B=3 / tokens=4 verifier pass at `chain_attn_mode='batched'`
  so every later proposal is rooted in concrete numbers.
- **M13.B** (no GPU until measure): instantiate already-templated fused kernels
  whose transposed-layout entry points are missing today (numerically
  identical exposure, not new algorithms).
- **M13.C** (no GPU until measure): C-side per-MoE-layer dispatcher that
  collapses ~10 Python/ctypes round-trips per layer into one C call.
  Numerically identical to the current Python sequence; gated by env so we
  can A/B cleanly.
- **M13.D**: re-evaluate graph replay after B+C. The point is to make the
  captured DAG small enough that `hipGraphLaunch` per-node overhead is no
  longer competitive with direct dispatch, i.e. to make M12.1 load-bearing.
- **M13.E** (contingency only if D leaves ≥10% on the table): sort-by-expert
  pre-pass for the selected MoE GEMVs.

Unchanged ground rules: every step is gated by exact-AR on the 9-prompt
`mtp-bench.py --mode hipengine-current` suite. No metric promotion until
A→D produces a retained `benchmarks/results/` artifact with the matching
`benchmarks/README.md` + `CHANGELOG.md` rows.

### M13.A static launch audit (B=3, tokens=4, chain_attn_mode='batched')

Model: Shisa packed PARO Qwen3.5/3.6 with 40 layers (30 `linear_attention`,
10 `full_attention`), 256 experts, top-k=8, hidden=4096, FP16 activations.
Verifier rows = B+1 = 4 (one parent + three drafts per cycle). All numbers
below are derived from `hipengine/runtime/qwen35_paro_runner.py` and
`hipengine/runtime/qwen35_paro.py` at HEAD (`fe3bea0`), not from rocprof.
The per-pass total is then reconciled against the measured 1052-launch
figure from `benchmarks/results/2026-05-22-hipengine-mtp-verifier-rocprof-w7900-current-m12.6c.json`.

Default env at the time of audit (per `docs/MTP.md` M12.6 entry + runtime defaults):

- `HIPENGINE_W4_MULTI_ROW_PACK8_SITES = full_qk,linear_qkv_z,dense_gate_up,single_full_o,single_shared_down,single_dense_down`
- `HIPENGINE_SMALL_BATCH_DECODE_THRESHOLD = 7`
- `HIPENGINE_VERIFY_MOE_GROUPED_MIN_TOKENS = 16` (→ at tokens=4 we use `run_moe_c1_fp16`, **not** grouped)
- `HIPENGINE_W4_PREFILL_SMALLBATCH_TILE_M = 16` (rows ≤ 8)

This launch-count table is historical. Current 2026-06-11 defaults add
`single_linear_out` and `single_full_v` to the M12.6 safe mask, use M16.4
split-output output-tiled W4 for `shared_gate_up`, and route the hot MoE path
through the C dispatcher.

#### Per-`linear_attention` layer (`run_linear_attention_moe_chain_tloop_layer_fp16`, tokens=4)

| # | Source site | Kernel(s) launched | Launches | Notes |
|---|---|---|---:|---|
| 1 | `input_rmsnorm_fp16` | `paro_rmsnorm_out_fp16` | 1 | |
| 2 | `rotate_linear_attention_inputs_fp16` | `paro_rotate2_fp16` | 1 | tokens>1 → fused-barrier path disabled |
| 3 | `project_linear_attention_qkv_z_fp16` | 2× `gemv_awq_pack8_transposed_fp16` | 2 | tokens=4 ≤ small_batch=7; two single GEMVs (M7.C.6 alias-safe split) |
| 4 | `project_linear_attention_ab_fp16` | 2× `dense_gemv_out_fp16` (or `rocblas_gemm_ex`) | 2 | tokens>1 split |
| 5 | `qwen35_linear_attn_chain_conv_decode_fp16_tloop` | conv | 1 | |
| 6 | `qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_tloop_fp16` | GDN t-loop | 1 | |
| 7 | `project_linear_attention_out_fp16` | `f32_to_fp16` + `paro_rotate1_fp16` + `awq_fusedw4_prefill_transposed_fp16` (single_linear_out **not** in safe mask → prefill path) | 3 | |
| 8 | `post_attention_add_rmsnorm_fp16` | `paro_add_rmsnorm_out_fp16` | 1 | |
| 9 | `run_moe_c1_fp16` → `route_moe_topk_shared_fp16` | `qwen35_router_topk_shared_out_fp16` | 1 | tokens>1 path |
| 10 | `selected_moe_gate_up_pack8_fp16` | `paro_rotate1_fp16` + `gemv_awq_selected_dual_pack8_transposed_fp16` (already mul_mat_id) | 2 | |
| 11 | `activate_rotate_moe_down_fp16` | `silu_mul_dual_rotate_out_fp16` | 1 | |
| 12 | `selected_moe_down_pack8_fp16` | `gemv_awq_selected_pack8_transposed_fp16` (already mul_mat_id) | 1 | |
| 13 | `shared_expert_fp16` → `shared_expert_paro_w4_fp16` (layer_type=linear → `small_batch=False`) | `paro_rotate2_fp16` (gate+up) + `awq_fusedw4_prefill_dual_fp16` + `silu_mul_separate_out_fp16` + `paro_rotate1_fp16` (down) + `awq_fusedw4_prefill_fp16` (down) | 5 | shared_gate_up **not** in safe mask → fall through to prefill kernels |
| 14 | `combine_moe_c1_shared_residual_fp16` | `weighted_sum_shared_gate_combine_residual_batch_out_fp16_f32w` | 1 | |
| 15 | per-layer next_hidden copy in `_iterate_verify_chain_layers` | `hipMemcpyAsync` (D2D) | 1 | `out.ptr != next_hidden.ptr` — `run_moe_c1_fp16` returns `scratch.moe_out`; the M12.6 `out=` write-through path is **not wired into the batched verifier** |

**Per linear-attention layer: 24 launches.** Across 30 linear-attention layers: **720 launches/pass.**

#### Per-`full_attention` layer (`_run_full_attention_chain_batched`, tokens=4)

| # | Source site | Kernel(s) launched | Launches | Notes |
|---|---|---|---:|---|
| 1 | `input_rmsnorm_fp16` | `paro_rmsnorm_out_fp16` | 1 | |
| 2 | `rotate_full_attention_inputs_fp16` | `paro_rotate3_fp16` | 1 | tokens>1 → fused-barrier path disabled |
| 3 | `project_full_attention_qkv_fp16` | 2× `gemv_awq_pack8_transposed_fp16` (Q/K small-batch split) + `awq_fusedw4_prefill_transposed_fp16` (V — `single_full_v` not in safe mask) | 3 | M7.C.6 alias-safe split |
| 4 | `prepare_full_attention_qkv_fp16` | `qwen35_split_qgate_fp16` + `fp16_to_f32` + `qwen35_head_rmsnorm_partial_rotary_positions_f32_bf16` | 3 | |
| 5 | `append_full_attention_kv_fp16_batch` | `qwen35_write_paged_kv_mixed_value_fp16_spans` | 1 | |
| 6 | `prefill_full_attention_gqa_gate_fp16` | `qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans` | 1 | |
| 7 | `project_full_attention_o_fp16` | `paro_rotate1_fp16` + `gemv_awq_pack8_multi_row_transposed_fp16` (`single_full_o` IS in safe mask → multi-row path) | 2 | |
| 8 | `post_attention_add_rmsnorm_fp16` | `paro_add_rmsnorm_out_fp16` | 1 | |
| 9 | `run_moe_c1_fp16` → `route_moe_topk_shared_fp16` | `qwen35_router_topk_shared_out_fp16` | 1 | |
| 10 | `selected_moe_gate_up_pack8_fp16` | `paro_rotate1_fp16` + `gemv_awq_selected_dual_pack8_transposed_fp16` | 2 | |
| 11 | `activate_rotate_moe_down_fp16` | `silu_mul_dual_rotate_out_fp16` | 1 | |
| 12 | `selected_moe_down_pack8_fp16` | `gemv_awq_selected_pack8_transposed_fp16` | 1 | |
| 13 | `shared_expert_fp16` → `shared_expert_paro_w4_fp16` (layer_type=full + tokens≤7 → `small_batch=True`) | `paro_rotate2_fp16` + `gemv_awq_dual_pack8_transposed_fp16` + `silu_mul_dual_rotate_out_fp16` + `gemv_awq_pack8_transposed_fp16` | 4 | small-batch path skips standalone down rotate (fused into silu) |
| 14 | `combine_moe_c1_shared_residual_fp16` | `weighted_sum_shared_gate_combine_residual_batch_out_fp16_f32w` | 1 | |
| 15 | per-layer next_hidden copy | `hipMemcpyAsync` (D2D) | 1 | same gap as linear-attn: `run_moe_c1_fp16` does not receive `out=next_hidden` |

**Per full-attention layer: 23 launches.** Across 10 full-attention layers: **230 launches/pass.**

#### Sample + accept + commit + embedding

| Stage | Source site | Kernel(s) | Launches |
|---|---|---|---:|
| Embedding | `_iterate_verify_chain_layers` head | `embedding_lookup_batch_fp16_i64` | 1 |
| Sample | `_sample_verify_rows_from_hidden` | `paro_rmsnorm_out_fp16` + `fp16_to_bf16` + `w8a16_linear_bf16_f32_multi_row` (M12.2) + `argmax_f32_rows_i32` | 4 |
| Accept | `_launch_verify_accept_summary` | `dflash_accept_chain_i32` | 1 |
| Commit (M12.4 fast path) | `_commit_bulk_linear_states` | `linear_state_pair_commit_i32` + 0–2 H2D pointer-table refreshes | 1–3 |

**Tail: 7–9 launches/pass.**

#### Per-pass total — static vs. measured

| Component | Static (audit) | Measured (rocprof m12.6c) | Δ |
|---|---:|---:|---:|
| 30 × linear-attention layer | 720 | — | |
| 10 × full-attention layer | 230 | — | |
| Embedding + sample + accept + commit | 7–9 | — | |
| **Total** | **957–959** | **~1052** | **+93–95** |

Residual ~93 launches the static audit doesn't predict: these are almost
certainly per-layer `runtime_memset` (`_memset_tensor` calls scattered
through MoE/shared/grouped scratch resets) and the H2D pointer-table
refreshes inside `_commit_bulk_linear_states`. They aren't visible in the
source walk because they're embedded in helpers called once per layer; the
follow-up M13.A.2 task should enumerate every `runtime.memset_async` and
`memcpy_async` call inside the verify path to close this 9% gap.

#### Where the launch budget actually lives

From the M12.6c rocprof families × audit attribution:

Note: the table below is a historical attribution snapshot. The current
2026-06-11 post-P1/default stack has moved the linear-attn out-proj and
C-dispatch linear shared-down sites off the fused small-batch prefill path; use
the current benchmark rollup artifacts for live counts.

| Family | Calls/pass | Audit attribution |
|---|---:|---|
| `w4_single_gemv` (=`gemv_awq_pack8_transposed_fp16`) | 140 | 30×2 linear-attn QKV/Z split (60) + 10×2 full-attn Q/K split (20) + 30×1 shared down small-batch (0, full-attn only ≈ 10) + 10×1 full-attn O proj (10) + 10×1 full-attn shared gate_up dual (counted as dual) + remainder ≈ "small-batch single GEMVs landed by M7.C.6 + M12.6 mask" |
| `moe_paro_rotate_in` (= `paro_rotate{1,2,3}`) | 190 | 30 linear-attn ×4 rotates (qkv-z + out + gate_up + shared.gate_up + shared.down = 5 ⇒ 150) + 10 full-attn ×4 rotates (qkv + o + gate_up + shared.gate_up ≈ 4 ⇒ 40). Family name covers ALL `paro_rotate*` launches, not only MoE ones. |
| `w4_single_prefill_smallbatch` | 70 | 30 linear-attn shared down prefill (30) + 30 linear-attn out_proj prefill (30) + 10 full-attn shared down/aux (≈10) |
| `w4_dual_prefill_smallbatch` | 30 | 30 linear-attn shared gate+up prefill (full-attn shared gate+up uses dual GEMV not prefill at tokens=4) |
| `moe_gate_up_dual_gemv` (mul_mat_id) | 40 | one per MoE layer per gate+up op — already collapsed |
| `moe_down_gemv` (mul_mat_id) | 40 | one per MoE layer per down op — already collapsed |
| `linear_attention_gdn_decode` | 30 | one per linear-attn layer |
| `w8a16_linear` (LM head) | 1 | M12.2 multi-row |
| Tail (~520): rmsnorm, paged_kv, decode_attn, silu, combine, router, runtime_copy/memset | ~520 | input_rmsnorm + post_attn_rmsnorm + qkv-norm + silu_mul × shared + silu_mul_dual_rotate + combine + router + paged_kv append + paged_attn prefill + per-layer next_hidden D2D + memsets |

#### Audit conclusions

1. **Selected MoE is already mul_mat_id.** No further consolidation possible
   at the selected-expert GEMV level. Any further MoE win has to come from
   *expert-grouping* (M13.E contingency) or from collapsing the surround.

2. **The 40 layers × ~5 rotates/layer = 190 `paro_rotate*` launches** are the
   single largest unfused launch family by count. Each is ~5 µs of pure
   dispatch. Two cheap wins exist here:

   - **(M13.B.1)** `gemv_awq_selected_dual_pack8_transposed_rotate_out_{fp16,bf16}`:
     the templated kernel already exists in `kernels/hip_gfx1100/quant/paro_awq_gemv.hip`
     (line 1466, `gemv_awq_selected_dual_pack8_strided_rotate_out_kernel`,
     templated on `qweight_transposed`). Only the strided extern-C wrappers
     are exposed (`hipengine_gemv_awq_selected_dual_pack8_strided_rotate_out_{fp16,bf16}`
     at line 2610 / 2774). Adding the transposed instantiation removes the
     `paro_rotate1` launch in `selected_moe_gate_up_pack8_*` for every MoE
     layer that uses selected dual GEMV (40 MoE ops → ~40 paro_rotate
     launches/pass eliminated). **Numerically identical** to today's
     two-launch sequence.
   - **(M13.B.2)** A `paro_rotate_into_gemv_awq_pack8_transposed_*` fold for
     the shared-expert down rotate at linear-attention layers (30 launches/pass).
     Less obvious; needs a quick survey of whether the templated rotate-in
     kernel exists for the non-selected variants.

3. **The per-layer next_hidden D2D copy is unwired write-through, not a fix.**
   The batched verifier path at `qwen35_paro_runner.py:~3175` calls
   `state.run_moe_c1_fp16(mlp_input, residual, scratch=moe_scratch, tokens=rows)`
   **without passing `out=next_hidden`**. M12.6's `out=` parameter on
   `run_moe_c1_fp16` exists exactly for this purpose. Wiring it through
   removes 1 D2D `hipMemcpyAsync` per layer = **40 launches/pass eliminated**
   for free, numerically identical. The c1_loop full-attention path has the
   same gap. **This is M13.B.0, the freebie.**

4. **The 957–1052 launch range is the surround, not the GEMVs.** Even with
   M13.B.0 + M13.B.1 + M13.B.2 we still emit ~870 launches/pass. The C-side
   per-MoE-layer dispatcher (M13.C) does not reduce launches, but it reduces
   the per-launch Python overhead and shortens the captured graph's record
   sequence. M13.D is where graph capture (M12.1) finally pays out, if at all.

#### M13.A follow-ups (no GPU yet)

- M13.A.1: file an issue/note to wire `out=next_hidden` through `run_moe_c1_fp16`
  in both the linear-attention chain_tloop and full-attention batched paths.
  Trivial source change; numerically identical; eliminates 40 D2Ds/pass.
- M13.A.2: enumerate every `runtime.memset_async` and `memcpy_async` call
  reachable from `_iterate_verify_chain_layers`. Close the 93-launch gap
  between static audit (957–959) and rocprof (1052).
- M13.A.3: confirm whether the `gemv_awq_pack8_transposed_rotate_*` symbol
  exists in the templated kernel body; if not, decide whether M13.B.2 is a
  cheap instantiation or a new kernel.

All three follow-ups are source-only; defer GPU validation to the M13.D
measure block.

### M13 implementation track

| # | Sub-task | Goal / acceptance gate | Status |
|---|---|---|---|
| M13.0 | Plan + audit write-up in `docs/MTP.md` + `WORKLOG.md` | Source-only plan landed; later phases reference this section. | **Done** 2026-05-23 |
| M13.A | Static per-pass launch audit (this section) | Reconcile static count vs M12.6c rocprof to <10% gap; identify the actual structural fat. | **Done** 2026-05-23 (static 957–959 vs measured 1052, 93-launch residual attributed to memsets + pointer-table refreshes; M13.A.2 will close the gap) |
| M13.B.0 | Wire `out=next_hidden` through `run_moe_c1_fp16` in batched + chain_tloop verifier paths | Exact-AR on 9-prompt suite; rocprof shows ~40 fewer `runtime_copy` launches/pass. | **Retained** 2026-05-23 W7900: exact `9/9`, cycle cost `3.716 -> 3.613` AR-token equivalents (-2.8%), verifier launches/pass `1052 -> 1011.6`, `runtime_copy` family `52 -> 12.6`; artifacts `benchmarks/results/2026-05-23-hipengine-mtp-bench-suite-w7900-m13.b0.json` and `benchmarks/results/2026-05-23-hipengine-mtp-verifier-rocprof-w7900-m13.b0.json`. |
| M13.B.1 | Expose transposed-layout instantiation of `gemv_awq_selected_dual_pack8_*_rotate_out_{fp16,bf16}` | Exact-AR on 9-prompt suite; rocprof shows ~40 fewer `paro_rotate1` launches/pass. | **Kernel landed, default-off** 2026-05-23 W7900: bit-exact LDS round-trip (Option C) added to the existing strided rotate-out kernel; transposed FP16 extern + Python wrapper + registry entry added; `selected_moe_gate_up_pack8_fp16` env-gated. Smoke + 9-prompt suite stay exact-AR with identical token sequences. Measurement rejected the default-on: launches drop 1011.6 → 971.6 (−40, as expected) but `moe_gate_up_dual_gemv` ms/pass `1.86 → 14.21` (+664%) because the kernel re-does the full in-LDS rotation in every `(out_pack, row)` block. For verifier shape (`out_packs ≈ 192`, `rows = 32` with 8-way redundancy per token) the redundant rotation work is ~1500× the unfused chain, blowing up total kernel time by +12.4 ms/pass. Default `HIPENGINE_MOE_FUSED_ROTATE=0`; opt-in for shape-specific experiments. M13.B.3 (proper HBM-staged variant) is the right structural fix. Artifacts: `benchmarks/results/2026-05-23-hipengine-mtp-{bench-suite,verifier-rocprof}-w7900-m13.b1-fusedon-rejected.json`. |
| M13.B.2 | Survey + (if cheap) add transposed-layout rotate-in fold for non-selected pack8 GEMVs used by shared expert | Exact-AR on 9-prompt suite; rocprof shows ~30 fewer `paro_rotate1` launches/pass. | **Kernel landed, default-off** 2026-05-23 W7900: survey found the existing HBM-staged `gemv_awq_dual_pack8_transposed_rotate_staged_fp16` (used by the attention Q/K rotate-fuse) is structurally correct for the shared-expert small-batch path. Patched the kernel barrier from `prior + 1 == rotate_blocks` to `prior + 1 == rotate_blocks * gridDim.y` and dropped the launcher `rows != 1` rejection so it can serve the verifier batched shape (tokens=4). Wired into `shared_expert_paro_w4_fp16` small_batch branch (10 full-attention layers at B=3 batched). Exact-AR holds across all four mode combinations (`chain_attn_mode in {batched, c1_loop}` × `graph_mode in {off, validate}`, fused-on and fused-off all produce identical token sequences). Rocprof shows the expected `moe_paro_rotate_in: 190 → 180 (-10) calls/pass`, but the staged launcher's implicit `hipMemsetAsync(barrier, 0, 8)` adds the same +10 launches to the `other` family. **Net launch delta = 0**, kernel time `17.315 → 17.407 ms/pass` (+0.5%). Defaulted off via `HIPENGINE_SHARED_EXPERT_FUSED_ROTATE`; kernel infra kept for a future keyed-barrier variant. Same lesson as M13.B.1: the kernel design has a hidden host-side overhead that swallows the dispatch saving. Artifacts: `benchmarks/results/2026-05-23-hipengine-mtp-verifier-rocprof-w7900-m13.b2-{fusedon-rejected,fusedoff-baseline}.json`. Tracked as M14.fuse.barrier. |
| M13.C | C-side per-MoE-layer dispatcher gated by `HIPENGINE_MOE_C1_LAYER_C_DISPATCH=1` | Exact-AR on 9-prompt suite when env on; behavior unchanged when env off; cycle_cost at `graph_mode=off` drops by host-dispatch savings. | **Deferred to M14.dispatch.1** 2026-05-23 W7900: cProfile measurement of the verifier hot path showed `run_moe_c1_fp16` own-time is 3 µs/call (40 layers × ~25 verifier passes × 3 µs = 3 ms total Python overhead per decoded token), not the 30+ ms the M13.A audit speculated. The dominant per-launch cost is the ctypes argument marshaling + HIP runtime dispatch (`hipLaunchKernelGGL`) itself, both of which a Python-level dispatcher cannot remove. A real C-side dispatcher (one new `.hip` translation unit per dispatcher, containing all kernel launchers, JIT-built as a single library with one ctypes entry point) would save ~6–8 ms/pass but is a ~300–500 LoC change touching the build system. Outside the scope of M13's launch-count attack; tracked as M14.dispatch.1. |
| M13.D | Re-evaluate graph replay after B+C | At least one of `graph_mode=auto` / `validate` beats `graph_mode=off` cycle_cost by ≥5% on the 9-prompt suite with exact-AR. | **Rejected** 2026-05-23 W7900: M13.B.0 dropped launches/pass from 1052 to 1011.55 (-3.8%) but graph replay still loses to direct dispatch at this launch count.  9-prompt suite at B=3, batched, max-tokens=64: `graph_mode=off cycle_cost=3.639`, `graph_mode=auto cycle_cost=3.782` (+3.9% worse), `graph_mode=validate cycle_cost=7.727` (validate overhead). Same conclusion as M12.1 — ROCm 7.x per-graph-node `hipGraphLaunch` overhead is comparable to direct dispatch at >900 launches. The graph capture infra remains landed (M12.1) but stays opt-in pending further launch reductions via M14.dispatch.1 + M14.fuse.*. Artifacts: `benchmarks/results/2026-05-23-hipengine-mtp-bench-suite-w7900-m13.d-{graphoff,graphauto-rejected,graphvalidate-rejected}.json`. |
| M13.B.3 | Properly-staged selected-MoE rotate+GEMV (HBM staging, rotate once per x_row, barrier, GEMV reads back) | Replaces M13.B.1 default-off kernel; should land both -40 launches AND ≤ baseline kernel time. Numerically bit-exact via the staged HBM scalar_t write. | **Gate/up staged attempt rejected; down-side slice superseded/no-held.** Selected gate/up staged rotate is bit-exact and cuts launches/pass but regresses kernel time, so it remains `HIPENGINE_SELECTED_MOE_STAGED_ROTATE=0` default-off. The down-side staged slice (`silu_mul_dual_rotate_out + selected_pack8_gemv`) was a 2026-06-03 verifier-window win, but the current exact D32 stack is faster with the fallback; the 2026-06-12 `decode_batched + graph_off + skip` compound retest regressed wall `21.661 -> 21.763 ms/cycle`. |
| M13.E | Sort-by-expert pre-pass for selected MoE GEMVs (contingency only) | Only if D leaves ≥10% on the table. | **Deferred to M14.fuse.5** 2026-05-23: M13.D was rejected, not "left work on the table" — graph replay's 3.9% penalty does not point at MoE GEMV weight-bandwidth waste, it points at dispatch overhead.  Sort-by-expert addresses a different lever (weight reuse across expert lanes within a row); it should only be revisited if a future profile shows the selected MoE GEMV is bandwidth-bound (not the case today).  Tracked as M14.fuse.5. |

Ground rule for the whole M13 block: any code change that touches MoE
numerics (M13.B.2 or beyond) needs the existing exact-AR gate plus the
`benchmarks/results/` rollup discipline in `AGENTS.md`.

## M14 — deferred MTP follow-ups (post-M13)

Catalog of work that M13 surfaced but did **not** execute, with the lineage
entry that surfaced each item and the gating condition under which it
should be revisited.  Items are not ordered by priority; some are tiny
plumbing fixes, others are full kernel implementations.  Re-prioritize at
the start of M14 against whatever cycle-cost gap remains after M13.D.

### Prompt-rendering diagnostic (2026-05-24)

User-requested check after the DFlash vs MTP discussion: compare MTP acceptance
under raw prompt text, Qwen chat-template `enable_thinking=true`, and Qwen
chat-template `enable_thinking=false`.  The harness now supports
`scripts/mtp-bench.py --mode hipengine-current --prompt-render
{raw,qwen_chat_thinking_off,qwen_chat_thinking_on}`; raw remains the default so
old artifacts stay comparable.

W7900/gfx1100, Qwen3.6-35B-A3B-PARO-MTP-BF16, B=3, batched verifier,
`graph_mode=off`, one run per prompt:

| Prompt/render | Decode | Exact | MTP/AR | Acceptance | Accepted/cycle | Cycle cost |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `code_python` raw | 64 | yes | `0.498x` | `0.521` | `1.52` | `5.10` |
| `code_python` Qwen thinking-on | 64 | yes | `0.410x` | `0.359` | `1.06` | `5.03` |
| `code_python` Qwen thinking-off | 64 | yes | `0.352x` | `0.267` | `0.78` | `5.05` |
| `code_cpp` raw | 64 | yes | `0.498x` | `0.487` | `1.46` | `4.95` |
| `code_cpp` Qwen thinking-on | 64 | yes | `0.410x` | `0.348` | `1.03` | `5.01` |
| `code_cpp` Qwen thinking-off | 64 | **no** | n/a | n/a | n/a | n/a |
| `code_cpp` Qwen thinking-on | 32 | yes | `0.496x` | `0.543` | `1.58` | `5.30` |
| `code_cpp` Qwen thinking-off | 32 | yes | `0.312x` | `0.211` | `0.60` | `5.13` |

Artifact:
[`2026-05-24-hipengine-mtp-thinking-render-w7900-diagnostic.json`](../benchmarks/results/2026-05-24-hipengine-mtp-thinking-render-w7900-diagnostic.json).

Conclusion for the current packed A3B+MTP artifact: explicit Qwen
thinking-off is **not** an MTP acceptance win on these code prompts.  It lowers
acceptance on exact rows and can trigger a special-token exact-AR mismatch on
the D64 C++ chat-template row.  This differs from HipFire's dense 27B DFlash
thinking-off anchor and reinforces that prompt-mode policy must be measured per
proposal family and model pair.  The safe MTP default for now is still raw prompt
text for llama.cpp-suite comparability, with chat-template thinking-on preferred
over thinking-off when chat rendering is explicitly tested.

### Bookkeeping / housekeeping gaps

| # | Item | Lineage | Gating condition / when to revisit |
|---|---|---|---|
| M14.book.1 | Enumerate every `runtime.memset_async` and `memcpy_async` call reachable from `_iterate_verify_chain_layers` and close the ~93-launch gap between the M13.A static audit (957–959/pass) and the M12.6c rocprof (1052/pass). Most likely candidates: per-layer `_memset_tensor` calls in MoE/router/grouped scratch resets + commit-table H2D pointer-table refreshes in `_commit_bulk_linear_states` fast path. | M13.A | Any time we want the static and measured launch counts to reconcile; mandatory before claiming a definitive launch-count target for M13.D. |
| M14.book.2 | Validate M13.B.0 (`out=next_hidden` / `out=row_out` write-through) on the **verify_tree** path with a real DDTree drafter, not just chain. The code already wires it through `_run_full_attention_tree_batched` and `run_linear_attention_moe_tree_tloop_layer_fp16` but only chain modes were exercised on the W7900 smoke. | M13.B.0 | When a DDTree-shaped MTP benchmark is added or when DDTree work resumes from `docs/DFLASH.md`. |
| M14.book.3 | BF16 siblings of the M13.B.0 `out=` forwarding: `run_full_attention_moe_c1_layer_bf16`, `run_linear_attention_moe_*_bf16`, `run_dense_mlp_residual_bf16`, `shared_expert_down_combine_residual_bf16`, and the corresponding `state.run_moe_c1_bf16(...)` call sites in the verifier orchestrator. Today the verifier path is FP16-only, so this is a consistency item, not a hot-path fix. | M13.B.0 | When/if a BF16 verifier (proposer-only or alternate quant) re-enters the picture. |
| M14.book.4 | The FP16 prefill/varlen helpers also need `out=` forwarding for consistency: `run_full_attention_moe_prefill_layer_fp16`, `run_full_attention_moe_prefill_varlen_layer_fp16`, `run_linear_attention_moe_c1_layer_fp16` (tokens=1 helper), `run_linear_attention_moe_packed_prefill_layer_fp16`. Same pattern as M13.B.0; not on the verifier hot path. | M13.B.0 | When prefill-mode launch counts become a measured bottleneck (currently overshadowed by verifier cycle). |
| M14.book.5 | `chain_attn_mode='c1_loop'` 9-prompt suite at `max-tokens=64` fails exact-AR on one prompt (the long-context prompt): MTP gets stuck in a `token 248045` repeat loop after ~56 generated tokens, while the AR baseline continues with `846, 198, 8917, ...`.  Confirmed pre-existing at HEAD (predates M13.B.0 — also fails before this session's changes).  Short-prompt smokes (`scripts/mtp_chain_e2e_smoke.py --decode-tokens 8`) pass exact-AR identically in c1_loop and batched, so this is a long-context numeric-drift issue specific to the c1_loop path. The hot path is `batched`, so this does not block M13.D; tracked here for future debugging. | M13.D measurement (incidental c1_loop suite failure at 64 tokens) | When/if c1_loop becomes a measured path again (e.g. for adaptive `B` selection at small B). |

### Kernel fusion candidates (post-M13.B.1 cost-model lesson)

M13.B.1 surfaced a hard rule: **a fused rotate+GEMV kernel that re-does
the full LDS rotation in every `(out_pack, row)` block multiplies
rotation work by ~`out_packs × row_replication_per_x_row`**.  For the
verifier shape this is ~1500×, which dominates the kernel budget.  The
right structural design is HBM-staged (rotate once per x_row, write to
HBM, GEMV blocks read back).  Apply this rule before greenlighting any
further rotate+GEMV fuses.

| # | Item | Lineage | Gating condition / when to revisit |
|---|---|---|---|
| M14.fuse.1 | Properly-staged selected-MoE gate/up rotate+GEMV (HBM staging, rotate once per x_row, atomic barrier, GEMV reads back). Replaces the M13.B.1 default-off kernel. Should land both **−40 launches/pass and ≤ baseline kernel time**. Numerically bit-exact via the same scalar_t HBM write the existing `gemv_awq_dual_pack8_transposed_rotate_staged_kernel` (non-selected) already uses. **This is M13.B.3 in the M13 tracker; copied here for visibility.** | M13.B.1 | **Attempted/rejected 2026-06-03:** staged gate/up is exact and removes launch count, but kernel time regresses (`15.344 -> 15.611 ms/pass` in the B=3/decode4 verifier window). Keep `HIPENGINE_SELECTED_MOE_STAGED_ROTATE=0`; revisit only with a different cost model (not another global-spin staged kernel). |
| M14.fuse.2 | Down-side fuse: `silu_mul_dual_rotate_out + selected_pack8_gemv` (the down projection). Same launch-count savings as M13.B.1 (~30 launches/pass for the silu+rotate kernel that currently runs before the down GEMV). Has the SAME cost-model risk as M13.B.1 because the down GEMV is also large-grid. Must use HBM staging from day one; never use an in-LDS-only design. | M13.B.1 lesson + M13 framing | **Superseded/no-held 2026-06-11/12:** HBM-staged/keyed down-side kernel was a 2026-06-03 verifier-window win (`1019.00 -> 989.33` calls/pass, kernel `15.369 -> 14.949 ms`) and remains useful for bisection, but current graph-auto MTP pays capture-safe barrier/fill overhead. The exact D32 9-prompt suite is faster with the unfused fallback: actual `0.6699x -> 0.6784x`, cycle `27.648 -> 27.408 ms/cycle`, verify `22.377 -> 22.131 ms/cycle`, identical acceptance. After graph-off became current best, the `decode_batched + graph_off + skip + HIPENGINE_SELECTED_MOE_DOWN_STAGED=1` compound also stayed exact but regressed ratio `0.8252x -> 0.8204x`, cycle `21.661 -> 21.763 ms/cycle`, and verify `16.511 -> 16.628 ms/cycle`. `HIPENGINE_SELECTED_MOE_DOWN_STAGED=1` is opt-in only. |
| M14.fuse.3 | Shared-expert rotate+pack8 GEMV fold (non-selected). Currently `paro_rotate1_fp16` + `gemv_awq_pack8_transposed_fp16` for the shared-expert gate/up/down rotations consume ~30 paro_rotate launches/pass at the verifier. Check whether the existing `gemv_awq_dual_pack8_transposed_rotate_staged_kernel` (already HBM-staged) covers this shape; if so, expose a single-output sibling and wire it. | M13.A audit (rotate launches in shared expert), M13.B.2 placeholder | Post-M13.D, evaluate against the lesson: shape is `out_packs × 1 row` (no expert lane replication), so the cost model is friendlier than M13.B.1 was. Likely safe to fuse with HBM staging — but only if M14.fuse.barrier ships first, otherwise the per-launch `hipMemsetAsync` will swallow the saving (see M13.B.2 measurement). |
| M14.fuse.barrier | Keyed/persistent barrier for the `gemv_awq_dual_pack8_transposed_rotate_staged_kernel` so the launcher does not need to `hipMemsetAsync(barrier, 0, 8)` on every call. Two viable designs: (1) keyed barrier — host passes a monotonically increasing `barrier_value` int32 per launch; the rotate phase compares `barrier[0] - prior_offset == rotate_blocks * rows`; the GEMV phase spins until `barrier[1] >= barrier_value`. (2) double-buffered barrier slots (idx flips per launch). Both eliminate the +N runtime memset launches that swallowed the M13.B.2 dispatch savings. | M13.B.2 measurement (net -10 paro_rotate vs +10 memsets = 0 net launches) | **Kernel/plumbing landed default-off 2026-06-02:** keyed BF16/FP16 externs plus shared-expert FP16 opt-in path initialize each barrier once and pass cumulative count/epoch values. W7900/gfx1100 MTP reload + persistent smokes pass exact-AR at B=3/decode4. Short rocprof: calls/pass `1019.00 -> 1009.33` (-9.67), `moe_paro_rotate_in` `190 -> 180` calls/pass, but kernel time/pass `15.350 -> 15.365` (+0.015 ms). DFlash chain E2E one-prompt smoke also passed exactness (`all_exact_match_ar=true`) with the local 35B DFlash drafter, but was diagnostic-only speed-wise. Keep default-off; useful as prerequisite, not a promoted speed path. |
| M14.dispatch.1 | C-side per-MoE-layer dispatcher (originally M13.C). Implementation: `kernels/hip_gfx1100/dispatch/moe_c1_dispatch.hip` plus the `hipengine.runtime.moe_c1_dispatch` ctypes bridge, now default-on as `HIPENGINE_MOE_C1_C_DISPATCH`. It collapses Python/ctypes entry points around the MoE subgraph but still launches the same underlying kernels. | M13.C measurement, M14.dispatch.1 implementation, M16.2 launch-cost probe | **Landed but not a speed lever by itself.** The measured C-dispatch row was parity (`cycle_cost off=3.707`, `on=3.696`), and M16.2 later proved a native loop that reissues the same grids cannot remove the GPU workgroup-scheduling launch residual. Treat this as infrastructure for reduced-DAG work, not a template for a full-layer C-only rewrite. |
| M14.fuse.4 | Router op fusion: `route_moe_topk_shared_fp16` is currently 1 small launch per MoE layer (40/pass) but the underlying op chain (logits GEMV + softmax + top-k + gather) could be a single composite. Vulkan/llama.cpp has `MUL_MAT_ID_MUL` as their router fold. Earlier hipEngine D1.5 cooperative-router probe was rejected (`OPTIMIZE.md`), but post-M13 the topology may be different. | M13.A audit (router family at 80 calls/pass after counting paroquant variants) | Only if total post-M13.C launch count is still > 2× llama.cpp shape. |
| M14.fuse.5 | Selected-expert *grouped* GEMV (sort tokens by expert before the GEMV) so each expert tile is read once from HBM regardless of B. This is the llama.cpp `mmid` slow-path pattern for tokens>8/expert. At B=3, top_k=8, 4× expert lanes per token = some expert collisions, so the savings would come from weight reuse, not just from fewer launches. **This is M13.E in the M13 tracker; copied here for visibility.** | M13 framing + WORKLOG 2026-05-22 ("useful MoE path still needs real expert grouping/sorting") | Only if M13.D leaves ≥10% on the table AND a profile shows MoE GEMV weight bandwidth is the bottleneck (not dispatch). |

### Numerics / contracts to track

| # | Item | Lineage | Gating condition |
|---|---|---|---|
| M14.num.1 | The M12.6 W4 multi-row pack8 site mask defaults to `full_qk,linear_qkv_z,dense_gate_up,single_full_o,single_shared_down,single_dense_down,single_linear_out,single_full_v`. `single_linear_out` and `single_full_v` were promoted on 2026-06-11 after current D32 9-prompt exact gates; `single_linear_out` also restored the no-env `translation` prompt. `shared_gate_up` is superseded by the M16.4 split-output output-tiled route. | M12.6 (pre-M13) + 2026-06-11 refresh | Keep `HIPENGINE_W4_MULTI_ROW_PACK8_SITES=all` experimental unless any future site has exact current-stack prompt-suite evidence and retained profile evidence. |
| M14.num.2 | The fused-rotate Option-C LDS round-trip kernel body change in `gemv_awq_selected_dual_pack8_*_rotate_out_kernel` is currently dormant (no production caller). If a future code path turns it on (e.g. M14.fuse.1's staged variant reuses parts of the same kernel template), confirm the LDS round-trip is still desired vs the original "higher-precision" Option-A behavior. | M13.B.1 | When M14.fuse.1 starts; pick the contract explicitly. |
| M14.num.3 | If the proposer ever uses a different selected-MoE GEMV kernel than the verifier (e.g. a CUDA-only or dense-BF16 variant for MTP weights), exact-AR will break in subtle prompt-dependent ways. The verifier and proposer MUST share kernel symbols on the MoE selected path. | M13 framing | Any time `hipengine/speculative/mtp_native.py` changes its dispatch. |

### Out of scope for M14, but worth noting

| # | Item | Lineage | Why deferred |
|---|---|---|---|
| M14.oos.1 | Tree-shaped MTP drafts (DDTree). `docs/DFLASH.md` covers this; the verifier infrastructure already supports `verify_tree` mode but no MTP drafter produces tree drafts. | Pre-M13, mentioned in M12 plan | DDTree is a separate axis; not part of MTP cycle-cost work. |
| M14.oos.2 | Long-context tuning. The 9-prompt suite is short-prompt-dominated. | M12 framing | Get the short-prompt cycle-cost row first. |
| M14.oos.3 | Quantizing MTP weights (proposer BF16 → W4 or AWQ). | M12 framing | Not required for cycle-cost ≤ 2.0; revisit if MoE bandwidth becomes the new bottleneck. |
| M14.oos.4 | Adaptive B / fallback policy (M12.8). | M12 tracker | Sequenced after `C_3 < 2.0` is reached. |
| M14.oos.5 | Cross-arch ports (CUDA / gfx1151) of any new MoE kernels. | M12 framing | Land on gfx1100 first, get a retained row, then port. |

All M14 items are explicitly *not blocking* the M13 path to a retained
row.  They are the inventory of "things we knew about but did not need to
act on yet".  Revisit at the M13 closure (post-M13.D) to decide which
become the next named phase.

## M15 — the launch-submission wall (2026-06-08, current active framing)

This is the controlling diagnosis after tasks #28–#31. It supersedes the
"shave a kernel family" framing of M7–M14: those micro-fusions each move
`C_3` by <5% (inside run-to-run noise) because the verifier is no longer
kernel-bound — it is **launch-submission bound**.

### The measurement that reframes everything

W7900/gfx1100, MTP B=3 verifier window, 11 steady passes
(`benchmarks/results/2026-06-07-hipengine-mtp-verifier-rocprof-family-rollup.json`,
task #29):

| Quantity | Value |
|---|---:|
| Host verify window | **37.88 ms/pass** |
| Summed kernel time | 18.46 ms/pass (48.7% of window) |
| **Host residual** (window − kernel) | **19.42 ms/pass (51.3%)** |
| Launches/pass | **971** |
| D2H syncs/pass | 7 (compact GPU-accept payload only) |
| Implied host cost per launch | (37.88 − 18.46) / 971 ≈ **20 µs/launch** |

The host residual is **not** Python/ctypes and **not** D2H sync:

1. **Not Python/ctypes.** The C-side MoE dispatcher (`moe_c1_dispatch.hip`,
   `HIPENGINE_MOE_C1_C_DISPATCH=1`, default-on) collapses ~13 Python calls/layer
   into one C call and was measured **parity** (M14.dispatch.1: cycle_cost
   `off=3.707`, `on=3.696`). Removing the Python loop did not move the wall.
2. **Not D2H sync.** Only 7 tiny scalar reads/pass (the GPU-accept payload);
   the 40-layer forward has no host sync inside it — it is pure async enqueue.
3. **Not graph-fixable on ROCm 7.x.** M12.1 and M13.D both measured HIP graph
   replay as neutral-to-worse at >900 launches: `hipGraphLaunch` walks every node
   and submits it to the GPU queue at ~the same per-node cost as direct dispatch.
   CUDA graph node replay is ~1–2 µs/node; ROCm 7.x is ~the direct-launch cost.

Conclusion: the residual is the **GPU command-queue submission cost of 971
separate kernels** (~20 µs each), which is fundamental to having 971 kernels and
is **not** removed by dispatcher batching or graph replay on this platform.

### Why AR escapes the wall and the verifier does not

AR decode runs at 111 tok/s ≈ **8.94 ms/token** through the same 40 layers.
971 launches × 20 µs = 19.4 ms could not fit in 8.94 ms — so AR is **not** on a
971-launch path. AR's c=1 path uses the fused *decode* kernels
(`run_linear_attention_out_proj_fp16`, the fused-rotate `tokens==1` GEMV path,
graph replay) and issues far fewer launches. The verifier's tokens>1 chain
(`run_linear_attention_moe_chain_tloop_layer_fp16`, the M7.C.6 split-single /
`awq_fusedw4_prefill_*` projections) is a **separate, less-fused code path** that
never inherited AR's decode-kernel consolidation. Task #31 (`decode_batched`)
was the first step of closing that gap, but only for the 10 full-attention
layers; the 30 linear-attention layers + MoE surround still run the heavy
tokens>1 shapes.

### Launch budget — where the 971 launches live (task #29 rollup)

| Family | launches/pass | kernel ms/pass | Notes |
|---|---:|---:|---|
| selected_moe (router/gate_up/down/silu/combine + 190 `paro_rotate`) | 400 | 4.98 | already `mul_mat_id`; fat is the surround |
| shared_dense_w4 projection bucket | 300 | 6.58 | largest **kernel** bucket; unsafe W4 sites still prefill-style |
| rmsnorm_misc + format/split | 173 | 0.52 | tiny elementwise; producer-fusion candidates |
| linear_conv_gdn | 60 | 2.73 | already chain t-loop |
| full_attn attention+KV | 20 | 0.45 | already cut by task #31 (`decode_batched`) |
| lm_head_top1 | 3 | 1.45 | M12.2 weight-shared |
| commit/accept | 15 | 0.29 | GPU-resident |

### The only path to `C_3 ≤ 2.0`

`C_3 ≤ 2.0` means cycle wall ≤ 2.0 × 8.94 ≈ **17.9 ms**. We are at ~45 ms
(`decode_batched`). With proposer ~4 ms + overhead ~4 ms, the verify window must
reach ≈10 ms. At a ~20 µs/launch floor, **even zero kernel time caps the
launch budget at ≈500**; with realistic kernel time the budget is lower. So the
verifier must drop from **971 → ≈300–400 launches/pass** *and* keep kernel time
≈flat. Removing 30–80 launches at a time is within noise and must be **batched**
into one measured landing, not committed piecemeal.

The launch count must fall by ~2.5–3×. That requires consolidating the per-layer
surround (≈24 launches/layer → ≈8), in the priority order the budget implies:

1. **M15.1 — small-batch decode-shaped linear-attention layer** (mirror task #31
   for the 30 linear layers): fuse the projection surround (rmsnorm→rotate→QKV/Z,
   AB, out-proj) into decode-shaped kernels that read each weight stream once for
   all `B+1` rows, the way AR's c=1 path already does. Largest single target:
   720 of 971 launches/pass live in linear-attention layers.

   - **M15.1a landed (2026-06-08, default-on).** The small-batch (`2<=tokens<=7`)
     linear QKV/Z (30 layers) and full Q/K (10 layers) projections were
     re-pointed from per-row single GEMVs (`grid=(out_pack,row)`, weight
     re-streamed per row) to the bit-exact weight-amortized
     `gemv_awq_pack8_multi_row_decode_transposed_fp16` (weight tile read once for
     all `B+1` rows) behind `HIPENGINE_W4_MULTI_ROW_SMALL_BATCH`. W7900 B=3
     rocprof: `w4_single_gemv` family `2.460 -> 1.873 ms/pass` (-23.9%), total
     kernel `17.05 -> 16.33 ms/pass` (-4.2%), launches/pass unchanged (981);
     `verify_ms/cycle` min `-2.4%` (B=3) / `-3.8%` (B=5), exact AR.
     Confirms the weight-amortization thesis (the win scales with row count) but
     is modest because these projections are only ~14% of verifier kernel time.
     Artifact: `benchmarks/results/2026-06-08-hipengine-mtp-m15.1-verifier-projection-multirow.json`.
   - **Remaining M15.1 work:** the larger projections still on per-row/prefill
     paths — linear `out_proj`, shared-expert gate/up/down, full-attn V — are the
     unsafe W4 sites and overlap M15.2. The rotate/rmsnorm surround and the
     M7.C.6 two-single→one-dual merge are the launch-count half (M15.3).
2. **M15.2 — exact small-batch W4 multi-row pack8 GEMV for the historically unsafe sites**
   (`single_linear_out` and `single_full_v` promoted 2026-06-11;
   `shared_gate_up` superseded by M16.4 split-output; M14.num.1). This is
   both the largest *kernel* bucket (6.58 ms) and a launch source. **Blocked by
   argmax fragility, not by kernel numerics (2026-06-08).** A bit-exact multi-row
   Marlin-K GEMV now exists (`gemv_paro_marlin_k_fma_multi_row_fp16`,
   `tests/test_qwen35_paro_marlin_k_multi_row.py`: each row is byte-identical to
   single-row Marlin-K, which is what AR uses at rows==1 for these sites). Even
   so, historical routing of `single_full_v` / `single_linear_out` through it
   flipped top-1 on the fragile `translation` / `summarize` prompts at 64 tokens,
   while the
   baseline (`awq_fusedw4_prefill_fp16`) stays exact. Root cause: the verifier's
   accumulated-chain input to these projections already differs slightly from
   AR's rows==1 input, and the argmax in degenerate/repetitive regions (the
   `248045` repeat loop) is on a knife's edge — so *any* numerics change can tip
   it, even one that matches AR's dequant formula. The default `prefill` path is
   exact only by coincidence on these prompts. Exact weight-amortization of these
   sites therefore needs the verifier's *input* to match AR bit-for-bit (a
   chain-state problem), not just a better projection kernel. The kernel +
   gate are landed default-off (`HIPENGINE_MARLIN_K_MULTI_ROW_SITES`, empty
   default site set) as infrastructure and a reproducible fragility probe.

   Side discovery: `chain_attn_mode=decode_batched` is **not** exact-AR on the
   full 9-prompt suite at 64 tokens — the baseline (no env) already fails
   `translation` (early token-6 reorder + `248045` repeat). `batched` is exact
   9/9. This extends M14.book.5 (the c1_loop 64-token drift) to decode_batched
   and should be tracked before decode_batched is used for suite-wide exactness.
3. **M15.3 — producer-side rotate/format fusion** (fold `paro_rotate` into the
   producing RMSNorm/SiLU/add, never into the GEMV consumer — that is the
   M13.B.1 redundancy trap). Targets the 190 `paro_rotate` + 173 format/misc
   launches. Cheap per-launch, exact-safe, but only worth landing batched with
   M15.1.
4. **M15.4 — re-evaluate graph replay** only after 1–3 drop the count below
   ~400, where ROCm per-node submission may finally beat direct dispatch (task
   #34's precondition).

Discipline: M15.1–M15.3 are decode-kernel R&D and belong in `~/amd-gpu-tuning/`
first (per `AGENTS.md`), then port stable kernels here behind a flag with a
bit-exact RED test and an MTP+DFlash exact-AR smoke. No retained row until the
*batched* launch reduction moves `C_3` outside noise with exact AR preserved.

DFlash shares this wall: its chain verifier uses the same tokens>1 trunk path,
so M15.1/M15.2 land for both providers (task #35). The 27B-dense profile-route
`1.350×` row is real but offline/non-deployable; it does not change the
launch-submission diagnosis for the online exact path.

### Exactness policy — exact default vs approximate tier (2026-06-08)

M15.2 proved the M12.6 "unsafe" sites can be argmax-fragile *independent of
dequant numerics*: even a multi-row kernel that is byte-identical to AR's
single-row Marlin-K flipped top-1 on `translation`/`summarize`, because the verifier's
accumulated-chain input already differs slightly from AR's rows==1 input and the
argmax in degenerate regions is knife-edge. That raises the obvious question:
should the default verifier even require bit-exact AR equality?

Position for the exact default path:

1. **A flip changes the output, and it cascades.** Greedy commits the verifier's
   top-1, so one flip at token *i* makes tokens *i..N* diverge from baseline. The
   flips are not only in degenerate tails (`translation` flips at token 6, mid
   content). The flipped token is *not lower quality* — both picks are within
   float noise of the true logits — but it is no longer the baseline's output.
2. **Relaxing the default buys little right now.** The verifier is
   launch-submission bound (≈51% host residual); weight-amortizing the unsafe W4
   bucket is kernel-time only and, even fully realized, leaves `C_3 ≈ 4.7` —
   still launch-bound and still below AR. Relaxing exact-AR does not unlock the
   launch wall, graph replay, or beating AR. So it would forfeit the one
   guarantee that differentiates us (vs e.g. hipfire's non-exact DFlash; see the
   2026-06-02 audit) for a non-decisive gain.
3. **Approximate belongs in an explicit opt-in tier, not a relaxed default.**
   That tier is task #36 (deeper-row acceptance despite mismatch, record first
   divergence, `performance_claim=false`, never default), gated by *task-level
   quality* (pass@k / perplexity parity over a real eval), not by exact-AR.

Note: main already carries some relaxed-acceptance semantics in places; this is
recorded here as policy intent (default = exact-AR; approximate = opt-in, quality
-gated), and reconciling main's existing relaxations with this tiering is a
follow-up, not part of the current grind.

**Working decision for the current push:** do not relax the default to chase the
unsafe-W4 bucket, and do not spend cycles measuring approximate quality yet.
Get the *exact* path to beat AR first by grinding down the launch count and
verifier shape (M15.1/M15.3); revisit the approximate tier only once exact MTP/
DFlash beats AR and we want a higher-speed opt-in.

### Top-k oracle — the MTP head is ready; C_B is the wall (2026-06-08)

To decide whether "more tokens/cycle" (acceptance) can push MTP positive, we
measured the **top-k oracle**: instrument the proposer to emit the per-depth
vocab top-k (`topk_f32_rows_i32` over the lm-head logits) and, in
`scripts/mtp_chain_e2e_smoke.py`, record the rank of the target's chosen depth-1
token (`verify.target_top1[0]`, populated with `HIPENGINE_VERIFY_GPU_ACCEPT=validate`)
in the MTP head's top-8. Artifact:
`benchmarks/results/2026-06-08-hipengine-mtp-topk-oracle.json`.

W7900, quicksort, B=3, exact D32 (11 cycles): **depth-1 top-1 = 82%, in-top-8 =
100%, rescuable (rank 2..8) = 18%, unrescuable = 0%.** D64 (24 cycles) is
consistent: in-top-8 = 96%, top-1 = 79%.

Interpretation — this is the decisive economics finding:

1. **The MTP head is excellent and acceptance is realizable.** The target's
   greedy token is essentially always in the head's depth-1 top-8, so a
   root-branching tree could drive depth-1 acceptance to ~100% and rescue the
   ~18% zero-accept cycles. The model is *not* the blocker. This is why the same
   model class beats AR on llama.cpp/hipfire.
2. **But the perfect-accept ceiling is < 1.0 at B≤5 because C_B exceeds B+1.**
   Measured C_B (decode_batched, sublinear after M15.1/M15.3): B1 4.25, B2 4.53,
   B3 4.89, B5 6.14 → ceilings 0.47 / 0.66 / 0.82 / 0.98. The ~4.25 floor at B=1
   is the **ROCm launch-submission cost**, not model quality. llama.cpp/hipfire
   run at C_B≈2 (CUDA dispatch/graph replay ~10× cheaper per launch), so the same
   acceptance yields >1.0×; we cannot, because our C_B is ~2× theirs.
3. **Therefore MTP-positive is C_B-bound, not acceptance-bound.** Tree drafts are
   necessary-but-insufficient: they realize the head's quality but cannot close a
   ~2× dispatch gap. At B≥7 the ceiling does cross 1.0 (C_B sublinear), but the
   required acceptance (~84%) exceeds the single-layer head's deep-draft horizon
   (avg_accepted saturates ~1.4–1.8).

**Decision:** stop chasing MTP-positive via tokens/cycle. The model is ready; the
wall is the ROCm dispatch floor (`C_B`), which only a megakernel / fewer-larger-
kernel rewrite (or a ROCm graph-replay improvement) can move — the hard problem
M15 already mapped. The tractable deployable >1.0× exact win is **DFlash
profile-routing**, which sidesteps per-cycle `C_B` by routing profitable prompts
to spec and the rest to AR (already 1.35× offline; see `DFLASH.md`). Build MTP
tree drafts only if/when `C_B` is cut to the ~2–3 range.

## M16 — the C_B ≤ 2 program (lower the dispatch floor)

The top-k oracle proved the MTP head is ready: at `C_B ≈ 2` this head + a
root-branch tree beats AR comfortably. Everything now hinges on cutting `C_B` from
~4.25 to ~2.0. **This is achievable on our hardware**: hipfire runs DFlash on the
same W7900/gfx1100 at multiples of AR (see `DFLASH.md` 2026-06-02), so the ~20 µs/
launch we measure is *our* cost, not a hard ROCm floor. **M16.1 + M16.2 (2026-06-08)
located that cost precisely:** the 1-block dispatch floor is ~5.6 µs/launch, but
per-launch cost **scales with grid size** (GPU workgroup scheduling) — real hot
kernels launch thousands of blocks, which accounts for the ~20 µs/op residual. It is
**not** host/Python (M14.dispatch.1 removed Python → parity), **not** arg-marshaling,
and **not** graph-removable. So the lever is **fewer/larger kernels (M16.3)**, not a
native loop (M16.2, parity) or graphs (M16.5, neutral). llama.cpp's low `C_B` comes
from its kernel structure + low D2H/sync, which is the *shape* to copy.

### The C_B budget (what "≤2" requires)

```text
C_B = (launches * per_launch_us/1000 + kernel_ms_per_pass) / ar_ms_per_token
```

W7900 35B, B=3 (task #29): `ar_ms ≈ 9.0`, verify window **37.9 ms** =
**19.4 ms host residual** (≈20 µs × 941 launches) + **18.5 ms kernel**. `C_1 ≈ 4.25`
is almost all the residual + fixed overhead. For `C_3 ≤ 2.0` the verify window must
fall to **≤ 18 ms** — a ~2.1× cut. A concrete decomposition target:

| Term | now | target | how |
|---|---:|---:|---|
| host residual (launch submission) | 19.4 ms | **~5 ms** | **launch count → ~300 (M16.3 fewer/larger kernels)** is the only lever. M16.2 showed the per-launch residual is GPU workgroup-scheduling (scales with grid; not host/Python/args), so a native loop is *parity* and a graph is *neutral* (≤1.13×). The residual is ~941 launches × (5.6 µs floor + grid-dispatch); cutting launches is the only way down. |
| kernel time (B+1 rows) | 18.5 ms | **~12 ms** | finish weight-amortization + larger fused kernels |
| → verify window | 37.9 ms | **~17 ms** | → `C_3 ≈ 1.9` |

`C_B` is already **sublinear in B** after M15.1/M15.3, so once the floor drops, a
higher-B root-branch tree (the oracle's 100%-in-top-8 head) carries the win.

### What we have already learned (don't re-litigate)

- **Python/ctypes is *not* the bottleneck.** The C-side MoE dispatcher
  (M14.dispatch.1) was parity. The ~20 µs/launch is HIP runtime dispatch + queue
  submission + inter-kernel GPU idle, not marshaling.
- **Pairwise op fusion is occupancy/redundancy-walled.** Consumer-side rotate
  fusion (M13.B.1/.2) hit a redundant-work trap; producer-side (M15.4) hit an
  occupancy trap (one-block-per-row serializes the rotate). Op-pair fusion
  cannot reach ≤2; only *fewer, larger* kernels or graph replay can.
- **HIP graph replay is neutral vs a native launch loop — now confirmed in clean
  isolation (M16.1, 2026-06-08).** The earlier neutral verdict (M12.1, M13.D) was
  on the *full* verify path with bucket churn + validation; M16.1 re-tested it with
  trivial fixed-arg kernels, one stream, steady-state replay: graph replay and a
  direct C launch burst both hit **~5.6 µs/node** and a graph is **1.00×** vs the
  native loop at 941 nodes. The reason is now clear: the ~5.6 µs floor is GPU-side
  dispatch, which graphs cannot remove — graphs only remove *host* issue cost, and
  a tight native loop already does that. So graphs (M16.5) are **de-prioritized**;
  the lever is the native loop (M16.2) for the host residual and fewer nodes
  (M16.3) for the floor.

### Tracks (ordered cheapest-diagnostic first)

| # | Track | First concrete step / acceptance gate | Why it can reach ≤2 |
|---|---|---|---|
| **M16.1** | **Isolate the ROCm graph-replay per-node cost** ✅ **Done 2026-06-08 — see result below.** | `scripts/graph_node_microbench.py`: trivial fixed-arg one-block kernels issued back-to-back from a C loop (zero Python per-node), same burst captured into one graph + replayed in steady state; µs/node graph vs direct, N∈{50…2000}. | **Resolved (not 2–5 nor 20):** the clean per-node floor is **~5.6 µs/node** and a HIP graph is **1.00×** vs a native launch loop. Graphs don't beat a native loop (the floor is GPU dispatch). → program is **native loop (M16.2) + fewer/larger kernels (M16.3/M16.4); graphs (M16.5) de-prioritized.** |
| **M16.2** | ~~Native C++ verify hot loop~~ **(predicted PARITY by the M16.2 arg/grid probe — do NOT build) ❌ 2026-06-08** | Was: move the whole 40-layer verify forward into one native TU. | **Disproven as a lever.** The launch residual is GPU command-processor *workgroup-scheduling* (scales with grid size; ~12–20 µs/op for the real kernels' thousand-block grids), which is host-language-independent (M14.dispatch.1 removed Python → parity), arg-count-independent, and graph-neutral. A native loop calls the same `hipLaunchKernelGGL` with the same grids → it cannot reduce this. See the M16.2 result block. Pivot to M16.3. |
| **M16.3** | **Structural launch-count reduction (megakernels, not op-pairs)** | Fuse a whole MoE op (router→gate_up→silu→down→combine) and a whole attention block into single kernels, sized for the B+1 verifier rows. Gate: exact-AR (bit-exact RED) + launches/pass 941 → ≤ ~400 with flat-or-lower kernel time. | The only fusion class that beats the occupancy/redundancy walls. At ~300–400 launches even ROCm's current per-launch cost gives a ~6 ms residual. |
| **M16.4** | **Finish weight-amortization of the kernel half** | Known prefill-style W4 sites are now covered on the exact path: `shared_gate_up` by split-output output-tiled W4, and `single_linear_out` / `single_full_v` by the M12.6 default safe mask after exact 9-prompt evidence. Further work should come from new profile buckets, not re-testing these sites. Gate: exact-AR 9-prompt + kernel ms/pass down. | Drives kernel time toward ~flat-in-B, the other ~6 ms of the budget. |
| **M16.5** | ~~Re-enable graph buckets after M16.2/M16.3~~ **(de-prioritized by M16.1)** | Per `(B+1, kv_bucket)` capture/replay. Gate: ≥X% `C_B` improvement on an exact row. | **M16.1 showed a HIP graph is 1.00× vs a native launch loop**, so graphs add ~nothing over M16.2. Keep the capture infra landed/opt-in, but do not invest here for `C_B` — revisit only if a future ROCm exposes cheaper graph dispatch. |

### M16.1 result — the dispatch floor is ~5.6 µs/node and graphs are neutral (2026-06-08, measured)

`scripts/graph_node_microbench.py` on W7900/gfx1100 (require-cached build, 80 reps,
N∈{50,100,200,500,941,2000}; three independent runs agree within 0.1 µs). A
trivial one-block kernel (writes memory so it can't be elided) is issued back-to-
back `N` times from a **single C call** (zero Python/ctypes per-node overhead), and
the *same* C burst is captured into **one HIP graph** and replayed in steady state.

| N (nodes) | direct C burst µs/node | graph replay µs/node | graph vs direct |
|---:|---:|---:|---:|
| 50 | 6.07 | 5.79 | 1.05× |
| 200 | 5.75 | 5.65 | 1.02× |
| 941 | **5.61** | **5.61** | **1.00×** |
| 2000 | 5.60 | 5.61 | 1.00× |

**Findings.**
1. The clean per-node dispatch floor on gfx1100 is **~5.6 µs/node** (converged,
   stable 50→2000) — *CUDA-class*, not the ~20 µs the full path shows.
2. **A HIP graph does not beat a native launch loop (1.00× at 941).** The ~5.6 µs
   is GPU-side dispatch; graphs only remove *host* issue cost, which a tight native
   C loop already removes. This re-confirms M12.1/M13.D's "graph neutral" verdict in
   clean isolation and explains *why*.
3. **Residual decomposition — CORRECTED by the M16.2 probe (2026-06-08).** This
   block originally inferred the ~15 µs/op gap above the 1-block floor was "host-side
   per-op overhead a native loop removes." **That inference was wrong.** The M16.2
   follow-up (arg-count + grid-size probes, below) shows the gap is GPU
   *workgroup-scheduling* that scales with grid size — the 1-block micro-bench
   underestimated real-kernel dispatch because real hot kernels launch thousands of
   blocks. The gap is **not** Python (M14.dispatch.1 removed Python → parity) and
   **not** arg-marshaling. So a native verify loop (M16.2) does **not** collapse the
   residual; **fewer/larger kernels (M16.3)** is the lever. See the M16.2 result block.
4. Going **below** ~5.3 ms requires **fewer nodes** (M16.3 structural megakernels;
   each removed node saves ~5.6 µs) and lower kernel time (M16.4) — not graphs.

Artifact: `benchmarks/results/2026-06-08-hipengine-m16.1-graph-node-replay-microbench.json`.

### M16.2 result — the launch residual is GPU workgroup-scheduling, so a native loop is PARITY (2026-06-08, measured)

M16.2 was "build a native C++ verify loop to remove host/Python per-op overhead."
Before building a multi-thousand-line 40-layer C forward, the load-bearing premise
was tested cheaply by extending `scripts/graph_node_microbench.py` with (a) a
16-arg kernel (`--kernels tiny,wide`) and (b) a forced grid-size sweep
(`--grid-sweep`). W7900/gfx1100, require-cached, 80 reps. **The premise was disproven.**

**Arg count is irrelevant.** A 2-arg vs 16-arg kernel are identical: direct `5.62`
vs `5.62` µs/launch at N=941, graph `1.00×` for both. So the per-launch cost is not
argument marshaling.

**Per-launch cost scales with GRID SIZE (workgroup count), and graphs barely help:**

| grid blocks | direct µs/launch | graph µs/launch | graph vs direct |
|---:|---:|---:|---:|
| 1 | 5.63 | 5.61 | 1.00× |
| 128 | 5.62 | 5.61 | 1.00× |
| 1024 | 7.27 | 6.47 | 1.12× |
| 2048 | 7.97 | 7.08 | 1.13× |
| 8192 | 12.19 | 11.28 | 1.08× |
| 65536 | 51.40 | 50.48 | 1.02× |

The hot W4 GEMV launches `dim3(out_packed, rows)` = thousands of blocks (e.g.
`512 × 4 = 2048` for a hidden-4096 B+1 projection; more for gate_up). At ~2048
blocks the dispatch is ~8 µs/launch, at ~8192 ~12 µs — exactly matching the real
path's ~12–20 µs/op residual. So **the launch residual is GPU command-processor
workgroup scheduling**, exposed because the trivial kernels drain instantly.

**Why M16.2 (native loop) is PARITY** — three independent lines of evidence:
1. **M14.dispatch.1** already bundled ~13 ctypes calls into 1 per MoE layer
   (removed the Python boundary for ~480 launches/pass) and measured **parity**
   (cycle_cost off `3.707` / on `3.696`). Removing Python doesn't help.
2. **Arg count is irrelevant** (above) — so it isn't ctypes/marshaling either.
3. **The cost scales with grid and is graph-neutral** (above) — it is GPU-side
   workgroup dispatch. A native loop issues the **same** `hipLaunchKernelGGL` with
   the **same** grids, so it pays the **same** dispatch. Predicted parity, and not
   worth the 40-layer exact-AR rewrite risk.

**Consequence for the M16 program.** The only lever for the launch residual is
**M16.3 (fewer, larger kernels)**: each fused kernel removes a full dispatch
payment, and more compute-per-launch lets the GPU hide dispatch behind compute.
Graphs (M16.5) stay de-prioritized (≤1.13× even at large grids). M16.2 is closed
as predicted-parity. Artifact:
`benchmarks/results/2026-06-08-hipengine-m16.2-launch-cost-arg-grid-scaling.json`.

### M16.3 progress — census + staged-rotate confirmed-negative (2026-06-09, measured)

> **Implementation spec for the M16.3 megakernel campaign lives in
> [docs/MEGAKERNEL.md](MEGAKERNEL.md)** (target kernel, the T1 self-consistent +
> KL accuracy strategy that drops bit-exact-vs-legacy, the GGUF-first
> simplification, and the staged RED-first build plan). This block is the
> economics/measurement record.


Fresh launch census of the **batched B=3 verifier** (the economics path), W7900/
gfx1100: **931 launches/pass, 15.97 ms kernel/pass.** No single family dominates
(>78/pass); launch count is spread ~1/layer across ~9 families. Biggest buckets:
`paro_rotate` (1+2+3) ~146/pass, `router` (logits+select) 77, `rmsnorm`
(norm+add) 78, `copyBuffer` D2D 55, selected/shared GEMVs ~38 each. The old
120/pass `runtime_memset` is **gone** (fillBufferAligned 0.6/pass — already
eliminated; do not chase). Artifact
`benchmarks/results/2026-06-09-hipengine-m16.3-launch-census-batched-b3.json`.

**Cheapest-first, the existing bit-exact staged-rotate flags were re-measured on
the current tree (post M15.x/M16.4) and stay default-OFF — they REGRESS `C_B`:**

| config | `C_B` (B=3) | exact | launches removed |
|---|---:|---|---|
| baseline (default) | **4.67** | ✓ | — |
| `SHARED_EXPERT_FUSED_ROTATE=1` | 5.13 | ✓ | ~68/pass (rotate2) |
| `+ SELECTED_MOE_STAGED_ROTATE=1` | 5.06 | ✓ | ~146/pass (rotate1+2) |

This is **empirical proof on the current tree** that op-pair *staging* fusion
cannot reach `C_B ≤ 2`: removing ~68–146 **small-grid** rotate launches via
HBM-staged keyed-barrier kernels saves only the ~5.6 µs/launch dispatch floor
(M16.1) per launch, which is **smaller** than the barrier-spin + staged HBM
round-trip the staged kernel adds. Consistent with M13.B.1/M15.4. Artifact
`benchmarks/results/2026-06-09-hipengine-m16.3-staged-rotate-recheck.json`.

**Consequence — the first true megakernel must consolidate REAL work + HBM
intermediate traffic across big-grid GEMVs, not move small-grid plumbing behind
a barrier.** The flagship target is the **selected-expert FFN**:
fuse `gate_up → silu → down → combine` into one kernel where each block carries
one `(token, expert)` pair, keeping the 512-d intermediate on-chip so the
gate_up-output write + down-input read vanish and ~3 big-grid launches/layer
collapse to 1 (~114 launches/pass + the intermediate HBM traffic). Removing
small-grid launches (rotate/rmsnorm/router) is at best neutral, at worst
negative, so it is **de-prioritized** as a standalone lever.

### Sequencing logic

1. **M16.1 + M16.2 are done (2026-06-08) and resolved the strategy** — see both
   result blocks below. M16.1 found the clean 1-block floor is **~5.6 µs/node** and a
   HIP graph is **neutral** (1.00×). M16.2 then probed arg count + grid size and
   **corrected M16.1's over-optimistic native-loop read**: per-launch cost is
   arg-independent but **scales with grid size** (GPU workgroup scheduling), so the
   real ~20 µs/op residual is command-processor-bound, not host/Python. With
   M14.dispatch.1's Python-removal parity, that means **a native loop (M16.2) is
   PARITY and graphs (M16.5) are neutral**. The program is therefore **M16.3
   (fewer/larger kernels) FIRST** as the sole launch-residual lever, then **M16.4**
   for kernel time. M16.2 and M16.5 are closed/de-prioritized.
2. hipfire's low `C_B` on gfx1100 therefore comes from **fewer/larger kernels +
   less D2H/sync**, not from a native hot loop *per se* (the loop language is
   parity here) — copy its kernel structure, not the assumption that re-hosting the
   loop in C/C++ is itself the win. Do not copy its non-exact acceptance policy
   (see the 2026-06-02 audit).
3. Every track keeps the exact-AR gate; `C_B` improvements are only retained with a
   recorded economics artifact and the bit-exact RED tests that M15 established.

## Closing the gap with llama.cpp MTP — kernel roadmap (2026-05-21)

Historical note: this section captured the pre-M7.C.6 kernel-family roadmap.
It remains useful as a cost catalog, but M12 supersedes it as the active plan.
The question is now sharper: why is llama.cpp's verifier cycle close to ~2
AR-token equivalents while ours costs 3.2–6.9 AR-token equivalents? Treat M7/M8/M9/M10
items below as candidate M12 sub-primitives, not as the primary roadmap.

### Architectural lessons from llama.cpp (`/home/lhl/llama.cpp/llama.cpp-hip`)

llama.cpp MTP for Qwen3.5/3.6 has the same model boundary we already model
([qwen35moe.cpp `graph_mtp`](../../../llama.cpp/llama.cpp-hip/src/models/qwen35moe.cpp)
lines ~555–727) and the same draft/verify dance
([common/speculative.cpp `common_speculative_impl_draft_mtp`](../../../llama.cpp/llama.cpp-hip/common/speculative.cpp)
lines ~409–775). What it has that we don’t:

1. **One `mul_mat_id` kernel per MoE op.** Their `ggml_cuda_mul_mat_vec_f`
   (and `mul_mat_vec_q` for quantized weights) launches a single 3D grid
   `(rows_in_dst, n_expert_used, n_tokens)` and reads `ids[token, slot]` inside
   the kernel to pick which expert tile to GEMV against. There is no
   host-side dispatch loop; all 60 expert GEMVs at B=3 collapse to **1 launch
   per (layer, MoE op)** instead of our 60 tiny launches per pass.
   References: `ggml/src/ggml-cuda/mmvf.cu` (kernel), `ggml/src/ggml-cuda/mmid.cu`
   (the slow-path token-sort helper used only when `n_tokens > 8` per expert).
   The fast path is mmvf/mmvq with the ids tensor itself — no sort, no scatter.
2. **Fused gate+up matmul.** `gate_up_proj` is a single expert weight stack;
   `build_moe_ffn` issues one `mul_mat_id` for gate_up, then views it as two
   halves. We already pack `mtp.layers.0.mlp.experts.gate_up_proj` the same
   way in `loading/mtp.py`; the runtime currently splits it back into two
   GEMVs because dispatch doesn’t have a fused selected-expert path.
3. **One captured graph per verifier row count.** The entire trunk forward
   (RMSNorm → GQA → MoE → residual … ×30 layers + lm_head) compiles once
   per `(B+1, kv_pos_bucket)` and replays as a single HIP graph each cycle.
   We have graph capture (Task #47) but it didn’t move the needle because the
   *kernel* time — not launch time — dominates.
4. **MTP head is a standalone “draft model”.** llama.cpp’s
   `convert_hf_to_gguf.py --mtp` produces a separate `mtp-*.gguf` containing
   only the MTP block + shared head. `ctx_dft` loads it as a normal model
   with `LLAMA_CONTEXT_TYPE_MTP` and its own KV cache; the draft loop is just
   `llama_decode(ctx_dft, batch)` per step. This is the clean separation our
   shared-verifier story already promises — we just need our proposer to ride
   the same kernel registry as the target.
5. **Backend sampling.** `llama_sampler_chain_add(… top_k(10))` runs as a ggml
   op on-device; the host gets only the sampled id back. Our GPU-fast accept
   (Task #48) is the equivalent on the verify side; we still do a host loop
   to greedy-pick draft tokens. Move that to a tiny GPU top-1 + write.

### Verifier break-even math (gfx1151, packed PARO+MTP-BF16)

From Task #52 profiling, B=3, perfect-accept assumption:

```
T_AR_step      ~= 22 ms        # AR decode tok/s ~= 45.5
T_verify_B3    ~= 52 ms wall   # ~45 ms kernel + ~7 ms host
MTP/AR (B=3)   ~= 22×3 / 52 = 1.27×   # ceiling, perfect accept
MTP/AR measured = 0.87×              # i.e. avg_accepted < 3 per cycle
```

To hit a **≥1.5×** target at B=3 the verifier must drop below
`22×3 / 1.5 = 44 ms` wall — ~15% off today. To hit **≥2.0×** it must drop
below `33 ms` — ~37% off today.

#### Master scoreboard — verifier wall-time budget (B=3, 4 rows, gfx1151)

Updated as each phase lands. “Baseline (ms)” was Task #52’s informal estimate;
M7.0’s rocprofv3 trace (`benchmarks/results/2026-05-21-hipengine-mtp-verifier-rocprof-baseline.json`) replaces it with measured per-pass kernel times.
“Target (ms)” is the post-phase projection. “Actual (ms)” is filled in from the
rocprofv3 + wall-time artifact when the phase commits. Negative deltas mean
we beat the target.

| Cost component                      | Phase | Task #52 est. (ms) | M7.0 measured (ms) | Target (ms) | Actual (ms) | Δ vs baseline | Δ vs target | Artifact / source |
|-------------------------------------|:-----:|-------------------:|-------------------:|------------:|------------:|---------------:|-------------:|-------------------|
| MoE chain (gate_up + down + rotate + silu + router + combine + w4_dual) | M7 | ~20 | **17.0** | 11–13 | _TBD_ | _TBD_ | _TBD_ | M7.0 artifact |
| GDN chain t-loop (4 rows)           | M7.B*  |          ~10  |          **13.1** |    10–11  |       _TBD_ |          _TBD_ |        _TBD_ | M7.0 artifact (new phase, see below) |
| Small-batch prefill kernels (QKV / shared-expert / dense MLP @ tokens=4) | M7.C* |        n/a  |          **11.0** (M7.0) → **6.97** (M7.C.6) |     2–3   |       **6.97** |        **−4.03** |        **+4 to target** | M7.0 artifact (new phase, see below). Originally mis-labeled “runtime_memset” due to classifier bug. M7.C.6 landed 2026-05-21: `awq_fusedw4_prefill_dual_fp16` 60 → 30 calls/pass (−6.02 ms kernel), replaced by `gemv_awq_pack8_transposed_fp16` (+4.94 ms). Real win is at wall-time: MTP tok/s **23.96 → 27.74** (+15.8%). |
| LM head W8A16 (4 rows)              | M9    |          ~7.5 |           **9.9** |     6–7   |       _TBD_ |          _TBD_ |        _TBD_ | M7.0 artifact |
| Pre-attention chain (QKV + RoPE + norms) | M8 |          ~5  |           **~3.3** (w4_dual_gemv + rmsnorm + paged_kv + decode_attn + attn_gate) |     2.5   |       _TBD_ |          _TBD_ |        _TBD_ | M7.0 artifact. Likely no-op once M7.C lands (the QKV prefill→GEMV switch *is* the pre-attention win). |
| Host-side Python overhead           | M10   |           ~7  |          **~9** (host_window − kernel = 65−56) |     2–3   |       _TBD_ |          _TBD_ |        _TBD_ | M7.0 artifact |
| Other (runtime_copy + paro_rotate + misc) |  —  |          —   |          **~4.6** |     ~4    |       _TBD_ |             —  |          —   | M7.0 artifact |
| **Total verifier wall (host)**      |       |             ~52  |             **~65** | **~35–40** |       _TBD_ |          _TBD_ |        _TBD_ | M7.0 artifact |
| **MTP/AR ceiling @ B=3 (22×3/wall)** |       |          **1.27×**|          **1.02×** | **1.65–1.88×** |   _TBD_ |          _TBD_ |        _TBD_ |                   |
| **MTP/AR measured @ B=3**           |       |          **0.87×**|          **~0.6×** (16-tok decode, 50% accept) | **1.0–1.4×** | _TBD_ |        _TBD_ |        _TBD_ | M7.0 + smoke      |

\* M7.B and M7.C were added after M7.0 revealed the actual top costs differ
from Task #52’s estimate. Both are simpler than the M7 MoE kernel work and may
land first.

Real-world MTP/AR depends on accepted-token density (50–80% measured on the
stable quicksort prompt at B=3, mean ~64% in M7.0 run; the persistent-b3
run on 2026-05-19 with 8-token decode hit 100%). The M7.0 measured ceiling
(1.02×) is below the Task #52 estimate (1.27×) because per-pass kernel time
is 56 ms vs Task #52’s 45 ms, plus ~9 ms host — i.e. “45 ms kernel + 7 ms
host” understates today’s actual cost by ~10 ms. The post-M7…M10 ceiling
range (1.65–1.88×) and measured target (1.0–1.4×) are correspondingly more
conservative than the original 2.06–2.36× / 1.3–1.7× projection. This paragraph
is now superseded by M12.0: M7.C.6 landed and improved wall throughput, but the
B=3 cycle still costs 4.33 AR-token equivalents on the 32-token quicksort
sweep, so a 1.5× measured row is contingent on reshaping the verifier loop, not
just landing M7/M9 as standalone kernel-family optimizations.

### Phase M7 — batched selected-expert GEMV (superseded by M12.4 as top-level priority)

Goal: replace the 60 tiny per-pass MoE GEMV launches with O(layer) launches.

ABI (registry key `("hip_gfx1151", "moe_selected_expert", quant, variant)`):

```text
moe_selected_expert_gemv(
    A:    [n_tokens, n_embd]                 fp16/bf16,           # token rows
    W:    [n_experts, n_out, n_embd]         bf16 | awq_q4 stack, # expert weight stack
    ids:  [n_tokens, n_expert_used]          int32,               # selected experts
    Y:    [n_tokens, n_expert_used, n_out]   fp16/bf16,           # per-slot output
    bias_or_scale:  optional,
) -> Y
```

Weighted-combine and shared-expert add stay outside the kernel; they are
cheap pointwise ops we already have.

#### M7 tracker

Projected savings target: **~4–6 ms** of the 17 ms MoE chain at B=3 / 30
layers (revised down from 8–12 after M7.0). Each row is a separately
landable unit with its own correctness + rocprofv3 gate. Fill `Actual (ms)`
and `Status` as each row commits.

**Pre-condition (M7.0):** code inspection (2026-05-21) confirmed the existing
`gemv_awq_selected_dual_pack8_transposed_bf16` is already a llama.cpp-style
mul_mat_id kernel — grid `(out_packed_a+out_packed_b, rows)` with
`rows = tokens * num_experts_per_tok`, reading `selected[row]` inside the
kernel. So “60 tiny launches” in Task #52’s analysis is “30 layers × 2 fused
MoE ops”, not “1 launch per expert”. M7 work is small-batch tile retuning
of the existing kernel, not a brand-new layer-array kernel.

##### M7.0 measured per-pass breakdown (B=3, 4 rows, gfx1151)

Artifact: `benchmarks/results/2026-05-21-hipengine-mtp-verifier-rocprof-baseline.json`.
Command: `python3 scripts/mtp_verifier_rocprof.py --prompt-tokens "$(cat
quicksort-tokens)" --decode-tokens 24 --candidate-budget 3 --steady-state-skip 2`.

Run: 9 verifier passes total, **7 kept** after dropping the 2 cold cycles.
Acceptance 64% mean (vs 100% on the 8-token persistent_b3 diagnostic from
2026-05-19). Per-pass numbers:

| Family                                  | calls/pass | ms/pass | share | avg μs | max μs | Notes |
|-----------------------------------------|-----------:|--------:|------:|-------:|-------:|-------|
| linear_attention_gdn_decode             |        30  |   13.07 | 23.4% |  435.7 |  513.8 | #1 cost; already chain_tloop. → M7.B |
| w8a16_linear (lm_head)                  |         1  |    9.93 | 17.7% | 9930.4 | 10326.8| Single big launch, bandwidth-bound. → M9 |
| **w4_dual_prefill_smallbatch** (QKV / dense MLP / shared-expert gate+up @ tokens=4) | 60 | **7.40** | 13.2% | 123.4 |  239 | **Wrong kernel for small batch**. Sites: `project_full_attention_qkv_fp16` / `project_linear_attention_qkv_z_fp16` / `shared_expert_paro_w4_fp16` / `dense_mlp_paro_w4_fp16` gate `if tokens == 1: GEMV; else: PREFILL` — lower the threshold to e.g. `tokens > 7`. → M7.C |
| moe_down_gemv                           |        70  |    5.99 | 10.7% |   85.6 |  307.3 | Already mul_mat_id; tile retune. → M7.4 |
| moe_gate_up_dual_gemv                   |        70  |    5.93 | 10.6% |   84.7 |  283.5 | Already fused gate+up. → M7.4 |
| **w4_single_prefill_smallbatch** (shared-expert down / dense-MLP down @ tokens=4) | 60 | **3.61** | 6.5% | 60.2 | 162 | Same kernel-choice bug as above, single-tensor path. → M7.C |
| moe_paro_rotate_in                      |       310  |    1.78 |  3.2% |    5.8 |   27.3 | Many small launches; low margin. |
| w4_dual_gemv (small-token QKV / shared) |        80  |    1.69 |  3.0% |   21.1 |  101.2 | The kernel we WANT the M7.C sites to use. |
| decode_attention                        |        40  |    1.11 |  2.0% |   27.7 |   77.8 | Lean already. |
| router                                  |       140  |    1.05 |  1.9% |    7.5 |   78.9 | Lean already. |
| (runtime_copy + rmsnorm + silu + combine + other small ops) | ~720 | ~5.6 | ~10% | | | |
| **TOTAL per pass**                      |     1838   | **56.0**| 100%  |        |        | Host window: **~65 ms** (kernel 56 + host ~9). |

**Important classifier correction (2026-05-21):** the original `_family`
classifier in `scripts/mtp_verifier_rocprof.py` matched bare `"fill"` which
false-matched `"prefill"`. The two `awq_fusedw4_prefill_*` kernels (11 ms
combined) were therefore mis-labeled `runtime_memset` in the first M7.0
report. The artifact at
`benchmarks/results/2026-05-21-hipengine-mtp-verifier-rocprof-baseline.json`
has been re-processed with the corrected classifier; the kernel CSV / wall
times are unchanged. **There is no memset bottleneck.** M7.C is
rechartered around the small-batch kernel-choice bug below.

##### M7.0 findings vs. Task #52 plan assumptions

1. **MoE is NOT the dominant bottleneck.** Combined MoE chain (gate_up +
   down + rotate + silu + router + combine + w4_dual_gemv attn) = **17.0 ms**
   (~30% of pass), not 20 ms / 44% as Task #52 estimated. The existing
   kernel is already llama.cpp-style. M7 reach drops from 8–12 ms to
   **4–6 ms** (small-batch tile retune of the existing kernel).
2. **GDN chain_tloop is the actual #1 cost: 13.1 ms.** Plan had this as
   “unchanged / already chain_tloop”. 30 launches × 436 μs avg = real
   bottleneck. Added new phase **M7.B** with ~2–3 ms reach (chain length /
   tile / wave-group sweep).
3. **The “memset 11 ms” finding was a classifier bug.** It is actually two
   `awq_fusedw4_prefill_*_fp16` kernels (60 + 60 calls/pass = 11.0 ms)
   firing because `project_full_attention_qkv_fp16`,
   `project_linear_attention_qkv_z_fp16`,
   `shared_expert_paro_w4_fp16` and `dense_mlp_paro_w4_fp16` are gated
   `if tokens == 1: GEMV; else: PREFILL`. At tokens=4 the prefill kernel
   fires even though `gemv_awq_dual_pack8_kernel`’s grid is
   `(out_packed_a + out_packed_b, row)` and already supports `rows > 1`.
   The MoE selected GEMV runs at 85 μs avg; the prefill path runs at 60–123
   μs avg for the same row count. **M7.C is now a small-batch threshold
   fix** with the same ~8–9 ms reach — but it’s a few lines of dispatch
   changes (`if tokens <= 7` instead of `if tokens == 1`), not a runtime
   rewrite. **Still the highest ROI per LoC.**
4. **LM head is 9.9 ms** (vs Task #52’s 7.5 ms estimate). M9 reach adjusted
   to ~3 ms.
5. **Pre-attention chain (M8) is only ~3.3 ms total** (w4_dual_gemv +
   rmsnorm + decode_attention + paged_kv + attn_gate). Task #52’s “~5 ms”
   was high; M8 reach now ~1–1.5 ms. **Note**: M8 was “fused pre-attention
   composite” in the original plan, but M7.C absorbs the bulk of that
   reach (the prefill→GEMV switch *is* the pre-attention QKV win), so M8
   may reduce to a no-op once M7.C lands.
6. **Host-side Python overhead is ~9 ms** vs Task #52’s 7 ms estimate. M10
   ~5 ms reach still plausible.

Revised total reachable savings: **~23–27 ms** (was 18.5–22.5 ms in the
original plan).
Verifier wall: 65 ms → **~38–42 ms**.
Ceiling MTP/AR @ B=3 perfect-accept: **~1.65–1.88×** (was 2.06–2.36×).
Measured MTP/AR @ B=3 60% accept: **~1.0–1.4×** (was 1.3–1.7×).
**A 1.5× measured row is now contingent on landing M7 + M7.C + M9** — not
M7 alone. M7.C is the highest single-phase ROI; landing it first puts us
at ~57 ms verifier wall before any kernel work.

##### M7.0 tooling notes (for next rocprofv3 run)

- **rocprofv3 1.1.0 silently drops `--selected-regions true`** output on this
  gfx1151 host, even when roctxProfilerResume/Pause symbols are correctly
  resolved via the therock SDK overlay. Workaround: full
  `--kernel-trace --marker-trace`, post-process by filtering kernel-CSV
  Start_Timestamp against the roctxRangePush window ns boundaries (markers
  named `mtp_verify_pass_N`).
- **Marker CSV uses `Function` column**, not `Marker_Name` / `Marker_Text`
  (rocprofv3 1.1.0 schema).
- **JIT compile under rocprofv3 spawns subprocesses** that each attach as
  separate rocprofv3 instances (hipcc → clang-23 → lld), producing
  hundreds of “tool initialization” / “tool finalization” log lines and
  breaking output. Fix: set `HIPENGINE_COMPILER_VERSION_FILE` env var so
  the JIT cache key stays stable; pre-warm the build cache by running the
  smoke once before rocprofv3.
- **Do not wrap the prompt-suite/economics parent harness in rocprofv3.**
  `scripts/mtp-bench.py --mode hipengine-current` runs
  `mtp_prompt_suite_economics.py`, which shells out to
  `mtp_verifier_economics.py`, which shells out again to
  `mtp_chain_e2e_smoke.py`. Profiling the parent propagates profiler/JIT state
  into nested Python children and can look like an hour-long hang if a cache
  artifact is missing. Use `scripts/mtp_verifier_rocprof.py` for verifier
  kernel breakdowns, or pre-warm and profile only the final smoke child.
- **SDK ROCTX library needs sysdeps on LD_LIBRARY_PATH**: the therock
  `librocprofiler-sdk-roctx.so.1` depends on `librocm_sysdeps_dw.so.1`,
  which lives under `<sdk_core>/lib/rocm_sysdeps/lib`. Without it,
  `ctypes.CDLL('libroctx64.so')` succeeds against the legacy library but
  has no `roctxProfilerResume` symbol and the marker calls silently no-op.
- `scripts/mtp_verifier_rocprof.py` handles all of the above automatically.
  Re-run with `--steady-state-skip N` to drop more cold cycles.

| #   | Sub-task                                                                                   | Variant            | Projected savings (ms) | Status   | Actual savings (ms) | Notes / artifact |
|-----|--------------------------------------------------------------------------------------------|--------------------|-----------------------:|----------|--------------------:|------------------|
| M7.0| **rocprofv3 re-baseline** — LANDED. 7 steady-state B=3 verifier passes traced via roctxRangePush markers; per-pass: 56 ms kernel + ~9 ms host = ~65 ms wall. Top families: GDN 13.1, memset 11.0, lm_head 9.9, MoE down 6.0, MoE gate_up 5.9. | n/a | 0 (diagnostic) | ✅ **Landed** | 0 | `benchmarks/results/2026-05-21-hipengine-mtp-verifier-rocprof-baseline.json` |
| M7.1| CPU-reference fixture: 4-tok / 30-layer + 8-tok routed MoE, KL≤0.05 / top-1≥0.90 oracle    | n/a                |                     0  | **Superseded** | n/a | The current sprint uses exact 9-prompt D32 gates plus targeted kernel/unit fixtures. New structural kernels still need RED fixtures, but this broad legacy M7 oracle is not the active gate. |
| M7.2| Tune existing `gemv_awq_selected_dual_pack8_transposed_bf16` for 32-row batch (per M7.0: already llama.cpp-style, just needs small-batch micro-tuning). Variant `dense_bf16` for the MTP proposer side. | `dense_bf16` | 1–2 | **Superseded / no-hold as phrased** | n/a | Selected gate/up remains a real compute bucket (`1.893 ms/pass`) but simple launch-width/tile retunes were measured as noise/regression; the proposer BF16 expert loop was addressed by route-batched expert kernels instead. |
| M7.3| Land `dense_bf16` for MTP proposer (down_proj + gate_up_proj fused), route via registry    |`dense_bf16`        |                  ~1–2  | **Superseded / retained differently** | `-0.44 ms/cycle` wall from route batching | `HIPENGINE_MTP_PROPOSER_ROUTE_BATCHED_EXPERT=1` batches top-8 gate/up, SiLU, down, and ordered accumulation while preserving route order; see top table and `benchmarks/results/2026-06-12-hipengine-mtp-proposer-route-batched-expert-retained.json`. |
| M7.4| AWQ pack8 small-batch tile retune: 70 calls/pass at 86 μs avg; target 60 μs avg via LDS / wave-tile sizing for the 32-row case. | `awq_q4_pack8` | ~3–4 | **Superseded by M12.6/M16.4/reduced-DAG work** | n/a | Current selected gate/up/down kernels are already the production selected-GEMV shape; further wins need structural reduced-DAG kernels, not another local tile retune. |
| M7.5| Skip: existing `dual_pack8_transposed` already fuses gate+up (one launch / layer / 2×n_ff_exp output). Reclassify as no-op. | n/a | 0 | ✅ **Landed (n/a)** | 0 | M7.0 confirmed |
| M7.6| Verify: rocprofv3 shows MoE chain runs ≤11 ms total at B=3/30 layers, KL≤0.05 vs CPU-ref    | both               |                     0  | **Superseded** | n/a | Use the current `842 calls/pass` profile and exact D32 prompt suite instead. The active top buckets are selected gate/up `1.893 ms/pass`, selected down `1.204 ms/pass`, shared/dual W4 `1.834 ms/pass`, and router `0.504 ms/pass`; the old `≤11 ms` phase gate is no longer the scoreboard. |
| **M7 total** |                                                                                |                    |              **4–6**   | **Closed / historical** | n/a | Useful work is absorbed by M7.C, M12.6, M16.4, proposer route batching, and current reduced-DAG rows. |

#### M7.B tracker — GDN chain t-loop tuning (resolved by M16.5/M16)

Historical target: **~2–3 ms** of the old 13.1 ms GDN chain t-loop budget.
This lane is no longer pending. Warp-shuffle reductions and `VTILE=4`
dv-tiling are retained; the current best `842`-launch verifier profile has GDN
at about **1.77 ms/pass** (`30` calls/pass). Do not add a separate GDN scalar
pre-pass as the next reduced-DAG unit: `VTILE=4` already computes the shared
q/k scales and `beta`/`decay` once per four `dv` columns, while adding a new
pre-pass would add another launch to a dispatch-floored verifier. `VTILE=8`
was rechecked and no-held because higher VGPR pressure offset the smaller grid.

| #   | Sub-task                                                                                | Projected savings (ms) | Status   | Actual savings (ms) | Notes / artifact |
|-----|-----------------------------------------------------------------------------------------|-----------------------:|----------|--------------------:|------------------|
| M7.B.1| Confirm GDN chain_tloop launch parameters match the 4-row B=3 case (not stuck on 8-row defaults) |                  ~1   | **Closed** | `-0.196 ms/pass` from shuffle reductions | `benchmarks/results/2026-06-09-hipengine-mtp-m16.5-gdn-chain-shuffle-reductions.json` |
| M7.B.2| Tile-size sweep on the 32-context decode shape; promote per CPU-ref correctness gate     |                  ~1–2  | **Closed / VTILE=4 retained** | `-0.56 ms/pass` kernel time in the M16 profile; current exact stack keeps the retained GDN kernel | `benchmarks/results/2026-06-09-hipengine-m16-gdn-chain-dvtiling.json`; `VTILE=8` no-held in `benchmarks/results/2026-06-11-hipengine-mtp-gdn-vtile8-rocprof.json` |
| **M7.B total** |                                                                               |              **~2–3** | **Closed** | Kernel-time headroom banked; not a current wall lever | Broader wall work stays on reduced-DAG/proposer/acceptance-density, not more local GDN tiling. |

#### M7.C tracker — small-batch prefill→GEMV switch (CLOSED / HISTORICAL)

Projected savings target: **~8–9 ms** of the 11.0 ms combined budget for the
two `awq_fusedw4_prefill_*_fp16` kernels. This phase was originally framed as
“runtime_memset elimination” — see M7.C.1 below for what the M7.0 trace actually
showed once the classifier bug was fixed.

2026-06-12 status: do not reopen this tracker as written. The current exact B=3
MTP path has already absorbed the useful safe-mask/split-output lessons through
M7.C.6, M12.6, M16.4, and later C-dispatch routing; the locked current profile
has no `w4_single_prefill_smallbatch` bucket left. New work should start from
the live `842 calls/pass` profile and reduced-DAG table above, not from the old
threshold-bump plan.

Dispatch sites to change (all in `hipengine/runtime/qwen35_paro.py`,
`if tokens == 1: ... else: awq_fusedw4_prefill_*` blocks):

| Site                                          | Line(s) | Kernels removed @ tokens=4 | Replacement |
|-----------------------------------------------|--------:|---------------------------|-------------|
| `project_full_attention_qkv_fp16`             |   ~2087 | `awq_fusedw4_prefill_dual_fp16` | `gemv_awq_dual_pack8_transposed_fp16` |
| `project_linear_attention_qkv_z_fp16`         |   ~3391 | `awq_fusedw4_prefill_dual_fp16` | `gemv_awq_dual_pack8_transposed_fp16` |
| `shared_expert_paro_w4_fp16` (gate+up)        |   ~5025 | `awq_fusedw4_prefill_dual_fp16` | `gemv_awq_dual_pack8_transposed_fp16` |
| `shared_expert_paro_w4_fp16` (down)           |   ~5089 | `awq_fusedw4_prefill_fp16`      | `gemv_awq_pack8_transposed_fp16`      |
| `dense_mlp_paro_w4_fp16` (gate+up)            |   ~5218 | `awq_fusedw4_prefill_dual_fp16` | `gemv_awq_dual_pack8_transposed_fp16` |
| `dense_mlp_paro_w4_fp16` (down)               |   ~5282 | `awq_fusedw4_prefill_fp16`      | `gemv_awq_pack8_transposed_fp16`      |

The replacement kernels already accept `rows > 1` (grid is
`(out_packed_a + out_packed_b, row)` in `gemv_awq_dual_pack8_kernel`). The
BF16 paths (`shared_expert_paro_w4_bf16`, etc.) already do exactly this for
all token counts — per the BF16 docstring: “BF16 has no fused prefill kernel,
so the same dual GEMV (which accepts rows > 1) is used for every tokens
value”. M7.C extends the same logic to the FP16 path for small batches.

| #   | Sub-task                                                                                | Projected savings (ms) | Status   | Actual savings (ms) | Notes / artifact |
|-----|-----------------------------------------------------------------------------------------|-----------------------:|----------|--------------------:|------------------|
| M7.C.1| **Identify culprit** — LANDED. Six dispatch sites listed above use prefill kernels for `tokens > 1`. The 11 ms “memset” was a `_family` classifier substring match against “fill” in “prefill”. | 0 (diagnostic) | ✅ **Landed** | 0 | this section + corrected M7.0 artifact |
| M7.C.2| Add a `_small_batch_decode_threshold` constant + env override; change the six dispatch sites from `if tokens == 1` to `if tokens <= _small_batch_decode_threshold()`. | ~6–8 | ⚠️ **Partial / reverted** | 0 (kept helper only) | Investigation report: see below + commit log |
| M7.C.3| Cross-check: prefill batches (16+ tokens) still take the prefill kernel; verify with a `--rocprof-warmup-cycles 0 --prefill-only` smoke run | 0 | **Superseded** | n/a | Historical prefill guard. Not part of the current B=3 verifier hot path; rerun only if prefill dispatch changes. |
| M7.C.4| Correctness: full B=3 chain still exact-AR-match on the quicksort fixture; KL/top-1 unchanged | 0 | **Closed** | n/a | Covered by later exact quicksort and 9-prompt D32 rows for M7.C.6, M12.6 `single_linear_out`/`single_full_v`, M16.4, and current best `0.927x` stack. |
| M7.C.5| Re-run M7.0 rocprof; new per-pass kernel ms drops by ~7–8 ms (the prefill kernels fall out, replaced by ~85 μs / 4 μs/row GEMVs at < 3 ms total). | 0 | **Superseded** | n/a | The current profile is the authority: `842 calls/pass`, `12.636 ms` kernel/pass, `16.220 ms` host/pass, and no `w4_single_prefill_smallbatch` bucket. |
| M7.C.6| **Split dual GEMV into two single GEMVs at `tokens > 1`** for `project_full_attention_qkv_fp16` (site #1) and `project_linear_attention_qkv_z_fp16` (site #2), mirroring the bf16 sibling pattern at `project_linear_attention_qkv_z_bf16` line ~1075. Adds an `elif tokens <= _small_batch_decode_threshold():` branch that issues two `gemv_awq_pack8_transposed_fp16` calls writing each view's backing pointer directly. | ~6–8 (revised: **~1 ms kernel + ~4 ms wall**) | ✅ **Landed** | **+15.8%** MTP tok/s (23.96 → 27.74) | benchmarks/results/2026-05-21-hipengine-mtp-m7c6-small-batch-dispatch-split.json |
| **M7.C total** |                                                                               |              **~6–8** | **Closed** |  **+15.8% MTP tok/s** from M7.C.6; later paths supersede the old tracker | Use the live reduced-DAG profile for further work. |

##### M7.C.2 investigation report (2026-05-21)

Implemented the naive threshold bump across all 10 sites (six `tokens == 1`
gates plus the two `rows > 1` gates inside `project_pack8_fp16` plus the
`if tokens != 1` paro_rotate1 fall-throughs at the shared-expert /
dense-MLP gate-up paths). Default threshold set to 7 (verifier B ≤ 6).
Added `HIPENGINE_SMALL_BATCH_DECODE_THRESHOLD` env override.

**Result: reverted.** The smoke harness exact-AR-match gate failed at
threshold=7. Bisecting the 10 sites isolated the divergence to **two
specific sites** with a row-stride aliasing bug:

- `project_full_attention_qkv_fp16` (line 2002): the dual GEMV writes
  `q_proj_key` (shape `(tokens, 2*q_width + kv_width)`) with row stride
  `2*q_width + kv_width`. But `q_proj` and `key_bf16` are *views* into
  `q_proj_key` with contiguous strides `2*q_width` and `kv_width`. At
  tokens > 1 the view strides do not match the dual GEMV’s row stride,
  so downstream kernels like `qwen35_split_qgate_fp16` read garbage rows.
- `project_linear_attention_qkv_z_fp16` (line 3337): same pattern with
  `qkv_z` as the combined buffer and `qkv` / `z` as the per-row views.

The **BF16 sibling already knows about this**: see
`project_linear_attention_qkv_z_bf16` lines 1075–1092 — the multi-token
path uses TWO separate single GEMVs writing `qkv` and `z` independently,
with an explicit comment: *“The dual GEMV writes row-major [qkv,z] per
token. Native prefill conv/GDN consumes contiguous [tokens,qkv] and
[tokens,z] streams, so split multi-token prefill into two projections.”*
The BF16 code was written with this constraint in mind; the FP16 sibling
never needed it because its multi-token path always called the
`awq_fusedw4_prefill_*` kernel (which writes two separate buffers).

Sites that DO NOT have this bug, but were also part of the reverted
change:
- `shared_expert_paro_w4_fp16` / `dense_mlp_paro_w4_fp16` (sites 4989,
  5025, 5089, 5182, 5218, 5282): the small-batch path writes
  `scratch.shared_up` which is its own backing tensor of shape
  `(tokens, 2*intermediate)` — no aliasing. These sites are safe to bump
  but are *not exercised* by the BF16 verifier (the BF16 path uses
  `shared_expert_paro_w4_bf16`, which already does the right thing).
- `project_pack8_fp16` helper (lines 491, 527): single-output GEMV with
  contiguous output buffer — no aliasing. Safe.

The “safe subset” (helper + sites #3–#6) was measured under rocprofv3:
- threshold=1 (baseline): **54.69 ms / pass** kernel time
- threshold=7 (safe subset active): **59.29 ms / pass**
- Local saving: `w4_single_prefill_smallbatch` 3.49 ms → `w4_single_gemv`
  3.05 ms = −0.44 ms
- Cache effect: downstream `linear_attention_gdn_decode`, `w8a16_linear`,
  `moe_*_gemv` kernels show +3 ms collectively from changed cache
  footprint
- **Net: −0.44 + ~+3 = ~+2.6 ms regression**

So the safe subset alone is a net regression, the unsafe subset breaks
correctness, and the proper fix (M7.C.6) is required to unlock the
reach. We left in:
- The `_small_batch_decode_threshold()` helper (infrastructure for
  M7.C.6 and future small-batch dispatch decisions).
- The corrected family classifier in
  `scripts/mtp_verifier_rocprof.py` (committed earlier as part of M7.C.1).
- The investigation comments on the reverted dispatch sites so the next
  agent finds the bug without re-discovering it.

Reverted code restores exact-AR-match on the 24-token quicksort fixture
with accepted lengths `[3, 3, 2, 0, 2, 0, 0, 1, 3]` identical to
baseline.

Design notes:
- Grid `(n_out_tiles, n_expert_used, n_tokens)`. Each block reads `expert = ids[token, slot]` and indexes `W[expert, tile, :]`.
- For BF16 dense weights (MTP proposer): straight fp16/bf16 vec-mat reduction with a 32-thread warp per row tile. Variant `"dense_bf16"`.
- For AWQ pack8 weights (target verifier): same kernel structure, swap inner dequant. Variant `"awq_q4_pack8"`. Reuse the dequant microcode from `gemv_awq_selected_*`.
- For n_tokens ≤ 8: keep `ids` in registers; no shared-memory scratch. Matches llama.cpp’s `MMVF_MAX_BATCH_SIZE = 8` fast path.
- Fused gate+up: weights `[n_experts, 2*n_ff_exp, n_embd]`; output `[n_tokens, n_expert_used, 2*n_ff_exp]`; SwiGLU stays as the existing pointwise op.

Plugin discipline:
- Registry key uses four-axis form (`backend, layer, quant, variant`). Routing lives in `hipengine/kernels/registry.py` and `hipengine/dispatch/fusion.py`, **not** in `if backend ==` / `if quant ==` branches.
- Both `dense_bf16` and `awq_q4_pack8` variants register under the same layer key `"moe_selected_expert"`; proposer and verifier pick the same kernel family with different variant tags.
- The unfused per-expert GEMV chain stays registered as fallback.

### Phase M8 — fused pre-attention sub-path (SECOND PRIORITY)

Goal: collapse the ~90 launches per verifier pass for RMSNorm → QKV → RoPE → q_norm/k_norm into one composite launch per layer.

ABI (registry key `("hip_gfx1151", "pre_attention_fused", quant, variant)`):

```text
pre_attention_fused(
    H_in:     [n_tokens, n_embd]    bf16,        # post-prev-residual
    W_qkv:    AWQ-packed Q/K/V
    W_q_norm, W_k_norm:  bf16,
    rope_cos, rope_sin: fp16,
    pos_ids:  int32,
) -> Q, K, V    # already RoPE’d, q/k normalized
```

Unfused fallback: RMSNorm → QKV GEMV → split → rotate → q_norm/k_norm. Stays registered; the fused composite registers as a separate layer key.

#### M8 tracker

Projected savings target: **~3 ms** out of the ~5 ms pre-attn chain at B=3 / 30 layers. The composite is correct-by-construction iff each step matches the unfused fallback on the same fixture; promote per-step.

2026-06-12 status: this tracker is historical. The current exact stack already
keeps the useful small-B projection work through M7.C.6/M12.6/M16.4. The broad
producer-side RMSNorm+rotate class was refreshed and no-held on the current P1
stack, and the full-attn split+key-cast micro-fusion was also no-held despite
removing launches. Reopen only as a new structural full-attention primitive with
fresh exact-suite evidence, not from these old pending rows.

| #   | Sub-task                                                                              | Projected savings (ms) | Status   | Actual savings (ms) | Notes / artifact |
|-----|---------------------------------------------------------------------------------------|-----------------------:|----------|--------------------:|------------------|
| M8.1| CPU-reference fixture: 4-tok / 30-layer pre-attn input → (Q, K, V) post-RoPE oracle    |                     0  | **Superseded** | n/a | New full-attn/linear-attn composite kernels still need targeted RED fixtures; this old broad fixture is not the active gate. |
| M8.2| Fused RMSNorm + QKV GEMV (skip RoPE) — collapse ~60 launches → ~30                    |                  ~1–1.5| **No-hold as current design** | 0 retained | Refreshed `HIPENGINE_FUSED_RMSNORM_ROTATE=1` stayed exact but worsened kernel/host on the current stack. |
| M8.3| Add RoPE inside the kernel (cos/sin from constant buffer)                              |                  ~1   | **No-hold as current design** | 0 retained | Covered by the same producer-side RMSNorm/rotate and rotate-staging evidence; reopen only with a new scheduling-safe structural primitive. |
| M8.4| Add q_norm / k_norm inside the kernel (collapse final ~30 launches)                    |                  ~0.5–1| **No-hold as micro-fusion** | 0 retained | Full-QKV split+key-cast and final RMSNorm+cast launch-only fusions were exact but same-suite negative. |
| M8.5| Verify: rocprofv3 shows 1 launch per layer pre-attn at B=3, KL≤0.05 vs unfused        |                     0  | **Superseded** | n/a | Current profile/table above is authoritative; no retained one-launch pre-attn primitive exists. |
| **M8 total** |                                                                          |               **~3**   | **Closed / historical** | 0 retained from this tracker | Continue with current reduced-DAG structural work, not the old M8 micro-fusion ladder. |

### Phase M9 — parallelized LM head over verifier rows (THIRD PRIORITY)

Goal: cut the 7.5 ms 4-row W8A16 lm_head projection by ~30%.

Switch from current `gemv_w8a16` chained over rows to a row-parallel split-k variant that streams the 248320-row weight matrix once per pass and computes all 4 rows in parallel.

Same ABI as today’s lm_head; new variant under `("hip_gfx1151", "lm_head", "w8a16", "row_parallel")`. Promote only if it beats the current path on B ∈ {2, 3, 4, 8}.

#### M9 tracker

Projected savings target: **~2.5 ms** of the ~7.5 ms LM head at B=3, 4 verifier rows. Bandwidth-bound — the win comes from streaming the weight matrix once, not from arithmetic.

2026-06-12 status: superseded by M12.2. The verifier LM head is already
multi-row/weight-shared and now appears as one `w8a16_linear` launch/pass at
about `1.45 ms/pass` in the current profile. The later fused verifier LM-head
diagnostic removed one argmax launch but made the W8A16 body slower, so it is
no-held for MTP.

| #   | Sub-task                                                                          | Projected savings (ms) | Status   | Actual savings (ms) | Notes / artifact |
|-----|-----------------------------------------------------------------------------------|-----------------------:|----------|--------------------:|------------------|
| M9.1| Row-parallel split-k kernel: grid `(n_out_tiles, n_tokens)`, single weight stream  |                  ~2–2.5| **Done via M12.2 / further fusion no-held** | `66.33 -> 68.77 tok/s` at M12.2; current `~1.45 ms/pass` | M12.2 streams W8A16 weights once for verifier rows. Do not reopen the fused LM-head path without a new W8A16 body that beats the current one-launch multi-row kernel. |
| M9.2| Promote per `B ∈ {2, 3, 4, 8}` sweep; gate via existing lm_head correctness test  |                  ~0   | **Superseded** | n/a | B sweeps are now acceptance-density endgame work after diagnostics; not an LM-head promotion gate. |
| **M9 total** |                                                                      |              **~2.5** | **Closed / historical** | retained by M12.2 | Current LM-head work is not the next wall-cut lever. |

### Phase M10 — align proposer with target dispatch

Once Phase M7 lands the `dense_bf16` variant of the MoE GEMV, the native MTP proposer (`hipengine/speculative/mtp_native.py`) plugs into the same registry path. This phase also eliminates host-side overhead (~7 ms/pass per Task #52).

No new model quant required — BF16 MTP weights stay. If we later quantize MTP, register an `awq_q4_pack8` variant under the same layer key and the proposer picks it up via the variant axis (no code branching).

#### M10 tracker

Projected savings target: **~5 ms** of host-side overhead (the ~7 ms baseline minus an irreducible ~2 ms for batch prep + sampling read).

2026-06-12 status: the useful proposer work landed through narrower measured
slices: skip unused reads/results/snapshots, stream-order and pack scalar H2D,
direct KV writes, fused router top-k+softmax, and route-batched expert kernels.
The remaining graph-capture item is tracked as M12.7 and is profile-bound to the
remaining proposer host gap.

| #    | Sub-task                                                                                                   | Projected savings (ms) | Status   | Actual savings (ms) | Notes / artifact |
|------|------------------------------------------------------------------------------------------------------------|-----------------------:|----------|--------------------:|------------------|
| M10.1| Route `mtp.layers.0.mlp.experts.gate_up_proj` through `moe_selected_expert` / `dense_bf16` (removes Task #50 blocker) |              ~2     | **Superseded / retained differently** | part of `-0.44 ms/cycle` route-batched expert win | The retained sidecar design uses route-batched BF16 expert GEMVs rather than the old registry path. |
| M10.2| Keep selected-expert ids on-device throughout the proposer chain (only D2H sync per draft step = sampled tok) |              ~1.5–2 | **Done / partial retained** | see proposer skip + route-batched rows | Router top-k ids stay on device for expert GEMVs; diagnostic host reads are skipped unless requested. Device-chain candidate buffering was tried and no-held. |
| M10.3| GPU top-1 + write kernel for the next draft seed (proposer never reads top1 to host)                       |                  ~1–1.5| **No-hold as attempted** | 0 retained | Device-chain candidate buffering kept candidates on device but regressed D32 wall/ratio. The proposer still needs the sampled token id at cycle boundaries. |
| M10.4| Re-capture HIP graph for the post-M7/M10 proposer chain (one captured graph per draft depth)               |                  ~0.5–1| **Open as M12.7, profile-bound** | _TBD_ | Direct capture must solve by-value cache destination/context-length dynamics in `NativeMtpChainProposer.advance()`. |
| **M10 total** |                                                                                              |              **~5**   | **Mostly closed; M12.7 remains** | retained proposer wall moved to `~1.25 ms/cycle` | Use the top priority table for current proposer work. |

### Phase M11 — fixed-depth chain bucket sweep on the fast verifier

After M7 lands (the only mandatory phase for a 1.5× row), re-run the B=1/2/3 sweep with same-session AR. Pick the operating point that maximizes measured MTP/AR.

Keep the existing exact-equality gate. The benchmark rollup (`benchmarks/README.md` + `benchmarks/CHANGELOG.md` + JSON artifact) is the only path to a retained speed claim, per `AGENTS.md`.

#### M11 tracker — operating-point sweep (post-M7…M10)

This phase doesn’t add per-kernel savings; it picks the chain depth and acceptance policy that maximize end-to-end MTP/AR on the fast verifier. Projected MTP/AR is the perfect-accept ceiling at the listed B given the post-M10 verifier wall; “measured target” assumes 60–80% acceptance.

2026-06-12 status: current retained operating point is B=3 at `0.927x`,
`19.334 ms/cycle`, and `2.012` visible tokens/cycle. Do not run policy sweeps
from this stale table yet; the top-level acceptance-density backlog now owns
B=4/B=5 diagnostics after the next wall-cut units land.

| #    | Operating point                                                  | Projected ceiling MTP/AR | Projected measured MTP/AR | Status   | Actual measured MTP/AR | Artifact / source |
|------|------------------------------------------------------------------|-------------------------:|--------------------------:|----------|-----------------------:|-------------------|
| M11.1| Chain B=2 (drafts/verify=2, target rows=3)                       |                  ~1.5×  |              ~1.2–1.4×    | **Deferred** | _TBD_ | Run only as part of the acceptance-density diagnostic sweep. |
| M11.2| Chain B=3 (drafts/verify=3, target rows=4) — default candidate   |                  ~2.0×  |              ~1.3–1.7×    | **Current retained operating point** | `0.927x` | Current exact 9-prompt D32 best: `19.334 ms/cycle`, `2.012` visible tokens/cycle. |
| M11.3| Chain B=4 (drafts/verify=4, target rows=5)                       |                  ~2.0×  |              ~1.2–1.6×    | **Deferred to acceptance endgame** | _TBD_ | First higher-B diagnostic after per-depth counters; promote only if suite ratio improves. |
| M11.4| DDTree B=4/8 (tree drafts)                                       |                _≥2.0×_  |                    _TBD_  | **No-hold for current B=3 tree; defer hybrid retest** | B=3 tree `0.61x` vs chain `0.76x` old row | Revisit only if histograms show recoverable first-rejection cases. |
| **M11 retained row** |                                                          |               **>=1.5x** |                  **>=1.3x** | **Superseded by current top table** | `0.927x` current best | Use benchmarks rollup/top table for retained rows; use acceptance backlog for future B sweeps. |

### Out-of-scope (don’t pre-build)

- Cross-arch (CUDA / gfx1100) variants of the new kernels. Land on gfx1151
  first, get a retained row, then port. Backend tree is peer-structured so
  porting is a per-arch task.
- Quantizing MTP weights. Not required for a 1.5× row; revisit if MoE
  bandwidth becomes the new bottleneck after M7.
- Tree-shaped MTP drafts. Chain at B ∈ {2,3} is enough for the first row;
  DDTree is a separate axis covered by `docs/DFLASH.md`.
- Long-context tuning. Get the short-prompt row first; long-context is a
  separate validation matrix.

## Do-not-chase list

Until the shared verifier is exact and faster than serial c=1 verification, do
not spend iterations on:

- deeper fixed-B sweeps;
- margin guards or root-topk rescue;
- adaptive depth policy;
- draft-token/sec headlines;
- allocator-only buffer cleanup;
- graph dry-runs that still replay c=1 target steps;
- final quantized-MTP packaging;
- attention-only tuning when profile says linear/MoE/lm-head dominates;
- any speed claim that omits same-session AR and exact equality.

## First concrete hipEngine tasks after DFlash

1. Confirm `docs/DFLASH.md` D1-D2 verifier/accept pieces are landed and faster
   than serial c=1 on chain B=1/2/4/8 for the shisa packed target on native
   `gfx1151`.
2. Add MTP tensor metadata/materialization for the retained packed-target MTP
   artifact; use the Quark W8A8 + BF16 MTP artifact only as a bring-up/reference
   source unless it becomes the measured target.
3. Add native MTP proposal oracle tests against parent fixed fixtures.
4. Implement `MtpDraftProvider` producing `DraftBatch` chain rows.
5. Feed MTP drafts into the shared chain verifier and accept kernel.
6. Add exact commit-state tests for reject, partial, and full accept.
7. Benchmark B=1/2/3, then B=5 only if the split supports it, always with
   same-session packed-target AR.
8. Revisit top-k/tree policy only after a flat-chain MTP row beats AR.
