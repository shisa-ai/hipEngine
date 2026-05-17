# hipEngine Optimization Grind Plan

Status: 2026-05-17 (reorganized as per-category candidate tables).

Scope: Qwen3.5-35B-A3B-PARO `w4_paro` on W7900/gfx1100, batch-1 prompt/decode rows first.
Goal: close every prefill/decode gap to source-lineage `nano-vllm-amd` parent **and** llama.cpp
HIP/Vulkan on retained comparison shapes (512/128, 4K/128, 32K/128, 128K/128), while preserving
hipEngine's existing peak-memory advantage and torch-free runtime invariant.

This document is the **live punchlist**. Each candidate is a row in a per-lane table with:

| Column | Meaning |
| --- | --- |
| **ID** | Stable label (e.g. `P1.1`, `D2.3`). Use this in commits / `WORKLOG.md` / multiloop tags. |
| **Candidate** | Short description of the change. |
| **Source / lineage** | Where the evidence/precedent lives (parent file, kernel, llama.cpp shader, etc.). |
| **Expected prefill Δ** | Best-guess uplift on `prefill_tok_s` from parent/llama.cpp evidence and ROOFLINE/Amdahl. |
| **Expected decode Δ** | Best-guess uplift on `decode_tok_s`. |
| **Memory** | Expected peak-allocated delta (must respect §3 guardrails). |
| **Risk / prereqs** | Audit/profile/blocker prereqs, correctness hazards, parent negative results to avoid. |
| **Status** | `pending`, `in-progress`, `accepted`, `rejected`, `parked`, or `deferred`. |
| **Result / evidence** | Filled in when the lane is run: measured Δ, artifact path, fixture KL/top-1, rocprof note. |

Status legend:

- **pending** — open, not yet attempted in hipEngine.
- **in-progress** — claimed in `WORKLOG.md`; kernel/wrapper edits in flight.
- **accepted** — measured win, committed, retained benchmark row updated.
- **rejected** — tried, gave no-op or regression, reverted. Record measured delta + artifact.
- **parked** — known-blocked by a prerequisite, or upstream parent already rejected; do not redo without new evidence.
- **deferred** — out of scope for the current batch-1 sweep; planned for a later phase (c>N, multi-GPU, MTP/DFlash).

Cross-links:

- `docs/PREFILL.md` — native prefill architecture, AOTriton evidence, profile/Amdahl analysis.
- `docs/KERNELS.md` — kernel catalog, port playbook, source-lineage drift workflow.
- `docs/ROOFLINE.md` — RDNA3/W7900 perf model and anti-rabbit-hole rules.
- `docs/BENCHMARK.md` and `benchmarks/README.md` — promotion contract and rollup.
- `docs/MARLIN.md` / `docs/DFLASH.md` / `docs/MTP.md` / `docs/GGUF.md` — large lanes covered by their own design docs; this file lists the *entry-point* candidate.
- Parent: `~/amd-gpu-tuning/docs/OPTIMAL.md`, `PLAN-PAROQUANT2.md`, `PLAN-LONGCONTEXT.md`,
  `docs/LLAMACPP-VULKAN.md`, `PR_COMMENT-llamacpp-hip-unroll600.md`, `LESSONS-LEARNED.md`.

---

## 1. Current scoreboard

Current hipEngine rows are resident-runner comparison-table diagnostics run with the deployment
defaults:

```text
--attn-aotriton-min-tokens 512
--graph-replay-decode
--prefill-linear-chunk-size 1024
--prefill-moe-chunk-size 1024
--prefill-full-attn-query-chunk-size 4096
--prefill-full-attn-post-chunk-size 1024
--prefill-full-attn-rope-chunk-size 1024
```

Source: `benchmarks/results/2026-05-16-hipengine-qwen35-comparison-tables-diagnostic.json`.
Re-score with:

```bash
python3 scripts/qwen35_compare_tables.py all
```

### 1.1 vs `nano-vllm-amd` parent (`docs/OPTIMAL.md` 2026-05-13)

| Workload | Prefill delta | Decode delta | Peak memory delta | Lift needed to win prefill | Lift needed to win decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512/128 | -13.3% | -5.7% | -0.28 GiB ✅ | **+15.4%** | **+6.0%** |
| 4K/128 | -7.3% | -1.7% | -1.77 GiB ✅ | **+7.9%** | **+1.7%** |
| 32K/128 | +0.3% ✅ | -4.9% | -0.68 GiB ✅ | already ahead | **+5.2%** |
| 128K/128 | +9.7% ✅ | -2.5% | -3.76 GiB ✅ | already ahead | **+2.5%** |

### 1.2 vs llama.cpp HIP (`PLAN-LONGCONTEXT.md` split rows)

| Workload | Prefill delta | Decode delta | Lift needed |
| --- | ---: | ---: | --- |
| 512/128 | -9.0% | +27.6% ✅ | **+9.9%** prefill |
| 4K/128 | +15.1% ✅ | +26.0% ✅ | none |
| 32K/128 | +26.1% ✅ | +22.0% ✅ | none |
| 128K/128 | +41.1% ✅ | +6.5% ✅ | none |

### 1.3 vs llama.cpp Vulkan (`PLAN-LONGCONTEXT.md` split rows)

| Workload | Prefill delta | Decode delta | Lift needed |
| --- | ---: | ---: | --- |
| 512/128 | +22.0% ✅ | -14.4% | **+16.9%** decode |
| 4K/128 | +46.9% ✅ | -8.4% | **+9.1%** decode |
| 32K/128 | +67.1% ✅ | -4.2% | **+4.4%** decode |
| 128K/128 | +108.6% ✅ | -5.3% | **+5.6%** decode |

### 1.4 The compact goal

To beat both `nano-vllm-amd` and llama.cpp Vulkan across the retained board we need roughly:

- **Prefill:** +15% at 512, +8% at 4K. 32K/128 and 128K/128 are already ahead of parent.
- **Decode:** +6% at 512, +2-5% at 4K/32K/128K vs parent; +17% at 512, +5-9% at 4K/32K/128K vs Vulkan.
- **Memory:** preserve the current peak-allocated advantage everywhere; keep 512/128 and 4K/128 under 24 GiB.

The decode lift is the steeper climb. Per `docs/PREFILL.md` §"Optimization diagnosis (2026-05-16)"
and parent `PLAN-PAROQUANT2.md` §11 Amdahl, the only way to find +15% decode at 512/4K is **compound
wins across non-W4 buckets** — rotation + RMSNorm fusion, replay dispatch reduction, and selective
attention/W4 work, in that order. Single-knob kernel rewrites alone cannot get there.

---

## 2. Strategy in one paragraph

Do not start with another blind kernel multiloop. First capture matched ROCTX / `rocprofv3
--kernel-trace` profiles for hipEngine on the comparison rows so we know which bucket each P/D
candidate actually targets on this stack (Lane M). The remaining short/mid prefill miss is most
likely bulk dense/shared-expert GEMV-shaped work (Lane P1) plus AOTriton glue (Lane P2); the parent
runs these as framework `F.linear(...)` GEMMs and llama.cpp HIP's prefill jumps +166% with one
compiler flag. The decode miss is the compound of replay dispatch fanout (~660-900 dispatches/token
on parent), rotation + RMSNorm boundary launches (combined ~20% of decode bucket per the parent
rocprof audit), and a small W4-launch-floor tail; we attack each in audit-first order and land the
parent's already-validated wins (Marlin-K vec8 layout, fused selected-MoE shared gate-sigmoid skip)
where they port cleanly. Long-context is mostly chunking-bound and already parity/ahead of parent
on prefill — the next 32K/128K decode levers are attention split-cap retuning and the grouped-GQA
producer family. Memory stays a feature: every candidate must keep 512/4K under 24 GiB and must not
reintroduce duplicate W4 qweight residency.

---

## 3. Non-negotiable promotion gates

These apply to **every** row below before it can move from `in-progress` to `accepted`.

1. **Correctness first.** The relevant fixture gates must pass before a number is retained. For
   the Qwen3.5/PARO batch-1 path that means at least:
   - `python3 scripts/qwen35_native_prefill_fixture_gate.py --max-layers 40 ...`
   - `python3 scripts/qwen35_decode_graph_fixture_gate.py --max-layers 40 ...`
   - and any new kernel-family CPU-reference / smoke gate from `docs/TESTING.md` and `docs/KERNELS.md`.
2. **No hidden torch in the hot path.** `import torch` is never in any module reached by
   `hipengine.LLM.generate()`. Profiler-only Python wrappers are allowed.
3. **Registry, not backend branches.** New paths register under `(backend, layer, quant, variant)`.
   No `if backend == "..."` or `if quant == "..."` in engine/model/dispatch.
4. **Memory budget.** Default 512/128, 4K/128, and 4K/4K rows stay under 24 GiB peak. Long-context
   rows may exceed only when explicitly labeled W7900 diagnostic; current chunked 128K/128 is
   already below 24 GiB and must not regress.
5. **Retained perf rows update the rollup.** `benchmarks/README.md`, `benchmarks/CHANGELOG.md`, and
   the compact JSON artifact in `benchmarks/results/` all move with each accepted row.
6. **Generated-sample equality.** A retained row matches the parent fixture token stream (or, when
   re-seeded, matches a known-good generated sample). Speed without sample equality is a
   correctness bug per the LESSONS-LEARNED RoPE / NaN history.

---

## 4. Lane M — Measurement (profile our own kernels before optimizing them)

The parent and llama.cpp throughput rows are the comparison baselines and are not in question.
What we needed was our **own** per-kernel rocprof data. That baseline is now landed via
`rocprofv3 --kernel-trace --selected-regions` using `roctxProfilerResume/Pause` around prefill and
measured decode graph. Raw traces stay under `/tmp/hipengine-rocprof-qwen35-audit/`; the committed
summary is `benchmarks/results/2026-05-17-hipengine-qwen35-rocprof-amdahl-diagnostic.json`.

Caveat: rocprofv3 1.2.3 asserts in finalization when tracing 64/128 graph replays on this host
(`retired dangling correlation IDs`). The decode Amdahl rows therefore profile **16 one-step graph
replays** and scale per token; the throughput scoreboard in §1 remains the real 128-token run.

| ID | Candidate | Source / lineage | Expected prefill Δ | Expected decode Δ | Memory | Risk / prereqs | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M.3 | Collect matched `rocprofv3 --kernel-trace` profiles for hipEngine 512/128, 4K/128, 32K/128 with the comparison-table flags. Retain only compact summaries under `benchmarks/results/`. | `docs/KERNELS.md` "Pre-optimization audit"; parent `~/amd-gpu-tuning/scripts/summarize_rocprof_kernels.py` bucketing precedent. | n/a | n/a | n/a | Used `--compiler-version-file` + `--require-cached-build`; selected-region profiling avoids profiled `hipcc` and marker-trace graph replay crashes. | accepted | `benchmarks/results/2026-05-17-hipengine-qwen35-rocprof-amdahl-diagnostic.json` |
| M.4 | Per-bucket Amdahl table for hipEngine 512/128 decode replay-only window. | Parent rocprof tail audit `artifacts/paroquant2_rocprof_audit_20260515_iter30/`; ROOFLINE §5. | n/a | n/a | n/a | 16 traced graph replays; 877 dispatches/token and 7.27 ms GPU kernel time/token at 512/128. | accepted | §6 table below + same artifact |

---

## 5. Lane P — Prefill

Measured selected-region prefill Amdahl (`rocprofv3 --kernel-trace`, 40 layers, comparison-table
flags; kernel time only):

| Bucket / family | 512/128 share | 4K/128 share | 32K/128 share | Main candidate(s) |
| --- | ---: | ---: | ---: | --- |
| MoE selected compact WMMA (`moe_awq_wmma`) | 26.2% | 21.5% | 15.6% | P1.4 thresholding |
| Linear-attention GDN prefill | 20.5% | 21.1% | 15.7% | P3.1 boundary fusion |
| W4 prefill GEMM | 17.9% | 17.1% | 12.7% | layout-preserving fusion only |
| Shared-expert W8A16 | 15.3% | 15.5% | 11.6% | P1.2 / P1.3 bulk dense path |
| AOTriton prefill attention | 1.9% | 5.7% | 30.2% | P2 already mostly settled; P4 deferred |
| Rotation / RoPE | 6.7% | 7.7% | 5.7% | P3.1 |
| Router | 4.5% | 4.6% | 3.4% | P3.2 |
| Linear-attention conv | 1.8% | 1.8% | 1.3% | low priority |
| RMSNorm | 1.2% | 1.7% | 1.3% | fused only when paired with P3.1 |
| MoE metadata | 0.8% | 0.8% | 0.6% | P3.3 stays low priority |

Kernel totals: 512/128 = 196.9 ms (82% of host prefill), 4K/128 = 1.576 s (96%), 32K/128 =
16.97 s (98%). The short/mid prefill miss is therefore not attention anymore; it is per-layer bulk
work: MoE compact WMMA + GDN + W4 + shared expert. AOTriton becomes the long-context top bucket at
32K, but 32K/128 and 128K/128 prefill are already ahead of parent, so P1/P3 stay ahead of P4. W.1 was tested after this audit and is neutral/noisy, not a remaining lever.

### 5.1 P1 — Bulk dense / shared-expert GEMM-shaped paths (parent uses `F.linear(...)`)

Parent multi-row `ParoQuantDenseLinear.forward(...)` and `ParoQuantSharedExpert.forward(...)` fall
through to `F.linear(...)` (rocBLAS/Tensile bulk GEMM). hipEngine currently uses row-shape GEMV
kernels for the same work. This is the leading P0 hypothesis from `docs/PREFILL.md`.

| ID | Candidate | Source / lineage | Expected prefill Δ | Expected decode Δ | Memory | Risk / prereqs | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1.1 | Torch-free hipBLASLt/rocBLAS ctypes wrapper for linear-attention A/B BF16/FP16 dense projections (replace `project_linear_attention_ab_fp16(...)` row GEMV pair). | `docs/PREFILL.md` P0; parent ledger `native_aux_dense_linear_calls=280`. | regressed with rocBLAS | neutral/slightly negative measured | neutral | Tested as `HIPENGINE_LINEAR_AB_PREFILL_ROCBLAS_MIN_TOKENS=2` using `rocblas_gemm_ex` FP32 accumulation; the skinny N=32 A/B shape is faster with the current custom row-GEMV kernels. Must not import torch. | rejected (rocBLAS prototype) | `benchmarks/results/2026-05-17-hipengine-qwen35-qwen36-p11-rocblas-ab-rejected.json` |
| P1.2 | Torch-free bulk dense path for shared expert gate/up SiLU during prefill (replace `w8a16_shared_gate_up_silu_fp16(...)` for `tokens >= threshold`). | `docs/PREFILL.md` P0; parent ledger `native_shared_expert_dense_calls=80`; parent W8A16 prefill path. | +0.5% at 512, +2.2% at 4K legacy W8A16 measured | neutral measured | neutral | Retained as token-tiled W8A16 gate/up (`token_tile=2`) for legacy shared expert only when `tokens >= 1024`; existing per-token kernel remains fallback / opt-out (`HIPENGINE_SHARED_GATE_UP_PREFILL_TOKEN_TILE=0`). Packed PARO sidecars continue on W4 prefill kernels. | accepted (large legacy prompts) | `benchmarks/results/2026-05-17-hipengine-qwen35-qwen36-p12-shared-gate-up-token-tile-diagnostic.json` |
| P1.3 | Torch-free bulk dense shared down + combine for prefill (`w8a16_shared_down_combine_residual_fp16`). | Same as P1.2. | +0.9% at 512/4K legacy W8A16 measured | neutral measured | neutral | Retained as token-tiled W8A16 shared down+combine (`token_tile=2`) for legacy prefill `tokens >= 2`; existing fused tail remains fallback / opt-out (`HIPENGINE_SHARED_DOWN_COMBINE_PREFILL_TOKEN_TILE=0`) and preserves sigmoid/residual semantics. Packed PARO sidecars continue on W4 shared expert + separate batch combine. | accepted (legacy prefill) | `benchmarks/results/2026-05-17-hipengine-qwen35-qwen36-p13-shared-down-combine-token-tile-diagnostic.json` |
| P1.4 | Empirical crossover threshold for compact WMMA vs bulk GEMM/GEMV across remaining candidates (analogous to parent `WMMA_MIN_TOKENS=64`). | Parent `LESSONS-LEARNED.md` "Compact WMMA prefill crossover" + `docs/OPTIMAL.md`. | compact WMMA decisively wins at 128+ measured | neutral measured | +0.008-0.251 GiB vs GEMV fallback | Retain `HIPENGINE_MOE_PREFILL_COMPACT_WMMA_MIN_TOKENS=2`: tokens=1 keeps c1 GEMV/decode, tokens>=2 uses grouped compact WMMA. Diagnostic override can force GEMV fallback; no backend/quant branch in hot dispatch. | accepted (threshold=2) | `benchmarks/results/2026-05-17-hipengine-qwen35-qwen36-p14-moe-wmma-threshold-diagnostic.json` |
| P1.5 | Sweep `-mllvm -amdgpu-unroll-threshold-local=600` build flag on the hipEngine prefill kernels. | `~/amd-gpu-tuning/PR_COMMENT-llamacpp-hip-unroll600.md`: llama.cpp HIP pp512 **+166%** at this flag; multi-model +6-232%. Parent PAROQUANT trial was **neutral** on `v8` kernels (E1 in `PLAN-PAROQUANT2.md` §12). | neutral/noisy measured | neutral/noisy measured | neutral | Default profiles already include the flag; `HIPENGINE_DISABLE_UNROLL600=1` strips only the unroll pair for ablation while preserving `-mcumode`. | accepted (neutral default) | `benchmarks/results/2026-05-17-hipengine-qwen35-qwen36-w1-unroll600-ablation-diagnostic.json` |
| P1.6 | Selective `-mcumode` build profile on hot prefill kernels. | `PR_COMMENT-llamacpp-hip-unroll600.md` threshold bracket table; parent ROOFLINE notes CU mode is build-profile dependent on gfx1100. | +0-2% on top of P1.5 | neutral | neutral | `HIPENGINE_PREFILL_MCUMODE=1` appends `-mcumode` only to remaining prefill-profile artifacts; most hot dual-use kernels already use decode `-mcumode`, and compact WMMA already requested it explicitly. | rejected (neutral/noisy) | `benchmarks/results/2026-05-17-hipengine-qwen35-qwen36-p16-prefill-mcumode-rejected.json` |

### 5.2 P2 — AOTriton glue and full-attention prelude

| ID | Candidate | Source / lineage | Expected prefill Δ | Expected decode Δ | Memory | Risk / prereqs | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P2.1 | AOTriton `attn_fwd_compact_varlen` Q/gate + K prelude fusion: read FP16 `Q\|gate` and FP16 K directly, emit gate FP16 + BF16 Q + FP32/BF16 K in one kernel; removes split + key cast launches + `query_raw`/`key_raw` scratch. | `docs/PREFILL.md` low-risk fusion audit, candidate #3; parent grouped-prefill kernel template. | throughput-neutral/slightly negative measured | neutral | small memory cleanup retained | Two diagnostics landed: cast-glue fusion is within run noise, gate-rotate fusion is neutral/slightly negative. Kept on tree as a launch/memory cleanup; not a perf lever. | rejected (perf); accepted (memory cleanup) | `benchmarks/results/2026-05-16-hipengine-qwen35-aotriton-cast-glue-diagnostic.json`, `…-aotriton-gate-rotate-diagnostic.json` |
| P2.2 | AOTriton V3 `attn_fwd_compact_varlen` ABI (shape-streaming compact-varlen params). | `docs/PREFILL.md` `aotriton_release.toml`; vendored 0.11.2b with the 12 BF16 head-dim-256 gfx11xx forward images Qwen3.5 needs. | landed via V3 ABI; throughput within run noise | neutral | within memory diagnostic noise | V3 ABI is the call path; 0.12 image upgrade only worth re-opening if M.3 says AOTriton time is a non-trivial prefill bucket. | accepted (ABI landed) | `benchmarks/results/2026-05-16-hipengine-qwen35-aotriton-v3-prefill-diagnostic.json` |
| P2.3 | Default deployment policy `--attn-aotriton-min-tokens 512`; native attention is diagnostic fallback only. | `docs/PREFILL.md` AOTriton sweep table (4K native = 662 tok/s; threshold-512 AOTriton = 2346 tok/s). | n/a (already deployed) | neutral | n/a | `PrefillConfig` and benchmark defaults use threshold 512; `0` is an explicit diagnostic override. | accepted (deployment policy) | `benchmarks/results/2026-05-16-hipengine-qwen35-aotriton-threshold-sweep-diagnostic.json` |

### 5.3 P3 — Boundary fusion for linear-attention and MoE prefill

These three are listed in `docs/PREFILL.md` "Additional low-risk prefill fusion audit (2026-05-16)"
as the recommended order; all are validated by source structure, not measurement yet.

| ID | Candidate | Source / lineage | Expected prefill Δ | Expected decode Δ | Memory | Risk / prereqs | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P3.1 | Fuse `qwen35_gdn_prefill_rmsnorm_gate_fp16` + `paro_rotate1_fp16` tail before `awq_fusedw4_prefill_strided_fp16`; writes `out_rot` directly when `head_v_dim == group_size`. | `docs/PREFILL.md` audit candidate #1; `head_v_dim == group_size` for Qwen3.5/PARO so the shape is safe. | +0.5% at 4K measured, negative at 32K measured | neutral/slightly negative at 32K measured | unchanged measured (scratch still reserves `recurrent_bf16`) | Diagnostic `HIPENGINE_LINEAR_GDN_PREFILL_ROTATE_FUSED=1` preserves fallback. Kernel smoke is bit-exact and fixture gate passes, but 32K regresses and memory does not drop without a larger scratch-planning change. | rejected (diagnostic opt-in only) | `benchmarks/results/2026-05-17-hipengine-qwen35-qwen36-p31-gdn-rotate-rejected.json` |
| P3.2 | Prefill-only router shared-gate `sigmoid()` fused into top-k path so grouped prefill skips `w8a16_shared_gate_sigmoid_fp32`. | `docs/PREFILL.md` audit candidate #2. | +0.21% at 512/128 measured, -0.23% at 4K/128 measured on Qwen3.5 legacy | decode neutral/noisy measured | unchanged measured | Diagnostic `HIPENGINE_PREFILL_ROUTER_SHARED_GATE_SIGMOID_FUSED=1` is prefill-only, legacy-shared-expert-only, and preserves raw shared-gate logits for c=1 decode and packed shared-expert combine. Correctness and rocprof smoke pass, but the E2E gain is not robust. | rejected (diagnostic opt-in only) | `benchmarks/results/2026-05-17-hipengine-qwen35-qwen36-p32-router-sigmoid-rejected.json` |
| P3.3 | MoE metadata fanout collapse: combine `moe_group_prefix` + `moe_wmma_tile_map` and initialize `scatter_offsets`/`tile_expert` in the same small metadata kernel. | `docs/PREFILL.md` audit candidate #5. | not measured via A/B; optimistic upper-bound ≤0.27% at 512/128 and ≤0.12% at 4K/32K from M.3 profile | neutral | neutral | Existing M.3 rocprof shows all MoE metadata below 0.85% of prefill kernel time, and the specific prefix+tile-map+two-memset target is below material threshold. Do not add another diagnostic kernel until c>N or scheduler profiling makes metadata multi-percent. | deferred (profile-gated; no implementation) | `benchmarks/results/2026-05-17-hipengine-qwen35-p33-moe-metadata-fanout-deferred.json` |
| P3.4 | Templated FP16-input segment conv wrapper for packed linear-attention path; remove `fp16_to_f32` cast and `qkv_f32` scratch in c>N packed prefill. | `docs/PREFILL.md` audit candidate #4. | +0-1% c=1; bigger lift on c>N | neutral | -0.05 GiB on c>N | Affects compact c>N more than batch-1; schedule after the batch-1 board closes. | deferred (Lane S, c>N) | — |

### 5.4 P4 — Native full-attention prefill kernel (long-term replacement of AOTriton)

| ID | Candidate | Source / lineage | Expected prefill Δ | Expected decode Δ | Memory | Risk / prereqs | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P4.1 | Hand-rolled HIP Flash-Attention-2 forward kernel with WMMA tile, online softmax, GQA reuse, in-kernel fused gate post-pass; uses AOTriton output as oracle. | `docs/PREFILL.md` "Options for fast prefill attention without `torch`"; `docs/KERNELS.md` planned `qwen35_causal_gqa_gate_fp16`. | +0-15% over AOTriton at the Qwen3.5 fixed shape; mostly a packaging/portability win | neutral | -0.1 GiB (no AOTriton image cache) | 3-6 weeks. Do **not** start before P1/P2/P3 are settled. Per `docs/PREFILL.md` "Explicit non-goals", **do not** start until AOTriton is wrapped and used as oracle. | deferred (Phase 4+) | — |

### 5.5 Long-context prefill (32K/128, 128K/128)

Chunked long-context prefill (`PrefillConfig.linear_chunk_size`, `moe_chunk_size`,
`full_attn_query/post/rope_chunk_size`) is **landed** and is the source of the current 32K/128K
advantage over parent: +8.9% prefill at 32K and the unblock for 128K, with -14.4 GiB at 32K and
-3.8 GiB at 128K (artifact `benchmarks/results/2026-05-16-hipengine-qwen35-prefill-chunking-diagnostic.json`).
Keep as default policy.

| ID | Candidate | Source / lineage | Expected prefill Δ | Expected decode Δ | Memory | Risk / prereqs | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P5.2 | Long-context chunk-size auto-tuner that respects per-shape memory budget instead of static defaults. | Parent `OPTIMAL.md` long-prefill overrides; current hipEngine used static defaults. | 128K/128 `+2.44%` vs static measured; 32K keeps static chunks (`+0.67%` noise) | neutral (`-0.40%` at 128K measured; short/32K noisy positive) | 128K uses +1.30 GiB vs static when budget allows q8192; 512/4K unchanged; budget-limited policy stays static | Default `PrefillConfig.auto_tune_chunk_sizes=True`: <32K unchunked, 32K uses 1024/4096 chunks, ≥128K raises full-attn query chunk to 8192 only when budget ≥24.5 GiB. Manual chunk flags override. Fixture gates pass. | accepted (default auto policy) | `benchmarks/results/2026-05-17-hipengine-qwen35-p52-prefill-chunk-autotune-accepted.json` |

---

## 6. Lane D — Decode

Measured selected-region decode Amdahl (`rocprofv3 --kernel-trace --selected-regions`, 16 one-step
graph replays because rocprofv3 crashes at 64/128 traced replays on this host; shares are kernel
time only and scale per token):

| Bucket / family | 512/128 share | 4K/128 share | 32K/128 share | Calls/token | Main candidate(s) |
| --- | ---: | ---: | ---: | ---: | --- |
| Selected-MoE W4 GEMV | 17.9% | 18.3% | 15.5% | 80 | D1.4; D2.1 where applicable |
| W8A16 linear / lm-head / dense | 15.7% | 15.7% | 13.4% | 81 | D5.2 |
| W4 single pack8 GEMV | 13.4% | 13.6% | 11.6% | 50 | D2.1 |
| W4 dual pack8 GEMV | 11.8% | 11.7% | 10.0% | 40 | D2.1; D1.1 |
| Decode attention | 11.4% | 10.5% | 22.9% | 10-20 | D3.1 / D3.2 / D3.3 |
| Rotation / RoPE | 9.4% | 9.6% | 8.4% | 160 | D1.1 |
| Router | 5.8% | 5.8% | 5.1% | 80 | D1.5 / D5.3 |
| Linear-attention GDN decode | 5.2% | 5.4% | 4.8% | 30 | D5.1 |
| RMSNorm / add-RMSNorm | 3.3% | 3.4% | 3.0% | 91 | D1.2 |
| Dense GEMV | 1.2% | 1.2% | 1.2% | 30 | already fused for A/B decode (D5.4) |
| MoE combine | 1.2% | 1.2% | 1.0% | 40 | D1.4 |

Decode graph replay emits **877 dispatches/token** in this profile. Kernel time/token is 7.27 ms at
512, 7.23 ms at 4K, and 8.60 ms at 32K; host decode steps are slower because graph replay and host
bookkeeping sit outside kernel duration. The parent-borrowed "boundary fusion dominates W4" story
was too strong for hipEngine: W4 + selected-MoE GEMV is ~43% of short-context kernel time, W8A16 is
~16%, and boundary/glue buckets (rotation + router + GDN + RMSNorm + combine) are ~25%.

### 6.1 D1 — Dispatch reduction and boundary fusion (largest non-W4 buckets)

| ID | Candidate | Source / lineage | Expected prefill Δ | Expected decode Δ | Memory | Risk / prereqs | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D1.1 | Fuse rotation into the same-input W4 GEMV producer for paired q/k/v stacked attention projections. | Parent `gemv_awq_selected_dual_pack8_strided_rotate_out` precedent (already ported as a kernel surface in hipEngine, but selected/strided only); `PLAN-PAROQUANT2.md` D4 rejected at per-output-pack granularity but valid at projection granularity. | neutral | **+2-4% at 512/128**, +1-3% at 4K/128 | neutral | Must rotate **once** per residual block (not once per output pack); D4 rejection in `PLAN-PAROQUANT2.md` §12.4 is the cautionary tale. | rejected | Opt-in rotate-once staged dual-pack8 kernel is correct and visible in rocprof, but 512/128 graph decode regressed `115.450 -> 110.457 tok/s` (`-4.32%`); default remains off. Artifact: [`2026-05-17-hipengine-qwen35-d11-rotate-dual-pack8-fusion-rejected.json`](../benchmarks/results/2026-05-17-hipengine-qwen35-d11-rotate-dual-pack8-fusion-rejected.json). |
| D1.2 | Fuse `paro_rmsnorm` / `add_rmsnorm` producer into the following projection where the normalized vector is single-use. | Parent linear-attn QKV/Z + full-attn Q/K precedents (`afb7b16`, `FULL_ATTN_QK_PACK8_FUSED`); extend only if the producer can be staged once and consumed by all needed projections. | neutral | no code change; safe single-use slice is ≤0.04% kernel time at 512/128 | neutral | Must keep fast pack8/repacked layout; per LESSONS-LEARNED "Output buffers alone are rarely enough under graph replay" — only count wins that change arithmetic / data reuse. | deferred (no-op) | M.4 shows RMSNorm/add-RMSNorm is only `3.30%/3.37%/2.97%` of decode kernel time at 512/4K/32K, and static dataflow shows the 40 input RMSNorms fan out to multiple attention projections while the 40 add-RMSNorms fan out to router + selected/shared expert paths. The only clear single-use producer is final RMSNorm -> lm-head (~`0.04%` kernel-time upper bound), too small to justify a row-staged W8A16/pack8 fusion after D1.1's barrier regression. Revisit after D1.3/D1.6 if a multi-consumer row-staging design exists. Artifact: [`2026-05-17-hipengine-qwen35-d12-rmsnorm-producer-fusion-deferred.json`](../benchmarks/results/2026-05-17-hipengine-qwen35-d12-rmsnorm-producer-fusion-deferred.json). |
| D1.3 | Same-input projection fusion for remaining adjacent c=1 GEMV pairs not yet fused (consult M.4 buckets for ordering). | Parent linear-attn `LINEAR_ATTN_QKV_Z_PACK8_FUSED` `+0.74%`, full-attn `FULL_ATTN_QK_PACK8_FUSED` `+0.41%`; LESSONS-LEARNED full-attn Q/K/V pack8 widen was **rejected** because the wider kernel erased launch savings. | neutral | no code change; no standalone remaining pair with arithmetic/data-reuse upside | neutral | Must preserve pack8 / repacked layout. Pure launch-count fusion is sub-1% under graph replay; only counts if arithmetic/reuse also improves. | rejected (no-op) | Static D1.3 inventory found linear-attn QKV/Z, full-attn Q/K, dense A/B, selected-MoE gate/up, and shared-expert gate/up already fused while preserving pack8/repacked layout. The only material unfused same-input slice is full-attn V beside the retained Q/K dual path (10 of the 50 W4 single-GEMV calls/token), but parent full-attn Q/K/V pack8 widening was correct yet slower/no-win (`116.357/107.412` vs retained Q/K `116.721/107.703` tok/s at 512/4K). Do not add a D1.3 broad widening kernel; leave any narrower K/V retest to D1.6. Artifact: [`2026-05-17-hipengine-qwen35-d13-same-input-projection-fusions-rejected.json`](../benchmarks/results/2026-05-17-hipengine-qwen35-d13-same-input-projection-fusions-rejected.json). |
| D1.4 | Selected-MoE post-op fold: combine selected-expert weighted-sum + add + sigmoid + residual into one kernel (Vulkan `MUL_MAT_ID_ADD_ID_MUL` shape). | Parent `selected-MoE silu/down-rotation fusion` (`fbff0fe`) precedent; `PLAN-PAROQUANT2.md` §11.5.2; `LLAMACPP-VULKAN.md` graph fusion analysis. | neutral | **+1-2% at 512/128, +1% at 4K/128** | neutral | Sorted-lane semantics: weighted scatter cannot naively fuse without atomics or layout change. Re-read parent F2 WMMA M16 lesson before attempting larger combined kernels. | rejected (further fold) | The safe c=1 post-op fold is already the default `weighted_sum_shared_gate_combine_residual_out_*` path and remains correct. Thread probe found no combine-kernel lever (~`2.44 us` median at 64/128/256 threads); direct selected-down+combine is rejected by parent target-shape evidence (`13.38 -> 16.52 us`, `0.81x`) because it collapses `out_pack * active_experts` grid parallelism. Current 512/128 default diagnostic: `2284.652` prefill tok/s, `115.755` decode tok/s, `18.176 GiB`; no D1.4 decode improvement claimed. Artifact: [`2026-05-17-hipengine-qwen35-d14-selected-moe-postop-fold-rejected.json`](../benchmarks/results/2026-05-17-hipengine-qwen35-d14-selected-moe-postop-fold-rejected.json). |
| D1.5 | Router top-k + softmax + scatter fold (Vulkan `MUL_MAT_ID_MUL` shape). | `LLAMACPP-VULKAN.md`; parent `PLAN-PAROQUANT2.md` §11.5.2. | neutral | **+0.5-1.5%** | neutral | Router currently uses one-block-per-expert producer for occupancy; naive fold collapses occupancy. Use cooperative producer + tail scatter pattern. | rejected | Opt-in cooperative producer (`HIPENGINE_PARO_ROUTER_TOPK_COOP=1`) preserves the 257-row logits grid and passes router/model correctness, but graph-replay decode regressed at both required shapes: 512/128 `115.931 -> 114.856 tok/s` (`-0.93%`) and 4K/128 `116.887 -> 116.106 tok/s` (`-0.67%`). Micro rocprof shows kernel-only router time improving (`logits 3.08 us + select 5.08 us` vs coop `7.08 us`), but the required counter memset/atomic tail and higher producer VGPR (`24 -> 40`) erase the win under graph replay. Artifact: [`2026-05-17-hipengine-qwen35-d15-router-coop-fold-rejected.json`](../benchmarks/results/2026-05-17-hipengine-qwen35-d15-router-coop-fold-rejected.json). |
| D1.6 | Decode k_proj + v_proj fused launch (parent `gemv_awq_dual_pack8` for QKV; extend to k/v stacked decode). | Parent `LESSONS-LEARNED.md` "Tiny c=1 projections are often launch-bound"; `PLAN-PAROQUANT2.md` §11.5.2. | neutral | opt-in correct but neutral/regressed; default unchanged | slight opt-in increase | Already fused in hipEngine for some pairs (full-attn Q/K, linear-attn QKV/Z). Use M.4 to enumerate remaining same-input pairs before coding. | rejected | Opt-in `HIPENGINE_PARO_FULL_ATTN_KV_PACK8_FUSED=1` preserves pack8/repacked layout by routing full-attn decode through single Q plus dual K/V into contiguous KV scratch, but it changes pairing rather than launch count. Correctness gate passed (`final_kl=0`, generated match), yet graph replay did not show a robust decode win: 512/128 `115.495 -> 115.627 tok/s` (`+0.11%`, noise) and 4K/128 `117.301 -> 117.053` (`-0.21%`), with slight tracked-peak increases. Default remains retained Q/K dual + single V. Artifact: [`2026-05-17-hipengine-qwen35-d16-kv-pack8-fusion-rejected.json`](../benchmarks/results/2026-05-17-hipengine-qwen35-d16-kv-pack8-fusion-rejected.json). |

### 6.2 D2 — W4 layout / Marlin-K vec8 port (the only retained parent W4 win)

`PLAN-PAROQUANT2.md` §11 documents that **most** Marlin-K work (FMA-only, Q8-FMA staging, sudot4,
all inner-loop ISA experiments) regressed or no-opped on the parent. The only retained win is the
**vec8 FMA inner loop + qweight-neutral replacement** (parent commits `7718fff` + `1522293`). The
rejected/parked siblings (B1-B7, C1-C5, D1-D6, E1-E6, F1-F4 in §12.2-§12.6) live in §11's
Do-Not-Chase list, not as candidate rows.

| ID | Candidate | Source / lineage | Expected prefill Δ | Expected decode Δ | Memory | Risk / prereqs | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D2.1 | Port parent Marlin-K vec8 FMA kernel + qweight-neutral repack-on-load to hipEngine non-expert `ParoQuantLinear` modules. | `docs/MARLIN.md` (full port plan); parent `nano-vllm-amd@7718fff`, `1522293`; `tools/paro_marlin_k_repack_reference.py`. | neutral/slightly positive measured | **+5.6%** at 512/128 and 4K/128 measured on hipEngine after qweight-neutral loading | -0.411 GiB tracked peak vs pack8 fallback in the retained hipEngine row | Preserves fast pack8 decode/prelude view through a zero-copy `qweight_pack8_decode` alias over `qweight_mk`; opt-out `HIPENGINE_PARO_MARLIN_K_REPLACE=0` remains available. | accepted | Retained: fallback pack8/raw -> default Marlin-K decode 512/128 `109.061 -> 115.137 tok/s` (+5.57%) and 4K/128 `110.088 -> 116.263` (+5.61%); fixture/graph gates pass (`max_kl=0.0396`, top-1 `100%`, final KL `0`). Artifact `benchmarks/results/2026-05-17-hipengine-qwen35-d21-marlin-k-qweight-neutral-diagnostic.json`. |
| D2.2 | Polish residual Marlin-K metadata residency (`qzeros_mk`, `scales_mk`) to 0 GiB. | `PLAN-PAROQUANT2.md` §12.1 A4 — `parked` upstream because every removal regressed decode. | n/a | neutral | -0.02 GiB | Parent already evaluated; reopen only if 24 GiB gate becomes tight. | parked | — |
| D2.3 | Activation pre-quantize to Q8 once per residual block; downstream W4 GEMV reads Q8 directly. | `PLAN-PAROQUANT2.md` §11.5.3 + §12.4 D1 (parked: needs ABI). Prerequisite for any future `sudot4` lane. | neutral | unknown; parent unable to land due to ABI gap | -0.05 GiB | Needs a torch-free per-residual Q8 ABI (act + per-chunk fp16 scale tensor). Not a quick win. | parked, blocked-by: torch-free Q8 activation ABI | — |

### 6.3 D3 — Long-context attention (32K, 128K)

| ID | Candidate | Source / lineage | Expected prefill Δ | Expected decode Δ | Memory | Risk / prereqs | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D3.1 | Port parent `nano-vllm-amd@52ebcd9` grouped-GQA paged split-K producer to hipEngine long-context decode. | `PLAN-LONGCONTEXT.md`; `LESSONS-LEARNED.md` "Grouped-GQA paged split-K producer"; hipEngine wrappers `qwen35_paged_full_attn_decode_split_k_gqa_*`. | neutral | **+5-12% at 32K/128**, +5-10% at 128K/128 (parent measured `+11.2%` 32K, `+11.0%` 128K) | neutral | Defaults on for Qwen3.5 GQA when `num_splits >= 64` or context >=4096; opt-out env. Validate per-shape correctness. | accepted | Retained: GQA opt-out -> default decode 32K/128 `70.064 -> 99.560 tok/s` (+42.1%), 128K/128 `30.789 -> 63.368` (+105.8%); graph fixture KL `0`; peak 32K `20.320 GiB`, 128K `23.288 GiB`. Artifact `benchmarks/results/2026-05-17-hipengine-qwen35-d31-d33-grouped-gqa-long-context-diagnostic.json`. |
| D3.2 | Re-tune `NANOVLLM_AMD_PAGED_ATTN_MAX_SPLITS=512` cap on the grouped-GQA producer (parent D3.1 follow-up). | Parent `LESSONS-LEARNED.md` "Post-GQA split-cap retune": 128K/128 decode +12.4% after grouped-GQA producer landed. | neutral | **+5-12% at 128K/128** on top of D3.1; smaller at 32K | neutral | Must be evaluated *after* D3.1 — same knob was rejected pre-GQA. Re-test cheap sweeps after every structural change. | accepted (512 cap rejected) | Retuned after D3.1: keep hipEngine default max-splits `4096`/no effective 128K cap. Forced `HIPENGINE_PAGED_ATTN_MAX_SPLITS=512` changed 128K/128 decode `63.368 -> 62.647 tok/s` (-1.14%) while saving only `0.034 GiB`, so reject the cap. Same artifact as D3.1. |
| D3.3 | Lower `NANOVLLM_PARO_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT` from `4096` to `1024` (parent retained at `78482b6`). | `PLAN-LONGCONTEXT.md` short/mid threshold sweep. | neutral | **+18% at 2K/128 and 3K/128** (parent measured); 4K/128 +0.7% | neutral | New default validation runs the graph/eager fixture at `1024/128`. | accepted | Retain default `HIPENGINE_PARO_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT=1024`: 1K/128 default split decode `113.242 tok/s` vs forced old threshold `92.486` (+22.4%). 512/128 and 4K/128 warmup-4 decode stayed within 1% vs D1.5 baseline (`-0.26%`, `-0.53%`). Same artifact as D3.1. |
| D3.4 | Online-softmax/FlashAttention-style grouped-GQA producer rewrite. | `LESSONS-LEARNED.md` "One-pass streaming needs a correctness fixture before E2E promotion" — parent attempt rejected (32K validation mismatch). | neutral | uncertain (+5-20%) but currently blocked on correctness | neutral | Needs an attention-output/logit fixture comparing producer outputs, split partials, top logits, greedy tokens against the retained exact path **before** E2E. | parked, blocked-by: correctness fixture per `LESSONS-LEARNED.md`. | — |
| D3.5 | INT8 paged-KV decode path. | `PLAN-LONGCONTEXT.md` INT8 KV status; focused staged plan in `docs/KVCACHE.md`; parent device-context INT8 only useful at very long context, neutral/negative at 32K. | neutral | -3-10% at 32K, +0-3% at 128K (parent measured) | -50% KV bytes | Must be no-shadow: no persistent BF16 KV backing or full-cache staging. Needs a fused gate-reduce + end-to-end quality check; LESSONS-LEARNED W8A8 NaN history is the cautionary tale. | deferred (post batch-1) | — |

### 6.4 D4 — Decode launch floor and replay graph hygiene

The "do not revisit multi-step graph replay" guardrail lives in §11's Do-Not-Chase list. M.3 already
produces the per-family dispatch count / kernel time / gap data the earlier `D4.1 replay-only
harness` row described, so it is folded into Lane M.

| ID | Candidate | Source / lineage | Expected prefill Δ | Expected decode Δ | Memory | Risk / prereqs | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D4.2 | Cap dispatches/token below 700 via batched D1.1-D1.6 fusions (Vulkan-style graph-level fusion). | `LLAMACPP-VULKAN.md`; parent `PLAN-PAROQUANT2.md` §11.5.2 (~660/tok current parent floor); Vulkan ~fewer than 200/tok at the same shape. | neutral | no code change; below-700 target unsupported by D1 dataflow | neutral | Per `LESSONS-LEARNED.md` "Output buffers alone are rarely enough", launch-count-only fusion under graph replay is sub-1%; must change data flow / reuse to count. | rejected | M.4 baseline is 877 dispatches/token, so `<700` needs ≥178 fewer dispatches/token. D1.2/D1.3/D1.6 add zero dataflow-safe count reduction; D1.1 regressed, D1.4 safe fold is already default while direct down+combine was parent-rejected, and D1.5 is launch-count neutral with counter memset and regresses. Even optimistic rejected-path accounting lands around 767 dispatches/token. See `benchmarks/results/2026-05-17-hipengine-qwen35-d42-dispatch-cap-rejected.json`. |
| D4.4 | Per-kernel `__launch_bounds__` retune after rotation/RMSNorm/W4 fusion changes (LESSONS-LEARNED "Runtime thread-count knobs must honor kernel launch bounds"). | `LESSONS-LEARNED.md` Task 23 audit. | neutral | no code change; retune not triggered by retained default fusions | neutral | Must cross-check against statically allocated shared memory + reduction scratch; never accept a knob value that bypasses `__launch_bounds__`. | deferred | Static audit found no wrapper bypass of source launch bounds: pack8/selected/Marlin-K launch ≤128 under `__launch_bounds__(128,4)`, compact WMMA/fusedW4 launch 32 under `__launch_bounds__(32,*)`. D1.1/D1.2/D1.3/D1.6 did not retain a default kernel that changes the resource envelope; prior stricter compact-WMMA/fusedW4 launch-bound trials regressed or spilled. Reopen only after a default fusion is retained or parent kernel R&D produces stable evidence. See `benchmarks/results/2026-05-17-hipengine-qwen35-d44-launch-bounds-deferred.json`. |

### 6.5 D5 — Decode glue / secondary kernel audits

These are smaller than the W4/attention headline lanes, but M.4 says they still matter:
W8A16 is ~16% of short-context decode kernel time, while GDN/router/rotation/RMSNorm/glue compound
into the dispatch floor.

| ID | Candidate | Source / lineage | Expected prefill Δ | Expected decode Δ | Memory | Risk / prereqs | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D5.1 | Audit `qwen35_gdn_recurrent_rmsnorm_gate_lowp_kernel` for vec8 / occupancy headroom on c=1 decode. | `docs/KERNELS.md` rocprof note; ROOFLINE §9 RDNA3 occupancy rules. | neutral | no retained change | neutral | Stop condition met: Qwen3.5 shape already has 8-way head-k loops, 128 threads exactly covering 128 value lanes, `VGPR=56`, and `Scratch_Size=0`; deeper barrier/multi-value rewrites belong in parent kernel R&D with stronger correctness proof. | accepted (stop) | `benchmarks/results/2026-05-17-hipengine-qwen35-d51-gdn-decode-audit.json`; rocprof decode-shape probe: BF16/FP16 medians `8.760/8.720 us`, `VGPR=56`, scratch `0`, workgroup `128`; smoke and graph fixture gates pass (`final_kl=0`). |
| D5.2 | Audit W8A16 decode kernels (`w8a16_linear_kernel`, `w8a16_linear_lowp_out_kernel`) for tile/occupancy headroom; **not** fused argmax. | M.4: W8A16 family is 15.7% / 15.7% / 13.4% of 512/4K/32K decode kernel time. Parent notes fused `lm_head + argmax` is not a lever, but the W8A16 matvec itself is still large. | neutral | no retained change | neutral | Stop condition met: ISA metadata has no spills/scratch and thread/tile probes show current defaults are best. | accepted (stop) | `benchmarks/results/2026-05-17-hipengine-qwen35-d52-w8a16-decode-audit.json`; keep lm-head W8A16 at 128 threads, shared lowp W8A16 at 64 threads; reject c=1 fused shared gate/up+SiLU replacement. |
| D5.3 | Router top-k cooperative producer (one workgroup all experts) — avoid the naive logits+top-k fusion that collapses occupancy. | `LESSONS-LEARNED.md` "Router fusion is the opposite case"; current router uses one block per expert. | neutral | +0.5-1% | neutral | Without inter-block sync, naive same-kernel fused top-k is racy or collapses occupancy. Use cooperative producer pattern. | rejected | Covered by D1.5: the atomic last-producer pattern preserves logits occupancy and is correct, but the counter reset/atomic tail regresses 512/128 and 4K/128 graph replay. Reopen only for a graph-level fusion or persistent initialized counter design that removes the extra memset node without racing. |
| D5.4 | Linear-attention A/B decode same-input fusion. | LESSONS-LEARNED retained win precedent. | neutral | +0.6-1% (parent measured) | neutral | Already live in hipEngine: `project_linear_attention_ab_fp16` dispatches `dense_dual_gemv_out_fp16` on the `tokens == 1` decode path; prefill keeps two unfused GEMVs, and the P1.1 rocBLAS bulk replacement was rejected for the skinny N=32 shape. | accepted | `hipengine/runtime/qwen35_paro.py:2795` |

### 6.6 D6 — DFlash / MTP / multi-token speculative decode

| ID | Candidate | Source / lineage | Expected prefill Δ | Expected decode Δ | Memory | Risk / prereqs | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D6.1 | Native compact/c-aware target verifier + GPU accept summaries (`DFLASH.md` Task #15). | `docs/DFLASH.md`; parent Python harness ~0.96× AR with 1.20 verify rows/output. | neutral | -3% to +50% (depends on verify cost; parent Python `~0.963×` AR; native target is ≥ 1.1×) | +1-2 GiB for verify scratch | Heavy infrastructure work; speculative path is `blocked` until verifier lands. | deferred (Phase 4) | `benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json` |
| D6.2 | MTP draft plugin on the shared DFlash verifier (`MTP.md`). | `docs/MTP.md`; parent B=5 native row `83.88 tok/s` vs AR `120.04 tok/s` = `0.699×`. | neutral | -30% to +20% (depends on acceptance + verify cost) | +0.5 GiB MTP head | DFlash verifier must land first; otherwise this reproduces the parent's ~0.7× AR result. | deferred (post-DFlash) | — |

---

## 7. Lane A — Memory (a feature, not a casualty)

hipEngine currently beats the parent peak-memory row on every retained comparison context. Any
candidate above must preserve this. The rows below are **guardrails**, not candidates that move.

| ID | Guardrail | Current value | Risk | Status |
| --- | --- | --- | --- | --- |
| A.1 | Default 512/128 peak | 18.58 GiB (parent 18.86 GiB) | New BLAS/WMMA bulk paths must report extra scratch and not duplicate W4 layouts. | accepted |
| A.2 | Default 4K/128 peak | 19.88 GiB (parent 21.64 GiB) | Same. | accepted |
| A.3 | Default 32K/128 peak | 20.69 GiB (parent 21.37 GiB) | Long-context chunked policy is the default; do not silently revert. | accepted |
| A.4 | Default 128K/128 peak | 23.66 GiB (parent 27.42 GiB) | Stay below 24 GiB; the differentiator vs parent. | accepted |
| A.5 | Alias ownership for qweight views (Marlin-K-style zero-copy) | n/a | Aliases must be non-owning tensors; never create two owning `DeviceTensorAllocation` records for the same pointer. | accepted |

---

## 8. Lane W — Compiler / build profile sweeps

Cheap to run, sometimes large. Per LESSONS-LEARNED, every sweep must be paired with `Scratch_Size=0`
and VGPR audits, and treated as **per-kernel build flag**, not a global default.

| ID | Candidate | Source / lineage | Expected prefill Δ | Expected decode Δ | Memory | Risk / prereqs | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W.1 | `-mllvm -amdgpu-unroll-threshold-local=600` on hot prefill kernels (linear-attn GDN, MoE compact WMMA, full-attn prelude, shared expert). | `PR_COMMENT-llamacpp-hip-unroll600.md` (llama.cpp +166% prefill, near-neutral decode); parent E1 (`PLAN-PAROQUANT2.md` §12.5): neutral on Marlin-K FMA. | neutral/noisy measured | neutral/noisy measured | neutral | Keep the current default, but stop treating it as an optimization lever. The no-unroll ablation preserves `-mcumode`; hot-library metadata stayed `private_segment_fixed_size=0`, no SGPR/VGPR spills, and unchanged max VGPR. | accepted (neutral default) | `benchmarks/results/2026-05-17-hipengine-qwen35-qwen36-w1-unroll600-ablation-diagnostic.json` |
| W.2 | `-mcumode` build profile on hot decode kernels. | `PR_COMMENT-llamacpp-hip-unroll600.md` table; ROOFLINE §1.1. | n/a (default) | n/a | n/a | Already in `hipengine/core/build.py:47` default flags and `kernels/hip_gfx1100/wmma/paro_awq_wmma.py` extra flags. | accepted | `hipengine/core/build.py:47` |
| W.3 | Per-kernel `__attribute__((amdgpu_waves_per_eu(...)))` retune after rotation/RMSNorm fusion lands. | Parent E4 rejected for Marlin-K but landed kernels in hipEngine have different VGPR profile. | neutral | +0-2% | neutral | Re-evaluate per kernel; do not blanket-apply. | pending | — |

---

## 9. Lane S — Serving / c>N (deferred until batch-1 is green)

c=2/4/8 native compact prefill correctness is already accepted (`benchmarks/README.md`); the
remaining work is decode and benchmark contract. Defer until the batch-1 board is green.

| ID | Candidate | Source / lineage | Expected aggregate decode Δ | Memory | Risk / prereqs | Status |
| --- | --- | --- | --- | --- | --- | --- |
| S.1 | c-aware decode graph buckets with fixed active-slot metadata (replace `scheduler_serial_slot_bridge`). | `PREFILL.md` Lane 5; `benchmarks/results/2026-05-15-hipengine-qwen35-c8-scheduler-serial-bench-blocked.json`. | +50-100% aggregate tok/s at c=8 over serial bridge | +0.5-1 GiB per c step | Needs `KVLiveSpans` per-slot + per-slot decode graph keys. | deferred |
| S.2 | c=N benchmark protocol for `512/128` rows. | `docs/BENCHMARK.md` c=N protocol. | n/a | n/a | Tied to S.1 throughput claim eligibility. | deferred |
| S.3 | RadixCache prefix caching (Phase 3). | `docs/PLAN.md` Phase 3. | depends on prefix-hit ratio | varies | Mini-sglang `RadixCache` is the reference. | deferred |

---

## 10. Lane K — Other quant formats and models

Out of scope for the Qwen3.5/PARO batch-1 board, listed so future grinds inherit the table layout.

| ID | Candidate | Source / lineage | Expected prefill Δ | Expected decode Δ | Memory | Status |
| --- | --- | --- | --- | --- | --- | --- |
| K.1 | GGUF Q4_K_M / Q8_0 loader + `w4_gguf` quant plugin. | `docs/GGUF.md`; llama.cpp HIP/Vulkan parity. | n/a | n/a (separate model file) | n/a | deferred (Phase 2) |
| K.2 | Native HIP FA-2 forward kernel registered under a per-shape variant key (replaces AOTriton at gfx1100 fixed shape). | `docs/PREFILL.md` "Recommended phased plan" Phase 3. | +0-15% over AOTriton | neutral | -0.1 GiB | deferred (after P1/P2/P3/D1/D2/D3 settle) |
| K.3 | Strix Halo / gfx1151 backend retune. | `docs/PLAN.md` Multi-backend tree; `hipengine/kernels/hip_gfx1151/`. | n/a | n/a | n/a | deferred (Phase 5+) |

---

## 11. Do-not-chase list

Parent-rejected / hipEngine-out-of-scope. Do not open a multiloop on these without **new** profile
evidence (e.g. structural changes that invalidate the earlier rejection).

| Avoid | Why | Source |
| --- | --- | --- |
| Naive `sudot4`/dp4a over current PARO/AWQ layout | 3.92-9.72× slower than tuned FMA; layout + activation staging dominate. | `PLAN-PAROQUANT.md` two scratch trials, `PLAN-PAROQUANT2.md` §11.3.3 |
| LDS staging as the default hypothesis | RDNA3 parent evidence repeatedly found barrier/occupancy costs > reuse benefits. | `LESSONS-LEARNED.md` "LDS is not free", `PLAN-PAROQUANT2.md` E6 |
| Multi-step graph replay | Parent tested 1/2/4/8/16; no reliable gain; 4K/4K diverged at token 581. | `LESSONS-LEARNED.md`, `OPTIMAL.md` |
| Thread-count sweeps without source/profile justification | Many regress; `__launch_bounds__` and LDS scratch must be checked first. | `LESSONS-LEARNED.md` Task 23 |
| Fusion that abandons pack8/repacked fast layout | Saving one launch can lose more in memory layout. | `LESSONS-LEARNED.md` `MOE_GATE_UP_ROTATE_FUSED` |
| Address-only V-loop polish on long-attention | Parent rejected; next attention attempt needs real online/tiled or parallel accumulation structure. | `LESSONS-LEARNED.md` |
| Perf rows without generated-token / logit sanity | Previous fast rows were invalid when recurrence/RoPE/state was wrong (Qwen RoPE, W8A8 NaN). | `LESSONS-LEARNED.md` "Fast rows are invalid until output sanity proves they are real" |
| Hand-rolled FA-2 before AOTriton is wrapped and used as oracle | Iters 1-49 demonstrated the cost of optimizing without a perf oracle. | `docs/PREFILL.md` "Explicit non-goals" |
| Marlin-K B1-B7 / C1-C5 / E1-E6 / F1/F2 inner-loop experiments | All rejected upstream; documented evidence. | `PLAN-PAROQUANT2.md` §12.2-§12.6 |
| Cargo-cult `TORCH_BLAS_PREFER_HIPBLASLT=1` | rocBLAS beat hipBLASLt on tested BF16 GEMM shapes on this W7900 stack. | `LESSONS-LEARNED.md` "rocBLAS is currently faster than hipBLASLt" |

---

## 12. First concrete punchlist (next 4-6 iterations)

Order is chosen from the M.3/M.4 profile, not from parent folklore.

P1.1-P1.4 are closed: rocBLAS A/B rejected; legacy shared-expert token tiling retained; selected-MoE compact WMMA threshold retained at tokens>=2.

1. **D2.1 + D5.2** — Marlin-K vec8/qweight-neutral port plus a W8A16 decode-kernel audit. Measured
   short-context decode kernel time is ~43% W4/selected-MoE GEMV and ~16% W8A16.
2. **D1.1 + D1.4 + D1.5** — Rotation-into-projection, selected-MoE post-op fold, and router fold.
   These target the 160 rotation calls/token, 40 combine calls/token, and 80 router calls/token.
3. **D3.1 + D3.2 + D3.3** — Grouped-GQA producer port + split-cap retune + paged-decode min-context
   threshold. Attention jumps from ~11% short-context to 23% at 32K.

Re-score the board after each retained row:

```bash
python3 scripts/qwen35_compare_tables.py all
```

The batch-1 board is **green** when, in the comparison table:

- prefill beats parent and llama.cpp HIP/Vulkan at 512/4K/32K/128K;
- decode beats parent and llama.cpp HIP/Vulkan at 512/4K/32K/128K;
- peak memory remains below parent on rows where parent memory is known, and below 24 GiB on
  short/mid contexts.

---

## 13. Reference map

| Topic | Primary reference |
| --- | --- |
| Parent optimal flags/rows | `~/amd-gpu-tuning/docs/OPTIMAL.md` |
| hipEngine prefill architecture + Amdahl audit | `docs/PREFILL.md` |
| hipEngine kernel catalog and port gates | `docs/KERNELS.md` |
| RDNA3 performance model | `docs/ROOFLINE.md` and `~/amd-gpu-tuning/docs/ROOFLINE.md` |
| Benchmark protocol and artifact rules | `docs/BENCHMARK.md` |
| Marlin-K W4 layout port plan | `docs/MARLIN.md` |
| DFlash / MTP speculative plans | `docs/DFLASH.md`, `docs/MTP.md` |
| GGUF quant lane | `docs/GGUF.md` |
| Current comparison rows artifact | `benchmarks/results/2026-05-16-hipengine-qwen35-comparison-tables-diagnostic.json` |
| llama.cpp HIP/Vulkan split rows + Vulkan source analysis | `~/amd-gpu-tuning/PLAN-LONGCONTEXT.md`, `~/amd-gpu-tuning/docs/LLAMACPP-VULKAN.md` |
| Compiler flag evidence | `~/amd-gpu-tuning/PR_COMMENT-llamacpp-hip-unroll600.md` |
| Parent ParoQuant 2 punchlist (~150 candidates, most resolved) | `~/amd-gpu-tuning/PLAN-PAROQUANT2.md` §12 |
| Parent ParoQuant forward plan | `~/amd-gpu-tuning/PLAN-PAROQUANT.md` |
| Long-context evidence | `~/amd-gpu-tuning/PLAN-LONGCONTEXT.md` |
| Hard-won rules and parent negative results | `~/amd-gpu-tuning/LESSONS-LEARNED.md` |
