# hipENGINE Optimization Grind Plan

Status: 2026-05-17 (reorganized as per-category candidate tables).

Scope: Qwen3.5-35B-A3B-PARO `w4_paro` on W7900/gfx1100, batch-1 prompt/decode rows first.
Goal: close every prefill/decode gap to source-lineage `nano-vllm-amd` parent **and** llama.cpp
HIP/Vulkan on retained comparison shapes (512/128, 4K/128, 32K/128, 128K/128), while preserving
hipENGINE's existing peak-memory advantage and torch-free runtime invariant.

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

- **pending** — open, not yet attempted in hipENGINE.
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

Current hipENGINE rows are still **diagnostic resident-runner rows** (`performance_claim=false`),
not accepted public `LLM.generate()` throughput rows. They use:

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

Do not start with another blind kernel multiloop. First promote the measurement harness and capture
matched ROCTX / `rocprofv3 --kernel-trace` profiles for every comparison row. The remaining short/mid
prefill miss is most likely bulk dense/shared-expert GEMV-shaped work (Lane P1) plus AOTriton glue
(Lane P2); the parent runs these as framework `F.linear(...)` GEMMs and llama.cpp HIP's prefill jumps
+166% with one compiler flag. The decode miss is the compound of replay dispatch fanout (~660-900
dispatches/token), rotation + RMSNorm boundary launches (combined ~20% of decode bucket per the
parent rocprof audit), and a small W4-launch-floor tail; we attack each in audit-first order and
land the parent's already-validated wins (Marlin-K vec8 layout, fused selected-MoE shared
gate-sigmoid skip) where they port cleanly. Long-context is mostly chunking-bound and already
parity/ahead of parent on prefill — the next 32K/128K decode levers are attention split-cap
retuning and the grouped-GQA producer family. Memory stays a feature: every candidate must keep
512/4K under 24 GiB and must not reintroduce duplicate W4 qweight residency.

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

## 4. Lane M — Measurement and protocol promotion (blocks everything)

These are not optimizations. They make every other lane's status field meaningful. **Do these
first**; lanes P/D/A/W are wasted iterations otherwise.

| ID | Candidate | Source / lineage | Expected prefill Δ | Expected decode Δ | Memory | Risk / prereqs | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M.1 | Promote first accepted `LLM.generate()` row for Qwen3.5/PARO 512/128 and 4K/128 with same flags as the comparison-table diagnostic. | `docs/BENCHMARK.md` acceptance protocol; current rows live in `benchmarks/results/2026-05-16-hipengine-qwen35-comparison-tables-diagnostic.json` and are `performance_claim=false`. | n/a | n/a | n/a | Needs repeated-run policy + retained sample/decode-graph gate per `docs/BENCHMARK.md`. | pending | — |
| M.2 | Add a `scripts/qwen35_compare_tables.py` auto-refresh hook from the current retained artifact. | Today the script is intentionally hardcoded; refresh it whenever a retained row moves. | n/a | n/a | n/a | Must keep the script the human checkpoint per lane. | pending | — |
| M.3 | Collect matched `rocprofv3 --kernel-trace` + ROCTX profiles for hipENGINE 512/128, 4K/128, 32K/128 with the comparison-table flags. Retain only compact summaries under `benchmarks/results/`. | Templates in `docs/OPTIMIZE.md` (earlier version); `docs/KERNELS.md` "Pre-optimization audit"; parent `~/amd-gpu-tuning/KERNEL_BLOCKERS.md` rocprof shim. | n/a | n/a | n/a | Verify `rocprofv3` is not blocked by the `spirv-expand-step` LLVM crash on this host (`tools/rocprof_torch_site/sitecustomize.py` shim if needed). | pending | — |
| M.4 | Build a per-bucket Amdahl table for hipENGINE 512/128 decode (replay-only window, after subtracting warmup/capture/prefill) — analogous to the parent rocprof tail audit in `PLAN-PAROQUANT2.md` §11.9. | Parent rocprof tail audit `artifacts/paroquant2_rocprof_audit_20260515_iter30/`; ROOFLINE §5. | n/a | n/a | n/a | Decode replay window must not include eager-validation steps. | pending | — |
| M.5 | Match-baseline llama.cpp Vulkan Q4_K_M on **this** machine at 4K/4K to confirm/refute the 122.2 tok/s ceiling. | Parent `PLAN-PAROQUANT2.md` §11.5.7 and §12.6 F4: local Vulkan rerun showed Qwen3.6-35B-A3B MXFP4 4K/128 tg = 112.12 tok/s, **not** the +9% Vulkan headline. Need a Q4_K_M GGUF for parity. | n/a | n/a | n/a | Headline decode-vs-Vulkan target may be smaller (~+5%) than the table in §1.3 suggests. | pending | — |

Until M.1-M.5 land, no row below can move to `accepted`; only `in-progress`/`rejected`/`parked` is valid.

---

## 5. Lane P — Prefill

The Amdahl break-up of 512/128 prefill (40-layer hipENGINE rocprof, `docs/PREFILL.md` §"Optimization
diagnosis 2026-05-16"):

```
qwen35_gdn_prefill_recurrent_k2          17.9% (linear-attn recurrent state)
gemm_awq_selected_dual_pack8_wmma         14.8% (MoE selected pack8 WMMA)
qwen35_paged_full_attn_prefill_gqa        11.4% (AOTriton at 4K closes this)
awq_fusedw4_prefill_fp16_kernel<*,true>   10.1% (Q/K stacked W4 prefill)
gemm_awq_selected_pack8_wmma               8.5% (MoE singleton pack8)
w8a16_shared_down_combine_residual         7.0% (shared-expert down + combine)
w8a16_shared_gate_up_silu                  6.8% (shared-expert gate+up)
awq_fusedw4_prefill_fp16<*,false>          6.4% (V/O/linear-out W4 prefill)
paro_rotate1                               4.0% (PARO rotation)
qwen35_router_logits_token_tile            3.8% (router logits)
```

Parent docs and `docs/PREFILL.md` agree the **per-layer non-attention bulk work** is the residual
gap (P1 lanes below), with AOTriton already closing the old 4K full-attention cliff (P2) and the
linear-attention/MoE rms+rotate boundaries open for fusion (P3).

### 5.1 P1 — Bulk dense / shared-expert GEMM-shaped paths (parent uses `F.linear(...)`)

Parent multi-row `ParoQuantDenseLinear.forward(...)` and `ParoQuantSharedExpert.forward(...)` fall
through to `F.linear(...)` (rocBLAS/Tensile bulk GEMM). hipENGINE currently uses row-shape GEMV
kernels for the same work. This is the leading P0 hypothesis from `docs/PREFILL.md`.

| ID | Candidate | Source / lineage | Expected prefill Δ | Expected decode Δ | Memory | Risk / prereqs | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1.1 | Torch-free hipBLASLt/rocBLAS ctypes wrapper for linear-attention A/B BF16/FP16 dense projections (replace `project_linear_attention_ab_fp16(...)` row GEMV pair). | `docs/PREFILL.md` P0; parent ledger `native_aux_dense_linear_calls=280`. | +5-12% on 512/128, +3-7% on 4K/128 | neutral-to-+1% (decode path uses pack8) | neutral; one new BLAS handle | LESSONS-LEARNED "rocBLAS is currently faster than hipBLASLt on this W7900 stack" — pin rocBLAS, not hipBLASLt-preferred. Must not import torch. | pending | — |
| P1.2 | Torch-free bulk dense path for shared expert gate/up SiLU during prefill (replace `w8a16_shared_gate_up_silu_fp16(...)` for `tokens >= threshold`). | `docs/PREFILL.md` P0; parent ledger `native_shared_expert_dense_calls=80`; parent W8A16 prefill path. | +3-7% | neutral (decode uses unfused W8A16 fallback) | -0.0-0.2 GiB (avoids `shared_intermediate` scratch growth) | Keep existing tiled W8A16 path as fallback. Quantized shared expert is currently part of the memory advantage; do not lose it on decode. | pending | — |
| P1.3 | Torch-free bulk dense shared down + combine for prefill (`w8a16_shared_down_combine_residual_fp16`). | Same as P1.2. | +2-5% | neutral | neutral | Combine kernel currently fuses sigmoid + residual; bulk dense version needs the same fused tail or an equivalent post-pass. | pending | — |
| P1.4 | Empirical crossover threshold for compact WMMA vs bulk GEMM/GEMV across linear-attn A/B and shared expert (analogous to parent `WMMA_MIN_TOKENS=64`). | Parent `LESSONS-LEARNED.md` "Compact WMMA prefill crossover" + `docs/OPTIMAL.md`. | enables P1.1-P1.3 wins | neutral | neutral | Must hot-path-dispatch by token count without `if quant ==` branches. | pending | — |
| P1.5 | Sweep `-mllvm -amdgpu-unroll-threshold-local=600` build flag on the hipENGINE prefill kernels. | `~/amd-gpu-tuning/PR_COMMENT-llamacpp-hip-unroll600.md`: llama.cpp HIP pp512 **+166%** at this flag; multi-model +6-232%. Parent PAROQUANT trial was **neutral** on `v8` kernels (E1 in `PLAN-PAROQUANT2.md` §12). | +10-40% on 512/128 prefill **if** it triggers; could also be neutral | neutral | neutral | Per-kernel build profile experiment; not a default. Verify no Scratch_Size spills. Parent E1 showed neutral on Marlin-K FMA — hipENGINE has more shape variety, so worth retesting. | pending | — |
| P1.6 | Selective `-mcumode` build profile on hot prefill kernels. | `PR_COMMENT-llamacpp-hip-unroll600.md` threshold bracket table; parent ROOFLINE notes CU mode is build-profile dependent on gfx1100. | +0-2% on top of P1.5 | neutral | neutral | Test only after P1.5 lands or rejects. | pending | — |

### 5.2 P2 — AOTriton glue and full-attention prelude

| ID | Candidate | Source / lineage | Expected prefill Δ | Expected decode Δ | Memory | Risk / prereqs | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P2.1 | AOTriton `attn_fwd_compact_varlen` Q/gate + K prelude fusion: read FP16 `Q\|gate` and FP16 K directly, emit gate FP16 + BF16 Q + FP32/BF16 K in one kernel; removes split + key cast launches + `query_raw`/`key_raw` scratch. | `docs/PREFILL.md` low-risk fusion audit, candidate #3; parent grouped-prefill kernel template. | +1-3% at 512/128, +1-2% at 4K/128 (mostly latency hiding) | neutral | -0.05-0.2 GiB | Previous AOTriton cast/gate fusions were throughput-neutral/slightly negative; require profile proof before spending more than a small spike. | pending | — |
| P2.2 | AOTriton 0.12 / V3 upgrade and shape-streaming kernel pull (`PLAN-PAROQUANT2.md` style "stream only used variants"). | `docs/PREFILL.md` `aotriton_release.toml`; AOTriton 0.11.2b baseline is vendored through Git LFS with only the 12 BF16 head-dim-256 gfx11xx forward images hipENGINE needs. | +0-3% if newer kernels are faster on Qwen3.5 shape | neutral | -0.1 GiB if shape streaming lands | Wrapper ABI must stay stable; pin manifest is the contract. | parked, blocked-by: M.5 measurement of upstream Vulkan ceiling on this machine. | — |
| P2.3 | Keep `--attn-aotriton-min-tokens 512` as the code/default deployment policy and keep native attention as a diagnostic fallback only. | `docs/PREFILL.md` AOTriton sweep table (4K native = 662 tok/s; threshold-512 AOTriton = 2346 tok/s). | n/a (already deployed) | neutral | n/a | AOTriton is a baseline runtime dependency; `PrefillConfig` and benchmark defaults use threshold 512, while `0` remains an explicit diagnostic override. | accepted (deployment policy) | `benchmarks/results/2026-05-16-hipengine-qwen35-aotriton-threshold-sweep-diagnostic.json` |

### 5.3 P3 — Boundary fusion for linear-attention and MoE prefill

These three are listed in `docs/PREFILL.md` "Additional low-risk prefill fusion audit (2026-05-16)"
as the recommended order; all are validated by source structure, not measurement yet.

| ID | Candidate | Source / lineage | Expected prefill Δ | Expected decode Δ | Memory | Risk / prereqs | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P3.1 | Fuse `qwen35_gdn_prefill_rmsnorm_gate_fp16` + `paro_rotate1_fp16` + `awq_fusedw4_prefill_strided_fp16` tail into a single GDN-out kernel; removes one launch and `recurrent_bf16` materialization on all 30 linear-attn layers. | `docs/PREFILL.md` audit candidate #1; `head_v_dim == group_size` for Qwen3.5/PARO so the shape is safe. | +2-4% at 4K/128, +1-2% at 32K/128 | neutral | -0.05-0.15 GiB (drop `recurrent_bf16` scratch) | Keep two-kernel path as fallback for non-Qwen3.5 shapes. Fixture gates must pass per-layer. | pending | — |
| P3.2 | Prefill-only router shared-gate `sigmoid()` fused into top-k path so grouped prefill skips `w8a16_shared_gate_sigmoid_fp32`. | `docs/PREFILL.md` audit candidate #2. | +0.3-1% at 512/128, +0.5-1.5% at 4K/128 | must not change c=1 decode (combine kernel applies sigmoid itself) | neutral | The c=1 `weighted_sum_shared_gate_combine_residual_*` kernel expects raw shared-gate logits; do not flip semantics for the decode path. | pending | — |
| P3.3 | MoE metadata fanout collapse: combine `moe_group_prefix` + `moe_wmma_tile_map` and initialize `scatter_offsets`/`tile_expert` in the same small metadata kernel. | `docs/PREFILL.md` audit candidate #5. | +0-0.5% | neutral | neutral | Small payoff; do after profiler confirms metadata is visible. | parked, policy: small payoff per source audit, do after M.3 says metadata bucket is non-negligible. | — |
| P3.4 | Templated FP16-input segment conv wrapper for packed linear-attention path; remove `fp16_to_f32` cast and `qkv_f32` scratch in c>N packed prefill. | `docs/PREFILL.md` audit candidate #4. | +0-1% c=1; bigger lift on c>N | neutral | -0.05 GiB on c>N | Affects compact c>N more than batch-1; schedule after the batch-1 board closes. | deferred (Lane S, c>N) | — |

### 5.4 P4 — Native full-attention prefill kernel (long-term replacement of AOTriton)

| ID | Candidate | Source / lineage | Expected prefill Δ | Expected decode Δ | Memory | Risk / prereqs | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P4.1 | Hand-rolled HIP Flash-Attention-2 forward kernel with WMMA tile, online softmax, GQA reuse, in-kernel fused gate post-pass; uses AOTriton output as oracle. | `docs/PREFILL.md` "Options for fast prefill attention without `torch`"; `docs/KERNELS.md` planned `qwen35_causal_gqa_gate_fp16`. | +0-15% over AOTriton at the Qwen3.5 fixed shape; mostly a packaging/portability win | neutral | -0.1 GiB (no AOTriton image cache) | 3-6 weeks. Do **not** start before P1/P2/P3 are settled. Per `docs/PREFILL.md` "Explicit non-goals", **do not** start until AOTriton is wrapped and used as oracle. | deferred (Phase 4+) | — |

### 5.5 Long-context prefill (32K/128, 128K/128)

| ID | Candidate | Source / lineage | Expected prefill Δ | Expected decode Δ | Memory | Risk / prereqs | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P5.1 | Already-landed: chunked long-context prefill (`PrefillConfig.linear_chunk_size`, `moe_chunk_size`, `full_attn_query/post/rope_chunk_size`). | `docs/PREFILL.md` chunking checkpoint; `benchmarks/results/2026-05-16-hipengine-qwen35-prefill-chunking-diagnostic.json`. | +5.7% (4K), +8.9% (32K), unblocks 128K | neutral | -1.8 GiB (4K), -14.4 GiB (32K), -3.8 GiB (128K vs parent) | None — landed; keep as default policy. | accepted | `WORKLOG.md` 2026-05-16 chunked rerun. |
| P5.2 | Long-context chunk-size auto-tuner that respects per-shape memory budget instead of static defaults. | Parent `OPTIMAL.md` long-prefill overrides; current hipENGINE uses static defaults. | +1-3% at 32K/128 | neutral | -0.3 GiB at 32K, -0.5 GiB at 128K | Must verify same fixture gates pass for each chunk size; do not regress 512/4K. | pending | — |

---

## 6. Lane D — Decode

Per-bucket Amdahl break-up of the parent rocprof tail audit (0.8B 4K/128, `PLAN-PAROQUANT2.md`
§11.9; treat as steering, not as the hipENGINE bucket map until M.4 lands):

```
paged full-attention decode      22.8%
small glue / elementwise         17.3%   <-- launch-fanout pile
rotation / RoPE                  14.5%
W4 dual pack8 GEMV               12.9%
W4 single pack8 GEMV             10.2%
W8A16 lm-head + dense linear      8.4%
RMSNorm / add-RMSNorm             6.9%
linear-attention GDN decode       6.9%
```

Combined "fusion-eligible boundary" buckets (rotation + RMSNorm + small glue) are ~39%, larger than
W4 GEMV (~23%). The W4 inner loop is **not** the headline lever — boundary fusion and dispatch
reduction are.

### 6.1 D1 — Dispatch reduction and boundary fusion (largest non-W4 buckets)

| ID | Candidate | Source / lineage | Expected prefill Δ | Expected decode Δ | Memory | Risk / prereqs | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D1.1 | Fuse rotation into the same-input W4 GEMV producer for paired q/k/v stacked attention projections. | Parent `gemv_awq_selected_dual_pack8_strided_rotate_out` precedent (already ported as a kernel surface in hipENGINE, but selected/strided only); `PLAN-PAROQUANT2.md` D4 rejected at per-output-pack granularity but valid at projection granularity. | neutral | **+2-4% at 512/128**, +1-3% at 4K/128 | neutral | Must rotate **once** per residual block (not once per output pack); D4 rejection in `PLAN-PAROQUANT2.md` §12.4 is the cautionary tale. | pending | — |
| D1.2 | Fuse `paro_rmsnorm` / `add_rmsnorm` producer into the following projection where the normalized vector is single-use. | Parent linear-attn QKV/Z + full-attn Q/K precedents (`afb7b16`, `FULL_ATTN_QK_PACK8_FUSED`); extend to V projection (full-attn) and shared-expert projection. | neutral | **+1-2%** | neutral | Must keep fast pack8/repacked layout; per LESSONS-LEARNED "Output buffers alone are rarely enough under graph replay" — only count wins that change arithmetic / data reuse. | pending | — |
| D1.3 | Same-input projection fusion for remaining adjacent c=1 GEMV pairs not yet fused (audit M.4 result first). | Parent linear-attn `LINEAR_ATTN_QKV_Z_PACK8_FUSED` `+0.74%`, full-attn `FULL_ATTN_QK_PACK8_FUSED` `+0.41%`; LESSONS-LEARNED Q/K/V pack8 widen was **rejected** because it abandoned pack8 layout. | neutral | **+0.5-1%** per fusion, compounds | neutral | Must preserve pack8 / repacked layout. Pure launch-count fusion is sub-1% under graph replay; only counts if arithmetic/reuse also improves. | pending | — |
| D1.4 | Selected-MoE post-op fold: combine selected-expert weighted-sum + add + sigmoid + residual into one kernel (Vulkan `MUL_MAT_ID_ADD_ID_MUL` shape). | Parent `selected-MoE silu/down-rotation fusion` (`fbff0fe`) precedent; `PLAN-PAROQUANT2.md` §11.5.2; `LLAMACPP-VULKAN.md` graph fusion analysis. | neutral | **+1-2% at 512/128, +1% at 4K/128** | neutral | Sorted-lane semantics: weighted scatter cannot naively fuse without atomics or layout change. Re-read parent F2 WMMA M16 lesson before attempting larger combined kernels. | pending | — |
| D1.5 | Router top-k + softmax + scatter fold (Vulkan `MUL_MAT_ID_MUL` shape). | `LLAMACPP-VULKAN.md`; parent `PLAN-PAROQUANT2.md` §11.5.2. | neutral | **+0.5-1.5%** | neutral | Router currently uses one-block-per-expert producer for occupancy; naive fold collapses occupancy. Use cooperative producer + tail scatter pattern. | pending | — |
| D1.6 | Decode k_proj + v_proj fused launch (parent `gemv_awq_dual_pack8` for QKV; extend to k/v stacked decode). | Parent `LESSONS-LEARNED.md` "Tiny c=1 projections are often launch-bound"; `PLAN-PAROQUANT2.md` §11.5.2. | neutral | **+1-1.5%** | neutral | Already fused in hipENGINE for some pairs (full-attn Q/K, linear-attn QKV/Z). Audit remaining same-input pairs via M.4 before coding. | pending | — |

### 6.2 D2 — W4 layout / Marlin-K vec8 port (the only retained parent W4 win)

`PLAN-PAROQUANT2.md` §11 documents that **most** Marlin-K work (FMA-only, Q8-FMA staging, sudot4,
all inner-loop ISA experiments) regressed or no-opped on the parent. The only retained win is the
**vec8 FMA inner loop + qweight-neutral replacement** (parent commits `7718fff` + `1522293`).
Everything else in §12.2-§12.6 (B1-B7, C1-C5, D1-D6, E1-E6, F1-F4) is `rejected` or `parked` upstream
— **do not redo them blind.** They are listed below as informational so future agents see the
guardrails.

| ID | Candidate | Source / lineage | Expected prefill Δ | Expected decode Δ | Memory | Risk / prereqs | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D2.1 | Port parent Marlin-K vec8 FMA kernel + qweight-neutral repack-on-load to hipENGINE non-expert `ParoQuantLinear` modules. | `docs/MARLIN.md` (full port plan); parent `nano-vllm-amd@7718fff`, `1522293`; `tools/paro_marlin_k_repack_reference.py`. | -0.7-0.3% (decode-pack8 view; prefill marginally slower on parent rerun) | **+1.1-2.6% on 35B 4K/128**, +1.5% on 35B 512/128 (parent §11.11 evidence) | +0.023 GiB peak metadata residue | Must preserve fast pack8 view path for prefill. Parent acceptance gate already passed correctness + 24 GiB memory. JIT extension name + cache path must be added to `docs/KERNELS.md` and `AGENTS.md` per `PLAN-PAROQUANT2.md` §0.6. | pending | — |
| D2.2 | Polish residual Marlin-K metadata residency (`qzeros_mk`, `scales_mk`) to 0 GiB. | `PLAN-PAROQUANT2.md` §12.1 A4 — `parked` upstream because every removal regressed decode. | n/a | neutral | -0.02 GiB | Parent already evaluated; reopen only if 24 GiB gate becomes tight. | parked, policy: parent A4 evaluated and parked; reopen only with new memory pressure. | — |
| D2.3 | Activation pre-quantize to Q8 once per residual block; downstream W4 GEMV reads Q8 directly. | `PLAN-PAROQUANT2.md` §11.5.3 + §12.4 D1 (parked: needs ABI). Needed prerequisite for any future `sudot4` lane. | neutral | unknown; parent unable to land due to ABI gap | -0.05 GiB | Needs a torch-free per-residual Q8 ABI (act + per-chunk fp16 scale tensor). Not a quick win. | parked, blocked-by: torch-free Q8 activation ABI | — |
| D2.4 (informational) | Marlin-K FMA inner-loop bandwidth tweaks (`int4` qweight load, `XVec8` activation vec-load, scale vec-load, `v_perm_b32` nibble unpack, `__half2` FMA, accumulator/register reshape, weight-hoist). | `PLAN-PAROQUANT2.md` §12.2 B1-B7 — **all rejected upstream**. | — | — | — | Do not redo without new evidence; parent has artifacts and ISA proofs. | parked (parent rejected B1-B7) | — |
| D2.5 (informational) | Marlin-K shape work (multi-row WG, templated WGSIZE, dual-projection fused, selected-MoE fused, strided fallback). | `PLAN-PAROQUANT2.md` §12.3 C1-C5 — C1 rejected (multi-row); C2 rejected (WGSIZE template); C3 parked (superseded by qweight-neutral view); C4 parked (selected-MoE inactive in 35B serving); C5 parked. | — | — | — | Same. | parked (parent rejected C1-C5) | — |
| D2.6 (informational) | Marlin-K codegen sweeps: unroll-600 (E1), launch_bounds (E2), `__builtin_assume` (E3), waves_per_eu (E4), nontemporal qweight load (E5), LDS prefetch (E6). | `PLAN-PAROQUANT2.md` §12.5 — all rejected/parked. | — | — | — | Same. | parked (parent rejected E1-E6) | — |
| D2.7 (informational) | Marlin-K frontier: Triton GEMV (F1), WMMA c=1 INT4 (F2), megakernel attention (F3), Vulkan-on-this-machine (F4). | `PLAN-PAROQUANT2.md` §12.6 — F1/F2 rejected, F3/F4 parked. | — | — | — | F4 (Vulkan calibration) is the same as M.5 above. | parked (parent rejected F1/F2; F4 = M.5) | — |
| D2.8 (informational) | Naive `sudot4` over current PARO/AWQ layout. | Parent `PLAN-PAROQUANT.md` two scratch trials + §11.3.3: 3.92-9.72× slower than tuned FMA. | — | — | — | Per `docs/OPTIMIZE.md` Do-Not-Chase list. | parked (parent rejected) | — |

### 6.3 D3 — Long-context attention (32K, 128K)

| ID | Candidate | Source / lineage | Expected prefill Δ | Expected decode Δ | Memory | Risk / prereqs | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D3.1 | Port parent `nano-vllm-amd@52ebcd9` grouped-GQA paged split-K producer to hipENGINE long-context decode. | `PLAN-LONGCONTEXT.md`; `LESSONS-LEARNED.md` "Grouped-GQA paged split-K producer"; current hipENGINE wrapper exists (`qwen35_paged_full_attn_decode_split_k_gqa_*`). | neutral | **+5-12% at 32K/128**, +5-10% at 128K/128 (parent measured `+11.2%` 32K, `+11.0%` 128K) | neutral | Defaults on only for `num_splits >= 64`; opt-out env. Validate per-shape correctness. Stack with D3.2 retune. | pending | — |
| D3.2 | Re-tune `NANOVLLM_AMD_PAGED_ATTN_MAX_SPLITS=512` cap on the grouped-GQA producer (parent D3.1 follow-up). | Parent `LESSONS-LEARNED.md` "Post-GQA split-cap retune": 128K/128 decode +12.4% after grouped-GQA producer landed. | neutral | **+5-12% at 128K/128** on top of D3.1; smaller at 32K | neutral | Must be evaluated *after* D3.1 — same knob was rejected pre-GQA. Re-test cheap sweeps after every structural change. | parked, blocked-by: D3.1 | — |
| D3.3 | Lower `NANOVLLM_PARO_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT` from `4096` to `1024` (parent retained at `78482b6`). | `PLAN-LONGCONTEXT.md` short/mid threshold sweep. | neutral | **+18% at 2K/128 and 3K/128** (parent measured); 4K/128 +0.7% | neutral | New default validation runs the graph/eager fixture at `1024/128`. | pending | — |
| D3.4 | Online-softmax/FlashAttention-style grouped-GQA producer rewrite. | `LESSONS-LEARNED.md` "One-pass streaming needs a correctness fixture before E2E promotion" — parent attempt rejected (32K validation mismatch). | neutral | uncertain (+5-20%) but currently blocked on correctness | neutral | Needs an attention-output/logit fixture comparing producer outputs, split partials, top logits, greedy tokens against the retained exact path **before** E2E. | parked, blocked-by: correctness fixture per `LESSONS-LEARNED.md`. | — |
| D3.5 | INT8 paged-KV decode path. | `PLAN-LONGCONTEXT.md` INT8 KV status; parent device-context INT8 only useful at very long context, neutral/negative at 32K. | neutral | -3-10% at 32K, +0-3% at 128K (parent measured) | -50% KV bytes | Needs a fused gate-reduce + end-to-end quality check; LESSONS-LEARNED W8A8 NaN history is the cautionary tale. | deferred (post batch-1) | — |

### 6.4 D4 — Decode launch floor and replay graph hygiene

| ID | Candidate | Source / lineage | Expected prefill Δ | Expected decode Δ | Memory | Risk / prereqs | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D4.1 | Replay-only profile harness: emit dispatches/token, per-family dispatch count, kernel time/token, inter-kernel gaps for 512/128 and 4K/128. | `docs/ROOFLINE.md`; parent steering signal ~894 replay-path dispatches/token after projection fusions. | n/a | n/a (enables D1.x, D4.2) | n/a | Must subtract prefill/warmup/capture from trace. | pending | — |
| D4.2 | Cap dispatches/token below 700 via batched D1.1-D1.6 fusions (Vulkan-style graph-level fusion). | `LLAMACPP-VULKAN.md`; parent `PLAN-PAROQUANT2.md` §11.5.2 (~660/tok current parent floor); Vulkan ~fewer than 200/tok at the same shape. | neutral | **+4-8%** when stacked | neutral | Per `LESSONS-LEARNED.md` "Output buffers alone are rarely enough", launch-count-only fusion under graph replay is sub-1%; must change data flow / reuse to count. | pending | — |
| D4.3 | Keep one-step decode graph replay as the retained graph shape; do **not** revisit multi-step capture. | `LESSONS-LEARNED.md` "Multi-step graph replay" — parent tested 1/2/4/8/16 step capture, no reliable gain; 4K/4K diverged at token 581. | n/a | -3-7% if reverted accidentally | n/a | This is a guardrail, not a candidate. | accepted (guardrail) | `WORKLOG.md` decode-graph fixture gate. |
| D4.4 | Per-kernel `__launch_bounds__` retune after rotation/RMSNorm/W4 fusion changes (LESSONS-LEARNED "Runtime thread-count knobs must honor kernel launch bounds"). | `LESSONS-LEARNED.md` Task 23 audit. | neutral | +0-2% | neutral | Must cross-check against statically allocated shared memory + reduction scratch; never accept a knob value that bypasses `__launch_bounds__`. | pending | — |

### 6.5 D5 — Decode glue / small wins ledger

These are individually sub-1% but compound. Parent §11.10 found that **GDN recurrent decode**, **PARO
single rotation**, **router select/logits**, and **W8A16 lm-head** combined still dominate ~25% of
decode wall after fusion polish.

| ID | Candidate | Source / lineage | Expected prefill Δ | Expected decode Δ | Memory | Risk / prereqs | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D5.1 | Audit `qwen35_gdn_recurrent_rmsnorm_gate_lowp_kernel` for vec8 / occupancy headroom on c=1 decode (currently `VGPR=56`, `LDS=1616`). | `docs/KERNELS.md` rocprof note; ROOFLINE §9 RDNA3 occupancy rules. | neutral | +0.5-2% | neutral | Per LESSONS-LEARNED, deep-loop unroll is the largest RDNA3 GEMV lever for `N/blockDim.x < 64`. | pending | — |
| D5.2 | W8A16 lm-head decode tile or fused argmax (parent decode profile shows lm-head 8.4%). | LESSONS-LEARNED W8A16 lm-head retained win + parent decode rocprof; current hipENGINE wrapper is `lm_head_fp16_argmax_bf16`. | neutral | +0-1% | neutral | Per parent decode notes "Fused `lm_head + argmax` is **not** a current lever" — argmax was only 0.2% of selected-region kernel time. Stop at audit unless profile says otherwise. | parked (parent already audited) | — |
| D5.3 | Router top-k cooperative producer (one workgroup all experts) — avoid the naive logits+top-k fusion that collapses occupancy. | `LESSONS-LEARNED.md` "Router fusion is the opposite case"; current router uses one block per expert. | neutral | +0.5-1% | neutral | Without inter-block sync, naive same-kernel fused top-k is racy or collapses occupancy. Use cooperative producer pattern. | pending | — |
| D5.4 | Linear-attention A/B decode same-input fusion (already done in parent / hipENGINE: `linear_attn_AB_fused`). | LESSONS-LEARNED retained win precedent. | neutral | already +0.6-1% | neutral | Confirm hipENGINE has this. If not, port. | pending (status audit) | — |

### 6.6 D6 — DFlash / MTP / multi-token speculative decode

| ID | Candidate | Source / lineage | Expected prefill Δ | Expected decode Δ | Memory | Risk / prereqs | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D6.1 | Native compact/c-aware target verifier + GPU accept summaries (`DFLASH.md` Task #15). | `docs/DFLASH.md`; parent Python harness ~0.96× AR with 1.20 verify rows/output. | neutral | -3% to +50% (depends on verify cost; parent Python `~0.963×` AR; native target is ≥ 1.1×) | +1-2 GiB for verify scratch | Heavy infrastructure work; speculative path is `blocked` until verifier lands. | deferred (Phase 4) | `benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json` |
| D6.2 | MTP draft plugin on the shared DFlash verifier (`MTP.md`). | `docs/MTP.md`; parent B=5 native row `83.88 tok/s` vs AR `120.04 tok/s` = `0.699×`. | neutral | -30% to +20% (depends on acceptance + verify cost) | +0.5 GiB MTP head | DFlash verifier must land first; otherwise this reproduces the parent's ~0.7× AR result. | deferred (post-DFlash) | — |

---

## 7. Lane A — Memory (a feature, not a casualty)

hipENGINE currently beats the parent peak-memory row on every retained comparison context. Any
candidate above must preserve this. The rows below are **guardrails**, not candidates that move.

| ID | Guardrail | Current value | Risk | Status |
| --- | --- | --- | --- | --- |
| A.1 | Default 512/128 peak | 18.58 GiB (parent 18.86 GiB) | New BLAS/WMMA bulk paths must report extra scratch and not duplicate W4 layouts. | accepted |
| A.2 | Default 4K/128 peak | 19.88 GiB (parent 21.64 GiB) | Same. | accepted |
| A.3 | Default 32K/128 peak | 20.69 GiB (parent 21.37 GiB) | Long-context chunked policy is the default; do not silently revert. | accepted |
| A.4 | Default 128K/128 peak | 23.66 GiB (parent 27.42 GiB) | Stay below 24 GiB; the differentiator vs parent. | accepted |
| A.5 | AOTriton is vendored through Git LFS and remains the baseline runtime dependency. | n/a | `attn_aotriton_min_tokens=512` is the default; `0` is diagnostic only and must not be used for current-fastest rows. | accepted |
| A.6 | Alias ownership for qweight views (Marlin-K-style zero-copy) | n/a | Aliases must be non-owning tensors; never create two owning `DeviceTensorAllocation` records for the same pointer. | accepted |

---

## 8. Lane W — Compiler / build profile sweeps

Cheap to run, sometimes large. Per LESSONS-LEARNED, every sweep must be paired with `Scratch_Size=0`
and VGPR audits, and treated as **per-kernel build flag**, not a global default.

| ID | Candidate | Source / lineage | Expected prefill Δ | Expected decode Δ | Memory | Risk / prereqs | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W.1 | `-mllvm -amdgpu-unroll-threshold-local=600` on hot prefill kernels (linear-attn GDN, MoE compact WMMA, full-attn prelude, shared expert). | `PR_COMMENT-llamacpp-hip-unroll600.md` (llama.cpp +166% prefill, near-neutral decode); parent E1 (`PLAN-PAROQUANT2.md` §12.5): neutral on Marlin-K FMA. | +10-50% on a hot prefill kernel **if** it triggers; could be neutral | -0-2% | neutral | Verify `Scratch_Size=0` and VGPR not increased. Decode 0.6% regression observed in some PR_COMMENT models; cross-check Qwen3.5 retained sample. | pending | — |
| W.2 | `-mcumode` build profile on hot decode kernels. | `PR_COMMENT-llamacpp-hip-unroll600.md` table; default hipENGINE already uses `-mcumode` per ROOFLINE §1.1. | n/a (already default) | n/a | n/a | Confirm wrappers actually compile with `-mcumode`; some build paths may have dropped it. | pending (status audit) | — |
| W.3 | Per-kernel `__attribute__((amdgpu_waves_per_eu(...)))` retune after rotation/RMSNorm fusion lands. | Parent E4 rejected for Marlin-K but landed kernels in hipENGINE have different VGPR profile. | neutral | +0-2% | neutral | Re-evaluate per kernel; do not blanket-apply. | pending | — |

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

Parent-rejected / hipENGINE-out-of-scope. Do not open a multiloop on these without **new** profile
evidence (e.g. structural changes that invalidate the earlier rejection).

| Avoid | Why | Source |
| --- | --- | --- |
| Naive `sudot4`/dp4a over current PARO/AWQ layout | 3.92-9.72× slower than tuned FMA; layout + activation staging dominate. | `PLAN-PAROQUANT.md`, `PLAN-PAROQUANT2.md` §11.3.3, §12.6 D2.8 |
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

Order is chosen to maximize signal-per-iteration. **M.x is gating; do it first.**

1. **M.1 / M.2** — Promote first accepted `LLM.generate()` rows + auto-refresh the comparison table.
   Without this, every other "+X%" below is unverifiable.
2. **M.3 / M.4** — Matched rocprof profiles + per-bucket Amdahl table for 512/128 decode and prefill.
   Confirms or falsifies P1 / D1 bucket hypotheses before code.
3. **M.5** — Local llama.cpp Vulkan Q4_K_M at 4K/4K. Calibrates headline decode targets (the +14.4%
   gap may already be smaller).
4. **P1.1 + P1.2 + P1.4** — Bulk dense rocBLAS for linear-attn A/B and shared-expert gate/up SiLU,
   with empirical token-count threshold. This is the largest single prefill lever per `PREFILL.md`.
5. **W.1** — `-mllvm -amdgpu-unroll-threshold-local=600` per-kernel sweep on the four hot prefill
   kernels above. Cheap one-iteration probe; could compound with P1.
6. **D1.1 + D1.4 + D2.1** — Rotation-into-projection fusion at the right granularity, selected-MoE
   post-op fold, and Marlin-K vec8 + qweight-neutral port. Stack these toward the +6% / +17%
   decode lift.
7. **D3.1 + D3.2 + D3.3** — Grouped-GQA producer port + split-cap retune + paged-decode min-context
   threshold. Closes 32K/128 and 128K/128 decode against parent.

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
| hipENGINE prefill architecture + Amdahl audit | `docs/PREFILL.md` |
| hipENGINE kernel catalog and port gates | `docs/KERNELS.md` |
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
