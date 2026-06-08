# hipEngine Dense Prefill Optimization Plan — Qwen3.6-27B-PARO

Status: 2026-05-18 (initial draft, all candidates `pending`).

Scope: `z-lab/Qwen3.6-27B-PARO` `w4_paro` on W7900/gfx1100, batch-1 (`c=1`)
prefill and short/mid-context decode rows. Sister doc to
[`docs/OPTIMIZE.md`](OPTIMIZE.md), which covers the Qwen3.5-35B-A3B-PARO MoE
workload. Both documents share the same lane format and the same promotion
gates.

Goal: close the dense prefill gap to llama.cpp HIP at 4K/128, then 512/128,
without regressing decode (where we already lead) or peak memory. Phase ordering
follows `AGENTS.md` "audit first, optimize second": **no kernel work begins
before Lane M lands a per-kernel rocprof Amdahl table for this model.**

This document is the **live punchlist** for the dense 27B workload. Each
candidate is a row in a per-lane table with the same columns as
`docs/OPTIMIZE.md`:

| Column | Meaning |
| --- | --- |
| **ID** | Stable label (e.g. `Q36D-P1.1`). Use this in commits / `WORKLOG.md` / multiloop tags. |
| **Candidate** | Short description of the change. |
| **Source / lineage** | Where the evidence/precedent lives (parent file, kernel, llama.cpp impl, reference repo, etc.). |
| **Expected prefill Δ** | Best-guess uplift on `prefill_tok_s` from upstream evidence and ROOFLINE/Amdahl. |
| **Expected decode Δ** | Best-guess uplift on `decode_tok_s`. |
| **Memory** | Expected peak-allocated delta (must respect §3 guardrails). |
| **Risk / prereqs** | Audit/profile/blocker prereqs, correctness hazards, parent negative results to avoid. |
| **Status** | `pending`, `in-progress`, `accepted`, `rejected`, `parked`, or `deferred`. |
| **Result / evidence** | Filled in when the lane is run: measured Δ, artifact path, fixture KL/top-1, rocprof note. |

Cross-links:

- [`docs/OPTIMIZE.md`](OPTIMIZE.md) — Qwen3.5-35B-A3B-PARO MoE punchlist; many
  Lane M / Lane D rows there apply directly to the dense path.
- [`docs/PREFILL.md`](PREFILL.md) — native prefill architecture, AOTriton
  evidence, profile/Amdahl analysis.
- [`docs/KERNELS.md`](KERNELS.md) — kernel catalog, port playbook,
  source-lineage drift workflow.
- [`docs/ROOFLINE.md`](ROOFLINE.md) — RDNA3/W7900 perf model and
  anti-rabbit-hole rules.
- [`docs/BENCHMARK.md`](BENCHMARK.md) and
  [`benchmarks/README.md`](../benchmarks/README.md) — promotion contract and
  rollup.
- Reference repos under `~/amd-gpu-tuning/reference/`:
  `atlas/kernels/gb10/qwen3.6-27b/nvfp4/gated_delta_rule.cu`,
  `atlas/kernels/gb10/common/gated_delta_rule_wy.cu`,
  `vllm/vllm/model_executor/layers/mamba/gdn_linear_attn.py`,
  `FlashQLA/flash_qla/ops/gated_delta_rule/chunk.py`,
  `exllamav3/exllamav3/architecture/qwen3_next.py`.

---

## 1. Workload and current scoreboard

### 1.1 Architecture (from local config snapshot)

| Field | Value |
| --- | ---: |
| `hidden_size` | 5120 |
| `intermediate_size` (dense MLP) | 17408 |
| `num_attention_heads` | 24 |
| `num_key_value_heads` | 4 |
| `head_dim` | 256 |
| `num_experts` | 0 (dense MLP) |
| Layer count | 64 (48 `linear_attention` + 16 `full_attention`, `full_attention_interval=4`) |
| `linear_num_value_heads` × `linear_value_head_dim` | 48 × 128 |
| `linear_num_key_heads` × `linear_key_head_dim` | 16 × 128 |
| `linear_conv_kernel_dim` | 4 |

Implication: every prefill token traverses **48 GDN linear-attention layers** and
**16 full-attention layers**, then a **dense W4 PARO MLP** (`5120 → 17408 ×
2 → 5120`) per layer. There is **no top-k expert sparsity** to exploit on the
MLP side.

### 1.2 Current diagnostic measurements (2026-05-18, W7900/gfx1100)

Source artifact:
[`benchmarks/results/2026-05-18-hipengine-gfx1100-qwen36-27b-paro-diagnostic.json`](../benchmarks/results/2026-05-18-hipengine-gfx1100-qwen36-27b-paro-diagnostic.json).

| Workload | Decode mode | Prefill tok/s | Decode tok/s | Tracked peak GiB |
| --- | --- | ---: | ---: | ---: |
| 512/128 | eager | 632.640 | 32.013 | 24.233 |
| 512/128 | graph_replay | 630.739 | 32.955 | 24.233 |
| 4K/128 | eager | 636.451 | 28.777 | 26.839 |
| 4K/128 | graph_replay | 631.673 | 29.567 | 26.839 |

**Status:** diagnostic only. There is no committed Qwen3.6-27B KL / top-1 oracle
fixture yet; landing one is `Q36D-K.1` below and is a hard prereq for promoting
any Lane P row from `in-progress` to `accepted`.

### 1.3 vs llama.cpp HIP (reference run, same machine)

llama.cpp HIP `Qwen3.6-27B-Q4_K_M.gguf`, `-fa 1 -ngl 99`:

```
| qwen35 27B Q4_K - Medium  | 15.92 GiB | 27.32 B | ROCm | 99 |  1 |  pp4096 | 818.58 ± 5.61 |
| qwen35 27B Q4_K - Medium  | 15.92 GiB | 27.32 B | ROCm | 99 |  1 |   tg128 |  25.42 ± 0.03 |
build: 232f46658 (9214)
```

Comparison vs hipEngine 4K/128 graph row:

| Workload | hipEngine prefill | llama.cpp prefill | Δ vs llama.cpp | hipEngine decode | llama.cpp decode | Δ vs llama.cpp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4K / 128 | 631.67 | 818.58 | **-22.8% prefill** | 29.57 | 25.42 | **+16.4% decode ✅** |

Lift to win 4K prefill: roughly **+29.6%** (or `+186.9 tok/s`).

### 1.4 The compact goal

To beat llama.cpp HIP on the dense path while preserving the decode lead and
peak-memory advantage we need approximately:

- **Prefill:** +30% at 4K/128. 512/128 prefill is a secondary target; the
  short-context gap is harder to close without a structural change.
- **Decode:** preserve existing margin (`+16.4%` at 4K/128, `~+30%` at 512/128
  vs llama.cpp tg128). Acceptable to give up `~1-2%` decode for a `>10%` prefill
  win, but no row goes accepted that breaks decode generated-sample equality.
- **Memory:** keep 4K/128 well below the W7900 24 GiB practical gate is no
  longer possible at this model scale (current peak 26.84 GiB at 4K/128 is over
  the gate); the gate becomes "fits in 44 GiB device memory headroom and does
  not regress vs the diagnostic baseline." Document the over-24-GiB status on
  every retained row.

### 1.5 Why MoE wins and dense lags (problem statement)

Two structural deltas, in order of effect:

1. **No top-k MLP sparsity.** Qwen3.5-35B-A3B-PARO MoE only touches `~2/8`
   experts per token; the *grouped-compact WMMA prefill* path
   (`gemm_awq_selected_dual_pack8_wmma_compact_*` in
   `hipengine/kernels/hip_gfx1100/wmma/paro_awq_wmma.hip`) exploits that
   sparsity. The dense MLP touches 100% of `intermediate=17408` weights for
   every token via `awq_fusedw4_prefill_dual_fp16_kernel` and
   `awq_fusedw4_prefill_fp16_kernel` in
   `hipengine/kernels/hip_gfx1100/quant/paro_awq_gemv.hip`.
2. **More linear-attention serial state.** The 27B dense model has 48
   `linear_attention` layers (vs the 35B-A3B MoE which has fewer), and each
   prefill chunk runs `qwen35_gdn_prefill_recurrent_k2_kernel`
   (`hipengine/kernels/hip_gfx1100/linear_attn/gdn.hip`) with a strictly serial
   `for (token = 0; token < tokens; ++token)` outer loop. At 4K context that is
   `48 × 4096 ≈ 196k` strictly sequential token-steps with `__syncthreads`
   between each, with no parallelism across the time axis. This is the single
   largest structural prefill miss.

---

## 2. Strategy in one paragraph

Do not start with another blind kernel multiloop. **Capture matched
`rocprofv3 --kernel-trace --selected-regions` profiles** (Lane M) for the dense
27B 4K/128 and 512/128 workloads first, using the same selected-region
infrastructure that landed
`benchmarks/results/2026-05-17-hipengine-qwen35-rocprof-amdahl-diagnostic.json`
for the 35B-A3B MoE path. Expectation, to be confirmed by data: the top three
prefill kernel-time buckets are `qwen35_gdn_prefill_recurrent_k2_kernel`
(serial state), `awq_fusedw4_prefill_*_kernel` (W4 dense MLP WMMA), and
`paro_rotate*_kernel` (PARO activation rotations). Once that ranking is signed
evidence, the lane order is:

1. **Lane K — Correctness oracle.** Land a Qwen3.6-27B KL / top-1 fixture at
   the same context lengths as the bench rows. Without it no prefill change
   can graduate from diagnostic to accepted (per `docs/BENCHMARK.md` evidence
   policy and `AGENTS.md` post-run quality gates).
2. **Lane P1 — Chunkwise / WY-chunkwise GDN prefill.** Replace the serial-time
   recurrent kernel with a chunkwise port modeled on the atlas / FLA / vllm
   references. This is the only structural lever; expected to be the dominant
   prefill win.
3. **Lane P2 — PARO rotate fusion into W4 prefill.** Eliminate dedicated
   `paro_rotate{1,2}_fp16` activation passes by folding the rotation into the
   head of `awq_fusedw4_prefill_*_kernel`'s K-loop.
4. **Lane P3 — Dense MLP launch coalescing / fusion.** `silu_mul +
   paro_rotate1` fusion is the low-risk first step; full
   gate+up+silu+rotate+down fusion is the larger lever once Lane P2 lands.
5. **Lane P4 — Tuning / config.** Larger linear/MLP chunk sizes (only after P1
   removes the serial bottleneck), AOTriton min-tokens confirmation, W4 prefill
   tile/launch-bounds sweep guarded by `LESSONS-LEARNED.md` rules.

---

## 3. Non-negotiable promotion gates

Same gates as `docs/OPTIMIZE.md` §3, repeated here for emphasis. Every row
below clears them all before moving from `in-progress` to `accepted`.

1. **Correctness first.** Before retaining any number, the relevant fixture
   gates pass:
   - Qwen3.6-27B KL / top-1 oracle (Q36D-K.1) at the same context length as the
     measured row.
   - Native prefill correctness equivalent to
     `scripts/qwen35_native_prefill_fixture_gate.py` for the dense 27B model.
   - Decode generated-sample equality with the eager baseline at the same
     prompt and seed.
2. **No hidden torch in the hot path.** `import torch` is never in any module
   reached by `hipengine.LLM.generate()`. Profiler-only Python wrappers are
   allowed. Per `hipengine/AGENTS.md` "Architectural Invariants".
3. **Registry, not backend branches.** New paths register under
   `(backend, layer, quant, variant)`. No `if backend == "..."` or
   `if quant == "..."` in engine / model / dispatch.
4. **Memory budget.** 4K/128 already exceeds the W7900 24 GiB gate; do not
   regress vs the diagnostic baseline (`26.839 GiB` at 4K/128). 512/128 must
   stay below its diagnostic baseline (`24.233 GiB`).
5. **Retained perf rows update the rollup.** `benchmarks/README.md`,
   `benchmarks/CHANGELOG.md`, and the compact JSON artifact in
   `benchmarks/results/` all move with each accepted row.
6. **Generated-sample equality.** A retained row matches the oracle token
   stream. Speed without sample equality is a correctness bug per the
   `LESSONS-LEARNED.md` RoPE / NaN history.

---

## 4. Lane M — Measurement (profile our own kernels first)

Audit-first is mandatory. The `qwen35_paro_bench.py` benchmark already supports
the dense 27B model, and the selected-region rocprof harness from
[`docs/OPTIMIZE.md`](OPTIMIZE.md) Lane M is reusable. Capture and commit a
compact summary before opening any Lane P row.

| ID | Candidate | Source / lineage | Expected prefill Δ | Expected decode Δ | Memory | Risk / prereqs | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q36D-M.1 | Capture matched `rocprofv3 --kernel-trace --selected-regions` profiles for Qwen3.6-27B-PARO at 512/128 and 4K/128 with deployment defaults (`--attn-aotriton-min-tokens 512`, `--graph-replay-decode`). Retain only compact summary under `benchmarks/results/`. | `docs/OPTIMIZE.md` Lane M.3 precedent; `AGENTS.md` "Pre-optimization grid/occupancy audit (MANDATORY)". | n/a | n/a | n/a | Use `--compiler-version-file` + `--require-cached-build` + `roctxProfilerResume/Pause` to avoid profiled `hipcc` and rocprofv3 marker-trace graph-replay crashes. Prebuild via `scripts/prebuild_native.py` first. | pending | — |
| Q36D-M.2 | Per-bucket Amdahl table for hipEngine dense 27B prefill at 512/128 and 4K/128 (kernel time only). Emit one CSV/JSON with `(KernelName, Calls, TotalDurationNs, ShareOfPrefill)` per workload. | Parent rocprof tail audit precedent; `docs/ROOFLINE.md` §5. | n/a | n/a | n/a | Sums `DurationNs` per `KernelName` from M.1 raw traces; reuses `~/amd-gpu-tuning/scripts/summarize_rocprof_kernels.py` style bucketing. | pending, blocked-by: Q36D-M.1 | — |
| Q36D-M.3 | Per-bucket Amdahl table for hipEngine dense 27B decode replay-only window at 512/128 and 4K/128. | Parent decode rocprof audit `artifacts/paroquant2_rocprof_audit_20260515_iter30/`. | n/a | n/a | n/a | rocprofv3 1.2.x asserts at 64/128 traced graph replays on this host; profile 16 one-step graph replays and scale per token, matching the OPTIMIZE.md M.4 protocol. | pending, blocked-by: Q36D-M.1 | — |
| Q36D-M.4 | Confirm whether 4K/128 prefill actually uses the AOTriton compact-varlen kernel or falls back to the GQA-spans path. | `aotriton_attention` flag in M.1 trace must list the AOTriton kernel name; `docs/PREFILL.md` AOTriton evidence. | n/a | n/a | n/a | Diagnostic JSON's `prefill_execution_detail.aotriton_attention` is `false` at smoke `tokens=8`; confirm at 4K. If AOTriton is not in the 4K rocprof, that is itself a Lane P bug, not a kernel optimization. | pending, blocked-by: Q36D-M.1 | — |

The Lane M evidence resolves the ranking of Lane P1 vs P2 vs P3. Until then,
the priorities below are **inferred** from source structure and the upstream
references, not from rocprof for this specific model.

---

## 5. Lane K — Correctness oracle (hard prereq for any retained Lane P row)

| ID | Candidate | Source / lineage | Expected prefill Δ | Expected decode Δ | Memory | Risk / prereqs | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q36D-K.1 | Land a Qwen3.6-27B-PARO KL / top-1 fixture at 512 and 4K context. CPU-reference oracle with KL ≤ 0.05 and top-1 ≥ 90% on the bench prompt. | `docs/TESTING.md` correctness gate; `docs/KERNELS.md` "Validation matrix"; existing `scripts/qwen35_native_prefill_fixture_gate.py` template. | n/a | n/a | n/a | Re-use the same prompt/seed pair the bench uses (`token_id=9707`, repeated to `prompt-length`) so prefill correctness, decode generated-sample equality, and the rocprof Amdahl rows all share inputs. | pending | — |
| Q36D-K.2 | Add `finite_prefill_logits` and `decode_step_graph_validation` booleans to the bench JSON for this model, matching the OPTIMIZE.md retained-row schema. | `AGENTS.md` "Post-Run Quality Gates"; current diagnostic artifact lists those fields as `null`. | n/a | n/a | n/a | Pure harness change; no kernel risk. Prevents future ambiguity about whether retained rows are NaN-clean. | pending | — |

---

## 6. Lane P — Prefill candidates

### 6.1 P1 — Chunkwise GDN prefill (highest expected leverage)

The `qwen35_gdn_prefill_recurrent_k2_kernel`
(`hipengine/kernels/hip_gfx1100/linear_attn/gdn.hip`) iterates tokens serially.
At 4K with 48 linear-attention layers this is the dominant prefill cost in
the source-structure analysis (Lane M.1 to confirm). All major upstream
LLM-inference stacks that run Qwen3.5/3.6/qwen3-next-style hybrids use a
chunkwise / WY-chunkwise reformulation that parallelizes the recurrence within
a chunk; FLA, vllm `gdn_linear_attn`, FlashQLA, exllamav3, and the atlas
reference kernels all do this.

| ID | Candidate | Source / lineage | Expected prefill Δ | Expected decode Δ | Memory | Risk / prereqs | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q36D-P1.1 | Port WY-chunkwise GDN prefill (2-pass algorithm) to a new `qwen35_gdn_prefill_chunkwise_*` kernel family. Preserve the existing `(query, key, value, beta, decay, recurrent_state, out)` ABI exactly; add only a chunk size parameter. Provide a CPU reference oracle bit-equal to the existing serial recurrent kernel at chunk_size=1. | `~/amd-gpu-tuning/reference/atlas/kernels/gb10/qwen3.6-27b/nvfp4/gated_delta_rule.cu` (`gated_delta_rule_chunk2/chunk3`); `~/amd-gpu-tuning/reference/atlas/kernels/gb10/common/gated_delta_rule_wy.cu` (WY-chunkwise spec); FLA `chunk_gated_delta_rule`; vllm `gdn_linear_attn.py`; exllamav3 `qwen3_next.py`. | **+15% to +25% at 4K/128** (largest single lever); +5-10% at 512/128 | neutral (decode still uses recurrent state-update kernels, not the prefill kernel) | neutral or slightly higher (chunk staging buffers); must keep tracked peak ≤ diagnostic baseline | Highest-priority structural change. Correctness contract must include: (a) bit-equal output vs serial recurrent at chunk_size=1, (b) KL ≤ 0.05 / top-1 ≥ 90% vs Q36D-K.1 oracle at chunk_size in {64, 128, 256}, (c) numerically stable for the gate decay range observed in this checkpoint. The 27B PARO uses `linear_value_head_dim=128`, which matches the typical FLA chunk sizes. | pending, blocked-by: Q36D-M.1 (signed evidence GDN is the top bucket), Q36D-K.1 (oracle) | — |
| Q36D-P1.2 | After P1.1 lands: increase `PrefillConfig.linear_chunk_size` autotune ceiling so 4K context runs 1-2 layer chunks instead of 4. Re-tune `_PREFILL_LINEAR_CHUNK` defaults for the new chunkwise GDN throughput shape. | `hipengine/runtime/prefill.py` `_resolve_prefill_config_for_length`; `docs/OPTIMIZE.md` P5.2 chunk-autotuner precedent. | +1-3% on top of P1.1 | neutral | `+0.1 to +0.5 GiB` at 4K (larger chunk staging) | Only meaningful after P1.1; with the serial kernel, chunk size only changes launch count, not parallelism. Re-confirm 4K ≤ baseline peak. | pending, blocked-by: Q36D-P1.1 | — |
| Q36D-P1.3 | Chunkwise GDN segments kernel: extend P1.1 to the `_segments_kernel` variant used by packed `c>N` prefill. | Existing `qwen35_gdn_prefill_recurrent_k2_segments_kernel`; same upstream references as P1.1. | not measurable on `c=1`; sets up future `c>N` dense prefill | neutral | neutral | Defer until `c=1` Lane P1.1 is `accepted`. Same correctness contract (bit-equal at chunk_size=1, KL/top-1 vs oracle). | deferred (Lane S, c>N) | — |

### 6.2 P2 — PARO rotate fusion into W4 prefill

The dense MLP currently runs `paro_rotate2_fp16(hidden) → awq_fusedw4_prefill_dual_fp16(gate+up) → silu_mul_separate_out_fp16 → paro_rotate1_fp16(intermediate) → awq_fusedw4_prefill_fp16(down)` per layer per chunk. The two `paro_rotate*` launches are **separate full-activation read+write passes** before each W4 prefill. The rotation is row-local (no cross-token dependency), so it can be fused into the kernel that already loads `x_row` for the WMMA inner loop.

| ID | Candidate | Source / lineage | Expected prefill Δ | Expected decode Δ | Memory | Risk / prereqs | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q36D-P2.1 | Fuse `paro_rotate2_fp16` into the head of `awq_fusedw4_prefill_dual_fp16_kernel`. Apply scale + krot rotation pairs to `b_reg[tn]` while it is being loaded; eliminate the dedicated rotate launch and the rotated-input scratch (`shared_gate_input`, `shared_up_input`). Templated on `qweight_transposed`. | Parent precedent `nano-vllm-amd@gemv_awq_dual_pack8_transposed_rotate_staged_kernel` (rotate-fused GEMV) extended to the prefill WMMA kernel; existing `paro_rotate2_kernel` for the per-pair scale + sincosf math. | **+5% to +15% at 4K/128** (eliminates 1 full read+write of hidden activations per dense MLP layer per chunk; 64 layers × N chunks at 4K) | neutral | `-0.05 to -0.10 GiB` (drop `shared_gate_input` / `shared_up_input` scratch tensors) | Mechanical fusion. Correctness contract: bit-equal vs the unfused chain at FP32 accumulation; KL ≤ 0.05 vs Q36D-K.1. Risk: register pressure on the WMMA kernel may spill VGPRs and erase the win — `__launch_bounds__(32, 8)` already targets full occupancy. Profile with `rocprofv3 -i pmc.txt` for `VGPR_Count` and `Scratch_Size` after the change. | pending, blocked-by: Q36D-K.1, Q36D-M.1 (confirm rotate is meaningful share) | — |
| Q36D-P2.2 | Fuse `paro_rotate1_fp16` into the head of `awq_fusedw4_prefill_fp16_kernel` (down projection). Same fusion pattern as P2.1, single rotation per call. | Same as P2.1. | **+3% to +10% at 4K/128** (eliminates 1 full read+write of `intermediate=17408`-wide activations per dense MLP layer per chunk) | neutral | `-0.05 to -0.20 GiB` (drop `shared_down_input` scratch where safe; intermediate buffer remains for SiLU) | Same correctness/VGPR risk as P2.1. Down projection's intermediate is wider (17408) than gate/up's hidden (5120), so the launch-count win and the HBM-traffic win are larger here than in P2.1. Land P2.1 first to validate the fusion pattern. | pending, blocked-by: Q36D-P2.1 | — |
| Q36D-P2.3 | Fuse `paro_rotate2_fp16` into the linear-attention QKV/Z and full-attention QKV pack8 prefill paths where applicable, mirroring the dense MLP fusion. | Existing rotate-fused selected-MoE precedents (`gemv_awq_selected_dual_pack8_strided_rotate_out_kernel`); per-call rotate launch counts in M.1. | +1-3% additional | neutral | small additional savings | Only meaningful if M.1 shows attention-side rotate is non-trivial share. Lower priority than dense-MLP fusion because it touches the smaller projection paths. | pending, blocked-by: Q36D-P2.1, Q36D-M.1 | — |

### 6.3 P3 — Dense MLP launch coalescing / fusion

After Lane P2 the dense MLP shrinks to `awq_fusedw4_prefill_dual_fp16 (gate+up,
rotate-fused) → silu_mul_separate_out_fp16 → awq_fusedw4_prefill_fp16 (down,
rotate-fused) → combine_residual`. The 17408-wide intermediate is still
materialized in HBM between SiLU and down. Two reductions remain.

| ID | Candidate | Source / lineage | Expected prefill Δ | Expected decode Δ | Memory | Risk / prereqs | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q36D-P3.1 | Fuse `silu_mul_separate_out_fp16 + paro_rotate1_fp16` into one kernel that reads gate/up outputs and writes the rotated down-input directly. After P2.2 lands, this collapses to fusing SiLU·multiply into the head of the rotate-fused down `awq_fusedw4_prefill_fp16_kernel`. | Existing `silu_mul_dual_rotate_out_fp16` precedent for the `tokens==1` decode path; same source idea applied at `tokens>1`. | +1% to +3% at 4K/128 | neutral (decode already uses `silu_mul_dual_rotate_out`) | `-0.05 to -0.10 GiB` (collapse `shared_intermediate` ↔ `shared_down_input` redundancy) | Lower risk after P2.2 is validated. Correctness contract: bit-equal to the unfused chain at FP16 with FP32 SiLU accumulation; KL ≤ 0.05 vs Q36D-K.1. | pending, blocked-by: Q36D-P2.2 | — |
| Q36D-P3.2 | Producer-consumer fused dense MLP: `awq_fusedw4_prefill_dual (gate+up) → register-resident SiLU·multiply → register-resident rotate1 → awq_fusedw4_prefill (down)` in one tile-resident kernel. Avoids HBM round-trip for the 17408-wide intermediate. | No direct upstream precedent; closest is GGML's fused `gate*up*silu*down` micro-kernels in some quants. Big change, not a port. | **+3% to +8% at 4K/128**; possibly the only path to close the remaining gap if Lane P1+P2 do not land enough alone | neutral | unclear; depends on whether the 17408-wide tile fits register/LDS budget. Likely net-zero or small negative tracked-peak. | High implementation risk; large kernel surface; needs sizable LDS scratch (17408 FP16 = 34KiB intermediate per row, so per-tile staging only). Defer until Lane P1+P2 results show whether it is needed to close the gap. **Do not start without M.1 evidence and Lane P1+P2 measurements.** | deferred (gated by P1+P2 results) | — |

### 6.4 P4 — Tuning, config, and W4 kernel hygiene

Smaller, lower-risk levers. Audit-driven; only lands if M.1 / M.2 / M.3 show a
non-trivial share.

| ID | Candidate | Source / lineage | Expected prefill Δ | Expected decode Δ | Memory | Risk / prereqs | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q36D-P4.1 | Confirm `attn_aotriton_min_tokens=512` actually routes 4K/128 prefill through the AOTriton compact-varlen path; if the diagnostic harness defaulted to the GQA-spans path, that alone is a meaningful prefill regression. | `docs/OPTIMIZE.md` P2.3; `docs/PREFILL.md` AOTriton evidence; current diagnostic JSON shows `aotriton_attention=false` at smoke-only `tokens=8`. | TBD; up to **+3% at 4K/128** if the bench is currently using the wrong attention kernel | neutral | neutral | Pure config / harness check. Land Q36D-M.4 first; the answer determines whether this is a real lever or a no-op. | pending, blocked-by: Q36D-M.4 | — |
| Q36D-P4.2 | W4 prefill kernel tile / launch-bounds sweep for the dense 27B GEMM shapes (`5120 → 17408 × 2`, `17408 → 5120`). Confirm the autoselected `tile_m=32, tile_n=32` is actually optimal at these sizes. | `LESSONS-LEARNED.md` "Audit first, optimize second" + "Deep loop unrolling is a major RDNA3 GEMV lever"; existing `_launch_fusedw4_prefill_*` autoselect logic in `paro_awq_gemv.py`. | TBD; per `LESSONS-LEARNED.md` likely small (1-3%) and noisy under graph replay | neutral | neutral | Only after Lane P1+P2 land. **Do not** chase WMMA tile micro-tuning until Lane P1 has fixed the structural prefill bottleneck — per `AGENTS.md` and `LESSONS-LEARNED.md`, the kernel time-share ranking will shift after P1, and this row should be re-evaluated then. Skip if M.1 shows W4 prefill is < 10% of dense 27B kernel time. | deferred (gated by P1+P2 results) | — |
| Q36D-P4.3 | `-mllvm -amdgpu-unroll-threshold-local=600` ablation on the dense W4 prefill kernel build profile. | `~/amd-gpu-tuning/PR_COMMENT-llamacpp-hip-unroll600.md`: llama.cpp HIP pp512 +166% with this flag; hipEngine `docs/OPTIMIZE.md` P1.5 retained as **neutral default**. | likely neutral (already in default profile per OPTIMIZE.md P1.5) | neutral | neutral | Confirm via `HIPENGINE_DISABLE_UNROLL600=1` ablation that the dense path is already getting the flag. If yes, document and close. If no, enable and re-bench. | pending | — |
| Q36D-P4.4 | Check whether `dense_mlp_paro_w4_fp16` `tokens==1` and `tokens>1` branches diverge in any way that matters for prefill chunking (e.g. whether the prefill chunk size of 1024 picks `tokens>1` consistently, and whether the very first prefill row at `tokens==1` smoke ever happens in the bench). | `hipengine/runtime/qwen35_paro.py:4735+` `dense_mlp_paro_w4_fp16` (the `tokens==1` decode path uses `gemv_awq_dual_pack8_transposed_fp16` + `silu_mul_dual_rotate_out_fp16`; the `tokens>1` prefill path uses `awq_fusedw4_prefill_dual_fp16` + `silu_mul_separate_out_fp16`). | n/a | n/a | n/a | Pure code-reading audit. Document any stale `tokens==1` shape that should never fire during prefill. | pending, blocked-by: Q36D-M.1 | — |

---

## 7. What not to chase (anti-rabbit-hole list)

`docs/ROOFLINE.md` §11 and `LESSONS-LEARNED.md` document the upstream
parent-rejected directions. The following dense-specific items are explicitly
**not** candidates until the listed precondition is met. Do not silently open a
multiloop on these.

- **Wave64 / `-mwavefrontsize64`** for any kernel except an isolated probe.
  Per `AGENTS.md` "Wavefront size policy", wave64 is non-default on gfx1100,
  smoke probes show cross-64-lane shuffle reductions are not actually
  cross-half-wave even with the flag, and the WMMA `_w32` intrinsics required
  by the W4 prefill kernels conflict with `-mwavefrontsize64`.
- **rocBLAS / hipBLASLt for the dense GEMM shapes.** `docs/OPTIMIZE.md` P1.1
  measured rocBLAS slower than the existing custom row-GEMV / fused-W4-WMMA
  path on the linear-attn A/B shapes; the dense MLP W4 prefill shape is also
  custom (W4 → FP16 dequant + WMMA). Do not introduce rocBLAS as a baseline
  unless a non-W4 BF16/FP16 path actually appears in the dense 27B prefill
  bucket and outranks the structural Lane P1 work.
- **dp4a / sudot4 W4 micro-optimizations.** Parent `PLAN-PAROQUANT2.md` §11
  lists these as parked / regressed across many lanes. Any retry requires a
  torch-free Q8-once-per-residual ABI (parent D1, parked) plus signed
  evidence that the W4 prefill bucket is the dominant remaining cost after
  Lane P1+P2. Currently neither precondition is met.
- **`MoE selected compact WMMA` thresholding for the dense path.** Dense
  MLP has no top-k expert structure to exploit; the grouped-compact WMMA
  kernel is irrelevant here. Do not import that code path into the dense
  MLP.
- **Building a new "fast" GDN kernel from scratch instead of porting WY-chunkwise.**
  The chunkwise / WY-chunkwise reformulation is the well-understood
  parallelization of this recurrence; multiple correct CUDA / Triton / HIP
  implementations exist as references. Reinventing the math without a CPU
  reference oracle violates `docs/TESTING.md` correctness gate and the
  `LESSONS-LEARNED.md` "audit first" rule.
- **Grouped-GQA producer attention rewrites at short context.** `docs/OPTIMIZE.md`
  D3 evidence shows long-context attention rewrites have small impact at 4K
  and noise-level impact at 512. The dense 27B 4K prefill bucket for
  AOTriton attention is small (Lane M.4 to confirm); do not chase it
  before Lane P1+P2.

---

## 8. Out of scope for this document

- **Multi-request `c>N` packed prefill** for the dense 27B path. Lane S in
  `docs/OPTIMIZE.md`. Q36D-P1.3 will track the `c>N` chunkwise GDN segments
  port when the `c=1` board closes.
- **Long-context (32K, 128K) dense prefill.** Out of immediate scope; the
  current diagnostic 4K/128 already exceeds the 24 GiB W7900 gate, and
  longer contexts are a memory problem before they are a throughput problem
  on this 27B dense model. Revisit after Lane P1.1 lands.
- **Speculative decode / DFlash for the dense path.** `docs/DFLASH.md` and
  `docs/SPECULATIVE-DECODE.md` cover this; orthogonal to single-request
  prefill optimization here.
- **gfx1151 (Strix Halo).** A separate workload with different occupancy
  math; tracked under `docs/ROOFLINE-gfx1151.md` if and when it matters for
  this model.

---

## 9. Reproduction

The diagnostic baseline is reproducible with the same harness committed in
`scripts/qwen35_paro_bench.py`. Recipe (run from
`/home/lhl/amd-gpu-tuning/hipEngine`):

```bash
# 1. Prebuild kernels outside any profiler (avoids hipcc inside the profiled process).
mamba run -n therock python3 scripts/prebuild_native.py \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt

# 2. Reproduce the 4K/128 graph row.
mamba run -n therock python3 scripts/qwen35_paro_bench.py \
    --model /models/huggingface/hub/models--z-lab--Qwen3.6-27B-PARO/snapshots/f0797088d8e0312aac0b5969bec1e6e5c6fb3ff3 \
    --backend hip_gfx1100 \
    --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 4 \
    --attn-aotriton-min-tokens 512 \
    --graph-replay-decode \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build \
    --json /tmp/qwen36-27b-paro-gfx1100-4k-128-graph.json

# 3. Lane M.1: rocprofv3 selected-region kernel trace, same workload.
PYTHONPATH=/home/lhl/amd-gpu-tuning/tools/rocprof_torch_site${PYTHONPATH:+:$PYTHONPATH} \
NANOVLLM_REQUIRE_CACHED_BUILD=1 \
NANOVLLM_HIPCC_VERSION_FILE=/tmp/hipengine-hipcc-version.txt \
mamba run -n therock rocprofv3 --kernel-trace -f csv \
    -o /tmp/hipengine-rocprof-qwen36-27b-4k/trace \
    -- python3 scripts/qwen35_paro_bench.py \
       --model /models/huggingface/hub/models--z-lab--Qwen3.6-27B-PARO/snapshots/f0797088d8e0312aac0b5969bec1e6e5c6fb3ff3 \
       --backend hip_gfx1100 \
       --prompt-length 4096 --decode-tokens 1 --warmup-decode-tokens 0 \
       --attn-aotriton-min-tokens 512 \
       --no-graph-replay-decode \
       --compiler-version-file /tmp/hipengine-hipcc-version.txt \
       --json /tmp/qwen36-27b-paro-gfx1100-4k-trace-bench.json
```

llama.cpp HIP comparison row reproduction (from `~/llama.cpp/llama.cpp-hip`):

```bash
build/bin/llama-bench -fa 1 -m /models/gguf/Qwen3.6-27B-Q4_K_M.gguf -p 4096
# Expect: pp4096 ≈ 818 ± 6 tok/s, tg128 ≈ 25.4 tok/s on W7900/gfx1100.
```

Both models are the same parameter scale (27B) and same group-size W4 quant
class. The architectural delta between hipEngine PARO and llama.cpp Q4_K_M is
real (PARO has activation rotations; Q4_K_M does not), but is out of scope for
the optimization plan above — closing the prefill gap is about restructuring
the linear-attention recurrence (Lane P1) and removing redundant activation
passes (Lane P2/P3), not about changing the quantization scheme.

---

## 10. Update protocol

- Update this file when a Lane P row moves from `pending` → `in-progress` →
  `accepted` / `rejected` / `parked` / `deferred`. Same protocol as
  [`docs/OPTIMIZE.md`](OPTIMIZE.md).
- Every accepted row also moves
  [`benchmarks/README.md`](../benchmarks/README.md),
  [`benchmarks/CHANGELOG.md`](../benchmarks/CHANGELOG.md), and writes a
  compact JSON artifact under `benchmarks/results/`.
- `WORKLOG.md` records the per-iteration evidence; this file records the
  *plan* and the *retained outcome*.
- If the architectural assumptions change (different model, different
  attention pattern, new quant), open a new sister doc instead of warping
  this one.
