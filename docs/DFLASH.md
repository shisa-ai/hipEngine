# hipEngine DFlash / DDTree Native Implementation Plan

> Status: implementation plan. This document converts the DFlash lessons from
> `~/amd-gpu-tuning` into a hipEngine port plan. Kernel R&D and benchmark
> exploration stay in `~/amd-gpu-tuning`; the production path belongs here as a
> torch-free, native C++/HIP hot loop.

## Thesis

The Python/PyTorch DFlash harness in `~/amd-gpu-tuning` has reached diminishing
returns. It proved correctness, acceptance accounting, the parent-indexed tree
kernel shape, and memory-safe state rings, but it still verifies rows too slowly
relative to autoregressive decode.

hipEngine is the right destination for the real implementation because the speed
problem is no longer a draft-policy problem. It is a native runtime problem:

- one target forward over `[root, draft/tree nodes...]` per cycle;
- no PyTorch tensors in the hot loop;
- no per-depth Python loops;
- no per-cycle `torch.empty`/clone churn;
- stable device buffers and scratch addresses;
- device-side argmax/accept/commit summaries;
- graph-capturable C++/HIP execution once fixed shapes are stable.

The immediate 2026-05 `dflash` branch target is
**z-lab/Qwen3.6-35B-A3B-DFlash** as the drafter against the
**shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5-packed** target model on native
Strix Halo `gfx1151` (`--offload-arch=gfx1151`). The W7900/gfx1100 and Quark
rows below remain the measured parent evidence, but they are not a prediction
for this packed/gfx1151 lane. On gfx1151 we have roughly 48% of W7900 compute
but only ~30% of W7900 memory bandwidth (optimistic read ceiling ~221 GB/s), so
bytes are more expensive and compute-per-byte is higher; a raw C++/HIP verifier
that increases row reuse and avoids PyTorch/host overhead may shift DFlash from
near-break-even to worthwhile. This is a hypothesis until same-session AR rows
on the packed target prove it.

The same infrastructure now also has an MTP-facing scaffold: provider-neutral
chain `DraftBatch` compilation, target-attached `mtp.*` metadata/loading, and a
local PARO+MTP-BF16 artifact assembled from the packed PARO trunk plus Qwen's
MTP sidecar. DFlash remains the first native block-verifier target; see
[`MTP.md`](MTP.md) for the target-attached multi-token predictor plan that
reuses this verifier/commit path rather than forking it.

## Current hipEngine status (2026-05-18)

The API scaffolding exists (`DraftBatch`, `TargetVerifyBatch`,
`TargetVerifyBuffers`, `TargetStateCommitBuffers`, `AcceptResult`,
`TargetAcceptSummary`, `TargetCommitPlan`, `DraftModel`, `Verifier`,
`KVTransaction`, and verify-shaped graph keys), and the first full-model native
B+1 chain verifier now runs in `scripts/dflash_chain_e2e_bench.py`.  DFlash/DDTree
is still **not** an accepted throughput path because the native verifier is
correct but slower than same-session AR and slower than the previous serial
fallback diagnostic.  Older blocker context is retained in
[`2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json`](../benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json).
Current status:

- the latest c=8 resident batch artifact still reports
  `scheduler_serial_slot_bridge`, `serial_c1_layer_path`, and
  `throughput_claim_eligible=false`;
- `Qwen35ParoResidentSession` exposes `step_batch_serial()`, batch metadata,
  `speculative_execution_metadata()`, metadata-only `target_verify_batch()` /
  `verify_speculative_batch()` layout helpers, and a `commit_verified_state()`
  copy/select path; target-verifier buffers are validated against the resident
  transaction id and device, and state/KV commit buffers are checked against
  commit-row/accepted-row coverage before `dflash_commit_chain_i32` copies
  selected linear state, accepted K/V path rows, hidden taps, output-ring tokens,
  and position/context metadata. The full-model chain E2E path now also has a
  resident `native_bulk_bplus1` verifier that materializes fixed-budget
  `TargetVerifyBatch` rows, captures target hidden taps, samples row-wise top-1,
  validates `dflash_accept_chain_i32` against the CPU oracle, commits the
  selected linear state row, and keeps the serial in-place verifier as fallback;
- native prefill still stops at the three-layer linear prefix, with first
  unsupported layer 3 (`full_attention`);
- speculative metadata and KV transactions reject duplicate request ids,
  invalid transaction roles, accept-summary/commit-plan transaction mismatches,
  ambiguous accept-result selected rows, accept-result next-token metadata,
  target-verifier next-token output buffers, CPU target-top1 accept-summary
  oracles, accept-summary/transaction candidate-budget/topology mismatches, and inconsistent transaction terminal
  states, and the batch scheduler can validate active-request readiness, emit
  scheduler-owned speculative `TargetVerifyBatch`/`WorkItem` metadata, derive
  verify graph shape keys, cache graph/replay objects under those keys, begin
  speculative KV transactions, bundle scheduler-owned verify plans, bind those
  plans to same-transaction, same-candidate-budget/topology resident target-verifier device buffers, derive scheduler-owned
  commit plans from verifier accept summaries or target-top1 oracle outputs, bind those commit plans to
  same-transaction, same-device, row-covering state/KV commit device buffers, commit or roll back
  speculative KV transaction metadata, finalize accepted-token recording after KV commit, and record
  accepted speculative token summaries plus target next tokens against request budgets, while host KV
  transaction bookkeeping now accounts
  for `TargetVerifyBatch` candidate
  rows only (committed root rows are excluded from the speculative journal),
  tracks per-request candidate counts, rejects accepted counts larger
  than the verified candidate budget, validates accepted target paths, can
  select the per-request target row whose state would be committed, binds the
  summary to a transaction-scoped commit plan, validates target-verifier and
  state-commit device buffer shapes/dtypes, projects candidate rows into
  scheduler `WorkItem` metadata, and derives verify graph shape keys from the
  target row topology, the torch-free target-verify ladder comparator can
  compare serial c=1 vs bulk verify-chain row snapshots at each layer-family
  boundary with first-failing-stage diagnostics, the gfx1151 GPU top1 +
  `dflash_accept_chain_i32` smoke matches `TargetVerifyBatch.accept_from_top1`
  for reject/partial/full, multi-request real verifier rows, and budgeted
  no-bonus cases without full-logit host copies in the accept fast path, and the
  gfx1151 `dflash_commit_chain_i32` smoke proves reject/partial/full plus
  multi-request copy/select commits do not leak rejected suffix rows into
  canonical linear state, KV, hidden taps, output ids, or context metadata;
- `scripts/dflash_chain_e2e_bench.py` now runs a same-session full-model AR
  control and native DFlash chain smoke on the shisa packed target plus z-lab
  drafter.  After fixing the drafter rotary table from a hard-coded `10000` to
  the z-lab config `rope_theta=10000000`, three follow-up phases landed:
  - Phase A: `serial_in_place_single_slot` verifier (no per-candidate state
    copies because the verify loop never steps into a rejected candidate).
  - Phase B: drafter caches `projected_context_norm` across cycles and only
    re-projects newly committed rows on commit.
  - Phase C: drafter caches per-layer rotated K (FP32) and V (BF16) for context
    rows; per-cycle `propose()` only processes block-size query rows.
  The retained Phase A+B+C gfx1151 16-token smoke is exact/finite with `6/30`
  acceptance across 9 cycles and is still slower than AR (`0.289x`), but it used
  the serial fallback verifier
  ([artifact](../benchmarks/results/2026-05-18-hipengine-dflash-chain-full-model-e2e-phaseABC-diagnostic.json)).
  The retained native-B+1 smoke is also exact/finite, has GPU accept summary =
  CPU oracle, and performs one fixed B+1 verifier call per draft cycle
  (`target_bulk_forward_calls=10`, `target_forwards_per_draft_call=1.0`), but is
  slower (`0.124x` AR, `performance_claim=false`; artifact
  [`2026-05-18-hipengine-dflash-chain-full-model-e2e-nativebulk-diagnostic.json`](../benchmarks/results/2026-05-18-hipengine-dflash-chain-full-model-e2e-nativebulk-diagnostic.json)).
  A follow-up drafter HIP graph prototype captures and replays the fixed-shape
  `propose()` body exactly (`validation_passed=true`, 10/10 candidate paths), but
  exact `context_tokens` buckets have no reuse during decode, so it regresses to
  `133.8 ms/call` vs the no-graph `68.9 ms/call` baseline and remains diagnostic
  ([artifact](../benchmarks/results/2026-05-18-hipengine-dflash-drafter-graph-validate-diagnostic.json)).
  A first QKV projection fusion is bit-exact vs the unfused GPU path and
  rocprofv3 confirms `dflash_qkv_proj_bf16_mixed_kernel`, but the retained E2E
  row is neutral (`69.6 ms/call`, `0.122x` AR), so it stays opt-in via
  `--drafter-fusion qkv`
  ([artifact](../benchmarks/results/2026-05-18-hipengine-dflash-drafter-qkv-fusion-diagnostic.json)).
  The latest native-verifier warm-scratch attempt improves B={1,2,4,8} verify
  seconds by ~26-35% by avoiding verifier scratch churn and a pre-accept barrier,
  but remains `1.8x-4.9x` slower than serial c=1
  ([artifact](../benchmarks/results/2026-05-18-hipengine-dflash-verifier-warmscratch-speedgate-diagnostic.json)).
  A follow-on true-batched chain verifier (`--full-attn-chain-mode batched`)
  lands the proper bulk path: one batched RMSNorm + rotate + QKV projection +
  multi-token RoPE + prompt-style K/V append + gated GQA prefill attention +
  batched O proj + post-norm + forced c=1 MoE per full-attention layer.  It is
  exact vs c1_loop and same-session AR, 6-8% faster than the c1_loop bulk path
  at B=2/4, neutral or slightly slower at B=1/8, and still `2.0-5.0x` slower
  than serial c=1 across all B because each batched cycle still pays B+1 rows
  of multi-token MoE (non-coop router, multi-row pack8 GEMV) regardless of how
  early the chain is rejected
  ([artifact](../benchmarks/results/2026-05-19-hipengine-dflash-chain-batched-vs-c1-loop-speedgate-diagnostic.json)).
  This batched path is retained as the **infrastructure foundation for DDTree**
  (where tree branches make serial early-exit structurally impossible), not as
  a chain DFlash speed win.

  **DDTree foundation (2026-05-19)** lands on top of the chain batched path:
  a new tree-aware GQA prefill gate kernel (`qwen35_paged_full_attn_prefill_gqa_gate_tree_fp16_spans`)
  with a per-row `[rows, rows]` ancestor mask + `tree_committed_count` offset,
  plus host-side ancestor-mask + per-row cache-slot metadata, a
  `_run_full_attention_tree_batched` orchestrator, a `verify_tree_bulk_and_commit`
  session entry, and `_commit_tree_full_attention_kv` for post-accept K/V
  compaction (multi-cycle decode-safe).  The `dflash_accept_chain_i32` accept
  summary kernel already walks `parent_rows` and so handles tree topology with
  no kernel changes.  Three GPU correctness gates are retained:

  * `scripts/dflash_tree_attn_kernel_smoke.py` — chain-shaped ancestor mask
    reduces to the chain kernel byte-for-byte; branching mask filters
    siblings/cousins correctly.
  * `scripts/dflash_tree_e2e_smoke.py` — three canonical tree shapes (depth-2
    binary, depth-1 4-way branch, chain reduction) all pass finite_logits +
    gpu_accept_match_cpu + cpu_oracle_matches on PARO target weights; the
    root's `target_top1` is invariant across tree shapes (proves the mask
    correctly isolates root-level attention from verifier rows).
  * `dflash_chain_e2e_bench.py --tree-mode chain_as_tree` -- end-to-end
    decode loop wraps the chain drafter output as a degenerate (linear) tree
    and routes through `verify_tree_bulk_and_commit`.  Exact AR-match,
    GPU accept matches CPU, accept-count parity with chain at B={1,2,4,8}.
    Verify seconds are within 6% of chain batched (FASTER at B=1 / B=4),
    confirming the tree kernel's per-row ancestor-mask check adds NO
    meaningful overhead.

  DDTree is still `2.0-4.9x` slower than serial c=1 on this degenerate chain
  topology because B+1 per-cycle target compute remains the bottleneck.

  **Branching top-K DDTree MVP (2026-05-19)** adds the first real non-linear
  tree proposal path to the same benchmark: `--tree-mode branching_topk
  --tree-top-k 2`.  The drafter now asks `topk_f32_rows_i32` for row-wise top-K
  logits (K<=8) and the host compiles a balanced breadth-first flat tree from
  the per-depth top-K tables.  For B=4,K=2 the active candidate parents are
  `[-1, -1, 0, 1]`: two root siblings, then one continuation under each.  The
  verifier remains `verify_tree_bulk_and_commit`; accepted tokens come from the
  tree accept path, not chain prefix slicing.  Because a real branch can accept
  a non-contiguous verifier path (e.g. rows `[0, 2, 4]`),
  `verify_tree_bulk_and_commit` also compacts captured hidden taps into dense
  context rows before the drafter appends them, matching the existing K/V
  compaction semantics.

  Fresh gfx1151 speed gate, 8 decode tokens, B={1,2,4,8}, K=2:
  branching top-K passes exact same-session AR equality and GPU accept matches
  CPU for every B.  It improves over chain_batched / chain_as_tree at B=2/4/8
  (`14.65/12.73/9.29 tok/s` vs chain_batched `14.19/10.70/8.42` and
  chain_as_tree `13.93/12.07/8.47`) by accepting `5` draft tokens in `3`
  cycles at B=4/8.  It still loses to serial c=1 (`19.61/19.70/17.98/17.22
  tok/s`), so the retained row is diagnostic, not a throughput claim.  The next
  blocker is target verifier row cost (multi-row MoE/router/projection) rather
  than tree proposal correctness.
- no speculative throughput claim is allowed until the native compact/c-aware
  target verifier plus drafter path produces a retained chain win over
  same-session AR.

## Prior W7900/gfx1100 evidence from `~/amd-gpu-tuning`

The numbers in this section are retained as design evidence from the parent
workspace: W7900/gfx1100, Qwen3.5/Qwen3.6 PARO/Quark-family artifacts, and a
Python/PyTorch-assisted DFlash harness. They prove acceptance accounting,
correctness, and the verifier cost wall, but they are **not** the baseline for
the new `gfx1151` + shisa packed target. Every promoted hipEngine row for the
current branch must re-measure same-session AR and DFlash on the packed model.

### Best current Python-harness row

Latest retained HumanEval-class chain/bulk row after R1/R2/R3 and the pack8
row-threshold fixes:

| Metric | Current value |
| --- | ---: |
| AR decode | ~29.75 tok/s |
| DFlash decode | ~28.65 tok/s |
| vs AR | ~0.963x |
| target verify | ~1.754 s / 64 output tokens |
| DFlash draft | ~0.380 s / 64 output tokens |
| target verify rows/output | 1.203 |
| verify eta per row | ~0.678 AR-token |
| peak allocated | ~21.77 GiB |
| correctness | exact greedy AR match, finite logits |

Cost model in AR-token units per emitted output:

```text
verify_cost ~= rows/output * eta = 1.203 * 0.678 = 0.815
 draft_cost ~= 0.380s / 2.15s AR = 0.176
 overhead   ~= remaining              = 0.046
 total      ~= 1.037 AR-token/output  = 0.963x AR
```

So acceptance is good enough to be near break-even, but target verification is
not cheap enough. To reach meaningful speedups with the same acceptance:

| Goal | Max total AR-token/output | Required verify eta if draft+overhead unchanged |
| --- | ---: | ---: |
| 1.1x | 0.909 | <= ~0.57 |
| 1.38x (DDTree-MLX chain class) | 0.725 | <= ~0.42 |
| 1.5x | 0.667 | <= ~0.37 |

The native implementation must therefore reduce **per-row target verify cost**
and **draft/host overhead**. Policy tuning alone cannot produce a large win.

### What already worked

These are worth porting or preserving:

1. **R1: parent-indexed tree Conv1D/GDN t-loop kernels.**
   The corrected HIP kernels put tree nodes on an in-kernel `t=0..N` loop and
   keep head/channel slices on the grid. A parent read at node `t` reads a slot
   written earlier by the same thread because `parent_idx < t`. This removes
   the old depth-batched host loop.

2. **R2: one launch per recurrent layer.**
   The bulk verifier can drive the corrected kernels with `parent_ids[N]`.
   Kernel-level wins were large for Conv and modest for GDN, but E2E stayed
   near flat because W4 projections and host/runtime overhead dominated next.

3. **Pack8 row threshold fix.**
   The generic PARO pack8 GEMV path must stay active for small multi-row verify
   batches. Falling back to dequant + rocBLAS/Tensile at `rows > 8` caused a
   major bs>=12 cliff. The project default is now `NANOVLLM_PARO_GEMV_V8_MAX_ROWS=16`.

4. **Dual-pack8 multi-row gates.**
   `gemv_awq_dual_pack8` is row-agnostic and safe for small `N`; the Python
   gate was the limitation. E2E impact was neutral in the Python harness, but
   it removes real dispatches and should be part of the native path.

5. **R3: persistent per-layer node-state rings.**
   Reusing per-layer scratch for `tree_conv_state`, `tree_recurrent_state`, and
   row intermediates cut peak memory by ~0.94 GiB at bs=8 and ~2.10 GiB at
   bs=16. It was performance-neutral in Python because commit/allocations were
   not the main bottleneck, but the memory discipline is required for graph
   capture and fixed-address native execution.

### What did not move the wall-clock enough

- More adaptive path/hybrid policy before chain DFlash wins.
- Budget 16/22 as a default. Larger budgets increase verify work faster than
  they increase useful accepted output on this hybrid target.
- Allocation-only cleanup after R3. Memory improved; speed did not.
- Per-dispatch micro-fusions inside the Python harness once the verify window
  remained ~22% host-idle and W4 projection dominated GPU time.

## Reference implementations and what to copy

All references below should be treated as design inputs. Do not edit them from
hipEngine work; port ideas and, where license-compatible and approved, code.

| Reference | Local path | Useful files / concepts | Key lesson |
| --- | --- | --- | --- |
| `amd-gpu-tuning` DFlash plan | `~/amd-gpu-tuning/PLAN-DFLASH.md` | R1/R2/R3/R6/R7 entries, WORKLOG 2026-05-15 | Our measured failures and corrected kernel lineage. |
| Spec decode analysis | `~/amd-gpu-tuning/docs/SPECULATIVE-DECODE.md` | speed model, reference audit, break-even math | Verification efficiency is the metric, not raw acceptance. |
| Fresh-eyes audit | `~/amd-gpu-tuning/docs/DFLASH-FRESH-EYES.md` | side-by-side reference patterns | Every winning impl uses one native batched forward plus persistent state commit. |
| DDTree-MLX | `~/amd-gpu-tuning/reference/ddtree-mlx` | `ddtree_mlx/verify.py::tree_verify_forward`, `cache.py::tree_aware_path_commit`, `kernels.py`, `BENCHMARKS.md` | Budget=4 default; tree-aware GDN/Conv; commit as slot copy; chain DFlash wins first. |
| hipfire | `~/amd-gpu-tuning/reference/hipfire` | `crates/hipfire-arch-qwen35/src/speculative.rs`, `qwen35.rs::TreeVerifyCtx`, `forward_prefill_batch*`, `rdna-compute/src/dispatch.rs::gated_delta_net_q8_tree_batch_seq` | Closest C++/HIP/gfx1100 shape: persistent scratch, batched verify, tree parent indices, native hot loop. |
| Lucebox DFlash | `~/amd-gpu-tuning/reference/lucebox-hub/dflash` | `test_dflash` flow, ggml CUDA tree Conv/GDN variants | Single graph/ggml forward; `_persist` GDN writes state directly into persistent cache. |
| vLLM / SGLang DFlash | source refs listed in `PLAN-DFLASH.md` | DFlash proposer, target-verify mode, draft KV materialization | Separate draft context KV materialization from query-token draft forward. |

Reference headline numbers on Qwen3.5/3.6 27B-class DFlash targets:

| Impl | Hardware | Shape | vs AR |
| --- | --- | --- | ---: |
| Current Python harness | W7900 | HumanEval bs=8 chain/bulk | ~0.96x |
| DDTree-MLX | M3 Ultra | chain / chain+DDTree | 1.38x / 1.52x |
| Lucebox | RTX 3090 | HumanEval DDTree | 3.43x |
| hipfire | 7900 XTX/gfx1100 | HumanEval DDTree | 4.45x |

The gap is not explained by W7900 memory bandwidth. It is runtime shape,
quantized small-batch linears, persistent cache discipline, and graph/native
host overhead.

## gfx1151 / packed-target deltas

The current branch starts from the gfx1151 roofline in
`../amd-gpu-tuning/docs/ROOFLINE-gfx1151.md`:

- `gfx1151` has about **48%** of W7900's FP16/BF16/INT8/INT4 matrix compute but
  only about **30%** of W7900's theoretical external memory bandwidth, with a
  local measured read ceiling around **221 GB/s**.
- Weight/KV bytes are therefore more expensive than on W7900. Speculative
  verification only helps if the native path amortizes target weights across
  root+candidate rows and removes PyTorch/host overhead; copying extra rows or
  rebuilding draft context can erase the win quickly.
- Native kernels must be compiled for `--offload-arch=gfx1151`; retained rows
  should not rely on `HSA_OVERRIDE_GFX_VERSION=11.0.0` or `gfx1100` code objects.
- The target artifact is the shisa packed PARO model (packed shared expert and
  pack8 decode sidecars), not the older Quark W8A8 + BF16 MTP bring-up layout.
- The first benchmark question is whether chain DFlash on the packed target can
  beat same-session AR. DDTree and MTP remain follow-ons on the same verifier.

## Artifact metadata gate

Before materializing tensors or launching a DFlash benchmark, run the torch-free
metadata validator:

```bash
python3 scripts/dflash_validate_artifacts.py \
  --target-model /models/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-full4096-e5-packed/snapshots/501ef8635e5cfb5a7497d232358ca8d1afc0c66e \
  --drafter-model /models/huggingface/hub/models--z-lab--Qwen3.6-35B-A3B-DFlash/snapshots/42d3b34d588423cdae7ba8f53a8cf7789346a719 \
  --json /tmp/hipengine-dflash-artifact-validation.json
```

The validator reads only `config.json` plus safetensors headers. It checks the
packed target's PARO shared-expert sidecars and the DFlash drafter's `fc`,
`hidden_norm`, draft-layer attention/MLP tensors, block size, mask token, target
hidden tap ids, hidden/head dimensions, KV heads, and vocab size. It must pass
before benchmark rows are considered comparable.

## Non-negotiable design rules

1. **Native hot loop.**
   DFlash generation in hipEngine must not call PyTorch or HF Transformers in
   the measured loop. Python may load configs, build the engine, and launch a
   benchmark; the repeated decode cycle is C++/HIP/raw-pointer execution.

2. **Chain DFlash must beat AR before DDTree is promoted.**
   DDTree is a +10-15% topping in the most conservative reference. If
   topk=1/chain cannot beat AR, topk>1 policy work is premature.

3. **Verify is one target forward over `N` rows.**
   hipEngine's speculative plugin boundary stays `DraftBatch`: it carries
   candidate rows only, not the already-committed root. The verifier internally
   materializes a `TargetVerifyBatch` with root at slot 0 plus candidate rows:

   ```text
   target_verify(tokens[N], positions[N], parents[N], tree_mask[N,N], start_pos)
   ```

   For topk=1 chain, `parents = [-1, 0, 1, ...]` and the mask is causal over
   the block. For DDTree, `parents` and `tree_mask` come from the compiled flat
   tree.

4. **Tree nodes live inside kernels, not on the host grid.**
   Conv and GDN kernels loop over nodes internally. No host depth loop; no
   per-depth launches.

5. **Persistent scratch and cache rings are mandatory.**
   Every per-cycle tensor has a fixed owner and address: input ids, positions,
   parent ids, masks/bias, logits/top1, accept summary, hidden taps, per-layer
   conv/GDN node states, full-attention temporary K/V rows, and draft KV.

6. **Commit is a copy/select, not a re-forward.**
   The verifier writes per-node states. Commit selects the accepted path and
   copies the accepted final node's linear-attention state plus accepted full-
   attention K/V rows into the live cache. Re-forward is a debug fallback only.

7. **Small budgets first.**
   Default to chain budgets `{1,2,4,8}` and DDTree budget `4`. Do not promote
   `budget >= 16` until small budgets are saturated and memory gates pass.

8. **Device-side accept summary.**
   The hot loop may copy a compact summary to host, but it must not materialize
   full vocab logits or do token-by-token acceptance in Python. Target top1,
   accept length, bonus/correction token, and committed ids are device outputs.

9. **Measured quality gates stay attached to every row.**
   Every retained benchmark reports exact greedy equality, finite logits, AR
   tok/s, DFlash tok/s, `verify_eta`, rows/output, draft time, verify time,
   overhead time, peak memory, and generated sample equality.

## Runtime architecture target

### Core objects

Suggested C++/HIP-owned runtime objects, exposed through hipEngine's Python API
only at setup/benchmark boundaries:

```text
DFlashSession
  TargetModelRuntime target
  DraftModelRuntime draft
  DFlashBuffers buffers
  TargetVerifyScratch verify_scratch[max_N]
  DraftKVCache draft_kv
  DdTreeCompiler tree_compiler
  HipGraphCache graphs_by_shape
```

`DFlashBuffers` owns fixed-size device buffers:

```text
input_ids[N]
position_ids[N]
parent_ids[N]
depths[N]
ancestor_mask_or_bias[N, N]
draft_topk_ids[(B-1), K]
draft_logits_or_scores[(B-1), K]
target_top1_ids[N]
accept_flags[N]
accept_summary[small]
committed_ids[max_commit]
bonus_id[1]
```

`TargetVerifyScratch` owns per-layer scratch:

```text
linear layer l:
  conv_state_nodes[max_N, conv_state_shape]
  recurrent_state_nodes[max_N, recurrent_state_shape]
  qkv/z/AB/intermediate rows[max_N, ...]

full-attention layer l:
  tree_k_rows[max_N, kv_heads, head_dim]
  tree_v_rows[max_N, kv_heads, head_dim]
  attention workspace[split_k, ...]
```

### Per-cycle flow

```text
1. target has already produced the current root token and target hidden taps.
2. draft context KV is already materialized through committed target hidden rows.
3. draft_query_forward(root + mask/query rows) produces B-1 candidate distributions.
4. chain or DDTree compiler writes tokens/positions/parents/mask into DFlashBuffers.
5. target_verify_batch(...) runs one native target forward over all N rows.
6. device_accept_kernel compares target top1 to draft tree edges and writes summary.
7. commit_kernel/copy path installs accepted recurrent state and K/V rows.
8. output ring receives root + accepted draft tokens + target correction/bonus.
9. append newly committed target hidden rows into draft context KV.
10. repeat.
```

There should be one synchronization boundary per cycle at most: copy the compact
accept summary or output count if the host scheduler needs it. A graph-captured
fixed-shape path should eventually replay steps 4-8 with fresh buffer contents.

## Kernel and port plan

### Phase D0 — Documented source-lineage refresh

- Update `docs/KERNELS.md` / `docs/source_lineage.json` to include the corrected
  DFlash kernel source files and parent commits from `nano-vllm-amd`:
  - R1 tree Conv/GDN t-loop kernels (`b95eaa5` lineage).
  - R2 Python/wrapper integration (`69eb9d8` lineage, but port as C++/HIP API).
  - PARO pack8 small-row threshold default 16 (`6f0e468` lineage).
  - dual-pack8 multi-row gate proof (`5d8f496` lineage).
- Add fixture descriptions for chain `N={1,2,4,8}` and DDTree budget=4.

### Phase D1 — Native chain verifier API, no drafter yet

Goal: prove hipEngine can verify a fixed `[root, draft...]` chain through the
native target runtime with selectable state commit.

- Add `TargetVerifyBatch` C++/Python boundary object with device buffers for
  ids, positions, parents, and mask.
- Implement topk=1 chain compiler: `parents=[-1,0,1,...]`, causal block mask.
- Port/wire corrected tree Conv/GDN t-loop kernels into hipEngine's raw-pointer
  wrapper style.
- Wire full-attention verify to write K/V rows into tree K/V scratch, not live
  cache first.
- **Landed 2026-05-18:** implement commit for chain: install final accepted
  row's Conv/GDN-style linear state, compact/copy accepted full-attention K/V
  path rows, copy hidden taps/output ids, and update position/context metadata
  with `dflash_commit_chain_i32`.
- Correctness gate: same-session exact greedy equality on synthetic candidates
  where accepted length is forced to 0, partial, and full.

### Phase D2 — Device-side target top1 and accept summary

Goal: remove host acceptance work from the measured loop.

- Reuse/extend hipEngine GPU lm-head + argmax primitives for `N` rows.
- Add `dflash_accept_chain_kernel`:
  - inputs: draft ids, target top1 ids, `N`, remaining decode budget;
  - outputs: accepted draft count, commit row count, correction/bonus id,
    committed ids, full-accept flag.
- Keep an optional debug path that copies all target top1 ids for trace rows.
- Correctness gate: device accept summary equals CPU reference for crafted
  accept patterns and real DFlash outputs.

Status 2026-05-18: row-wise `argmax_f32_rows_i32`, row `lm_head_fp16_argmax_bf16_rows_i32`,
and `dflash_accept_chain_i32` are landed for gfx1100/gfx1151 registration. The gfx1151 smoke
`HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/dflash_accept_chain_smoke.py --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --debug-top1-readback`
passes CPU-oracle parity for crafted reject/partial/full chains, multi-request `TargetVerifyBatch`
rows from `compile_dflash_chain`, and remaining-budget no-bonus outputs. The follow-on
`dflash_commit_chain_i32` smoke commits those summaries into canonical state/KV/output buffers
without accepted-prefix target re-forward; integrated target-forward execution remains Phase D3/D5 work.

### Phase D3 — Native DFlash drafter and draft context KV

Goal: stop calling the HF/PyTorch drafter with full context hidden every cycle.

- **Partial landed 2026-05-18:** load z-lab DFlash drafter BF16 weights through
  hipEngine loaders via raw safetensors payload offsets (no torch/NumPy BF16
  dependency), expose root/query request planning, materialize root+mask token
  ids/absolute positions/BF16 embeddings with `dflash_prepare_noise_inputs_bf16_i32`
  or FP16 target-embedding conversion with `dflash_prepare_noise_inputs_f16_to_bf16_i32`,
  run native target-hidden projection (`fc + hidden_norm`) with
  `dense_gemv_out_bf16` + direct-weight `dflash_rmsnorm_bf16`, validate BF16 dense
  projection to FP32 (`dflash_dense_bf16_to_f32`) for Q/K-style drafter
  projections, validate BF16 add/concat/SiLU/dense BF16 outputs for residual and
  MLP wiring, validate direct-weight head RMSNorm+rotary
  (`dflash_head_rmsnorm_rotary_f32`), and validate the correctness-first
  non-causal GQA attention primitive `dflash_gqa_attention_f32_bf16` against a
  NumPy BF16 oracle.
- **Landed 2026-05-18:** add `DFlashDraftKVCacheOwner` /
  `DFlashDraftKVCacheSpec` for fixed per-layer context K/V buffers, append-plan
  capacity checks, metadata reporting (`key_bytes`, `value_bytes`, `total_bytes`,
  phases `full_context_rebuild` / `append_materialize` / `query_only_drafter`),
  NumPy reference tests proving append-only K/V materialization matches a
  full-context rebuild prefix without clobbering suffix rows, and gfx1151
  `materialize_dflash_draft_kv_append_from_projected()` smoke that projects only
  newly appended rows, applies K norm/RoPE, writes fixed K/V cache rows, and
  updates positions/live-count metadata.
- Draft forward computes only root/query rows; context K/V are read from draft KV.
- **Landed 2026-05-18:** add compact draft lm-head top-k primitive
  `topk_f32_rows_i32`, candidate-only `DraftBatch` emission from top-k rows, and
  a deterministic one-layer tiny DFlash decoder-block smoke. Native top-k
  matches `fixtures/dflash/drafter_root_query_parent_fixture.json`, generated
  from the parent/PyTorch `dflash.py` harness, exactly (`[[5,9,6],[8,2,5]]`)
  with native-vs-parent logits `max_abs=4.802e-03`.
- **Landed 2026-05-18:** add `scripts/dflash_chain_correctness_harness.py`, a
  stable-prompt correctness loop for budgets `N={2,4,8}` and reject/partial/full
  cases. It connects deterministic drafter candidates → candidate-only
  `DraftBatch` → `TargetVerifyBatch` root insertion → GPU accept summary → GPU
  commit-copy check, records generated ids and commit rows, matches same-session
  AR token streams exactly, keeps finite draft/verify flags true, and marks
  `throughput_claim_eligible=false`.
- **Landed 2026-05-18:** add `scripts/dflash_chain_e2e_bench.py`, a full-model
  diagnostic driver that executes the packed target and native DFlash drafter in
  one resident target session with same-session AR control. It captures target
  hidden taps on device, proposes a top-1 chain through z-lab drafter weights,
  verifies through either the fallback `serial_in_place_single_slot` verifier or
  the default `native_bulk_bplus1` verifier, and emits schema-2 rows with
  acceptance, split timings, D2H counts, graph status, backend/arch, memory, and
  promotion eligibility. The retained gfx1151 smoke artifact after Phase A+B+C
  is exact/finite but slower than AR (`0.289x`) and `performance_claim=false`.
  Drafter per-call sync time dropped from `~95-100 ms` to `~68 ms` (-32%);
  decode tok/s rose from `~14.7` to `~18.3` (median across 5 runs, +24%).  The
  follow-up retained native-B+1 artifact is exact and proves one target verifier
  call per draft cycle, but regresses to `0.124x` AR because the tiny-row verifier
  path is still launch/kernel dominated.  The drafter graph prototype proves
  exact graph replay of the fixed-shape `propose()` body, but exact-context graph
  keys do not repeat in decode (`cache_entries=10`, no hits), so graph validation
  doubles drafter time rather than reducing launch overhead.  The first QKV
  projection fusion is correct and profiled, but neutral (`69.6 ms/call` vs
  `68.9 ms/call` no-fusion) because the mixed-output branchy grid does not remove
  the dominant work.  Warm verifier scratch and accept reordering reduce native
  verifier latency, but B={1,2,4,8} still fails the faster-than-serial gate.
- Remaining integration work: optimize/capture/fuse the native bulk verifier and
  pursue higher-leverage drafter fusions (attention/O-proj or MLP families) or
  reusable context-bucket-safe graph kernels; promote only if the full-model
  chain beats same-session AR.

### Phase D4 — DDTree compiler and tree verify

Goal: implement topk>1 DDTree without changing the target verifier shape.

- Build a CPU reference DDTree compiler first:
  - inputs: per-position draft topk ids/scores;
  - outputs: flat `tokens[N]`, `parents[N]`, `depths[N]`, `positions[N]`,
    `ancestor_mask[N,N]`, edge map from parent node to draft token.
- Default budget: `4` excluding root. Add explicit opt-in for `8`; do not use
  `16/22` until small budgets win.
- Implement device buffer upload/fill for compiled tree.
- Verify with the same target forward as chain mode.
- Add `dflash_accept_tree_kernel` to follow the accepted path from target top1
  comparisons across tree edges.
- Commit accepted path:
  - linear layers: copy final accepted node state to live state;
  - full-attention layers: compact accepted DFS/path K/V rows into consecutive
    live cache positions;
  - hidden taps: copy committed rows for draft KV append.
- Correctness gate: exact greedy equality; DDTree acceptance path matches a
  CPU tree-walk oracle; no DFS-state contamination.

### Phase D5 — Graph capture and fixed-shape replay

Goal: convert fixed `N` rows into low-overhead graph replay.

- **Landed 2026-05-18:** fixed verify graph bucket keys for chain `N={2,4,8}`
  include backend, active C, context/page buckets, mode, draft depth, tree shape,
  top-k, experts, replay steps, and fixed buffer address fingerprints.
- **Landed 2026-05-18:** `scripts/dflash_verify_graph_capture_smoke.py`
  validates fixed-address replay for N={2,4,8} against direct mode exactly and
  records graph validation in
  `benchmarks/results/2026-05-18-hipengine-dflash-verify-graph-buckets-diagnostic.json`.
  Rare page-bucket shapes fall back to direct launch semantics with an explicit
  fallback reason.
- **Landed 2026-05-18:** `scripts/dflash_chain_e2e_bench.py --drafter-graph
  {off,auto,validate}` prototypes HIP graph capture for the native DFlash
  drafter `propose()` body.  Validation mode proves graph replay candidate
  equality vs direct fallback, but the retained E2E artifact records the blocker:
  exact `context_tokens` buckets are unique per decode cycle, so there are no
  cache-hit replays and no speedup.
- Warm up JIT/build outside capture.
- Capture only kernels and device copies whose addresses are stable.
- Do not bake per-cycle scalar values into graph nodes unless they live in
  device buffers read at replay time.
- Validation: graph replay exact output equality vs direct mode for every bucket;
  report `decode_step_graph_validation=true` / graph validation artifact fields.

### Phase D6 — Benchmark and promotion

Initial retained shapes:

| Shape | Purpose |
| --- | --- |
| HumanEval/53, decode=64, chain N=2/4/8 | compare directly to current Python harness |
| HumanEval medium, decode=128, chain N=4/8 | acceptance robustness |
| code/instruct/prose mini-suite, decode=64 | genre sensitivity |
| 4K prompt / 128 decode, chain N=4/8 | long-context sanity before promotion |
| DDTree budget=4 after chain wins | topping, not baseline |

Promotion gates:

- exact same-session AR equality;
- finite prefill/draft/verify logits;
- DFlash chain > 1.10x AR on HumanEval short before DDTree promotion;
- DDTree budget=4 improves chain by >= 5% without memory/correctness regressions;
- peak allocation under the active gate for the model/workload;
- compact artifact under `benchmarks/results/` and rollup update per
  `docs/BENCHMARK.md`.

## DDTree details to preserve

### Flat tree ABI

DDTree is not a different verifier; it is a different way to fill the same
verifier-internal `TargetVerifyBatch`. The public `DraftBatch` still carries
candidate rows only; the verifier inserts the root row:

```text
slot 0: root / current target token
slot i: candidate tree node
parents[i]: parent slot index, or -1 for root
positions[i]: committed_position + depth[i]
mask[i,j]: 0 if j is an ancestor/self of i, -inf otherwise
```

The flat order must be topological: `parents[i] < i`. That is the property the
Conv/GDN t-loop kernels rely on.

### Acceptance semantics

For a tree edge `parent -> child` labeled with draft token `token(child)`, the
child is accepted only if target top1 at `parent` equals `token(child)`. The
accepted output is the longest followed path from root. If no child matches,
commit root plus the target correction/bonus. If a path fully accepts, use the
last accepted node's target prediction as the next root/bonus according to the
same semantics as the chain DFlash harness.

Never commit draft-only tokens beyond what target verification accepted.

### State semantics

- Linear-attention Conv/GDN state for node `i` is the state after consuming the
  token at slot `i` along that node's parent path.
- Full-attention K/V row for node `i` is stored in tree scratch first. On commit,
  accepted path rows are copied into live consecutive cache positions.
- Rejected sibling/subtree rows must not remain visible in live state or live KV.

### Budget policy

Default DDTree budget is `4` because:

- DDTree-MLX found 5 verified nodes to be the sweet spot for this hybrid model.
- Higher budgets increase recurrent verify work and memory pressure quickly.
- Our current chain/bulk evidence already shows budget 8 is near break-even;
  budget 12/16 needed a pack8 threshold fix and did not beat budget 8.

Use HumanEval/code prompts for speed gates. Instruct/prose are required for
robustness reporting but should not be expected to show hipfire-style multipliers.

## Future optimizations after the native baseline

Do not start these before D1-D6 establish a winning native chain path.

1. **Grouped small-N linears / projection batching.**
   Current profiles show W4 pack8 GEMV dominates verify GPU time. If the native
   chain path still has `eta > 0.55`, investigate small-N grouped GEMM/GEMV
   variants for QKV/Z/out and MLP paths.

2. **Boundary fusion.**
   Fuse RMSNorm + rotate + pack8 projection where profile shows launch overhead
   and memory traffic are significant. Keep unfused fallbacks registered.

3. **Persistent-cache GDN variant.**
   Lucebox's `_persist` idea writes accepted recurrent state directly into the
   persistent target state buffer. Consider after copy-commit is correct and
   profiled hot.

4. **Quantized KV / Q8 state.**
   hipfire's Q8/asym KV is part of its memory and bandwidth story. Port only
   after BF16/BF16-state correctness and speed are established.

5. **HIP graph multi-bucket cache.**
   Add graph buckets for multiple budgets and prompt regimes after a single
   fixed bucket proves exact and faster.

6. **Speculative server scheduling.**
   Once c=1 native DFlash wins, integrate with batching/admission. Do not mix
   server scheduling questions with first c=1 verifier bring-up.

## Anti-patterns / stop signs

- A DFlash speed claim without `verify_eta`, rows/output, draft time, and exact
  AR equality is not actionable.
- A path that replays accepted prefixes through the target model is a debug
  path, not the production path.
- A tree verifier that launches per depth or per node from the host is not the
  reference shape.
- Adaptive controller work before chain DFlash beats AR is premature.
- Budget 16/22 as a default is unsupported by the strongest references.
- Full logits copied to host per verify row will destroy the intended economics.
- Python scalar `.item()` / CPU list conversion inside the hot loop is a bug.
- Any kernel micro-optimization without a rocprof time-share audit belongs in
  `~/amd-gpu-tuning`, not hipEngine.

## First concrete hipEngine tasks

1. Refresh `docs/MTP.md`/this plan for the `gfx1151` + shisa packed target and
   port the parent benchmark metric schema without inheriting PyTorch hot-loop
   assumptions (`scripts/dflash_speculative_bench.py` owns the artifact shape).
2. Add DFlash source-lineage entries and fixtures for corrected tree Conv/GDN
   plus z-lab DFlash drafter metadata.
3. Validate packed target and drafter safetensors/config metadata offline.
4. Add a native chain `TargetVerifyBatch` with fixed device buffers and CPU
   reference acceptance tests.
5. Port corrected tree Conv/GDN t-loop wrappers into
   `hipengine/kernels/hip_gfx1100/linear_attn/` with `gfx1151` alias coverage.
6. Wire chain verify through the Qwen3.6/Qwen3.5 PARO target runtime with
   persistent node state rings and K/V scratch.
7. **Landed 2026-05-18:** add GPU top1 + chain accept summary (`argmax_f32_rows_i32`,
   row lm-head, `dflash_accept_chain_i32`) with gfx1151 smoke parity vs CPU oracle.
8. **Landed 2026-05-18:** add `dflash_commit_chain_i32` verified state/KV/output
   copy-select with reject/partial/full and multi-request non-leakage smoke.
9. Benchmark HumanEval/code chain N=1/2/4/8 against same-session packed-target
   AR on native `gfx1151`.
10. Only after chain > AR: add DDTree budget=4 compiler and tree accept/commit.

## Round-2 optimization plan (post-MTP M13)

> **Trigger:** Re-engage DFlash optimization once MTP M13.C (C-side per-layer
> dispatcher) lands and the shared native verifier has measurably improved on
> the 9-prompt `mtp-bench.py --mode hipengine-current` suite. MTP and DFlash
> share the same target verifier; verifier wins land in MTP first because it
> iterates without a second model load, then port to DFlash.
> See [`MTP.md`](MTP.md#m13--launch-count--host-dispatch-consolidation-2026-05-23)
> for the current verifier track. This section is the **next** DFlash round,
> not work in progress.

### Where Round-1 left us (and what changed in MTP since)

Round-1 (Phase D1–D3 native bring-up, 2026-05-18/19) landed a correct chain
verifier and an operational DFlash drafter, but every speed gate failed:

- Phase A+B+C chain DFlash on the packed PARO target: `0.289x` AR on gfx1151,
  serial fallback verifier
  ([artifact](../benchmarks/results/2026-05-18-hipengine-dflash-chain-full-model-e2e-phaseABC-diagnostic.json)).
- Native B+1 verifier: `0.124x` AR — exact and correct, but per-row wall
  worse than serial because the tiny-row path is launch/kernel dominated
  ([artifact](../benchmarks/results/2026-05-18-hipengine-dflash-chain-full-model-e2e-nativebulk-diagnostic.json)).
- True-batched chain verifier (`--full-attn-chain-mode batched`): 6–8% faster
  than `c1_loop` at B=2/4, but still 2.0–5.0x slower than serial c=1 across
  all B because each batched cycle still pays B+1 rows of multi-token MoE
  ([artifact](../benchmarks/results/2026-05-19-hipengine-dflash-chain-batched-vs-c1-loop-speedgate-diagnostic.json)).

The DFlash and MTP paths share the same target verifier shape and the same
wall. MTP picked up the verifier optimization track in M11–M13 because the
MTP prompt suite iterates without a second model load; the verifier wins port
back to DFlash. As of M13.B.0 (2026-05-23) the MTP wall is `0.53x` AR with
`cycle_cost = 3.61 AR-token-eq` for B=3 verifier
([artifact](../benchmarks/results/2026-05-23-hipengine-mtp-bench-suite-w7900-m13.b0.json)),
still under unity.

### Reference baseline: BeeLlama v0.2.0 (RTX 3090)

`~/beellama.cpp/CHANGELOG.md` v0.2.0 (DFlash on llama.cpp b9275 CUDA 13.1) on a
single RTX 3090 with Qwen 3.6 27B Q5_K_S target + DFlash drafter Q4_K_M:

| Workload | Baseline | DFlash | Speedup | Acceptance (acc/draft / acc/total) |
| --- | ---: | ---: | ---: | --- |
| Task store module ~1K tok | 37.2 tok/s | 163.9 tok/s | **4.40x** | 67.7% / 89.2% |
| KV report module ~1K tok | 34.6 tok/s | 157.7 tok/s | 4.56x | 58.8% / 88.9% |
| Doubly-linked list ~4K tok | 36.8 tok/s | 130.8 tok/s | 3.56x | 50.4% / 86.8% |
| Multi-turn coding ~28K tok | 33.3 tok/s | 64.6 tok/s | 1.94x | 24.9% / 72.9% |

Existence proof on gfx1100: hipfire on 7900 XTX gets `4.45x` on HumanEval
DDTree (see the Reference-numbers table above). The 4–5x DFlash class is
hardware-achievable on AMD; our `0.53x` is a software gap, not a hardware
ceiling.

Key structural choices in BeeLlama (cited for cross-reference, not for
blind copying):

1. **Verifier is one `llama_decode([id_last, draft0, …, draftN])` through the
   same ggml graph as AR decode** — no separate "batched verifier" code path.
   `tools/server/server-context.cpp` ~line 3936.
2. **DFlash drafter is a small 1-layer block-diffusion model with
   cross-attention over a ring of recent target hiddens**, with K/V projection
   caching across cycles so only newly committed rows are re-projected
   (`src/models/dflash_draft.cpp` `llm_build_dflash_draft` ~line 691;
   `dflash_kv_cache_ready_for_window` ~line 203).
3. **Drafter cross-context bucketed by power-of-2 (<=128) then 128-aligned**
   (`src/llama-context.cpp` `cross_bucket()` ~line 3649) so cycle-to-cycle the
   graph reservation is reused after ~7–8 buckets fill.
4. **Reduced verifier logits**: `llama_set_dflash_verify_logits(ctx, true, top_k)`
   makes the target graph emit `ggml_topk_ext` / `ggml_argmax_ext` in-graph and
   skip full-vocab readback (`src/models/qwen35.cpp` ~line 160).
5. **Hidden capture is graph-embedded** as a `ggml_cpy` into per-layer GPU
   rings, not a follow-up D2D pass (`src/models/qwen35.cpp` ~line 80).

### The ROCm 7.x graph-replay ceiling ("Gap 3")

`hipGraphLaunch` per-node overhead on ROCm 7.x at our ~1052-kernel DAG matches
direct `ctypes → hipModuleLaunchKernel` overhead. MTP M12.1 (2026-05-22) landed
HIP graph capture for the batched verifier, validated exact-AR, and measured
`33.3 ms cycle wall in both graph=auto and graph=off`
([artifact](../benchmarks/results/2026-05-22-hipengine-mtp-m12.1-w7900-graph-capture-diagnostic.json)).
The Python round-trip *is* removed on replay; ROCm's per-node graph runtime
overhead replaces it ~1:1.

CUDA on RTX 3090 with a smaller (~500–800 node) DAG does not have this
property — `cudaGraphLaunch` is a real win in BeeLlama's regime. We cannot
copy that piece directly. Closing this gap requires either (a) reducing the
GPU launch count via actual fusion (subject to the cost-model wall in L1
below) or (b) accepting it as a ROCm runtime characteristic at this DAG size.

### Things we have already tried (do NOT repeat without a different cost model)

| Attempt | Artifact | Result | Lesson |
| --- | --- | --- | --- |
| HIP graph capture/replay for batched chain verifier (MTP M12.1) | [`2026-05-22-...m12.1-w7900-graph-capture-diagnostic.json`](../benchmarks/results/2026-05-22-hipengine-mtp-m12.1-w7900-graph-capture-diagnostic.json) | Cycle wall unchanged (33.3 ms both graph=auto and graph=off); exact-AR preserved | At ~1052 graph nodes on ROCm 7.x, `hipGraphLaunch` per-node overhead ≈ ctypes overhead. Graph capture wins require fewer nodes first, not better keys. |
| DFlash drafter HIP graph capture (Phase D5 prototype) | [`2026-05-18-...drafter-graph-validate-diagnostic.json`](../benchmarks/results/2026-05-18-hipengine-dflash-drafter-graph-validate-diagnostic.json) | Validation passes (10/10 candidates); exact `context_tokens` buckets are unique per decode cycle → 0 cache-hit replays; replay regresses to 133.8 ms/call vs no-graph 68.9 ms/call | Drafter graph bucket keys must match BeeLlama's `cross_bucket()` shape. Exact `n_enc` keys do not repeat. |
| QKV projection fusion for DFlash drafter | [`2026-05-18-...drafter-qkv-fusion-diagnostic.json`](../benchmarks/results/2026-05-18-hipengine-dflash-drafter-qkv-fusion-diagnostic.json) | Bit-exact, rocprofv3 confirms `dflash_qkv_proj_bf16_mixed_kernel` runs; retained E2E neutral (69.6 ms/call vs 68.9 ms no-fusion) | Mixed-output branchy grid did not amortize the dominant work. Single-kernel fuse without a tile/work-amortization analysis is unlikely to win. |
| Verifier warm-scratch reuse | [`2026-05-18-...verifier-warmscratch-speedgate-diagnostic.json`](../benchmarks/results/2026-05-18-hipengine-dflash-verifier-warmscratch-speedgate-diagnostic.json) | Verify seconds −26–35% at B={1,2,4,8} but still 1.8–4.9x slower than serial c=1 | Reduces verifier scratch churn but does not reduce per-row target compute. Necessary, not sufficient. |
| True-batched chain verifier vs c1_loop | [`2026-05-19-...batched-vs-c1-loop-speedgate-diagnostic.json`](../benchmarks/results/2026-05-19-hipengine-dflash-chain-batched-vs-c1-loop-speedgate-diagnostic.json) | 6–8% faster at B=2/4 over c1_loop; neutral or slower at B=1/8; still 2.0–5.0x slower than serial c=1 | B+1-row MoE/router cost grows roughly linearly with B; cycle wall pays B+1 rows of multi-token MoE regardless of early chain rejection. Retained as DDTree infrastructure foundation only. |
| Branching top-K DDTree (B=4, K=2) | retained 2026-05-19 row in `dflash_chain_e2e_bench` | Exact-AR, GPU-accept matches CPU, beats chain at B=2/4/8 but still loses to serial c=1 | Tree topology is correctness-free but verifier row-cost dominates; DDTree before chain DFlash > AR is premature (matches Anti-patterns rule above). |
| Selected-MoE rotate+GEMV fusion (MTP M13.B.1) | [`2026-05-23-...m13.b1-fusedon-rejected.json`](../benchmarks/results/2026-05-23-hipengine-mtp-verifier-rocprof-w7900-m13.b1-fusedon-rejected.json) | −40 launches/pass but **+71.8% kernel time**; `moe_gate_up_dual_gemv` ms/pass +664% | Cost model: a fuse that saves N launches but multiplies per-block work by M loses when M × block_count > N × launch_overhead. For verifier shape M ≈ `out_packs × top_k` ≈ 192 × 8 = ~1500x rotation work. |
| Shared-expert transposed-rotate fold (MTP M13.B.2) | [`2026-05-23-...m13.b2-fusedon-rejected.json`](../benchmarks/results/2026-05-23-hipengine-mtp-verifier-rocprof-w7900-m13.b2-fusedon-rejected.json) | −10 paro_rotate launches but +10 implicit `hipMemsetAsync` barrier resets → net 0; kernel time +0.5% | Single-kernel fuses with implicit host-side init (barrier resets, scratch zeros) swallow the dispatch saving. Account for host-side per-kernel overhead, not just launch count. |

### Round-2 punchlist (in dependency order)

Baseline for "Expected Δ" columns: MTP M13.B.0 W7900 (`0.53x` AR, `cycle_cost
= 3.61`, `verify = 25.1 ms`, acceptance 30%,
[artifact](../benchmarks/results/2026-05-23-hipengine-mtp-bench-suite-w7900-m13.b0.json)).
Port each MTP win into DFlash before measuring DFlash-side rows. **A row
cannot move to "completed" without filling in the Actual Δ column.**

Exact-AR equality on the 9-prompt suite is mandatory for every row and is not
repeated in the Gate column.

| # | Task | Gate | Expected Δ | Actual Δ | Status |
| --- | --- | --- | --- | --- | --- |
| R2.1 | Pull MTP M13.C (C-side per-layer dispatcher) through to the DFlash chain verifier launch path once it lands in MTP. | `kernel_calls/pass` unchanged; host verify wall drops; rocprofv3 shows no in-kernel-time regression. | verify 25→20 ms (−20%); cycle_cost 3.61→2.9; 0.53x→0.65x | **Regression resolved; expected win not realized.** M14.dispatch.0-alpha (argtypes caching) landed 2026-05-23 as foundation: cycle_cost parity (3.61→3.64 within ±17% std, cProfile -6.6% launcher tottime). M14.dispatch.1-beta first looked regressed (verify 25.0→31.5 ms/cycle) but clean rerun isolated it to one-time lazy dispatcher/fn-table warmup charged to cycle 1 (`code_python` first cycle 271.8 ms vs 64.6 ms). Prewarming globals during resident build fixes the artifact: clean 9-prompt suite env ON vs OFF is parity (`cycle_cost 3.707→3.696`, `verify_ms 24.92→24.81`, exact all prompts). Artifacts: [on-prewarm](../benchmarks/results/2026-05-23-hipengine-mtp-bench-suite-w7900-m14.dispatch.1-prewarm-on-diagnostic.json), [off-baseline](../benchmarks/results/2026-05-23-hipengine-mtp-bench-suite-w7900-m14.dispatch.1-prewarm-off-baseline.json). | Partial: dispatcher infra is safe/default-off but does not provide the projected 20% verify win; move to next verifier reductions |
| R2.2 | Land DFlash drafter `propose()` chain on packed PARO target + z-lab drafter, re-measure same-session AR. Build on the existing Phase D3 work; close out the half-built drafter forward and wire to `verify_chain_bulk_and_commit`. | Exact greedy AR equality on quicksort + 3 representative prompts; finite logits across cycles. Retained DFlash row > current best Round-1 (`0.289x`). | New retained DFlash row at >=0.7x AR (acceptance bump from MTP 30% to DFlash ~50–65%). | _TBD_ | Pending R2.1 |
| R2.3 | Drafter cross-context bucketing matching BeeLlama `cross_bucket()` (<=16→16; <=128→next pow2; >128→128-aligned). Replace the exact `context_tokens` graph key. | Drafter graph cache hit rate >=50% after first 8 cycles on real decode; replay validates exact candidate equality. | Drafter time 68.9 ms/call → ~25–40 ms/call (graph replay amortizes for steady-state cycles). | _TBD_ | Pending R2.2 |
| R2.4 | Reduced-logits verifier path wired through DFlash chain accept summary. Confirm no full-vocab tensor materializes in steady state. | `HIPENGINE_VERIFY_GPU_ACCEPT=1` returns exact-AR; rocprofv3 shows no full-vocab lm-head kernel and no full-vocab D2H copy in the verifier window. | verify −0.5–1 ms/cycle (small; most groundwork already in MTP M12.6+). | _TBD_ | Pending R2.2 |
| R2.5 | Drafter K/V projection caching for the cross-attention window (BeeLlama `dflash_kv_cache` analog at `src/models/dflash_draft.cpp:203`). Re-project only newly committed rows; reuse cached K/V for the rest of the ring. | Cached K/V matches full re-projection bit-exact for crafted windows; finite candidate logits across 32 decode cycles. | Drafter time −25–40% in steady state. Combined with R2.3, drafter ~12–25 ms/call. | _TBD_ | Pending R2.2 |
| R2.6 | Adaptive draft budget B (BeeLlama `profit` controller analog in `tools/server/server-adaptive-dm.h`). Switch DFlash off when measured baseline wins. | Profit/no-profit transitions logged; observed speedup never regresses below 0.95x AR on any retained prompt. | Maintains best-of-(DFlash, AR) across genre mix; protects low-acceptance regimes (multi-turn coding) where chain DFlash regresses. | _TBD_ | Pending R2.4 |
| R2.7 | DDTree budget=4 branching (lands only if R2.5 closes chain > AR by >=10%). | Tree-shape verify accepts dense path; DDTree improves chain by >=5%. | +5–15% over chain (matches DDTree-MLX / Lucebox topping). | _TBD_ | Pending R2.5 win |
| R2.8 | Reduce verifier graph node count via principled fusion, **with the L1 cost-model check completed before implementation**. Survey candidates: add+RMSNorm pair, RoPE+QKV-cur pair, RoPE+QKV-noise pair — fuses where per-block work scales with `tokens`, not `out_packs × top_k`. | Each fuse passes `saved_launches × launch_overhead > added_per_block_work × block_count` on paper before code lands; bit-exact vs unfused chain after. | Each accepted fuse: −10–30 launches/pass, kernel time within ±noise. Aggregate target: ~100–200 fewer launches/pass. | _TBD_ | Pending R2.6 |
| R2.9 | Re-evaluate HIP graph capture (M13.D analog) after R2.8 drops node count toward ~600–750. | At least one of `graph_mode=auto/validate` beats `graph_mode=off` by >=5% on the 9-prompt suite. | If node count drops below ~750, graph capture may start paying; otherwise mark as confirmed ROCm runtime ceiling at this DAG size and stop. | _TBD_ | Pending R2.8 |

Promotion rule (carried from MTP.md): no DFlash speed row is accepted as a
performance claim until the economics artifact shows
`avg_visible_tokens_per_cycle / cycle_cost_ar_tokens > 1.0` on the same
prompt/workload, with exact AR equality and accepted-token provenance
preserved.

### M14.dispatch.1-beta design notes (implemented; historical)

Alpha (`hipengine/core/ctypes_cache.py` + 38 wrapper refactors) and the C-side dispatcher are committed. The clean post-prewarm result is parity rather than the projected win; keep the original design notes below as context for why this path was attempted:

1. **New TU** `kernels/hip_gfx1100/dispatch/moe_c1_dispatch.cpp` (plain C++, no
   HIP includes — it only calls existing `extern "C" hipengine_*` launchers via
   typed function pointers). Built via the existing `build_hip(sources=[...],
   family="moe_c1_dispatch", ...)` infra.
2. **One extern-C entry point** `hipengine_moe_c1_dispatch_fp16(const FnTable*
   fns, const Args* args)` where:
   - `FnTable` holds the 13 `void*` function pointers (router, paro_rotate1,
     gemv_awq_selected_dual, silu_mul_dual_rotate, gemv_awq_selected_pack8,
     5 shared-expert kernels, combine).
   - `Args` holds the ~45 `void*` ptrs and ~10 `int64_t` dims + 1 stream.
   - Inside the function, each `void*` is cast to its typed function-pointer
     signature and called in sequence with the matching subset of args. The
     compiler can keep state in registers across the calls.
3. **Python side**: a `MoeC1DispatchCache` object built once per layer at
   warmup that pre-resolves all 13 function pointers via `signed_kernel_fn`
   (so argtypes is set), pre-resolves all 45 weight tensor pointers (cached on
   the LayerRuntime), and snapshots the dims that are constant for the layer
   (hidden_size, num_experts, top_k, etc.). At runtime, `run_moe_c1_fp16`
   just updates the variable ptrs (hidden, residual, scratch, out) and the
   variable dims (tokens, group_size) in the cached `Args` struct, then makes
   one ctypes call.
4. **Two paths**: handle linear-attention vs full-attention shared-expert
   variants either via two separate entry points
   (`hipengine_moe_c1_dispatch_fp16_linear` / `_full`) or via a single entry
   point with a `shared_expert_kind` enum dispatched in-C.
5. **Gate** with `HIPENGINE_MOE_C1_C_DISPATCH=1` env var (default off) so we
   can A/B and never have to revert.

Expected savings (per M13.C cProfile attribution): 6–8 ms/pass = ~3–5%
cycle_cost reduction. Asymmetric across AR/spec (verifier has more launches
per cycle, so per-launch overhead reduction helps cycle_cost specifically),
unlike the alpha-level argtypes caching which sped AR and spec proportionally.

LoC budget: ~250–350 LoC total (150 C, 100 Python, 50 build-system, 50 tests).

### Lessons carried forward (living table)

Update this table whenever a Round-2 row lands with a non-trivial finding,
positive or negative.

| # | Lesson | First learned | Applies to |
| --- | --- | --- | --- |
| L1 | **Fusion cost model**: a fuse saving N launches but multiplying per-block work by M loses when M × block_count > N × launch_overhead. For verifier shape, M ≈ `out_packs × top_k` is typically 1000–2000x. Check the math before writing the kernel. | MTP M13.B.1 (2026-05-23) | Every speculative kernel fuse proposal |
| L2 | **Hidden host overhead**: single-kernel fuses with implicit host-side init (barrier resets, scratch zeros, lazy allocation) swallow the dispatch saving. Count host-side per-kernel cost, not just `hipModuleLaunchKernel` calls. | MTP M13.B.2 (2026-05-23) | Staged HBM kernels with atomic-style barriers |
| L3 | **ROCm 7.x graph ceiling**: HIP graph capture does not pay at >~1000-node DAGs because `hipGraphLaunch` per-node overhead ≈ direct dispatch. Reduce node count first; capture second. CUDA on consumer NVIDIA does not have this property at the same node count. | MTP M12.1 (2026-05-22) | Any HIP graph capture/replay work |
| L4 | **Bucket-shape graph keys**: exact-shape graph cache keys do not repeat in decode. Use BeeLlama-style power-of-2 / stride-aligned buckets (`cross_bucket()` shape) for any cross-context-dependent graph. | DFlash Phase D5 (2026-05-18) | Drafter graph capture; any context-dependent graph |
| L5 | **B+1 MoE linearity**: B+1-row MoE/router cost grows roughly linearly with B in the current "batched" path. Cycle cost in AR-token-eq tracks B+1 closely. Lowering per-row cost is the only way to win at B >= 4. | DFlash 2026-05-19 batched-vs-c1-loop | All speculative budgets B >= 2 |
| L6 | **Tree before chain is premature**: tree topology is correctness-free (DDTree exact-AR holds at the kernel level), but tree-before-chain > AR yields nothing. Topology helps 5–15%; drafter quality helps 50–200%. | DFlash 2026-05-19 branching top-K | Tree/DDTree work ordering |
| L7 | **Drafter quality dominates**: native MTP (replicated single decoder layer) caps acceptance ~30% at B=3. DFlash-class drafters (1-layer cross-attention over hidden ring) reach 60–90%. Drafter quality is the single largest visible/cycle lever. | MTP M11–M13 baseline acceptance | Drafter design choices |
| L8 | **Warm scratch is necessary, not sufficient**: persistent rings reduce verifier wall ~25–35% but do not unlock break-even alone. Keep as a building block. | DFlash 2026-05-18 warm-scratch | All scratch/cache discipline work |
| L9 | **Symmetric Python-side optimizations don't move cycle_cost**: argtypes caching, library handle caching, raw-int call sites — all real wins (−6.6% kernel-launcher tottime in M14.dispatch.0-alpha) but they speed AR and spec proportionally so the AR-tok-eq ratio stays flat. To move cycle_cost, the optimization must be asymmetric at the actual bottleneck. M14.dispatch.1 proved that bundling ctypes calls alone is not enough once steady-state launch/kernel work dominates. | M14.dispatch.0-alpha / M14.dispatch.1 (2026-05-23) | All host-overhead optimizations |
| L10 | **Prewarm before measuring cycles**: lazy ctypes/build-cache setup inside verifier cycle 1 can look like a persistent regression when economics averages over cycles. M14.dispatch.1's apparent `verify_ms 25.0→31.5` regression was just `code_python` cycle 1 `271.8 ms` vs `64.6 ms`; prewarming globals during resident build restored parity (`cycle_cost 3.707→3.696`). Any verifier benchmark must separate first-cycle warmup from steady-state. | M14.dispatch.1 prewarm (2026-05-23) | Benchmark harnesses; optional dispatcher/graph/fusion caches |

