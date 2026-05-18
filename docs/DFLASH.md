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

The same infrastructure should later support MTP and other speculative decoders,
but DFlash is the first native block-verifier target. See [`MTP.md`](MTP.md) for
the target-attached multi-token predictor plan that reuses this verifier/commit
infrastructure after DFlash lands.

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
  doubles drafter time rather than reducing launch overhead.
- Remaining integration work: optimize/capture/fuse the native bulk verifier and
  replace exact-context drafter graphs with reusable context-bucket-safe kernels
  or fusion; promote only if the full-model chain beats same-session AR.

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
