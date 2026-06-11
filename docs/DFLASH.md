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

### 2026-05-25 27B B=4 down-projection result

The retained Qwen3.6-27B-PARO dense + z-lab DFlash B=4/D64 9-prompt
branch-copy suite now defaults verifier-sized W4 MLP down projections
(`shared_expert.down_proj` and dense `mlp.down_proj`, `B+1<=8`) to the standard
row-wise pack8 GEMV path:

| Metric | Prior branch-copy | Down-GEMV default |
| --- | ---: | ---: |
| exact rows | 9/9 | 9/9 |
| AR decode | 32.83 tok/s | 32.83 tok/s |
| DFlash decode | 28.47 tok/s | 30.40 tok/s |
| vs AR | 0.867x | 0.926x |
| median row vs AR | 0.919x | 0.983x |
| avg accept length | 2.56 | 2.56 |
| target verify rows/output | 1.40 | 1.40 |

Selected-region verifier rocprof confirms this is a verifier-cost reduction,
not an acceptance change: total selected kernel time moves `1413.8 -> 1285.1
ms`; `awq_fusedw4_prefill_fp16` drops `555.7 -> 149.3 ms`; row-wise
`gemv_awq_pack8` rises `215.6 -> 469.1 ms`. The faster multi-row
weight-sharing down-projection path remains rejected because it was only `8/9`
exact on the same suite (`code:json_yaml_continuation`, token 48). Roll back
the promoted exact path with `HIPENGINE_W4_DOWN_PROJ_SMALL_BATCH=prefill`; force
the rejected prefill-dequant diagnostic with
`HIPENGINE_W4_DOWN_PROJ_SMALL_BATCH=multi_row`.

### 2026-05-26 GPU1 exact-dequant multi-row and profile route

GPU1 was selected with `HIP_VISIBLE_DEVICES=1` (RX 7900 XTX/gfx1100).  The 27B
dense target+drafter pair OOMs on this 24 GiB card during model load, so this
round used the fitting 35B A3B packed lane for full-model checks and a small GPU
unit test for the kernel arithmetic gate.

- Added `HIPENGINE_W4_DOWN_PROJ_SMALL_BATCH=multi_row_decode`: it keeps the
  weight-sharing multi-row down-projection launch shape, but matches the
  row-wise pack8 GEMV f32 dequantization instead of the old FP16 prefill-WMMA
  compatibility dequantization.  Synthetic GPU tests are bit-exact vs
  `gemv_awq_pack8_transposed_fp16` for rows `{2,5,8}`.
- On 35B A3B B=4/D16 4-prompt branch-copy, the opt-in mode stayed exact and
  moved all-chain DFlash `44.97 -> 46.87 tok/s` (+4.2%) with summed target
  verify seconds `1.200 -> 1.144` (-4.6%), but it is still far below AR and
  remains **opt-in** pending a larger exact suite.
- Added `scripts/dflash_build_profile_route_manifest.py`, which turns prior
  exact same-session rows into a zero-probe `{AR, chain}` manifest.  On the GPU1
  35B A3B D16 slice no chain row beat AR, so the generated manifest selects AR
  for all four prompts and measures exact `1.021x` vs same-session AR (variance
  around plain AR, not a speculative speed claim).

Retained artifact:
[`2026-05-26-hipengine-dflash-gpu1-multi-row-decode-profile-route.json`](../benchmarks/results/2026-05-26-hipengine-dflash-gpu1-multi-row-decode-profile-route.json).

### 2026-05-31 W7900 27B zero-probe profile route

The multiloop optimize pass first re-ran the exact B=4/D64 9-prompt W7900 lane
with the `multi_row_decode` default, then generated a profile-history route from
that exact same-session row:

```bash
PYTHONPATH=. python3 scripts/dflash_build_profile_route_manifest.py --input /tmp/multiloop-dflash-27b-w7900-baseline-source-iter2.json --output /tmp/multiloop-dflash-27b-w7900-profile-route-iter2-manifest.json --min-chain-speedup 1.0 --default-route ar
```

The manifest selects `chain` for five prompt IDs and `ar` for the four known
losers: `code:class_continuation`, `code:json_yaml_continuation`,
`code:humaneval_add`, `instruct:simple_qa_no_template`, and
`instruct:simple_qa_qwen_static_chat` route to DFlash chain; quicksort,
function-continuation, sort-third, and the short GSM8K-style prompt route to AR.

| Metric | all-chain `multi_row_decode` | profile route | profile route + verifier graph | graph-aware profile route + verifier graph | graph-aware + graph + bulk-direct | + budget-prefix drafter query | + `single_full_v` W4 site | + `single_linear_out` W4 site | threshold4 + json→AR | threshold4 regenerated route | + terminal AR tail | + json terminal20 route |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exact rows | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 |
| AR decode | 32.57 tok/s | 32.65 tok/s | 32.38 tok/s | 32.34 tok/s | 32.49 tok/s | 32.67 tok/s | 32.47 tok/s | 32.60 tok/s | 32.47 tok/s | 32.38 tok/s | 32.40 tok/s | 32.42 tok/s |
| DFlash/spec decode | 31.75 tok/s | 34.63 tok/s | 36.81 tok/s | 37.55 tok/s | 38.22 tok/s | 38.65 tok/s | 38.99 tok/s | 40.22 tok/s | 40.69 tok/s | 40.94 tok/s | 42.07 tok/s | 43.76 tok/s |
| vs AR | 0.975x | 1.061x | 1.137x | 1.161x | 1.176x | 1.183x | 1.201x | 1.234x | 1.253x | 1.265x | 1.298x | 1.350x |
| route mix | 9 chain / 0 AR | 5 chain / 4 AR | 5 chain / 4 AR | 7 chain / 2 AR | 7 chain / 2 AR | 7 chain / 2 AR | 7 chain / 2 AR | 7 chain / 2 AR | 6 chain / 3 AR | 7 chain / 2 AR | 7 chain / 2 AR + AR tail | 8 chain / 1 AR + prompt tail |

Verifier HIP graph capture (`--verifier-graph auto`) on the same route was the
first multiloop result to clear the numeric `>1.10x` gate: exact `9/9`,
`36.81 tok/s`, `1.137x` AR.  Rebuilding the profile manifest from an exact
all-chain verifier-graph row made the route graph-aware: quicksort and the short
GSM8K-style prompt became chain winners, yielding a 7-chain / 2-AR route at
`37.55 tok/s`, `1.161x` AR.  On this exact D64 gate, `bulk_direct` canonical
commit also stayed exact and improved the graph-aware route to `38.22 tok/s`,
`1.176x` AR.  Finally, limiting the drafter query rows to the root+B prefix
(`--drafter-query-mode budget_prefix`) stayed exact and nudged the route to
`38.65 tok/s`, `1.183x` AR.  Adding opt-in `single_full_v` to the W4
multi-row site mask also stayed exact and reduced summed target-verify time by
about 1%, reaching `38.99 tok/s`, `1.201x` AR.  Adding `single_linear_out` on
top of that mask stayed exact too, cut target-verify time by another 3.6%, and
reached `40.22 tok/s`, `1.234x` AR.  Lowering
`HIPENGINE_SMALL_BATCH_DECODE_THRESHOLD` to `4` made the verifier QK/QKV path
faster but was non-exact for `code:json_yaml_continuation`; forcing that prompt
to AR yielded an exact 6-chain / 3-AR route at `40.69 tok/s`, `1.253x` AR.
Regenerating the threshold4 route from all-chain row evidence then added
`code:function_continuation` as an exact/profitable chain winner while keeping
json-YAML and sort-third on AR, reaching `40.94 tok/s`, `1.265x` AR.  Adding a
default-off terminal AR tail guard (`--terminal-ar-tokens 5`) then skipped only
cycles whose remaining decode horizon was below a full B=4 draft window, avoiding
adaptive probing while cutting rows/output from `1.259` to `1.220` and reaching
a 3-run median `42.07 tok/s`, `1.298x` AR.  The follow-up retained route lets
the profile manifest override terminal AR cutoffs per prompt: json-YAML, the
previously fast but non-exact chain row, now routes through chain for its safe
prefix and switches to AR with `terminal_ar_tokens=20`, while the other prompts
keep `terminal_ar_tokens=5`.  This exact 8-chain / 1-AR route reaches a 3-run
median `43.76 tok/s`, `1.350x` AR.  Chain rows report graph validation success
with `captured_validated_miss`/`replayed` statuses.  This remains a
**diagnostic profile-history route**, not a promoted default: the route depends
on prior prompt history rather than a deployable online classifier, verifier
graph capture is still opt-in, `bulk_direct` exactness is only established for
this gate, budget-prefix proposals can differ from the z-lab block-query
contract, threshold `4` has known non-exact chain rows, and prompt-specific
terminal-tail AR routing is only proven on this D64 suite. `single_linear_out`
and `single_full_v` were later promoted into the default exact-safe W4 site mask
by the 2026-06-11 MTP D32 9-prompt gates.
Retained artifacts:
[`2026-05-31-hipengine-dflash-27b-profile-route-multiloop.json`](../benchmarks/results/2026-05-31-hipengine-dflash-27b-profile-route-multiloop.json),
[`2026-05-31-hipengine-dflash-27b-profile-route-verifier-graph.json`](../benchmarks/results/2026-05-31-hipengine-dflash-27b-profile-route-verifier-graph.json),
[`2026-05-31-hipengine-dflash-27b-graph-aware-profile-route.json`](../benchmarks/results/2026-05-31-hipengine-dflash-27b-graph-aware-profile-route.json),
[`2026-05-31-hipengine-dflash-27b-graph-aware-route-bulk-direct.json`](../benchmarks/results/2026-05-31-hipengine-dflash-27b-graph-aware-route-bulk-direct.json),
[`2026-05-31-hipengine-dflash-27b-graph-aware-route-budget-prefix.json`](../benchmarks/results/2026-05-31-hipengine-dflash-27b-graph-aware-route-budget-prefix.json),
[`2026-05-31-hipengine-dflash-27b-graph-aware-route-single-full-v.json`](../benchmarks/results/2026-05-31-hipengine-dflash-27b-graph-aware-route-single-full-v.json),
[`2026-05-31-hipengine-dflash-27b-graph-aware-route-single-linear-out.json`](../benchmarks/results/2026-05-31-hipengine-dflash-27b-graph-aware-route-single-linear-out.json),
[`2026-05-31-hipengine-dflash-27b-threshold4-json-ar-route.json`](../benchmarks/results/2026-05-31-hipengine-dflash-27b-threshold4-json-ar-route.json),
[`2026-05-31-hipengine-dflash-27b-threshold4-regenerated-route.json`](../benchmarks/results/2026-05-31-hipengine-dflash-27b-threshold4-regenerated-route.json),
[`2026-05-31-hipengine-dflash-27b-threshold4-terminal-ar-tail.json`](../benchmarks/results/2026-05-31-hipengine-dflash-27b-threshold4-terminal-ar-tail.json), and
[`2026-05-31-hipengine-dflash-27b-threshold4-json-terminal20-route.json`](../benchmarks/results/2026-05-31-hipengine-dflash-27b-threshold4-json-terminal20-route.json).

### 2026-06-11 hardening rerun: 27B dense DFlash accepted

After the verify graph-capture fixes landed for the shared MTP/DFlash verifier,
the deployable 27B dense lane was rerun on W7900/gfx1100 with the production
native-bulk settings: Qwen3.6-27B-PARO dense target, z-lab Qwen3.6-27B-DFlash
drafter, 9 prompts, D64, `B=4`, `top_k=2`, `whole_cycle_gate=0.90`,
`native_bulk_bplus1`, `full_attn_chain_mode=batched`, `branch_copy`, and verifier
graph `auto`.

Result: **`40.10 tok/s` DFlash vs `32.57 tok/s` same-session AR = `1.231x`**,
with exact AR equality on `9/9` rows and finite AR/draft/verify logits. Aggregate
rows/output is `1.160`, average accept length is `2.237`, and multi-token
acceptance is `0.616`. This supersedes the previous retained `1.1615x`/`1.164x`
deployable gate rows for the same lane.

The rerun also fixed the artifact decision metadata for native-bulk rows:
`scripts/dflash_chain_e2e_bench.py` now promotes a native bulk artifact to
`status=accepted` and `performance_claim=true` when all exact/finite correctness
gates pass and aggregate speedup exceeds the `>1.10x` rule. Artifact:
[`2026-06-11-hipengine-dflash-27b-dense-hardening-rerun.json`](../benchmarks/results/2026-06-11-hipengine-dflash-27b-dense-hardening-rerun.json).

### 2026-06-08 deployable routing: online whole-cycle drafter-confidence gate

The 27B profile-route reaches `1.35x` but is **non-deployable** — it routes whole
prompts from prior measured per-prompt speedup. The online `AdaptiveBudgetController`
only reaches safety (`~1.0x`) because it pays speculative probe costs to learn
acceptance. We closed part of that gap with a **cheap pre-verifier signal**.

**Oracle (does drafter confidence predict acceptance?).** Instrumented the bench
to emit, per cycle, the drafter's depth-1 top-1 softmax probability `p1`
(`--draft-top-k 2`; `HIPENGINE_DFLASH_CONF_ORACLE_OUT=<jsonl>`) alongside the
accepted count. W7900, 27B PARO + z-lab DFlash, 9-prompt D64 (161 cycles):
**`corr(p1, accepted) = 0.705`**; `p1` mean is **0.669 when accepted=0** vs
**0.954 when accepted≥1**; `P(accept≥1)` is `~0.25` for `p1<0.8`, **0.94** for
`p1∈[0.9,0.97]`, **1.00** for `p1≥0.97`. The cheap drafter confidence is a strong,
deployable acceptance predictor.

**Whole-cycle gate.** `--whole-cycle-gate thr` (CLI flag; env
`HIPENGINE_DFLASH_WHOLE_CYCLE_GATE=thr` retained as a backward-compat override for
artifact reproduction): if the drafter's depth-1 `p1 < thr`, drop the whole cycle
to AR (verify root only); else run the full chain. This is a *whole-cycle*
decision — unlike `--draft-p-min`, which truncates the chain mid-stream (cutting
good deeper drafts; it measured `0.92x` at `p_min=0.8`, worse than all-chain).
The flag requires `--tree-mode chain --draft-top-k 2` and is mutually exclusive
with `--draft-p-min`; the resolved threshold is recorded in the artifact
`workload.whole_cycle_gate`. When both the flag (`>0`) and env are set, the flag
wins.

W7900, 27B 9-prompt D64, exact `9/9` on every row:

| Routing | speedup vs AR | deployable? |
| --- | ---: | --- |
| all-chain (no gate) | 1.027x | yes (but barely >AR) |
| whole-cycle gate @0.85 | 1.099x | **yes (online, no history)** |
| **whole-cycle gate @0.90** | **1.147x** | **yes** |
| whole-cycle gate @0.95 | 1.106x | yes (over-gates) |
| offline profile-route | 1.35x | **no** (needs prior per-prompt history) |

So the confidence gate captures roughly half the offline profile-route's gain but
is **fully online and exact** — the first deployable >1.10x exact DFlash row on
this lane. Threshold is per-model/suite (clean-separation point ~0.9 here),
calibrated from the oracle. Artifact:
[`2026-06-08-hipengine-dflash-deployable-confidence-gate.json`](../benchmarks/results/2026-06-08-hipengine-dflash-deployable-confidence-gate.json).
**Promoted to a CLI flag (2026-06-08).** The env gate is now the first-class
`--whole-cycle-gate` flag (exact-AR re-verified on a 1–2 prompt smoke after the
promotion; the env var still activates the gate when the flag is left at its `0.0`
default). Remaining follow-ups: wire it into the engine DFlash decode API;
threshold auto-calibration (a short warmup window to set `thr` online from the
oracle's clean-separation point); optionally combine with terminal-AR-tail and
per-prompt budget levers toward the offline `1.35x` ceiling.

### 2026-06-02 hipfire replication, exactness audit, and importable lessons

Retained diagnostic:
[`2026-06-02-hipfire-vs-hipengine-27b-4096-512-diagnostic.json`](../benchmarks/results/2026-06-02-hipfire-vs-hipengine-27b-4096-512-diagnostic.json)
(`performance_claim=false`).  The 4096/512 shape row used token id `9707`
repeated 4096 times to force the same prompt length across tokenizers; it is a
shape/perf diagnostic, not a quality prompt.

| Engine | Mode | Prefill tok/s | Decode tok/s | Same-session vs AR | Peak VRAM GiB | Correctness / notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| hipEngine 27B PARO | AR target-only | 629.02 | 28.46 | 1.00x | 31.64 | forced 512 decode, graph replay |
| hipEngine 27B PARO+DFlash | DFlash B=4 | n/a | 24.95 | 0.891x | 33.18 | exact vs same-session AR; accept length 4 every cycle |
| hipfire MQ4/q8 | AR baseline | 591.97 | 33.30 | 1.00x | 18.69 | `--ar-baseline` still loads draft, so peak includes target+draft |
| hipfire MQ4/q8 + DFlash | DFlash B=16 | 595.22 | 180.12 | 5.41x | 18.78 | emitted 513 tokens for max 512; first 512 tokens match AR; normalized-to-512 decode is 179.77 tok/s |

The speed class is real on favorable prompts: hipfire's synthetic row still
matches the first 512 target-AR tokens and is ~5.4x its AR baseline.  However,
the same implementation is not an exact-greedy path by default on a broader
prompt audit.  The original 10-prompt token-id comparison between hipfire AR and
default hipfire DFlash found strict exact rows `1/10`, prefix-equal-to-min rows `6/10`,
and hard mismatches before the shared output length on `4/10` rows
(`hipfire_merge_sort_thinking_off`, `code:quicksort_prefix`,
`instruct:simple_qa_no_template`, `instruct:simple_qa_qwen_static_chat`).  A
slower `--no-tape` + `HIPFIRE_PREFILL_BATCHED=0` rerun did not repair those
mismatches.  A follow-up stable-fixture reproducer now lives at
`scripts/hipfire_dflash_exactness_audit.py`; on the first 10 committed
`fixtures/dflash/stable_prompts.jsonl` rows (`--max 128 --no-chatml --kv-mode q8`)
it reports strict exact rows `1/10`, prefix-equal-to-shared-length rows `8/10`,
hard mismatches `2/10` (`code:quicksort_prefix` at token 54 and
`code:function_continuation` at token 42), and over-emission past `max=128` on
`7/10` rows.  Artifact:
[`2026-06-02-hipfire-dflash-exactness-audit.json`](../benchmarks/results/2026-06-02-hipfire-dflash-exactness-audit.json).

Interpretation: a mismatch before the shared output length means DFlash emitted
and accepted a token that the target model's greedy AR run did **not** choose at
the same position.  For hipEngine's exact mode that is simply incorrect output,
not a speculative speedup.  It may still be usable by a system that explicitly
opts into approximate speculative generation and validates quality by task-level
metrics, but it cannot satisfy our exact-token gate (`exact_match_ar` for every
retained row).  Over-emitting past `max` also inflates emitted-token throughput
unless results are truncated/normalized.

Reproducer command:

```bash
PYTHONPATH=. HIP_VISIBLE_DEVICES=0 \
python3 scripts/hipfire_dflash_exactness_audit.py \
  --demo /tmp/hipfire-target/release/examples/dflash_spec_demo \
  --target /home/lhl/.hipfire/models/qwen3.6-27b.mq4 \
  --draft /home/lhl/.hipfire/models/qwen36-27b-dflash-mq4.hfq \
  --prompts fixtures/dflash/stable_prompts.jsonl --max-prompts 10 \
  --max 128 --ctx 8192 --kv-mode q8 --temp 0.0 --no-chatml \
  --json benchmarks/results/2026-06-02-hipfire-dflash-exactness-audit.json
```

Importable exact-safe lessons for the next hipEngine pass:

1. Prefer a verifier-native hot loop with persistent scratch/state over adapting
   an AR session per cycle; avoid full-logit materialization and host token loops.
2. Use B=8/B=16 only behind exact, high-accept routes and a profit/VRAM gate;
   hipfire's win occurs at near-perfect acceptance, while our measured B=15 row
   still loses when acceptance does not scale.  A same-day hipEngine synthetic
   4096/512 repeated-token sweep confirms the conditional: B=8 is exact at
   `33.80 tok/s` (`1.211x` AR, avg accept `7.98`), and B=15 is exact at
   `38.72 tok/s` (`1.389x` AR, avg accept `15.0`), but combined-process peak
   VRAM rises to `41.07 GiB`.  Artifact:
   [`2026-06-02-hipengine-dflash-27b-4096-512-b8-b15-diagnostic.json`](../benchmarks/results/2026-06-02-hipengine-dflash-27b-4096-512-b8-b15-diagnostic.json).
3. Keep profile/history routing, per-prompt draft budgets, and terminal AR tails
   as exact-safe policy levers.  Online probes are only a fallback because failed probes are costly.
4. Treat Q8/asym KV as a full storage+attention kernel-family project, not a
   flag flip; hipfire's low VRAM comes from target format plus q8 KV kernels.
5. Copy tape/rollback and fixed-address graph ideas only when same-session AR
   equality survives the full prompt suite; non-exact acceptance shortcuts stay
   out of the promoted hipEngine path.

### 2026-05-26 W7900 27B multi-row-decode default

The real 27B dense lane was validated on GPU0/W7900 (`HIP_VISIBLE_DEVICES=0`)
after the GPU1 OOM.  `multi_row_decode` passed the B=4/D64 9-prompt
branch-copy exact gate and is now the default verifier-sized W4 down-projection
mode for `B+1<=8`.  Force the prior row-wise path with
`HIPENGINE_W4_DOWN_PROJ_SMALL_BATCH=gemv`, roll back further with `prefill`, and
keep `multi_row` as the rejected prefill-dequant diagnostic.

| Metric | Prior row-wise default | `multi_row_decode` default |
| --- | ---: | ---: |
| exact rows | 9/9 | 9/9 |
| AR decode | 32.71 tok/s | 32.50 tok/s |
| DFlash decode | 30.20 tok/s | 31.74 tok/s |
| vs AR | 0.923x | 0.977x |
| target verify sum | 15.75 s | 14.83 s |
| target verify rows/output | 1.40 | 1.40 |
| avg accept length | 2.56 | 2.56 |

An explicit env-on run measured `31.95 tok/s` and summed verifier `14.72 s`, so
the retained no-env default run is within normal same-session variance.  This is
default-on exact speed work, not a speculative throughput claim: aggregate speed
is still below AR and below the `>1.10x` promotion gate.  Retained artifact:
[`2026-05-26-hipengine-dflash-27b-multi-row-decode-default.json`](../benchmarks/results/2026-05-26-hipengine-dflash-27b-multi-row-decode-default.json).

### 2026-05-25 adaptive D64 probe guard

On the same 27B B=4/D64 suite, the old adaptive default
`--adaptive-probe-amortization-tokens 64` still paid one startup DFlash probe on
every prompt.  That improved all-chain `0.926x` to `0.976x`, but it was still
worse than AR because losing prompts paid a full failed verifier cycle.  The
benchmark harness default is now `128`, so a 64-token decode uses AR fallback
unless the caller explicitly lowers the probe guard:

| Policy | vs AR | DFlash/spec tok/s | rows/output | accepted draft tokens |
| --- | ---: | ---: | ---: | ---: |
| all-chain down-GEMV | 0.926x | 30.40 | 1.40 | 412 |
| adaptive probe guard 64 | 0.976x | 32.03 | 1.06 | 93 |
| adaptive probe guard 128 | 0.998x | 32.76 | 1.00 | 0 |
| offline `{AR, chain}` oracle | 1.046x | 34.34 | mixed | n/a |

This is a safety default, not a speedup: online probing remains too expensive
for D64 unless a prompt classifier can predict the four chain-winning prompts
without paying a speculative cycle.

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

5. **Verifier-sized W4 down-projection GEMV.**
   For Qwen3.6-27B dense DFlash B=4, shared+dense MLP down projections at
   `B+1<=8` are cheaper through the standard row-wise pack8 GEMV path than
   through `awq_fusedw4_prefill_fp16`, while preserving the exact AR suite.

6. **R3: persistent per-layer node-state rings.**
   Reusing per-layer scratch for `tree_conv_state`, `tree_recurrent_state`, and
   row intermediates cut peak memory by ~0.94 GiB at bs=8 and ~2.10 GiB at
   bs=16. It was performance-neutral in Python because commit/allocations were
   not the main bottleneck, but the memory discipline is required for graph
   capture and fixed-address native execution.

### llama.cpp PR #21845 follow-up: small-row verifier projections

[`ggml-org/llama.cpp#21845`](https://github.com/ggml-org/llama.cpp/pull/21845)
adds a SYCL MTP verifier path that handles multiple RHS columns (`ncols <= 8`)
in one quantized mat-vec kernel.  The direct code is not a DFlash port target,
but the same small-row rule applies to DFlash's root+candidate verifier rows:
when `rows=B+1` is in the 2..8 range, projection kernels should share each
quantized weight stream across rows whenever that preserves exact target-AR
outputs.

Carry these checks into the next DFlash verifier profile pass:

- Re-audit `rows == 1` / `tokens == 1` gates in verifier-hot projection paths;
  a small verifier batch should not fall back to row-wise GEMV just because the
  optimized-layout gate was written for single-token decode.
- For B=1/2/4/8 DFlash chain/tree rows, record which W4/W8 projections use
  multi-row/read-once-weight kernels and which still use prefill or row-loop
  fallbacks.  Keep this as a callsite-marked rocprof table, not an inference
  from kernel names alone.
- Treat MTP M12.2 (W8A16 LM-head weight sharing) and M12.6 (gated W4 pack8
  multi-row sites) as the local precedent.  Extend coverage only when exact
  same-session AR rows remain green; otherwise keep the site default-off and log
  the numerical mismatch.
- Do not chase a standalone SYCL-style port.  The deliverable is a DFlash/MTP
  shared verifier dispatch audit plus exact multi-row coverage for the remaining
  profitable sites, especially W4 down/full/linear projections with `rows<=8`.

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
| hipfire | `~/amd-gpu-tuning/reference/hipfire` | `crates/hipfire-arch-qwen35/src/speculative.rs`, `qwen35.rs::TreeVerifyCtx`, `forward_prefill_batch*`, `rdna-compute/src/dispatch.rs::gated_delta_net_q8_tree_batch_seq` | Closest C++/HIP/gfx1100 shape: persistent scratch, batched verify, tree parent indices, native hot loop. Copy the runtime shape, **not** its default exactness policy; see the 2026-06-02 audit above. |
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
host overhead.  Reference headline numbers are not exact-greedy claims unless a
same-session target-AR token audit proves equality; hipfire's default DFlash path
fails that audit on several local prompts, so treat it as a throughput/existence
proof rather than an exact acceptance policy.

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
   hipfire's Q8/asym KV is part of its memory and bandwidth story.  The
   2026-05-25 triage confirmed this is not a small runtime flag in hipEngine:
   paged KV write, paged/tree attention, and AOTriton cache-backed paths are
   BF16-typed today.  Treat Q8/asym KV as a full storage + verifier-attention
   kernel-family port.

5. **HIP graph multi-bucket cache.**
   Add graph buckets for multiple budgets and prompt regimes after a single
   fixed bucket proves exact and faster.

6. **Speculative server scheduling.**
   Once c=1 native DFlash wins, integrate with batching/admission. Do not mix
   server scheduling questions with first c=1 verifier bring-up.

## Anti-patterns / stop signs

- A DFlash speed claim without `verify_eta`, rows/output, draft time, and exact
  AR equality is not actionable.
- A speculative path that accepts a token different from same-session target AR
  is approximate generation, not exact DFlash.  It is only usable behind an
  explicit non-exact quality policy, never under hipEngine's exact-token gate.
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

### R2.2 W7900 result: real z-lab drafter is correct, still below AR

After the missing drafter snapshot was restored, the R2.2 command ran on W7900
with the packed PARO target plus `z-lab/Qwen3.6-35B-A3B-DFlash` revision
`42d3b34d588423cdae7ba8f53a8cf7789346a719`:

```bash
python3 scripts/dflash_chain_e2e_bench.py \
  --target-model /models/hipengine/Qwen3.6-35B-A3B-PARO-full4096-e5-packed-MTP-BF16 \
  --drafter-model z-lab/Qwen3.6-35B-A3B-DFlash \
  --backend hip_gfx1100 --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --max-prompts 4 --decode-tokens 16 --draft-budgets 4 \
  --verifier-mode native_bulk_bplus1 --full-attn-chain-mode batched \
  --hardware-gpu 'AMD Radeon Pro W7900' \
  --json benchmarks/results/2026-05-23-hipengine-dflash-r2.2-w7900-b4-d16-4prompt-diagnostic.json
```

Result: exact same-session AR equality and finite logits on quicksort plus three
representative code prompts. Aggregate AR is `108.17 tok/s`; DFlash chain is
`44.69 tok/s`, or `0.413x` AR. This beats the old Round-1 best (`0.289x`) by
~43% but is far below the expected `>=0.7x` and not promotable. Acceptance is
better than native MTP but not BeeLlama-class: avg accepted draft tokens/cycle
`1.42`, visible tokens/cycle `2.46`, multi-token accept `38.5%`, full-B=4 accept
`11.5%`. The cycle wall is split between target verify (`31.1 ms/cycle`) and
drafter (`23.8 ms/cycle`) for total `55.1 ms/cycle`; with AR at ~`9.24 ms/token`,
cycle cost is ~`5.96` AR-token equivalents. The row-level spread is large:
quicksort stays at `0.287x` while `class_continuation` reaches `0.645x`, so
profit control and prompt gating remain required even after kernel wins.

A follow-up `--sync-draft-phases` one-prompt B=4/D8 diagnostic splits the
drafter wall accurately: target verify `36.9 ms/cycle`, drafter `25.0 ms/cycle`,
drafter decoder layers `19.6 ms/cycle`, drafter lm-head `2.8 ms/cycle`, and
drafter top-k/readback `0.4 ms/cycle`
([artifact](../benchmarks/results/2026-05-23-hipengine-dflash-r2.2-w7900-b4-d8-sync-phase-diagnostic.json)).
Context projection + KV-cache extension is already effectively free
(`0.022 ms/cycle`, rebuild rows `0`, cached rows `369` in the sync artifact), so
the BeeLlama-style K/V caching item is not a remaining large lever in this code
path.

**Implication for the punchlist:** R2.2 is complete/unblocked but failed the
0.7x target. R2.5 is already present as append-only projected-context + per-layer
K/V caches; remaining drafter work is R2.3 shape-safe graph capture or actual
decoder-layer kernel reduction. R2.4/R2.8 must still reduce target verifier cost.
DDTree remains premature until chain DFlash is near or above AR.

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
| R2.1 | Pull MTP M13.C (C-side per-layer dispatcher) through to the DFlash chain verifier launch path once it lands in MTP. | `kernel_calls/pass` unchanged; host verify wall drops; rocprofv3 shows no in-kernel-time regression. | verify 25→20 ms (−20%); cycle_cost 3.61→2.9; 0.53x→0.65x | **Regression resolved; expected win not realized.** M14.dispatch.0-alpha (argtypes caching) landed 2026-05-23 as foundation: cycle_cost parity (3.61→3.64 within ±17% std, cProfile -6.6% launcher tottime). M14.dispatch.1-beta first looked regressed (verify 25.0→31.5 ms/cycle) but clean rerun isolated it to one-time lazy dispatcher/fn-table warmup charged to cycle 1 (`code_python` first cycle 271.8 ms vs 64.6 ms). Prewarming globals during resident build fixes the artifact: clean 9-prompt suite env ON vs OFF is parity (`cycle_cost 3.707→3.696`, `verify_ms 24.92→24.81`, exact all prompts). Dispatcher is now default-on with `HIPENGINE_MOE_C1_C_DISPATCH=0` opt-out. Artifacts: [on-prewarm](../benchmarks/results/2026-05-23-hipengine-mtp-bench-suite-w7900-m14.dispatch.1-prewarm-on-diagnostic.json), [off-baseline](../benchmarks/results/2026-05-23-hipengine-mtp-bench-suite-w7900-m14.dispatch.1-prewarm-off-baseline.json). | Done for plumbing: dispatcher infra is safe/default-on but does not provide the projected 20% verify win; continue with R2.2/R2.4 verifier reductions |
| R2.2 | Land DFlash drafter `propose()` chain on packed PARO target + z-lab drafter, re-measure same-session AR. Build on the existing Phase D3 work; close out the half-built drafter forward and wire to `verify_chain_bulk_and_commit`. | Exact greedy AR equality on quicksort + 3 representative prompts; finite logits across cycles. Retained DFlash row > current best Round-1 (`0.289x`). | New retained DFlash row at >=0.7x AR (acceptance bump from MTP 30% to DFlash ~50–65%). | **Correct/unblocked; speed still fails.** z-lab drafter metadata validates (`91/91` tensors) with the local packed target (`722/722`). W7900 B=4/D16 over quicksort + 3 representative code prompts: exact/finite all rows, AR `108.17 tok/s`, DFlash `44.69 tok/s` = `0.413x` AR; acceptance avg `1.42` draft tokens/cycle, visible `2.46` tokens/cycle, multi-token accept `38.5%`; verify `31.1 ms/cycle`, drafter `23.8 ms/cycle`, total `55.1 ms/cycle`. Artifact: [r2.2-w7900](../benchmarks/results/2026-05-23-hipengine-dflash-r2.2-w7900-b4-d16-4prompt-diagnostic.json); previous blocked preflight: [missing-drafter](../benchmarks/results/2026-05-23-hipengine-dflash-r2.2-w7900-missing-drafter-blocked.json). | Done for correctness and Round-1 improvement (+43% over `0.289x`); not promoted because chain remains < AR and < expected `0.7x` |
| R2.3 | Drafter cross-context bucketing matching BeeLlama `cross_bucket()` (<=16→16; <=128→next pow2; >128→128-aligned). Replace the exact `context_tokens` graph key. | Drafter graph cache hit rate >=50% after first 8 cycles on real decode; replay validates exact candidate equality. | Drafter time 68.9 ms/call → ~25–40 ms/call (graph replay amortizes for steady-state cycles). | **Architecturally correct on W7900, but no perf win (drafter is GPU-bound).** Landed behind `--drafter-bucket cross_bucket`: bucketed GQA attention with device-resident live-context scalar, compact two-pass iteration over [0..live) ⊕ [bucket..bucket+B), bit-equivalent to unbucketed kernel when live==bucket. 4-prompt B=4 D=16 W7900: validation_failures=0 across 4 buckets / 22 replays, 85% cache-hit replay rate, aggregate AR `0.389x` (vs R2.2 `0.413x`, -6%); drafter `28.23 ms/cycle` mean (vs R2.2 `23.87 ms/cycle`, +18%). Per-cycle replay math (prompt 1: 1 capture + 8 replay): replay cycle `~24.2 ms` ≈ R2.2 direct `~23.75 ms` (saves only `~0.45 ms` = `~3.6 us` per launch over `124` launches/cycle); first-capture cycle costs `~48 ms` (direct + validate-launch). For 3–9 cycles per prompt the capture-amortization is unfavorable. Matches the M14.dispatch.0/1 finding: drafter wallclock is dominated by 8 decoder-layer kernel-execution time (`19.6 ms/cycle` synced), not host launch overhead. Artifacts: [r2.3-w7900](../benchmarks/results/2026-05-23-hipengine-dflash-r2.3-w7900-b4-d16-4prompt-diagnostic.json), [sync-phase](../benchmarks/results/2026-05-23-hipengine-dflash-r2.2-w7900-b4-d8-sync-phase-diagnostic.json). | Architecturally landed (correctness gated), kept as opt-in via `--drafter-bucket cross_bucket`; no retained perf row. Next levers must move kernel-execution time, not launch overhead |
| R2.4 | Reduced-logits verifier path wired through DFlash chain accept summary. Confirm no full-vocab tensor materializes in steady state. | `HIPENGINE_VERIFY_GPU_ACCEPT=1` returns exact-AR; rocprofv3 shows no full-vocab lm-head kernel and no full-vocab D2H copy in the verifier window. | verify −0.5–1 ms/cycle (small; most groundwork already in MTP M12.6+). | _TBD_ | Ready after R2.2; likely small but low-risk verifier cleanup |
| R2.5 | Drafter K/V projection caching for the cross-attention window (BeeLlama `dflash_kv_cache` analog at `src/models/dflash_draft.cpp:203`). Re-project only newly committed rows; reuse cached K/V for the rest of the ring. | Cached K/V matches full re-projection bit-exact for crafted windows; finite candidate logits across 32 decode cycles. | Drafter time −25–40% in steady state. Combined with R2.3, drafter ~12–25 ms/call. | **Already present before R2.2; no remaining large delta.** The current `NativeDFlashChainDrafter` uses append-only projected-context and per-layer rotated-K/V caches (`warmup_context()` + `commit_context_rows()`). R2.2 sync: context projection `0.088 ms` total over 4 cycles (`0.022 ms/cycle`), `context_projection_rebuild_rows=0`, `kv_cache_cached_rows=369`; decoder layers, not context/KV projection, dominate. | Done/no-op for Round-2; keep cache discipline but focus next on R2.3 decoder graphing or verifier cost |
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
5. **Gate** with `HIPENGINE_MOE_C1_C_DISPATCH=0` opt-out. This was originally
   opt-in during validation; after prewarm fixed the first-cycle artifact it is
   default-on.

Expected savings (per M13.C cProfile attribution): 6–8 ms/pass = ~3–5%
cycle_cost reduction. Asymmetric across AR/spec (verifier has more launches
per cycle, so per-launch overhead reduction helps cycle_cost specifically),
unlike the alpha-level argtypes caching which sped AR and spec proportionally.

LoC budget: ~250–350 LoC total (150 C, 100 Python, 50 build-system, 50 tests).

## Round-3 optimization plan (post-R2.3)

After R2.2 (real z-lab drafter) and R2.3 (bucketed HIP graph cache) the bottleneck shape is fully measured on W7900 / gfx1100:

- **Mean cycle wall ≈ 62 ms** (verifier `36.9 ms` + drafter `25.0 ms`); R2.2 sync-phase artifact.
- **Mean visible tokens / cycle ≈ 2.42**, per-prompt range `[1.67, 4.00]` (see break-even table below).
- **Cycle cost @ AR rate ≈ 6.65 AR-tok-equivalents** (vs 107 tok/s AR baseline); break-even needs either `visible/cycle ≥ 6` (impossible at B=4) or `wall ≤ 22–36 ms` (-35% to -73%, prompt-dependent).
- **Per-replay launch overhead saved by R2.3 graph capture ≈ `~3.6 µs/launch × ~124 launches = ~0.45 ms` ≈ 2% of cycle**. Host-launch overhead is decisively NOT the bottleneck (matches M14.dispatch.0/1 finding for MTP).
- **Drafter dense GEMVs run at ~3% of W7900 bandwidth limit** (current naive per-output-element kernel; see R3.3 roofline below).

Round-3 splits the remaining work into three parallel tracks plus a regression-guard track:

| Track | What | Win criterion | Carry from R2 |
| --- | --- | --- | --- |
| **A** | **Drafter kernel work** — push drafter wall toward bandwidth-bound floor (currently 3% of peak). | Drafter cycle wall ≤ 8 ms on W7900 (-60% from `19.6 ms` decoder layers + small overhead). | New: R3.3 paper → R3.4 kernel |
| **B** | **Verifier fusion** — cost-model-gated kernel fuses to drop verifier wall. | Verifier wall ≤ 30 ms (-19% from `36.9 ms`); each fuse passes the L1 cost model on paper before code. | R2.4, R2.8 (re-scoped as R3.6, R3.7) |
| **C** | **Topology / budget** — turn high-acceptance prompts into wins; protect low-acceptance prompts. | At least one retained prompt at ≥ 1.0x AR; suite mean ≥ 0.7x AR. | R2.6 → R3.1; R2.7 → R3.5 |
| **G** | **Regression guard** (cross-cuts all tracks) — never lose more than 5% to AR per prompt. | Adaptive controller: `dflash_tok_s / ar_tok_s ≥ 0.95` on every retained prompt across 9-prompt suite. | R3.1 |

Final Round-3 win state (all three tracks plus guard): on the 9-prompt suite, **aggregate AR ratio ≥ 1.0x** with no single prompt below `0.95x`, exact greedy AR equality, and an artifact whose `avg_visible_tokens_per_cycle / cycle_cost_ar_tokens > 1.0` (the promotion rule).

### Per-prompt break-even reference (W7900, R2.2 4-prompt B=4 D=16)

Ground truth from [r2.2-w7900](../benchmarks/results/2026-05-23-hipengine-dflash-r2.2-w7900-b4-d16-4prompt-diagnostic.json):

| Prompt | Accept rate | Accepted/cycle | Visible/cycle | Cycle wall (ms) | Cycle cost (AR-tok-eq) | Need wall < (ms) | Wall Δ needed | Archetype |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 (quicksort) | 20% | 0.67 | 1.67 | 56.83 | 6.17 | 15.4 | **-73%** | unwinnable → must route to AR |
| 2 | 42% | 1.67 | 2.67 | 53.23 | 5.83 | 24.3 | -54% | mid → needs both tracks A+B |
| 3 | **80%** | **3.00** | **4.00** | 56.08 | 6.18 | 36.3 | **-35%** | **achievable** → priority target |
| 4 | 38% | 1.29 | 2.29 | 53.14 | 5.56 | 21.8 | -59% | mid → needs A+B |
| **mean** | 40% | 1.42 | 2.42 | 54.82 | 5.94 | 22.1 | -60% |  |

Prompt 3 is the achievable-win archetype that R3.3/R3.4 (drafter wall) + R3.5 (DDTree) target. Prompt 1 is the unwinnable archetype that R3.1 (adaptive controller) must route to AR.

R2.2 sync-phase ground truth (quicksort, synchronized phases): `decoder_layers 19.60 ms` (78% of drafter wall), `lm_head 2.83 ms`, `context_projection 0.02 ms` (cached), `topk_and_readback 0.40 ms`, `final_norm 0.04 ms`, `noise_prepare 0.05 ms`. Source: [sync-phase](../benchmarks/results/2026-05-23-hipengine-dflash-r2.2-w7900-b4-d8-sync-phase-diagnostic.json).

### Round-3 punchlist

Exact-AR equality is mandatory for every row and is NOT repeated in the Gate column. Items are dependency-ordered; Phase-1 rows (R3.1–R3.3, R3.8) need 0 GPU time.

| # | Task | Track | Depends on | LoC est. | Gate | Expected Δ | Actual Δ | Status |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- |
| R3.1 | Adaptive budget controller — per-prompt or rolling-window profit signal that switches between DFlash and AR; BeeLlama `tools/server/server-adaptive-dm.h` analog. | C / G | none (data already in R2.2/R2.3 artifacts) | ~180 (Python + tests) | Controller never costs > 5% vs AR on any single prompt in the 9-prompt suite; state transitions logged in artifact. | Aggregate AR ratio cannot regress; low-accept prompts (prompt 1 archetype) route to AR and are no longer drag on the mean. | Native bulk→c=1 AR handoff fixed in two layers: c=1 decode scratch is canonicalized after bulk commits, and chain native bulk now scores on a branch slot then replays the accepted prefix through the canonical c=1 slot before drafter context commit. Strict probe policy now starts adaptive mode in `AR_PROBE`, uses one-cycle probes, demotes after one negative-profit cycle, and requires `--adaptive-probe-amortization-tokens 64` extra remaining tokens beyond the normal horizon before probing. With `--adaptive-min-remaining-tokens 128`, D160 4-prompt W7900 routes entirely to AR (`draft_calls=0`, rows/output `1.0`) and is exact at `0.995x` AR; D32 9-prompt guard is exact at `0.991x` AR, row min `0.955x`. | **Controller safety fixed / default-off as a speed path**; failed/negative-profit probe cost is avoided for D32/D160, but the retained policy does this by declining to speculate when the horizon cannot amortize a failed probe. DFlash speed promotion still needs a genuinely profitable longer-horizon probe or lower-cost canonical commit. |
| R3.2 | Paper L1 cost-model writeup for every verifier-side and drafter-side fuse candidate. NO kernel code; produces a markdown table inside DFLASH.md (or a separate `docs/DFLASH_FUSION_COSTMODEL.md` if it grows). | B (paper) | none | ~250 LoC markdown | Each candidate has explicit `saved_launches × launch_overhead` vs `added_per_block_work × block_count` calculation; each labeled pass/fail; output is the work order for R3.6. | Selects ≥ 1 verifier fuse + ≥ 1 drafter fuse that pass the L1 check. | Cost model landed in `docs/DFLASH_FUSION_COSTMODEL.md`; shortlist is C1 drafter add+rmsnorm + C5 verifier QKV-prepare fusion; no local fuse is a primary wall lever. | **Done (paper)**; R3.6 should stay narrow and not distract from R3.4. |
| R3.3 | Drafter dense kernel roofline + WMMA design doc. NO kernel code; produces a roofline analysis and a proposed kernel signature. | A (paper) | none | ~200 LoC markdown | Document captures: (a) per-block work/launches of current `dflash_dense_bf16_to_bf16_kernel`; (b) W7900 BW-bound floor for 16-row × 2048-dim BF16 GEMV (`~9 µs`); (c) measured per-op cost (`~306 µs` → 3% of BW peak); (d) proposed 16×16×16 BF16 WMMA tiling with LDS-staged input; (e) expected delta if kernel reaches 30–50% of BW peak. | Drafter dense roofline at `~25–50 µs/op`, ~5–10× over current. Decoder wall projection `19.6 → 5–8 ms`. Cycle wall projection `62 → ~45 ms` (prompt 3 break-even). | GPU-sanity doc landed in `docs/DRAFTER_DENSE_ROOFLINE.md`: current dense op mix is `16.32 ms/cycle` (~83% of decoder wall), 3–6% BW; 30% BW projects decoder `19.6 → ~6.3 ms`, cycle `62 → ~48.8 ms`. | **Done (paper + GPU sanity)**; R3.4 implementation is justified. |
| R3.4 | Implement R3.3's proposed WMMA dense kernels (`bf16_to_bf16` and `bf16_to_f32`) for the drafter; register against `KernelKey(...)` per AGENTS.md (no `if backend == ...` branches). | A (code) | R3.3 (paper sign-off) | ~430 (HIP + Python + tests) | KL ≤ 0.05 + top-1 ≥ 90% vs `kernels/cpu_reference/`; `rocprofv3 --kernel-trace` shows kernel duration in expected range and BW utilization ≥ 30%; same-session AR exact match on 9-prompt suite. | Drafter decoder `19.6 → 5–8 ms`; cycle wall `62 → ~45 ms`. Combined with R3.1 routing, prompt 3 archetype reaches ≥ 1.0x AR. | RDNA3 `v_wmma_f32_16x16x16_bf16` 16x16x16 wave32 tiling landed default-on (`HIPENGINE_DFLASH_DRAFTER_DENSE=wmma`); 16-row dense ops 1.7–5.7× faster (e.g. `(16,2048,6144)` `0.523 → 0.091 ms`); per-cycle dense `16.32 → ~3.88 ms`. 9-prompt B=4 D=32 same-session DFlash: aggregate `0.446 → 0.636x` AR; drafter wall `23.50 → 9.09 ms/cycle` (-61%); cycle `52.83 → 37.21 ms`; exact-AR all 9 prompts; best `0.911x` (class_continuation). | **Done (default-on)**; necessary but insufficient — still <1.0x AR aggregate. R3.6/R3.5/R3.7 still required. |
| R3.5 | DDTree wiring from chain-drafter row-wise top-K to existing `verify_tree_bulk_and_commit` (already landed gfx1151; bring up gfx1100). Carry-over from R2.7. | C (code) | R3.4 (chain wall drop) AND chain ≥ 1.0x AR on ≥ 1 prompt | ~200 (Python + tests) | DDTree branching_topk path runs on gfx1100 with K=2, exact-AR holds, accept rate per cycle improves ≥ 5% over chain on prompt 3 archetype. | +5–15% over chain on high-accept prompts; combined with R3.4 enables retained promotion. | gfx1100 4-prompt smoke: branching_topk K=2 B=4 functional and exact (4/4 prompts), aggregate `0.656 → 0.625x` (-5%), helps low-accept prompts (`+0.086` quicksort, `+0.042` json) but regresses high-accept (`-0.18` function, `-0.22` class). Tree verifier adds `~2.2 ms/cycle`; tree depth 2 caps acceptance vs chain depth 4 on high-accept prompts. B=8 K=2 worse across the board. Promotion requires per-prompt routing (gated on R3.1 c=1 fallback fix). | **Functional/non-promotable**; needs R3.1 adaptive routing or a depth-aware tree shape before default-on. |
| R3.6 | Implement ≥ 1 verifier-side fuse and ≥ 1 drafter-side fuse from R3.2's pass list. Each fuse keeps its unfused fallback per AGENTS.md. | B (code) | R3.2 (cost-model sign-off) | ~360 (HIP + Python + tests) per fuse | Bit-exact vs unfused chain on `kernels/cpu_reference/`; `rocprofv3` shows `saved_launches` matches prediction; 9-prompt cycle_cost reduction ≥ 3%. | Verifier `36.9 → 33–35 ms`. Combined with R3.4 cycle wall `45 → ~40–42 ms`. | **Drafter C1 landed default-off** (`HIPENGINE_DFLASH_DRAFTER_ADD_RMSNORM=on`): fused `dflash_add_rmsnorm_bf16` is bit-equal to the unfused chain on all `(rows, hidden_size)` parity tests and on the same-session DFlash 9-prompt B=4 D=32 W7900 run (`exact_match_ar=9/9`, `avg_accept_length=1.527` identical to R3.4 baseline). Drafter `9.09 → 9.03 ms/cycle` (-0.06 ms, -0.7%); cycle `37.21 → 37.19 ms` (-0.05%); aggregate AR `0.6359 → 0.6398` (+0.6%, within sampling noise). Saves the predicted `8 launches/cycle × ~3.6 µs = ~29 µs` exactly. **C5 (verifier QKV-prepare) deferred** because the cost model predicts ~`72 µs/pass` saving (≈ 0.2% of cycle) and a 3-variant fused kernel would cost ~400-600 LoC for sub-3% gate impact. | **C1 done (default-off)**, paper-PASS / cycle-gate FAIL by design (matches R3.2 prediction); C5 deferred. R3.7 + R3.1 are the higher-leverage Round-3 levers. |
| R3.7 | Reduced-logits verifier wire-through for DFlash chain accept (carry-over from R2.4); confirm via rocprof that no full-vocab lm-head kernel and no full-vocab D2H copy run in the verifier window. | B (code) | none (infra exists in MTP M12.6+) | ~50 (Python) | `HIPENGINE_DFLASH_VERIFY_FUSED_LM_HEAD=on` returns exact-AR; rocprofv3 shows no full-vocab lm-head kernel in verifier window. | -0.5 to -1 ms verifier. Small but free. | Fused W8A16 LM-head + argmax rows landed behind `HIPENGINE_DFLASH_VERIFY_FUSED_LM_HEAD=on`. Correctness: 6-shape parity test is bit-exact vs unfused LM-head + `argmax_f32_rows_i32`; 9-prompt B=4 D=32 chain bench remains exact with identical `avg_accept_length=1.527`. Rocprof gate passes: no `w8a16_linear_multi_row_kernel`, no `argmax_rows_stage1_i32_kernel`, no full-vocab D2H in the verifier window. Perf A/B is negative on W7900: fused-on `0.621x` AR vs fused-off `0.637x` AR, verifier `27.04 → 28.00 ms/cycle`, because stage-1 grid occupancy collapses from ~248K vocab blocks to `243 × 5` long-running blocks. | **Done / default-off**; architectural gate closed, speed promotion rejected on gfx1100. Future use needs a different work schedule, larger verifier row count, or true reduced-vocab sampling. |
| R3.8 | BeeLlama v0.2.0 profile read + writeup. Compare their default config and kernels to ours. | reference | none | ~120 LoC markdown | Notes document (`docs/BEELLAMA_PROFILE.md`) captures: their default B, drafter architecture / param count, verifier batching strategy, kernel fusion tricks they call out. Maps each finding back to R3.x items. | Validates whether 4.4x is achievable on similar hardware or reflects a different setup. Does not directly produce a code change. | Done in [`docs/BEELLAMA_PROFILE.md`](BEELLAMA_PROFILE.md). Default `B=16` flat DFlash with greedy drafter and adaptive `profit` controller (per-context-bucket EWMA over depth ladder `{0..8,10,12,14,16}` with hysteresis + baseline reprobe). Drafter is shallow-and-wide (5 layers @ `n_embd=5120` for Qwen3.5-27B). Reduced-logits verifier path (`set_dflash_verify_logits`) is real and has a long sampler/grammar blocklist. `4.40x` headline is RTX 3090 + Q5/Q4 GGUF target (~3× lower target memory traffic than our BF16 target) + adaptive B up to 16, so not directly comparable to W7900 BF16 R3.4 numbers. Confirms R3.1 + R3.7 are the right next levers. | **Done (paper)**; informs R3.1 controller fix priority and R3.7 streaming argmax-over-vocab kernel design |

### Verifier-Side WMMA Audit (2026-05-25)

The drafter R3.4 result does **not** transfer to the current target verifier as
a simple "turn WMMA on for all small rows" change.

Two verifier-side checks were run on W7900/gfx1100:

- Existing W4 projection dispatch forced to prefill/WMMA-style kernels for
  small verifier rows (`HIPENGINE_SMALL_BATCH_DECODE_THRESHOLD=1`
  `HIPENGINE_W4_MULTI_ROW_PACK8=off`) was exact on the 27B B=4/D64 one-prompt
  direct-bulk row, but slowed `0.862x -> 0.852x` AR and verifier mean
  `99.17 -> 100.36 ms/cycle`.
- New dense-GEMV WMMA variants for the true dense verifier family
  (`project_linear_attention_ab_fp16`) were added behind
  `HIPENGINE_VERIFY_DENSE_GEMV_WMMA=on`.  Unit correctness passed, but the
  skinny A/B shapes under-occupy WMMA: `5x5120x48` was `0.0066 ms` with the
  scalar GEMV kernel vs `0.0322 ms` with WMMA, and `5x2048x32` was
  `0.0062 ms` vs `0.0159 ms`.  Full-model checks stayed exact but regressed:
  27B direct-bulk `0.862x -> 0.835x` AR, 35B safe replay `0.255x -> 0.242x`.

Conclusion: the easy verifier-side R3.4 analogue is rejected.  The verifier
projection tax is not the same shape as the drafter's 16-row x 2048/6144 BF16
dense stack.  Keep the new dense WMMA gate default-off for reproducibility, and
prioritize lower-cost canonical commit / tree topology / Q8 KV over another
small-row WMMA dispatch flip.

### B+1=16 verifier threshold check (2026-05-25)

Follow-up to the WMMA audit: the relevant question is whether B=15
(`B+1=16` verifier rows) moves the 27B dense verifier into a profitable
prefill-shaped regime.  It does improve row efficiency, but not enough.

Retained diagnostic:
[`2026-05-25-hipengine-dflash-b15-verifier-prefill-threshold-diagnostic.json`](../benchmarks/results/2026-05-25-hipengine-dflash-b15-verifier-prefill-threshold-diagnostic.json).

Measured on Qwen3.6-27B-PARO dense + z-lab DFlash, W7900/gfx1100, one
quicksort prompt, D64, `--draft-budgets 15`, `native_bulk_bplus1`,
`full-attn-chain-mode batched`, and `canonical-commit-mode bulk_direct`:

| Mode | Correctness | AR tok/s | DFlash tok/s | vs AR | Verifier cost | Notes |
| --- | --- | ---: | ---: | ---: | --- | --- |
| default prefill-at-16 | exact | 33.17 | 19.31 | `0.582x` | `167.63 ms/cycle` | avg accept `2.94`, rows/output `4.02` |
| force GEMV cutoff at 16 | exact | 32.93 | 17.11 | `0.519x` | `194.19 ms/cycle` | `HIPENGINE_SMALL_BATCH_DECODE_THRESHOLD=16`; slower |
| default + A/B WMMA | exact | 33.01 | 19.13 | `0.580x` | `169.51 ms/cycle` | `HIPENGINE_VERIFY_DENSE_GEMV_WMMA=on`; neutral/slower |

The default path already routes the 16-row verifier through the prefill-style W4
projection kernels where current call-site gates allow it.  Forcing the affected
sites back to GEMV is slower, so the "16 rows unlocks prefill" part is true.
The economics still fail because acceptance does not scale with the larger
budget: B=15 accepts only `2.94` draft tokens/cycle on this prompt, while
target rows/output rises to `4.02`.  B=4 stays the measured optimum for this
27B DFlash lane.

### Qwen3.6-27B PARO/DFLASH lane check (2026-05-25)

The z-lab Qwen3.6-27B lane is dense PARO (`num_experts=0`,
`dense_paro_w4`), not the 35B A3B shared-expert target family.  The DFlash
metadata gate now accepts dense PARO target runtime tensors and the benchmark
artifact labels local Hugging Face snapshot paths by model id/revision instead
of inheriting the 35B defaults.

Measured on W7900/gfx1100 with the local 27B target snapshot
`84f86409151d4f2ec86dc0b6a096d5f6daa7f207` and DFlash snapshot
`0919688658996800f86b895034249700e9481106`:

| Mode | Shape | Correctness | AR tok/s | DFlash tok/s | vs AR | Key economics |
| --- | --- | --- | ---: | ---: | ---: | --- |
| safe replay | B=4, D64, 4 prompts | exact `4/4` | 32.60 | 14.69 | `0.451x` | avg accept `2.53`, rows/output `2.41`, 63-64 replay rows per prompt |
| branch copy | B=4, D64, 4 prompts | exact `4/4` | 32.80 | 28.19 | `0.859x` | avg accept `2.53`, rows/output `1.41`, zero replay rows, 144 state copies |
| direct bulk | B=4, D64, 4 prompts | exact `4/4` | 32.69 | 27.43 | `0.839x` | avg accept `2.53`, rows/output `1.41` |
| branch copy | B=4, D160, 1 prompt | exact `1/1` | 32.91 | 24.82 | `0.754x` | avg accept `2.14`, rows/output `1.59`, zero replay rows |
| direct bulk | B=4, D160, 1 prompt | exact `1/1` | 32.71 | 23.83 | `0.728x` | avg accept `2.14`, rows/output `1.59` |
| direct bulk budget sweep | B={1,2,4,8,12,15}, D64, 1 prompt | exact `6/6` | 32.75 | 21.45 aggregate | `0.655x` aggregate | B4 is best: `0.823x`; B8/B12/B15 regress to `0.706/0.644/0.582x` |

Interpretation: 27B dense PARO/DFLASH is a materially better speculation lane
than the current 35B A3B DFlash path.  It also confirms the budget conclusion:
B=4 remains the measured optimum; the 8-24-token range pays too many verifier
rows before acceptance catches up.  The remaining blocker is not drafter
quality on the best prompts, it is exact commit cost.  Branch-copy commit is the
first lower-cost exact canonical mode that materially closes the replay gap:
27B D64 four-prompt improves `0.451x -> 0.859x` AR and 35B A3B D32 one-prompt
improves `0.255x -> 0.340x` AR, both exact.  It still does not beat AR, but it
should replace safe replay as the conservative speed-work baseline when
`native_bulk_bplus1` chain mode is available.  Direct bulk remains diagnostic
until canonical-state equivalence is proven broadly.

Artifact:
[`2026-05-25-hipengine-dflash-qwen36-27b-w7900-diagnostic.json`](../benchmarks/results/2026-05-25-hipengine-dflash-qwen36-27b-w7900-diagnostic.json).

### Lower-cost commit / topology / Q8 KV triage (2026-05-25)

Retained diagnostic:
[`2026-05-25-hipengine-dflash-canonical-tree-q8-triage.json`](../benchmarks/results/2026-05-25-hipengine-dflash-canonical-tree-q8-triage.json).

- **Canonical commit:** `--canonical-commit-mode branch_copy` is exact on 27B
  D64 four-prompt, 27B D160 one-prompt, and 35B A3B D32 one-prompt.  It avoids
  c=1 replay by verifying on a branch slot and copying the committed branch
  state back to canonical.  A bounded-KV copy variant now passes the live
  context row count into `copy_slot_state`; this is neutral at D64 because the
  resident capacity is still one 256-token KV block, but it avoids copying
  unused future rows at longer capacities.
- **Tree topology:** `chain_as_tree` is exact but slightly slower than chain on
  27B B=4/D64 one-prompt (`0.843x` AR).  Balanced `branching_topk K=2` is
  exact but much slower at B=4 (`0.659x`) and B=8 (`0.612x`) because breadth
  steals depth from the top-1 path: avg accept drops from `2.50` to `1.74` at
  B=4.  Do not promote current balanced branching topology before a depth-aware
  tree shape exists.
- **Q8 KV:** current hipEngine verifier hot paths are BF16-KV typed end to end:
  paged KV write stores BF16, paged/full/tree attention kernels read BF16, and
  AOTriton cache-backed prefill explicitly rejects non-BF16 KV tensors.  hipfire
  obtains its KV advantage from a full storage+attention family (`Q8_0` /
  asym3 writers plus matching batched/tree attention kernels), not from a small
  flag.  A c=1-only Q8 path would not accelerate native bulk DFlash verify, so
  Q8/asym KV stays a dedicated kernel-family port rather than a micro-iteration.

### 27B B=4 routing/profile diagnostic (2026-05-25)

Retained diagnostic:
[`2026-05-25-hipengine-dflash-27b-routing-profile-diagnostic.json`](../benchmarks/results/2026-05-25-hipengine-dflash-27b-routing-profile-diagnostic.json).

The 9-prompt Qwen3.6-27B B=4/D64 branch-copy baseline confirms the reviewer
direction on per-prompt routing, but rejects balanced tree promotion:

| Policy | Correctness | AR tok/s | Policy tok/s | vs AR | Key result |
| --- | --- | ---: | ---: | ---: | --- |
| chain branch-copy | exact `9/9` | 32.83 | 28.47 | `0.867x` | 3 prompts beat AR: class `1.028x`, HumanEval add `1.109x`, simple QA `1.048x` |
| branching_topk K=2 | exact `9/9` | 32.92 | 21.16 | `0.643x` | Worse than chain on every prompt; avg accept falls instead of improving |
| adaptive one-cycle probe | exact `9/9` | 32.64 | 31.35 | `0.960x` | Probe cost/noise misses class/simple-QA winners and charges every loser |
| profile route `{AR, chain}` | exact `9/9` | 32.76 | 33.24 | `1.015x` | No-probe manifest routes only the 3 measured chain winners to DFlash |

The offline oracle over `{AR, chain, tree}` selects tree on `0/9` prompts,
chain on `3/9`, and AR on `6/9`, for `1.019x` AR.  The measured profile route
lands close at `1.015x`, but this is still below the formal `>1.10x` speed
gate and is not a deployable classifier.

Cycle accounting also narrows the next wall: branch-copy commit/state copy is
not the 27B B=4 bottleneck.  The chain baseline spends about `103-108 ms` per
DFlash cycle in target verify and `~20 ms` in the drafter, while commit/state
copy is ~`0.12 ms/cycle` (`commit_fraction ~= 0.1%`).  The next speed work
should focus on the B=4 target verifier cost and on profile/history-based
routing; online one-cycle probing is a safety fallback for this D64 shape, not
a speed path.

### 27B B=4 verifier rocprof + down-projection check (2026-05-25)

Retained diagnostic:
[`2026-05-25-hipengine-dflash-27b-verifier-rocprof-down-proj-diagnostic.json`](../benchmarks/results/2026-05-25-hipengine-dflash-27b-verifier-rocprof-down-proj-diagnostic.json).

The benchmark harness now has `--roctx` and
`--rocprof-selected-region dflash_verify` so `rocprofv3 --selected-regions`
can profile only the target-verifier windows.  On this host, selected regions
require the therock ROCTX SDK shim on `LD_LIBRARY_PATH`; the system
`/opt/rocm/lib/libroctx64.so` exposes range markers but not
`roctxProfilerResume/Pause`.

Verifier-only rocprof for the 27B B=4/D64 quicksort row confirms the current
wall is W4 target projection, not attention or commit:

| Verifier family | Kernel time | Share | Calls | Notes |
| --- | ---: | ---: | ---: | --- |
| `awq_fusedw4_prefill_fp16` single W4 projections | `555.7 ms` | `39.3%` | 2304 | Dense-MLP down projection dominates |
| Existing multi-row W4 dense | `300.4 ms` | `21.2%` | 1458 | Mostly gate/up and safe sites |
| Other W4/GEMV dense | `224.9 ms` | `15.9%` | 4032 | Includes verifier-size single GEMV |
| Linear/GDN chain state | `165.5 ms` | `11.7%` | 2610 | Recurrent linear-attn state path |
| Full attention | `62.0 ms` | `4.4%` | 288 | Not the primary wall at this context |

Experiment: route verifier-sized shared+dense MLP down projections through the
existing `gemv_awq_pack8_multi_row_transposed_fp16` path.  On the one-prompt
trace, `awq_fusedw4_prefill_fp16` drops `555.7 -> 146.7 ms`, selected-region
kernel time drops `1413.8 -> 1137.2 ms`, and target verify wall drops
`2.00 -> 1.62 s`; single-prompt speed improves `0.817x -> 0.971x` AR.

Promotion is rejected for correctness/economics:

| Variant | Scope | Correctness | vs AR | Result |
| --- | --- | --- | ---: | --- |
| shared+dense down multi-row + branch-copy | 9 prompts | failed `8/9` exact | `1.003x` | `code:json_yaml_continuation` diverges at token 48 |
| shared-down-only default mask | 9 prompts | exact `9/9` | `0.863x` | no suite win over the `0.867x` baseline |
| dense-down-only + branch-copy | first 4 prompts | failed | `1.011x` | reproduces the JSON/YAML failure |
| dense-down + canonical replay | first 4 prompts | exact | `0.498x` | replay fixes drift but costs too much |

Conclusion: dense-down multi-row is the first concrete verifier-cost lever that
can reach break-even, but branch-copy canonical state is not exact with it.
Do not default this dispatch.  Next useful work is a lower-cost exact canonical
commit/state-canonicalization path for dense-down multi-row, or a deterministic
verifier down-projection path that preserves the branch-copy speed.

### R3.1 design notes — Adaptive budget controller

**Goal:** Default DFlash on, but never lose to AR by more than 5% on any prompt. Required to make any future R3.4/R3.6 win promotable across the full 9-prompt suite (not just the prompt 3 archetype).

**Status 2026-05-24:** host controller and artifact logging remain default-off, but the native-bulk→c=1 AR handoff blocker is fixed. Root cause was verifier-sized resident scratch reused by later `tokens=1` decode: split views such as linear-attention `qkv/z` and `a/b` used offsets based on the bulk row count, while c=1 projection kernels write compact one-row layouts. `Qwen35ParoResidentSession` now canonicalizes linear-attention and MoE/MLP decode scratch after native bulk commits and on c=1 entry. The 4-prompt B=4 D=16 W7900 acceptance run with `--adaptive-budget on --verifier-mode native_bulk_bplus1 --full-attn-chain-mode batched` is exact (`4/4` rows) and AR fallback cycles are near same-session AR (`mean/p50 9.90/9.60 ms`, 24 cycles) instead of the interim root-only bulk fallback (`~18 ms/token`). Aggregate diagnostic speed is still below AR (`0.529x`, `performance_claim=false`), so promotion still requires the full 9-prompt guard and/or further DFlash speed work.

**Short-horizon guard addendum (2026-05-24):** a failed DFlash probe costs too
much to amortize on short decode windows, so the adaptive controller now has an
optional `--adaptive-min-remaining-tokens N` horizon guard. When the remaining
decode length is below `N`, the controller routes to AR with
`decision_reason=remaining_tokens_guard`; the terminal AR path skips unused
drafter hidden-tap capture and drafter context commits because no future DFlash
probe can occur as the remaining horizon only decreases. On the 4-prompt B=4
D=16 W7900 diagnostic with `N=128`, all rows are exact, `draft_calls=0`,
`target_verify_rows_per_output_token=1.0`, aggregate guarded decode is
`108.64 tok/s` vs same-session AR `108.01 tok/s` (`1.006x`, row range
`0.988–1.028x`). This is a controller safety result, not a speculative speedup.

**llama.cpp HIP MTP mining addendum (2026-05-24):** local
`/home/lhl/llama.cpp/llama.cpp-hip` contributes two relevant MTP ideas.
Commit `3e12fbdea` splits pre-norm hidden extraction from raw-logit copying
for MTP prompt decode; hipEngine already has the analogous device-resident
hidden-tap path and compact accept summaries, so there is no direct DFlash code
port there.  Branch `origin/gg/mtp-graphs-improve` commit `d7b1fd2af`
re-enables `p_min` for MTP drafts by sampling top-K and stopping low-confidence
tails.  hipEngine now mirrors that as default-off DFlash benchmark plumbing:
`--draft-top-k K --draft-p-min P` trims chain candidates at the first
low-confidence row and, when enabled, compacts the native verifier batch to the
surviving active rows instead of verifying padded inactive suffix rows.  This is
not a perf claim until a retained W7900/gfx1100 artifact proves exact-AR and an
economics win.

Follow-up diagnostic: direct use of that confidence gate as a speed lever is not
promotable at B=4/D16. `--draft-top-k 2 --draft-p-min 0.70` cut verifier rows
to `1.08/output` but dropped average acceptance to `0.91` and measured only
`0.414x` AR; `p_min=0.55` measured `0.383x` AR. Keep top-K confidence as
controller telemetry, not as a default chain verifier trimming policy.

**Strict probe addendum (2026-05-24):** failed probes were still too expensive
after the canonical commit fix because the controller paid four initial DFlash
cycles plus two retry probes before locking out.  Adaptive mode now starts in
`AR_PROBE` instead of assuming DFlash, uses one-cycle probes by default,
demotes from `DFLASH` after one negative-profit cycle, and adds
`--adaptive-probe-amortization-tokens` (default `64`).  A probe only runs when
`remaining_tokens >= adaptive_min_remaining_tokens + adaptive_probe_amortization_tokens`;
already-promoted `DFLASH` still uses the normal remaining-token horizon.  On
the D160 4-prompt W7900 guard with `min_remaining=128`, the new margin blocks
the unamortizable initial probe (`probe_amortization_guard` for the first 33
cycles, then `remaining_tokens_guard`), preserving exactness with
`draft_calls=0`, rows/output `1.0`, aggregate `0.995x` AR, and per-row range
`0.991–1.004x`.  The D32 9-prompt guard remains exact with `draft_calls=0`,
rows/output `1.0`, aggregate `0.991x` AR, row min `0.955x`.

**Budget sweep addendum (2026-05-24):** the current z-lab drafter config has
`block_size=16`, so the native chain can sweep at most `B=15` today (root row
plus draft rows), not B=24.  A W7900 D32 4-prompt sweep over
`B={1,2,4,8,12,15}` with the safe canonical replay path stayed exact on all
24 rows and found B=4 is the best current budget: mean AR ratios
`0.292/0.321/0.343/0.307/0.271/0.273x` for B1/B2/B4/B8/B12/B15.  Acceptance
saturates after B=4 (`avg_accept 1.89 -> 2.11`) while verifier rows/output
keeps rising (`2.84 -> 4.09 -> 6.50`), so the 8-24-token optimum seen in other
testing does not carry over to this runtime/model shape.  A short-window
`bulk_direct` diagnostic was exact and improved aggregate speed
`0.291x -> 0.417x` AR with the same B=4 optimum, but prior D160 `bulk_direct`
failed exactness on longer class/json rows; direct bulk remains diagnostic
until canonical bulk state is fixed.  `--drafter-query-mode budget_prefix`
reduced B=4 drafter wall only `8.91 -> 8.56 ms/cycle` because WMMA dense still
uses one 16-row tile, and it is neutral overall.

**Win state:**
- For every prompt in the 9-prompt suite, `spec_tok_s_with_controller / ar_tok_s ≥ 0.95`.
- Per-cycle artifact records: `mode_used`, `cycle_wall_ms`, `visible_tokens`, `profit_ms`, `controller_state`.
- Aggregate AR ratio is ≥ max(R2.2 baseline `0.413x`, R3.4 mean after kernel work).

**Approach (recommended algorithm):**

Reference: BeeLlama `tools/server/server-adaptive-dm.h`.

Per-cycle profit signal:

```
ar_ms_per_token  = 1000.0 / ar_decode_tok_s_estimate  # rolling avg of recent AR cycles
cycle_wall_ms    = drafter_ms + verifier_ms
visible_ar_eq_ms = visible_tokens_this_cycle * ar_ms_per_token
profit_ms        = visible_ar_eq_ms - cycle_wall_ms   # positive = DFlash winning
```

State machine:

```
state ∈ {DFLASH, AR_LOCKED, AR_PROBE}

init: state = AR_PROBE, probe_cycles = 1, profit_window = deque(maxlen=8)

on each cycle:
  if remaining_tokens < adaptive_min_remaining_tokens:
    run AR cycle; do not consume cooldown/probe state; continue
  elif state == AR_PROBE and remaining_tokens < adaptive_min_remaining_tokens + adaptive_probe_amortization_tokens:
    run AR cycle; do not consume cooldown/probe state; continue
  if state == DFLASH:
    run DFlash cycle; profit_window.append(profit_ms)
    if mean(profit_window[-1:]) < -5.0:
      state = AR_LOCKED; cooldown = 32
  elif state == AR_LOCKED:
    run AR cycle; cooldown -= 1
    if cooldown == 0:
      state = AR_PROBE; probe_cycles = 1; probe_profit = []
  elif state == AR_PROBE:
    run DFlash cycle; probe_profit.append(profit_ms); probe_cycles -= 1
    if probe_cycles == 0:
      if mean(probe_profit) > 0.5:
        state = DFLASH; profit_window.extend(probe_profit)
      else:
        state = AR_LOCKED; cooldown = 32
```

**Integration points (concrete):**
- New module `hipengine/speculative/adaptive_budget.py` containing `AdaptiveBudgetController` class with `should_use_dflash() -> bool`, `record(cycle_metrics)`, `summary() -> dict`.
- Call sites: `scripts/dflash_chain_e2e_bench.py:run_dflash_tokens` and `_run_dflash_chain_on_session`, between the budget check and the drafter `propose()` call.
- New CLI flag `--adaptive-budget {off, on}` (default `off` during validation; flip to `on` after measured pass).
- New CLI flag `--adaptive-probe-amortization-tokens N` (default `64`) for the extra remaining-token margin required before startup/retry probes.
- Artifact summary key: `spec.adaptive_budget` containing controller state transitions + per-cycle log.

**Grind budget:**
- ~120 LoC Python (controller + integration).
- ~60 LoC tests (synthetic profit traces).
- Effort: 0.5–1 day. **Abort if:** controller still flaps even with cooldown ≥ 16; the right fix is to add hysteresis or per-prompt routing, not more iteration.

**Risk:** Controller flapping on borderline prompts. Mitigated by the cooldown and 4-cycle moving average. Worst case the controller misroutes one or two prompts; the gate (`≥ 0.95x per prompt`) catches this.

**Acceptance bench command:**

```bash
python3 scripts/dflash_chain_e2e_bench.py \
  --target-model /models/hipengine/Qwen3.6-35B-A3B-PARO-full4096-e5-packed-MTP-BF16 \
  --drafter-model z-lab/Qwen3.6-35B-A3B-DFlash \
  --backend hip_gfx1100 --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --max-prompts 9 --decode-tokens 32 --draft-budgets 4 \
  --verifier-mode native_bulk_bplus1 --full-attn-chain-mode batched \
  --adaptive-budget on \
  --hardware-gpu 'AMD Radeon Pro W7900' \
  --json benchmarks/results/<date>-hipengine-dflash-r3.1-w7900-adaptive-controller.json
```

### R3.2 design notes — Verifier fusion L1 cost-model template

**Goal:** Produce a written cost-model verdict for every plausible fuse before any kernel work, so R3.6 starts from a paper-approved short list. Required by AGENTS.md L1 ("Check the math before writing the kernel").

**Win state:**
- A markdown table inside DFLASH.md (or `docs/DFLASH_FUSION_COSTMODEL.md` if it grows past ~200 lines) with one row per candidate fuse.
- Each row labels pass / fail / inconclusive with explicit numbers.
- The output is a 1–3-item shortlist for R3.6 to implement.

**Status 2026-05-23:** complete in [`docs/DFLASH_FUSION_COSTMODEL.md`](DFLASH_FUSION_COSTMODEL.md). Shortlist: C1 drafter add+rmsnorm + C5 verifier QKV-prepare fusion, with optional C4 combine+nextnorm only if API work is clean. Negative finding: local fuses are sub-millisecond polish, not the primary wall lever; R3.4 dense WMMA remains the main Round-3 wall-reduction track.

**Approach — cost model formula:**

```
LAUNCH_OVERHEAD = 3.6 µs / launch                      # measured (R2.3)
VERIFIER_LAUNCHES_PER_PASS ≈ 1011                       # MTP M13.B.0 rocprof
DRAFTER_LAUNCHES_PER_CYCLE ≈ 124                        # 8 layers × ~15 + outer
W7900_BW = 864 GB/s
W7900_BF16_WMMA_PEAK ≈ 120 TFLOPS

saved_ms     = saved_launches * LAUNCH_OVERHEAD
added_ms     = (added_redundant_compute_or_memory_per_block / peak_throughput) * blocks_per_pass

PASS  iff  saved_ms > added_ms
```

**Template per candidate (coder fills in):**

```
### Fuse candidate: <name>
- Phase: drafter | verifier
- Replaces kernels: A (current ms/pass, current launches/pass), B (...)
- New fused kernel signature: <sig>
- Saved launches: <delta>
- Saved overhead: <delta * 3.6 µs>
- Per-block added work expression (in terms of tokens, heads, hidden, block_size, etc.):
- Block count per pass:
- Added kernel time per pass:
- Verifier verdict: PASS / FAIL
- Drafter verdict: PASS / FAIL
- Notes / risks:
```

**Candidates to evaluate (initial list; coder may add):**

1. **add + RMSNorm fuse** (drafter and verifier post-attention and post-MLP residual paths).
2. **RoPE + QKV-cur fuse** (verifier query-side; per-block work scales with `tokens × heads`).
3. **RoPE + QKV-noise fuse** (drafter query-side; scales with `block_size × heads`).
4. **SiLU + mul + down-proj fuse** (drafter MLP).
5. **Final-norm + lm-head fuse** (drafter last row only).
6. **K rotate + KV write fuse** (drafter; replaces concat + rotary as separate launches).
7. **Verifier write-through extensions** (M13.B.0 already wrote `next_hidden` through; survey for any remaining write-throughs in router/expert combine paths).

**Grind budget:**
- ~250 LoC markdown.
- Effort: 1 day to do the math carefully for all 7 candidates. **Abort if:** the candidate list is exhausted with 0 passing fuses; that's a real finding (record it as L15 and skip R3.6 entirely).

**Risk:** Mis-estimating `added_per_block_work` for fuses that have asymmetric compute (e.g. RMSNorm reduction across hidden). Mitigation: cross-check against rocprofv3 per-kernel time of the existing unfused kernels (the per-kernel time IS the cost; we just need to subtract any redundant work that the fuse introduces).

### R3.3 design notes — Drafter dense kernel roofline

**Goal:** Quantify the gap between current naive drafter dense GEMVs and the W7900 roofline, then specify a WMMA-tiled replacement kernel signature. This is the gate document for R3.4 (the actual kernel rewrite).

**Status 2026-05-23:** complete in [`docs/DRAFTER_DENSE_ROOFLINE.md`](DRAFTER_DENSE_ROOFLINE.md) with fresh W7900 microbench artifact [`benchmarks/results/2026-05-23-hipengine-dflash-r3.3-w7900-dense-microbench.json`](../benchmarks/results/2026-05-23-hipengine-dflash-r3.3-w7900-dense-microbench.json). The current dense op mix costs `16.32 ms/cycle` (~83% of the `19.60 ms` decoder-layer wall) at only `3–6%` of W7900 BW. R3.4 WMMA at 30% BW projects dense `16.3 → 3.0 ms` and decoder `19.6 → ~6.3 ms`.

**Win state:**
- A roofline document (in DFLASH.md or `docs/DRAFTER_DENSE_ROOFLINE.md`) with:
  1. Current kernel analysis (per-block work, launch shape, measured per-op time).
  2. W7900 BW-bound and FLOPS-bound floors for the relevant shapes.
  3. Concrete tiling proposal (16×16×16 BF16 WMMA) with LDS strategy.
  4. Expected drafter wall delta at 30% and 50% of BW peak.
- A C-style kernel signature (`hipengine_dflash_dense_bf16_to_bf16_wmma(...)`) ready to be implemented in R3.4.

**Approach — current kernel analysis:**

Source: `hipengine/kernels/hip_gfx1100/speculative/dflash_drafter.hip:dflash_dense_bf16_to_bf16_kernel` (and the `_to_f32` sibling).

- Launch shape: `dim3(out_features, rows, 1)` blocks × `dim3(threads)` threads (typically 128).
- Per block: computes one output element via an `in_features`-length reduction in shared memory.
- For drafter shape (`rows=16`, `in=2048`, `out=2048`): **32K blocks** of 128 threads each, doing a 2048-element reduction (`128 threads × 16 ops + log2(128) syncthreads`).

**Roofline math (W7900):**

- Peak memory bandwidth: `864 GB/s`.
- Peak BF16 WMMA throughput: `~120 TFLOPS`.
- Per-op weight bytes: `out × in × 2 = 2048 × 2048 × 2 = 8 MB`.
- Per-op input bytes (reused across all output tiles via LDS): `rows × in × 2 = 64 KB`.
- Per-op FLOPS: `2 × rows × out × in = 2 × 16 × 2048 × 2048 ≈ 134 MFLOPS`.
- BW-bound floor: `8 MB / 864 GB/s ≈ 9.3 µs`.
- FLOPS-bound floor: `134 MFLOPS / 120 TFLOPS ≈ 1.1 µs`.
- **Roof = max(BW, FLOPS) = 9.3 µs** (memory-bound).

**Measured:** drafter decoder `19.6 ms` synced; ~7 dense ops per layer (Q, K, V, O, gate, up, down) × 8 layers = `56` dense ops per cycle. `19.6 ms / 56 ops ≈ 350 µs/op`. At `9.3 / 350 ≈ 2.7% of BW peak`.

**Proposed WMMA kernel signature:**

```c++
extern "C" int hipengine_dflash_dense_bf16_to_bf16_wmma(
    const uint16_t* x,       // [rows, in]      BF16, row-major
    const uint16_t* weight,  // [out, in]       BF16, row-major (weight[o, i])
    uint16_t* out,           // [rows, out]     BF16, row-major
    int64_t rows,            // = block_size (16); enforce rows % 16 == 0
    int64_t in_features,     // = hidden (2048); enforce in % 16 == 0
    int64_t out_features,    // = hidden / intermediate; enforce out % 16 == 0
    int64_t stream);
```

**Tile layout (gfx1100 native 16×16×16 BF16 WMMA):**

- One WMMA call: `D[16×16] = A[16×16] × B^T[16×16] + C[16×16]` (FP32 accumulator, BF16 inputs).
- For `(rows=16, in=2048, out=2048)`: M=16 (one tile row), K-tile count = `in / 16 = 128`, N-tile count = `out / 16 = 128`.
- One thread block per output tile: `128` blocks total per dense op.
- Each block iterates `128` K-tiles, accumulates in registers, writes `16×16` BF16 to `out`.

**LDS strategy:**

- Input `x[16, 2048]` is reused across all output tiles → each block loads its 16×16 input slab per K-tile iteration (`512 B` per K-tile × 128 K-tiles = `64 KB` per block).
- Weight `weight[out_tile_16, k_tile_16]` is unique per (N-tile, K-tile) → per-block weight read = `16 × 2048 × 2 = 64 KB`. Total weight read across all blocks = `128 × 64 KB = 8 MB` ✓ matches per-op weight size.
- LDS budget per block: `2 × 16 × 16 × 2 B (double-buffered A + B) = 1 KB`. Plenty of headroom on RDNA3 (64 KB LDS / WG).

**Expected delta (conservative — 30% of BW peak):**

- Per-op: `9.3 µs / 0.3 ≈ 31 µs`.
- 56 ops per cycle × 31 µs = `1.7 ms` total dense work.
- Drafter wall: replace ~15–18 ms of dense ops with `~1.7 ms` → decoder layers ~`4–7 ms` total.
- Total drafter wall: `~5–9 ms` (attention/norm ~2–3 ms remain).
- Cycle wall: `verifier (37) + drafter (8) ≈ 45 ms`. **Prompt 3 (visible 4.0) within 8 ms of break-even.**

**Stretch (50% of BW peak):**

- Per-op: `~19 µs` → `1.0 ms` total dense.
- Drafter wall: `~4–6 ms` total.
- Cycle wall: `~41–43 ms`. **Prompt 3 wins; prompt 2 borderline.**

**Acceptance gate (R3.4 implementation phase):**

- KL ≤ 0.05 + top-1 ≥ 90% vs `kernels/cpu_reference/` BF16 GEMM (AGENTS.md correctness gate).
- `rocprofv3 --kernel-trace` shows the new kernel name with duration in the `20–60 µs` range AND occupancy/BW utilization ≥ 30%.
- 9-prompt suite same-session AR exact match on every prompt.
- New JSON artifact under `benchmarks/results/` with full per-cycle phase breakdown.

**Reference kernels to read before writing R3.4 code:**

- `kernels/hip_gfx1100/linear/` — existing WMMA kernels for the target model.
- `kernels/hip_gfx1100/quant/` — `gemv_awq_*_dual_pack8` series; closest analog for batched small-row WMMA.
- `~/amd-gpu-tuning/docs/OPTIMAL.md` — parent workspace's gfx1100 WMMA tile catalog (READ ONLY).

**Grind budget:**
- ~200 LoC markdown for the roofline doc.
- Effort: 0.5 day. **Abort if:** the roofline math says the current kernel is already ≥ 30% of BW peak (it isn't, based on the 3% measurement, but verify the assumptions hold for ALL dense shapes, including non-square `(rows=16, in=2048, out=intermediate)` ones).

### R3.4 design notes — Drafter dense WMMA implementation

(Filled in by coder after R3.3 paper sign-off.)

**Must include:**
- Per AGENTS.md, the kernel registers against `KernelKey("hip_gfx1100", "dflash_dense", "w4_paro", "bf16_to_bf16_wmma")` and `KernelKey("hip_gfx1100", "dflash_dense", "w4_paro", "bf16_to_f32_wmma")`. **No `if backend == ...` branches** in dispatch code.
- Drafter Python wrappers gated via env flag `HIPENGINE_DFLASH_DRAFTER_DENSE={wmma, naive}`, default `naive` until validated.
- Unfused fallback (the existing `dflash_dense_bf16_to_*` kernels) remain registered — AGENTS.md L2 invariant.
- Numerics: BF16 inputs, FP32 accumulator, BF16 output via the standard round-to-even path. Bit-exact to the existing kernel within FP rounding tolerance (KL gate).

**Grind budget:**
- ~250 LoC HIP (kernel + launcher).
- ~100 LoC Python (wrapper + registration).
- ~80 LoC tests (cpu_reference parity + small smoke).
- Effort: 1–2 days. **Abort if:** after 2 iterations the WMMA kernel cannot reach ≥ 25% of BW peak; record the finding as L15 and revisit kernel design (likely needs a tile-shape variant like 32×16×16 for the `intermediate` outputs).

### R3.5 design notes — DDTree wiring on gfx1100

**Gated**: do NOT start unless R3.4 closes the prompt-3 archetype to ≥ 1.0x AR.

**Goal:** Use existing chain drafter `--draft-top-k` output to compile a balanced breadth-first tree and route through `verify_tree_bulk_and_commit` (already landed on gfx1151 — 2026-05-19 DDTree work). Bring up the gfx1100 path.

**Win state:**
- gfx1100 DDTree branching_topk path runs at K=2, depth=4 (B=4 effective).
- Exact-AR holds on the 9-prompt suite.
- Prompt 3 archetype shows ≥ +5% over chain on AR ratio.

**Approach:**
- Chain drafter already supports row-wise top-K via `--draft-top-k=K`; metadata flows through `DraftBatch.candidate_tokens` extra dims.
- The `_build_flat_fan_tree_target_batch` helper in `scripts/dflash_chain_e2e_bench.py` already compiles a depth-1 tree. For depth-D, reuse the breadth-first compiler from the 2026-05-19 work.
- Route through `verify_tree_bulk_and_commit`.

**Grind budget:** ~150 LoC + ~50 tests; 1 day.

### R3.6 design notes — Verifier fusion implementation (post-R3.2)

(Filled in by coder after R3.2 cost-model sign-off.)

**Per-fuse requirements (AGENTS.md):**
- Register under `KernelKey(...)` per backend.
- Carry an unfused fallback.
- Pass `kernels/cpu_reference/` correctness gate.
- Be gated behind an env flag during validation; default-on only after suite cycle_cost reduction ≥ 3%.

**Grind budget:** ~360 LoC per fuse (HIP + Python + tests); 1–2 days per fuse. Aim to ship 1–2 in Round-3.

### R3.7 design notes — Reduced-logits verifier wire-through

**Pure plumbing.** Most of the infrastructure exists from MTP M12.6+.

**Steps:**
1. Confirm `HIPENGINE_VERIFY_GPU_ACCEPT=1` returns exact AR for DFlash chain.
2. `rocprofv3 --kernel-trace --start <prefill_end> --end <verify_end>` shows NO `lm_head_full_vocab` kernel and NO `hipMemcpy` of `[B+1, vocab_size]` shape.
3. If a full-vocab kernel is still present in the verifier window: identify call site, replace with reduced-logits accept-summary path.

**Grind budget:** ~50 LoC; 0.5 day.

#### R3.7 implementation outcome (2026-05-24, gfx1100/W7900)

**Status: rocprof gate PASSED, perf gate FAILED — landed default-off.**

**Step 1 — D2H gate (already passing):** `_verify_gpu_accept_enabled()` is
default-on (`HIPENGINE_VERIFY_GPU_ACCEPT=1` default).  In the
`verify_chain_bulk_and_commit` GPU-accept path, `_read_verify_top1` is NOT
called; only the small `[request_count, ~7]` int32/uint8 accept payload is
read D2H per cycle.  No full-vocab D2H copy occurs.  This holds for the
unfused path (R3.6 baseline) and was confirmed via the per-cycle `d2h`
counters in the bench (`full_logits_readbacks: 0`, `vector_reads: 448`
of small accept summary tensors).

**Step 2/3 — kernel-window gate (initially failing):** rocprof on the
unfused R3.6 baseline showed `w8a16_linear_multi_row_kernel` running 2x in
the 4-token verifier window at **1794.52 µs/call** — this is the
`lm_head_full_vocab` kernel that materializes the
`[rows=B+1, vocab=248320] FP32 = 4.97 MB/cycle` logits buffer in HBM.  The
downstream `argmax_rows_stage1/stage2` kernels add only 13 + 3 µs/cycle.

**Implemented fused path (`HIPENGINE_DFLASH_VERIFY_FUSED_LM_HEAD=on`):**
New kernel `hipengine_w8a16_lm_head_argmax_rows_bf16` (in
`kernels/hip_gfx1100/linear/lm_head.hip`) does the W8A16 GEMV and per-block
argmax in one launch.  Reuses the unchanged `argmax_rows_stage2_i32_kernel`
for the cross-block reduction.  Each stage-1 block of 256 threads
cooperatively processes a `chunk = 1024` vocab tile for one verifier row
with the SAME 256-way LDS-reduce dot pattern as
`w8a16_linear_multi_row_kernel`, so per-row logits are bit-exact vs the
unfused path.  After each row's reduction, thread 0 keeps the running
`(max_value, min_index_on_tie)` and writes one pair to per-block scratch.

**Numerics gate:** parity test `tests/test_w8a16_lm_head_argmax_rows.py`
(4 shapes covering rows∈{1,3,4,5}, hidden∈{1024,2048}, vocab∈{2048,4096,8192,16384})
asserts BIT-EXACT match on both indices and FP32 top-1 values vs the
unfused `w8a16_linear_bf16_f32_multi_row` → `argmax_f32_rows_i32` chain.
All 6 tests pass.

**E2E gate:** 9-prompt B=4 D=32 same-session DFlash chain bench (
`scripts/dflash_chain_e2e_bench.py`, `--full-attn-chain-mode batched`,
`--verifier-mode native_bulk_bplus1`) holds **exact-AR on all 9 prompts**
with `avg_accept_length=1.527` identical to the R3.6 baseline.

**Rocprof gate (1-prompt, 4-token, fused-on):**

| Kernel | unfused (R3.6) | fused-on (R3.7) |
| --- | --- | --- |
| `w8a16_linear_multi_row_kernel` | 2 calls × 1794.5 µs | **NOT PRESENT** |
| `argmax_rows_stage1_i32_kernel` | 2 calls × 12.6 µs | **NOT PRESENT** |
| `argmax_rows_stage2_i32_kernel` | 2 calls × 3.5 µs | 2 calls × 4.2 µs |
| `w8a16_lm_head_argmax_rows_stage1_kernel` (NEW) | — | 2 calls × **2902.2 µs** |

The full-vocab lm-head kernel is gone from the verifier window.  Step 2 of
the R3.7 spec is satisfied.

**Perf gate (9-prompt B=4 D=32, same-session A/B):**

| Metric | fused-OFF | fused-ON | Δ |
| --- | ---: | ---: | ---: |
| drafter ms/cycle | 9.111 | 9.108 | -0.003 (noise) |
| **verifier ms/cycle** | **27.04** | **28.00** | **+0.96 (+3.6%)** |
| cycle ms | 36.34 | 37.30 | +0.96 (+2.6%) |
| spec tok/s | 69.26 | 67.52 | -2.5% |
| AR | 0.4464 | 0.4464 | identical |
| accept length | 1.527 | 1.527 | identical |
| exact-AR rows | 9/9 | 9/9 | preserved |

**The fused kernel is ~1 ms/cycle SLOWER** than the unfused pair on
gfx1100/W7900 despite eliminating the 5 MB/cycle full-vocab logits
writeback and one argmax kernel launch.

**Root cause: GPU occupancy collapse.**
- Unfused `w8a16_linear_multi_row_kernel` grid: `dim3(vocab=248320)` →
  248K small blocks, each doing 5 cooperative dot products (one per token).
  ~2588 active blocks per CU on 96 CUs, excellent occupancy.
- Fused stage-1 grid: `dim3(stage1_blocks=243, rows=5)` → only **1215
  blocks**, each doing 1024 cooperative dot products.  ~12 active blocks
  per CU, GPU under-occupied; long-running blocks can't fill the machine.

The GEMV is bandwidth-bound at the W7900's ~864 GB/s HBM throughput.  The
writeback we eliminated (5 MB / 864 GB/s ≈ 6 µs theoretical) doesn't
compensate for the 200x reduction in block-level parallelism.  This
matches the R3.2 cost model's prediction (`~72 µs/pass theoretical
saving`) and the BeeLlama profile finding that their fused
`set_dflash_verify_logits` design wins on RTX 3090 because Ada is
launch-bound at 248K block dispatches in a way RDNA3 isn't.

**Decision: land default-off.**
- AGENTS.md compliance: env-flag (`HIPENGINE_DFLASH_VERIFY_FUSED_LM_HEAD`,
  default `off`), unfused fallback retained as the registered path,
  cost-model-approved fuse pattern documented.
- Bit-exact correctness gate passes (parity tests + 9-prompt e2e).
- Rocprof gate passes (no `w8a16_linear_multi_row_kernel` in verifier
  window when on).
- Future use: if a future variant (e.g., reduced-logits sampling for
  lm-head pruning, or multi-batch verify with B+1 ≥ 16 changing the
  optimal grid) finds a perf win, the fused kernel + dispatch is already
  in place to be flipped to default-on without further plumbing.

**Files:**
- Kernel: `hipengine/kernels/hip_gfx1100/linear/lm_head.hip`
- Python wrapper + registry: `hipengine/kernels/hip_gfx1100/linear/lm_head.py`
- Runtime dispatch: `hipengine/runtime/qwen35_paro_runner.py::_sample_verify_rows_from_hidden` + `_dflash_verify_fused_lm_head_enabled()`
- Tests: `tests/test_w8a16_lm_head_argmax_rows.py` (6 tests, all pass)
- Bench artifacts: `/tmp/dflash_r37_fused_on_v2.json`, `/tmp/dflash_r37_fused_off_v2.json`
- Rocprof DB: `/tmp/dflash_r37_fused_on_rocprof/epyc/*_results.db`

### R3.8 design notes — BeeLlama v0.2.0 profile read

**Read order:**
1. BeeLlama v0.2.0 release notes (linked in earlier session).
2. `tools/server/server-adaptive-dm.h` (profit controller — cross-references R3.1).
3. `src/models/dflash_draft.cpp` (drafter — cross-references R3.3).
4. `src/llama-context.cpp:~3649` (`cross_bucket()` — already referenced in L4/L12; sanity-check we match it).
5. Their published benchmark methodology (model, prompts, B, settings).

**Output:** `docs/BEELLAMA_PROFILE.md` covering:
- Default `B` and rationale.
- Drafter architecture / param count vs ours.
- Verifier batching strategy and any fused kernels.
- Whether their 4.4x is comparable to our hardware/model/prompt suite or reflects different inputs.
- Maps each finding back to a Round-3 punchlist row.

**Grind budget:** ~120 LoC markdown; 0.5 day.

### Round-3 LoC budget

| Track | Item | LoC type | Estimate | GPU needed? |
| --- | --- | --- | ---: | --- |
| C/G | R3.1 controller + tests | Python | ~180 | minimal (final validation only) |
| B (paper) | R3.2 fusion cost-model writeup | Markdown | ~250 | no |
| A (paper) | R3.3 drafter dense roofline writeup | Markdown | ~200 | no |
| A (code) | R3.4 WMMA drafter dense kernel | HIP + Python + tests | ~430 | yes |
| C (code) | R3.5 DDTree wiring | Python + tests | ~200 | yes (gated) |
| B (code) | R3.6 1–2 verifier fuses | HIP + Python + tests | ~360 – ~720 | yes |
| B (code) | R3.7 reduced-logits wire-through | Python | ~50 | minimal |
| ref | R3.8 BeeLlama profile notes | Markdown | ~120 | no |
| **Total** |  |  | **~1,790–2,150** |  |

Phase-1 (paper-only, blocks 0 GPU): **R3.1 + R3.2 + R3.3 + R3.8 ≈ 750 LoC** (mostly markdown + Python).
Phase-2 (GPU): **R3.4 + R3.5 + R3.6 + R3.7 ≈ 1,040–1,400 LoC** (mostly HIP + Python + tests).

### Round-3 acceptance principles (carry from Round-2)

- Exact greedy AR equality preserved on every retained prompt (same-session control).
- Each new kernel: correctness gate (KL ≤ 0.05 + top-1 ≥ 90% vs `kernels/cpu_reference/`).
- Each fuse: bit-exact vs unfused chain via the cpu_reference oracle; unfused fallback registered.
- Each adaptive-controller transition: logged in artifact.
- Each paper claim: backed by measured numbers from `benchmarks/results/`, never hand-waved.
- Promotion rule still applies: no DFlash speed row is accepted as a performance claim until the economics artifact shows `avg_visible_tokens_per_cycle / cycle_cost_ar_tokens > 1.0`.

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
| L11 | **Real DFlash drafter quality is necessary but not sufficient**: restored z-lab DFlash lifts W7900 chain from Round-1 `0.289x` to `0.413x` AR, but acceptance (`1.42` accepted draft tokens/cycle, `2.46` visible tokens/cycle) is still below the ~`5.96` AR-token-equivalent cycle cost. Both sides must move: drafter caching/quality to raise visible tokens, and verifier/fusion work to lower cycle cost. | R2.2 W7900 z-lab drafter rerun (2026-05-23) | R2.3/R2.5 drafter work; R2.4/R2.8 verifier work; DDTree promotion timing |
| L12 | **Bucketed graph capture is a kernel-shape problem, not a cache-key tweak**: exact-context graph capture misses because `context_tokens` changes every cycle, but simply bucketing the key would be wrong today. The drafter concat/attention kernels place query K/V immediately after `context_tokens` rows and loop over `total_kv=context+block`; replaying a captured graph at a different live context would attend to the wrong/padded rows. | R2.2 sync-phase triage (2026-05-23) | R2.3 design; any context-dependent HIP graph capture |
| L13 | **Drafter on W7900 is GPU-kernel-bound, not host-launch-bound**: bucketed HIP graph capture (R2.3) saves only `~3.6 µs / launch × ~124 launches/cycle = ~0.45 ms` (~2% of cycle). Cycle wall is dominated by 8 decoder layers' kernel time (`~19.6 ms` synced). Same finding as M14.dispatch.0/1 for MTP. Future drafter wins must move kernel work, not launches. | R2.3 W7900 4-prompt B=4 D=16 (2026-05-23) | Any drafter launch-overhead optimization; HIP graph capture re-eval; ROCm → CUDA portability claims |
| L14 | **Drafter dense GEMVs run at ~3% of W7900 bandwidth peak**: the naive per-output-element `dflash_dense_bf16_to_bf16_kernel` runs at ~`306–350 µs/op` on `(16 rows × 2048 in × 2048 out)` vs a BW-bound floor of `~9.3 µs`. A 16×16×16 BF16 WMMA-tiled kernel with LDS-staged input should give 5–10× and is the single biggest remaining drafter lever on W7900. | R3.3 roofline analysis (2026-05-23) | All small-batch BF16 GEMV kernels in hipEngine; drafter rewrites; any non-WMMA dense kernel |
