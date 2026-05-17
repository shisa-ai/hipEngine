# hipEngine Native Bulk Prefill Plan

> Status: final implementation spec, corrected 2026-05-15. This document is
> the authoritative prefill punchlist for Qwen3.5-35B-A3B-PARO. `docs/PLAN.md`
> remains the architecture source of truth; update both files if the architecture
> changes.

## TL;DR

We are **not** landing throwaway intermediate prefill paths. hipEngine already
has correct reference implementations: the original `nano-vllm-amd` native bulk
engine and hipEngine's validated serial resident path. Use those as oracles and
build the complete native path directly.

Final target:

- One `Qwen35ParoResidentSession.prefill_native(...)` call embeds the whole
  prompt (or a configured prompt chunk) into `[T, hidden]` and runs every layer
  in bulk.
- Linear-attention layers use native conv/GDN prefill and update their prompt
  tail state once per layer.
- Full-attention layers use native batched Q/K/V projection, batched head
  RMSNorm+RoPE, batched KV append, and native causal GQA prefill attention.
- MoE uses the grouped/compact parent route over prompt rows, not the existing
  c1 selected-row MoE path as the retained implementation.
- Generation and benchmark scripts call `prefill_native(...)`; serial/token-loop
  paths remain only for reproduction and correctness comparisons.
- Compact c>N prefill packs multiple requests into a prompt slab and executes
  native kernels over that slab. Per-request invocation is an oracle/fallback for
  debugging, not a retained c>N throughput path.

Explicitly skipped as retained implementations:

| Skipped path | Allowed use |
| --- | --- |
| `linear_prefix_token_major_suffix` | Existing artifact reproduction only. |
| Layer-major full-attention row loop through c=1 decode kernels | Stage oracle/probe only; do not wire into generation or retain perf rows. |
| c1-style selected-row MoE as the prefill path | Oracle for grouped MoE and bring-up probes only. |
| Per-request c>N packed fallback | Debug/equality oracle only; no c>N throughput claim. |

Implementation landing policy: native pieces may land independently in code
behind `require_full_native=False` or test-only/probe entrypoints, using the
oracle paths above to fill missing pieces during bring-up. The first retained
prefill performance artifact is captured only after all native pieces are
present, `PrefillConfig.require_full_native=True` is the default, and the c1
selected-row MoE path no longer appears in the production prefill code path.
In-progress measurements live in `WORKLOG.md`; `benchmarks/README.md` keeps the
current 117.24 tok/s c=1 row as the retained performance baseline until native
prefill beats it. The first full single-request native correctness artifact is
accepted, but it is diagnostic-only for throughput.

Scope note: this plan targets `z-lab/Qwen3.5-35B-A3B-PARO` MoE hybrid. Dense
`Qwen3.5-0.8B-PARO` needs tied-lm-head and dense PARO MLP support first; that is
a separate loader/runtime task.

## Terms and shape conventions

| Term | Meaning |
| --- | --- |
| `T` | Prompt rows for one request in one prefill call or internal chunk. |
| `T_total` | Rows in a compact prompt slab packed across multiple requests. |
| `C` | Active decode requests; not the same as prompt rows. |
| Bulk prefill | Layer input/output buffers are `[rows, hidden]`; kernels operate on prompt rows. |
| Grouped/compact MoE | Parent-style route that scatters routed rows by expert and runs grouped bulk kernels. |
| Append-then-attend | First native full-attn design: append all prompt K/V rows to paged cache, then causal attention reads prefix+prompt K/V entirely from cache. |

KV span convention for this repo:

- KV **append** spans use `live_counts[row] = absolute_position` (0-based write
  position), matching the preserved parent writer ABI.
- KV **attention/decode** spans use `live_counts[row] = context_length` (1-based
  visible length).
- For prompt row `r` with `start_position`, append position is
  `start_position + r`; attention context length is `start_position + r + 1`.

## Evidence: current gap and references

Parent native engine retained rows (Qwen3.5-35B-A3B-PARO, W7900, BF16/FP16
activations, W4 PARO weights):

| Shape | Prefill tok/s | Decode tok/s | Notes (`~/amd-gpu-tuning/docs/PARO.md`) |
| --- | ---: | ---: | --- |
| 512 / 128 | 554.21 | 64.71 | `bench_paro_native_engine.py --prefill-mode bulk`, lm_head dense GEMV |
| 4096 / 128 | 2140.71 | 60.32 | bulk, lm_head dense GEMV, 24GB path |
| 4096 / 4096 | 2155.60 | 56.79 | bulk, lm_head dense GEMV, 24GB path |
| 512 / 32 | 2682.66 | 116.26 | parent fixture row recorded in `fixtures/qwen35_paro/parent_512_32_seed1234.json` |

hipEngine current rows on the same 35B fixture:

| Shape | Prefill tok/s | Decode tok/s | Artifact / notes |
| --- | ---: | ---: | --- |
| 512 / 32 | 117.24 | 101.68 | `benchmarks/results/2026-05-15-hipengine-qwen35-c1-parent-fixture-accepted.json`; prompt runs as sequential resident steps |
| 512 / 32 | 45.72 fixture / 46.96 repeated-token diagnostic | 101.61 | `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefill-full-single-request-accepted.json`; `single_request_native_full`, correctness accepted (`max_kl=0.0168`, top-1 100%), no perf row promoted |
| c=8 8/1 | 115.08 | 108.89 | `scheduler_serial_slot_bridge` diagnostic, not native compact batching |

Correctness/blocker artifacts already retained:

- `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefix-scratch-restore-sweep.json`
  — native linear prefix accepted through layers 0..2.
- `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefix-serial-suffix-full40-accepted.json`
  — native linear prefix plus token-major serial suffix matches serial resident
  outputs; no throughput claim.
- `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefill-full-single-request-accepted.json`
  — final single-request native prefill correctness gate accepted on the 512/32
  parent fixture (`max_kl=0.0168`, top-1 100%, generated IDs match serial and
  parent); diagnostic timing remains slower than serial and parent baselines.
- Earlier blocked boundary artifacts (`native-prefill-full-attn-boundary-blocked`,
  `native-prefill-plan-blocked`) are superseded by the accepted full native
  orchestration artifact above.

Reference files:

- `~/amd-gpu-tuning/scripts/bench_paro_native_engine.py`
- `~/amd-gpu-tuning/nano-vllm-amd/nanovllm/native/qwen35/paroquant.py`
- `~/amd-gpu-tuning/docs/PARO.md`
- `~/amd-gpu-tuning/docs/OPTIMAL.md`

### 2026-05-16 AOTriton V3 parent-gap audit

Latest single-request diagnostic rows use hipEngine's AOTriton V3
compact-varlen GQA path (`--attn-aotriton-min-tokens 512`) and real
hipEngine-owned memory accounting.  They supersede the older bring-up rows above
for the current parent-parity gap discussion.  AOTriton is now a mandatory,
vendored baseline runtime dependency for the gfx1100 Qwen3.5/PARO path, and the
`LLM.generate()`/benchmark defaults select the threshold-512 policy plus decode
HIP graph replay.  `attn_aotriton_min_tokens=0` is retained only as an explicit
native-attention diagnostic override.

| Workload | Parent prefill tok/s | hipEngine AOTriton V3 prefill tok/s | Prefill delta | Parent decode tok/s | hipEngine decode tok/s | Decode delta | Peak allocated delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 / 128 | 2696.4 | 2333.4 | -13.5% | 116.05 | 101.30 | -12.7% | -0.17 GiB |
| 4096 / 128 | 2741.5 | 2379.7 | -13.2% | 113.05 | 102.41 | -9.4% | -0.84 GiB |

Source artifacts:

- Parent: `benchmarks/results/2026-05-13-source-lineage-qwen35-paro-optimal-512-128.json`
  and `benchmarks/results/2026-05-13-source-lineage-qwen35-paro-optimal-4k-128.json`
  (`nano-vllm-amd@59195ed`, OPTIMAL flags, decode graph replay).
- hipEngine original AOTriton audit: `benchmarks/results/2026-05-16-hipengine-qwen35-aotriton-v3-memory-diagnostic.json`
  (`a00c244`, no decode graph replay, AOTriton opt-in threshold 512).
- hipEngine decode-graph follow-up: `benchmarks/results/2026-05-16-hipengine-qwen35-decode-graph-replay-diagnostic.json`
  (same opt-in threshold, one-step HIP graph replay, graph-vs-eager fixture gate).
- hipEngine threshold sweep: `benchmarks/results/2026-05-16-hipengine-qwen35-aotriton-threshold-sweep-diagnostic.json`
  (32/64/128/256/512/1024/4096 prompt sweep, threshold-512 fixture and graph gates).

The important shape signal is that the residual prefill gap is almost the same
at 512 and 4K after AOTriton V3: the old 4K native-attention cliff is closed, so
the remaining gap is unlikely to be the quadratic attention core.  The decode
follow-up moved hipEngine decode to `109.34` tok/s at 512 and `110.30` tok/s at
4K, narrowing the parent decode gap to `-5.8%` and `-2.4%`; it does not change
the prefill diagnosis.  The threshold sweep moved the AOTriton policy from
"pending" to "recommended opt-in threshold 512".  The remaining prefill gap
points at per-layer non-attention bulk work and launch/cast glue.

#### AOTriton threshold sweep (2026-05-16)

Single-run diagnostic sweep on W7900/gfx1100, repeated token id `9707`,
`max_layers=40`, cached HIP builds, real hipEngine memory accounting.  Short
prompt rows use `--decode-tokens 0 --warmup-decode-tokens 0` to isolate prefill;
512/128 and 4K/128 rows use one-step decode graph replay.  Correctness gates:
`qwen35_native_prefill_fixture_gate.py` with `--attn-aotriton-min-tokens 512`
passed (`max_kl=0.039568870612619614`, top-1 agreement `1.0`, fixture IDs
match), and `qwen35_decode_graph_fixture_gate.py` passed (`final_kl=0.0`, graph
IDs match eager and fixture).

| Prompt tokens | Native attention prefill tok/s | Forced AOTriton prefill tok/s | AOTriton delta | Native peak GiB | AOTriton peak GiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 605.657 | 504.397 | -16.7% | 18.331 | 18.334 |
| 64 | 994.069 | 829.395 | -16.6% | 18.345 | 18.350 |
| 128 | 1464.792 | 1304.824 | -10.9% | 18.371 | 18.381 |
| 256 | 1892.317 | 1826.457 | -3.5% | 18.429 | 18.449 |
| 512 | 2146.479 | 2284.584 | +6.4% | 18.541 | 18.580 |
| 1024 | 1815.743 | 2498.659 | +37.6% | 18.763 | 18.842 |
| 4096 | 662.419 | 2356.051 | +255.7% | 20.099 | 20.414 |

The measured crossover is between 256 and 512 prompt tokens.  Among tested
threshold policies (`0`, `32`, `64`, `128`, `256`, `512`), threshold `512` is the
only one that avoids the short-prompt regressions while still selecting the fast
path at the first prompt length where AOTriton wins.  Because AOTriton is now
vendored through Git LFS and treated as a baseline runtime dependency,
`PrefillConfig.attn_aotriton_min_tokens` defaults to `512`; pass `0` only for
native-attention diagnostics.

| Workload | Native prefill tok/s | AOTriton threshold-512 prefill tok/s | Prefill delta | Native decode tok/s | AOTriton decode tok/s | Peak GiB delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 / 128 | 2125.642 | 2270.750 | +6.8% | 109.225 | 109.123 | +0.039 |
| 4096 / 128 | 662.873 | 2345.670 | +253.9% | 109.980 | 110.091 | +0.315 |

#### Long-shape checkpoint (2026-05-16)

Follow-up diagnostic checkpoint with the same baseline AOTriton policy
(`--attn-aotriton-min-tokens 512`) and one-step decode graph replay:
`benchmarks/results/2026-05-16-hipengine-qwen35-long-checkpoint-diagnostic.json`.
No new long-context oracle fixture was run; this row inherits the threshold-512
fixture gates above and is not a promoted performance claim.

| Workload | hipEngine prefill tok/s | hipEngine decode tok/s | hipEngine tracked peak GiB | Parent/source comparison | Notes |
| --- | ---: | ---: | ---: | --- | --- |
| 4K / 4K | 2379.818 | 108.930 | 20.529 | Local parent rerun: 2728.305 prefill / 104.963 decode / 21.719 GiB | hipEngine prefill -12.8% vs parent, decode +3.8%; parent 4K/4K replay row has known graph/eager divergence at token 581, so it is comparison context only. |
| 32K / 128 | 1718.308 | 93.933 | 35.100 | `~/amd-gpu-tuning/docs/OPTIMAL.md`: 1880 prefill / 98.8 decode / 21.37 GiB | Pre-chunk checkpoint: hipEngine -8.6% prefill, -4.9% decode, but much higher tracked peak because it lacked the parent's long-context chunking. |
| 128K / 128 | blocked: OOM | — | — | `~/amd-gpu-tuning/docs/OPTIMAL.md`: 914 prefill / 62.6 decode / 27.42 GiB | Pre-chunk attempt reserved unchunked linear-attention scratch and failed at `linear_attn.out_rot`; replace with chunked retest tables. |

Parent long-context rows use chunking overrides (`NANOVLLM_PARO_PREFILL_LINEAR_CHUNK_SIZE`,
`NANOVLLM_PARO_MOE_CHUNK_SIZE`, and full-attention query/post/RoPE chunks).  hipEngine's
`PrefillConfig` now exposes and wires matching knobs in the single-request path:
linear layers run as contiguous chunks, full-attention chunks append KV then run
bottom-right-aligned causal AOTriton over the cached prefix.

Retained chunking checkpoint:
`benchmarks/results/2026-05-16-hipengine-qwen35-prefill-chunking-diagnostic.json`.
Companion quick-table artifact:
`benchmarks/results/2026-05-16-hipengine-qwen35-comparison-tables-diagnostic.json`.
Run `python3 scripts/qwen35_compare_tables.py {nano-vllm-amd,llama.cpp-hip,llama.cpp-vulkan,all}`
to print separate prefill/decode/memory comparison tables.  All hipEngine rows use
`--attn-aotriton-min-tokens 512 --graph-replay-decode`; the chunked policy mirrors
parent long-context knobs: linear/MoE/post/RoPE chunks `1024`, full-attention query
chunk `4096`.

| Workload | Unchunked prefill tok/s | Chunked prefill tok/s | Prefill delta | Unchunked decode tok/s | Chunked decode tok/s | Tracked peak GiB | Parent OPTIMAL | Chunked/current vs parent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 512 / 128 | not rerun (chunk sizes exceed prompt) | 2216.487 | no-op policy | — | 109.105 | 18.581 | 2557.0 prefill / 115.7 decode / 18.86 GiB | prefill -13.3%, decode -5.7%, peak -0.28 GiB |
| 4K / 128 | 2370.229 | 2504.959 | +5.7% | 110.168 | 110.117 | 20.415 → 19.875 | 2703.0 prefill / 112.0 decode / 21.64 GiB | prefill -7.3%, decode -1.7%, peak -1.77 GiB |
| 32K / 128 | 1731.976 | 1886.344 | +8.9% | 93.867 | 93.923 | 35.100 → 20.688 | 1880.0 prefill / 98.8 decode / 21.37 GiB | prefill +0.3%, decode -4.9%, peak -0.68 GiB |
| 128K / 128 | blocked: OOM | 1002.409 | unblocked | — | 61.051 | OOM → 23.656 | 914.0 prefill / 62.6 decode / 27.42 GiB | prefill +9.7%, decode -2.5%, peak -3.76 GiB |

Takeaway: chunking fixes the 128K scratch OOM and removes the 32K memory cliff.
Long-context prefill is now at/above the parent long-context rows in this
resident-runner diagnostic, while decode remains slightly behind parent.  The
512/128 no-op-chunk row remains a short-context prefill gap and is included so
the quick comparison script covers the same context set for every baseline.

#### Parent vs hipEngine prefill call structure

| Stage | nano-vllm-amd OPTIMAL parent | hipEngine AOTriton V3 path | Audit finding |
| --- | --- | --- | --- |
| Layer mix | 40 layers: 30 linear-attention, 10 full-attention. | Same model layer sequence. | Layer coverage is not the gap. |
| Full-attention core | `ParoQuantFullAttentionLayer.prefill_native(...)` projects Q/K/V, applies head norm/RoPE, appends KV, then calls `torch.nn.functional.scaled_dot_product_attention(..., enable_gqa=True)` on BF16 Q/K/V. | `run_full_attention_moe_prefill_layer_fp16(...)` does the same prelude, then calls `v3::flash::attn_fwd` compact-varlen GQA with BF16 Q/K/V and a separate BF16-output gate post-pass. | Attention launch fanout is fixed; remaining attention overhead is now casts/post-pass around AOTriton, not the core SDPA algorithm. |
| Linear-attention A/B dense projections | Multi-row prefill falls through `ParoQuantDenseLinear.forward(...)` to `F.linear(...)` (`native_aux_dense_linear_calls` appears in parent ledgers). | `project_linear_attention_ab_fp16(...)` launches two row-wise `dense_gemv_out_fp16(...)` kernels for `tokens > 1`. | Likely prefill gap: parent uses rocBLAS/Tensile-style bulk GEMM; hipEngine uses scalar row/column GEMV kernels for a bulk matrix problem. |
| Shared expert during prefill | `ParoQuantSharedExpert.forward(...)` uses W8A16 only for `x.shape[0] == 1`; multi-row prefill uses dense `F.linear(...)` gate/up and down (`native_shared_expert_dense_calls` in parent ledgers). | `shared_expert_gate_up_silu_fp16(...)` and `shared_expert_down_combine_residual_fp16(...)` use custom W8A16 row/column kernels for all `tokens > 1`. | Likely prefill gap and also explains hipEngine's slightly lower peak memory: hipEngine quantizes this branch but does not yet have a tiled bulk W8A16/dense-GEMM implementation. |
| Grouped routed MoE | Compact stacked MoE, compact WMMA tile map, dual gate/up WMMA, fused SiLU+down-rotate, single down WMMA, weighted-lane accumulation. | Same compact WMMA route is ported and wired in `run_moe_grouped_compact_fp16(...)`. | Probably not the first gap unless a matched profile disproves parity. |
| Full-attention Q/K/V/O W4 projections | Parent multi-row W4 projections use pack8 replacement and bulk AWQ prefill paths once row count exceeds GEMV thresholds. | hipEngine uses fused W4 prefill kernels for dual Q/K, QKV/Z, and single V/O/out projections. | Needs profiler verification, but current source structure does not show an obvious missing parent optimization here. |
| Decode | Parent retained rows use `--decode-use-step-graph-replay`. | hipEngine now has one-step HIP graph replay with device token/position state and fixture validation (`qwen35_decode_graph_fixture_gate.py`). | Explains most prior decode delta; remaining decode gap is small at 4K and separate from the prefill gap. |

#### Prioritized prefill gap table

| Priority | Gap / hypothesis | Evidence | Why it can explain 512 and 4K | Next action |
| --- | --- | --- | --- | --- |
| P0 | Replace bulk dense GEMV-style kernels with real bulk GEMM/WMMA for linear-attention A/B and shared-expert prefill. | Parent source uses `F.linear(...)` for multi-row `ParoQuantDenseLinear` and multi-row `ParoQuantSharedExpert`; parent ledgers show `native_aux_dense_linear_calls=280` and `native_shared_expert_dense_calls=80`. hipEngine source uses `dense_gemv_out_fp16(...)` for A/B and scalar W8A16 shared kernels. | These costs scale roughly linearly with prompt rows and occur in every layer/MoE layer, matching the near-constant -13% residual gap at both 512 and 4K. | Capture a matched 512/128 ROCTX+`rocprofv3` profile first; if confirmed, add a torch-free rocBLAS/hipBLAS or tiled WMMA bulk dense path, starting with shared expert gate/up+down and linear A/B. |
| P1 | Avoid or fuse AOTriton dtype/post-pass glue. | **Landed as diagnostics:** single-request AOTriton prefill writes BF16 Q directly from head-norm/RoPE, reuses the already-appended BF16 paged KV cache for K/V, fuses BF16 attention × FP16 gate into PARO rotate1, and aliases the old gated scratch as BF16 AOTriton output. | These changes remove Q/K/V cast launches plus the separate gate launch and reduce 4K tracked peak memory by ~0.39 GiB cumulatively, but single-run throughput stayed neutral/slightly negative; this is no longer a leading explanation for the -13% residual gap. | Keep the cast-glue and gate-rotate artifacts as evidence; prioritize the P0 bulk dense/shared-expert gap unless matched profiler attribution says attention post-pass still dominates. |
| P2 | Keep AOTriton threshold policy evidence-backed. | **Sweep landed as diagnostic:** forced AOTriton is slower at 32/64/128/256 and faster at 512/1024/4096; threshold 512 is the first tested policy that avoids short-prompt regressions while keeping the fixed 4K path. | Not a gap for current comparison, but it determines the default full-attention path without hurting short prompts. | Keep code default `512` now that AOTriton is vendored through Git LFS; use `0` only for native-attention diagnostics and rerun the sweep when AOTriton or the prelude changes. |
| P2 | Decode graph replay parity. | **Landed as diagnostic:** one-step HIP graph replay records generated IDs on device for the fixture gate and reaches 109.34 tok/s (512/128) / 110.30 tok/s (4K/128), reducing the parent decode gap to -5.8% / -2.4%. | Decode delta is separate from prefill, but it affects end-to-end comparison tables. | Keep the graph gate in the benchmark protocol; remaining decode work should wait until prefill default/threshold and P0 bulk kernels are settled. |
| P3 | Matched profiler attribution. | This audit is source/ledger based; no matched hipEngine-vs-parent prefill kernel-time table exists yet. | Prevents tuning the wrong family if dense/shared kernels are not the top residual. | Retain compact 512/128 and 4K/128 profiler summaries with ROCTX ranges before landing invasive kernel work. |

#### Additional low-risk prefill fusion audit (2026-05-16)

Source audit only; no new GPU measurement in this pass.  The goal was to find
small launch/materialization cleanups left after the AOTriton Q/K/V cast and
gate+rotate work, not to replace the P0 bulk dense/shared-expert gap.

| Rank | Area | Current hipEngine prefill sequence | Parent/source comparison | Candidate | Risk / why not already done |
| --- | --- | --- | --- | --- | --- |
| 1 | Linear-attention output tail | `qwen35_gdn_prefill_rmsnorm_gate_fp16(...) -> paro_rotate1_fp16(...) -> awq_fusedw4_prefill_strided_fp16(...)` in `run_linear_attention_prefill_*_fp16`. | Parent computes `hidden_outputs = rmsnorm(recurrent) * silu(z)` then calls PARO `out_proj`; hipEngine already owns both lowp GDN gate and PARO rotate kernels. | Add a FP16 `gdn_prefill_rmsnorm_gate_rotate` kernel that computes per-value-head RMSNorm+SiLU gate into LDS, applies the PARO rotate1 group, and writes `out_rot` directly. Removes one launch and the `recurrent_bf16` materialization on all 30 linear-attention layers. | Low/medium. Safe when `head_v_dim == group_size` (Qwen3.5/PARO uses the natural 128-wide groups); keep the existing two-kernel path as fallback for other shapes. |
| 2 | MoE shared gate | `route_moe_topk_shared_fp16(...)` writes the shared-gate logit, then grouped prefill launches `w8a16_shared_gate_sigmoid_fp32(...)`, then `w8a16_shared_down_combine_residual_fp16(...)` consumes the sigmoid. | Parent's c=1 fused shared-gate combine computes sigmoid inside combine; hipEngine precomputes it once to avoid recomputing `expf` per hidden tile. | Add a prefill-only router variant (or select-kernel flag) that overwrites the shared-gate column with `sigmoid(logit)` after top-k selection. Then grouped prefill can skip `w8a16_shared_gate_sigmoid_fp32(...)` without recomputing sigmoid inside shared down. | Low. Must not change the c=1 route because `weighted_sum_shared_gate_combine_residual_*` expects raw shared-gate logits and applies sigmoid itself. |
| 3 | Full-attention Q/gate + K prelude | `qwen35_split_qgate_fp16(...)`, `fp16_to_f32(key)`, then `qwen35_head_rmsnorm_partial_rotary_positions_*` for Q/K head norm + RoPE. | Parent's torch graph views/chunks Q and normalizes/rotates Q/K; hipEngine has explicit split/cast launch glue because the raw-pointer ABI needs materialized FP32 Q/K inputs. | Add an AOTriton-first fused prelude that reads FP16 `q_proj` (`Q|gate`) and FP16 K projection directly, writes gate FP16, BF16 Q, and FP32 K (or BF16 K if appending directly) while doing head RMSNorm + vector-position RoPE. Removes split and key-cast launches plus `query_raw`/`key_raw` scratch for the AOTriton path. | Medium. More pointer/stride plumbing than math risk; keep native non-AOTriton path unchanged until the AOTriton fixture is green. |
| 4 | Packed linear-attention segment conv | Compact c>N path casts `qkv` FP16 to `qkv_f32` before `qwen35_linear_attn_conv_prefill_segments_f32(...)`; single-request prefill already has the lowp-input `qwen35_linear_attn_conv_prefill_fp16(...)`. | Parent segment path is newer than the original c=1 prefill and does not expose this exact torch-free split. | Add a templated/FP16-input segment conv wrapper to remove the `fp16_to_f32(...)` cast and `qkv_f32` scratch in packed prefill. | Low, but affects compact c>N more than the current c=1 benchmark, so schedule after c=1 launch cleanups. |
| 5 | MoE metadata fanout | `_prepare_grouped_moe_prefill_metadata(...)` uses memset/count/prefix/tile-map/memset/scatter launches before expert WMMA. | Parent uses torch-side grouping plus native compact WMMA; hipEngine's explicit metadata kernels are correct but launch-heavy. | Combine `group_prefix` + `wmma_tile_map` and initialize `scatter_offsets`/`tile_expert` in the same small metadata kernel. | Low math risk but small payoff; do after profiler confirms metadata is visible. |

Defer for now: fusing input RMSNorm with PARO input rotation (requires a
row-wide reduction before group-local rotations), fusing rotate into generic W4
WMMA projection kernels (larger AWQ kernel rewrite), and folding
`weighted_lanes_sum` into shared down combine (sorted lanes make the selected
sum non-local unless the down kernel grows atomics or a different output
layout).  These are not "easy" launch fusions.

Recommended order if continuing launch-cleanup work: (1) GDN RMSNorm/gate +
linear-out rotate, (2) prefill-only router shared-gate sigmoid, (3) AOTriton
Q/gate+K prelude fusion.  Re-run the 512/32 fixture gate after each and keep
512/128 + 4K/128 rows diagnostic-only unless repeated runs show a real
throughput improvement.

## Current hipEngine inventory

`docs/KERNELS.md` is authoritative for exact landed kernels and gates. If this
section disagrees with `docs/KERNELS.md`, `docs/KERNELS.md` wins and the follow-up
change should fix both files.

Landed and usable now:

| Area | Current usable pieces |
| --- | --- |
| Runtime state | `embedding_lookup_batch_{bf16,fp16}_i64`, mapped variants, `set_i64_vector`, scalar/vector decode position helpers. |
| Linear-attn prefill | `qwen35_linear_attn_conv_prefill_f32`, segment-aware `qwen35_linear_attn_conv_prefill_segments_f32`, `qwen35_linear_attn_prefill_prepare_f32_fp16`, `qwen35_gdn_prefill_recurrent_k2_f32`, segment-aware `qwen35_gdn_prefill_recurrent_segments_k2_f32`, `qwen35_gdn_prefill_rmsnorm_gate_fp16`. |
| Linear layer orchestrator | `run_linear_attention_moe_c1_layer_fp16(tokens=T)` already selects prefill conv/GDN when `tokens > 1`; final path must replace its c1 MoE tail with grouped MoE. |
| Full-attention decode/prelude | Existing c=1 Q/K/V projection, vector-position RoPE prefill prelude, KV append, native append-then-attend causal GQA prefill kernel including varlen/block-diagonal `cu_seqlens` ABI, context/GQA decode, gate, output projection. Decode kernels remain useful as oracle only for prefill attention. |
| KV append | `qwen35_write_paged_kv_mixed_value_fp16_prompt_spans(...)` appends all prompt rows into one request cache; row-major `*_batch_spans(...)` remains for c>N-shaped caches. Both consume per-row append positions in `spans.live_counts`. |
| KV metadata | `KVLiveSpans` already carries `request_ids`, `row_positions`, and `span_role`; compact prefill needs wiring/population, not a span redesign. |
| Graph primitives | `hipengine.core.hip.HipRuntime` exposes HIP graph capture/instantiate/launch; c=1 decode graph replay exists with max-replay split sizing and optional generated-token recording for gates. |

Missing for the final path:

| Area | Required final work |
| --- | --- |
| Public API wiring | **Landed:** `prefill_native(...)` is the default single-request path; compact c>N uses `prefill_native_packed(slab)` and generated-equality gates now pass for c=2/4/8 prompt8. |
| Full-attn retained orchestration | **Landed for c=1:** batched Q/K/V + vector RoPE + prompt KV append + native causal prefill attention are wired and fixture-gated. |
| Grouped/compact MoE | **Landed for c=1:** grouped scatter/gather and compact AWQ WMMA expert kernels are wired into native prefill. |
| Compact c>N slab | **Correctness landed:** `CompactPromptSlab`, `bucketize_by_block_count`, physical slot metadata, segment-aware linear-attn conv/GDN, varlen/block-diagonal full-attn via `cu_seqlens`, grouped compact MoE, and final-row commit are wired through `prefill_native_packed(slab)`; c-aware decode graph replay and retained throughput remain future work. |
| Prefill config/tuning | Add typed `PrefillConfig`; no hot-path env lookups. |

## Final API and config contract

Add this public session API:

```python
def prefill_native(
    self,
    token_ids: Sequence[int],
    *,
    sample: bool = True,
    require_full_native: bool | None = None,
) -> Qwen35ParoAutoregressiveStepResult | None:
    """Run full native prefill from position 0 through len(token_ids)-1.

    If sample=True, return next-token logits/argmax from the final prompt row.
    If require_full_native is None, use PrefillConfig.require_full_native; an
    explicit per-call value overrides the config default. The final default is
    full-native required: unsupported configs raise NotImplementedError rather
    than silently using token-loop fallbacks.
    """
```

Add a typed config object, e.g. `hipengine/runtime/prefill.py`:

```python
@dataclass(frozen=True)
class PrefillConfig:
    linear_chunk_size: int = 0
    full_attn_query_chunk_size: int = 0
    full_attn_post_chunk_size: int = 0
    full_attn_rope_chunk_size: int = 0
    moe_chunk_size: int = 0
    moe_grouped_device_gather: bool = True
    moe_stacked_compact: bool = True
    require_full_native: bool = True
```

Semantics:

- Public `prefill_native(token_ids, ...)` starts at position 0 on a fresh
  session. Non-zero external `start_position` is not a public API.
- Final native prefill requires `T >= config.linear_conv_kernel_dim` (typically
  4 for Qwen3.5/PARO) because the linear-attention conv prefill kernels require
  enough rows. Shorter prompts raise `ValueError`; no production serial fallback
  is added for this corner unless a future dedicated short-prompt native kernel
  lands.
- Internal chunking may process the prompt as multiple contiguous chunks, but it
  must preserve exactly the same final conv/recurrent state and KV cache as a
  single full-prompt call.
- If `sample=False`, the method performs all state/KV updates and returns
  `None`.
- After prefill, copy the final row into `self.hidden`, restore decode scratch
  sizes, and set `position_buf = T - 1`, `context_buf = T` so the next decode
  step appends at position `T`.
- Keep `prefill_linear_tokens_native(...)` only as a compatibility alias for
  retained artifact reproduction; update `scripts/qwen35_paro_bench.py` and new
  call sites to use `prefill_native(...)`.
- `hipengine/generation/qwen35_paro.py` should call `session.prefill_native(...)`
  directly; no generation-time serial prompt loop except an explicitly requested
  diagnostic mode.
- Prefill work does not change decode policy: multi-token decode scheduling and
  any future `Qwen35ParoOneTokenGenerator` rename/behavior changes are out of
  scope here. This plan only replaces prompt setup.

Path labels for artifacts:

| Label | Meaning | Retained perf claim? |
| --- | --- | --- |
| `serial_step_loop` | Existing token-by-token resident prefill. | Baseline only. |
| `native_prefill_full_single_request` | Final single-request native prefill: native full-attn + grouped MoE. | Yes. |
| `native_prefill_compact_cN` | Final multi-request compact slab path. | Yes. |
| `oracle_row_loop_full_attention` | c=1 row loop used by probes/tests. | No. |
| `oracle_c1_selected_moe_rows` | c1 selected-row MoE used as grouped-MoE oracle. | No. |
| `oracle_per_request_packed_fallback` | c>N metadata debug path invoking one request at a time. | No. |

## Final single-request native prefill pipeline

For one request with prompt length `T`:

1. **Prepare prompt tensors**
   - Validate `token_ids` and capacity.
   - Fill/copy `prefill_token_ids[int64, T]` and `prefill_positions[int64, T] =
     arange(T)`.
   - Resolve the embedding op through the backend/model dispatch path. Qwen3.5
     PARO uses FP16 hidden buffers, so the concrete gfx1100 launch is
     `embedding_lookup_batch_fp16_i64(...)` into `prefill_hidden[T, hidden]`.

2. **Layer-major execution with no production row-loop fallbacks**
   - Maintain `hidden[T, H]` and `next_hidden[T, H]` double buffers.
   - Invariant: every row of `next_hidden[0:T]` is written before the layer-end
     `hidden, next_hidden = next_hidden, hidden` swap.
   - For each layer, route by model layer type through plugin/registry keys; do
     not add backend/quant branches in engine code.

3. **Linear-attention layer final path**
   - Input RMSNorm over `T` rows.
   - PARO rotations/projections over `[T, H]`.
   - Native conv prefill + GDN recurrent prefill over the prompt rows.
   - Update the layer's conv/recurrent state to the prompt tail exactly once.
   - Output projection over `T` rows.
   - Post-attention add/RMSNorm.
   - Run final grouped/compact MoE (below), not retained c1 selected-row MoE.

4. **Full-attention layer final path**
   - Input RMSNorm over `T` rows.
   - Batched PARO Q/K/V projections producing contiguous:
     - `query_proj: fp16[T, num_q_heads, 2 * head_dim]` (query + gate),
     - `key_raw_lowp: fp16/bf16[T, num_kv_heads, head_dim]`,
     - `value: fp16[T, num_kv_heads, head_dim]`.
   - Split/cast query/gate as needed and run batched Q/K head RMSNorm + RoPE with
     per-row positions via `qwen35_head_rmsnorm_partial_rotary_positions_f32_bf16`.
     The existing scalar-position kernel is oracle-only for prefill.
   - Specific scalar bug to avoid: current `prepare_full_attention_qkv_fp16(...)`
     casts only one row of K (`kv_width` elements) from FP16 to FP32. The bulk
     path must cast `T * kv_width` elements.
   - Append all `T` K/V rows to the single request's paged cache with
     `qwen35_write_paged_kv_mixed_value_fp16_prompt_spans(...)`, append spans
     `live_counts = positions`, and `span_role="prefill"`.
   - Run native causal GQA prefill attention over the paged cache using context
     spans `live_counts = positions + 1`.
   - Run output projection over `T` rows.
   - Post-attention add/RMSNorm.
   - Run final grouped/compact MoE.

5. **Final norm/lm head**
   - Only the final prompt row is sampled when `sample=True`.
   - The final hidden row seeds subsequent decode; no extra prompt-token decode
     step is allowed.

### Full-attention prefill kernel contract

First native design is **append-then-attend from cache**:

- Append all prompt K/V rows to paged BF16 KV cache first.
- Attention reads prefix+prompt keys/values entirely from the paged cache.
- Future optimization may read prefix-from-cache plus prompt-from-scratch to
  avoid one HBM round-trip; do not combine that two-source design with the first
  native kernel.

Register a gfx1100 kernel such as:

```python
KernelKey("hip_gfx1100", "full_attn_prefill", "w4_paro", "qwen35_causal_gqa_gate_fp16")
```

`w4_paro` is the model/dispatch identity, as with existing rotary/attention
registrations; the attention kernel itself does not dequantize weights.

Mirror the existing GQA split-K gate-fused decode shape
(`qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_spans`). What differs from
decode: the new kernel processes `T` query rows instead of one, consumes
`positions[row]`/per-row context spans, and applies a causal mask
`cache_position <= positions[row]`. Scratch layout, gate fusion, split-K/reduce
shape, and softmax scale should otherwise match decode.

- Inputs:
  - query `fp32[T, num_q_heads, head_dim]`,
  - gate `fp16[T, num_q_heads, head_dim]`,
  - BF16 paged key/value cache,
  - `KVLiveSpans` with per-row context lengths,
  - output buffer `fp16[T, num_q_heads * head_dim]`.
- For row `r`, attend only to cache positions `<= positions[r]`.
- GQA mapping: `kv_head = q_head // (num_q_heads // num_kv_heads)`.
- Apply the same softmax scale and gate semantics as decode.
- Output is **post-gate FP16**, ready for `project_full_attention_o_fp16(tokens=T)`.

### Grouped/compact MoE final path

The existing `run_moe_c1_fp16(tokens=T)` path is a correctness oracle only. The
retained prefill path must use grouped/compact MoE over prompt rows.

Required ports/wiring:

- Confirm router wrappers cover `[T, hidden]` and keep native router top-k.
- `moe/group_scatter.hip` count, prefix, scatter/scatter_gather, gather packed
  hidden, and WMMA tile-map metadata kernels are landed; the current grouped
  prefill wire-up builds `lane_to_row` in the weighted-lane combine kernel.
- The grouped compact route is wired over packed/sorted lanes and registered as
  `moe_prefill/w4_paro/qwen35_grouped_compact`; compact WMMA gate/up and down
  expert kernels are now the default grouped expert path. A retained throughput
  claim still requires full single-request prefill orchestration and benchmark
  artifact closure.
- The selected-row c1 path is registered only as
  `moe_prefill/w4_paro/qwen35_selected_c1_rows` oracle/fallback coverage; native
  multi-token prefill layer orchestration routes to grouped compact instead.
- Port the Qwen3.5/PARO-used subset of `quant/w8a16_moe.hip` shared/bulk
  variants if/when the parent call graph requires variants beyond the existing
  W8A16 shared expert wrappers; do not port all 17 variants speculatively.
- Port `moe/w8a8_grouped.hip` only if the W4 PARO parent retained path actually
  uses it.
- Port `wmma/wmma_i8_gemm.hip` for long-prompt grouped GEMM once the pack8
  grouped path is correctness-accepted.
- Register:
  - `(hip_gfx1100, moe_prefill, w4_paro, qwen35_grouped_compact)` as retained,
  - `(hip_gfx1100, moe_prefill, w4_paro, qwen35_selected_c1_rows)` only as an
    oracle/fallback key for tests.

## Compact c>N prompt batching final path

Final c>N prefill packs multiple requests into one slab. Per-request invocation
is allowed for debugging/equality tests only.

```python
@dataclass(frozen=True)
class CompactPromptSlab:
    token_ids: Tensor        # int64[T_total]
    positions: Tensor        # int64[T_total], absolute positions per row
    cu_seqlens_q: Tensor     # int32[N + 1]
    cu_seqlens_k: Tensor     # int32[N + 1]
    row_to_request: Tensor   # int64[T_total]
    request_ids: Tensor      # int64[N]
    block_tables: Tensor     # int32[T_total, blocks_per_request] == KVLiveSpans.base_offsets reshaped for the current batch-writer ABI
    append_counts: Tensor    # int64[T_total], 0-based append positions
    context_counts: Tensor   # int64[T_total], 1-based visible lengths
```

Kernel ABI convention: `cu_seqlens_q`/`cu_seqlens_k` define the varlen
block-diagonal attention segments passed to the native causal prefill kernel.
`row_to_request` remains scheduler/debug metadata and is used for validation,
state routing, and output ownership; it is not the primary mask input to the
attention kernel.

Final compact requirements:

- `ResidentBatchScheduler.next_compact_prefill_slabs(chunk_size=...)` forms
  compact slab descriptors for requests with prefill work; legacy
  `next_prefill_work(...)` remains the serial diagnostic path.
- An explicit `bucketize_by_block_count` step in the scheduler runs before slab
  construction and emits one slab per uniform block-table length.
- `Qwen35ParoResidentSession.prefill_native_packed(slab)` is present and
  fail-closed until the remaining packed full-attn and final commit stages
  land; it must eventually run the same native layer logic over `T_total` rows.
- Current batch KV writer constraint: `_check_write_batch_shape(...)` computes
  one `block_table_len = base_offsets.numel // rows`, so every row in a writer
  call must expose the same block-table length. Final scheduler policy should
  bucket slabs by common `blocks_per_request`; cross-bucket requests launch as
  separate native slabs. A true varlen block-table writer is a future kernel
  port, not a reason to use a serial per-request fallback.
- Linear-attention conv/GDN is segment-aware: `f32_segments` conv consumes
  `cu_seqlens` + state slots and `f32_k2_segments` GDN commits each request's
  recurrent tail independently. Packed prefill orchestration must call these
  landed kernels rather than retaining per-request invocation.
- Native causal prefill attention is var-len/block-diagonal:
  `qwen35_varlen_causal_gqa_gate_fp16` consumes `cu_seqlens_q/k`, row-shaped
  block tables, context counts, and positions. Packed prefill orchestration must
  call this landed kernel so a query row attends only to its request segment and
  positions not greater than the query position.
- Native compact prefill is non-speculative and commits canonical KV inline for
  admitted prompt rows. `CompactPromptSlab.slot_ids` carries the physical slots;
  `_commit_packed_prefill_final_rows(...)` commits each segment tail hidden row
  plus position/context metadata after packed layer execution. `KVPolicy.begin_transaction/commit/rollback` hooks remain
  for speculative verify/draft paths, not this ordinary prefill path.
- `prefill_native_packed(slab)` now runs the native compact prefill path and
  returns one final-row sample per request. Decode after the seed still uses
  `step_batch_serial`; replace that serial decode bridge only after c-aware
  decode graph replay lands.

## CPU references and oracles

Before registering new retained gfx1100 layer keys, add or identify the matching
correctness oracle:

- `hipengine/kernels/cpu_reference/ops.py` includes a torch-free NumPy
  `full_attn_prefill` CPU reference for tiny causal-GQA fixtures using
  pre-appended K/V.
- Add CPU-reference or row-by-row c1 oracle coverage for grouped MoE stages.
- Use hipEngine's serial resident path and the parent `nano-vllm-amd` native bulk
  path as external stage/e2e oracles.
- Row-loop full-attention and c1 selected-row MoE may be implemented as test-only
  helpers if useful, but they must not be wired into generation or retained
  performance artifacts.

## Graph capture and tuning

Do not chase graph capture before the final native kernels are in place and
roofline/profiler data says dispatch is material.

- Low-level HIP graph wrappers already live in `hipengine/core/hip.py`.
- `Qwen35ParoResidentSession.capture_decode_graph(...)` exists for decode.
- Add a prefill graph cache only after native single-request and compact c>N
  paths are correct.
- Graph keys should use a small prechosen/power-of-two T bucket set, not one
  graph per exact prompt length.
- `PrefillConfig` chunk sizes mirror parent knobs:
  - `linear_chunk_size`,
  - `full_attn_query_chunk_size`,
  - `full_attn_post_chunk_size`,
  - `full_attn_rope_chunk_size`,
  - `moe_chunk_size`.
- Defaults must match retained parent OPTIMAL flags on W7900 once measured.

## Validation and definition of done

No intermediate perf wins are retained. The doc is complete when these final
artifacts/gates exist.

### Single-request native prefill done

Required checks:

1. Unit/CPU-reference tests for `full_attn_prefill` and grouped MoE stages.
2. Stage probes vs serial resident and/or parent native bulk for:
   - Q/K/V projection layout,
   - batched RoPE,
   - KV append,
   - causal attention post-gate output,
   - grouped MoE output,
   - full layer hidden output.
3. Full 40-layer fixture gate on `fixtures/qwen35_paro/parent_512_32_seed1234.json`:
   greedy generated IDs match the serial resident path, with KL ≤ 0.05 and
   top-1 agreement ≥ 90% on logits at each sampled position.
4. Chunk-equivalence sweep: for non-zero
   `PrefillConfig.{linear_chunk_size, full_attn_query_chunk_size,
   full_attn_post_chunk_size, full_attn_rope_chunk_size}` values, final hidden
   row and KV cache contents match the single-chunk run within the stage-probe
   tolerance, and generated decode IDs/logits satisfy the same KL/top-1 gate.
5. `rocprofv3 --kernel-trace` proves native full-attn prefill and grouped MoE
   kernels ran, with expected names and plausible durations.
6. `LLM.generate`/`Qwen35ParoOneTokenGenerator` uses `prefill_native(...)` by
   default and satisfies the fixture ID/KL/top-1 gate above.

Retained artifact target:

```text
benchmarks/results/2026-05-XX-hipengine-qwen35-native-prefill-full-single-request-accepted.json
```

It must include model, quant, workload shape, W7900 hardware, exact command,
peak memory, correctness gate, kernel names, and comparison to the current
117.24 tok/s c=1 fixture and parent rows.

### Compact c>N prefill done

Required checks:

1. c=2/4/8 generated-token equality vs independent serial c=1 sessions.
2. Finite logits and per-request state/KV bounds checks.
3. Native compact kernels run; no per-request prompt loop in the retained prefill path.
4. At c=8/T=512, prefill tok/s improves over `scheduler_serial_slot_bridge` by
   at least 2× before retaining a throughput claim. This perf row is still
   pending; c=2/4/8 prompt8 correctness is accepted but not a throughput claim.

Accepted correctness artifacts:

```text
benchmarks/results/2026-05-15-hipengine-qwen35-c2-native-compact-prefill-correctness-accepted.json
benchmarks/results/2026-05-15-hipengine-qwen35-c4-native-compact-prefill-correctness-accepted.json
benchmarks/results/2026-05-15-hipengine-qwen35-c8-native-compact-prefill-correctness-accepted.json
```

Retained throughput artifact target:

```text
benchmarks/results/2026-05-XX-hipengine-qwen35-native-prefill-compact-c8-accepted.json
```

Any retained performance row also updates `benchmarks/README.md`,
`benchmarks/CHANGELOG.md`, and a compact JSON artifact under
`benchmarks/results/`.

Correctness is non-negotiable: a faster prefill that fails the parent fixture is
a regression, not a win.

## Optimization diagnosis (2026-05-16): the 4K gap is one kernel

This section captures the trace-driven diagnosis after the 49-iteration
`prefill-perf` multiloop plateaued at 2039 tok/s @ 512/128, plus the
standing-rule reasoning chain for the next optimization spike. It is
standalone evidence: a future agent should be able to read this section and
reproduce the decision without re-running the audit.

### Where we stand

Measured with the standard bench command on the parent 512/32 fixture and the
4K/128 repeated-token diagnostic; both runs use `require_full_native=True`.

| Shape          | hipEngine | nano-vllm-amd (parent) | parent / hipEngine |
| -------------- | --------: | ---------------------: | -----------------: |
| 512 prefill    | 2039 tok/s | 2589 tok/s              | +27 %              |
| 4K prefill     |  659 tok/s | 1681 tok/s              | +155 %             |

The 4K gap is the load-bearing one. At T=4K, hipEngine spends 6.21 s in
prefill vs the parent's ≈ 2.44 s. Multiloop iters 1–49 optimized only the 512
metric and treated 4K as a no-regression guard; that left the long-context
path structurally unexamined.

### Trace comparison

From `rocprofv3 --kernel-trace` on `qwen35_paro_bench.py` with
`--prompt-length {512,4096} --decode-tokens 0 --max-layers 40` and matching
flags from `~/amd-gpu-tuning/scripts/run_moe2_baselines.py::COMMON_ENV` on the
parent side. Numbers are summed across the 40 layers.

Top kernel buckets, hipEngine 512 prefill (total kernel time 229.77 ms):

| ms      | calls | avg us  | kernel                                                |
| ------: | ----: | ------: | ----------------------------------------------------- |
|   41.22 |    30 |  1373.9 | `qwen35_gdn_prefill_recurrent_k2_kernel`              |
|   33.96 |    40 |   849.0 | `gemm_awq_selected_dual_pack8_wmma_compact_kernel`    |
|   26.16 |    10 |  2615.8 | `qwen35_paged_full_attn_prefill_gqa_gate_fp16_kernel<true>` |
|   23.21 |    80 |   290.2 | `awq_fusedw4_prefill_fp16_kernel<32,32,true>`         |
|   19.48 |    40 |   487.0 | `gemm_awq_selected_pack8_wmma_compact_kernel`         |
|   16.05 |    40 |   401.2 | `w8a16_shared_down_combine_residual_fp16_kernel`      |
|   15.56 |    40 |   389.0 | `w8a16_shared_gate_up_silu_fp16_kernel`               |
|   14.81 |    50 |   296.2 | `awq_fusedw4_prefill_fp16_kernel<32,32,false>`        |
|    9.24 |    80 |   115.5 | `paro_rotate1_kernel<_Float16>`                       |
|    8.62 |    40 |   215.5 | `qwen35_router_logits_token_tile_kernel<_Float16,4>`  |

Top kernel buckets, hipEngine 4K prefill (total kernel time 6171.07 ms):

| ms       | calls | avg us    | kernel                                                |
| -------: | ----: | --------: | ----------------------------------------------------- |
|  4572.38 |    10 |  457237.5 | `qwen35_paged_full_attn_prefill_gqa_gate_fp16_kernel<false>` |
|   391.64 |    30 |   13054.8 | `qwen35_gdn_prefill_recurrent_k2_kernel`              |
|   199.96 |    40 |    4999.0 | `gemm_awq_selected_dual_pack8_wmma_compact_kernel`    |
|   170.16 |    80 |    2127.0 | `awq_fusedw4_prefill_fp16_kernel<32,32,true>`         |
|   133.07 |    80 |    1663.4 | `paro_rotate1_kernel<_Float16>`                       |
|   124.89 |    40 |    3122.3 | `w8a16_shared_down_combine_residual_fp16_kernel`      |
|   117.35 |    40 |    2933.8 | `w8a16_shared_gate_up_silu_fp16_kernel`               |
|   116.15 |    40 |    2903.8 | `gemm_awq_selected_pack8_wmma_compact_kernel`         |
|    86.33 |    50 |    1726.7 | `awq_fusedw4_prefill_fp16_kernel<32,32,false>`        |
|    65.39 |    40 |    1634.7 | `qwen35_router_logits_token_tile_kernel<_Float16,4>`  |

Key observations:

1. **The full-attention prefill kernel template flips between 512 and 4K.** At
   T=512 the trace runs `qwen35_paged_full_attn_prefill_gqa_gate_fp16_kernel<true>`
   (split-K enabled, 26.16 ms / 10 layers); at T=4K it runs the same kernel as
   `<false>` (split-K disabled, **4572.38 ms / 10 layers, 457 ms per layer**).
   8× the tokens produces 175× the kernel time. That is super-quadratic; a
   correctly-tiled Flash-Attention-style kernel scales as T² in compute but
   stays HBM-bandwidth-bound and finishes ≈ 64× of the T=512 cost, not 175×.
2. **`<false>` is 74 % of all 4K kernel time** (4572 / 6171 ms). Closing this
   one bucket is worth more than every other optimization in the multiloop
   combined.
3. **`paro_rotate1` is also super-linear** (115 us → 1663 us, 14.4× growth for
   8× tokens). Nano-vllm-amd's analogous `paroquant_rotate_kernel` is
   near-linear (36 us → 240 us, 6.7×). This is a secondary but real RDNA3
   tiling/occupancy issue, not the headline.
4. **GDN recurrent prefill scales as ~9.5×** (41 → 392 ms) — roughly linear
   for 8× tokens with mild overhead, and within 7 % of nano-vllm-amd parity.
   Multiloop iters 36–37 were working against a real ceiling there; further
   grinding on that bucket is unlikely to pay.
5. **MoE compact-WMMA + W8A16 shared family scales ~6–8×** as expected for
   linear MoE work. Combined hipEngine MoE+shared kernel time is ≈ 1.27× the
   nano-vllm-amd equivalent, because nano-vllm-amd silently opts OUT of compact
   WMMA at long T and dispatches `hipBLASLt` HGEMM with per-shape autotuned
   tiles (MT96×96×32, MT128×48×32, MT96×32×32 observed). Compact WMMA is a
   correct prefill path but is not the W7900-optimal one at T ≥ 1K.

### Why our kernel mis-scales

Direct read of
`hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.hip:1039–1193`
(`qwen35_paged_full_attn_prefill_gqa_gate_fp16_kernel`) shows four
structural problems, all visible in the source:

1. **LDS scratch scales with `max_context_len`.** Line 1083:
   `extern __shared__ float shared[]; float* scores = shared; float* partial =
   scores + max_context_len; float* q_shared = partial + blockDim.x;`. The
   `scores` buffer is `max_context_len * 4 B` per block: 2 KiB at T=512,
   16 KiB at T=4K, ~128 KiB at T=32K. RDNA3 ships ~64 KiB LDS per CU, so block
   residency drops from ≥8 blocks/CU at 512 to ≤3 blocks/CU at 4K to
   single-block-per-CU at 32K. Occupancy collapse compounds the T² cost.
2. **The V@scores epilogue is a fully serial T-deep inner loop, per output
   dim.** Line 1170:
   `for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
   float acc = 0.0f; for (int64_t token = 0; token < visible_len; ++token)
   { ... acc += scores[token] * value_cache[v_offset]; } }`. Each thread
   walks the full T axis sequentially, fetching every V row from HBM with no
   LDS staging. At T=4K that is 4096 serial multiply-accumulates per
   `(thread, output_dim)` pair, with one HBM load each.
3. **GQA KV sharing is missing.** Line 1084: `kv_head = q_head / kv_group`,
   computed independently per block. With 16 Q-heads and 2 KV-heads, each of
   the 8 Q-heads in a KV group has its own block that re-streams the same
   K/V cache through HBM. That is 8× redundant K/V bandwidth.
4. **The `<true>` / `<false>` template flip is a red herring.** It toggles
   `SHORT_BLOCK256` for short-context block-table inlining (line 1090–1097);
   it does not change the inner attention algorithm. Both branches share the
   serial V-loop above. T=512 looks acceptable only because all three issues
   are small at that length.

Observed 512→4K scaling is 178× for 8× length. A correctly-tiled Flash-Attention
implementation is O(T²) in compute but stays bandwidth-bound and runs in
≈64× the T=512 cost; the extra ≈3× is exactly issues (1) + (2) + (3)
compounding. The `<false>` branch is not a one-off bug; the entire kernel
family is pre-Flash-Attention.

**The kernel does carry one piece of fused logic we have to preserve.** Lines
1191, 1350, 1410:
`out[...] = static_cast<half_t>(acc * sigmoid_f32(gate_v))`. The attention
epilogue multiplies the per-`(row, q_head, dim)` output by
`sigmoid(gate[row, q_head, dim])`, where `gate` is a separate FP16 tensor
produced by the upstream QKV projection split. AOTriton's `attn_fwd*` API
has no gate input; any AOTriton-based replacement must add a trivial
elementwise post-pass kernel (`out *= sigmoid(gate)`) immediately after the
attention call to maintain model semantics. At T=4K, head_dim=128,
num_q_heads=16 the post-pass is one HBM-bandwidth-bound pass over
≈ 4096 × 16 × 128 = 8.4 M FP16 elements per layer — expected cost ≤ 0.2 ms
per layer, well inside noise.

A Flash-Attention-style fix to the existing kernel is not a tuning change; it
is an algorithmic rewrite: tile Q in registers, stream K/V chunks through LDS,
maintain online softmax running statistics across the K loop, share K/V
fetches across the GQA group, and apply causal masking inline. That is
several thousand lines of HIP plus several iterations of LDS bank-layout
tuning. We are not going to get there inside the existing multiloop budget by
turning knobs on the current kernel.

### Options for fast prefill attention without `torch` in the hot path

The four-axis registry and torch-free hot path invariants mean we cannot just
clone nano-vllm-amd's call to `F.scaled_dot_product_attention`. The viable
plugin keys all keep `import torch` out of the generation path.

| Option                                          | Source on disk / API                                                                                                 | gfx1100 support | Effort  | Expected 4K result vs current |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | --------------- | ------- | ----------------------------- |
| AOTriton 0.11.2b standalone C++ ABI             | `~/.cache/hipengine/aotriton/0.11.2b/{include,lib}/` after `scripts/fetch_aotriton.sh`; symbols `aotriton::v2::flash::attn_fwd{,_compact_varlen}` in `libaotriton_v2.so.0.11.2`; pinned in `aotriton_release.toml`. | yes, 396 pretuned gfx11xx forward variants | 2–3 days | ≈ 1700 tok/s (closes ~94 % of 4K gap) |
| Hand-rolled HIP FA-2 with WMMA                  | new code under `kernels/hip_gfx1100/attention/`; oracle = AOTriton output                                            | yes (we write it) | 3–6 weeks | 1300–1900 tok/s depending on tuning |
| Composable Kernel `ck_tile/01_fmha`             | `~/amd-gpu-tuning/reference/composable_kernel/example/ck_tile/01_fmha/`                                              | **no** — `known_fails_gfx{90a,942,950}.txt` only; CDNA-targeted | n/a | not applicable on W7900 |
| vLLM-vendored CK FA (CK fork inside vLLM)       | `~/vllm/flash-attention/csrc/composable_kernel/CMakeLists.txt` builds for `gfx1100;gfx1101;gfx1102`                  | yes (claimed) | 1–2 weeks (build + wrap) | uncertain, likely 1400–1700 tok/s |
| FlashAttention-2 (Dao-AILab upstream)           | wheels in `~/Downloads/`, CDNA-only HIP path                                                                          | no            | n/a     | not applicable on W7900 |
| Patch the existing `<false>` branch in place    | `hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.{hip,py}`                                                  | yes (ours)    | unclear | uncapped; the path needs an FA rewrite, not a patch |

AOTriton specifics worth recording so a future agent does not re-derive them:

- `AOTRITON_NS::v2::flash::attn_fwd_compact_varlen` takes `cu_seqlens_q`,
  `cu_seqlens_k`, `max_seqlen_q`, `max_seqlen_k`, `is_causal`, and a
  `hipStream_t`-equivalent stream. This matches our existing
  `CompactPromptSlab.cu_seqlens_q/k` ABI almost verbatim and the `KVLiveSpans`
  prefill role; no scheduler changes are required.
- The tensor type is `AOTRITON_NS::TensorView<N>` — a `(void* ptr, shape[N],
  stride[N], dtype)` descriptor. There is no `torch::Tensor` anywhere in the
  AOTriton public headers (`include/aotriton/{flash,runtime,util,dtypes,cpp_tune}.h`).
- Disk footprint on gfx11xx, verified on this host post-`scripts/fetch_aotriton.sh`
  (`du -sh ~/.cache/hipengine/aotriton/0.11.2b/`):
    - Combined tarball download: 4.9 MB runtime + 475 MB compressed gfx11xx
      images = **480 MB on the wire**.
    - `libaotriton_v2.so.0.11.2` ≈ 5 MB.
    - Default prune keeps only `flash/attn_fwd/` (396 forward variants for
      bf16/fp16 × head_dim ∈ {16…256} × causal × …); drops every `bwd_*`,
      `bwd_preprocess*`, and `debug_*` subdir. Post-prune cache: **159 MB**.
    - Tighter prune to the exact shapes hipEngine invokes
      (bf16/fp16 × head_dim 128 × causal=true × dropout=false × no-bias) is
      the per-GPU kernel streaming future-work item below; not implemented.
- The 396 pretuned variants do per-shape kernel selection at call time; this
  is the value we would lose by hand-rolling. The default attn_fwd-only prune
  is safe because hipEngine's attention shape set is fixed at model-load time
  and small (one head_dim per registered model, causal-only forward).

### Why "surely native HIP beats Triton" is not a fast path

Triton lowers to AMDGPU LLVM IR through MLIR and emits the same instruction
class — `v_wmma_*`, `ds_read_b128`, `v_dual_*` — that a hand-written HIP
kernel would. On gfx1100 the Triton tax over a perfectly-tuned hand kernel is
typically 5–15 %, often less. The existing `<false>` branch is not slow
because HIP cannot match Triton; it is slow because it does not implement the
Flash-Attention algorithm. Catching up to AOTriton requires implementing FA-2
correctly first; only then does the per-shape hand-tuning headroom open up.

A hand-written FA-2 spike that lands in less than 30 days will almost
certainly underperform AOTriton on gfx1100, because AOTriton already ships
shape-specialized binaries and our spike will run one tile schedule. Using
AOTriton as the perf oracle is what makes a later native port tractable.

### Recommended phased plan

**Phase 1 — AOTriton attention plugin** (next multiloop spike).

- Add a new kernel-build module under `hipengine/kernels/hip_gfx1100/attention/`
  (`aotriton_wrap.{cc,py}`) that links against the manifest-pinned
  `libaotriton_v2.so` and exposes a stable `extern "C"` surface around
  `attn_fwd_compact_varlen` (the varlen path matches `CompactPromptSlab`).
  `ctypes` dlopens hipEngine's own wrapper `.so`, not AOTriton directly; see
  "Stable-ABI shim, not raw dlopen" below.
- Register `KernelKey("hip_gfx1100", "full_attn_prefill", "w4_paro",
  "aotriton_attn_fwd")` alongside
  `(... , "qwen35_causal_gqa_gate_fp16")`. No `if backend=="..."`,
  no `if quant=="..."`; the model layer asks the registry for an attention
  prefill key and gets one. The existing kernel stays registered as the
  short-T variant.
- Threshold via `PrefillConfig.attn_aotriton_min_tokens` (default `512`, per
  the retained threshold sweep); decode and short prefill continue on the
  existing hand-rolled kernel where it is fine.
- Add a tiny **gate-fusion post-pass kernel** in the same module:
  `out[row, q_head, dim] *= sigmoid(gate[row, q_head, dim])` over the
  AOTriton output. The existing prefill kernel fuses this inside its
  epilogue (`paged_attn_decode.hip:1191`) and we must preserve the
  semantics. Single elementwise pass; ≤ 0.2 ms at T=4K, head_dim=128,
  num_q_heads=16. Reuse the existing decode-side gate kernel pattern at
  `paged_attn_decode.hip:316,329` for the math; only the launch shape
  changes.
- Use the Git-LFS-vendored, pinned-manifest scheme described in "AOTriton
  distribution and pinning strategy" below: pin in `aotriton_release.toml`,
  vendor/update with `scripts/vendor_aotriton.sh`, and resolve via
  `aotriton_runtime_tree()`. Do not add a submodule; do not depend on
  PyTorch's bundled copy.
- Correctness gate: re-run `scripts/qwen35_native_prefill_fixture_gate.py` on
  `fixtures/qwen35_paro/parent_512_32_seed1234.json` and the 4K repeated-token
  diagnostic; require `passed=true`, `max_kl <= 0.05`, top-1 ≥ 90 %, and
  generated IDs equal to the serial path with the AOTriton variant active.
- Perf gate: 512/128 median prefill_tok_s ≥ current best (2039), 4K/128 ≥ 1500
  before a row is retained; target band 4K/128 ≥ 1700 after one round of
  threshold tuning.
- Plugin-registry compliance is the load-bearing invariant here: AOTriton
  enters as a new variant key, never as a branch in dispatch code.

**Phase 2 — hipBLASLt for MoE projection at T ≥ 1K** (parallel, independent).

- Wrap `hipblasLtMatmul` from `/opt/rocm/lib/libhipblaslt.so.1.2` for the
  shared-expert W8A16 path and grouped-stacked MoE projection at T ≥ 1K, where
  nano-vllm-amd's trace shows it dispatches HGEMM tiles instead of compact
  WMMA. Register as `(hip_gfx1100, shared_expert | moe_prefill, w8a16 |
  w4_paro, hipblaslt_hgemm)` variants; compact WMMA stays as the short-T
  variant.
- Expected delta: 5–10 % at 512, 15–25 % at 4K, on top of Phase 1.

**Phase 3 — native HIP FA-2 port** (optional, only if AOTriton bundle is
unacceptable or per-shape headroom is measurable).

- Reference: vLLM-vendored CK FA on `gfx1100;gfx1101;gfx1102`, ck_tile/01_fmha
  algorithm pattern. AOTriton output is the correctness and perf oracle.
- 3–6 weeks. Expected per-shape gain over AOTriton: 0–15 % at our exact shape
  (head_dim 128, kv_heads 2, num_q_heads 16, causal, BF16 paged cache,
  post-gate FP16 output). The unique win is fusing the existing post-gate
  semantics into the FA epilogue, which AOTriton cannot do for us.

### AOTriton distribution and pinning strategy

AOTriton (`https://github.com/ROCm/aotriton`) is under active development:
ABI churn is real between minors, release artifacts are matrixed across ROCm
minors, and the version PyTorch bundles is mangled (the conda installs on this
host show `libaotriton_v2.so.torch` symlinks under `torch/lib/`). hipEngine
pins one upstream release in `aotriton_release.toml` and now vendors the exact
pruned gfx11xx runtime/images needed by the Qwen3.5/PARO inference path under
`hipengine/kernels/hip_gfx1100/attention/aotriton_runtime/`. Binary payloads
are tracked with Git LFS. No submodule. No PyTorch dependency.

#### Pinned version: 0.11.2b

We pin AOTriton 0.11.2b (released 2026-01-28). Rationale:

- The 0.11.x release line ships a `manylinux_2_28_x86_64-rocm7.0-shared`
  build whose `libaotriton_v2.so.0.11.2` NEEDs `libamdhip64.so.7` directly.
  Our host ROCm is 7.2.2; no `libamdhip64.so.6` compat shim is required.
- The V2 API we depend on (`attn_fwd_compact_varlen`) is present and
  signature-stable in 0.11.x (`include/aotriton/v2/flash.h`). V3 is the new
  default but V2 is "frozen for backward compatibility", not removed; our
  wrapper continues to call V2.
- The 0.11b release notes warn that gfx1100 is "experimental" due to
  "massive accuracy problems". That warning is **training-only**:
  `test/adiffs/gfx1100.txt` in 0.11.2b lists 436 failing tests, 100% of which
  are in `test_backward.py` (backward/training kernels). Zero forward-pass
  failures are listed. hipEngine is inference-only, so the warning does not
  apply. As an additional sanity check, 0.11.1b restored Navi31 support via
  an alternative wheel mechanism and 0.11.2b shipped an updated `gfx11xx`
  image tarball; the gfx11xx images tarball has 60k+ downloads.

Non-targets:

- 0.11.210b is a gfx942-only ASAN debug build.
- 0.11.52b is a gfx1250-only tech preview.
- 0.10b and earlier ship a single combined tarball but only against ROCm
  ≤7.0 (the rocm7.0-shared assets for 0.10b would technically still work,
  but we gain nothing by staying older).
- 0.8.x requires `libamdhip64.so.6`. We deliberately do not support that
  path now that ROCm 7 is the deployed runtime.

#### Two-tarball release layout (changed in 0.11.x)

0.11.x split runtime and GPU images into separate tarballs. For gfx1100 we
need both:

| Asset | URL fragment | Size | Contents |
| --- | --- | ---: | --- |
| Runtime | `aotriton-0.11.2b-manylinux_2_28_x86_64-rocm7.0-shared.tar.gz` | 4.9 MB | `lib/libaotriton_v2.so*`, `include/aotriton/{flash,runtime,util,v2}.h` |
| GPU images (gfx11xx) | `aotriton-0.11.2b-images-amd-gfx11xx.tar.gz` | 475 MB | `lib/aotriton.images/amd-gfx11xx/flash/{attn_fwd,bwd_*,debug_*}/*.aks2` |

Both share the same top-level `aotriton/` directory and merge cleanly into
one cache tree. The committed vendor tree is pruned further than the generic
fetch cache: it keeps the runtime library/headers plus the 12 BF16 head_dim=256
`gfx11xx` forward-attention images used by Qwen3.5/PARO AOTriton prefill. The
vendored footprint is ~24 MiB on disk here (~42 MB logical file bytes) before
Git LFS pointer substitution.

#### Manifest schema

`hipengine/kernels/hip_gfx1100/attention/aotriton_release.toml` is the
source of truth for the pin. Schema (see the file for the live values):

```toml
[aotriton]
version = "0.11.2b"
so_name = "libaotriton_v2.so.0.11.2"
rocm_min = "7.0"
rocm_max = "7.x"

[[aotriton.archives]]
kind = "runtime"
url = "...rocm7.0-shared.tar.gz"
sha256 = "5501a0a3..."
size_bytes = 4884827

[[aotriton.archives]]
kind = "images"
arch = "amd-gfx11xx"
url = "...images-amd-gfx11xx.tar.gz"
sha256 = "83929963..."
size_bytes = 475390993

[aotriton.prune]
keep_flash_subdirs = ["attn_fwd"]
keep_archs = ["amd-gfx11xx"]

[aotriton.vendor]
relative_path = "aotriton_runtime/0.11.2b"
image_glob = "lib/aotriton.images/amd-gfx11xx/flash/attn_fwd/FONLY__＊bf16@16_256_*___gfx11xx.aks2"
image_count = 12
```

#### Lookup chain at module load

`hipengine/kernels/hip_gfx1100/attention/aotriton.py:aotriton_runtime_tree()`
resolves the runtime in this order. First hit wins; nothing else is
consulted.

1. **`HIPENGINE_AOTRITON_LIB`** — explicit path to `libaotriton_v2.so`.
   The matching `include/` and `aotriton.images/` trees must live at the
   standard sibling paths. Developer override; not for production.
2. **Explicit `root` argument or `HIPENGINE_AOTRITON_HOME`** — cache root that
   contains `<version>/lib/libaotriton_v2.so`. Missing explicit roots fail
   loudly instead of silently falling back.
3. **Vendored package tree** —
   `hipengine/kernels/hip_gfx1100/attention/aotriton_runtime/<version>/`.
   This is the baseline path for normal checkouts.
4. **Default external cache `${HOME}/.cache/hipengine/aotriton/<version>/`** —
   useful for pin refreshes or local experiments.
5. **`/opt/rocm/lib/libaotriton_v2.so`** — only when its SONAME matches the
   manifest's pinned `so_name`. Not shipped by ROCm 7.2.2 today; reserved
   for future ROCm releases that begin bundling AOTriton.
6. Nothing found → `AotritonNotInstalledError` pointing at Git LFS or
   `scripts/fetch_aotriton.sh`. AOTriton is the baseline runtime dependency;
   install LFS objects with `git lfs pull` after clone if the vendored payload
   is missing.

There are no other env vars. `HIPENGINE_AOTRITON_SOURCE_ROOT` and
`HIPENGINE_AOTRITON_RUNTIME_ROOT` (used during the 0.8.x spike) have been
removed; if you find them in docs or commit messages, they predate the
0.11.2b cleanup.

#### Bring-up checklist

Normal checkout bring-up:

```bash
git lfs install
git lfs pull
```

The vendored tree should then exist at:

```text
hipengine/kernels/hip_gfx1100/attention/aotriton_runtime/0.11.2b/
```

`pip install hipengine` from a source tree or wheel includes the vendored
runtime as package data once the LFS objects are present. Refreshing the
vendored baseline is explicit:

```bash
scripts/vendor_aotriton.sh --force
  [--local-tarball-dir PATH]   # reuse already-downloaded release tarballs
  [--skip-fetch]               # copy from an already-populated cache
```

`scripts/fetch_aotriton.sh` remains available for pin refreshes, offline mirrors,
or external-cache overrides:

```bash
scripts/fetch_aotriton.sh
  [--dest ~/.cache/hipengine/aotriton]   # default; override if cache lives elsewhere
  [--no-prune]                           # default prunes to flash/attn_fwd; --no-prune keeps everything
  [--local-tarball-dir PATH]             # reuse already-downloaded tarballs by filename
  [--no-verify-sha]                      # opt-out for offline mirrors only
  [--force]                              # re-extract on top of an existing version directory
  [--dry-run]                            # print the plan as JSON without downloading
```

The helper downloads both pinned tarballs (~480 MB total), verifies SHA256
against the manifest, extracts into `~/.cache/hipengine/aotriton/<version>/`,
prunes per the manifest, and writes `MANIFEST.local.json` recording
provenance. To update the vendored baseline after changing the manifest pin,
run `scripts/vendor_aotriton.sh --force`; it fetches (or reuses) the cache,
copies the pruned runtime/images into `aotriton_runtime/<version>/`, updates
`MANIFEST.vendor.json`, and then the fixture gate must be re-run.

#### Why not just `pip install aotriton`

There is no standalone PyPI wheel that ships the gfx11xx tile database
usefully. The pip distribution channel is PyTorch's; relying on it couples
hipEngine to a torch version we explicitly do not import. The standalone
tarballs at `https://github.com/ROCm/aotriton/releases/` are the
upstream-blessed distribution; that is what we pin.

#### Stable-ABI shim, not raw dlopen

The C++ entry we want is
`AOTRITON_NS::v2::flash::attn_fwd_compact_varlen(...)`. Mangled, the symbol
is ~120 characters and varies across AOTriton minors. Linking Python
`ctypes` against that directly would break on every upstream rebuild.
Instead:

1. `hipengine/kernels/hip_gfx1100/attention/aotriton_wrap.cc` includes
   `<aotriton/v2/flash.h>` from the resolved cache and links against
   `lib/libaotriton_v2.so`, exposing a small `extern "C"` surface
   (`hipengine_aotriton_attn_fwd_compact_varlen`,
   `hipengine_aotriton_attn_fwd_compact_varlen_gqa_per_q_head`,
   `hipengine_aotriton_gate_mul_fp16_inplace`,
   `hipengine_aotriton_check_gpu`).
2. `aotriton_wrap.py` dlopens *our* wrapper .so (built by the same hipcc JIT
   path that builds all other gfx1100 kernels), not AOTriton directly.
3. Bumping AOTriton: edit `aotriton_release.toml`, re-run
   `scripts/vendor_aotriton.sh --force`, re-build the wrapper (seconds via JIT
   cache), re-run the fixture gate.

If AOTriton 0.12 renames the entrypoint, only `aotriton_wrap.cc` changes;
the registry key, runtime call site, and Python ABI remain stable.

PyTorch dispatch policy is unrelated: `pytorch/pytorch#166397` (Nov 2025)
marked gfx1100 as "experimental" in PyTorch's SDPA backend matrix. That is
a PyTorch QA policy decision, not a statement about AOTriton kernel
correctness on gfx1100. hipEngine calls AOTriton directly via its C++ ABI
and is unaffected.

#### Future work: per-GPU kernel streaming (not implemented)

The gfx11xx images tarball is 475 MB compressed because it bundles every
shape variant across gfx1100/1101/1102/1103 plus the full forward + backward
+ debug surface. hipEngine only invokes the forward `attn_fwd` family on one
physical GPU at a time, and even within `attn_fwd` only calls a handful of
shape variants for the model in use. There is an obvious opportunity to
stream only the necessary kernels per detected GPU and per
manifest-declared shape signature.

Sketch of the eventual design (do **not** implement as part of the 0.11.2b
landing; recorded here so a future iteration can pick it up):

- Detect the host GPU at module load (`rocminfo` or HIP runtime device
  properties; we already do something similar for backend selection).
- The manifest grows a `[[aotriton.shapes]]` table declaring the
  (`dtype`, `head_dim`, `causal`, `BLOCK_M`, ...) tuples hipEngine actually
  invokes for each registered model. Filename pattern
  `FONLY__＊<dtype>@<BLOCK_M>_<head_dim>_<causal>_<...>___gfx11xx.aks2` is
  decodable (verified in `aotriton-0.11.2b-images-amd-gfx11xx.tar.gz`).
- Fetcher resolves the cross product to a small file list and downloads only
  those .aks2 + their .json metadata, either by HTTP range over the upstream
  tarball or by mirroring the pruned set on a hipEngine-controlled location
  (GitHub release of a `hipengine-aotriton-shapes-gfx1100` artifact, S3,
  etc.). A 32-file pruned set is ~3 MB.
- Cache invalidation: keyed on (manifest version, GPU arch string, shape
  signature hash). New shapes trigger an incremental pull.
- Graceful degradation: if the AOTriton selector requests a variant not in
  the pruned cache, fall back to the hand-rolled kernel for that call and
  emit one diagnostic so the manifest can grow.

The 0.11.2b vendor step already mitigates the original ~500 MB fetch by
committing only the 12 BF16 head-dim-256 forward images hipEngine uses.  Future
shape streaming would reduce wheel/source-checkout footprint further, but it is
not required for the baseline runtime dependency.

### Explicit non-goals for the next spike

- Do not patch the `<false>` branch of the existing prefill attention kernel.
  The algorithmic gap is FA-vs-not-FA, not tile-tuning; patches there waste
  iterations.
- Do not write a from-scratch FA-2 before AOTriton is wrapped. Without an
  oracle ceiling we cannot tell a good hand-rolled kernel from a mediocre one;
  iters 1–49 demonstrate the cost of optimizing without one.
- Do not introduce `import torch` in `hipengine/runtime/`,
  `hipengine/generation/`, `hipengine/models/`, `hipengine/dispatch/`, or any
  kernel module reached by `LLM.generate()`. AOTriton is loaded via dlopen and
  called through `ctypes`; the existing dlopen pattern in
  `hipengine/core/hip.py` is the template.
- Do not branch on `backend == "..."` or `quant == "..."` in dispatch or model
  code to route to AOTriton. Use the kernel registry; that is what the
  four-axis design exists for.

### Reproduction commands

Trace comparison evidence above was produced with:

```bash
# hipEngine 512/0 trace
rocprofv3 --kernel-trace -d /tmp/iter50-shared-down-tile8-trace -o trace -- \
  python3 scripts/qwen35_paro_bench.py --token-id 9707 \
    --prompt-length 512 --decode-tokens 0 --warmup-decode-tokens 0 \
    --max-layers 40 \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build --json /tmp/iter50.json

# hipEngine 4K/0 trace
rocprofv3 --kernel-trace -d /tmp/iter52-4k-profile-trace -o trace -- \
  python3 scripts/qwen35_paro_bench.py --token-id 9707 \
    --prompt-length 4096 --decode-tokens 0 --warmup-decode-tokens 0 \
    --max-layers 40 \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build --json /tmp/iter52.json
```

Parent comparison numbers (2589 tok/s @ 512, 1681 tok/s @ 4K) are from a
peer-system audit at the same git tree under
`~/amd-gpu-tuning/scripts/bench_paro_native_engine.py --model-preset
qwen35-a3b-paro --prompt-len {512,4096} --decode-len 0 --prefill-mode bulk
--no-warmup` with the `COMMON_ENV` flags from
`~/amd-gpu-tuning/scripts/run_moe2_baselines.py`. Re-running the parent profile
on this host is blocked behind ongoing GPU contention; the numbers above are
recorded against the parent audit transcript and treated as the comparison
baseline until reproduced locally.

## References

- `docs/PLAN.md` — architecture, phase roadmap, extensibility, KV ABI.
- `docs/KERNELS.md` — live kernel catalog and port playbook.
- `docs/BENCHMARK.md` — benchmark protocol and artifact rollup rules.
- `docs/TESTING.md` — RED/GREEN workflow, fixtures, correctness gates.
- `docs/ROOFLINE.md` — W7900/RDNA3 performance model.
- `docs/DFLASH.md` — related speculative path using the same batch-shaped ABI.
- `~/amd-gpu-tuning/docs/PARO.md` — parent retained rows and config.
- `~/amd-gpu-tuning/docs/OPTIMAL.md` — parent optimal Qwen3.5/PARO route and flags.
- `~/amd-gpu-tuning/scripts/bench_paro_native_engine.py` — parent `prefill_bulk(...)` reference.
- `~/amd-gpu-tuning/nano-vllm-amd/nanovllm/native/qwen35/paroquant.py` — parent layer implementations.
- `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefill-full-attn-boundary-blocked.json` — current full-attention boundary.
- `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefix-scratch-restore-sweep.json` — accepted linear-prefix correctness.
- `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefix-serial-suffix-full40-accepted.json` — accepted legacy suffix correctness.
- `benchmarks/results/2026-05-15-hipengine-qwen35-c1-parent-fixture-accepted.json` — current c=1 perf/correctness baseline.
- `benchmarks/results/2026-05-15-hipengine-qwen35-c8-scheduler-serial-bench-blocked.json` — current c=8 serial bridge diagnostic.
