# hipENGINE Optimization Grind Plan

Status: 2026-05-16

Scope: Qwen3.5-35B-A3B-PARO `w4_paro` on W7900/gfx1100, batch-1 prompt/decode rows first.

This document is the working plan for turning the current Qwen3.5/PARO resident-runner diagnostics into a path that beats the source-lineage `nano-vllm-amd` rows and the llama.cpp HIP/Vulkan comparison rows across prefill, decode, and memory.

It complements:

- `docs/PREFILL.md` — native prefill architecture and AOTriton/chunking evidence.
- `docs/KERNELS.md` — kernel catalog, source-lineage map, and port gates.
- `docs/ROOFLINE.md` — RDNA3/W7900 performance model and anti-rabbit-holes.
- `docs/BENCHMARK.md` and `benchmarks/README.md` — benchmark evidence policy and rollup.
- Parent references under `/home/lhl/amd-gpu-tuning/`, especially `docs/OPTIMAL.md`, `PLAN-PAROQUANT*.md`, `PLAN-LONGCONTEXT.md`, `docs/LLAMACPP-VULKAN.md`, and `LESSONS-LEARNED.md`.

## Current scoreboard

Current hipENGINE rows are **diagnostic resident-runner rows**, not accepted public `LLM.generate()` throughput rows yet. They use:

```text
--attn-aotriton-min-tokens 512
--graph-replay-decode
--prefill-linear-chunk-size 1024
--prefill-moe-chunk-size 1024
--prefill-full-attn-query-chunk-size 4096
--prefill-full-attn-post-chunk-size 1024
--prefill-full-attn-rope-chunk-size 1024
```

Source: `benchmarks/results/2026-05-16-hipengine-qwen35-comparison-tables-diagnostic.json`. Quick view:

```bash
python3 scripts/qwen35_compare_tables.py all
```

### hipENGINE vs nano-vllm-amd parent

| Workload | Prefill delta | Decode delta | Peak memory delta |
| --- | ---: | ---: | ---: |
| 512/128 | -13.3% | -5.7% | -0.28 GiB |
| 4K/128 | -7.3% | -1.7% | -1.77 GiB |
| 32K/128 | +0.3% | -4.9% | -0.68 GiB |
| 128K/128 | +9.7% | -2.5% | -3.76 GiB |

To beat parent everywhere, hipENGINE needs roughly:

| Workload | Needed prefill lift | Needed decode lift | Memory status |
| --- | ---: | ---: | --- |
| 512/128 | +15.4% | +6.0% | already lower |
| 4K/128 | +7.9% | +1.7% | already lower |
| 32K/128 | already ahead | +5.2% | already lower |
| 128K/128 | already ahead | +2.5% | already lower |

### hipENGINE vs llama.cpp HIP

| Workload | Prefill delta | Decode delta | Memory |
| --- | ---: | ---: | --- |
| 512/128 | -9.0% | +27.6% | llama.cpp split rows have no retained memory |
| 4K/128 | +15.1% | +26.0% | — |
| 32K/128 | +26.1% | +22.0% | — |
| 128K/128 | +41.1% | +6.5% | — |

To beat llama.cpp HIP everywhere, the only current throughput miss is 512/128 prefill: about +9.9% needed.

### hipENGINE vs llama.cpp Vulkan

| Workload | Prefill delta | Decode delta | Memory |
| --- | ---: | ---: | --- |
| 512/128 | +22.0% | -14.4% | llama.cpp split rows have no retained memory |
| 4K/128 | +46.9% | -8.4% | — |
| 32K/128 | +67.1% | -4.2% | — |
| 128K/128 | +108.6% | -5.3% | — |

Vulkan is the decode target. To beat it everywhere, hipENGINE needs roughly +16.9% decode at 512, +9.1% at 4K, +4.4% at 32K, and +5.6% at 128K.

## Strategy in one paragraph

Do not start with another blind kernel multiloop. First promote the measurement harness and collect clean ROCTX/`rocprofv3` profiles for the exact comparison rows. The prefill miss is mostly short/mid-context and likely comes from bulk dense/shared-expert work where the parent uses framework GEMM-style paths while hipENGINE still uses row/GEMV-style custom kernels. The decode miss is smaller but hits every parent/Vulkan row; it likely needs compound wins from replay-only dispatch reduction, rotation/projection fusion that preserves fast layouts, and targeted full-attention decode work only where profiles show attention dominates. Memory is currently a strength; every candidate must preserve the <24 GiB PARO usability posture for 512/4K and must not reintroduce duplicate full-model W4 layouts.

## Non-negotiable promotion gates

1. **Correctness first.** The relevant fixture gates must pass before a number is retained. For this path that means at least:
   - `python3 scripts/qwen35_native_prefill_fixture_gate.py --max-layers 40 ...`
   - `python3 scripts/qwen35_decode_graph_fixture_gate.py --max-layers 40 ...`
   - and any new kernel-family CPU-reference or smoke gate from `docs/TESTING.md` / `docs/KERNELS.md`.
2. **No hidden torch in the hot path.** A profiler run may use Python to launch the benchmark, but `hipengine.LLM.generate()` and runtime hot modules stay torch-free.
3. **Registry, not backend branches.** New paths register under `(backend, layer, quant, variant)`. Do not add `if backend == ...` or `if quant == ...` in engine/model dispatch.
4. **Memory budget.** Default 512/128, 4K/128, and 4K/4K PARO rows should stay below 24 GiB peak. Long-context rows may exceed that only when explicitly labeled W7900 diagnostics; current chunked 128K/128 is already below 24 GiB and should not regress casually.
5. **A retained perf row updates the rollup.** Any retained benchmark updates `benchmarks/README.md`, `benchmarks/CHANGELOG.md`, and a compact artifact in `benchmarks/results/`.

## Lane 0 — make the scoreboard promotion-ready

The current comparison table is useful, but it is still backed by resident-runner diagnostics with `performance_claim=false`. Before claiming we beat anything, close the protocol gap.

### 0.1 Promote the Qwen3.5/PARO command path

Goal: produce accepted `LLM.generate()` or explicitly approved equivalent rows for the same comparison workloads.

Acceptance:

- Same model, quant, hardware, prompt/decode shapes, AOTriton threshold, decode graph replay, and chunk policy as the comparison script.
- Generated sample / fixture gates remain green.
- Artifact schema follows `docs/BENCHMARK.md`; if repeated runs are too expensive, the artifact says why and keeps diagnostics separate from accepted claims.
- `benchmarks/README.md` moves the first hipENGINE row out of "Blocked / diagnostic benchmark attempts" only when the protocol is satisfied.

### 0.2 Keep comparison-table data single-sourced

`scripts/qwen35_compare_tables.py` is intentionally hardcoded. It should still be refreshed whenever the retained current rows move.

Acceptance:

- Script rows match the retained artifact.
- `python3 scripts/qwen35_compare_tables.py all` remains the human checkpoint after each optimization lane.

## Lane 1 — audit before touching kernels

### 1.1 Collect matched hipENGINE profiles

Run replay/chunk/AOTriton rows with ROCTX so profiles can separate build, prefill, warmup, capture, and measured decode.

Command template:

```bash
COMMON="--token-id 9707 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build \
  --attn-aotriton-min-tokens 512 --graph-replay-decode \
  --prefill-linear-chunk-size 1024 --prefill-moe-chunk-size 1024 \
  --prefill-full-attn-query-chunk-size 4096 \
  --prefill-full-attn-post-chunk-size 1024 \
  --prefill-full-attn-rope-chunk-size 1024 --roctx"

rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-profile/qwen35-512 -- \
  python3 scripts/qwen35_paro_bench.py $COMMON --prompt-length 512 --json /tmp/qwen35-512-profile.json
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-profile/qwen35-4k -- \
  python3 scripts/qwen35_paro_bench.py $COMMON --prompt-length 4096 --json /tmp/qwen35-4k-profile.json
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-profile/qwen35-32k -- \
  python3 scripts/qwen35_paro_bench.py $COMMON --prompt-length 32768 --json /tmp/qwen35-32k-profile.json
```

Retain only compact summaries under `benchmarks/results/`; raw CSVs stay in `/tmp`.

### 1.2 Profile questions to answer

For each workload, summarize:

| Question | Why it matters |
| --- | --- |
| What are the top kernel buckets in prefill and measured decode separately? | Prevents another 49-iteration local optimum on the wrong metric. |
| Are linear-attention A/B and shared-expert prefill kernels visible as row/GEMV-style work? | Confirms or falsifies the leading short-prefill hypothesis from `docs/PREFILL.md`. |
| How many dispatches per measured decode token remain under hipENGINE graph replay? | `docs/ROOFLINE.md` treats dispatch/fusion as a first-order decode lever. |
| Which decode bucket explains the Vulkan gap at 512/4K? | Vulkan's biggest advantage is c=1 decode, not prefill. |
| At 32K/128 and 128K/128, is grouped-GQA attention still the first positive decode bucket? | Determines whether to spend the next long-context pass on attention or W4/rotation. |
| Are any hot kernels spilling (`Scratch_Size > 0`) or under-occupying CUs? | Required before shape/thread-count changes. |

### 1.3 Matched parent/source context

Use parent docs as steering evidence, not as proof that hipENGINE has the same bottleneck:

- `~/amd-gpu-tuning/docs/OPTIMAL.md`: parent flags, parent rows, long-prefill chunk overrides, graph-replay decode notes.
- `~/amd-gpu-tuning/PLAN-PAROQUANT2.md`: Amdahl postmortem; W4 inner-loop rewrites alone did not move E2E because launch/fusion and non-W4 buckets dominated.
- `~/amd-gpu-tuning/PLAN-LONGCONTEXT.md`: grouped-GQA split-K decode successes, split-cap retune, and rejected address-only V-loop polishing.
- `~/amd-gpu-tuning/LESSONS-LEARNED.md`: audit-first rules, correctness gates, layout-before-dot-intrinsics, LDS caution, graph state contract.
- `~/amd-gpu-tuning/docs/LLAMACPP-VULKAN.md`: Vulkan decode wins come from workgroup shape, ACO scheduling, activation packing, and graph-level fusion; not generic backend magic.

## Lane 2 — short/mid prefill gap: bulk dense and shared expert

This is the highest-probability lane for beating `nano-vllm-amd` and llama.cpp HIP prefill at 512/128 while keeping our long-context memory advantage.

### Hypothesis

After AOTriton closes the old 4K full-attention cliff, the residual prefill gap is not the quadratic attention core. Source and ledger evidence in `docs/PREFILL.md` point at per-layer non-attention bulk work:

| Area | Parent/source-lineage behavior | Current hipENGINE concern |
| --- | --- | --- |
| Linear-attention A/B projections | Parent multi-row path falls through dense `F.linear(...)`/GEMM-style work. | hipENGINE uses row/column dense GEMV wrappers for multi-row prompt work. |
| Shared expert prefill gate/up/down | Parent multi-row shared expert uses dense `F.linear(...)` paths. | hipENGINE uses custom W8A16 row/column kernels for all token counts. |
| Routed MoE compact WMMA | Parent compact WMMA route. | hipENGINE route is ported and chunked; less likely first gap unless profile says otherwise. |
| Full-attention core | Parent SDPA; hipENGINE AOTriton V3. | Old cliff is fixed; remaining cast/post-pass changes were neutral/slightly negative. |

### 2.1 Build a torch-free bulk dense path

Options, in preferred order after profile confirmation:

1. **hipBLAS/rocBLAS ctypes wrapper** for FP16/BF16 GEMM-shaped prefill work.
   - Good first experiment because parent's advantage here is likely GEMM/Tensile-style bulk execution.
   - Keep it behind kernel/linear plugin registration; do not import torch.
2. **Custom tiled WMMA kernels** for fixed Qwen3.5 shapes if BLAS call overhead/layout conversion is too high.
   - Use only for `tokens >= 64` or a measured threshold.
   - Keep existing GEMV paths for short/no-op and c=1.
3. **Hybrid dispatch threshold** based on measured crossover.
   - Parent compact WMMA crossover is ~64 prompt tokens for MoE; dense/shared may have a different threshold.

Initial target families:

- Linear-attention A/B dense projections.
- Shared expert gate/up + SiLU and shared down/combine during prefill.

Acceptance:

- Fixture gates pass for default/no-op chunk and actual chunk rows.
- 512/128 prefill improves by at least +8% without decode regression >1%.
- 4K/128 prefill improves or stays within noise; 32K/128 and 128K/128 stay finite and below current memory unless explicitly blocked.
- Profile after the change shows the targeted bucket moved, not just noise.

### 2.2 Then apply low-risk prefill fusions

Only after the bulk dense/shared gap is measured or falsified:

| Candidate | Expected benefit | Guardrail |
| --- | --- | --- |
| GDN RMSNorm+gate + PARO rotate tail fusion | Removes one launch and `recurrent_bf16` materialization across 30 linear-attention layers. | Keep the existing two-kernel path as fallback; require layer/fixture equality. |
| Prefill-only router shared-gate sigmoid | Lets grouped prefill skip a separate shared-gate sigmoid kernel. | Do not change c=1 decode, where combine kernels expect raw shared-gate logits. |
| AOTriton Q/gate+K prelude fusion | Removes split/cast/headnorm/RoPE glue and some scratch in the AOTriton path. | Previous AOTriton cast/gate fusions were throughput-neutral; require profile proof before spending more than a small spike. |
| Compiler profile experiment (`-mllvm -amdgpu-unroll-threshold-local=600`) | llama.cpp HIP saw large prefill gains from this flag. | Treat as a measured build-profile experiment, not a default; compare exact same kernels and check decode/memory. |

## Lane 3 — decode gap: compound dispatch, rotation, and W4 wins

We need decode +2% to +6% versus parent depending on context and +4% to +17% versus llama.cpp Vulkan. No single small tweak is likely enough at 512; plan for compound wins.

### 3.1 Start with replay-only decode profile

Do not use mixed traces that include eager validation decodes as the serving-loop source of truth. For hipENGINE, profile the measured graph-replay window and count:

- dispatches/token;
- per-family dispatch count;
- kernel time per token;
- inter-kernel gaps when trace timestamps allow;
- top buckets after subtracting prefill/warmup/capture ranges.

The current `docs/ROOFLINE.md` parent-derived steering signal is about 894 replay-path dispatches/token after projection fusions. Confirm hipENGINE's number before choosing fusion work.

### 3.2 Fusion rules for decode

Good decode fusion removes a launch **and** preserves or improves the memory layout. Bad fusion saves a launch by falling back to a slower qweight/activation layout.

Prioritize:

1. **Same-input projection fusion** where not already landed.
   - Existing wins include linear-attention QKV/Z and full-attention Q/K pack8 fusion.
   - Re-profile for remaining same-input pairs before writing code.
2. **Rotation/projection boundary fusion** when the rotated activation is consumed once by the following fast-layout projection.
   - Do not recompute a shared rotation per output pack.
   - Do not abandon the repacked/pack8 fast path to save one rotation launch.
3. **RMSNorm/add-RMSNorm producer fusion** only where the normalized vector is single-use.
   - Avoid folding a row-wide norm into every output-pack GEMV block.
4. **MoE post-op consolidation** that keeps selected-expert W4 GEMV grid parallelism.
   - Router logits + top-k cannot naively fuse into one block per token; that would recreate under-occupancy.

Acceptance:

- 512/128 decode improves by at least +3% per retained unit, or a smaller improvement composes with a clearly measured bucket reduction.
- 4K/128, 32K/128, and 128K/128 do not regress beyond noise.
- Graph replay fixture gate passes and graph state contract remains explicit.

### 3.3 W4 GEMV / Marlin-K caution

Parent `PLAN-PAROQUANT2.md` is a warning: many W4 inner-loop rewrites passed correctness but did not improve E2E. The pack8 kernel had a launch/fixed-cost floor; `sudot4` with in-kernel activation quantization regressed badly.

Do not port or default a Marlin-K/dot4 lane just because it looks theoretically better. Reopen W4-layout work only if the hipENGINE profile shows W4 GEMV is the first decode bucket after launch/fusion cleanup, and then isolate:

- activation pre-quantization once per residual vs in-kernel Q8 staging;
- no duplicate full-model qweight residency;
- kernel-level speed and E2E speed separately;
- no peak-memory regression.

If a qweight-neutral aliasing path is used, represent aliases as non-owning tensors. Never create two owning `DeviceTensorAllocation` records for the same pointer.

### 3.4 Long-context decode attention lane

At 32K/128 and 128K/128, the gap to parent/Vulkan is only ~4-6%, but attention can dominate long-context decode. hipENGINE already has the parent grouped-GQA split-K family wired; next work should be profile-triggered.

Valid next attention work:

- Confirm grouped-GQA split-K producer remains the hot positive bucket at 32K/128 or 128K/128.
- Prototype an exact BF16 online/tiled grouped-GQA producer only behind a fixture gate.
- Preserve the current split partial/reduce ABI at first.
- Read llama.cpp HIP/Vulkan `-fa` and hipfire tile structures for layout, but keep Qwen3.5 fixed-shape HIP code and current fallback.

Do **not** repeat rejected parent work without a structural change:

- address-only V-loop polishing;
- unroll-only grouped-GQA loops;
- thread-count sweeps around the same producer;
- Q-register local cache trials;
- split-reduce-only polish while reduce is a tiny bucket.

## Lane 4 — memory stays a feature, not a casualty

hipENGINE currently beats the parent peak-memory row on all retained comparison contexts and fixes the 128K scratch OOM with chunking. Preserve that advantage.

Rules:

- Keep chunked long-context policy as the default comparison policy until a better allocation plan exists.
- Keep AOTriton optional; code default remains `attn_aotriton_min_tokens=0` unless packaging makes the runtime dependable.
- No default duplicate expert layout or duplicate full-model qweight layout.
- Any new BLAS/WMMA bulk path must state its extra scratch and peak high-water mark in the artifact.
- Memory wins count only with the same workload, quant, and correctness gates as speed rows.

Near-term memory checks:

| Check | Why |
| --- | --- |
| 512/128 and 4K/128 peak under 24 GiB after each lane | Product usability gate. |
| 32K/128 peak stays near current ~20.7 GiB | Ensures chunking did not regress. |
| 128K/128 peak stays below 24 GiB if possible | Current hipENGINE row is a differentiator versus parent `27.42 GiB`. |
| Alias ownership tests for qweight views | Avoid double-free if qweight-neutral layouts are introduced. |

## Lane 5 — after batch-1: c>N and serving throughput

The current comparison plan is batch-1. After the batch-1 board is green, the next user-visible unlock is c>N decode and continuous batching.

Current state from `benchmarks/README.md`:

- c=2/4/8 generated equality exists.
- Native compact prefill equality exists.
- Decode remains serial in `step_batch_serial()`; c-aware decode graph replay is not wired.

Next serving lane:

1. Define c=2/4/8 `512/128` benchmark rows using `docs/BENCHMARK.md` c=N protocol.
2. Add c-aware decode graph buckets with fixed active-slot metadata.
3. Reuse grouped/compact prefill slab work and per-slot KV spans.
4. Report aggregate and per-request tok/s; do not compare aggregate c>N to c=1 without ratios.

This lane should not block batch-1 parent/Vulkan closure unless a profile shows batch-1 launch overhead cannot be solved without a batched/fused decode shape.

## Do-not-chase list

Do not spend a new optimization loop on these without new profile evidence:

| Avoid | Reason |
| --- | --- |
| More AOTriton cast/gate glue tweaks as the first prefill lane | Existing Q/K/V cast and gate+rotate artifacts saved memory but were throughput-neutral/slightly negative. |
| Native prefill attention rewrite before profile says AOTriton is the bottleneck | AOTriton closed the old 4K cliff; residual prefill gap now points elsewhere. |
| Naive `sudot4`/dot4 over current PARO layout | Parent experiments regressed; layout and activation staging dominate. |
| LDS staging as a default hypothesis | RDNA3 parent evidence repeatedly found barrier/occupancy costs exceeding reuse benefits. |
| Multi-step graph replay | Parent tested it and did not promote it; one-step replay is the retained shape. |
| Thread-count sweeps without source/profile justification | Many were neutral or regressed; launch bounds and LDS scratch must be checked first. |
| Fusion that abandons pack8/repacked fast layout | Saving one launch can lose more in memory layout. |
| Address-only V-loop polish for long attention | Parent rejected it; next attention attempt needs real online/tiled or parallel accumulation structure. |
| Perf rows without generated-token/logit sanity | Previous fast rows were invalid when recurrence/RoPE/state was wrong. |

## First concrete punchlist

1. **Profile summaries.** Produce compact profile summaries for hipENGINE 512/128, 4K/128, and 32K/128 with the current comparison policy.
2. **Prefill bulk path decision.** Based on the profiles, either:
   - implement a torch-free bulk dense/shared-expert GEMM path, or
   - explicitly mark the hypothesis falsified and update this file with the new top prefill bucket.
3. **Decode replay profile.** Produce a replay-only dispatch/bucket table for 512/128 and 4K/128.
4. **First decode fusion.** Pick one fusion that removes real dispatch/memory traffic while preserving fast layout; validate with graph fixture and comparison rows.
5. **Accepted-row promotion.** Once a retained improvement is real, promote the corresponding hipENGINE row through the benchmark contract and refresh `scripts/qwen35_compare_tables.py` data/artifact.
6. **Re-score the board.** Run:

   ```bash
   python3 scripts/qwen35_compare_tables.py all
   ```

   The batch-1 board is green when:
   - prefill beats parent and llama.cpp HIP/Vulkan at 512/4K/32K/128K,
   - decode beats parent and llama.cpp HIP/Vulkan at 512/4K/32K/128K,
   - peak memory remains below parent where parent memory is known and below the PARO usability gate for short/mid contexts.

## Reference map

| Topic | Primary reference |
| --- | --- |
| Parent optimal flags/rows | `/home/lhl/amd-gpu-tuning/docs/OPTIMAL.md` |
| Current hipENGINE prefill diagnosis | `docs/PREFILL.md` |
| Kernel catalog and port gates | `docs/KERNELS.md` |
| Benchmark rules | `docs/BENCHMARK.md` |
| Current comparison rows | `benchmarks/results/2026-05-16-hipengine-qwen35-comparison-tables-diagnostic.json` |
| llama.cpp HIP/Vulkan comparison rows | `/home/lhl/amd-gpu-tuning/PLAN-LONGCONTEXT.md` |
| Vulkan-vs-HIP source analysis | `/home/lhl/amd-gpu-tuning/docs/LLAMACPP-VULKAN.md` |
| RDNA3 performance model | `docs/ROOFLINE.md` and `/home/lhl/amd-gpu-tuning/docs/ROOFLINE.md` |
| Parent negative evidence / gotchas | `/home/lhl/amd-gpu-tuning/LESSONS-LEARNED.md`, `/home/lhl/amd-gpu-tuning/PLAN-PAROQUANT2.md` |
