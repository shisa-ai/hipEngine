# hipEngine Roadmap

_Last updated: 2026-06-17_

This document is the strategic planning layer above `docs/PLAN.md` (architecture),
`docs/ROOFLINE.md` (performance model), `docs/BENCHMARK.md` (evidence policy),
and `docs/CONCURRENCY.md` (serving path). It records **where we choose to
compete**, **where we choose not to**, and the concrete next phases. Architecture
invariants and evidence rules in those docs still govern; this doc only sets
priorities and rationale.

When this doc and `docs/PLAN.md` conflict on *what to build next*, this doc wins.
When they conflict on *architectural invariants*, `docs/PLAN.md` wins.

## 1. Executive summary

hipEngine on W7900/gfx1100 is competitive with — but not clearly ahead of —
llama.cpp on c=1 token throughput, while holding a structural, large lead on
concurrent serving throughput and on agentic/structured-decoding workloads.
Two strategic shifts are now in motion:

1. **GGUF becomes the default quant path.** Custom PARO/AWQ packs remain as a
   proven high-performance reference for gfx1100, but new models and new
   deployments target GGUF (`Q4_K`, `Q3_K_L`, `Q8_0`, …) so users can consume
   the same quant files the llama.cpp ecosystem already produces. This trades a
   measured c=1 perf advantage for portability, ecosystem compatibility, and a
   single quant story across backends.
2. **StepFun 3.7 on gfx1151 (Strix Halo) is the next model+backend focus.**
   The gfx1151 backend is registered but not yet stood up; StepFun 3.7
   correctness (text-only, GGUF `Q3_K_L`, greedy decode) is the vehicle for
   standing it up, deferring perf, vision, NVFP4, and MTP until base decode is
   correct.

The defensible long-term differentiators are **concurrent serving throughput**,
the **native on-device sampler + agent runtime contract**, and **multi-backend
portability** (gfx1100 + gfx1151 from one four-axis registry) — not the c=1
kernel-speed race, which is compiler-disadvantaged on HIP and diminishing-returns.

## 2. Where we stand (W7900 / gfx1100, Qwen3.6-35B-A3B)

Retained and diagnostic rows from `benchmarks/README.md`:

| Axis | hipEngine | Best RDNA3 peer | Read |
|---|---|---|---|
| c=1 decode, GGUF Q4 | 86–109 tok/s across shapes | llama.cpp Vulkan Q4: 98–128 tok/s | Vulkan ~11–16% ahead on c=1 |
| c=1 decode, PARO W4 | ~100 tok/s (17% of 864 GB/s ceiling) | — | PARO W4 ≈ W8A8; "W4=2×W8" has not materialized |
| c=8 aggregate (512/128, diag) | 190 tok/s | llama.cpp Vulkan: 25; vLLM: 117 | **7.5× / 1.6× diagnostic lead** |
| Agentic sampling, eager | native bounded top-k 94.6 tok/s | host-logits top-k 32.2 tok/s | **2.94× host readback avoidance** |
| MTP/DFlash (35B-A3B) | 1.023× AR / 0.30–0.40× | — | Break-even or worse on this model |

Two uncomfortable facts we are not papering over:

- **llama.cpp Vulkan beats hipEngine GGUF on c=1 at every shape.** Per
  `docs/ROOFLINE.md` §9.6/§9.7 this is an ACO compiler-scheduling + subgroup-64
  advantage on the same ISA, not a kernel-authoring gap; no free compiler flag
  remains.
- **PARO W4 achieves only ~17% of its bandwidth ceiling** and has converged to
  W8A8-level tok/s. The "W4 should be 2× W8" thesis was not realized because W4
  GEMV is only 33.8% of decode time (Amdahl) and the dp4a path needs a layout
  rewrite that has not landed.

## 3. Strategic axes: where we compete, where we don't

### 3.1 c=1 decode speed — selective, not primary

`docs/ROOFLINE.md` §10/§11 ranks the levers and records what is falsified. The
honest position:

- **The only c=1 lever with >15% headroom is the W4 dp4a block-aligned layout
  rewrite** (§10 Priority 1: block-aligned layout + `v_dot4_i32_iu8` +
  activation reuse, expected ~1.20× E2E → ~120–135 tok/s). It is a weight-repack
  + kernel rewrite touching the four-axis registry, not a patch. The naive dp4a
  attempt is in §11 as falsified; the retry must use a fundamentally different
  (block-aligned) framing.
- **Everything else in §11 is closed**: wave32/no-LDS GEMV, LDS staging,
  multi-step graph replay, WMMA at c=1, forced paged attention at 512, dynamic
  LDS, the `-amdgpu-unroll-threshold` flag. Do not relitigate.
- **Fusion/megakernel economics only flip positive at c>1.** The fused FFN
  megakernel was 2.66× slower at c=1 (`docs/MEGAKERNEL.md`); at c=1 the wide
  selected-FFN kernels already fill the GPU and fusion adds register pressure.
  Fusion work is justified as part of the c>1 path (§3.2), not as a c=1 win.

**Decision:** c=1 kernel work is bounded to (a) the GGUF quant plugins needed
for the new default path, and (b) an optional dp4a layout rewrite for gfx1100
PARO if we want to retake short-context. It is not the primary axis and should
not block the GGUF or gfx1151 transitions.

### 3.2 Concurrent serving throughput — the primary moat

This is the structural differentiator. c>1 moves decode from bandwidth-bound
toward compute-bound, reusing the 1.5 GB/tok weight reads across sequences —
exactly where the W7900's 48 GB and the roofline's "200+ tok/s requires c>1"
conclusion (`docs/ROOFLINE.md` §3.2) point. hipEngine's continuous batching,
per-row sampler, native batched projections, and elastic KV pool already realize
this; llama.cpp's Vulkan serving layer collapses under concurrency on this
backend (106 → 25 aggregate at c=8).

The gap is that c=N is **not yet a retained claim** on W7900. Per
`docs/CONCURRENCY.md` the host-side scaffolding is mostly in place, but the
retained-gate evidence is not:

- Generated-token equality is green for c=2/4/8 on Qwen/PARO BF16, but the
  retained decode path is still labeled experimental pending **native
  projection/dispatch closure**, **full-native c4/c8 attention** (no rowchunk
  diagnostics), **graph-replay profiler evidence**, and **scaling evidence**.
- A recent W7900 c=8 rerun hit a HIP OOM that must be diagnosed and fixed
  before c=8 can be retained on this hardware.

**Decision:** hardening c=N (c4/c8) to retained status on W7900 is the
highest-leverage single piece of work in the gfx1100 tree. It converts a 7.5×/
1.6× *diagnostic* into a *retained serving-throughput claim* and is where the
roofline says 200+ tok/s lives. See Phase A.

### 3.3 Native sampler + agent runtime — differentiator no RDNA3 peer has

The 2026-06-16 native sampler diagnostics are the punchline: host-logits top-k
eager is 32.2 tok/s; native bounded top-k eager is 94.6 tok/s (2.94×) because
it avoids a full-vocab D2H readback per token. The native GPU sampler keeps
suppress-token masks, min-token/EOS policy, bounded top-k/top-p/min-p, and
bounded top-logprobs on device — exactly the primitives agentic and
structured-decoding workloads need (grammar constraints, tool-call token
forcing, stop sequences). The `docs/AGENTIC.md` agent-runtime contract (bounded
budgets, resumable generations, session/cache control, cancellation/deadlines,
`<think>` splitting, tool-call streaming) is a coherent "local agent runtime"
thesis that neither llama.cpp nor vLLM targets on RDNA3.

**Decision:** continue promoting native sampler coverage (full-vocab paths,
true batched c>N sampling) and the agent-runtime contract. This is the
workload-shape moat and compounds with the concurrency moat (concurrent
agentic requests on one GPU).

### 3.4 Speculative decode — model-class strategy, not a universal win

`docs/SPECULATIVE-DECODE.md` is explicit: 35B-A3B's 256-expert / top-8 /
sequential dispatch makes `verify(B) ≈ B × verify(1)`, breaking the
"verification is free" assumption. MTP landed at 1.023× AR; DFlash on 35B-A3B
is 0.30–0.40× (drafter-bound). Speculative only wins on the **27B dense**
target (DFlash 1.23×, MTP llama.cpp 1.59×).

**Decision:** keep the MTP/DFlash infrastructure; deploy it behind dense
targets where it wins. Do **not** market speculative decode as a 35B-A3B
differentiator. The active `m12-batched` multiloop (batched verifier economics)
is the right shape of work *for the dense model class*; do not extrapolate its
results to 35B-A3B.

### 3.5 DMS / KV compression — deferred on 35B-A3B

Per `docs/KVCACHE.md`: only 10 full-attention layers own dense KV; at 128K BF16
KV is ~2.7 GB (INT8: 1.35 GB). At 4K, KV read is ~84 MB/token — not the
bottleneck; attention is 18.6% of decode and mostly occupancy/reduction, not
raw KV bytes. DMS only matters at 32K+ where attention becomes 40–55%, and even
there the limiter (`docs/ROOFLINE.md` §7.3) is the grouped-GQA context
producer, not KV row count. INT8 KV already buys the capacity win at -3% speed.

**Decision:** DMS/compact KV is a long-context capacity/efficiency play for a
future model where attention dominates and INT8 KV is insufficient. FastDMS
stays a read-only reference. Do not prioritize a DMS port for 35B-A3B.

## 4. The GGUF transition

### 4.1 Why GGUF as the default

Custom PARO/AWQ packs gave us a measurable c=1 edge and proved the four-axis
registry + raw-pointer-kernel architecture. But they are a proprietary
on-ramp: every new model needs a custom repack, users cannot reuse the
ecosystem's quant files, and the perf advantage is narrowing against Vulkan.
GGUF is the lingua franca of the local-LLM ecosystem; defaulting to it buys:

- **Portability and ecosystem reuse** — users drop in the same `Q4_K_M`,
  `Q3_K_L`, `Q8_0` files llama.cpp produces.
- **A single quant story across backends** — GGUF quant plugins are
  backend-agnostic at the registry layer, which matters the moment gfx1151
  lands (§5).
- **A cleaner quant-plugin boundary** — file/checkpoint layout → host repack →
  explicit device layout → raw-pointer kernel → registry dispatch, the same
  pattern PARO already proved (`docs/GGUF.md`).

### 4.2 Current state (`docs/GGUF.md`)

GGUF intake is mature for Qwen3.5/3.6: file scanning, lazy tensor views,
CPU-dequant fallback, native GEMV correctness for `Q8_0`/`Q5_K`/`Q6_K`/`Q4_K`
(+ lossless PARO-style pack8 repack for `Q4_K`), full model materialization
and E2E correctness for `Q4_K_M`/`Q8_0`/`Q4_1`/`UD-Q4_K_XL`, resident decode,
all-GPU full attention, AOTriton prefill, decode graph replay with GPU
sampling, and dense-BF16 fallback for `Q4_1`/`F16`/`IQ4_XS`. The
`qwen35moe` GGUF public-generation bring-up works for the 35B-A3B UD-Q4_K_M
file. Remaining gaps: public full-model bulk prefill, retained throughput
parity rows vs the PARO path, and deeper WMMA/Marlin-style tuning.

### 4.3 What "shoring up GGUF" means concretely

- **Retained GGUF benchmark rows** for the Qwen targets at the standard shapes,
  recorded under `benchmarks/results/` with the full evidence policy
  (`docs/BENCHMARK.md`). GGUF is not "the default" until a retained row exists
  at each shape and the rollup (`benchmarks/README.md`) reflects it.
- **Q3_K_L coverage** — required for the StepFun 3.7 target (§5) and a
  generally useful long-context quant. Add the quant plugin + CPU oracle + HIP
  kernel following the `Q4_K` pattern.
- **Bulk prefill parity** for GGUF so prefill is not a regression vs the PARO
  path.
- **Pack8 sidecar cache** (Task #59 lineage) promoted from opt-in to default-on
  for the rank-3 expert tensors where it helps, behind the JIT cache contract.
- **GGUF as the documented default** in `docs/GGUF.md` and the public API once
  the retained rows land; PARO/AWQ stays as an opt-in high-performance path for
  gfx1100 users who want the c=1 edge.

GGUF must remain torch-free on the generate path and must not introduce
backend/quant special-casing in dispatch/engine/model code — it plugs in via
the four-axis registry like every other quant.

## 5. StepFun 3.7 on gfx1151 — the next model+backend focus

### 5.1 Why gfx1151, why now

gfx1151 (Strix Halo / Radeon 8060S, RDNA3.5) is the second backend in the
four-axis registry and the path to a portable, multi-backend engine. It is
registered but not yet stood up — no gfx1151-native kernels exist. Standing it
up via a real model forces the backend-peer discipline the architecture
demands (no "AMD directory", gfx1100 and gfx1151 are siblings; `docs/PLAN.md`
Architectural Invariants).

StepFun 3.7 is the vehicle: a current, relevant model whose GGUF `Q3_K_L`
text-only target exercises the new GGUF default path on the new backend
simultaneously. Strix Halo's `docs/ROOFLINE-gfx1151.md` (256 GB/s LPDDR5X, 40
CUs, 32 MiB L3) makes c=1 decode a bandwidth+occupancy problem — a good stress
test for the GGUF quant kernels and a realistic deployment target (APU local
LLM).

### 5.2 Scope and non-goals (Phase C)

**In scope:** text-only GGUF `Q3_K_L` StepFun 3.7 greedy decode, correct, on
gfx1151. Correctness gate per `docs/TESTING.md` and `docs/KERNELS.md`: KL ≤ 0.05
and top-1 agreement ≥ 90% vs `kernels/cpu_reference/` on fixture inputs, plus a
`rocprofv3 --kernel-trace` smoke showing the kernel ran under the expected
name with plausible duration.

**Explicitly deferred** (recorded so they are not silently picked up):
performance optimization, NVFP4, vision components, and MTP. These are gated
behind a correct base greedy decode. The active `stepfun-gguf-correctness`
multiloop (punchlist mode) is tracking this against a P0–P12 bring-up
checklist (see Phase C).

### 5.3 What gfx1151 bring-up requires

- A gfx1151-native kernel set: the GGUF quant GEMVs (`Q3_K_L` first, then the
  Qwen-proven `Q4_K`/`Q5_K`/`Q6_K`/`Q8_0`), attention (paged-KV-write + decode
  reading the `KVLiveSpans` ABI), RMSNorm, router, and the PARO rotation/state
  primitives StepFun needs — all raw-pointer kernels registered against
  `(hip_gfx1151, layer, quant, variant)`.
- A StepFun 3.7 model plugin following the existing Qwen model-plugin pattern,
  with its tensor-name mapping, RoPE/attention shape guards, and layer
  classification (linear-attention vs full-attention blocks).
- The `Q3_K_L` quant plugin + CPU oracle shared across backends (gfx1100 +
  gfx1151), proving the registry gives quant reuse for free.
- HIP-availability guards on all new tests so no-ROCm CI/publish runners skip
  rather than fail (`AGENTS.md` "During Work").

## 6. Phased plan

Phases are priority-ordered, not strictly sequential: Phase A (gfx1100
concurrency) and Phase B (GGUF retention) can overlap; Phase C (gfx1151 /
StepFun) starts once the GGUF Q3_K_L quant plugin from Phase B is shippable,
because StepFun consumes it.

### Phase A — Bank the concurrency moat (gfx1100, highest leverage)

Goal: turn the 7.5×/1.6× c=N diagnostic into retained rows.

1. Diagnose and fix the W7900 c=8 HIP OOM; confirm c=4 and c=8 memory
   headroom under the elastic KV pool.
2. Close the native projection/dispatch blocker for c=2/4/8 (the
   `CONCURRENCY.md` QKV/Z batch-GEMV closure set).
3. Land full-native c4/c8 attention with no rowchunk diagnostics.
4. Produce graph-replay profiler evidence (prebuilt `.so`, `require_cached`,
   no `hipcc` under the profiler — `AGENTS.md`).
5. Produce scaling evidence (c=1→2→4→8 aggregate) and emit retained
   `benchmarks/results/` artifacts + rollup rows.

Exit gate: retained c=4 and c=8 rows in `benchmarks/README.md` with the
correctness gate green and profiler evidence attached.

### Phase B — GGUF as the default quant path

Goal: retained GGUF rows + Q3_K_L coverage + GGUF documented as default.

1. Add retained GGUF benchmark rows (Q4_K_M, Q8_0) at the standard shapes vs
   the PARO baseline.
2. Implement the `Q3_K_L` quant plugin + CPU oracle + HIP kernel (gfx1100),
   with a bit-exact RED test and the correctness gate. (Required input to
   Phase C.)
3. Close GGUF bulk-prefill parity vs PARO.
4. Promote the pack8 expert sidecar cache to default-on where it helps, behind
   the JIT cache contract.
5. Update `docs/GGUF.md` to mark GGUF as the default; mark PARO/AWQ as the
   opt-in high-performance gfx1100 path.

Exit gate: retained GGUF rows at every standard shape; `Q3_K_L` correctness
gate green on gfx1100; `docs/GGUF.md` reflects the default.

### Phase C — StepFun 3.7 on gfx1151 (next model+backend focus)

Goal: correct text-only GGUF Q3_K_L StepFun 3.7 greedy decode on gfx1151.

1. Track the StepFun P0–P12 bring-up punchlist (`docs/STEPFUN.md`, authored in
   a parallel branch) as the correctness checklist for this phase.
2. Populate the `hip_gfx1151` kernel tree: GGUF `Q3_K_L` GEMV first, then the
   Qwen-proven GGUF quants, then attention (paged-KV-write + decode via
   `KVLiveSpans`), RMSNorm, router, and the StepFun rotation/state primitives.
3. Add the StepFun 3.7 model plugin with tensor-name mapping and layer
   classification.
4. Pass the correctness gate (KL ≤ 0.05, top-1 ≥ 90%) vs `cpu_reference/` and
   the `rocprofv3 --kernel-trace` smoke.
5. Add HIP-availability guards on all new tests.

Exit gate: `docs/STEPFUN.md` P0–P12 green; StepFun 3.7 text-only greedy decode
correct on gfx1151 with profiler evidence. Perf, NVFP4, vision, MTP remain
explicitly deferred and tracked as follow-on phases.

### Phase D — Differentiators that compound (ongoing, parallel)

These run alongside A–C and are not gated by them:

- **Native sampler coverage** — full-vocab paths, true batched c>N sampling,
  on-device constraint primitives (grammar, tool-call forcing).
- **Agent runtime contract** (`docs/AGENTIC.md`) — bounded budgets, resumable
  generations, session/cache control, cancellation/deadlines, streaming.
- **Speculative decode for dense targets** — keep MTP/DFlash infra; ship behind
  the 27B-dense target where it is 1.23×. Do not extrapolate to 35B-A3B.

### Phase E — Optional c=1 retake (gfx1100 PARO, only if prioritized)

The dp4a block-aligned layout rewrite (`docs/ROOFLINE.md` §10 Priority 1),
expected ~1.20× E2E → ~120–135 tok/s, retaking short-context from Vulkan. This
is a weight-repack + kernel rewrite touching the four-axis registry. Pursue
only if c=1 leadership is strategically required; it is lower-leverage than
A–C and competes against a compiler advantage we cannot fully close.

## 7. Explicitly not chasing (this roadmap period)

Recorded so effort is not re-spent on closed questions (see
`docs/ROOFLINE.md` §11 for the full falsified list):

- Naive dp4a on the AWQ/pack8 layout (falsified; only block-aligned retry is
  open, and only as Phase E).
- Wave32/no-LDS GEMV, LDS staging, multi-step graph replay, WMMA at c=1,
  forced paged attention at 512, dynamic LDS, `-amdgpu-unroll-threshold` for
  decode.
- DMS/compact KV for 35B-A3B (KV too small to matter; revisit only with a
  long-context-heavy model where attention is >50% of decode and INT8 KV is
  insufficient).
- Speculative decode marketed for 35B-A3B (break-even or worse; dense-only).
- Megakernel/fusion work targeted at c=1 (economics only flip at c>1).

## 8. Cross-references

| Doc | Role for this roadmap |
|---|---|
| `docs/PLAN.md` | Architecture invariants, phase roadmap, extensibility design. |
| `docs/ROOFLINE.md` | gfx1100 performance model, Amdahl analysis, priority map, falsified list. |
| `docs/ROOFLINE-gfx1151.md` | gfx1151/Strix Halo performance model for Phase C. |
| `docs/CONCURRENCY.md` | c=N serving path, contracts, retained-gate requirements (Phase A). |
| `docs/AGENTIC.md` | Agent runtime contract and native sampler story (Phase D). |
| `docs/GGUF.md` | GGUF intake plan and current state (Phase B). |
| `docs/KERNELS.md` | Kernel catalog, correctness gate, JIT cache gotcha, build profiles. |
| `docs/TESTING.md` | RED/GREEN workflow, correctness oracles, validation matrix. |
| `docs/BENCHMARK.md` | Evidence policy and benchmark gates. |
| `docs/SPECULATIVE-DECODE.md` / `docs/MTP.md` / `docs/DFLASH.md` | Speculative decode economics (Phase D, dense-only). |
| `docs/KVCACHE.md` | KV/INT8/DMS rationale for deferral (§3.5). |
| `docs/STEPFUN.md` | StepFun 3.7 / gfx1151 punchlist (Phase C). |
