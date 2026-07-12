# hipEngine

hipEngine is a ROCm-native local LLM inference engine designed from the ground
up for AMD RDNA GPUs (starting with gfx1100, gfx1151). It pairs a small
purpose-built Python host with a complete suite of custom-tuned HIP kernels
developed through 100+ iterations of profiling and tuning.

hipEngine has lightweight dependencies with no PyTorch required for fully
supported GPUs and models.

## Core principles

- **HIP-first, not CUDA-ported.** Kernels directly target AMD hardware like 
  gfx1100/RDNA3 with wave32, vec8 FMA, and the actual cache hierarchy.
- **Torch-free runtime.** `import torch` is **not** on the hot path. The
  runtime owns a thin `hipengine.Tensor` over raw HIP/CUDA device pointers and
  drives `hipblasLt`, `hipGraph`, AOTriton, and JIT builds through `ctypes`.
  Torch appears only as an optional dlpack bridge behind the `hipengine[torch]`
  extra (~125 MiB install including the vendored AOTriton subset vs ~2 GiB with
  torch).
- **Multi-backend from day one.** Kernels live under `kernels/hip_gfx1100/`,
  `kernels/hip_gfx1151/`, `kernels/cuda_sm86/`, `kernels/cpu_reference/` as
  peer trees.
- **Four-axis plugin registry.** Kernels are keyed by
  `(backend, layer, quant, variant)`. Models, quant schemes, and layers are
  plugins. No `if backend == "..."` or `if quant == "..."` branches in
  dispatch / engine / model code.
- **Fused + unfused coexist.** Every fused composite
  (`rmsnorm+rotate`, `gate_combine_residual`, …) has a numerically-equivalent
  unfused chain registered under its primitives, used as both fallback and
  correctness baseline.
- **Evidence-backed performance.** Every performance claim ships with
  model + quant + workload shape + hardware + exact command + correctness gate
  (KL ≤ 0.05, top-1 ≥ 90% vs `kernels/cpu_reference/`). See
  [`docs/BENCHMARK.md`](docs/BENCHMARK.md) and
  [`benchmarks/README.md`](benchmarks/README.md).

## Status

**v0.2.2 alpha.** The runtime hot path is torch-free by construction, and the
first two 35B-class model-loading surfaces are available on gfx1100 and gfx1151:
[shisa-ai/Qwen3.6-35B-A3B-PARO-packed](https://huggingface.co/shisa-ai/Qwen3.6-35B-A3B-PARO-packed)
(19.07 GiB, 4.68 bpw) in packed
[ParoQuant](https://github.com/shisa-ai/paroquant) format, plus Qwen3.6 GGUF
`Q4_K_M` / `Q4_K_S` files through the resident GGUF path. Older benchmark
artifacts may still show the historical
`Qwen3.6-35B-A3B-PARO-full4096-e5-packed` name or local MTP-BF16 assembly path;
those rows use the same packed PARO architecture and remain the evidence for the
numbers below.

- INT8 KV cache support has been added for PARO. Qwen 3 MoE's full 256K context window can fit in <24GB tracked memory; see [Memory Usage](#memory-usage).
- The OpenAI-compatible server now has resident context/KV preallocation, startup warmup, max-prompt scratch probing, bounded chat-shaped startup smoke, `/ready` diagnostics, request context admission, and `max_tokens=auto` defaults for chat requests that omit an output cap.
- Non-streaming completion/chat responses carry exact generated token IDs and all-choice counts under `hipengine.token_accounting`; `usage.completion_tokens` uses those IDs instead of re-tokenizing decoded text whenever the generator provides them.
- Direct generation and non-streaming `/v1/completions` accept the same exact token-ID rows for PARO and GGUF. The server returns input hashes/counts under `hipengine.prompt_token_accounting`, and `scripts/exact_token_generation.py` gates HTTP output against a direct 512/128 generated-ID oracle.
- Choice telemetry declares timing scope, covered rows, and ownership; packed PARO/GGUF timing shares a stable batch ID so benchmark consumers count copied group walls once.
- Non-streaming server responses expose request-scoped route caps, queue request/prompt groups, actual backend call widths, and speculative verifier rows independently; the benchmark harness deduplicates the shape by queue-group ID, so client c8 cannot be mislabeled as a width-8 verifier run.
- New server, retained PARO, GGUF, and HIP/Vulkan micro artifacts share one torch-free provenance contract with a concrete resolved backend/arch/device, content-derived model fingerprint, exact command/toolchain, and separate staged, unstaged, and untracked source state.
- The gfx1151 GGUF eager gate proves that `[9707] * 512` legitimately continues with token `9707` in both llama.cpp and hipEngine; four teacher-forced transitions match fresh serial-prefix hidden, Conv/GDN, and live KV state byte-for-byte. This is a [correctness artifact](benchmarks/results/2026-07-11-sol-g1-gfx1151-gguf-eager-p512-d4.json), not a throughput claim.
- The gfx1151 GDN prefill [exact matrix](benchmarks/results/2026-07-11-sol-g2-gfx1151-gdn-prefill-exact-matrix.json) passes 6/6 short, 512, segment-threshold, 4K, and chunk-boundary cases. The clean [interleaved A/B](benchmarks/results/2026-07-11-sol-g3-gfx1151-gdn-prefill-interleaved-ab.json) then rejects split-chain promotion: it is 5.19% slower at 512 and 6.70% slower at 4K across four balanced repetitions, with exact timed tokens. Fused remains the default; the exact split stays available as the unfused diagnostic fallback.
- The gfx1151 GGUF production graph [audit](benchmarks/results/2026-07-11-sol-g5-gfx1151-gguf-decode-graph-production-audit.json) passes 128/128 byte-exact hidden, recurrent-state, KV, and token checkpoints. For a 128-token long-greedy window, capture-inclusive throughput improves same-run eager 49.178 -> 49.233 tok/s (+0.112%); shorter, sampled, streaming, c>N, INT8-KV, and gfx1100 routes remain eager.
- The clean gfx1151 GGUF [residency census](benchmarks/results/2026-07-11-sol-g6-gfx1151-gguf-residency-audit.json) finds 733 unique source tensors and no default raw+replacement duplicates or optional sidecars. A Q4_K_M p512/d128 BF16-KV graph session owns 21.478 GiB, leaving 2.522 GiB to the 24 GiB gate; the graph adds no tracked buffer and about 308 KiB of sampled HIP graph/exec residency.
- `LLM.stream()` and `stream=true` chat completions run token-level resident decode, with Qwen/DeepSeek-style `<think>...</think>` spans split into `reasoning_content` in both streaming and non-streaming responses.
- Qwen 3.6 [Q4_K_M](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF?show_file_info=Qwen3.6-35B-A3B-UD-Q4_K_M.gguf) and [Q4_K_S](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF?show_file_info=Qwen3.6-35B-A3B-UD-Q4_K_S.gguf) GGUF support has landed (W7900 Q4_K_M/Q4_K_S sweeps are in [Performance](#performance) alongside packed PARO and llama.cpp HIP/Vulkan Q4_K_M baselines). The Q4_K_M link points at Unsloth's MTP-bearing GGUF repo so the same filename also works for llama.cpp draft-MTP comparisons. GGUF uses a substantial GGUF-specific runtime path with bulk prefill, eager resident decode, and on-load decode-repack into T16 tile layouts. Q4_K_S is the lower-memory secondary file; Q4_K_M is the active 1:1 llama.cpp comparison target and current 24 GiB BF16-KV support is mid-context unless a lower-memory KV/weight policy is enabled. GGUF also has a higher per-session load cost (~60 s vs ~38 s for PARO packed on the same W7900/TheRock stack) for the same decode-repack reason.
- Current gfx1100 and gfx1151 performance snapshots are summarized in [Performance](#performance) with hardware-separated tables and recent llama.cpp baselines.


## Hardware targets

| Backend | Hardware | Status |
| --- | --- | --- |
| `cpu_reference` | Any CPU, numpy | Correctness oracle; CI without GPU |
| `hip_gfx1100` | AMD Radeon Pro W7900 / RX 7900 XTX (RDNA3) | Active backend |
| `hip_gfx1151` | AMD Ryzen AI MAX+ 395 / Radeon 8060S (Strix Halo, RDNA3.5) | Active backend |
| `cuda_sm86` | NVIDIA Ampere consumer (3090-class) | Planned peer backend |

`backend="auto"` is the public API/server default. It maps exact `gfx1100` and
`gfx1151` detections to the matching HIP backend; unknown ROCm targets warn and
select `cpu_reference` where a CPU implementation exists. Users on nearby targets
such as `gfx1101`/`gfx1102` can force a backend with `backend="hip_gfx1100"`,
`--backend hip_gfx1100`, or `HIPENGINE_BACKEND=hip_gfx1100` after validating
correctness/performance.

Wave32 is the default for `hip_gfx1100` device code; wave64 is treated as an
isolated experiment with its own gates (see
[`docs/PLAN.md`](docs/PLAN.md#rdna3-wavefront-and-scheduling-caveat)).

## Memory Diagnostics Awaiting Rerun

The 2026-05-19 W7900 diagnostic showed the packed Qwen3.6 PARO model fitting a
128K context with BF16 KV and 256K with INT8 per-token/per-head KV under 24 GiB
tracked allocator peak. The artifact records hipEngine `ae229513` and exact
commands, but not the compiler version; treat this as a stale capacity
diagnostic until it is rerun with the full provenance and Qwen3.6 long-rollout
quality gates.

<!-- BEGIN TOPLINE:W7900_MEMORY_CAPACITY -->
No eligible capacity row; the dated artifact remains linked below pending rerun.
<!-- END TOPLINE:W7900_MEMORY_CAPACITY -->

Regardless of the difference in PARO weight storage (legacy or packed),
loaded-weight memory is approximately 16.4 GiB in VRAM.

The INT8 KV correctness gate attached to this artifact is the deterministic
Qwen3.5 PARO fixture `fixtures/qwen35_paro/parent_512_32_seed1234.json`
(512-token prompt,
32 greedy decode tokens): `max_kl=0.015328`, `mean_kl=0.001639`, top-1 agreement
100%, and generated IDs match BF16 KV exactly. Layer attention probes at context
64 and 520 also had top-1 agreement 100% with max quantized-vs-BF16 KL
`2.34e-7`. This is a fixture/regression gate, not a long-rollout perplexity
study, so long context generations may have unmeasured compounding errors.

The separate 128K/128 Qwen3.5 BF16-vs-INT8 gate measured -0.99% prefill tok/s
and -3.20% decode tok/s for INT8 KV.

See
[`benchmarks/results/2026-05-19-hipengine-qwen36-packed-int8-kv-readme-memory-diagnostic.json`](benchmarks/results/2026-05-19-hipengine-qwen36-packed-int8-kv-readme-memory-diagnostic.json),
[`benchmarks/README.md`](benchmarks/README.md#w7900-paro-context-capacity-2026-05-19),
and [`docs/KVCACHE.md`](docs/KVCACHE.md) for commands, artifacts, and the full
no-shadow memory audit.

### llama.cpp configuration note

The repository has no compact artifact or source revision for the former
llama.cpp Q8_0 memory tables, so those numbers are not toplines. The tested
configuration was:

```bash
--flash-attn on -ctk q8_0 -ctv q8_0 -c 262144 -b 128 -ub 128
```

A replacement capacity table must record the GGUF fingerprint, llama.cpp
commit/build, GPU, full command, and whole-card sampling artifact.

## Model Performance

### gfx1100 (Radeon RX 7900 XTX / Radeon Pro W7900)

**Status: stale diagnostic.** The linked record is the last complete same-host sweep,
measured on 2026-07-07 at hipEngine `b4edca09` with TheRock HIP
`7.13.26162-1140233ffe`. Its top-level artifact sets
`performance_claim=false`. The GGUF path repeatedly selected token `9707`;
that stream is externally and state-certified on gfx1151, but this old gfx1100
run predates the hardware-local oracle and current provenance contract.
hipEngine PARO uses W4 PARO/BF16 KV; the other columns use Q4_K_M GGUF with
BF16/f16 KV, so maxima are not same-quant wins.

<!-- BEGIN TOPLINE:W7900_SWEEP -->
No eligible model-throughput row; the `performance_claim=false` sweep remains linked below pending correctness rerun.
<!-- END TOPLINE:W7900_SWEEP -->

W7900 row sources: [`summary`](benchmarks/results/2026-07-07-w7900-gpu0-readme-refresh-20260707-104756-summary.json), [`hipEngine PARO`](benchmarks/results/2026-07-07-w7900-gpu0-readme-refresh-20260707-104756-hipengine-paro-packed-5run.json), [`hipEngine GGUF`](benchmarks/results/2026-07-07-w7900-gpu0-readme-refresh-20260707-104756-hipengine-gguf-q4km-5run.json), [`llama.cpp HIP`](benchmarks/results/2026-07-07-w7900-gpu0-readme-refresh-20260707-104756-llamacpp-hip-q4km-f16kv.json), and [`llama.cpp Vulkan`](benchmarks/results/2026-07-07-w7900-gpu0-readme-refresh-20260707-104756-llamacpp-vulkan-q4km-f16kv.json). Exact settings and the refresh blocker are in the canonical [`benchmarks/README.md`](benchmarks/README.md#w7900-model-sweep-2026-07-07).

### gfx1151 (AMD Ryzen AI MAX+ 395 / Radeon 8060S)

**Status: retained.** The Strix Halo rows were measured on 2026-07-11 from a
clean detached worktree at `d1231ee0` with TheRock HIP
`7.13.60980-c76140fa27`. hipEngine uses two discarded warmups plus five
measured repetitions per right-sized resident shape; llama.cpp uses one
internal warmup plus five samples per split phase. The top-level artifact
passes clean provenance, stable finite-output, variance, model/build/device,
and memory-scope gates.

<!-- BEGIN TOPLINE:GFX1151_SWEEP -->
#### Prefill tok/s

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | 994.866 | 430.767 | 1061.260 | 1067.770 |
| 1K/128 | 810.029 | 437.467 | 1043.230 | 1069.870 |
| 4K/128 | 671.985 | 403.946 | 1009.240 | 1016.580 |
| 32K/128 | 599.063 | 369.942 | 743.547 | 814.923 |
| 64K/128 | 511.248 | 334.395 | 573.611 | 660.974 |
| 128K/128 | 375.635 | 270.601 | 390.441 | 476.788 |

#### Decode tok/s

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | 66.753 | 49.536 | 50.939 | 62.396 |
| 1K/128 | 61.628 | 52.192 | 50.818 | 62.136 |
| 4K/128 | 62.715 | 52.999 | 50.126 | 60.097 |
| 32K/128 | 50.362 | 43.947 | 44.240 | 51.319 |
| 64K/128 | 42.032 | 37.477 | 39.326 | 44.422 |
| 128K/128 | 30.316 | 27.862 | 32.114 | 34.948 |

#### Peak memory GiB

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | 18.144 | 21.478 | 21.375 | 21.551 |
| 1K/128 | 18.367 | 21.710 | 21.387 | 21.501 |
| 4K/128 | 19.161 | 22.995 | 21.444 | 21.507 |
| 32K/128 | 19.864 | 23.559 | 21.987 | 22.191 |
| 64K/128 | 20.403 | 24.203 | 22.666 | 22.627 |
| 128K/128 | 22.124 | 25.493 | 23.862 | 24.254 |
<!-- END TOPLINE:GFX1151_SWEEP -->

The memory columns have different scopes: hipEngine reports tracked allocator
high-water, while llama.cpp reports absolute whole-device amdgpu GTT used,
sampled every 10 ms. Use them for within-column context growth, not small
cross-column allocator comparisons. Row sources: [`accepted summary`](benchmarks/results/2026-07-11-gfx1151-readme-refresh-20260711-d1231ee0-summary.json),
[`hipEngine PARO`](benchmarks/results/2026-07-11-gfx1151-readme-refresh-20260711-d1231ee0-hipengine-paro-packed-5run.json),
[`hipEngine GGUF`](benchmarks/results/2026-07-11-gfx1151-readme-refresh-20260711-d1231ee0-hipengine-gguf-q4km-5run.json),
[`llama.cpp HIP`](benchmarks/results/2026-07-11-gfx1151-readme-refresh-20260711-d1231ee0-llamacpp-hip-q4km-f16kv.json), and
[`llama.cpp Vulkan`](benchmarks/results/2026-07-11-gfx1151-readme-refresh-20260711-d1231ee0-llamacpp-vulkan-q4km-f16kv.json). Exact settings and gates are in the canonical [`benchmarks/README.md`](benchmarks/README.md#gfx1151-model-throughput).

### Current gfx1151 GGUF decode baselines

These are separate exact repeated-token SOL-G4/G5 controls. The model sweep
above excludes graph capture from steady decode throughput; SOL-G5 charges one
capture/instantiate and destroy to each 128-token window.

<!-- BEGIN TOPLINE:GFX1151_GGUF_EAGER -->
| Path | Platform and protocol | Result | Evidence status |
| --- | --- | ---: | --- |
| GGUF eager c1 | Radeon 8060S/gfx1151; Qwen3.6-35B-A3B UD-Q4_K_M; BF16 KV; `[9707] * 512`; TheRock HIP 7.15; TuneD accelerator-performance; clean scalar/candidate/scalar, 1 discarded + 4 measured runs per leg; 128 eager steps; graph off | **48.850 tok/s** (`20.471 ms/token`), **+0.309%** vs clean scalar control | Retained for this exact repeated-token protocol; control/candidate ranges do not overlap, every output ID is 9707, and the G1 hidden/state/KV oracle is linked |
| GGUF state-bound graph c1 | Radeon 8060S/gfx1151; same current model/KV/prompt/stack; 1 warmup + 4 measured rotating same-session runs; 128 steps; capture and destroy charged | **48.704 tok/s** (`20.532 ms/token`), **-0.293%** vs same-run eager; **+0.201%** vs scalar graph | Exact 128/128 state/KV/token replay, but current G5 rejects a graph-over-eager speed claim; graph default policy is tracked separately |
<!-- END TOPLINE:GFX1151_GGUF_EAGER -->

Artifacts: [`SOL-G4 eager audit`](benchmarks/results/2026-07-11-sol-g4-gfx1151-gguf-eager-decode-audit.json)
and [`SOL-G5 production graph audit`](benchmarks/results/2026-07-11-sol-g5-gfx1151-gguf-decode-graph-production-audit.json).

See [`benchmarks/README.md`](benchmarks/README.md) for the platform freshness
index, exact settings, run commands, and evidence status.

## Speculative decode (DFlash / MTP)

Every displayed route has its own same-protocol AR control. The exact/default
and `llama-compat` columns are separate because only `llama-compat` shares the
B2 natural24 structure used by the llama.cpp comparison.

<!-- BEGIN TOPLINE:SPECULATIVE -->
#### GGUF MTP comparison, Radeon 8060S/gfx1151

| Metric | hipEngine GGUF exact/default | hipEngine GGUF `llama-compat` | llama.cpp HIP |
| --- | ---: | ---: | ---: |
| Headline route | B5, fixed 10 cycles | B2, natural24/cyclecap24 | B2, natural24 diagnostic |
| Headline MTP decode | 61.98 tok/s (1.1312x own AR) | 71.52 tok/s (1.3055x own AR) | 71.91 tok/s (1.3835x own AR) |
| Matched natural24 B2 MTP decode | 52.04 tok/s (diagnostic) | 71.52 tok/s | 71.91 tok/s |
| Matched natural24 own AR | 54.80 tok/s | 54.79 tok/s | 51.98 tok/s |
| Matched natural24 cycle wall/output | 19.248 ms | 14.005 ms | 14.269 ms |
| State/commit contract | exact/default, serial-prefix preserving | direct partial commit/dp4a; accuracy-traded | native llama.cpp compatibility target |

`llama-compat` is the closer 1:1 performance comparison with llama.cpp because
both use B2 and the natural24 output horizon. It is still not an exact semantic
comparison: `llama-compat` is not serial-prefix-equivalent. The llama.cpp HIP
column is a locally instrumented, dirty-source diagnostic
(`performance_claim=false`), not a promoted standalone topline. The exact B5
headline uses a different fixed-cycle horizon and must not be ranked directly
against the two natural24 columns; its matched natural24 B2 control is shown
separately.

#### Dense PARO DFlash

| Path | Platform and protocol | Result | Evidence status |
| --- | --- | ---: | --- |
| DFlash B=4 online-gated | W7900/gfx1100; Qwen3.6-27B PARO target plus Qwen3.6-27B DFlash drafter; 9 prompts; 64 decode tokens | 40.10 vs 32.57 AR tok/s, **1.231x** | Retained under the recorded DFlash gate; source tree was dirty and must be refreshed before changing the claim |
<!-- END TOPLINE:SPECULATIVE -->

Artifacts: [`DFlash`](benchmarks/results/2026-06-11-hipengine-dflash-27b-dense-hardening-rerun.json),
[`exact MTP`](benchmarks/results/2026-07-02-ar-mtp-default-parallelattn-full.json),
and [`llama-compat` MTP](benchmarks/results/2026-07-03-ar-mtp-llama-compat-directcommit-nocopy-natural24-cyclecap24-f32head-full.json).
Matched natural24 controls: [`exact/default B1-B5`](benchmarks/results/2026-07-03-ar-mtp-default-natural24-budget-sweep-c1.json)
and [`llama.cpp HIP B2`](benchmarks/results/2026-07-02-llamacpp-mtp-stage-timing-b2-natural24-rerun.json).
Historical hipEngine OpenAI MTP server rows are excluded. The current raw-ID
route counts exact completion IDs across every choice and owns batch timing
once. The
corrected 2026-07-11 server matrix finds that compatibility MTP changes true-AR
IDs even at c1, so it must remain explicit-only despite diagnostic c1/c2 speed
gains; SOL-S1 routes automatic requests to exact/default AR while keeping the
compatibility hook explicit-only. See the
[`route-gate artifact`](benchmarks/results/2026-07-11-sol-s1-gfx1151-server-auto-route-gate.json)
and canonical [`benchmarks/README.md`](benchmarks/README.md#gfx1151-gguf-server-automatic-route-gate-2026-07-11).

The clean gfx1151 PARO DFlash S4 profile is exact but not competitive:
`9.68` versus `65.27 tok/s` AR (`0.148x`) at B4/32 tokens. Branch-copy is
faster but diverges at generated token 1, and fused target LM-head is 5.16%
slower than unfused. See the
[`compact profile`](benchmarks/results/2026-07-11-sol-s4-gfx1151-paro-dflash-profile.json)
and the canonical
[`benchmark analysis`](benchmarks/README.md#gfx1151-paro-dflash-s4-profile-2026-07-11).

## Concurrency

The current publishable gfx1151 table is the exact PARO production-routing
catalog below. c1 has a retained timing; c2-c8 use width-1 sessions because
every native candidate fails the independent-c1 oracle. P2 proves that serial
route through ragged c8-to-c1 EOS/cancel transitions and front/middle/tail
sparse slots. Native batching remains closed until a general c>N algorithm
passes the same token/state/KV gates. See
[`docs/CONCURRENCY.md`](docs/CONCURRENCY.md) for the design history.

The linked records keep gfx1100 and gfx1151 separate because the model files,
ROCm stacks, and comparison backends differ. *Aggregate* is total tok/s across
the batch; *per-sequence* is tok/s seen by one request. See
[`docs/VLLM_RDNA3.md`](docs/VLLM_RDNA3.md) for vLLM RDNA3 setup notes.

### gfx1100 / W7900 decode tok/s vs concurrency (Qwen3.6 35B-A3B, 512/128)

**Status: stale diagnostic.** This is a median-of-3 scaling snapshot, not an
apples-to-apples engine ranking. hipEngine uses PARO W4/BF16 KV, llama.cpp uses
Vulkan Q4_K_M/f16 KV, and vLLM uses GPTQ Int4. hipEngine and llama.cpp report
backend decode timing; vLLM reports OpenAI client wall throughput.

<!-- BEGIN TOPLINE:W7900_CONCURRENCY -->
No eligible concurrency row; the mixed-quant, mixed-timing sweep remains linked below pending rerun.
<!-- END TOPLINE:W7900_CONCURRENCY -->

Protocol: prompt 512, decode 128, 8 warmup decode tokens, median of 3.
hipEngine `c=1` uses the single-sequence graph-replay benchmark and `c>1` uses
the native batch benchmark. llama.cpp restarts its server for each concurrency
and repetition with `-np c -c 1024*c`.

Source artifacts:
[`hipEngine W7900`](benchmarks/results/2026-07-07-w7900-gpu0-readme-refresh-20260707-104756-hipengine-concurrency-w7900/summary.json),
[`llama.cpp Vulkan W7900`](benchmarks/results/2026-07-07-w7900-gpu0-readme-refresh-20260707-104756-llamacpp-vulkan-concurrency-w7900/summary.json),
[`vLLM local build W7900`](benchmarks/results/2026-07-07-w7900-gpu0-readme-refresh-20260707-104756-vllm-localbuild-gptq-int4-concurrency-c1-c8-w7900.json),
[`full W7900 refresh summary`](benchmarks/results/2026-07-07-w7900-gpu0-readme-refresh-20260707-104756-summary.json),
and [`vLLM RDNA3 notes`](docs/VLLM_RDNA3.md).

### gfx1151 / Radeon 8060S PARO exact shape catalog (2026-07-11, Qwen3.6 35B-A3B, 512/128)

**Status: retained c1 performance, c1-c8 routing correctness, and production
lifecycle safety.** Clean `a18ff7bc` uses the same exact 512-token fixture at
every width. c1 graph replay is retained; every c2-c8 native candidate fails
independent-c1 equality at generated index 2 and is explicitly routed through
width-1 sessions. Clean `6f1910c9` then passes ragged c8-to-c1 EOS/cancel
lifecycle coverage without compacting physical slots.

<!-- BEGIN TOPLINE:GFX1151_PARO_CURRENT -->
| Client c | Production backend groups | Exact classification | Retained aggregate decode |
| ---: | --- | --- | ---: |
| 1 | `1` | c1 oracle / accepted | **66.910 tok/s** (`14.946 ms/token`) |
| 2 | `1+1` | explicitly serial | no separate c>N claim |
| 3 | `1+1+1` | explicitly serial | no separate c>N claim |
| 4 | `1+1+1+1` | explicitly serial | no separate c>N claim |
| 5 | five width-1 groups | explicitly serial | no separate c>N claim |
| 6 | six width-1 groups | explicitly serial | no separate c>N claim |
| 7 | seven width-1 groups | explicitly serial | no separate c>N claim |
| 8 | eight width-1 groups | explicitly serial | no separate c>N claim |
<!-- END TOPLINE:GFX1151_PARO_CURRENT -->

Protocol: W4 PARO/BF16 KV, 40 layers, exact prompt-ID SHA-256
`b162b2d0...2388`, 8 warmup decode steps, 128 measured decode steps, and greedy
sampling. c1 is a clean median of three (`66.948/66.754/66.910 tok/s`). Native
c2-c8 diagnostic rates are withheld from the topline because all rows fail the
137-token oracle. The P2 gate uses prompt lengths 449 through 512 and matches
all eight generated sequences, 30 linear-state families, and 10 live K/V
families through EOS plus front/middle/tail sparse cancellation. Ragged prefill
uses the correctness-first `per_segment_ragged_exact` fallback and makes no
throughput claim. See the [P1 compact catalog](benchmarks/results/2026-07-11-sol-p1-gfx1151-paro-c1-c8-exact-catalog.json),
[P2 lifecycle artifact](benchmarks/results/2026-07-11-sol-p2-gfx1151-paro-ragged-lifecycle.json),
and [canonical run record](benchmarks/README.md#gfx1151-paro-exact-shaperouting-catalog-2026-07-11).

### gfx1151 / Radeon 8060S historical cross-engine concurrency (2026-06-15)

**Status: stale diagnostic.** hipEngine uses PARO W4/BF16 KV; llama.cpp uses
Vulkan Q4_K_S/f16 KV. vLLM did not produce a healthy server. The summary lacks
the measured hipEngine commit, and the then-used per-run device properties could
report gfx1100 even though the run forced `HIPENGINE_HIP_ARCH=gfx1151`.

<!-- BEGIN TOPLINE:GFX1151_CONCURRENCY -->
No eligible concurrency row; the `performance_claim=false` snapshot remains linked below pending rerun.
<!-- END TOPLINE:GFX1151_CONCURRENCY -->

Protocol: prompt 512, decode 128, 8 warmup decode tokens, median of 3. Primitive
c>1 attention/KV checks passed. The generated-token field used the older
batch-shaped reference and is not independent-c1 evidence. Profiler, scaling,
and provenance gates also did not pass.

Source artifacts: [`gfx1151 summary`](benchmarks/results/2026-06-15-gfx1151-readme-concurrency-20260615-213804-summary.json),
[`hipEngine PARO`](benchmarks/results/2026-06-15-gfx1151-readme-concurrency-20260615-122207-hipengine-paro/summary.json),
[`llama.cpp Vulkan`](benchmarks/results/2026-06-15-gfx1151-readme-concurrency-20260615-213804-llamacpp-vulkan/summary.json), and
[`vLLM blocked`](benchmarks/results/2026-06-15-gfx1151-readme-concurrency-20260615-122207-vllm-gptq-int4-blocked.json).

A 2026-06-13 RX 7900 XTX rerun reached c1/c2/c4 but c8 blocked with HIP OOM;
see [`XTX partial`](benchmarks/results/2026-06-13-hipengine-qwen35-concurrency-decode-latest-xtx-blocked-c8.json).
Replicate the W7900 hipEngine, llama.cpp Vulkan, and vLLM concurrency rows with:

```bash
scripts/run_w7900_readme_refresh.sh concurrency
scripts/run_w7900_readme_refresh.sh vllm
```

The exact settings and gfx1151 runner gap are recorded in
[`benchmarks/README.md`](benchmarks/README.md#readme-sweep-test-procedure).

## GGUF Support

As of v0.2.0, hipEngine includes resident Qwen3.6 GGUF support for `Q4_K_M` and
`Q4_K_S` model files (with more formats planned). This is a major runtime path,
not just a loader shim: GGUF has its own quant readers, bulk-prefill path,
decode-repacked T16 layouts, and fast-path controls.

Current caveats:

- PARO models take ~22s to load on the W7900 test host in the current refresh;
  GGUF Q4_K_M currently takes about 74s because decode-repack happens on load.
  On-disk caching could reduce startup time later, but would require additional
  storage for repacked layouts.
- GGUF has higher base weight residency than packed PARO before KV cache is the
  deciding factor. The full-attention KV slope is the same 10-layer Qwen3.6
  shape; the 24 GiB long-context gap is mostly the loaded-weight baseline.
  Packed PARO is ~19.07 GiB on disk, while the local GGUF tensor payloads are:

  | GGUF tensor family | Q4_K_M GiB | Q4_K_M mix | Q4_K_S GiB | Q4_K_S mix | Q4_K_S - Q4_K_M |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | Q4_K | 11.531 | 54.7% | 16.875 | 84.8% | +5.344 |
  | Q5_K | 6.531 | 31.0% | 0.000 | 0.0% | -6.531 |
  | Q8_0 | 1.932 | 9.2% | 1.932 | 9.7% | +0.000 |
  | Q6_K | 1.004 | 4.8% | 1.004 | 5.0% | +0.000 |
  | F32/BF16 metadata | 0.098 | 0.5% | 0.098 | 0.5% | +0.000 |
  | **Total tensor payload** | **21.097** | **100.0%** | **19.909** | **100.0%** | **-1.188** |

  In other words, `Q4_K_S` saves ~1.19 GiB versus `Q4_K_M` by replacing the
  selected-MoE `Q5_K` expert-down payload with `Q4_K`; it still starts above
  packed PARO, and hipEngine's resident T16/pack8 decode layouts add their own
  allocator shape. On 24 GiB cards, current `Q4_K_M` BF16-KV support is a
  mid-context path unless a lower-memory KV/weight policy is explicitly enabled.
  The current clean gfx1151 p512/d128 census is **21.478 GiB** owned/tracked:
  **20.461 GiB** replacement weights, **0.503 GiB** required raw token embedding,
  **0.097 GiB** dense weights/metadata, and **0.417 GiB** scratch/session buffers.
  It confirms the default T16 path replaces source layouts rather than retaining
  raw+packed copies; this short-context margin is not a 128K capacity claim.
- GGUF is close enough to PARO to share some high-level scheduling ideas, but in
  practice it needs substantial GGUF-only kernels and dispatch. The goal for
  future releases is to keep closing the remaining PARO/GGUF speed gap.


## Architecture at a glance

```
┌─────────────────────────────────────────────────────────────────┐
│  USER API                                                       │
│  hipengine.LLM.generate()           library API                 │
│  hipengine serve                    OpenAI-compatible server    │
├─────────────────────────────────────────────────────────────────┤
│  LOADING (torch-free)                                           │
│  safetensors mmap + hipMemcpyAsync / HF config / jinja2 chat    │
│  templates / HF tokenizers (Rust)                               │
├─────────────────────────────────────────────────────────────────┤
│  DISPATCH                                                       │
│  Scheduler / Block Manager (KVPolicy) / Prefix Cache            │
│  Fusion Planner (chain → kernel plan, fused preferred)          │
│  Model / Quant / Layer plugins / Engine loop (hipGraph replay)  │
├─────────────────────────────────────────────────────────────────┤
│  CORE (torch-free primitives)                                   │
│  hipengine.Tensor / device / memory / stream / graph / blas     │
│  build (hipcc subprocess + ctypes.CDLL + .so cache)             │
├─────────────────────────────────────────────────────────────────┤
│  KERNELS (backend-keyed, 120 __global__ in the Qwen/PARO port)  │
│  kernels/hip_gfx1100/  attention / linear_attn / moe / quant    │
│                        wmma / norm / rotary / fused             │
│  kernels/hip_gfx1151/  native target-arch peer backend          │
│  kernels/cuda_sm86/    (future)                                 │
│  kernels/cpu_reference/ correctness oracle, no GPU required     │
└─────────────────────────────────────────────────────────────────┘
```

Full layer diagram, plugin axes, KV cache ABI, and roadmap are in
[`docs/PLAN.md`](docs/PLAN.md).

## Installation

```bash
# PyPI wheel: runtime, JIT kernel sources, vendored AOTriton, and server
pip install hipengine

# Source checkout: fetch Git LFS payloads before an editable install
git lfs install
git lfs pull
pip install -e .

# with the optional dlpack torch bridge for user-boundary interop
pip install -e ".[torch]"

# dev / test
pip install -e ".[dev]"
```

Python 3.11+. A working ROCm install with `libamdhip64.so` on the loader path
is required for any GPU run; CPU-reference correctness tests run without a GPU.

### ROCm / TheRock setup for retained benchmark rows

For retained gfx1100 benchmark rows, use the pinned AMD TheRock environment in
[`docs/THEROCK.md`](docs/THEROCK.md), not an ad-hoc mixed `/opt/rocm` runtime.
Current retained rows use TheRock ROCm `7.13.0a20260423` with:

```text
HIP version: 7.13.26162-1140233ffe
```

On this host (`Linux 7.0.10-1-cachyos`, W7900 VBIOS `113-D7070100-138`, RX 7900
XTX VBIOS `113-EXT89622-001`), ROCm 7.14 nightly diagnostics showed GGUF prefill
and MTP wall-time regressions, so 7.13 remains the canonical stack until a newer
ROCm release beats the same gates. See `docs/THEROCK.md` for the exact `pip
install`/repair commands, clean process wrapper, and the upstream TheRock
[`RELEASES.md`](https://github.com/ROCm/TheRock/blob/main/RELEASES.md) reference.

The installed app exposes a small command group:

```bash
hipengine --help
hipengine serve --help
hipengine bench list
```

## Quickstart

Model loading does not start network downloads. Populate the Hugging Face cache
before using a repository ID:

```bash
hf download shisa-ai/Qwen3.6-35B-A3B-PARO-packed
```

Then construct `LLM` with the same repository ID:

```python
from hipengine import LLM, SamplingParams

llm = LLM("shisa-ai/Qwen3.6-35B-A3B-PARO-packed")
outputs = llm.generate(
    ["Hello, hipEngine."],
    SamplingParams(max_tokens=64, temperature=0.0),
)
print(outputs[0])
```

`LLM(model)` auto-detects `gfx1100` or `gfx1151` and selects the model plugin's
quantization. The Qwen3.6 GGUF path also selects T16 decode-repack plus the
retained WMMA-prefill/GEMV-decode session profile. Explicit `backend=` and
`quant=` arguments are overrides; supported PARO and GGUF models do not require
hipEngine environment variables. Unsupported registry combinations fail instead
of falling back to a torch path.

## OpenAI-compatible server

The OpenAI-compatible FastAPI layer is installed by default:

```bash
pip install hipengine
hipengine serve \
  --model shisa-ai/Qwen3.6-35B-A3B-PARO-packed \
  --served-model-name qwen-paro
```

`--model` accepts either a local filesystem path or a Hugging Face model ID
already present in the local HF cache; hipEngine resolves IDs locally and does
not download weights during startup.

Supported endpoints: `GET /v1/models`, `POST /v1/completions`, and
`POST /v1/chat/completions` with token-level SSE streaming, OpenAI-style tool
calling, and Qwen no-think controls. Chat responses separate `<think>` reasoning
into `reasoning_content` (matching the OpenAI reasoning-content convention). The
server eagerly warms the model on startup by default, logs startup load/warmup
timing, caps omitted chat `max_tokens` with `--chat-default-max-tokens` (default
4096), and has an explicit `--debug` mode for full request/response payload
logging. See [`docs/API.md`](docs/API.md) for request examples, bearer-token
auth, diagnostics, and current limitations.

## Documentation

| File | Purpose |
| --- | --- |
| [`docs/PLAN.md`](docs/PLAN.md) | Architecture, plugin axes, phase roadmap, LoC budgets |
| [`docs/BENCHMARK.md`](docs/BENCHMARK.md) | Benchmark protocols, baselines, correctness gate, artifact format |
| [`docs/TESTING.md`](docs/TESTING.md) | RED/GREEN workflow, correctness oracles, fixture policy |
| [`docs/KERNELS.md`](docs/KERNELS.md) | Kernel catalog, source-lineage drift workflow, JIT cache gotchas, build profiles |
| [`docs/ENVS.md`](docs/ENVS.md) | Environment variables, TheRock setup, benchmark/profiling profiles |
| [`docs/ROOFLINE.md`](docs/ROOFLINE.md) | RDNA3 / W7900 performance model and decision tree |
| [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md) | Implementation status and concrete milestones |
| [`docs/API.md`](docs/API.md) | OpenAI-compatible server usage and endpoint support |
| [`docs/PREFILL.md`](docs/PREFILL.md) | Native prefill implementation spec |
| [`docs/SAMPLING.md`](docs/SAMPLING.md) | Normal sampling parameter support plan |
| [`docs/MTP.md`](docs/MTP.md) | Multi-token prediction plan |
| [`docs/DFLASH.md`](docs/DFLASH.md) | DFlash draft-model speculative decode plan |
| [`docs/SOL-OPTIMIZATION.md`](docs/SOL-OPTIMIZATION.md) | gfx1151 PARO/GGUF optimization ledger and completion gates |
| [`docs/MTP-LLAMACPP-PARITY.md`](docs/MTP-LLAMACPP-PARITY.md) | Current GGUF MTP parity results and open reruns |
| [`docs/PARO-GGUF-MTP-TRANSFER.md`](docs/PARO-GGUF-MTP-TRANSFER.md) | PARO follow-up queue from GGUF/MTP server and verifier work |
| [`docs/HIP-vs-VULKAN.md`](docs/HIP-vs-VULKAN.md) | Current timing-contract v2 backend conclusions and portability gates |
| [`benchmarks/README.md`](benchmarks/README.md) | Canonical topline scoreboard, platform freshness, protocols, and refresh commands |
| [`AGENTS.md`](AGENTS.md) | Ground rules for every coding / review / benchmarking task |
| [`WORKLOG.md`](WORKLOG.md) | Append-only cross-session journal of decisions and measurements |

## Development

```bash
# narrowest test suite (CPU-only paths run without a GPU)
pytest -q

# kernel source-lineage drift check before any port
python3 scripts/check_lineage.py --kind kernel --diff stat
```

See [`AGENTS.md`](AGENTS.md) for the full workflow: when to run the
CPU-reference correctness gate, when to add a `rocprofv3 --kernel-trace` smoke,
and what a retained benchmark row requires.

## References & lineage

hipEngine is not a fork of any project; it is a brand new codebase with from-scratch
code and kernels. Of course it builds on the work of many others:

- [ROCm](https://github.com/ROCm/rocm) - of course this all sits on AMD's open-source
  compute stack, notably on [HIP](https://github.com/ROCm/rocm-systems/tree/develop/projects/hip).
- [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) - most of the original
  kernel tuning iteration loops used this as a host-layer. Some of the performance 
  limitations of the architecture motivated the hipEngine rewrite, but we remain
  greatful and deeply appreciative of nano-vllm as a great research platform.
- [ParoQuant](https://github.com/z-lab/paroquant) - after reviewing the current SOTA on model
  quantization, we chose ParoQuant as the first target due to both its excellent accuracy
  *and* its efficiency (QTIP/[YAQA](https://github.com/Cornell-RelaxML/yaqa-quantization) is 
  very cool but proved challenging to implement performant RDNA3 kernels)
- [FastDMS](https://github.com/shisa-ai/FastDMS) - our KVCache ABI is shaped by the lessons 
   learned from building our DMS reference implementation.

Greetz: [hipfire](https://github.com/Kaden-Schutt/hipfire), [Lucebox](https://github.com/Luce-Org/lucebox-hub), [DS4](https://github.com/antirez/ds4), [ExLlamaV3](https://github.com/turboderp-org/exllamav3) and ofc the og [llama.cpp](https://github.com/ggml-org/llama.cpp)

See also: [Marlin](https://github.com/IST-DASLab/marlin), [kernel-anvil](https://github.com/apollosenvy/kernel-anvil), [wmma_ops](https://github.com/glovepost/wmma_ops), [tilelang](https://github.com/tile-ai/tilelang), [fsr4-rdna3-optimization](https://github.com/lhl/fsr4-rdna3-optimization), [ROCm examples](https://github.com/ROCm/rocm-examples)


## License

hipEngine source code is licensed under **AGPL-3.0-or-later**. It is built and distributed
for anyone who has an AMD card that hasn't been living up to its compute potential.

Model weights, checkpoints, and external datasets remain under their own licenses.
