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
first two 35B-class model-loading surfaces are now available on gfx1100:
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

## Memory Usage

The 2026-05-19 W7900 diagnostic showed the packed Qwen3.6 PARO model fitting a
128K context with BF16 KV and 256K with INT8 per-token/per-head KV under 24 GiB
tracked allocator peak. The artifact records hipEngine `ae229513` and exact
commands, but not the compiler version; treat this as a stale capacity
diagnostic until it is rerun with the full provenance and Qwen3.6 long-rollout
quality gates.

<!-- BEGIN TOPLINE:W7900_MEMORY_CAPACITY -->
| Model | Context | KV cache | Sampled HIP peak | Allocator peak | Retained KV | Prefill | Decode |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 35B-A3B PARO | 128K | BF16 | 21.04 GiB | 21.88 GiB | 2.69 GB | 1091.9 tok/s | 62.2 tok/s |
| Qwen3.6 35B-A3B PARO | 128K | INT8 per token/head | 19.80 GiB | 20.89 GiB | 1.36 GB | 1076.5 tok/s | 60.0 tok/s |
| Qwen3.6 35B-A3B PARO | 256K | INT8 per token/head | 21.96 GiB | 23.71 GiB | 2.71 GB | 670.2 tok/s | 40.3 tok/s |
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

## Performance

### gfx1100 (Radeon RX 7900 XTX / Radeon Pro W7900)

**Status: stale diagnostic.** The table is the last complete same-host sweep,
measured on 2026-07-07 at hipEngine `b4edca09` with TheRock HIP
`7.13.26162-1140233ffe`. Its top-level artifact sets
`performance_claim=false`. The GGUF path repeatedly selected token `9707` and
is not a correctness-certified performance baseline. hipEngine PARO uses W4
PARO/BF16 KV; the other columns use Q4_K_M GGUF with BF16/f16 KV, so maxima are
not same-quant wins.

<!-- BEGIN TOPLINE:W7900_SWEEP -->
#### Prefill tok/s

| Workload | hipEngine PARO | hipEngine GGUF Q4_K_M | llama.cpp HIP Q4_K_M | llama.cpp Vulkan Q4_K_M |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | 2796.853 | 653.979 | 2502.690 | 2731.086 |
| 1K/128 | 2917.115 | 664.564 | 2423.728 | 2642.684 |
| 4K/128 | 2904.920 | 668.125 | 2294.828 | 2539.920 |
| 32K/128 | 2103.724 | 635.321 | 1680.677 | 1950.575 |
| 64K/128 | 1575.284 | 578.702 | 1319.054 | 1417.008 |
| 128K/128 | 1063.951 | 490.289 | 913.108 | 1075.764 |

#### Decode tok/s

| Workload | hipEngine PARO | hipEngine GGUF Q4_K_M | llama.cpp HIP Q4_K_M | llama.cpp Vulkan Q4_K_M |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | 112.207 | 35.838 | 79.603 | 107.216 |
| 1K/128 | 102.458 | 35.610 | 79.498 | 106.851 |
| 4K/128 | 102.918 | 34.836 | 78.627 | 102.677 |
| 32K/128 | 91.745 | 35.162 | 72.228 | 91.480 |
| 64K/128 | 77.213 | 35.592 | 66.437 | 83.106 |
| 128K/128 | 59.999 | 35.426 | 57.712 | 70.479 |

#### Peak GiB

| Workload | hipEngine PARO | hipEngine GGUF Q4_K_M | llama.cpp HIP Q4_K_M | llama.cpp Vulkan Q4_K_M |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | 21.029 | 25.492 | 21.598 | 21.260 |
| 1K/128 | 21.241 | 25.492 | 21.610 | 21.220 |
| 4K/128 | 21.973 | 25.492 | 21.666 | 21.278 |
| 32K/128 | 22.082 | 25.492 | 22.208 | 21.855 |
| 64K/128 | 22.082 | 25.492 | 22.887 | 22.512 |
| 128K/128 | 22.124 | 25.492 | 24.080 | 23.824 |
<!-- END TOPLINE:W7900_SWEEP -->

W7900 row sources: [`summary`](benchmarks/results/2026-07-07-w7900-gpu0-readme-refresh-20260707-104756-summary.json), [`hipEngine PARO`](benchmarks/results/2026-07-07-w7900-gpu0-readme-refresh-20260707-104756-hipengine-paro-packed-5run.json), [`hipEngine GGUF`](benchmarks/results/2026-07-07-w7900-gpu0-readme-refresh-20260707-104756-hipengine-gguf-q4km-5run.json), [`llama.cpp HIP`](benchmarks/results/2026-07-07-w7900-gpu0-readme-refresh-20260707-104756-llamacpp-hip-q4km-f16kv.json), and [`llama.cpp Vulkan`](benchmarks/results/2026-07-07-w7900-gpu0-readme-refresh-20260707-104756-llamacpp-vulkan-q4km-f16kv.json). Exact settings and the refresh blocker are in the canonical [`benchmarks/README.md`](benchmarks/README.md#w7900-model-sweep-2026-07-07).

### gfx1151 (AMD Ryzen AI MAX+ 395 / Radeon 8060S)

**Status: stale diagnostic.** The Strix Halo rows were measured on 2026-06-15
from a clean detached worktree at `64b86b9a` with TheRock HIP
`7.13.60980-c76140fa27`. The run used one measured repetition and no measured
warmup. Its summary omits the source/build provenance, which was recovered from
`WORKLOG.md`; the next refresh must emit it in the artifact.

<!-- BEGIN TOPLINE:GFX1151_SWEEP -->
#### Prefill tok/s

| Workload | hipEngine PARO | hipEngine GGUF Q4_K_M | llama.cpp HIP Q4_K_M | llama.cpp Vulkan Q4_K_M |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | 956.666 | 833.366 | 1016.696 | 1043.209 |
| 1K/128 | 1067.175 | 854.308 | 1069.681 | 1055.050 |
| 4K/128 | 1062.248 | 729.117 | 1021.186 | 1027.069 |
| 32K/128 | 822.255 | 619.570 | 742.869 | 809.619 |
| 64K/128 | 622.752 | 522.872 | 569.611 | 658.399 |
| 128K/128 | 425.727 | 384.011 | 384.959 | 473.651 |

#### Decode tok/s

| Workload | hipEngine PARO | hipEngine GGUF Q4_K_M | llama.cpp HIP Q4_K_M | llama.cpp Vulkan Q4_K_M |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | 66.967 | 56.581 | 51.640 | 62.434 |
| 1K/128 | 61.768 | 52.832 | 51.446 | 61.572 |
| 4K/128 | 62.910 | 53.638 | 49.581 | 60.012 |
| 32K/128 | 50.368 | 44.383 | 43.628 | 50.911 |
| 64K/128 | 41.966 | 37.741 | 38.604 | 44.010 |
| 128K/128 | 30.286 | 28.043 | 31.598 | 34.714 |

#### hipEngine tracked allocator peak GiB

| Workload | hipEngine PARO | hipEngine GGUF Q4_K_M |
| --- | ---: | ---: |
| 512/128 | 20.924 | 26.264 |
| 1K/128 | 20.926 | 26.264 |
| 4K/128 | 20.937 | 26.264 |
| 32K/128 | 21.047 | 26.264 |
| 64K/128 | 21.047 | 26.264 |
| 128K/128 | 21.248 | 26.264 |
<!-- END TOPLINE:GFX1151_SWEEP -->

On Strix Halo, `rocm-smi` / sysfs exposed only a 512 MiB VRAM aperture, so
cross-engine memory comparisons are omitted. Row sources: [`gfx1151 summary`](benchmarks/results/2026-06-15-gfx1151-readme-udq4km-20260615-040438-summary.json),
[`hipEngine PARO`](benchmarks/results/2026-06-15-gfx1151-readme-udq4km-20260615-040438-hipengine-paro-packed-1run.json),
[`hipEngine GGUF`](benchmarks/results/2026-06-15-gfx1151-readme-udq4km-20260615-040438-hipengine-gguf-ud-q4km-1run.json),
[`llama.cpp HIP`](benchmarks/results/2026-06-15-gfx1151-readme-udq4km-20260615-040438-llamacpp-hip-ud-q4km-f16kv.json), and
[`llama.cpp Vulkan`](benchmarks/results/2026-06-15-gfx1151-readme-udq4km-20260615-040438-llamacpp-vulkan-ud-q4km-f16kv.json). The canonical [`benchmarks/README.md`](benchmarks/README.md#gfx1151-model-sweep-2026-06-15) records the missing-runner blocker and next refresh protocol.

See [`benchmarks/README.md`](benchmarks/README.md) for the platform freshness
index, exact settings, run commands, and evidence status.

## Speculative decode (DFlash / MTP)

The table includes only contracts with a true same-protocol AR control. Exact
MTP and `llama-compat` use different state semantics and output horizons.

<!-- BEGIN TOPLINE:SPECULATIVE -->
| Path | Platform and protocol | Result | Evidence status |
| --- | --- | ---: | --- |
| DFlash B=4 online-gated | W7900/gfx1100; Qwen3.6-27B PARO target plus Qwen3.6-27B DFlash drafter; 9 prompts; 64 decode tokens | 40.10 vs 32.57 AR tok/s, **1.231x** | Retained under the recorded DFlash gate; source tree was dirty and must be refreshed before changing the claim |
| GGUF MTP exact B5 | Radeon 8060S/gfx1151; Qwen3.6-35B-A3B Q4_K_M; 10-prompt category suite; fixed 10 cycles; exact/default state semantics | 61.98 vs 54.79 AR tok/s, **1.1312x** | Retained for this fixed-cycle contract |
| GGUF MTP `llama-compat` B2 | Radeon 8060S/gfx1151; same GGUF and prompt suite; natural24/cyclecap24; direct-commit/dp4a compatibility semantics | 71.52 vs 54.79 AR tok/s, **1.3055x** | Retained for this compatibility contract; accuracy-traded and not serial-prefix-equivalent |
<!-- END TOPLINE:SPECULATIVE -->

Artifacts: [`DFlash`](benchmarks/results/2026-06-11-hipengine-dflash-27b-dense-hardening-rerun.json),
[`exact MTP`](benchmarks/results/2026-07-02-ar-mtp-default-parallelattn-full.json),
and [`llama-compat` MTP](benchmarks/results/2026-07-03-ar-mtp-llama-compat-directcommit-nocopy-natural24-cyclecap24-f32head-full.json).
OpenAI MTP server rows are excluded because their completion-token and
batch-timing accounting does not satisfy the topline contract. See the
canonical [`benchmarks/README.md`](benchmarks/README.md#blocked-and-diagnostic-benchmark-attempts).

## Concurrency (batched decode)

hipEngine has a native `c>1` decode path: a scheduler-owned compact prefill plus
a device-resident batched decode step (token feedback through `batch_lm_out_index`,
device batched LM-head argmax, on-stream position advance) that can be captured
and replayed as a single HIP graph. See [`docs/CONCURRENCY.md`](docs/CONCURRENCY.md)
for the design and the C3.0a/b/c decode-throughput work.

The snapshots below keep gfx1100 and gfx1151 separate because the model files,
ROCm stacks, and comparison backends differ. *Aggregate* is total tok/s across
the batch; *per-sequence* is tok/s seen by one request. See
[`docs/VLLM_RDNA3.md`](docs/VLLM_RDNA3.md) for vLLM RDNA3 setup notes.

### gfx1100 / W7900 decode tok/s vs concurrency (Qwen3.6 35B-A3B, 512/128)

**Status: stale diagnostic.** This is a median-of-3 scaling snapshot, not an
apples-to-apples engine ranking. hipEngine uses PARO W4/BF16 KV, llama.cpp uses
Vulkan Q4_K_M/f16 KV, and vLLM uses GPTQ Int4. hipEngine and llama.cpp report
backend decode timing; vLLM reports OpenAI client wall throughput.

<!-- BEGIN TOPLINE:W7900_CONCURRENCY -->
| Concurrency | hipEngine PARO decode aggregate | hipEngine per sequence | llama.cpp Vulkan decode aggregate | llama.cpp per sequence | vLLM OpenAI wall aggregate | vLLM per sequence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 114.98 | 114.98 | 105.63 | 105.63 | 21.32 | 21.32 |
| 2 | 113.34 | 56.67 | 156.06 | 78.03 | 40.61 | 20.31 |
| 4 | 158.25 | 39.56 | 76.52 | 19.13 | 78.41 | 19.60 |
| 8 | 189.59 | 23.70 | 26.47 | 3.31 | 116.44 | 14.55 |
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

### gfx1151 / Radeon 8060S PARO shape diagnostic (2026-07-10, Qwen3.6 35B-A3B, 512/128)

**Status: diagnostic, not retained.** This direct PARO batch measurement ran at
tracked-clean hipEngine `4175dabf` with detected and target arch gfx1151. It
uses opt-in retained-default recovery routes; the production server default
uses different routing. Odd widths fail generated-token equality; c1, dynamic
shrinking, profiler, and scaling gates are missing. Red-width timing is
withheld.

<!-- BEGIN TOPLINE:GFX1151_PARO_CURRENT -->
| Width | Aggregate decode tok/s | Per sequence tok/s | Median step ms | Exact gate | Measured route |
| ---: | ---: | ---: | ---: | --- | --- |
| 1 | Not rerun | Not rerun | Not rerun | Same-fixture timing missing | Single-sequence control required |
| 2 | 78.578 | 39.289 | 25.465 | Primitive pass; generated IDs 3/3 | Native full attention; selected-c1 MoE; batched LM-head |
| 3 | Withheld | Withheld | Withheld | Rejected at token index 4 | Grouped-compact MoE; selected-layer rowchunk2 |
| 4 | 99.616 | 24.904 | 40.158 | Primitive pass; generated IDs 3/3 | Selected-c1 MoE; all-layer rowchunk2; batched LM-head |
| 5 | Withheld | Withheld | Withheld | Rejected at token index 4 | Grouped-compact MoE; selected-layer rowchunk2 |
| 6 | 109.909 | 18.318 | 54.568 | Primitive pass; generated IDs 3/3 | Selected-c1 MoE; selected-layer rowchunk2; serial LM-head |
| 7 | Withheld | Withheld | Withheld | Rejected at token index 2 | Grouped-compact MoE; all-layer rowchunk2 |
| 8 | 115.515 | 14.439 | 69.254 | Primitive pass; generated IDs 3/3 | Selected-c1 MoE; all-layer rowchunk2; batched LM-head |
<!-- END TOPLINE:GFX1151_PARO_CURRENT -->

Protocol: W4 PARO/BF16 KV, 40 layers, fixed 512-token slices, 8 warmup decode
steps, 128 measured decode steps, and greedy sampling. Green widths report the
median of three direct backend runs and pass primitive plus 137-token generated
equality. See the [compact artifact](benchmarks/results/2026-07-10-gfx1151-paro-cn-current-diagnostic-summary.json)
and [canonical run record](benchmarks/README.md#gfx1151-paro-direct-exact-shape-diagnostic-2026-07-10).

### gfx1151 / Radeon 8060S historical cross-engine concurrency (2026-06-15)

**Status: stale diagnostic.** hipEngine uses PARO W4/BF16 KV; llama.cpp uses
Vulkan Q4_K_S/f16 KV. vLLM did not produce a healthy server. The summary lacks
the measured hipEngine commit, and the then-used per-run device properties could
report gfx1100 even though the run forced `HIPENGINE_HIP_ARCH=gfx1151`.

<!-- BEGIN TOPLINE:GFX1151_CONCURRENCY -->
| Concurrency | hipEngine PARO decode aggregate | hipEngine per sequence | llama.cpp Vulkan decode aggregate | llama.cpp per sequence | vLLM OpenAI |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 66.62 | 66.62 | 62.16 | 62.16 | Blocked: server unhealthy |
| 2 | 69.54 | 34.77 | 94.12 | 47.06 | Blocked |
| 4 | 88.39 | 22.10 | 119.51 | 29.88 | Blocked |
| 8 | 100.68 | 12.59 | 119.94 | 14.99 | Blocked |
<!-- END TOPLINE:GFX1151_CONCURRENCY -->

Protocol: prompt 512, decode 128, 8 warmup decode tokens, median of 3. Primitive
c>1 attention/KV checks and generated-token equality passed, but the retained
profiler, scaling, and provenance gates did not.

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
# one-time: fetch Git LFS payloads, including the vendored AOTriton runtime/images
git lfs install
git lfs pull

# runtime + OpenAI-compatible server (torch-free hot path)
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

## Quickstart (Phase 0 — bring-up only)

The public API surface is stable:

```python
from hipengine import LLM, SamplingParams

llm = LLM("/path/to/model", quant="w4_paro")  # backend="auto" by default
outputs = llm.generate(
    ["Hello, hipEngine."],
    SamplingParams(max_tokens=64, temperature=0.0),
)
print(outputs[0])
```

Today `LLM.generate()` only resolves to narrow Qwen3.5 / PARO bring-up paths
registered in `hipengine.generation`; unsupported `(model, backend, quant)`
combinations fail loudly rather than falling back to a generic torch path. See
[`docs/PLAN.md`](docs/PLAN.md) for the model / quant roadmap.

## OpenAI-compatible server

The OpenAI-compatible FastAPI layer is installed by default:

```bash
pip install hipengine
hipengine serve \
  --model shisa-ai/Qwen3.6-35B-A3B-PARO-packed \
  --quant w4_paro \
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
| [`docs/PARO-GGUF-MTP-TRANSFER.md`](docs/PARO-GGUF-MTP-TRANSFER.md) | PARO follow-up queue from GGUF/MTP server and verifier work |
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
