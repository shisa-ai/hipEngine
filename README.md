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

**v0.3.0 alpha.** The runtime hot path is torch-free by construction, and the
first two 35B-class model-loading surfaces are available on gfx1100 and gfx1151:
[shisa-ai/Qwen3.6-35B-A3B-PARO-packed](https://huggingface.co/shisa-ai/Qwen3.6-35B-A3B-PARO-packed)
(19.07 GiB, 4.68 bpw) in packed
[ParoQuant](https://github.com/shisa-ai/paroquant) format, plus Qwen3.6 GGUF
`Q4_K_M` / `Q4_K_S` files through the resident GGUF path. Older benchmark
artifacts may still show the historical
`Qwen3.6-35B-A3B-PARO-full4096-e5-packed` name or local MTP-BF16 assembly path;
those rows use the same packed PARO architecture and remain the evidence for the
numbers below.

- Model-aware `backend="auto"` / `quant="auto"` defaults select the registered
  PARO or GGUF route without environment-variable setup. Direct generation now
  supports exact token-id prompts, detailed outputs, logprobs, structured finish
  details, and backend execution telemetry.
- PARO and GGUF support ordinary sampling controls including top-k/min-p,
  penalties, logit bias, suppression, deterministic seeds, EOS/min-token policy,
  token stops, and multi-token stops. Covered PARO shapes use a native GPU
  sampler; unsupported shapes use an explicit host fallback.
- The OpenAI-compatible server includes capability/readiness discovery, token
  and context diagnostics, exact usage accounting, request batching, deadlines,
  cancellation, opt-in Prometheus metrics, and detailed streaming metadata.
- Local-agent support includes OpenAI-style tools, Qwen thinking controls,
  structured-output result validation, deterministic continuation handles, and
  app-local transcript sessions with fork, rollback, snapshot, and overflow
  policies.
- Qwen3.6 GGUF models with NextN tensors expose detailed MTP generation and a
  guarded explicit non-streaming server route. Dense PARO DFlash and the shared
  speculative proposal/verify/commit infrastructure are available as retained
  runtime and benchmark paths.
- PARO BF16 KV has retained W7900 evidence through 128K, and **208 Ki is the
  recommended safe BF16 cap** on a physical 24 GB XTX. The all-layer 256K INT8
  layout fits its tracked-memory gate but fails Qwen3.6 fidelity. The milder
  six-BF16/four-INT8 native layout passes the tested GGUF accuracy gates, but
  fails PARO accuracy and quality-preserving 256 Ki request-scratch capacity.
  Neither INT8 route is supported/default. Current capacity, throughput,
  speculative-decode, and concurrency evidence is reported below with separate
  gfx1100/gfx1151 provenance and correctness gates.

This remains an alpha, single-GPU release. Production PARO native `c>1` decode
is disabled pending independent-c1 correctness, app-local sessions do not reuse
resident KV, structured outputs are not grammar-constrained decoding, and the
server MTP route is explicit-only. See [the API limitations](docs/API.md#current-limitations)
and [concurrency status](docs/CONCURRENCY.md#current-answer) for the exact
boundaries.


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

On gfx1151, hipEngine sets `GPU_MAX_HW_QUEUES=1` before HIP loads because it
reduces a retained gfx11 low-power queue failure and is non-regressive at short
context. It is not a repeated-128K lifecycle guarantee: current production can
still hit the firmware/scheduler stall. A matched follow-up reproduces the stall
under both HIP 7.15 and HIP 7.13, so downgrading ROCm is not a safe workaround.
Explicit values are preserved; set `GPU_MAX_HW_QUEUES=4` before process start to
restore ROCm's documented default for diagnosis. gfx1100 is unchanged. See
[`docs/ENVS.md`](docs/ENVS.md) and the
[cross-stack lifecycle artifact](benchmarks/results/2026-07-15-gfx1151-128k-hip713-vs-715-lifecycle.json).

Wave32 is the default for `hip_gfx1100` device code; wave64 is treated as an
isolated experiment with its own gates (see
[`docs/PLAN.md`](docs/PLAN.md#rdna3-wavefront-and-scheduling-caveat)).

## Memory Usage

The clean 2026-07-13 profile-aware BF16 frontier (`5a49b16d`) directly tests
the current Qwen3.6 packed PARO model on a physical 24 GB gfx1100 card. The
automatic low-memory prefill profile makes **208 Ki the recommended safe BF16
cap** with 0.361 GiB observed headroom; 220 Ki completes but leaves only about
78 MiB and is edge-only. Separately, compact 256K all-layer INT8 (`d6504544`)
fits its tracked layout gate but fails fidelity. Native tail-four mixed KV saves
18.75% of K/V and passes the tested GGUF accuracy gates, but PARO accuracy and
the quality-preserving 256 Ki XTX request-scratch allocation reject. Accordingly,
256K remains diagnostic allocation capacity—not a supported route.

<!-- BEGIN TOPLINE:W7900_MEMORY_CAPACITY -->
| Route / profile | Hardware | Context/decode | Tracked peak | Observed device peak | Device/card margin | Capacity / quality status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| PARO BF16 KV reference | W7900, default chunks | 128K/128 | **22.124 GiB** | 21.107 GiB phase sample | n/a | Reference path |
| PARO BF16 KV, automatic 24 GB low-memory profile | RX 7900 XTX 24 GB | **208 Ki (212,992)/128** | **23.082 GiB** | **23.623 GiB** | **+0.361 GiB** | **Recommended practical safe cap** |
| PARO BF16 KV, automatic 24 GB low-memory profile | RX 7900 XTX 24 GB | 220 Ki (225,280)/128 | 23.369 GiB | **23.908 GiB** | **+0.076 GiB (~78 MiB)** | Physical pass, but **edge only—not safe cap** |
| PARO BF16 KV, default 48 GB-card profile | W7900 | 220 Ki (225,280)/128 | 24.090 GiB | at least 24.832 GiB | at most -0.848 GiB vs 24 GB card | Rejected for this larger-chunk profile |
| PARO INT8 per-token/head KV, FP16 scales | W7900 | 256K/128 | **22.971 GiB** | 21.041 GiB phase sample | +1.029 GiB tracked | **Rejected** by Qwen3.6 matched-context and task gates |
| PARO tail-four Hadamard-group32 mixed KV, BF16-oracle prefill | RX 7900 XTX 24 GB | 256 Ki (262,144)/128 request scratch | **23.469 GiB before failed allocation** | 22.566 GiB after clean OOM | 1.418 GiB free before request scratch; insufficient | **Rejected:** `HIP error 2` OOM and PARO fidelity failure; no segfault |
| PARO tail-four Hadamard-group32 mixed KV, direct-streaming control | RX 7900 XTX 24 GB | 256 Ki (262,144)/128 request scratch | **23.290 GiB** | **23.590 GiB** live sample | **+0.394 GiB** live | Allocation passes, but direct packed prefill is **correctness-rejected** |

The native explicit `tail4_hadamard_group32` layout keeps K/V for
full-attention layers `3,7,11,15,19,23` in BF16 and stores only layers
`27,31,35,39` as Hadamard-group32 INT8 with FP16 scales. At 262,400 retained
rows it uses `4,366,336,000` K/V bytes—**18.75% below BF16**—with no persistent
BF16 shadow. PARO's quality-preserving prefill uses a temporary BF16 oracle;
GGUF's post-quality layout audit reports zero persistent oracle/mirror buffers.
Native PARO still fails 1/11 prompts at 512/8 and 2/11 at 4K/16 (58.82%
worst-prompt top-1), and its 256 Ki quality-preserving request scratch OOMs.

The clean `c971262f` therock-7.15 GGUF-only closure passes all 11 prompts at
512/8 (max KL `0.007455`, top-1 100%) and 4K/16 (mean/max KL
`0.0001369/0.009926`, aggregate/minimum-prompt top-1 `99.47%/94.12%`) plus
bounded `mixed_v1` at 128K/16 (max KL `5.19e-5`, top-1 100%). At 128K,
persistent K/V is `2,185,297,920` bytes versus BF16 `2,689,597,440` bytes and
live owned memory falls `24.168 -> 23.698 GiB`. It still rejects promotion:
production 4K prefill/decode regress `0.67%/0.75%`, one-shot 128K decode
regresses `3.82%`, and production prefill allocates then frees
`1,075,838,976` bytes—byte-exact to four BF16 layer caches—raising allocator
high water `24.168 -> 24.700 GiB`. The transient attribution is inferred from
the exact bytes; it is not a persistent shadow. The policy remains explicit
and non-default. Evidence:
`benchmarks/results/2026-07-15-gfx1100-gguf-tail4-hadamard-clean-gate.json` and
`benchmarks/results/2026-07-14-gfx1100-native-tail4-hadamard-kv-outcome.json`.
<!-- END TOPLINE:W7900_MEMORY_CAPACITY -->

The INT8 layout retains 2,686,976,000 payload bytes plus 20,992,000 FP16 scale
bytes across ten full-attention layers and no BF16 K/V shadow. Its final
BF16-reference-token matched 128K/16 gate rejects at mean/max KL
`0.85128/4.97382` and 41.18% top-1 agreement. Format and mixed-policy screens
did not find a candidate that transferred through 4K.

The former llama.cpp Q8_0 pass is now a repeated-token saturation control, not
representative quality evidence. On identical Q4_K_M weights at exact mixed
4K/16, native Q8_0 rejects at mean/max KL `0.075654/1.26009` despite 94.12%
top-1; F16/F16 is exactly zero. K-only and V-only Q8 reach `0.096682` and
`0.243219` mean KL, while full Q8 benefits from non-additive K/V cancellation.
The repeated full-Q8 control is only `0.00000619` KL, confirming prompt content
as the dominant difference.

hipEngine shows the same protocol effect. Host per-head/group32/Hadamard all
pass repeated 4K/16 near `0.000002` KL but reject mixed at
`0.12779/0.28106/0.25180`. Pure native per-head INT8 rejects mixed at
`0.19038/2.99555`, 88.24% top-1, with all ten layers INT8 and no BF16 mirror.
Direct arithmetic is therefore not a universal fidelity repair. The separate
same-weight hipEngine-GGUF-BF16 versus llama.cpp-F16 bridge preserves 100%
top-1; its `0.26606` all-position mean KL is prompt-final dominated, while 16
decode rows average `0.000510`.

The original five-category free-generation reference is unscorable. In the
replacement restricted-choice diagnostic, INT8 flips one of two
BF16-qualified 4K answers (multihop `D -> C`) but retains all three qualified
32K answers. This shows that large KL can change a bounded functional decision
without implying every answer changes; it remains partial evidence, not support
for 256K INT8. Memory was measured once; timing is diagnostic.

See the
[`capacity/fidelity outcome`](benchmarks/results/2026-07-13-w7900-paro-int8-kv-accuracy-outcome.json),
[llama.cpp repeated-token Q8_0 control](benchmarks/results/2026-07-13-w7900-llamacpp-q8-kv-matched-quality.json),
[repeated/mixed prompt and native arithmetic isolation](benchmarks/results/2026-07-13-w7900-gguf-q8-kv-protocol-arithmetic-isolation.json),
[same-weight GGUF bridge](benchmarks/results/2026-07-13-w7900-gguf-llamacpp-matched-parity.json),
[bounded functional check](benchmarks/results/2026-07-13-w7900-paro-int8-kv-functional-mc.json),
[format screen](benchmarks/results/2026-07-13-w7900-paro-kv-format-ablation.json),
and [policy screen](benchmarks/results/2026-07-13-w7900-paro-kv-policy-ablation.json).

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

**Status: retained.** The GGUF column is the clean 2026-07-16 final
selector-unset BF16-KV sweep at `28b37356` on therock HIP 7.15: independent
right-sized sessions, one discarded warmup, three measured runs, and production
graph decode. All 18 GGUF final IDs are exact and maximum prefill/decode stdev
over median is `0.658%/0.223%`. PARO and llama.cpp retain their clean July 12
protocols. PARO is W4 PARO/BF16 KV; the other columns use Q4_K_M with BF16/F16
KV, so bold values are descriptive rather than same-quant wins. GGUF prefill now
beats llama.cpp HIP at every shape and Vulkan through 64K; GGUF decode beats HIP
everywhere and is closest to Vulkan at 4K (`-2.47%`).

<!-- BEGIN TOPLINE:W7900_SWEEP -->
#### Prefill tok/s

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **2917.732** | 2716.648 | 2412.320 | 2627.990 |
| 1K/128 | 2995.876 | **3052.541** | 2389.670 | 2631.750 |
| 4K/128 | 2943.038 | **2953.101** | 2255.080 | 2521.770 |
| 32K/128 | **2108.868** | 2078.038 | 1667.640 | 1943.920 |
| 64K/128 | **1584.131** | 1559.878 | 1291.820 | 1414.470 |
| 128K/128 | 1056.252 | 1037.378 | 891.949 | **1079.280** |

#### Decode tok/s

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **115.599** | 92.833 | 80.756 | 107.786 |
| 1K/128 | 103.238 | 98.148 | 80.805 | **107.555** |
| 4K/128 | **105.943** | 100.522 | 79.768 | 103.066 |
| 32K/128 | **92.438** | 88.240 | 74.304 | 91.835 |
| 64K/128 | 78.260 | 76.691 | 69.010 | **83.746** |
| 128K/128 | 60.663 | 62.669 | 60.933 | **70.833** |

#### Peak memory GiB

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **18.144** | 21.228 | 21.606 | 21.260 |
| 1K/128 | **18.367** | 21.295 | 21.618 | 21.220 |
| 4K/128 | **19.161** | 21.670 | 21.674 | 21.278 |
| 32K/128 | **19.864** | 22.234 | 22.216 | 21.855 |
| 64K/128 | **20.403** | 22.879 | 22.895 | 22.512 |
| 128K/128 | **22.124** | 24.168 | 24.089 | 23.824 |
<!-- END TOPLINE:W7900_SWEEP -->

W7900 row sources: [final hipEngine GGUF sweep](benchmarks/results/2026-07-16-gfx1100-gguf-final-optimization-sweep.json),
[July 12 accepted summary](benchmarks/results/2026-07-12-w7900-v030-8116c453-summary.json),
[hipEngine PARO](benchmarks/results/2026-07-12-w7900-v030-8116c453-hipengine-paro-packed-5run.json),
[superseded July 12 hipEngine GGUF](benchmarks/results/2026-07-12-w7900-v030-8116c453-hipengine-gguf-q4km-5run.json),
[llama.cpp HIP](benchmarks/results/2026-07-12-w7900-v030-8116c453-llamacpp-hip-q4km-f16kv.json),
[llama.cpp Vulkan](benchmarks/results/2026-07-12-w7900-v030-8116c453-llamacpp-vulkan-q4km-f16kv.json),
and [W7900 correctness oracle](benchmarks/results/2026-07-12-w7900-v030-gguf-eager-p512-d4.json).

### gfx1151 (AMD Ryzen AI MAX+ 395 / Radeon 8060S)

> Thanks to Framework for sending a dedicated Framework Desktop Strix Halo motherboard for this profiling and tuning work.

**Status: current IOMMU-off refresh retained through 64K; repeated GGUF 128K
blocked.** The clean 2026-07-17 table at `2edbb2ee` refreshes PARO, GGUF, and
both llama.cpp backends under `amd_iommu=off`. GGUF 512-64K passes clean
provenance, finite logits, exact final IDs, and the 5% variance gate; maximum
prefill/decode stdev over median is **0.122%/0.028%**, and all 15 IDs are
`9707`.

Relative to the previous published IOMMU-on rows, the arithmetic mean change
across 11 eligible hipEngine cells is **+4.60% prefill / +6.20% decode**; GGUF
alone averages **+8.84% / +5.84%**. This is directional, not causal, because
the hipEngine revision/routing also changed; a same-commit reboot A/B remains
necessary. The setting leaves zero IOMMU groups and disables the XDNA/NPU
driver. GGUF 128K still times out after a 584.059 tok/s warmup and 583.464 tok/s
measured pass, so no stale 128K number is carried forward. Bold values remain
descriptive because quant/KV types and memory scopes differ.

<!-- BEGIN TOPLINE:GFX1151_SWEEP -->
#### Prefill tok/s

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | 1298.259 | **1395.379** | 1184.628 | 1161.498 |
| 1K/128 | 1332.199 | **1481.943** | 1192.768 | 1154.327 |
| 4K/128 | 977.252 | **1444.733** | 1148.155 | 1114.081 |
| 32K/128 | 827.350 | **1132.215** | 843.252 | 873.573 |
| 64K/128 | 690.642 | **892.663** | 632.774 | 702.742 |
| 128K/128 | 498.101 | — (blocked) | 432.033 | **499.728** |

#### Decode tok/s

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **70.750** | 52.761 | 53.222 | 63.795 |
| 1K/128 | **65.905** | 54.658 | 53.044 | 63.391 |
| 4K/128 | **66.728** | 55.297 | 52.338 | 61.863 |
| 32K/128 | **53.458** | 45.983 | 45.946 | 52.286 |
| 64K/128 | 44.793 | 39.388 | 40.353 | **45.160** |
| 128K/128 | 32.615 | — (blocked) | 32.728 | **35.569** |

#### Peak memory GiB

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **18.039** | 21.478 | 21.375 | 21.551 |
| 1K/128 | **18.051** | 21.710 | 21.387 | 21.501 |
| 4K/128 | **19.026** | 22.995 | 21.444 | 21.507 |
| 32K/128 | **19.716** | 23.559 | 21.987 | 22.191 |
| 64K/128 | **20.344** | 24.203 | 22.666 | 22.627 |
| 128K/128 | **21.881** | — (blocked) | 23.862 | 24.254 |
<!-- END TOPLINE:GFX1151_SWEEP -->

The memory columns have different scopes: hipEngine reports tracked allocator
high-water, while llama.cpp reports absolute whole-device amdgpu GTT used,
sampled every 10 ms. Use them for within-column context growth, not small
cross-column allocator comparisons. Row source: [`current IOMMU-off refresh and
128K blocker`](benchmarks/results/2026-07-17-gfx1151-amd-iommu-off-topline-refresh.json).
Exact settings and gates are in the canonical
[`benchmarks/README.md`](benchmarks/README.md#gfx1151-model-throughput).

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
#### GGUF MTP comparison, Radeon Pro W7900/gfx1100

| Metric | hipEngine GGUF true AR | hipEngine GGUF exact/default | hipEngine GGUF `llama-compat` | llama.cpp HIP base AR |
| --- | ---: | ---: | ---: | ---: |
| Route | State-bound graph, no MTP | B3, fixed 10 cycles | B2, natural24/cyclecap24 | Natural25 request / 24 timed transitions |
| Decode | **98.75 tok/s fixed / 93.30 tok/s natural24** | 68.50 tok/s | 79.70 tok/s | 78.29 tok/s transition-normalized |
| Own true AR | same route | 98.75 tok/s | 93.30 tok/s | same route |
| MTP / own AR | 1.0000x | **0.6936x** | **0.8542x** | n/a |
| Draft acceptance | n/a | 73.53% | 82.95% | n/a |
| Accepted draft/output | n/a | 50.00% | 60.83% | n/a |
| Complete wall per output/transition | 10.718 ms natural24 | 14.696 ms | 12.578 ms | 12.774 ms |
| State/commit contract | serial autoregressive | serial-prefix preserving | direct partial commit/dp4a; accuracy-traded | native llama.cpp autoregressive |

The old `34.28-34.49 tok/s` true-AR denominator was an eager-only benchmark
path, not the fastest production no-MTP route. gfx1100 had no backend graph
capability even though the state-bound implementation was already shared with
gfx1151. A clean W7900 p512/d24 gate now passes all 24 hidden/GDN/KV/token
transitions and moves capture-inclusive wall from **30.536 to 12.514 ms/token
(2.4402x)**. The full natural24 suite matches every prior eager generated-token
preview/tail and moves **34.28 -> 93.30 tok/s** in the same MTP wrapper.

At the matched cross-engine boundary, hipEngine counts 240 complete post-prefill
transitions including graph capture/instantiate/close; llama.cpp build 9648
requests 25 outputs and counts the 240 timed transitions inside `predicted_ms`.
hipEngine is **93.30 versus 78.29 tok/s (+19.19%)**. BF16 versus F16 KV remains
disclosed. llama.cpp stays an external diagnostic with
`performance_claim=false` because its local instrumentation patchset is dirty
but preserved.

Neither MTP route beats the corrected production AR control. Exact/default
remains the semantic control; `llama-compat` remains explicit-only because
direct partial commit is not serial-prefix-equivalent. The fixed-cycle exact
and natural24 compatibility rows are different protocols and are not ranked
against each other.

##### W7900 `llama-compat` full-suite gate against graph AR

| Scope | Prompts | True AR tok/s | `llama-compat` tok/s | MTP / AR | Draft acceptance | Accepted/output | Cycle wall/output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 10 | **93.30** | 79.70 | **0.8542x** | 82.95% | 60.83% | 12.578 ms |
| Train | 6 | **93.73** | 82.01 | **0.8749x** | **88.12%** | 61.81% | 12.224 ms |
| Heldout | 4 | **92.67** | 76.47 | **0.8252x** | **76.00%** | 59.38% | 13.110 ms |
| `code` | 4 | **93.63** | 86.99 | **0.9291x** | 95.38% | 64.58% | 11.523 ms |
| `general_en` | 2 | **90.99** | 75.87 | **0.8338x** | 75.68% | 58.33% | 13.212 ms |
| `general_ja` | 2 | **94.38** | 72.17 | **0.7647x** | 69.23% | 56.25% | 13.889 ms |
| `mixed_ja_en` | 2 | **93.98** | 78.71 | **0.8375x** | 82.86% | 60.42% | 12.744 ms |

All four categories and heldout lose to graph AR despite unchanged strong draft
acceptance. This corrects the earlier false MTP-win conclusion without changing
the compatibility semantics. Artifact:
[`2026-07-12-w7900-gfx1100-gguf-graph-ar-refresh.json`](results/2026-07-12-w7900-gfx1100-gguf-graph-ar-refresh.json).

#### GGUF MTP comparison, Radeon 8060S/gfx1151

| Metric | hipEngine GGUF exact/default | hipEngine GGUF `llama-compat` | llama.cpp HIP |
| --- | ---: | ---: | ---: |
| Route | B5, fixed 10 cycles | B2, natural24/cyclecap24 | B2, natural25 request / 24 timed transitions |
| Canonical/native MTP decode | 56.39 tok/s (0.9895x own AR) | **81.90 tok/s (1.4423x own AR)** | 70.99 tok/s native (1.3530x own AR; not cross-engine comparable) |
| Cross-engine MTP decode-transition rate | n/a: fixed-cycle horizon | **81.75 tok/s** | 68.15 tok/s |
| Cross-engine own AR transition rate | n/a: fixed-cycle horizon | **56.78 tok/s** | 50.37 tok/s |
| Cross-engine MTP / own AR | n/a | 1.4396x | 1.3530x |
| Draft acceptance | 72.33% | 77.72% | 79.56% |
| Accepted draft/output | 53.49% | 59.58% | 57.60% |
| Full-cycle/predicted wall per counted output or timed transition | 17.808 ms/output | 12.233 ms/output | 14.673 ms/transition |
| State/commit contract | exact/default, serial-prefix preserving | direct partial commit/dp4a; accuracy-traded | native llama.cpp compatibility target |

The IOMMU-off exact/default B5 route improves **51.81 -> 56.39 tok/s** but
still narrowly trails true AR at **56.98 tok/s (0.9895x)**. Its train split is
1.0161x AR, while heldout is only 0.9339x, so the aggregate negative remains
the retained semantic-control result. `llama-compat` stays a separate,
explicit-only contract and is not serial-prefix-equivalent.

The cross-engine rows use the transition-matched timing contract: hipEngine
uses complete cycle wall; llama.cpp requests 25 outputs and counts the 24
transitions inside `predicted_ms`. hipEngine is **81.75 vs 68.15 tok/s
(+19.94%)** on that boundary. hipEngine uses BF16 KV while llama.cpp uses F16
KV. The llama.cpp server binary is byte-identical to the prior publication, but
its source checkout remains dirty/preserved and `performance_claim=false`.
As with the model sweep, hipEngine's prior IOMMU-on comparison is directional
because the measured revision changed; this is not a same-commit reboot A/B.

##### gfx1151 `llama-compat` full-suite gate

| Scope | Prompts | True AR tok/s | `llama-compat` tok/s | MTP / AR | Draft acceptance | Accepted/output | Cycle wall/output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 10 | 56.78 | **81.90** | **1.4423x** | 77.72% | 59.58% | 12.233 ms |
| Train | 6 | 57.21 | **82.99** | **1.4504x** | **82.08%** | 60.42% | 12.073 ms |
| Heldout | 4 | 56.15 | **80.32** | **1.4306x** | **71.79%** | 58.33% | 12.474 ms |
| `code` | 4 | 56.66 | **89.13** | **1.5731x** | 91.04% | 63.54% | 11.239 ms |
| `general_en` | 2 | 57.66 | **78.44** | **1.3605x** | 71.79% | 58.33% | 12.771 ms |
| `general_ja` | 2 | 56.79 | **79.32** | **1.3968x** | 69.23% | 56.25% | 12.629 ms |
| `mixed_ja_en` | 2 | 56.17 | **75.43** | **1.3430x** | 69.23% | 56.25% | 13.287 ms |

All four categories and the heldout split beat their true same-protocol AR
controls. Train/heldout draft acceptance remains **82.08% / 71.79%**; the gap
is kept visible rather than averaged away. The repeated-stream teacher-forced
oracle also passes byte-exact hidden, Conv/GDN, live-KV, and token state.

#### Dense PARO DFlash

| Path | Platform and protocol | Result | Evidence status |
| --- | --- | ---: | --- |
| DFlash B=4 online-gated | W7900/gfx1100; Qwen3.6-27B PARO target plus Qwen3.6-27B DFlash drafter; 9 prompts; 64 decode tokens | 40.10 vs 32.57 AR tok/s, **1.231x** | Retained under the recorded DFlash gate; source tree was dirty and must be refreshed before changing the claim |
<!-- END TOPLINE:SPECULATIVE -->

Artifacts: [`W7900 GGUF MTP transfer`](benchmarks/results/2026-07-12-w7900-gfx1100-gguf-mtp-transfer.json),
[`DFlash`](benchmarks/results/2026-06-11-hipengine-dflash-27b-dense-hardening-rerun.json),
and [`current gfx1151 IOMMU-off MTP refresh`](benchmarks/results/2026-07-17-gfx1151-amd-iommu-off-mtp-refresh.json).
Historical gfx1151 controls remain linked from the canonical benchmark record.
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

Current GGUF direct-model-step tables are retained separately for gfx1100 and
gfx1151. Both have exact native c2/c4/c8 graph routes and direct throughput;
only gfx1100 has also closed live OpenAI membership. The separate gfx1151 PARO
catalog remains c1-only for native performance because its c2-c8 candidates
fail the independent-c1 oracle and use width-1 sessions in production. See
[`docs/CONCURRENCY.md`](docs/CONCURRENCY.md) for the exact boundaries.

The linked records keep gfx1100 and gfx1151 separate because the model files,
ROCm stacks, and comparison backends differ. *Aggregate* is total tok/s across
the batch; *per-sequence* is tok/s seen by one request. See
[`docs/VLLM_RDNA3.md`](docs/VLLM_RDNA3.md) for vLLM RDNA3 setup notes.

### gfx1100 / W7900 direct and server GGUF concurrency (Qwen3.6 35B-A3B, 512/128)

**Status: retained direct native-c4/c8 model-step throughput and retained real
OpenAI SSE arbitrary-C server scaling.** All rows use `UD-Q4_K_M`, BF16 KV,
greedy top-1, W7900/gfx1100, and TheRock HIP 7.15. Timing scopes stay separate:
direct rows time synchronized graph steps; server rows time complete concurrent
SSE cycles including admission, prompt work, decode, delivery, and completion.

<!-- BEGIN TOPLINE:W7900_CONCURRENCY -->
| Direct route | Logical C | Native groups | Aggregate decode tok/s | Per-request tok/s | Aggregate / c1 | Aggregate / serial-c4 | TTFT p50 / p95 | Model-step ITL p50 / p95 | Tracked peak |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct c1 | 1 | 1x c1 | 85.469 | 85.469 | 1.000x | 1.009x | 0.209 / 0.209 s | 11.693 / 11.955 ms | 21.783 GiB |
| direct c2 | 2 | 1x c2 | 127.427 | 63.714 | 1.491x | 1.504x | 0.951 / 0.954 s | 15.765 / 16.023 ms | 22.394 GiB |
| direct c4 | 4 | 1x c4 | 184.575 | 46.144 | 2.160x | 2.178x | 2.020 / 2.023 s | 21.715 / 22.021 ms | 23.396 GiB |
| **direct c8** | **8** | **1x c8** | **246.872** | **30.859** | **2.888x** | **2.913x** | **3.475 / 3.479 s** | **32.414 / 32.749 ms** | **25.401 GiB** |
| chunked c8 control | 8 | 2x c4, serialized | 183.020 | 22.878 | 2.141x | 2.160x | 3.055 / 4.084 s | 43.767 / 44.281 ms | 26.069 GiB* |
| serial-c4 rate control | 4 | 4x c1, serialized | 84.738 | 21.185 | 0.991x | 1.000x | 0.548 / 0.877 s | 47.225 / 48.142 ms | 26.985 GiB* |

| Real OpenAI SSE route | Logical C | Physical execution | Aggregate generated tok/s | Per-request tok/s | Aggregate / logical-c1 | Aggregate / serial-c13 | Cycle wall p50 | Scheduler TTFT p50 / p95 | Scheduler ITL p50 / p95 | Cumulative tracked peak |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logical-c1 control | 1 | masked physical c8 | 25.583 | 25.583 | 1.000x | 0.807x | 5.003 s | 0.271 / 0.276 s | 36.499 / 39.809 ms | 29.312 GiB |
| physical c8 | 8 | 1x c8 | **136.122** | 17.015 | **5.321x** | 4.293x | 7.523 s | 1.751 / 2.088 s | 41.712 / 45.068 ms | 30.805 GiB* |
| grouped c9 | 9 | c8 + sparse c8 | 88.592 | 9.844 | 3.463x | 2.794x | 13.003 s | 1.709 / 2.235 s | 81.087 / 88.112 ms | 31.969 GiB* |
| **grouped c13** | **13** | **c8 + sparse c8** | **111.380** | **8.568** | **4.354x** | **3.513x** | **14.940 s** | **1.886 / 3.323 s** | **87.502 / 93.631 ms** | **32.869 GiB*** |
| serial-c13 bridge | 13 | 13x c1 serial | 31.708 | 2.439 | 1.239x | 1.000x | 52.479 s | 2.424 / 3.390 s | 382.821 / 396.004 ms | 32.869 GiB* |
<!-- END TOPLINE:W7900_CONCURRENCY -->

Direct protocol uses 128 decode transitions, one discarded warmup, and median
of three; one physical c8 is **2.888x** c1 and **+34.89%** over c4+c4, with a
**748 packed-native / 0 row-local / 0 copy** trace. Server protocol uses 512
exact prompt IDs and 128 generated outputs/request, a 20 ms admission window,
one discarded plus three measured bursts, and scheduler latency. Logical c1 is
honestly a masked physical-c8 production control; C9/C13 are multiple declared
buckets, never wider native widths. All **189/189** server requests match
resident prompt IDs, direct-c1 outputs, usage, and finish metadata. Grouped C13
is **4.354x** logical-c1 and **3.513x** serial; one exact c8→c13 live trace emits
**1,664/1,664** IDs at **107.284 aggregate tok/s** and drains ownership to zero.
Starred server memory is cumulative in one prepared process.

Artifacts: [`C4`](benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-c4-native-graph-scaling-closure.json),
[`E2 native c8`](benchmarks/results/2026-07-17-gfx1100-gguf-concurrency-e2-native-c8-scaling-closure.json),
[`E3 arbitrary C`](benchmarks/results/2026-07-17-gfx1100-gguf-concurrency-e3-arbitrary-c-correctness.json), and
[`F1 real server`](benchmarks/results/2026-07-17-gfx1100-gguf-concurrency-f1-server-scaling-closure.json).
Historical mixed-quant/mixed-scope results remain in
[`benchmarks/HISTORY.md`](benchmarks/HISTORY.md).

### gfx1151 / Radeon 8060S PARO direct c2/c4/c8 and production shape catalog (Qwen3.6 35B-A3B, 512/128)

**Status: direct and resident OpenAI c2/c4/c8 are retained; gfx1151 selects
them by package default.** Equivalent clean tree `e175e28f` (pushed as
`8c8cc15e`) generalizes the exact c2 route into true physical c4/c8 without c2
stacking. G5 attaches those identity-matched widths to one shared stable-slot
owner for public `LLM`, blocking OpenAI, and concurrent SSE. Explicit legacy
`=0` flags remain rollback opt-outs.

Three p512/d128 processes per width pass all **5,754/5,754** recorded IDs plus
all-layer state/KV, sparse lifecycle, ten-prompt category/heldout, primitive,
and cached-profiler gates.

| Explicit direct route | Median aggregate decode | Per-request decode | Classification |
| --- | ---: | ---: | --- |
| c1 graph | **70.810 tok/s** | 70.810 tok/s | independent reference |
| serial c2 bridge | **65.574 tok/s** | 32.787 tok/s | exact fallback control |
| native selected-batch c2 | **79.237 tok/s** | 39.619 tok/s | **retained direct c2; 1.1190x c1 / 1.2084x serial** |
| true physical c4 | **100.209 tok/s** | 25.052 tok/s | **retained direct c4; 1.4152x c1** |
| true physical c8 | **99.943 tok/s** | 12.493 tok/s | **retained direct c8; 1.4114x c1** |

The production table below uses the clean blocking F1 wall: 512 raw prompt IDs,
128 generated IDs/request, one warmup plus three measured bursts, and a fresh
server per width. All **68/68** warmup/measured/live rows are exact; c1/c2/c4/c8
scale to **47.124/51.962/60.323/61.253 aggregate tok/s** with <=0.994% variance.
The complementary exact-roundtrip SSE packet keeps all **100/100** rows exact at
**36.327/38.666/42.471/41.487/35.633 tok/s** for c1/c2/c4/c8/serial-c8; native
c8 is **1.164x** serial and live c4->c8 admission is **38.191 tok/s**. A 1+7 c8
stress adds **72/72** exact rows, and a no-native-flag OpenAI c4 gate run from
`/tmp` loads the packaged profile, observes physical widths 2/4, and records no
fallback.

<!-- BEGIN TOPLINE:GFX1151_PARO_CURRENT -->
| Client c | Production backend groups | Exact classification | Retained OpenAI aggregate |
| ---: | --- | --- | ---: |
| 1 | `1` | c1 oracle / accepted | **47.124 tok/s** |
| 2 | native `2` | retained physical width | **51.962 tok/s** |
| 3 | `2+1` | exact partition; not native c3 | no separate claim |
| 4 | native `4` | retained physical width | **60.323 tok/s** |
| 5 | `4+1` | exact partition; not native c5 | no separate claim |
| 6 | `4+2` | exact partition; not native c6 | no separate claim |
| 7 | `4+2+1` | exact partition; not native c7 | no separate claim |
| 8 | native `8` | retained physical width | **61.253 tok/s** |
<!-- END TOPLINE:GFX1151_PARO_CURRENT -->

Protocol: W4 PARO/BF16 KV, 40 layers, 8 warmup decode steps, 128 measured
decode steps, greedy sampling, TheRock HIP 7.15, one hardware queue, TuneD
`accelerator-performance`, and `amd_iommu=off`. All direct-width process
variance is <=0.054%. c4/c8 are **+41.52%/+41.14% vs c1**; c8 aggregate is
**0.265% below c4**, while its median model-step time is **0.183% faster than
two c4 steps**. c3/c5/c6/c7 retain no native-width claim. See the
[retained direct-c2/c4/c8 artifact](benchmarks/results/2026-07-18-gfx1151-paro-g3-native-c248-direct-retained.json),
[G5 blocking F1](benchmarks/results/2026-07-18-gfx1151-paro-g5-f1-server-scaling.json),
[G5 SSE](benchmarks/results/2026-07-18-gfx1151-paro-g5-sse-server-scaling.json),
[c8 repeatability](benchmarks/results/2026-07-18-gfx1151-paro-g5-c8-sse-repeatability.json),
[package-default OpenAI c4](benchmarks/results/2026-07-18-gfx1151-paro-g5-default-openai-c4.json),
and the [canonical run record](benchmarks/README.md#paro-concurrency-and-production-routing).

### gfx1151 / Radeon 8060S direct and server GGUF concurrency (Qwen3.6 35B-A3B, 512/128)

**Status: retained direct native-c2/c4/c8 model-step throughput and retained real
OpenAI SSE arbitrary-C server scaling.** All rows use `UD-Q4_K_M`, BF16 KV,
greedy top-1, one HIP hardware queue, and the active `amd_iommu=off` boot. Timing
scopes stay separate: direct rows time synchronized graph steps; server rows
time complete concurrent SSE cycles. The direct/category and E3 gates retain
**188,080** and **134,160** exact hidden comparisons, and the c8 trace is **748
packed-native / 0 row-local / 0 copies**.

<!-- BEGIN TOPLINE:GFX1151_CONCURRENCY -->
| Direct route | Logical C | Native groups | Aggregate decode tok/s | Per-request tok/s | Aggregate / c1 | Aggregate / serial-c4 | TTFT p50 / p95 | Model-step ITL p50 / p95 | Tracked peak |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct c1 | 1 | 1x c1 | 50.291 | 50.291 | 1.000x | 1.001x | 0.367 / 0.368 s | 19.875 / 20.162 ms | 21.783 GiB |
| direct c2 | 2 | 1x c2 | 72.262 | 36.131 | 1.437x | 1.439x | 2.176 / 2.176 s | 27.679 / 27.967 ms | 22.394 GiB |
| direct c4 | 4 | 1x c4 | 102.663 | 25.666 | 2.041x | 2.044x | 3.393 / 3.393 s | 38.980 / 39.295 ms | 23.396 GiB |
| **direct c8** | **8** | **1x c8** | **128.075** | **16.009** | **2.547x** | **2.550x** | **6.836 / 6.849 s** | **62.473 / 62.973 ms** | **25.401 GiB** |
| chunked c8 control | 8 | 2x c4, serialized | 102.724 | 12.841 | 2.043x | 2.045x | 5.089 / 6.787 s | 77.902 / 78.467 ms | 26.069 GiB* |
| serial-c4 rate control | 4 | 4x c1, serialized | 50.235 | 12.559 | 0.999x | 1.000x | 0.927 / 1.485 s | 79.643 / 80.637 ms | 26.985 GiB* |

| Real OpenAI SSE route | Logical C | Physical execution | Aggregate generated tok/s | Per-request tok/s | Aggregate / logical-c1 | Aggregate / serial-c13 | Cycle wall p50 | Scheduler TTFT p50 / p95 | Scheduler ITL p50 / p95 | Cumulative tracked peak |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logical-c1 control | 1 | masked physical c8 | 15.798 | 15.798 | 1.000x | 0.366x | 8.102 s | 0.432 / 0.434 s | 59.835 / 60.409 ms | 29.312 GiB |
| physical c8 | 8 | 1x c8 | **86.358** | 10.795 | **5.467x** | 2.003x | 11.858 s | 2.283 / 3.216 s | 64.781 / 68.431 ms | 31.291 GiB* |
| grouped c9 | 9 | c8 + sparse c8 | 57.691 | 6.410 | 3.652x | 1.338x | 19.969 s | 2.094 / 3.572 s | 125.097 / 132.002 ms | 32.008 GiB* |
| **grouped c13** | **13** | **c8 + sparse c8** | **73.065** | **5.620** | **4.625x** | **1.695x** | **22.774 s** | **3.588 / 5.298 s** | **132.468 / 143.079 ms** | **32.889 GiB*** |
| serial-c13 bridge | 13 | 13x c1 serial | 43.116 | 3.317 | 2.729x | 1.000x | 38.594 s | 3.599 / 5.319 s | 257.918 / 271.301 ms | 32.889 GiB* |
<!-- END TOPLINE:GFX1151_CONCURRENCY -->

Direct protocol uses 128 decode transitions, one discarded warmup, and the
median of three; one physical c8 is **2.544x** c1 and **+24.65%** over c4+c4.
Server protocol uses 512 exact prompt IDs and 128 generated outputs/request, a
20 ms admission window, one discarded plus three measured bursts, and scheduler
latency. C9/C13 are multiple declared buckets, never wider native widths. All
**189/189** requests match resident prompt IDs, direct-c1 outputs, usage, and
finish metadata. Grouped C13 is **4.619x** logical-c1 and **1.696x** serial; one
exact c8→c13 live trace emits **1,664/1,664** IDs at **70.093 aggregate tok/s**
and drains ownership to zero. Starred server memory is cumulative.

Artifacts: [`E1 direct correctness`](benchmarks/results/2026-07-17-gfx1151-gguf-concurrency-e1-direct-correctness.json),
[`E1 direct scaling`](benchmarks/results/2026-07-17-gfx1151-gguf-concurrency-e1-native-c8-scaling-closure.json),
[`E3 arbitrary C`](benchmarks/results/2026-07-18-gfx1151-gguf-concurrency-e3-arbitrary-c-correctness.json), and
[`F1 real server`](benchmarks/results/2026-07-18-gfx1151-gguf-concurrency-f1-server-scaling-closure.json).

### gfx1151 historical cross-engine concurrency (2026-06-15)

**Status: stale diagnostic.** hipEngine used PARO W4/BF16 KV; llama.cpp used
Vulkan Q4_K_S/f16 KV, and vLLM did not produce a healthy server. The summary
lacks the measured hipEngine commit and has incomplete device provenance. No
eligible historical row is inferred from the current direct GGUF table.

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
│  KERNELS (backend-keyed custom HIP implementations)             │
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
pip install "hipengine[torch]"

# dev / test
pip install -e ".[dev]"
```

Python 3.10+. A working ROCm install with `libamdhip64.so` on the loader path
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

Core endpoints are `GET /v1/models`, `POST /v1/completions`, and
`POST /v1/chat/completions`, with token-level SSE streaming, logprobs,
OpenAI-style tools, structured-output validation, and Qwen thinking controls.
hipEngine extensions provide readiness/capability discovery, token and context
diagnostics, and app-local session management. Chat responses separate
`<think>` reasoning into `reasoning_content`. The server eagerly warms the model
on startup, caps omitted chat `max_tokens` with
`--chat-default-max-tokens` (default 4096), and has an explicit `--debug` mode
for full request/response payload logging. See the complete
[`docs/API.md` endpoint table](docs/API.md#endpoints) for bearer-token auth,
request examples, feature contracts, diagnostics, and current limitations.

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
  grateful and deeply appreciative of nano-vllm as a great research platform.
- [ParoQuant](https://github.com/z-lab/paroquant) - after reviewing the current SOTA on model
  quantization, we chose ParoQuant as the first target due to both its excellent accuracy
  *and* its efficiency (QTIP/[YAQA](https://github.com/Cornell-RelaxML/yaqa-quantization) is 
  very cool but proved challenging to implement performant RDNA3 kernels)
- [FastDMS](https://github.com/shisa-ai/FastDMS) - our KVCache ABI is shaped by the lessons 
   learned from building our DMS reference implementation.

Greetz: [ROCmFPX](https://github.com/charlie12345/ROCmFPX), [hipfire](https://github.com/Kaden-Schutt/hipfire), [Lucebox](https://github.com/Luce-Org/lucebox-hub), [DS4](https://github.com/antirez/ds4), [ExLlamaV3](https://github.com/turboderp-org/exllamav3) and ofc the og [llama.cpp](https://github.com/ggml-org/llama.cpp)

See also: [Marlin](https://github.com/IST-DASLab/marlin), [kernel-anvil](https://github.com/apollosenvy/kernel-anvil), [wmma_ops](https://github.com/glovepost/wmma_ops), [tilelang](https://github.com/tile-ai/tilelang), [fsr4-rdna3-optimization](https://github.com/lhl/fsr4-rdna3-optimization), [ROCm examples](https://github.com/ROCm/rocm-examples)


## License

hipEngine source code is licensed under **AGPL-3.0-or-later**. It is built and distributed
for anyone who has an AMD card that hasn't been living up to its compute potential.

Model weights, checkpoints, and external datasets remain under their own licenses.
