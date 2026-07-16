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

**Status: retained through 64K; repeated 128K blocked.** GGUF is the clean
2026-07-15 selector-unset 1+3 production refresh at `61a27d72`, with the
current LCP stack and automatic one-queue gfx1151 process policy. The five
accepted shapes pass clean provenance, finite logits, exact final IDs, and the
5% variance gate; prefill/decode stdev over median is at most
**0.187%/0.049%**, all 15 measured IDs are `9707`, and tracked memory is
unchanged. Repeated 128K production completes warmup but can still enter the
100%/2.9 GHz, 44-46 W no-progress state on measured pass 1. Router rollback and
SDMA-disabled full controls reproduce it, so no stale hipEngine GGUF 128K value
is carried into the table. llama.cpp remains the clean July 11 reference; PARO
retains its July 12 exact recovery and scoped queue-isolation rows. Bold marks
the best raw value per row, but PARO is W4 PARO rather than Q4_K_M and memory
scopes differ, so emphasis is descriptive rather than a controlled same-math
allocator comparison.

<!-- BEGIN TOPLINE:GFX1151_SWEEP -->
#### Prefill tok/s

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | 1140.101 | **1294.885** | 1061.260 | 1067.770 |
| 1K/128 | 1208.343 | **1358.342** | 1043.230 | 1069.870 |
| 4K/128 | 1089.031 | **1365.720** | 1009.240 | 1016.580 |
| 32K/128 | 906.145 | **1034.845** | 743.547 | 814.923 |
| 64K/128 | 716.775 | **796.083** | 573.611 | 660.974 |
| 128K/128 | 474.641 | — (blocked) | 390.441 | **476.788** |

#### Decode tok/s

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **66.767** | 49.041 | 50.939 | 62.396 |
| 1K/128 | 61.746 | 51.623 | 50.818 | **62.136** |
| 4K/128 | **62.715** | 52.422 | 50.126 | 60.097 |
| 32K/128 | 50.342 | 43.572 | 44.240 | **51.319** |
| 64K/128 | 42.094 | 37.622 | 39.326 | **44.422** |
| 128K/128 | 30.386 | — (blocked) | 32.114 | **34.948** |

#### Peak memory GiB

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **18.039** | 21.478 | 21.375 | 21.551 |
| 1K/128 | **18.051** | 21.710 | 21.387 | 21.501 |
| 4K/128 | **19.026** | 22.995 | 21.444 | 21.507 |
| 32K/128 | **19.729** | 23.559 | 21.987 | 22.191 |
| 64K/128 | **20.403** | 24.203 | 22.666 | 22.627 |
| 128K/128 | **22.124** | — (blocked) | 23.862 | 24.254 |
<!-- END TOPLINE:GFX1151_SWEEP -->

The memory columns have different scopes: hipEngine reports tracked allocator
high-water, while llama.cpp reports absolute whole-device amdgpu GTT used,
sampled every 10 ms. Use them for within-column context growth, not small
cross-column allocator comparisons. Row sources: [`current hipEngine GGUF 512-64K refresh and 128K blocker`](benchmarks/results/2026-07-15-gfx1151-gguf-production-refresh-512-64k-128k-blocked.json),
[`PARO exact recovery`](benchmarks/results/2026-07-12-gfx1151-paro-prefill-recovery.json),
[`PARO 4K-128K AOTriton queue isolation`](benchmarks/results/2026-07-12-gfx1151-paro-aotriton-stream-isolation.json),
[`previous GGUF LCP-1/LCP-D1 rollup`](benchmarks/results/2026-07-14-gfx1151-gguf-lcp1-lcpd1-right-sized-3run.json),
[`queue-stall follow-up`](https://github.com/ROCm/ROCm/issues/5107#issuecomment-4979442043),
[`accepted July 11 matched summary`](benchmarks/results/2026-07-11-gfx1151-readme-refresh-20260711-d1231ee0-summary.json),
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
| Canonical/native MTP decode | 51.81 tok/s (0.9571x own AR) | **69.50 tok/s (1.2776x own AR)** | 69.44 tok/s native (1.3752x own AR; not cross-engine comparable) |
| Cross-engine MTP decode-transition rate | n/a: fixed-cycle horizon | **69.38 tok/s** | 66.66 tok/s |
| Cross-engine own AR transition rate | n/a: fixed-cycle horizon | **54.40 tok/s** | 48.47 tok/s |
| Cross-engine MTP / own AR | n/a | 1.2755x | 1.3752x |
| Draft acceptance | 72.33% | 77.72% | 79.56% |
| Accepted draft/output | 53.49% | 59.58% | 57.60% |
| Full-cycle/predicted wall per counted output or timed transition | 19.360 ms/output | 14.413 ms/output | 15.001 ms/transition |
| State/commit contract | exact/default, serial-prefix preserving | direct partial commit/dp4a; accuracy-traded | native llama.cpp compatibility target |

The current exact/default B5 route no longer beats true AR after the
correctness/state-lifecycle pass: **51.81 vs 54.14 tok/s (0.9571x)**. Its old
61.98 tok/s row is retained only as history. `llama-compat` remains a separate,
explicit-only semantic contract and is not serial-prefix-equivalent.

The cross-engine rows use the canonical transition-matched timing contract:
hipEngine uses complete cycle wall; llama.cpp requests 25 outputs and counts
the 24 transitions inside `predicted_ms`. This removes llama.cpp's native
one-untimed-token numerator advantage. hipEngine uses BF16 KV while llama.cpp
uses F16 KV, which remains a model-execution difference even with matched timer
boundaries. The captured llama.cpp source is dirty but fully preserved in the
repository patchset; the binary hash is authoritative and
`performance_claim=false`.

##### gfx1151 `llama-compat` full-suite gate

| Scope | Prompts | True AR tok/s | `llama-compat` tok/s | MTP / AR | Draft acceptance | Accepted/output | Cycle wall/output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 10 | 54.40 | **69.50** | **1.2776x** | 77.72% | 59.58% | 14.413 ms |
| Train | 6 | 54.44 | **70.96** | **1.3034x** | **82.08%** | 60.42% | 14.116 ms |
| Heldout | 4 | 54.33 | **67.42** | **1.2408x** | **71.79%** | 58.33% | 14.858 ms |
| `code` | 4 | 54.42 | **74.81** | **1.3747x** | 91.04% | 63.54% | 13.387 ms |
| `general_en` | 2 | 54.50 | **67.62** | **1.2407x** | 71.79% | 58.33% | 14.811 ms |
| `general_ja` | 2 | 54.40 | **66.60** | **1.2242x** | 69.23% | 56.25% | 15.042 ms |
| `mixed_ja_en` | 2 | 54.25 | **64.90** | **1.1964x** | 69.23% | 56.25% | 15.438 ms |

All four categories and the heldout split beat their true same-protocol AR
controls. Train/heldout draft acceptance is **82.08% / 71.79%**; the gap is
kept visible rather than averaged away.

#### Dense PARO DFlash

| Path | Platform and protocol | Result | Evidence status |
| --- | --- | ---: | --- |
| DFlash B=4 online-gated | W7900/gfx1100; Qwen3.6-27B PARO target plus Qwen3.6-27B DFlash drafter; 9 prompts; 64 decode tokens | 40.10 vs 32.57 AR tok/s, **1.231x** | Retained under the recorded DFlash gate; source tree was dirty and must be refreshed before changing the claim |
<!-- END TOPLINE:SPECULATIVE -->

Artifacts: [`W7900 GGUF MTP transfer`](benchmarks/results/2026-07-12-w7900-gfx1100-gguf-mtp-transfer.json),
[`DFlash`](benchmarks/results/2026-06-11-hipengine-dflash-27b-dense-hardening-rerun.json),
[`gfx1151 exact MTP`](benchmarks/results/2026-07-02-ar-mtp-default-parallelattn-full.json),
and [`gfx1151 llama-compat` MTP](benchmarks/results/2026-07-03-ar-mtp-llama-compat-directcommit-nocopy-natural24-cyclecap24-f32head-full.json).
The gfx1151 matched natural24 controls are [`exact/default B1-B5`](benchmarks/results/2026-07-03-ar-mtp-default-natural24-budget-sweep-c1.json)
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

### gfx1100 / W7900 direct GGUF concurrency (Qwen3.6 35B-A3B, 512/128)

**Status: retained direct decode-model-step packet; not production continuous
batching.** All rows use the same `UD-Q4_K_M`, BF16 KV, greedy-top1,
W7900/gfx1100, and synchronized graph-step timing. Aggregate counts every row
advanced by each logical transition; per-request is aggregate divided by live
rows. ITL excludes streaming-token D2H.

<!-- BEGIN TOPLINE:W7900_CONCURRENCY -->
| Route | Logical C | Native groups | Aggregate decode tok/s | Per-request tok/s | Aggregate / c1 | Aggregate / serial-c4 | TTFT p50 / p95 | Model-step ITL p50 / p95 | Tracked peak |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct c1 | 1 | 1x c1 | 84.907 | 84.907 | 1.000x | 1.009x | 0.209 / 0.209 s | 11.768 / 12.105 ms | 21.783 GiB |
| direct c2 | 2 | 1x c2 | 126.909 | 63.455 | 1.495x | 1.508x | 0.954 / 0.957 s | 15.802 / 16.024 ms | 22.394 GiB |
| **direct c4** | **4** | **1x c4** | **184.993** | **46.248** | **2.179x** | **2.199x** | **2.027 / 2.031 s** | **21.641 / 21.888 ms** | **23.396 GiB** |
| chunked c8 | 8 | 2x c4, serialized | 183.900 | 22.987 | 2.166x | 2.186x | 3.055 / 4.079 s | 43.487 / 44.014 ms | 25.731 GiB |
| serial-c4 control | 4 | 4x c1, serialized | 84.140 | 21.035 | 0.991x | 1.000x | 0.547 / 0.875 s | 47.481 / 48.465 ms | 26.646 GiB* |
<!-- END TOPLINE:W7900_CONCURRENCY -->

Protocol: prompt 512 per row, 128 decode transitions, one discarded full-route
warmup and median of three, one shared model load. Resident sessions grow
c1→c2→c4→c8 so c4 memory is scoped to four sessions. `serial-c4` runs after c8
and its starred memory retains eight resident slabs; it is a throughput control,
not the c4 memory row. C4 improves aggregate decode **117.88%** over c1 and
**119.86%** over serial-c4, while per-request rate is **45.53% lower** than c1;
Phase D has not admitted this tradeoff as a live server route.

Artifact: [`retained C4 closure`](benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-c4-native-graph-scaling-closure.json).
Historical mixed-quant/mixed-scope results remain in
[`benchmarks/HISTORY.md`](benchmarks/HISTORY.md).

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

Greetz: [hipfire](https://github.com/Kaden-Schutt/hipfire), [Lucebox](https://github.com/Luce-Org/lucebox-hub), [DS4](https://github.com/antirez/ds4), [ExLlamaV3](https://github.com/turboderp-org/exllamav3) and ofc the og [llama.cpp](https://github.com/ggml-org/llama.cpp)

See also: [Marlin](https://github.com/IST-DASLab/marlin), [kernel-anvil](https://github.com/apollosenvy/kernel-anvil), [wmma_ops](https://github.com/glovepost/wmma_ops), [tilelang](https://github.com/tile-ai/tilelang), [fsr4-rdna3-optimization](https://github.com/lhl/fsr4-rdna3-optimization), [ROCm examples](https://github.com/ROCm/rocm-examples)


## License

hipEngine source code is licensed under **AGPL-3.0-or-later**. It is built and distributed
for anyone who has an AMD card that hasn't been living up to its compute potential.

Model weights, checkpoints, and external datasets remain under their own licenses.
