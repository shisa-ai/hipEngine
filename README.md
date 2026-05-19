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

**v0.1.x.** The runtime hot path is torch-free by construction; kernel
families and registry plumbing are landing under
[`hipengine/kernels/hip_gfx1100/`](hipengine/kernels/hip_gfx1100/). Current
single-model tuning targets
[shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5-packed](https://huggingface.co/shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5-packed)
(19.07 GiB, 4.68 bpw) in packed
[ParoQuant](https://github.com/shisa-ai/paroquant) format.

- INT8 KV cache support has been added. Qwen 3 MoE's full 256K context window can fit in <24GB tracked memory; see [Memory Usage](#memory-usage).


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

With BF16 KV cache, hipEngine running the packed Qwen 3.6 PARO model fits
>128K context window in a 24GB-class memory budget. The INT8 KV cache option
(with FP16 per-token/per-head scales) uses the
`--kv-storage int8_per_token_head` flag and lets the **full 256K context** fit
under 24 GiB tracked allocator peak.

The numbers below are for
`shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5-packed` on W7900/gfx1100 with q3072
full-attention prefill chunks:

| Model                | Context | KV cache | Sampled peak | Allocator peak | Retained KV | Prefill      | Decode     |
| -------------------- | ------: | -------- | -----------: | -------------: | ----------: | -----------: | ---------: |
| Qwen3.6 35B-A3B PARO |    128K | BF16     |    21.04 GiB |      21.88 GiB |    2.69 GiB | 1091.9 tok/s | 62.2 tok/s |
| Qwen3.6 35B-A3B PARO |    128K | INT8     |    19.80 GiB |      20.89 GiB |    1.36 GiB | 1076.5 tok/s | 60.0 tok/s |
| Qwen3.6 35B-A3B PARO |    256K | INT8     |    21.96 GiB |      23.71 GiB |    2.71 GiB |  670.2 tok/s | 40.3 tok/s |

Regardless of the difference in PARO weight storage (legacy or packed),
loaded-weight memory is about the same - approximately 16.4GB in VRAM.

The INT8 KV correctness gate is currently the deterministic Qwen3.5 PARO
fixture `fixtures/qwen35_paro/parent_512_32_seed1234.json` (512-token prompt,
32 greedy decode tokens): `max_kl=0.015328`, `mean_kl=0.001639`, top-1 agreement
100%, and generated IDs match BF16 KV exactly. Layer attention probes at context
64 and 520 also had top-1 agreement 100% with max quantized-vs-BF16 KL
`2.34e-7`. This is a fixture/regression gate, not a long-rollout perplexity
study, so long context generations may have unmeasured compounding errors.

The same 128K/128 Qwen3.5 BF16-vs-INT8 run measured -0.99% prefill tok/s and
-3.20% decode tok/s for INT8 KV, so speed loss is also very small.

See
[`benchmarks/results/2026-05-19-hipengine-qwen36-packed-int8-kv-readme-memory-diagnostic.json`](benchmarks/results/2026-05-19-hipengine-qwen36-packed-int8-kv-readme-memory-diagnostic.json),
[`benchmarks/README.md`](benchmarks/README.md#blocked--diagnostic-benchmark-attempts),
and [`docs/KVCACHE.md`](docs/KVCACHE.md) for commands, artifacts, and the full
no-shadow memory audit.

## Performance

### gfx1100 (Radeon RX 7900 XTX / Radeon Pro W7900)

While we are far from [gfx1100 roofline](https://github.com/shisa-ai/hipEngine/blob/main/docs/ROOFLINE.md), the current gfx1100 implementation does well compared to Q4_K_M quants of recent llama.cpp builds (`b9042`) on the same model family. The latest W7900 packed rows use the default prefill policy: 512-token prompts stay unchunked and prompts above 1K use `1024/1024/4096/1024/1024` chunks.

### Prefill tok/s

| Workload | hipEngine shisa Qwen3.6 packed PARO | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: |
| 512/128 | **2500.565** | 2436.049 | 1816.927 |
| 4K/128 | **2899.685** | 2176.905 | 1705.093 |
| 32K/128 | **2115.050** | 1496.409 | 1128.554 |
| 128K/128 | **1054.291** | 710.213 | 480.539 |

### Decode tok/s

| Workload | hipEngine shisa Qwen3.6 packed PARO | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: |
| 512/128 | 111.516 | 85.487 | **127.515** |
| 4K/128 | 113.094 | 87.375 | **120.163** |
| 32K/128 | 97.594 | 76.994 | **98.073** |
| 128K/128 | 62.027 | 57.341 | **64.478** |

### Peak GiB

| Workload | hipEngine shisa Qwen3.6 packed PARO | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: |
| 512/128 | **18.123** | 21.125 | 20.844 |
| 4K/128 | **19.455** | 21.197 | 20.969 |
| 32K/128 | **20.267** | 21.738 | 21.533 |
| 128K/128 | **23.235** | 23.605 | 23.596 |

### gfx1151 (AMD Ryzen AI MAX+ 395 / Radeon 8060S)

The gfx1151 backend is a native `--offload-arch=gfx1151` peer backend using the same registry-keyed kernel surface. The Strix Halo snapshot below uses 256-row prefill chunks, which removed the 4K prefill gap without hurting long-context decode.

### Prefill tok/s

| Workload | hipEngine shisa Qwen3.6 packed PARO | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: |
| 512/128 | 983.206 | **1058.738** | 638.008 |
| 4K/128 | **1029.402** | 1004.220 | 595.400 |
| 32K/128 | **792.296** | 735.534 | 407.984 |
| 128K/128 | **413.489** | 376.070 | 181.453 |

### Decode tok/s

| Workload | hipEngine shisa Qwen3.6 packed PARO | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: |
| 512/128 | **62.060** | 50.537 | 57.615 |
| 4K/128 | **63.605** | 49.379 | 55.027 |
| 32K/128 | **50.629** | 43.435 | 44.576 |
| 128K/128 | 30.245 | **31.286** | 26.935 |

On Strix Halo, `rocm-smi` / sysfs expose only a 512 MiB VRAM aperture, so cross-engine memory comparisons are omitted here. The hipEngine allocator high-water mark for the chunk256 sweep was 17.997 GiB (512/128), 18.097 GiB (4K/128), 18.909 GiB (32K/128), and 21.877 GiB (128K/128).

See [`benchmarks/README.md`](benchmarks/README.md) for full protocol details,
correctness status, source-lineage targets, and external comparison baselines.

## Architecture at a glance

```
┌─────────────────────────────────────────────────────────────────┐
│  USER API                                                       │
│  hipengine.LLM.generate()           library API                 │
│  hipengine.server                   optional [server] extra     │
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

# core runtime (torch-free)
pip install -e .

# with the OpenAI-compatible server
pip install -e ".[server]"

# with the optional dlpack torch bridge for user-boundary interop
pip install -e ".[torch]"

# dev / test
pip install -e ".[dev]"
```

Python 3.11+. A working ROCm install with `libamdhip64.so` on the loader path
is required for any GPU run; CPU-reference correctness tests run without a GPU.

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

Install the optional server extra and run the FastAPI layer:

```bash
pip install -e ".[server]"
python -m hipengine.server \
  --model /path/to/model \
  --quant w4_paro \
  --served-model-name qwen-paro
```

Supported v0.1 endpoints: `GET /v1/models`, `POST /v1/completions`, and
`POST /v1/chat/completions` (including one-chunk SSE for `stream=true`). See
[`docs/API.md`](docs/API.md) for request examples, bearer-token auth, and
current limitations.

## Documentation

| File | Purpose |
| --- | --- |
| [`docs/PLAN.md`](docs/PLAN.md) | Architecture, plugin axes, phase roadmap, LoC budgets |
| [`docs/BENCHMARK.md`](docs/BENCHMARK.md) | Benchmark protocols, baselines, correctness gate, artifact format |
| [`docs/TESTING.md`](docs/TESTING.md) | RED/GREEN workflow, correctness oracles, fixture policy |
| [`docs/KERNELS.md`](docs/KERNELS.md) | Kernel catalog, source-lineage drift workflow, JIT cache gotchas, build profiles |
| [`docs/ROOFLINE.md`](docs/ROOFLINE.md) | RDNA3 / W7900 performance model and decision tree |
| [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md) | Implementation status and concrete milestones |
| [`docs/API.md`](docs/API.md) | OpenAI-compatible server usage and endpoint support |
| [`docs/PREFILL.md`](docs/PREFILL.md) | Native prefill implementation spec |
| [`docs/MTP.md`](docs/MTP.md) | Multi-token prediction plan |
| [`docs/DFLASH.md`](docs/DFLASH.md) | DFlash draft-model speculative decode plan |
| [`benchmarks/README.md`](benchmarks/README.md) | Current-fastest rollup and external comparison baselines |
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
